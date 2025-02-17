import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from tqdm import tqdm

class BacktestEngine:
    def __init__(self, data_dir: str = "ds_fee/market_data", output_dir: str = "output"):
        self.preprocessed_data = None
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_config()
        
    def _load_config(self):
        """加载路径配置"""
        self.spot_path = os.path.join(self.data_dir, "1m/spot")
        self.future_path = os.path.join(self.data_dir, "1m/future")
        
    def preprocess_data(self):
        """预处理市场数据（只执行一次）"""
        if self.preprocessed_data is not None:
            return
            
        print("开始加载并预处理市场数据...")
        
        # 并行读取现货和期货数据
        spot_df = self._load_minute_data(self.spot_path, is_future=False)
        future_df = self._load_minute_data(self.future_path, is_future=True)
        
        # 重置索引为列以便merge_asof正确合并
        spot_df_reset = spot_df.reset_index()
        future_df_reset = future_df.reset_index()
        
        # 时间对齐处理
        merged_df = pd.merge_asof(
            spot_df_reset.sort_values('timestamp'),
            future_df_reset.sort_values('timestamp'),
            on='timestamp',
            suffixes=('_spot', '_future'),
            tolerance=pd.Timedelta('1min')
        ).dropna()
        merged_df.set_index('timestamp', inplace=True)
        
        # 计算价差和年化收益率
        merged_df['price_spread'] = merged_df['close_future'] - merged_df['close_spot']
        merged_df['funding_yield'] = merged_df['funding_rate'] * 24 * 365
        
        self.preprocessed_data = merged_df
        print(f"数据预处理完成，总数据量：{len(merged_df)} 条")
        
    def _load_minute_data(self, path: str, is_future: bool) -> pd.DataFrame:
        """加载分钟级别数据"""
        all_files = []
        for file in tqdm(os.listdir(path), desc=f"Loading {'future' if is_future else 'spot'} data"):
            if file.endswith('.csv'):
                file_path = os.path.join(path, file)
                df = pd.read_csv(
                    file_path,
                    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'funding_rate'] if is_future 
                           else ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    parse_dates=['timestamp'],
                    dtype={'funding_rate': np.float32} if is_future else None
                )
                df.set_index('timestamp', inplace=True)
                all_files.append(df)
                
        return pd.concat(all_files).sort_index()
        
    def run_backtest(self, params: Dict[str, Any]):
        """执行回测"""
        if self.preprocessed_data is None:
            self.preprocess_data()
            
        print(f"开始回测，参数配置：{params}")
        
        # 初始化回测状态
        portfolio = {
            'cash': params['initial_capital'],
            'position': 0,
            'equity': [],
            'trades': []
        }
        
        # 策略逻辑主循环
        for idx, row in self.preprocessed_data.iterrows():
            current_spread = row['price_spread']
            expected_profit = abs(current_spread) - params['spread_threshold']
            
            # 开仓逻辑
            if abs(current_spread) > params['spread_threshold'] and portfolio['position'] == 0:
                # 计算合约数量
                qty = portfolio['cash'] * params['leverage'] / row['close_spot']
                direction = 1 if current_spread > 0 else -1
                
                # 记录交易
                portfolio['position'] = qty * direction
                portfolio['cash'] -= qty * row['close_spot'] * params['fee_rate']
                portfolio['trades'].append({
                    'timestamp': row.name,
                    'type': 'open',
                    'qty': qty,
                    'price_spot': row['close_spot'],
                    'price_future': row['close_future'],
                    'pnl': 0,
                    'exit_time': None
                })
            
            # 平仓逻辑
            elif portfolio['position'] != 0 and (
                (row.name - portfolio['trades'][-1]['timestamp']).total_seconds() >= params['max_hold_seconds'] 
                or expected_profit <= 0
            ):
                entry_time = portfolio['trades'][-1]['timestamp']
                exit_time = row.name
                if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                    holding_seconds = (exit_time - entry_time).total_seconds()
                else:
                    holding_seconds = 0
                holding_hours = holding_seconds / 3600
                funding_times = int(holding_hours // 8)
                funding_pnl = portfolio['position'] * row['funding_rate'] * funding_times
                
                spread_pnl = portfolio['position'] * (
                    (row['close_future'] - portfolio['trades'][-1]['price_future']) -
                    (row['close_spot'] - portfolio['trades'][-1]['price_spot'])
                )
                
                pnl = spread_pnl + funding_pnl
                
                portfolio['cash'] += pnl
                portfolio['position'] = 0
                portfolio['trades'][-1].update({
                    'pnl': pnl,
                    'exit_time': idx,
                    'spread_pnl': spread_pnl,
                    'funding_pnl': funding_pnl,
                    'total_pnl': pnl
                })
                
            # 记录权益曲线
            portfolio['equity'].append({
                'timestamp': row.name,
                'value': portfolio['cash'] + portfolio['position'] * (
                    row['close_future'] - row['close_spot']
                )
            })
            
        # 生成回测报告
        results = {
            'final_equity': portfolio['cash'],
            'max_drawdown': self._calculate_max_drawdown(portfolio['equity']),
            'total_trades': len(portfolio['trades']),
            'win_rate': self._calculate_win_rate(portfolio['trades']),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio['equity']),
            'equity_curve': portfolio['equity'],
            'trades': portfolio['trades']
        }
        
        self._save_results(results, params)
        return results
        
    def _calculate_max_drawdown(self, equity_curve):
        values = [e['value'] for e in equity_curve]
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        return max_drawdown
        
    def _calculate_win_rate(self, trades):
        if not trades:
            return 0
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trades)
        
    def _calculate_sharpe_ratio(self, equity_curve):
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp').sort_index()
        try:
            returns = equity_df['value'].astype(float).pct_change().dropna()
        except KeyError:
            return 0
        if returns.empty:
            return 0
        risk_free_rate = 0.04
        risk_free_per_minute = risk_free_rate / (252 * 1440)
        excess_returns = returns - risk_free_per_minute
        if excess_returns.std() == 0:
            return 0
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 1440)
        return float(sharpe)
    
    def _save_results(self, results, params):
        trades_df = pd.DataFrame(results['trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df.to_csv(os.path.join(self.output_dir, 'trades.csv'), index=False)
        
        positions_df = pd.DataFrame([
            {'timestamp': e['timestamp'], 'position': e['value'] - results['final_equity'] + e['value']} 
            for e in results['equity_curve']
        ])
        positions_df.to_csv(os.path.join(self.output_dir, 'positions.csv'), index=False)
        
        initial_capital = params.get('initial_capital', 100000)
        total_profit = results['final_equity'] - initial_capital
        
        start_time = pd.to_datetime(results['equity_curve'][0]['timestamp'])
        end_time = pd.to_datetime(results['equity_curve'][-1]['timestamp'])
        total_days = (end_time - start_time).days + 1
        
        annualized_return = (1 + total_profit/initial_capital) ** (365/total_days) - 1 if total_days > 0 else 0
        
        total_funding = sum(t.get('funding_pnl',0) for t in results['trades'])
        funding_ratio = total_funding / total_profit if total_profit != 0 else 0
        
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        monthly_returns = equity_df['value'].resample('M').last().pct_change().dropna()
        monthly_volatility = monthly_returns.std() * np.sqrt(12) if not monthly_returns.empty else 0

        risk_metrics = {
            'final_equity': results['final_equity'],
            'max_drawdown': results['max_drawdown'],
            'annualized_volatility': np.std([e['value'] for e in results['equity_curve']]) * np.sqrt(252*1440),
            'sharpe_ratio': results['sharpe_ratio'],
            'var_95': np.percentile([e['value'] for e in results['equity_curve']], 5),
            'annualized_return': annualized_return,
            'funding_ratio%': funding_ratio * 100,
            'monthly_volatility': monthly_volatility,
            'total_days': total_days
        }
        pd.DataFrame([risk_metrics]).to_csv(os.path.join(self.output_dir, 'risk_metrics.csv'), index=False)
        results['risk_metrics'] = risk_metrics

    def print_backtest_results(self, results, params):
        print(f"\n========== 回测结果汇总 =========="
            f"\n初始资金: {params['initial_capital']:,.2f}"
            f"\n最终资金: {results['final_equity']:,.2f}"
            f"\n盈亏金额: {results['final_equity'] - params['initial_capital']:+,.2f}"
            f"\n总交易次数: {results['total_trades']}"
            f"\n胜率: {results['win_rate']:.2%}"
            f"\n最大回撤: {results['max_drawdown']:.2%}"
            f"\n夏普比率: {results['sharpe_ratio']:.2f}"
            f"\n年化收益率: {results['risk_metrics']['annualized_return']:.2%}"
            f"\n资金费收益占比: {results['risk_metrics']['funding_ratio%']:.2f}%"
            f"\n月度波动率: {results['risk_metrics']['monthly_volatility']:.4f}"
            f"\n回测总天数: {results['risk_metrics']['total_days']}天"
            f"\n===================================")

    def visualize_results(self, results, save_to_file=True):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            plt.rcParams['figure.figsize'] = (14, 8)
            plt.rcParams['figure.dpi'] = 100
            
            timestamps = [pd.to_datetime(e['timestamp']) for e in results['equity_curve']]
            equity_values = [e['value'] for e in results['equity_curve']]
            spot_prices = [row.close_spot for row in self.preprocessed_data.itertuples()]
            future_prices = [row.close_future for row in self.preprocessed_data.itertuples()]

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            plt.subplots_adjust(hspace=0.3)
            
            ax1.plot(timestamps, spot_prices, label='Spot Price', linewidth=1)
            ax1.plot(timestamps, future_prices, label='Future Price', linewidth=1)
            
            buy_signals = [pd.to_datetime(t['timestamp']) for t in results['trades'] 
                         if t.get('pnl', 0) > 0 and t['exit_time'] is not None]
            sell_signals = [pd.to_datetime(t['timestamp']) for t in results['trades'] 
                          if t.get('pnl', 0) <= 0 and t['exit_time'] is not None]
            ax1.scatter(buy_signals, [self.preprocessed_data.loc[ts].close_spot for ts in buy_signals], 
                      marker='^', color='g', s=100, label='Buy')
            ax1.scatter(sell_signals, [self.preprocessed_data.loc[ts].close_spot for ts in sell_signals], 
                      marker='v', color='r', s=100, label='Sell')
            ax1.set_title('Price & Trading Signals')
            ax1.legend()
            ax1.grid(True)

            positions = [e['value'] - results['final_equity'] + e['value'] for e in results['equity_curve']]
            ax2.fill_between(timestamps, positions, alpha=0.3, color='b', label='Position')
            ax2.set_ylabel('Position')
            ax2.grid(True)

            ax3.plot(timestamps, equity_values, label='Equity')
            max_dd_start = timestamps[np.argmax(np.maximum.accumulate(equity_values) - equity_values)]
            max_dd_end = timestamps[np.argmax(equity_values[:np.argmax(np.maximum.accumulate(equity_values) - equity_values)])]
            ax3.axvspan(max_dd_start, max_dd_end, alpha=0.3, color='red', label='Max Drawdown')
            ax3.set_title('Equity Curve')
            ax3.legend()
            ax3.grid(True)

            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_to_file:
                output_path = os.path.join(self.output_dir, 'backtest_result.png')
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                print(f"\n可视化结果已保存至：{output_path}")
                
            plt.show()
            
        except ImportError:
            print("提示：安装matplotlib可查看资金曲线 → pip install matplotlib")

if __name__ == "__main__":
    def main():
        try:
            engine = BacktestEngine()
            
            params = {
                "spread_threshold": 0.005,
                "leverage": 3,
                "max_hold_seconds": 4 * 3600,
                "fee_rate": 0.0002,
                "initial_capital": 1500
            }
            
            print("正在预处理数据...")
            engine.preprocess_data()
            
            print("开始回测...")
            results = engine.run_backtest(params)
            
            engine.print_backtest_results(results, params)
            engine.visualize_results(results)
                
        except FileNotFoundError as e:
            print(f"数据加载失败：{e}\n请检查market_data目录结构是否符合要求")
        except Exception as e:
            print(f"回测异常：{str(e)}")

    main()
