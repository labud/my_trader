import pandas as pd
import numpy as np
import os
from typing import Dict, Any

class BacktestEngine:
    def __init__(self, config):
        """初始化回测引擎（进程安全版本）"""
        # 将Config对象转换为字典并添加默认值
        self.config = {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
        self.actual_start_date = None
        self.actual_end_date = None
        self.actual_funding_rate = None
        self.data_dir = getattr(config, 'data_dir', 'ds_fee/market_data/')
        self.output_dir = getattr(config, 'output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocessed_data = None
        self.spot_path = os.path.join(self.data_dir, "1m/spot")
        self.future_path = os.path.join(self.data_dir, "1m/future")
        
    def preprocess_data(self, verbose=True):
        """预处理市场数据（只执行一次）"""
        if self.preprocessed_data is not None:
            return
        if verbose:
            print("开始加载并预处理市场数据...")
        
        spot_df = self._load_minute_data(self.spot_path, is_future=False)
        future_df = self._load_minute_data(self.future_path, is_future=True)
        
        # 放宽合并条件并保留更多数据
        # 使用outer join保留所有时间戳并填充缺失值
        merged_df = pd.merge(
            spot_df.reset_index(),
            future_df.reset_index(),
            on='timestamp',
            how='outer',
            suffixes=('_spot', '_future')
        ).sort_values('timestamp')
        
        # 前向填充缺失值
        merged_df[['close_spot', 'close_future']] = merged_df[['close_spot', 'close_future']].ffill()
        merged_df = merged_df.dropna(subset=['close_spot', 'close_future']).set_index('timestamp')
        
        merged_df['price_spread'] = merged_df['close_future'] - merged_df['close_spot']
        merged_df['funding_yield'] = merged_df['funding_rate'] * 24 * 365
        
        # 转换为上海时区并包含完整时间信息
        tz = 'Asia/Shanghai'
        start = merged_df.index.min().tz_convert(tz).strftime('%Y-%m-%d %H:%M')
        end = merged_df.index.max().tz_convert(tz).strftime('%Y-%m-%d %H:%M')
        self.actual_start_date = start
        self.actual_end_date = end
        self.actual_funding_rate = merged_df['funding_rate'].mean()
        self.preprocessed_data = merged_df
        
        if verbose:
            print(f"数据预处理完成，总数据量：{len(merged_df)} 条")

    def _load_minute_data(self, path: str, is_future: bool) -> pd.DataFrame:
        """加载分钟级别数据（递归加载子目录）"""
        all_files = []
        for root, dirs, files in os.walk(path):
            all_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
        
        # 添加加载进度提示
        print(f"正在加载{len(all_files)}个数据文件，路径：{path}")
        
        dfs = []
        # 按文件名中的日期排序
        all_files.sort(key=lambda x: pd.to_datetime(os.path.basename(x).split('.')[0]))
        for f in all_files:
            try:
                df = pd.read_csv(
                    f,
                    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'funding_rate'] if is_future 
                           else ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    parse_dates=['timestamp'],
                    dtype={'funding_rate': np.float32} if is_future else None
                ).set_index('timestamp')
                dfs.append(df)
            except Exception as e:
                print(f"加载文件 {f} 失败: {str(e)}")
                continue
                
        combined_df = pd.concat(dfs).sort_index()
        print(f"合并后总数据量：{len(combined_df)} 条，时间范围：{combined_df.index.min()} ~ {combined_df.index.max()}")
        return combined_df

    def run_backtest(self, params: Dict[str, Any], verbose=False, saveResults=True):
        """执行回测"""
        # 设置默认参数
        merged_params = {
            'min_funding_rate': 0.0002,
            'leverage': 2,
            'max_hold_seconds': 8*3600,
            'risk_per_trade': 0.02,
            **self.config,
            **params,
            'fee_rate': float(self.actual_funding_rate or 0.0002)
        }
        
        if self.preprocessed_data is None:
            self.preprocess_data(verbose=verbose)

        portfolio = {
            'cash': params.get('initial_capital', 100000.0),
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
                # 增加资金费率方向过滤和风险管理
                if current_spread > 0 and row['funding_rate'] > params['min_funding_rate']:
                    direction = -1  # 做空期货（价差为正且资金费率为正）
                elif current_spread < 0 and row['funding_rate'] < -params['min_funding_rate']:
                    direction = 1   # 做多期货（价差为负且资金费率为负）
                else:
                    continue  # 不符合资金费率条件时不交易
                
                # 基于波动率的仓位管理
                atr = self._calculate_atr(window=20)
                position_size = (portfolio['cash'] * params['risk_per_trade']) / (atr * params['leverage'])
                qty = min(position_size, 
                        portfolio['cash'] * params['leverage'] / row['close_spot'])
                
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
                holding_seconds = (exit_time - entry_time).total_seconds() if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp) else 0
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
                
            # 记录权益曲线（确保每个时间点都有记录）
            current_value = portfolio['cash'] + portfolio['position'] * (row['close_future'] - row['close_spot'])
            portfolio['equity'].append({
                'timestamp': row.name,
                'value': current_value,
                'position': portfolio['position']  # 添加仓位记录
            })

        # 先初始化基础结果
        results = {
            'final_equity': portfolio['cash'],
            'max_drawdown': self._calculate_max_drawdown(portfolio['equity']),
            'total_trades': len(portfolio['trades']),
            'win_rate': self._calculate_win_rate(portfolio['trades']),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio['equity']),
            'equity_curve': portfolio['equity'],
            'trades': portfolio['trades']
        }
        
        # 单独计算风险指标并添加
        results['risk_metrics'] = self._calculate_risk_metrics(results, params)

        if saveResults:
            self._save_results(results, params)
        return results

    def _calculate_atr(self, window=20):
        """计算平均真实波幅(ATR)"""
        if self.preprocessed_data is None:
            return 0
            
        df = self.preprocessed_data.copy()
        df['high_low'] = df['high_future'] - df['low_future']
        df['high_close'] = np.abs(df['high_future'] - df['close_future'].shift())
        df['low_close'] = np.abs(df['low_future'] - df['close_future'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        return df['tr'].rolling(window=window).mean().iloc[-1]

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
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 1440))
    
    def _calculate_risk_metrics(self, results, params):
        """独立风险指标计算"""
        initial_capital = params.get('initial_capital', 100000)
        total_profit = results['final_equity'] - initial_capital
        
        # 使用预处理数据的时间范围计算总天数
        start_time = self.preprocessed_data.index.min().to_pydatetime()
        end_time = self.preprocessed_data.index.max().to_pydatetime()
        total_days = (end_time - start_time).days + 1  # 包含首尾两天
        
        annualized_return = (1 + total_profit/initial_capital) ** (365/total_days) - 1 if total_days > 0 else 0
        
        total_funding = sum(t.get('funding_pnl',0) for t in results['trades'])
        funding_ratio = total_funding / total_profit if total_profit != 0 else 0
        
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        monthly_returns = equity_df['value'].resample('ME').last().pct_change().dropna()
        monthly_volatility = monthly_returns.std() * np.sqrt(12) if not monthly_returns.empty else 0

        return {
            'final_equity': results['final_equity'],
            'max_drawdown': results['max_drawdown'],
            'annualized_volatility': np.std([e['value'] for e in results['equity_curve']]) * np.sqrt(252*1440),
            'sharpe_ratio': results['sharpe_ratio'],
            'var_95': np.percentile([e['value'] for e in results['equity_curve']], 5),
            'annualized_return': annualized_return,
            'funding_ratio%': funding_ratio * 100,
            'monthly_volatility': monthly_volatility,
            'total_days': total_days,
            'total_trades': results['total_trades'],
            'net_profit': total_profit,
            'win_rate': results['win_rate']
        }

    def _save_results(self, results, params):
        """数据存储方法"""
        # 保存交易记录
        trades_df = pd.DataFrame(results['trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df.to_csv(os.path.join(self.output_dir, 'trades.csv'), index=False)
        
        # 保存持仓记录
        positions_df = pd.DataFrame([
            {'timestamp': e['timestamp'], 'position': e['value'] - results['final_equity'] + e['value']} 
            for e in results['equity_curve']
        ])
        positions_df.to_csv(os.path.join(self.output_dir, 'positions.csv'), index=False)
        
        # 保存风险指标
        pd.DataFrame([results['risk_metrics']]).to_csv(
            os.path.join(self.output_dir, 'risk_metrics.csv'), 
            index=False
        )

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
            import matplotlib.font_manager as fm
            
            # 配置中文字体
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            plt.rcParams['figure.figsize'] = (14, 8)
            plt.rcParams['figure.dpi'] = 100
            
            # 获取字体属性
            font_path = '/System/Library/Fonts/PingFang.ttc'
            font_prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None
            
            # 确保使用统一的时间戳序列
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            # 统一时区并精确对齐到分钟级
            equity_df['timestamp'] = equity_df['timestamp'].dt.tz_localize(None).dt.floor('min')
            
            # 统一时区处理并移除时区信息
            preprocessed_data = self.preprocessed_data.tz_localize(None).copy()
            preprocessed_data.index = preprocessed_data.index.tz_localize(None).floor('min')
            
            # 创建完整时间索引（包含所有数据点）
            full_index = pd.date_range(
                start=preprocessed_data.index.min(),
                end=preprocessed_data.index.max(),
                freq='1min'
            )
            
            # 重新索引对齐数据
            # 使用索引代替timestamp列
            merged_df = equity_df.set_index('timestamp').reindex(full_index, method='ffill')
            
            # 初始化position_df并设置时区
            position_df = pd.DataFrame(results['equity_curve'])
            position_df['timestamp'] = pd.to_datetime(position_df['timestamp']).dt.tz_localize(None)
            position_df.set_index('timestamp', inplace=True)
            position_df.index = position_df.index.tz_localize('UTC')
            
            # 统一时区处理
            merged_df.index = merged_df.index.tz_localize('UTC')
            # 合并市场数据
            merged_df = merged_df.merge(
                self.preprocessed_data[['close_spot', 'close_future']],
                how='left',
                left_index=True,
                right_index=True
            )

            # 合并仓位数据并确保列名正确
            position_data = position_df[['position']].copy()
            position_data.columns = ['position']  # 明确设置列名
            
            # 直接赋值避免合并冲突
            merged_df['position'] = position_data['position']
            # 前向填充仓位数据
            merged_df['position'] = merged_df['position'].ffill().fillna(0)
            
            # 调试日志：检查合并后的列
            print("合并后的列:", merged_df.columns.tolist())
            
            timestamps = merged_df.index.tolist()
            equity_values = merged_df['value'].tolist()
            spot_prices = merged_df['close_spot'].tolist()
            future_prices = merged_df['close_future'].tolist()

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
            ax1.set_title('价格与交易信号', fontsize=12)
            ax1.set_ylabel('价格（USDT）', fontsize=10)
            ax1.legend(loc='upper left', frameon=False)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 直接从对齐后的数据框获取仓位数据
            positions = merged_df['position'].values
            ax2.fill_between(timestamps, positions, alpha=0.3, color='b', label='仓位')
            ax2.set_title('仓位变化', fontsize=12)
            ax2.set_ylabel('仓位数量', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper right', frameon=False)

            ax3.plot(timestamps, equity_values, label='资金曲线', linewidth=1.5)
            ax3.set_title('资金曲线与最大回撤', fontsize=12)
            ax3.set_ylabel('资金价值（USDT）', fontsize=10)
            # 使用回测结果中的最大回撤数据
            if results['max_drawdown'] > 0:
                # 从权益曲线中找到最大回撤时间段
                peak_value = max([e['value'] for e in results['equity_curve']])
                trough_value = peak_value * (1 - results['max_drawdown'])
                
                # 精确查找实际峰值索引
                peak_index = np.argmax([e['value'] for e in results['equity_curve']])
                
                # 查找波谷索引（从峰值之后开始）
                trough_index = peak_index + np.argmin(
                    [e['value'] for e in results['equity_curve'][peak_index:]])
                
                # 索引越界保护
                trough_index = min(trough_index, len(results['equity_curve'])-1)
                
                # 验证索引有效性
                if not (0 <= peak_index < len(results['equity_curve'])) or not (0 <= trough_index < len(results['equity_curve'])):
                    raise ValueError(f"无效索引 peak_index={peak_index}, trough_index={trough_index}")
                
                # 转换为图表时间戳
                max_dd_start = pd.to_datetime(results['equity_curve'][peak_index]['timestamp'])
                max_dd_end = pd.to_datetime(results['equity_curve'][trough_index]['timestamp'])
                
                # 标注到图表
                ax3.axvspan(max_dd_start, max_dd_end, alpha=0.3, color='red', 
                            label=f'Max Drawdown ({results["max_drawdown"]:.2%})')
            else:
                print("策略表现稳健，未检测到资金回撤")
            ax3.set_title('Equity Curve')
            ax3.legend()
            ax3.grid(True)

            # 扩展时间轴显示范围
            locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
            locator.intervald[3] = [7]  # 强制显示周刻度
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = [
                '%Y',        # 年
                '%m-%d',     # 月-日
                '%d',        # 日
                '%H:%M',     # 小时
                '%H:%M',     # 分钟
                '%S'         # 秒
            ]
            formatter.offset_formats = [
                '', 
                '',
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            
            plt.xticks(rotation=35, ha='right', fontsize=8)
            plt.subplots_adjust(bottom=0.15)
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
            from ds_fee.config import load_base_config
            config = load_base_config()
            engine = BacktestEngine(config)
            
            params = {
                "spread_threshold": 0.015,
                "leverage": 2,
                "max_hold_seconds": 8 * 3600,
                "fee_rate": 0.0002,
                "initial_capital": 5000,
                "min_funding_rate": 0.0002,  # 新增资金费率过滤阈值
                "risk_per_trade": 0.02       # 新增每笔交易风险比例
            }
            
            engine.preprocess_data()
            results = engine.run_backtest(params, verbose=True)
            engine.print_backtest_results(results, params)
            engine.visualize_results(results)
    
        except Exception as e:
            print(f"发生错误: {str(e)}")
            raise
    
    main()
