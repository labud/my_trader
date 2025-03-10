import pandas as pd
import numpy as np
from typing import Dict, Any
from .data_processor import DataProcessor
from ds_fee.strategy.funding_arbitrage import should_open_position, should_close_position
from ds_fee.utils.time_series import get_next_funding_time

class BacktestCore:
    def __init__(self, config, data_processor):
        self.config = config
        self.data_processor = data_processor
        self.processor = data_processor  # 使用传入的处理器实例
        self.initial_balance = getattr(config, 'initial_balance', 100000.0)

    def _calculate_atr(self, window):
        """计算平均真实波动范围(Average True Range, ATR)"""
        if self.processor.preprocessed_data is None:
            raise ValueError("需要先预处理数据")

        # 创建数据副本避免修改原始数据
        df = self.processor.preprocessed_data.loc[:, ['high_spot', 'low_spot', 'close_spot']]
        
        # 提取计算ATR所需的价格数据
        high = df['high_spot']      # 当日最高价
        low = df['low_spot']        # 当日最低价
        close = df['close_spot'].shift(1)  # 前一日收盘价（通过shift(1)获取）
        
        # 计算三种真实波幅
        tr1 = abs(high - low)           # 当日价格区间：最高价 - 最低价
        tr2 = abs(high - close)         # 当日最高价与前收价之差
        tr3 = abs(low - close)          # 当日最低价与前收价之差
        
        # 将三个序列合并后取每行最大值，得到真实波幅TR
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算TR的window周期简单移动平均，返回整个ATR序列
        # rolling(window)创建移动窗口
        # mean()计算窗口内平均值
        # 使用min_periods=1确保在数据点不足时使用所有可用数据
        return true_range.rolling(window=window, min_periods=1).mean()

    def run_backtest(self, params: Dict[str, Any], verbose=False):
        """执行期现套利策略的回测逻辑

        Args:
            params (Dict[str, Any]): 策略参数字典，包含：
                - spread_threshold (float): 开仓价差阈值
                - min_funding_rate (float): 最小资金费率要求
                - risk_per_trade (float): 每笔交易的风险比例
                - leverage (float): 杠杆倍数
                - max_hold_seconds (int): 最大持仓时间（秒）
                - atr_window (int): ATR计算周期，默认为20
            verbose (bool, optional): 是否打印详细日志. Defaults to False.

        Returns:
            Dict[str, Any]: 回测结果字典，包含：
                - cash (float): 最终现金余额
                - max_drawdown (float): 最大回撤
                - total_trades (int): 总交易次数
                - win_rate (float): 胜率
                - sharpe_ratio (float): 夏普比率
                - equity_curve (List[Dict]): 权益曲线，每个元素包含:
                    - timestamp: 时间戳
                    - value: 账户价值
                    - position: 持仓数量
                - trades (List[Dict]): 交易记录，每个元素包含:
                    - timestamp: 开仓时间
                    - type: 交易类型(open/close)
                    - qty: 交易数量
                    - price_spot: 现货价格
                    - price_future: 期货价格
                    - pnl: 收益
                    - exit_time: 平仓时间
                    - spread_pnl: 价差收益
                    - funding_pnl: 资金费收益
                    - total_pnl: 总收益
                - risk_metrics (Dict): 风险指标，包含15+个高级风险分析指标

        策略说明:
            1. 开仓条件：
               - 现货与期货价差超过阈值
               - 资金费率方向符合预期
               - 当前无持仓
            2. 仓位管理：
               - 基于ATR的动态仓位计算
               - 考虑账户余额和杠杆限制
            3. 平仓条件：
               - 达到最大持仓时间
               - 预期利润消失
            4. 收益计算：
               - 价差收益：期现价差变动
               - 资金费收益：每8小时结算一次
        """
        if self.processor.preprocessed_data is None:
            self.processor.preprocess_data(verbose=verbose)
        preprocessed_data = self.processor.preprocessed_data

        # 初始化资产组合（关键数据结构）
        portfolio = {
            'cash': self.initial_balance,  # 可用现金
            'position': 0,  # 当前持仓方向（正数做多，负数做空）
            'equity': [],   # 每日权益曲线
            'trades': []    # 交易记录明细
        }

        # 动态仓位计算（基于波动率的风险控制）
        atr_window = params.get('atr_window', 20)  # 获取ATR计算周期，默认为20
        atrs = self._calculate_atr(window=atr_window)  # 使用配置的ATR周期

        # 核心交易逻辑（逐笔数据回放）
        for idx, row in preprocessed_data.iterrows():
            # 计算当前期现价差（期货价格 - 现货价格）
            # 正值表示期货溢价，负值表示期货折价
            current_spread = row['price_spread']  # 单位：USDT
            
            # 计算预期收益（绝对价差 - 开仓阈值）
            # 当预期收益为正时，表示有套利机会
            expected_profit = abs(current_spread) - params['spread_threshold']  # 单位：USDT
            
            # 计算价差趋势（使用过去60分钟的价差数据，增加判断周期）
            current_idx = preprocessed_data.index.get_loc(idx)
            start_idx = max(0, current_idx - 60)
            spread_history = preprocessed_data['price_spread'].iloc[start_idx:current_idx+1]
            spread_trend = spread_history.diff().mean()  # 正值表示价差扩大趋势，负值表示价差收窄趋势
            spread_volatility = spread_history.std()  # 计算价差波动率
            
            # 计算距离下一个资金费率结算时间的分钟数
            current_time = row.name
            next_funding_time, minutes_to_funding = get_next_funding_time(current_time)
            
            # 判断当前是否处于高风险时段（UTC 16:00-24:00）
            is_high_risk_period = 16 <= current_time.hour < 24
            
            #========== 开仓逻辑 ===========#
            # 使用封装的开仓判断函数
            should_open, direction, open_reason = should_open_position(current_spread, row, minutes_to_funding, is_high_risk_period, params, self.processor)
            
            if should_open and portfolio['position'] == 0:
                # 使用更长期的ATR来评估风险
                # 结合短期和长期波动率，以获得更稳定的风险评估
                current_idx = atrs.index.get_loc(idx)
                start_idx = max(0, current_idx - 480)
                recent_atr = atrs.iloc[current_idx]  # 当前ATR（短期波动率）
                long_term_atr = atrs.iloc[start_idx:current_idx+1].mean()  # 8小时(480分钟)平均ATR（长期波动率）
                volatility = max(recent_atr, long_term_atr)  # 取较大值作为最终波动率估计
                
                # 仓位计算考虑波动率
                # 基于账户风险管理的动态仓位计算
                # position_size = (可用资金 * 每笔风险比例) / (波动率 * 杠杆)
                position_size = (portfolio['cash'] * min(params['risk_per_trade'], 0.01)) / (volatility * params['leverage'])  # 限制每笔交易风险不超过1%
                
                # 实际可开仓量计算（考虑手续费）
                # 确保开仓金额不超过账户可用保证金
                spot_fee_rate = self.processor.get_fee_rate('spot', is_maker=True)
                future_fee_rate = self.processor.get_fee_rate('future', is_maker=True)
                total_fee_rate = spot_fee_rate + future_fee_rate  # 开仓总手续费率（现货+期货）
                # max_position = 可用资金 * 杠杆 / (现货价格 * (1 + 总手续费率))
                max_position = portfolio['cash'] * params['leverage'] / (row['close_spot'] * (1 + total_fee_rate))
                # 取两个限制中的较小值作为最终开仓数量
                qty = min(position_size, max_position)  # 单位：个/张
                
                # 更新持仓状态
                portfolio['position'] = qty * direction
                # 分别计算现货和期货的开仓手续费
                spot_open_fee = abs(qty) * row['close_spot'] * spot_fee_rate
                future_open_fee = abs(qty) * row['close_future'] * future_fee_rate
                total_open_fee = spot_open_fee + future_open_fee
                portfolio['cash'] -= total_open_fee  # 扣除开仓总手续费
                portfolio['trades'].append({
                    'timestamp': row.name,
                    'type': 'open',
                    'qty': qty,
                    'direction': direction,
                    'price_spot': row['close_spot'],
                    'price_future': row['close_future'],
                    'funding_rate': row['funding_rate'],  # 记录开仓时的资金费率
                    'spread': current_spread,
                    'volatility': volatility,
                    'reason': open_reason,
                    'pnl': 0,
                    'open_fee_cost': total_open_fee,  # 记录开仓手续费
                    'exit_time': None
                })
            
            #========== 平仓逻辑 ==========#
            elif portfolio['position'] != 0:
                entry_trade = portfolio['trades'][-1]
                holding_seconds = (row.name - entry_trade['timestamp']).total_seconds()
                
                # 使用封装的平仓判断函数
                should_close, close_reason = should_close_position(current_spread, row, entry_trade, minutes_to_funding, 
                                   holding_seconds, is_high_risk_period, params, self.processor)
                
                if should_close:
                    # 计算资金费收益（按固定时间点结算）
                    # 1. 计算持仓时间并转换为小时
                    holding_hours = holding_seconds / 3600  # 单位：小时

                    # 2. 使用工具函数计算资金费收益
                    from ds_fee.utils.funding_fee import calculate_funding_fee
                    funding_pnl = calculate_funding_fee(
                        start_time=entry_trade['timestamp'],
                        end_time=row.name,
                        position=portfolio['position'],
                        funding_rates=preprocessed_data['funding_rate']
                    )
                    
                    # 计算价差收益（期现价差变化带来的收益）
                    # spread_pnl = 持仓量 * (期货价差变化 - 现货价差变化)
                    # 正向持仓时（做多期货做空现货），期货上涨、现货下跌时盈利
                    # 反向持仓时（做空期货做多现货），期货下跌、现货上涨时盈利
                    spread_pnl = portfolio['position'] * (
                        (row['close_future'] - entry_trade['price_future']) -  # 期货价格变化
                        (row['close_spot'] - entry_trade['price_spot'])      # 现货价格变化
                    )  # 单位：USDT
                    
                    # 计算总收益（考虑手续费成本）
                    # 1. 计算平仓手续费（现货和期货分别计算）
                    spot_close_fee = abs(portfolio['position']) * row['close_spot'] * self.processor.get_fee_rate('spot', is_maker=False)
                    future_close_fee = abs(portfolio['position']) * row['close_future'] * self.processor.get_fee_rate('future', is_maker=False)
                    close_total_fee = spot_close_fee + future_close_fee
                    # 2. 计算净收益 = 价差收益 + 资金费收益 - 手续费
                    pnl = spread_pnl + funding_pnl - close_total_fee  # 单位：USDT
                    
                    # 更新资产组合状态
                    portfolio['cash'] += pnl  # 更新可用资金
                    portfolio['position'] = 0  # 清空持仓
                    
                    # 记录平仓信息
                    close_reason = "达到最大持仓时间" if holding_seconds >= params['max_hold_seconds'] else "预期收益消失"
                    entry_trade.update({
                        'type': 'close',
                        'pnl': pnl - entry_trade['open_fee_cost'],
                        'exit_time': row.name,
                        'exit_price_spot': row['close_spot'],
                        'exit_price_future': row['close_future'],
                        'spread_pnl': spread_pnl,
                        'funding_pnl': funding_pnl,
                        'close_fee_cost': close_total_fee,
                        'holding_hours': holding_hours,
                        'close_reason': close_reason
                    })
            
            # 计算当前权益（考虑未实现盈亏和手续费）
            unrealized_pnl = 0
            if portfolio['position'] != 0:
                unrealized_spread_pnl = portfolio['position'] * (
                    (row['close_future'] - portfolio['trades'][-1]['price_future']) -
                    (row['close_spot'] - portfolio['trades'][-1]['price_spot'])
                )
                unrealized_fee = self._calculate_fee(portfolio['position'], row['close_spot'], 'spot', False)
                unrealized_pnl = unrealized_spread_pnl - unrealized_fee
            
            current_value = portfolio['cash'] + unrealized_pnl
            portfolio['equity'].append({
                'timestamp': row.name,
                'value': current_value,
                'position': portfolio['position'],
                'unrealized_pnl': unrealized_pnl
            })
        return self._generate_results(portfolio, params)

    def _generate_results(self, portfolio, params):
        """生成标准化的结果结构"""
        return {
            'cash': portfolio['cash'],
            'max_drawdown': self._calculate_max_drawdown(portfolio['equity']),
            'total_trades': len(portfolio['trades']),
            'win_rate': self._calculate_win_rate(portfolio['trades']),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio['equity']),
            'equity_curve': portfolio['equity'],
            'trades': portfolio['trades'],
            'risk_metrics': self._calculate_risk_metrics({
                'cash': portfolio['cash'],
                'max_drawdown': self._calculate_max_drawdown(portfolio['equity']),
                'total_trades': len(portfolio['trades']),
                'win_rate': self._calculate_win_rate(portfolio['trades']),
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio['equity']),
                'equity_curve': portfolio['equity'],
                'trades': portfolio['trades']
            }, params)
        }

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
    
    def _calculate_fee(self, position, price, market_type, is_maker=True):
        """计算交易手续费
    
        Args:
            position (float): 交易数量
            price (float): 交易价格
            market_type (str): 市场类型（'spot'或'future'）
            is_maker (bool, optional): 是否为挂单. Defaults to True.
    
        Returns:
            float: 交易手续费（USDT）
        """
        fee_rate = self.processor.get_fee_rate(market_type, is_maker)
        return abs(position) * price * fee_rate
    
    def _calculate_risk_metrics(self, results, params):
        """计算高阶风险指标（核心风险分析引擎）
        
        参数:
            results: 包含回测结果的字典
            params: 策略参数
            
        返回:
            包含15+个风险指标的字典，用于全面评估策略表现
        
        指标说明:
            - annualized_return: 年化收益率（复利计算）
            - funding_ratio%: 资金费收益占总收益比例
            - var_95: 95%置信度下的风险价值（最大潜在损失）
            - monthly_volatility: 月波动率（标准差年化）
        """
        # 初始化参数
        initial_capital = self.initial_balance
        total_profit = results['cash'] - initial_capital  # 总净利润
        
        #====== 时间维度分析 ======#
        # 计算回测周期总天数
        start_time = self.data_processor.preprocessed_data.index.min().to_pydatetime()
        end_time = self.data_processor.preprocessed_data.index.max().to_pydatetime()
        total_days = (end_time - start_time).days + 1  # 含当天
        
        #====== 收益指标 ======#
        # 年化收益率（复利公式）
        if total_days > 0:
            annualized_return = (1 + total_profit/initial_capital) ** (365/total_days) - 1 
        else:
            annualized_return = 0
        
        #====== 资金费分析 ======#
        # 计算资金费收益占比（套利策略核心收益来源之一）
        total_funding = sum(t.get('funding_pnl',0) for t in results['trades'])
        # 防止除零错误处理
        funding_ratio = total_funding / total_profit if total_profit != 0 else 0  
        
        #====== 波动率指标 ======#
        # 将权益数据转为DataFrame便于计算
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # 月波动率（标准差年化）
        monthly_returns = equity_df['value'].resample('ME').last().pct_change().dropna()
        monthly_volatility = monthly_returns.std() * np.sqrt(12) if not monthly_returns.empty else 0
        
        #====== 风险价值 ======#
        # 计算95% VaR（历史模拟法）
        equity_values = [e['value'] for e in results['equity_curve']]
        var_95 = np.percentile(equity_values, 5)  # 最差5%分位点的权益值
        
        return {
            # 基础指标
            'cash': results['cash'],  # 最终现金余额
            'max_drawdown': results['max_drawdown'],  # 最大回撤（来自其他函数）
            'total_days': total_days,  # 回测总天数
            'total_trades': results['total_trades'],  # 总交易次数
            
            # 收益指标
            'net_profit': total_profit,  # 净利润
            'annualized_return': annualized_return,  # 年化收益率（复利）
            'win_rate': results['win_rate'],  # 胜率
            
            # 风险指标
            'annualized_volatility': np.std(equity_values) * np.sqrt(252*1440),  # 年化波动率（分钟数据折算）
            'sharpe_ratio': results['sharpe_ratio'],  # 夏普比率
            'var_95': var_95,  # 风险价值（95%置信度）
            'monthly_volatility': monthly_volatility,  # 月波动率
            
            # 套利特有指标
            'funding_ratio%': funding_ratio * 100,  # 资金费收益占比
            'avg_holding_hours': self._calc_avg_holding_time(results['trades']),  # 平均持仓时间（需补充实现）
            
            # 策略参数回显（用于参数调优分析）
            'params': params  
        }

    #----- 补充指标函数示例 -----#
    def _calc_avg_holding_time(self, trades):
        """计算平均持仓时间（小时）"""
        if not trades:
            return 0
        total_seconds = sum(
            (t['exit_time'] - t['timestamp']).total_seconds()
            for t in trades if t['exit_time'] is not None
        )
        return total_seconds / len(trades) / 3600
