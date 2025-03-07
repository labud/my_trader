import pandas as pd
from typing import Dict, Any, Tuple

def should_open_position(current_spread: float,
                       row: pd.Series,
                       minutes_to_funding: float,
                       is_high_risk_period: bool,
                       params: Dict[str, Any],
                       processor) -> Tuple[bool, int, str]:
    """判断是否应该开仓
    
    Args:
        current_spread (float): 当前期现价差
        row (pd.Series): 当前时间点的市场数据
        minutes_to_funding (float): 距离下一个资金费率结算的分钟数
        is_high_risk_period (bool): 是否处于高风险时段
        params (dict): 策略参数
        processor: 数据处理器实例
        
    Returns:
        tuple: (should_open, direction, reason)
            - should_open (bool): 是否应该开仓
            - direction (int): 开仓方向，1为做多期货，-1为做空期货
            - reason (str): 开仓原因
    """
    # 基础条件：价差超过阈值
    if abs(current_spread) <= params['spread_threshold']:
        return False, 0, ""
        
    # 时间窗口条件：放宽结算时间窗口（5-30分钟）
    if not (5 <= minutes_to_funding <= 30):
        return False, 0, ""
        
    # 高风险时段提高开仓门槛
    risk_multiplier = 1.5 if is_high_risk_period else 1.1  # 降低风险系数
    if abs(current_spread) < params['spread_threshold'] * risk_multiplier:
        return False, 0, ""

    # 计算预期收益是否足够覆盖手续费
    spot_fee_rate = processor.get_fee_rate('spot', is_maker=True) + processor.get_fee_rate('spot', is_maker=False)
    future_fee_rate = processor.get_fee_rate('future', is_maker=True) + processor.get_fee_rate('future', is_maker=False)
    total_fee_rate = spot_fee_rate + future_fee_rate # 考虑开仓和平仓的总手续费
    
    # 预期收益需要显著高于手续费成本（降低收益要求）
    min_profit_multiplier = 2.0 if is_high_risk_period else 1.8  # 降低预期收益要求
    expected_fee_cost = row['close_spot'] * total_fee_rate
    if abs(current_spread) - expected_fee_cost * min_profit_multiplier <= 0:
        return False, 0, ""

    # 资金费率方向验证（降低资金费率要求）
    if current_spread > 0 and row['funding_rate'] > params['min_funding_rate']:
        direction = -1  # 期现价差为正，做空期货+做多现货
        reason = "正向价差超阈值且资金费率为正，接近结算时间"
        return True, direction, reason
    elif current_spread < 0 and row['funding_rate'] < -params['min_funding_rate']:
        direction = 1   # 期现价差为负，做多期货+做空现货
        reason = "反向价差超阈值且资金费率为负，接近结算时间"
        return True, direction, reason
        
    return False, 0, ""

def should_close_position(current_spread: float,
                        row: pd.Series,
                        entry_trade: Dict[str, Any],
                        minutes_to_funding: float,
                        holding_seconds: float,
                        is_high_risk_period: bool,
                        params: Dict[str, Any],
                        processor) -> Tuple[bool, str]:
    """判断是否应该平仓
    
    Args:
        current_spread (float): 当前期现价差
        row (pd.Series): 当前时间点的市场数据
        entry_trade (dict): 开仓交易记录
        minutes_to_funding (float): 距离下一个资金费率结算的分钟数
        holding_seconds (float): 当前持仓时间（秒）
        is_high_risk_period (bool): 是否处于高风险时段
        params (dict): 策略参数
        processor: 数据处理器实例
        
    Returns:
        tuple: (should_close, reason)
            - should_close (bool): 是否应该平仓
            - reason (str): 平仓原因
    """
    entry_spread = entry_trade['spread']
    spread_change = current_spread - entry_spread
    spread_change_pct = abs(spread_change / entry_spread)
    
    # 计算当前收益是否足够覆盖手续费
    spot_fee_rate = processor.get_fee_rate('spot', is_maker=False)
    future_fee_rate = processor.get_fee_rate('future', is_maker=False)
    total_fee_rate = spot_fee_rate + future_fee_rate
    current_fee_cost = row['close_spot'] * total_fee_rate
    
    # 根据持仓方向判断价差收敛
    if entry_trade['direction'] < 0:  # 做空期货+做多现货
        expected_profit = entry_spread - current_spread  # 期待价差收窄
    else:  # 做多期货+做空现货
        expected_profit = current_spread - entry_spread  # 期待价差扩大
        
    # 止损条件（考虑手续费成本）
    stop_loss_threshold = params['spread_threshold'] * (2.2 if is_high_risk_period else 1.8)
    if expected_profit <= -stop_loss_threshold:
        return True, "止损：预期收益显著低于阈值"
        
    # 获利了结（确保收益足够覆盖手续费）
    if expected_profit >= current_fee_cost * 2:
        return True, "止盈：收益已覆盖手续费成本"
        
    # 时间条件
    max_hold_time = params['max_hold_seconds'] * (0.8 if is_high_risk_period else 1.0)
    if holding_seconds >= max_hold_time:
        return True, "达到最大持仓时间"
        
    # 价差剧烈波动
    volatility_threshold = 0.3 if is_high_risk_period else 0.25
    if holding_seconds > 300 and spread_change_pct > volatility_threshold:
        return True, "价差波动超过阈值"
        
    # 资金费率结算时间管理
    if minutes_to_funding > 10 or minutes_to_funding <= 1:
        return True, "接近资金费率结算时间"
        
    return False, ""