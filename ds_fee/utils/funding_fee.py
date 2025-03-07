import pandas as pd

def calculate_funding_fee(start_time, end_time, position, funding_rates):
    """
    计算资金费率收益
    
    Args:
        start_time (pd.Timestamp): 开仓时间
        end_time (pd.Timestamp): 平仓时间
        position (float): 持仓数量
        funding_rates (pd.Series): 资金费率数据，索引为时间戳，值为资金费率
        
    Returns:
        float: 资金费收益（USDT）
    """
    # 初始化资金费收益
    funding_pnl = 0
    
    # 确保时间戳带有时区信息
    if start_time.tz is None:
        raise ValueError("start_time must be timezone-aware")
    if end_time.tz is None:
        raise ValueError("end_time must be timezone-aware")
    
    # 生成持仓期间的所有结算时间点
    funding_times = []
    current_time = pd.Timestamp(start_time.date(), tz=start_time.tz) + pd.Timedelta(hours=0)  # 当天的UTC 0点
    
    while current_time <= end_time:
        if current_time >= start_time:
            funding_times.append(current_time)
        current_time += pd.Timedelta(hours=8)  # 每隔8小时一个结算点
    
    # 计算每个结算时间点的资金费收益
    for funding_time in funding_times:
        # 获取该结算时间点最近的资金费率
        period_rate = funding_rates.loc[:funding_time].iloc[-1]
        funding_pnl += position * period_rate  # 使用该周期的实际资金费率
    
    return funding_pnl