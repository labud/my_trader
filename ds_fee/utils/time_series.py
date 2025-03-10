import pandas as pd
import numpy as np
from typing import Tuple, Optional

def process_single_time_series(
    time_series: pd.DatetimeIndex,
    target_tz: str = 'UTC'
) -> pd.DatetimeIndex:
    """
    统一处理单个时间序列的时区和类型转换
    
    Args:
        time_series (pd.DatetimeIndex): 需要处理的时间序列
        target_tz (str, optional): 目标时区. Defaults to 'UTC'.
    
    Returns:
        pd.DatetimeIndex: 处理后的时间序列
    """
    # 确保输入数据类型正确
    if not isinstance(time_series, pd.DatetimeIndex):
        time_series = pd.DatetimeIndex(pd.to_datetime(time_series))
    
    # 检查并统一时区
    if time_series.tz is None:
        time_series = time_series.tz_localize('UTC')
    
    # 转换到目标时区
    time_series = time_series.tz_convert(target_tz)
    
    return time_series

def get_next_funding_time(current_time: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    """
    计算下一个资金费率结算时间点和距离该时间点的分钟数
    
    Args:
        current_time (pd.Timestamp): 当前时间，必须带有时区信息
    
    Returns:
        Tuple[pd.Timestamp, float]: 返回(下一个结算时间, 距离下一个结算时间的分钟数)
    """
    if current_time.tz is None:
        raise ValueError("current_time must be timezone-aware")
        
    next_funding_time = pd.Timestamp(current_time.date(), tz=current_time.tz)
    
    # 根据当前时间确定下一个结算时间点（UTC 0:00, 8:00, 16:00）
    if current_time.hour < 8:
        next_funding_time += pd.Timedelta(hours=8)
    elif current_time.hour < 16:
        next_funding_time += pd.Timedelta(hours=16)
    else:
        next_funding_time += pd.Timedelta(days=1)
        
    # 计算距离下一个资金费率结算时间的分钟数
    minutes_to_funding = (next_funding_time - current_time).total_seconds() / 60
    
    return next_funding_time, minutes_to_funding