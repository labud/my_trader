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