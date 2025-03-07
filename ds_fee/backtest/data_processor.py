import pandas as pd
import numpy as np
import os
from typing import Dict, Any

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.preprocessed_data = None
        self.actual_start_date = None
        self.actual_end_date = None
        self.actual_funding_rate = None
        # 从配置文件读取手续费率
        self.fees = getattr(config, 'fees', {
            'spot': {'maker': 0.001, 'taker': 0.001, 'bnb_discount': 0.25},
            'future': {'maker': 0.0002, 'taker': 0.0004}
        })
        self.data_dir = getattr(config, 'data_dir', 'ds_fee/market_data/')
        self.spot_path = os.path.join(self.data_dir, "1m/spot")
        self.future_path = os.path.join(self.data_dir, "1m/future")

    def get_fee_rate(self, market_type: str, is_maker: bool = True, use_bnb: bool = True) -> float:
        """获取交易手续费率

        Args:
            market_type (str): 市场类型，'spot'或'future'
            is_maker (bool, optional): 是否为挂单. Defaults to True.
            use_bnb (bool, optional): 已废弃参数，不再使用. Defaults to True.

        Returns:
            float: 手续费率
        """
        fee_type = 'maker' if is_maker else 'taker'
        return self.fees.get(market_type, {}).get(fee_type, 0.001)

    def preprocess_data(self, verbose=True):
        """统一预处理市场数据，合并现货和期货数据并计算相关指标

        Args:
            verbose (bool, optional): 是否打印处理过程信息. Defaults to True.

        Returns:
            None: 处理后的数据存储在 self.preprocessed_data 中，结构如下：
            pd.DataFrame:
                Index:
                    timestamp (datetime64[ns, UTC]) - 时间戳索引
                Columns:
                    - open_spot (float): 现货开盘价
                    - high_spot (float): 现货最高价
                    - low_spot (float): 现货最低价
                    - close_spot (float): 现货收盘价
                    - volume_spot (float): 现货交易量
                    - open_future (float): 期货开盘价
                    - high_future (float): 期货最高价
                    - low_future (float): 期货最低价
                    - close_future (float): 期货收盘价
                    - volume_future (float): 期货交易量
                    - funding_rate (float): 资金费率
                    - price_spread (float): 期现差价 (= close_future - close_spot)
                    - funding_yield (float): 年化资金费率 (= funding_rate * 24 * 365)

        Example:
            timestamp                open_spot  high_spot  low_spot  close_spot  volume_spot  ...  funding_rate  price_spread  funding_yield
            2023-01-01 00:00:00     16500.0    16550.0   16480.0    16520.0     100.5      ...     0.0001        100.0         0.876
            2023-01-01 00:01:00     16520.0    16570.0   16500.0    16540.0     98.2       ...     0.0001        100.0         0.876
        """
        if self.preprocessed_data is not None:
            return
            
        spot_df = self._load_minute_data(self.spot_path, is_future=False)
        future_df = self._load_minute_data(self.future_path, is_future=True)

        merged_df = pd.merge(
            spot_df.reset_index(),
            future_df.reset_index(),
            on='timestamp',
            how='outer',
            suffixes=('_spot', '_future')
        ).sort_values('timestamp')
        
        merged_df[['close_spot', 'close_future']] = merged_df[['close_spot', 'close_future']].ffill()
        merged_df = merged_df.dropna(subset=['close_spot', 'close_future']).set_index('timestamp')
        
        merged_df['price_spread'] = merged_df['close_future'] - merged_df['close_spot']
        merged_df['funding_yield'] = merged_df['funding_rate'] * 24 * 365
        
        tz = 'Asia/Shanghai'
        self.actual_start_date = merged_df.index.min().tz_convert(tz).strftime('%Y-%m-%d %H:%M')
        self.actual_end_date = merged_df.index.max().tz_convert(tz).strftime('%Y-%m-%d %H:%M')
        self.actual_funding_rate = merged_df['funding_rate'].mean()
        self.dynamic_fee_rate = merged_df['fee_rate'].mean() if 'fee_rate' in merged_df else 0.0002
        self.preprocessed_data = merged_df

    def _load_minute_data(self, path: str, is_future: bool) -> pd.DataFrame:
        """加载分钟级别数据"""
        all_files = []
        for root, dirs, files in os.walk(path):
            all_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            
        dfs = []
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
                
        return pd.concat(dfs).sort_index()
