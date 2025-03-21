# binance_data_fetcher.py（优化版）
import ccxt
import os
import pandas as pd
import pytz
import re
import requests
import time
from datetime import datetime, timedelta

CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '1m',
    'days': 180,
    'chunk_size': 1000,
    'timezone': 'UTC',
    'data_dir': 'market_data',
    'overwrite': False,  # 文件覆盖控制
    'exchange': {
        'enableRateLimit': True,
        'timeout': 10000,
        'proxies': {
            'http': 'socks5h://127.0.0.1:7897',  # 使用SOCKS5协议
            'https': 'socks5h://127.0.0.1:7897',
            'websocket': 'socks5h://127.0.0.1:7897'
        },
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    },
    'retries': 5,
    'retry_delay': 10
}

class BinanceDataFetcher:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._init_exchanges()
        self.utc_tz = pytz.UTC
        self.timeframe_ms = self._get_timeframe_ms()
        self.spot_dir = os.path.join(CONFIG['data_dir'], CONFIG['timeframe'], 'spot')
        self.future_dir = os.path.join(CONFIG['data_dir'], CONFIG['timeframe'], 'future')
        os.makedirs(self.spot_dir, exist_ok=True)
        os.makedirs(self.future_dir, exist_ok=True)

    def _init_exchanges(self):
        self.spot_ex = ccxt.binance({
            **CONFIG['exchange'],
            'options': {'defaultType': 'spot'}
        })
        # USDT永续合约市场（使用binanceusdm专用类）
        self.future_ex = ccxt.binanceusdm({
            **CONFIG['exchange'],
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        self._verify_proxy()

    def _verify_proxy(self):
        test_urls = [
            'https://api.binance.com/api/v3/ping',
            'https://fapi.binance.com/fapi/v1/ping'
        ]
        for url in test_urls:
            try:
                response = requests.get(
                    url,
                    proxies=CONFIG['exchange']['proxies'],
                    headers=CONFIG['exchange']['headers'],
                    timeout=10
                )
                if self.verbose:
                    print(f"代理验证 {url} 状态码: {response.status_code}")
            except Exception as e:
                if self.verbose:
                    print(f"代理验证失败: {str(e)}")
                raise ConnectionError("代理服务器不可用")

    def _get_daily_files(self, start_date, end_date):
        """生成UTC日期范围"""
        # 转换为UTC时区
        start_date = start_date.astimezone(self.utc_tz)
        end_date = end_date.astimezone(self.utc_tz)

        date_range = pd.date_range(
            start=start_date.date(),
            end=end_date.date(),
            freq='D',
            tz=self.utc_tz
        )
        return [d.strftime('%Y-%m-%d') + '.csv' for d in date_range]

    def _file_needs_download(self, filename, market_type):
        target_dir = os.path.join(
            CONFIG['data_dir'],
            CONFIG['timeframe'],
            'spot' if market_type == 'spot' else 'future'
        )

        filepath = os.path.join(target_dir, filename)

        if not os.path.exists(filepath):
            return True  # 文件不存在需要下载
        return CONFIG['overwrite']  # 文件存在时根据覆盖标志决定

    def fetch_ohlcv(self, market_type):
        if market_type == 'future' and not self.future_ex.has['fetchOHLCV']:
            raise Exception(f"{self.future_ex.name} 不支持获取K线数据")

        exchange = self.spot_ex if market_type == 'spot' else self.future_ex
        # 统一使用UTC时区
        end_time = datetime.now(self.utc_tz)
        start_time = end_time - timedelta(days=CONFIG['days'])

        date_files = self._get_daily_files(start_time, end_time)

        for date_file in date_files:
            if not self._file_needs_download(date_file, market_type):
                if self.verbose:
                    print(f"跳过已存在文件: {date_file}")
                continue

            day_str = date_file.replace('.csv', '')
            day_start = int(pd.Timestamp(day_str).timestamp() * 1000)
            day_end = int((pd.Timestamp(day_str) + pd.Timedelta(days=1)).timestamp() * 1000)

            if self.verbose:
                print(f"\n开始获取{market_type}市场数据（日期：{day_str}）")

            daily_data = []
            since = day_start
            while since < day_end:
                try:
                    data = self._safe_fetch(
                        exchange.fetch_ohlcv,
                        CONFIG['symbol'],
                        CONFIG['timeframe'],
                        since=since,
                        limit=CONFIG['chunk_size']
                    )

                    if not data:
                        break

                    valid_data = [d for d in data if d[0] < day_end]
                    daily_data.extend(valid_data)

                    if len(data) == 0:
                        break
                    since = data[-1][0] + self.timeframe_ms

                    if len(data) < CONFIG['chunk_size']:
                        break

                except Exception as e:
                    if self.verbose:
                        print(f"分页获取失败: {str(e)}")
                    break

            if daily_data:
                self._save_daily_data(daily_data, market_type, day_str)

        return self._load_all_data(market_type)

    def _save_daily_data(self, data, market_type, day_str):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 确保时间戳为UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(self.utc_tz)

        # 转换为UTC日期对象比较
        target_date = pd.Timestamp(day_str, tz=self.utc_tz).date()
        df = df[df['timestamp'].dt.date == target_date]  # 修正时区比较

        if market_type == 'future':
            funding_df = self._get_daily_funding_rates(pd.Timestamp(day_str))
            if not funding_df.empty:
                df = pd.merge_asof(
                    df.sort_values('timestamp'),
                    funding_df,
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta('8h')  # 资金费率每8小时结算
                )
            else:
                df['funding_rate'] = None

        save_dir = self.spot_dir if market_type == 'spot' else self.future_dir
        filepath = os.path.join(save_dir, f"{day_str}.csv")

        if not df.empty:
            df.to_csv(filepath, index=False)
            if self.verbose:
                print(f"保存 {len(df)} 条{market_type}数据到 {filepath}")

    def _load_all_data(self, market_type):
        data_dir = os.path.join(
            CONFIG['data_dir'],
            CONFIG['timeframe'],
            'spot' if market_type == 'spot' else 'future'
        )

        # 获取所有CSV文件
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        sorted_files = sorted(all_files, key=lambda x: x.split('.')[0])

        dfs = []
        for file in sorted_files:
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(
                    file_path,
                    parse_dates=['timestamp'],
                    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'] +
                    (['funding_rate'] if market_type == 'future' else [])
                )
                df['market_type'] = market_type
                dfs.append(df)
            except Exception as e:
                if self.verbose:
                    print(f"加载文件 {file} 失败: {str(e)}")
                continue

        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs)

        return full_df.sort_values('timestamp').pipe(self._clean_loaded_data)

    def _clean_loaded_data(self, df):
        return (
            df
            .drop_duplicates('timestamp', keep='last')
            .rename(columns={
                'close': f"{df.iloc[0]['market_type']}_price",
                'funding_rate': 'funding_rate' if 'funding_rate' in df.columns else None
            })
            .pipe(lambda d: d[['timestamp', f"{d.iloc[0]['market_type']}_price"] +
                (['funding_rate'] if 'funding_rate' in d.columns else [])])
        )

    def _get_timeframe_ms(self):
        """动态解析时间单位 (支持 m/d/s/ms)"""
        timeframe = CONFIG['timeframe']

        match = re.match(r'^(\d+)([a-z]+)$', timeframe.lower())
        if not match:
            raise ValueError(f"无效时间框架格式: {timeframe}")

        value = int(match.group(1))
        unit = match.group(2)

        if unit.startswith('ms'):
            return value
        elif unit.startswith('s'):
            return value * 1000
        elif unit.startswith('m'):
            return value * 60 * 1000
        elif unit.startswith('d'):
            return value * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"不支持的时间单位: {unit}")

    def _safe_fetch(self, func, *args, **kwargs):
        for i in range(CONFIG['retries']):
            try:
                return func(*args, **kwargs)
            except ccxt.AuthenticationError as e:
                print(f"认证错误: {str(e)}")
                raise
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                if self.verbose:
                    print(f"网络错误（尝试 {i + 1}/{CONFIG['retries']}）: {str(e)}")
                if i < CONFIG['retries'] - 1:
                    sleep_time = CONFIG['retry_delay'] * (2 ** i)
                    if self.verbose:
                        print(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
            except ccxt.ExchangeError as e:
                if self.verbose:
                    print(f"交易所错误: {str(e)}")
                if 'Invalid symbol' in str(e):
                    raise ValueError(f"交易对错误: {CONFIG['symbol']}")
                raise
        raise Exception("超过最大重试次数")

    def _get_daily_funding_rates(self, day):
        try:
            symbol = CONFIG['symbol'].replace('/', '')
            if not symbol.isalnum():
                raise ValueError(f"非法交易对格式: {symbol}")

            start_time = int(day.timestamp() * 1000)
            end_time = start_time + 86400000  # 24小时

            all_rates = []
            current_since = start_time

            while current_since < end_time:
                # API请求参数
                params = {
                    'symbol': symbol,
                    'startTime': current_since,
                    'endTime': min(current_since + 3600000 * 24, end_time),
                    'limit': 1000
                }

                response = self._safe_fetch(
                    self.future_ex.fapipublic_get_fundingrate,
                    params=params
                )

                if not isinstance(response, list):
                    if self.verbose:
                        print(f"异常响应格式: {type(response)}")
                    break

                for item in response:
                    try:
                        funding_time = int(item['fundingTime'])
                        funding_rate = float(item['fundingRate'])

                        if funding_time < end_time:
                            all_rates.append({
                                'fundingTime': funding_time,
                                'fundingRate': funding_rate
                            })
                    except (KeyError, ValueError) as e:
                        if self.verbose:
                            print(f"数据解析异常: {str(e)} | 数据: {item}")

                # 更新分页参数
                if response:
                    last_time = int(response[-1]['fundingTime'])
                    current_since = last_time + 1
                else:
                    current_since += 3600000 * 24  # 推进24小时

            # 转换为DataFrame
            if all_rates:
                df = pd.DataFrame(all_rates)
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
                df['funding_rate'] = df['fundingRate']
                return df[['timestamp', 'funding_rate']].drop_duplicates()

            return pd.DataFrame()

        except Exception as e:
            if self.verbose:
                print(f"资金费率获取异常: {str(e)}")
            return pd.DataFrame()

# ==================== 优化后的数据处理 ====================
def process_and_save_data(spot_df, future_df, verbose=True):
    """合并现货和期货数据，并从期货数据中提取资金费率
    返回包含以下字段的DataFrame:
    - timestamp (UTC时区)
    - spot_price
    - future_price
    - funding_rate
    """

    # 合并价格数据
    merged_df = pd.merge_asof(
        spot_df.rename(columns={'spot_price': 'price'}),
        future_df.rename(columns={'future_price': 'price'}),
        on='timestamp',
        suffixes=('_spot', '_future'),
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )

    # 从期货数据中提取资金费率
    if 'funding_rate' in future_df.columns:
        # 前向填充资金费率（因为资金费率每8小时更新一次）
        merged_df['funding_rate'] = future_df.set_index('timestamp')['funding_rate'].reindex(
            merged_df['timestamp'], method='ffill', limit=1  # 最多向前填充1个周期
        ).values
    else:
        merged_df['funding_rate'] = None
        if verbose:
            print("警告：期货数据中未找到资金费率字段")

    # 数据清理
    final_df = merged_df[[
        'timestamp',
        'price_spot',
        'price_future',
        'funding_rate'
    ]].rename(columns={
        'price_spot': 'spot_price',
        'price_future': 'future_price'
    })

    # 去除可能存在的空值
    final_df = final_df.dropna(subset=['spot_price', 'future_price'])

    # 保存最终数据
    final_path = os.path.join(CONFIG['data_dir'], 'merged_dataset.csv')
    final_df.to_csv(final_path, index=False)
    if verbose:
        print(f"数据集已保存至: {final_path}")

    return final_df

# ==================== 主程序 ====================
if __name__ == "__main__":
    if verbose:
        print("=== 启动数据获取程序 ===")
    fetcher = BinanceDataFetcher(verbose=True)

    try:
        # 获取数据
        spot = fetcher.fetch_ohlcv('spot')
        future = fetcher.fetch_ohlcv('future')

        # 处理数据
        df = process_and_save_data(spot, future, verbose=True)

        # 数据验证
        if verbose:
            print("\n数据质量报告:")
            print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"总记录数: {len(df)}")
            print(f"缺失值统计:\n{df.isnull().sum()}")
            print("\n前5条记录:")
            print(df.head())
            print("\n最后5条记录:")
            print(df.tail())

    except Exception as e:
        if verbose:
            print(f"\n程序运行失败: {str(e)}")
        raise
        exit(1)
