import pandas as pd
import os
from tqdm import tqdm

def convert_folder_to_parquet(src_dir, dest_dir, columns=None, dtype=None):
    """将目录中的CSV文件批量转换为Parquet格式"""
    os.makedirs(dest_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
    
    for file in tqdm(files, desc=f'Processing {os.path.basename(src_dir)}'):
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file.replace('.csv', '.parquet'))
        
        # 分块读取和保存
        reader = pd.read_csv(src_path, usecols=columns, dtype=dtype, chunksize=10**5)
        for i, chunk in enumerate(reader):
            if i == 0:
                chunk.to_parquet(dest_path, engine='pyarrow')
            else:
                chunk.to_parquet(dest_path, engine='pyarrow', append=True)

if __name__ == "__main__":
    # 转换现货数据
    convert_folder_to_parquet(
        src_dir='ds_fee/market_data/1m/spot',
        dest_dir='ds_fee/market_data/1m/spot_parquet',
        columns=['timestamp', 'close'],
        dtype={'close': 'float32'}
    )
    
    # 转换合约数据
    convert_folder_to_parquet(
        src_dir='ds_fee/market_data/1m/future',
        dest_dir='ds_fee/market_data/1m/future_parquet',
        columns=['timestamp', 'close', 'funding_rate'],
        dtype={'close': 'float32', 'funding_rate': 'float32'}
    )
