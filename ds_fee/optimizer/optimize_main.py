import numpy as np
import argparse
import multiprocessing
from ds_fee.optimizer import GeneticOptimizer, RandomSearchOptimizer, GridSearchOptimizer
from ds_fee.optimizer.params import OptimizationParams
from ds_fee.backtest.data_processor import DataProcessor
from ds_fee.utils.config import load_base_config
from ds_fee.optimizer.shared_state import SharedState

# 使用BaseOptimizer中的parse_args方法
from ds_fee.optimizer.base import BaseOptimizer

def main():
    # 解析命令行参数
    args = BaseOptimizer.parse_args()
    
    # 加载基础配置
    config = load_base_config()
    
    # 初始化数据处理器
    data_processor = DataProcessor(config)
    data_processor.preprocess_data()
    
    # 创建优化参数实例
    params = OptimizationParams(
        population_size=args.population_size,
        generations=args.generations,
        samples=args.samples,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )
    
    # 从配置中获取参数优化范围
    param_ranges = config.get('optimizer', {}).get('param_ranges', {})
    
    # 根据参数创建对应的优化器
    optimizer_map = {
        'genetic': GeneticOptimizer,
        'random': RandomSearchOptimizer,
        'grid': GridSearchOptimizer
    }
    optimizer = optimizer_map[args.optimizer](config)
    #输出优化方法
    print(f"使用优化方法: {args.optimizer}")

    # 创建全局共享变量
    manager = multiprocessing.Manager()
    sharedState = SharedState(manager)

    # 执行优化
    results = optimizer.optimize(data_processor.preprocessed_data, params, sharedState)
    
    # 打印最优结果
    if results:
        print("\n优化完成! 最优参数组合:")
        best_result = results[0]
        print(f"参数: {best_result['params']}")
        print(f"指标: {best_result['metrics']}")
    else:
        print("\n未找到有效的优化结果")

if __name__ == '__main__':
    main()