import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
import os
import json
from ds_fee.backtest.BacktestEngine import BacktestEngine
from .params import OptimizationParams

import argparse

class OptimizationDashboard:
    def print_best_result(self, results):
        # 过滤无效结果
        valid_results = [r for r in results if r.get('metrics')]
        
        if valid_results:
            # 始终显示全局最优夏普组合
            best_sharpe = max(valid_results, 
                            key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf))
            print("\n📈 当前最优夏普组合:")
            for k, v in best_sharpe['params'].items():
                print(f"  ▸ {k}: {v}")
            print(f"  夏普比率: {best_sharpe['metrics']['sharpe_ratio']:.2f}")
            print(f"  年化收益: {best_sharpe['metrics']['annualized_return']:.2%}")
            print(f"  最大回撤: {best_sharpe['metrics']['max_drawdown']:.2%}")
            print(f"  是否合格: {'✅' if best_sharpe['qualified'] else '❌'}")

            # 显示全局收益最高的组合（即使不合格）
            best_profit = max(valid_results, 
                            key=lambda x: x['metrics'].get('annualized_return', -np.inf))
            print("\n🏆 全局最高收益组合:")
            for k, v in best_profit['params'].items():
                print(f"  ▸ {k}: {v}")
            print(f"  年化收益: {best_profit['metrics']['annualized_return']:.2%}")
            print(f"  夏普比率: {best_profit['metrics'].get('sharpe_ratio', 0):.2f}")
            print(f"  最大回撤: {best_profit['metrics']['max_drawdown']:.2%}")
            print(f"  是否合格: {'✅' if best_profit['qualified'] else '❌'}")

def evaluate_params(params, data, base_config, dashboard, verbose=False):
    """执行单个参数组合的回测评估"""
    if verbose:
        print(f"\n🔍 开始回测参数组合:", flush=True)
        for k,v in params.items():
            print(f"  {k}: {v}", flush=True)
    
    try:
        # 合并基础配置参数
        base_config_dict = base_config.to_dict() if hasattr(base_config, 'to_dict') else dict(base_config)
        full_config = {
            **base_config_dict,  # 基础配置参数
            **params            # 优化参数
        }
        
        # 初始化引擎
        engine = BacktestEngine(config=full_config)
        
        # 预处理数据
        engine.data_processor.preprocessed_data = data.copy()
        
        # 执行回测
        results = engine.backtest_core.run_backtest(params, verbose=False)
        metrics = results.get('risk_metrics', {})
        
        # 增加资金费收益权重的评估
        funding_ratio = metrics.get('funding_ratio%', 0) / 100  # 转换为小数
        funding_weight = 1.2  # 资金费收益权重
        
        # 计算加权夏普比率
        weighted_sharpe = metrics.get('sharpe_ratio', 0) * (1 + funding_ratio * funding_weight)
        
        # 更新评估指标
        result = {
            'params': params.copy(),
            'metrics': {
                'annualized_return': metrics.get('annualized_return', -np.inf),
                'max_drawdown': metrics.get('max_drawdown', 1.0),
                'sharpe_ratio': weighted_sharpe,  # 使用加权夏普比率
                'profit_factor': metrics.get('profit_factor', 0),
                'win_rate': metrics.get('win_rate', 0),
                'volatility': metrics.get('annualized_volatility', np.inf),
                'funding_ratio': funding_ratio
            },
            'error': None
        }
        
        # 从配置中获取风控参数
        risk_params = base_config.get('optimizer', {}).get('risk_params', {})
        min_annual_return = risk_params.get('min_annual_return', 0.03)
        max_drawdown = risk_params.get('max_drawdown', 0.30)
        min_sharpe = risk_params.get('min_sharpe', 0.8)
        min_win_rate = risk_params.get('min_win_rate', 0.30)
        
        # 检查是否满足风控要求
        result['qualified'] = (
            result['metrics']['annualized_return'] >= min_annual_return and
            result['metrics']['max_drawdown'] <= max_drawdown and
            result['metrics']['sharpe_ratio'] >= min_sharpe and
            result['metrics']['win_rate'] >= min_win_rate
        )
        
        return result
            
    except Exception as e:
        import traceback
        error_stack = traceback.format_exc()
        print(f"\n⚠️ 参数评估失败:")
        print(f"参数: {params}")
        print(f"错误信息: {str(e)}")
        print(f"错误堆栈:\n{error_stack}")
        return {'params': params, 'error': str(e), 'error_stack': error_stack}

class BaseOptimizer(ABC):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='参数优化器')
        parser.add_argument('--optimizer', type=str, choices=['genetic', 'random', 'grid'], default='grid',
                          help='优化方法: genetic(遗传算法), random(随机搜索), grid(网格搜索)')
        parser.add_argument('--population-size', type=int, default=50, help='遗传算法种群大小')
        parser.add_argument('--generations', type=int, default=20, help='遗传算法迭代次数')
        parser.add_argument('--samples', type=int, default=500, help='随机搜索采样数')
        parser.add_argument('--n-jobs', type=int, default=4, help='网格搜索并行数')
        parser.add_argument('--verbose', action='store_true', help='是否显示详细信息')
        return parser.parse_args()

    def __init__(self, base_config):
        self.base_config = base_config
        self.dashboard = OptimizationDashboard()

    @abstractmethod
    def validate_params(self, params) -> None:
        """验证优化器参数
        
        Args:
            params: OptimizationParams实例，包含优化器参数
            
        Raises:
            ValueError: 当参数验证失败时抛出
        """
        pass

    @abstractmethod
    def optimize(self, data, params):
        """优化方法的抽象接口，需要被具体的优化器实现
        
        Args:
            data: 预处理后的数据，DataFrame格式
            params: 优化器参数配置实例
        
        Returns:
            list: 排序后的优化结果列表，每个元素为字典格式：
                {
                    'params': 参数配置字典,
                    'metrics': 评估指标字典,
                    'qualified': 是否满足风控要求的布尔值
                }
        """
        pass

    def save_full_report(self, results, output_dir):
        """保存完整优化报告"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV结果
        df = pd.DataFrame([{
            **res['params'],
            **res['metrics'],
            'qualified': res['qualified']
        } for res in results])
        csv_path = f"{output_dir}/optimization_report_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # 保存元数据
        # 只处理合格结果
        qualified = [res for res in results if res['qualified']]
        meta = {
            "optimization_date": timestamp,
            "total_combinations": len(results),
            "qualified_count": len(qualified),
            "best_sharpe": max((res['metrics']['sharpe_ratio'] for res in qualified), default=None)
        } if qualified else {
            "optimization_date": timestamp,
            "total_combinations": len(results),
            "qualified_count": 0,
            "best_sharpe": None
        }
        json_path = f"{output_dir}/metadata_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)