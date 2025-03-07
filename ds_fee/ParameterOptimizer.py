import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import json
import multiprocessing
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from ds_fee.BacktestEngine import BacktestEngine

class OptimizationConfig:
    def __init__(
        self,
        min_annual_return=0.03,   # 最低年化收益3%
        max_drawdown=0.30,        # 允许最大回撤30%
        min_sharpe=0.8,           # 夏普比率≥0.8
        min_win_rate=0.30,        # 胜率≥30%
        max_position_days=15,     # 最大持仓15天
        max_volatility=0.40       # 波动率≤40%
    ):
        # 风险收益指标阈值
        self.min_annual_return = min_annual_return  # 最小年化收益率
        self.max_drawdown = max_drawdown            # 最大回撤
        self.min_sharpe = min_sharpe                # 最小夏普比率
        self.min_win_rate = min_win_rate            # 最小胜率
        self.max_position_days = max_position_days  # 最大持仓天数
        self.max_volatility = max_volatility        # 最大波动率

    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置对象"""
        return cls(**config_dict)

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

def serialize_preprocessed_data(df: DataFrame) -> dict:
    """序列化预处理数据为可传输格式"""
    return {
        'features': df.values.tolist(),  # 统一使用features作为数据键
        'index': df.index.tolist(),
        'columns': df.columns.tolist()
    }

def deserialize_preprocessed_data(data_dict: dict) -> DataFrame:
    """反序列化预处理数据为DataFrame"""
    return DataFrame(
        data=data_dict['features'],  # 同步修改为features
        index=pd.to_datetime(data_dict['index']),
        columns=data_dict['columns']
    )

def is_qualified(metrics, optimization_config):
    """检查参数组合是否满足风控要求（全局函数版本）"""
    return (
        metrics['annualized_return'] >= optimization_config.min_annual_return and
        metrics['max_drawdown'] <= optimization_config.max_drawdown and
        metrics['sharpe_ratio'] >= optimization_config.min_sharpe and
        metrics['win_rate'] >= optimization_config.min_win_rate
    )

def evaluate_params(params, serializable_data, optimization_config, verbose=False):
    """执行单个参数组合的回测评估（全局函数版本）"""
    if verbose:
        print(f"\n🔍 开始回测参数组合:", flush=True)
        for k,v in params.items():
            print(f"  {k}: {v}", flush=True)
    
    # 直接从序列化数据获取预处理结果
    try:
        # 合并基础配置参数（包含fee_rate）
        full_config = {
            **serializable_data['base_config'],  # 基础配置参数
            **params,                            # 优化参数
            'output_dir': 'output'
        }
        
        # 使用预处理的配置和数据直接初始化引擎
        engine = BacktestEngine(config=full_config)
        engine.preprocessed_data = deserialize_preprocessed_data(serializable_data['preprocessed_data'])
        
    except KeyError as e:
        raise ValueError(f"Invalid serialized data format: {str(e)}") from e
    
    results = engine.run_backtest(full_config, verbose=verbose, saveResults=False)
    
    metrics = results.get('risk_metrics', {})
    return {
        'params': params.copy(),
        'metrics': {
            'annualized_return': metrics.get('annualized_return', -np.inf),
            'max_drawdown': metrics.get('max_drawdown', 1.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', -np.inf),
            'profit_factor': metrics.get('profit_factor', 0),
            'win_rate': metrics.get('win_rate', 0),
            'volatility': metrics.get('annualized_volatility', np.inf)
        },
        'qualified': is_qualified(metrics, optimization_config) if metrics else False,
        'error': None
    }

def process_result(result, shared_completed, shared_best_score, shared_lock, total_combinations):
    """处理单个结果（全局函数版本）"""
    with shared_lock:
        shared_completed.value += 1
        elapsed = pd.Timestamp.now().strftime('%H:%M:%S')
        print(f"[{elapsed}] 已完成 {shared_completed.value}/{total_combinations} ({shared_completed.value/total_combinations:.1%})")
        
        if result and result.get('error'):
            print(f"⚠️ 参数组合 {result['params']} 执行失败: {result['error']}")
            return result
            
        if result and result.get('metrics'):
            current_score = result['metrics']['sharpe_ratio']
            if current_score > shared_best_score.value:
                shared_best_score.value = current_score
                print(f"\n🌟 发现新的最佳组合（夏普比率:{current_score:.2f}）:")
                for k, v in result['params'].items():
                    print(f"   ├ {k}: {v}")
                print(f"   ├ 年化收益: {result['metrics']['annualized_return']:.2%}")
                print(f"   └ 最大回撤: {result['metrics']['max_drawdown']:.2%}")
        return result

# 创建全局任务包装函数
def task_wrapper(params, serializable_data, optimization_config, verbose,
                shared_completed, shared_best_score, shared_lock, total_combinations):
    """并行任务包装函数"""
    try:
        result = evaluate_params(
            params=params,
            serializable_data=serializable_data,
            optimization_config=optimization_config,
            verbose=verbose
        )
    except Exception as e:
        return {'params': params, 'error': str(e)}
    return process_result(
        result=result,
        shared_completed=shared_completed,
        shared_best_score=shared_best_score,
        shared_lock=shared_lock,
        total_combinations=total_combinations
    )
        
class ParameterOptimizer:
    def __init__(self, base_config, param_space, optimization_config=None):
        # 转换配置对象为字典格式
        self.base_config = base_config
        self.param_space = param_space
        self.optimization_config = optimization_config or OptimizationConfig()
        self.dashboard = OptimizationDashboard()
        self.manager = multiprocessing.Manager()
        self.shared_completed = self.manager.Value('i', 0)
        self.shared_best_score = self.manager.Value('d', -np.inf)
        self.shared_lock = self.manager.Lock()

    def grid_search(self, data, n_jobs=-1, verbose=False):
        """并行化网格搜索优化"""
        grid = list(ParameterGrid(self.param_space))
        total_combinations = len(grid)
        
        print(f"\n🔧 正在初始化{total_combinations}个参数组合...", flush=True)
        print(f"⚙️ 启动{n_jobs if n_jobs != -1 else '全部'}个并行工作进程", flush=True)
        print(f"🕒 开始时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        # 将预处理数据和基础配置序列化
        serializable_data = {
            'base_config': self.base_config,  # 包含所有基础配置参数
            'preprocessed_data': serialize_preprocessed_data(data)
        }
        
        # 执行并行任务并实时处理结果
        with Parallel(n_jobs=n_jobs, verbose=5, timeout=3600) as parallel:
            results = []
            for result in parallel(delayed(task_wrapper)(
                params, 
                serializable_data,
                self.optimization_config,
                verbose,
                self.shared_completed,
                self.shared_best_score,
                self.shared_lock,
                total_combinations
            ) for params in grid):
                results.append(result)
        
        # 过滤合格结果
        qualified_results = [res for res in results if res.get('qualified')]
        self._save_results(qualified_results)
        
        # 最终结果展示
        self.dashboard.print_best_result(results)
        
        return sorted(qualified_results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)

    def _save_results(self, results):
        """保存优化结果和历史记录"""
        history_dir = "output/optimization_history"
        os.makedirs(history_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{history_dir}/optimized_params_{timestamp}.csv"
        
        # 构建包含完整指标的结果DataFrame
        df = pd.DataFrame([{
            **res['params'],
            'annualized_return': res['metrics']['annualized_return'],
            'max_drawdown': res['metrics']['max_drawdown'],
            'sharpe_ratio': res['metrics']['sharpe_ratio'],
            'win_rate': res['metrics']['win_rate'],
            'volatility': res['metrics']['volatility'],
            'profit_factor': res['metrics']['profit_factor'],
            'avg_holding_days': res['metrics']['avg_holding_days'],
            'total_trades': res['metrics']['total_trades'],
            'qualified': res['qualified']
        } for res in results if res['metrics']])
        
        # 保存本次结果
        df.to_csv(filename, index=False)
        
        # 更新优化日志
        log_entry = {
            'timestamp': timestamp,
            'total_combinations': len(results),
            'qualified_count': len([r for r in results if r['qualified']]),
            'best_sharpe': df['sharpe_ratio'].max() if not df.empty else None
        }
        log_path = f"{history_dir}/optimization_log.csv"
        pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=not os.path.exists(log_path))

    def _evaluate_params(self, params, data):
        """评估单个参数组合（类方法版本）"""
        return evaluate_params(
            params=params,
            serializable_data={
                'base_config': self.base_config,
                'preprocessed_data': serialize_preprocessed_data(data)
            },
            optimization_config=self.optimization_config,
            verbose=False
        )

    def genetic_optimization(self, data, population_size=50, generations=20, verbose=False):
        """遗传算法优化器"""
        # 初始化种群
        population = self._initialize_population(population_size)
        
        for gen in range(generations):
            # 评估适应度
            # 过滤无效评估结果
            evaluated_pop = [res for res in 
                           (self._evaluate_params(ind, data) for ind in population)
                           if res and res['metrics'] is not None]
            
            # 选择
            selected = self._selection(evaluated_pop)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            population = self._mutation(offspring)
            
            # 显示进度
            self.dashboard.print_best_result(evaluated_pop)
        
        return sorted(evaluated_pop, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)

    def _initialize_population(self, size):
        """生成初始种群"""
        population = []
        for _ in range(size):
            individual = {
                'spread_threshold': np.round(np.random.uniform(0.002, 0.006), 4),  # 价差范围0.2%-0.6%
                'leverage': np.random.choice([2, 3, 4]),                          # 杠杆倍数选项
                'max_hold_seconds': np.random.choice([3600, 7200, 14400]),         # 持仓时间1-4小时
                'take_profit': np.random.choice(np.arange(0.004, 0.012, 0.001)),  # 止盈范围0.4%-1.2% 步长0.1%
                'stop_loss': np.random.choice(np.arange(0.003, 0.008, 0.001))    # 止损范围0.3%-0.8% 步长0.1%
            }
            population.append(individual)
        return population

    def _selection(self, population):
        """锦标赛选择"""
        selected = []
        tournament_size = 5
        for _ in range(len(population)):
            candidates = np.random.choice(population, tournament_size, replace=False)
            winner = max(
                [c for c in candidates if c['metrics'] is not None], 
                key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf)
            )
            selected.append(winner['params'])
        return selected

    def _crossover(self, parents):
        """均匀交叉"""
        offspring = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < len(parents) else parents[0]
            child = {}
            for key in p1:
                if np.random.rand() < 0.5:
                    child[key] = p1[key]
                else:
                    child[key] = p2[key]
            offspring.append(child)
        return offspring

    def random_sampling(self, raw_data, samples=500):
        """随机采样优化"""
        print(f"\n🎲 开始随机采样，样本数: {samples}")
        
        # 统一数据序列化格式
        data = {
            'base_config': self.base_config,
            'preprocessed_data': serialize_preprocessed_data(raw_data)
        }
        
        results = []
        for i in range(samples):
            # 生成随机参数
            params = {
                'spread_threshold': np.round(np.random.uniform(0.002, 0.006), 4),
                'leverage': np.random.choice([2, 3, 4]),
                'max_hold_seconds': np.random.choice([3600, 7200, 14400]),
                'take_profit': np.random.choice(np.arange(0.004, 0.012, 0.001)),  # 从步长0.1%的离散值中选择
                'stop_loss': np.random.choice(np.arange(0.003, 0.008, 0.001))     # 从步长0.1%的离散值中选择
            }
            # 评估参数
            result = self._evaluate_params(params, data)
            results.append(result)
            # 更新进度
            if (i+1) % 10 == 0 or (i+1) == samples:
                print(f"进度: {i+1}/{samples} ({(i+1)/samples:.1%})")
                self.dashboard.print_best_result(results)
        # 过滤无效结果并排序
        valid_results = [r for r in results if r['metrics'] is not None]
        return sorted(valid_results, key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf), reverse=True)

    def _mutation(self, offspring):
        """高斯变异（更新参数范围）"""
        mutated = []
        for params in offspring:
            mutated_params = params.copy()
            # 价差阈值变异 (新范围0.002-0.006)
            if np.random.rand() < 0.1:
                mutated_params['spread_threshold'] = np.clip(
                    np.random.normal(params['spread_threshold'], 0.001),
                    0.002, 0.006
                ).round(4)
            # 杠杆倍数变异 (新范围2-4)
            if np.random.rand() < 0.1:
                mutated_params['leverage'] = np.random.choice([2, 3, 4])
            # 持仓时间变异 (新选项)
            if np.random.rand() < 0.1:
                mutated_params['max_hold_seconds'] = np.random.choice([3600, 7200, 14400])
            # 止盈参数变异 (从预定义步长中选择)
            if np.random.rand() < 0.1:
                mutated_params['take_profit'] = np.random.choice(np.arange(0.004, 0.012, 0.001))
            # 止损参数变异 (从预定义步长中选择)
            if np.random.rand() < 0.1:
                mutated_params['stop_loss'] = np.random.choice(np.arange(0.003, 0.008, 0.001))
            mutated.append(mutated_params)
        return mutated

    def _sharpe_scorer(self, estimator, X, y):
        """自定义夏普比率评分函数"""
        estimator.run_backtest()
        metrics = estimator.analyze_performance()
        return metrics['sharpe_ratio']

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

if __name__ == "__main__":
    import argparse
    import json
    from ds_fee.config import load_base_config
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='量化策略参数优化执行器', 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', choices=['grid','genetic','random'], default='genetic',
                      help='选择优化算法: grid-网格搜索, genetic-遗传算法, random-随机采样')
    parser.add_argument('--output', type=str, default='optimization_results',
                      help='结果输出目录路径')
    parser.add_argument('--jobs', type=int, default=-1,
                      help='并行任务数（-1表示使用全部CPU核心）')
    parser.add_argument('--verbose', action='store_true',
                      help='启用详细输出模式')
    args = parser.parse_args()

    # 初始化优化器配置
    base_config = load_base_config()
    param_ranges = {
        'spread_threshold': np.arange(0.002, 0.006, 0.0005),  # 调整到更合理的价差范围
        'leverage': [2, 3, 4],
        'max_hold_seconds': [3600, 7200, 14400],  # 1-4小时更合理的持仓时间
        'take_profit': np.arange(0.004, 0.012, 0.001),  # 新增止盈参数
        'stop_loss': np.arange(0.003, 0.008, 0.001)     # 新增止损参数
    }
    optimizer = ParameterOptimizer(base_config, param_ranges)  # 修正变量名错误

    try:
        # 使用已加载的基础配置初始化引擎
        engine = BacktestEngine(base_config)
        engine.preprocess_data()
        data = engine.preprocessed_data
        
        print(f"✅ 成功加载预处理数据（{len(data):,} 条记录）")

        # 执行优化
        print(f"🚀 开始参数优化（方法: {args.method.upper()}）")
        if args.method == 'grid':
            results = optimizer.grid_search(data, n_jobs=args.jobs, verbose=args.verbose)
        elif args.method == 'genetic':
            results = optimizer.genetic_optimization(data, verbose=args.verbose)
        elif args.method == 'random':
            results = optimizer.random_sampling(data, verbose=args.verbose)
        
        # 保存结果
        # 过滤无效结果并保存
        valid_results = [r for r in results if r['metrics'] is not None]
        optimizer.save_full_report(valid_results, args.output)
        print(f"\n🎉 优化完成！有效参数组合: {len(valid_results)} 个")
        print(f"📂 结果文件保存在: {os.path.abspath(args.output)}")
        
        # 处理优化结果
        if valid_results:
            qualified = [res for res in results if res['qualified']]
            print(f"\n🎉 优化完成！总参数组合: {len(results)} 合格参数: {len(qualified)}")
            
            if qualified:
                best_result = max(qualified, key=lambda x: x['metrics']['sharpe_ratio'])
                print("\n🏆 最佳参数组合:")
                for k, v in best_result['params'].items():
                    print(f"  ▸ {k:.<20} {v}")
                print(f"⭐ 夏普比率: {best_result['metrics']['sharpe_ratio']:.2f}")
            else:
                print("\n⚠️  没有找到符合风控要求的参数组合，建议：")
                print("1. 检查策略逻辑是否正确")
                print("2. 调整优化参数范围")
                print("3. 放宽风控指标阈值")
        else:
            print("\n❌ 优化失败：未得到任何有效结果")

    except FileNotFoundError as e:
        print(f"❌ 市场数据目录不存在: {e.filename}")
        print("请检查ds_fee/market_data目录结构是否符合要求")
    except pd.errors.EmptyDataError:
        print(f"❌ 数据文件为空或格式错误: {args.data}")
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        raise
