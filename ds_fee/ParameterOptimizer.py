import numpy as np
import pandas as pd
import os
import json
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
    def __init__(self):
        self.top_params = []
        self.total_tasks = 0
        self.completed = 0
        
    def start_progress(self, total):
        self.total_tasks = total
        self.completed = 0
        print(f"开始参数优化，总任务数: {total}")
        
    def update_progress(self, advance=1):
        self.completed += advance
        print(f"进度: {self.completed}/{self.total_tasks} ({self.completed/self.total_tasks:.1%})")
        
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

class ParameterOptimizer:
    def __init__(self, base_config, param_space, optimization_config=None):
        self.base_config = base_config
        self.param_space = param_space
        self.optimization_config = optimization_config or OptimizationConfig()
        self.dashboard = OptimizationDashboard()

    def _is_qualified(self, metrics):
        """检查参数组合是否满足风控要求"""
        return (
            metrics['annualized_return'] >= self.optimization_config.min_annual_return and
            metrics['max_drawdown'] <= self.optimization_config.max_drawdown and
            metrics['sharpe_ratio'] >= self.optimization_config.min_sharpe and
            metrics['win_rate'] >= self.optimization_config.min_win_rate
        )
    
    best_score = -np.inf
    best_params = None
    completed = 0

    # 创建进度回调函数
    def grid_search(self, data, n_jobs=-1, verbose=False):
        """并行化网格搜索优化"""
        grid = list(ParameterGrid(self.param_space))
        total_combinations = len(grid)
        self.dashboard.start_progress(total_combinations)
        
        print(f"\n🔧 正在初始化{total_combinations}个参数组合...", flush=True)
        print(f"⚙️ 启动{n_jobs if n_jobs != -1 else '全部'}个并行工作进程", flush=True)
        print(f"🕒 开始时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        # 初始化最佳结果跟踪
        self.best_score = -np.inf
        self.best_params = None

        # 原子操作进度跟踪
        from joblib.externals.loky import get_reusable_executor
        
        def process_result(result):
            # 使用原子操作更新进度
            self.completed += 1
            elapsed = pd.Timestamp.now().strftime('%H:%M:%S')
            print(f"[{elapsed}] 已完成 {self.completed}/{total_combinations} ({self.completed/total_combinations:.1%})")
            print(str(result))
            if result and result['metrics']:
                # 使用线程安全的方式更新最佳结果
                current_score = result['metrics']['sharpe_ratio']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_params = result['params'].copy()
                    print(f"\n🌟 发现新的最佳组合（夏普比率:{current_score:.2f}）:")
                    for k, v in self.best_params.items():
                        print(f"   ├ {k}: {v}")
                    print(f"   ├ 年化收益: {result['metrics']['annualized_return']:.2%}")
                    print(f"   └ 最大回撤: {result['metrics']['max_drawdown']:.2%}")
            return result

        try:
            # 将数据预处理为可序列化格式
            serializable_data = {
                'features': data.values.astype(np.float64).tolist(),
                'index': data.index.tz_localize(None).astype('datetime64[ns]').astype(np.int64).tolist(),
                'columns': data.columns.tolist(),
                'dtypes': {
                    'index': 'datetime64[ns]',
                    'features': 'float64'
                }
            }
            
            # 添加并行任务启动前检查点
            print(f"🔍 正在启动并行任务，首个参数组合示例: {grid[0]}", flush=True)
            print(f"🔍 最后一个参数组合示例: {grid[-1]}", flush=True)
            print(f"🔍 序列化数据类型检查: {type(serializable_data)} 长度: {len(serializable_data['features'])}", flush=True)
            
            # 启用joblib详细日志并添加超时设置
            # 显式收集结果并触发回调
            # 先序列化数据（避免在并行任务中重复处理）
            serializable_data = {
                'features': data.values.astype(np.float64).tolist(),
                'index': data.index.tz_localize(None).astype('datetime64[ns]').astype(np.int64).tolist(),
                'columns': data.columns.tolist(),
                'dtypes': {
                    'index': 'datetime64[ns]',
                    'features': 'float64'
                }
            }

            # 创建任务包装函数（简化版）
            def task_wrapper(params):
                result = self._evaluate_params(params, serializable_data, verbose)
                processed_result = process_result(result)
                return processed_result
            print("✅ 并行任务已成功启动", flush=True)
            # 执行并行任务并实时处理结果
            with Parallel(n_jobs=n_jobs, verbose=5, timeout=3600) as parallel:
                results = []
                for result in parallel(delayed(task_wrapper)(params) for params in grid):
                    results.append(result)
            
            # 过滤合格结果
            qualified_results = [res for res in results if res['qualified']]
            self._save_results(qualified_results)
            
            # 最终结果展示
            self.dashboard.print_best_result(results)
            
            return sorted(qualified_results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        finally:
            # 清理进度跟踪
            self.dashboard.total_tasks = 0
            self.dashboard.completed = 0
            
        return sorted(qualified_results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)

    def _evaluate_params(self, params, data, verbose=False):
        """执行单个参数组合的回测评估（子进程安全版本）"""
        if verbose:
            print(f"\n🔍 开始回测参数组合:", flush=True)
            for k,v in params.items():
                print(f"  {k}: {v}", flush=True)
            
        # 重建DataFrame数据（兼容原始DataFrame和序列化数据）
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            try:
                df = pd.DataFrame(
                    data['features'],
                    index=pd.to_datetime(data['index']),
                    columns=data['columns']
                )
            except KeyError as e:
                raise ValueError("Invalid data format. Expected serialized data with 'features' and 'index' fields") from e
        if verbose:
            print(f"✅ 数据重建完成，共{len(df)}条记录")
        
        # 创建可序列化的配置字典（避免传递复杂对象）
        config_dict = {
            'data_dir': 'ds_fee/market_data',
            'output_dir': 'output',
            **params
        }
        
        # 创建引擎并注入动态参数
        engine = BacktestEngine(config_dict)
        # 更新配置参数并设置默认手续费率
        actual_fee_rate = engine.actual_funding_rate or 0.0002  # 默认0.02%
        config_dict.update({
            'backtest': {
                'start_date': engine.actual_start_date,
                'end_date': engine.actual_end_date
            },
            'fee_rate': actual_fee_rate
        })
        # 重新创建引擎确保参数生效（使用合并后的配置）
        engine = BacktestEngine({
            **config_dict,
            'fee_rate': actual_fee_rate  # 显式传递有效费率
        })
        engine.preprocessed_data = df  # 直接注入预处理数据
        
        try:
            results = engine.run_backtest(config_dict, verbose=verbose, saveResults=False)
        except Exception as e:
            raise ValueError(f"⚠️ 回测失败: {str(e)}")
        
        # 确保risk_metrics存在  
        if 'risk_metrics' not in results:
            raise ValueError("⚠️ 回测结果缺少risk_metrics字段")
        if verbose:
            print("\n📊 回测结果详情:")
            print(f"✅ 回测成功完成")
            print(f"📅 回测期间: {config_dict['backtest']['start_date']} 至 {config_dict['backtest']['end_date']}")
            print(f"💸 手续费率: {config_dict['fee_rate']:.4f}")
            print(f"🔄 总交易次数: {results['risk_metrics']['total_trades']}")
            print(f"⏱️ 平均持仓天数: {results['risk_metrics'].get('avg_holding_days', 0):.1f}")
            print(f"💰 净利润: {results['risk_metrics']['net_profit']:.2f}")
            print(f"📉 最大回撤: {results['risk_metrics']['max_drawdown']:.2%}")
            print(f"🏆 年化收益率: {results['risk_metrics']['annualized_return']:.2%}")
            print(f"⚖️ 夏普比率: {results['risk_metrics'].get('sharpe_ratio', 0):.2f}")
        
        # 处理可能缺失的metrics字段
        metrics = results.get('risk_metrics')
        
        return {
            'params': params.copy(),
            'metrics': {
                'annualized_return': metrics.get('annualized_return', -np.inf),
                'max_drawdown': metrics.get('max_drawdown', 1.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', -np.inf),
                'profit_factor': metrics.get('profit_factor', 0),
                'win_rate': metrics.get('win_rate', 0),
                'volatility': metrics.get('annualized_volatility', np.inf),
                'avg_holding_days': metrics.get('avg_holding_days', 0),
                'total_trades': metrics.get('total_trades', 0),
                'net_profit': metrics.get('net_profit', -np.inf)
            },
            'qualified': self._is_qualified(metrics) if metrics else False,
            'error': None
        }

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

    def genetic_optimization(self, data, population_size=50, generations=20, verbose=False):
        """遗传算法优化器"""
        # 初始化种群
        population = self._initialize_population(population_size)
        
        for gen in range(generations):
            # 评估适应度
            # 过滤无效评估结果
            evaluated_pop = [res for res in 
                           (self._evaluate_params(ind, data) for ind in population)
                           if res['metrics'] is not None]
            
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
                'spread_threshold': np.round(np.random.uniform(0.002, 0.006), 4),
                'leverage': np.random.choice([2, 3, 4]),
                'max_hold_seconds': np.random.choice([3600, 7200, 14400]),
                'take_profit': np.round(np.random.uniform(0.004, 0.012), 4),
                'stop_loss': np.round(np.random.uniform(0.003, 0.008), 4)
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
            'features': raw_data.values.astype(np.float64).tolist(),
            'index': raw_data.index.tz_localize(None).astype('datetime64[ns]').astype(np.int64).tolist(),
            'columns': raw_data.columns.tolist(),
            'dtypes': {
                'index': 'datetime64[ns]',
                'features': 'float64'
            }
        }
        results = []
        for i in range(samples):
            # 生成随机参数
            params = {
                'spread_threshold': np.round(np.random.uniform(0.002, 0.006), 4),
                'leverage': np.random.choice([2, 3, 4]),
                'max_hold_seconds': np.random.choice([3600, 7200, 14400]),
                'take_profit': np.round(np.random.uniform(0.004, 0.012), 4),
                'stop_loss': np.round(np.random.uniform(0.003, 0.008), 4)
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
            # 止盈参数变异 (新范围0.004-0.012)
            if np.random.rand() < 0.1:
                mutated_params['take_profit'] = np.clip(
                    np.random.normal(params['take_profit'], 0.001),
                    0.004, 0.012
                ).round(4)
            # 止损参数变异 (新范围0.003-0.008)
            if np.random.rand() < 0.1:
                mutated_params['stop_loss'] = np.clip(
                    np.random.normal(params['stop_loss'], 0.001),
                    0.003, 0.008
                ).round(4)
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
    from .config import load_base_config
    
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
    param_space = {
        'spread_threshold': np.round(np.linspace(0.002, 0.006, 5), 4),    # 优化价差范围0.2%-0.6%
        'leverage': [2, 3, 4],                     # 杠杆倍数优化选项
        'max_hold_seconds': [3600, 7200, 14400],   # 持仓时间1-4小时
        'take_profit': np.round(np.linspace(0.004, 0.012, 4), 4),  # 止盈0.4%-1.2%
        'stop_loss': np.round(np.linspace(0.003, 0.008, 4), 4),     # 止损0.3%-0.8%
        'min_funding_rate': np.round(np.linspace(0.0005, 0.002, 4), 4),  # 最低资金费率0.05%-0.2%
        'risk_per_trade': [0.01, 0.02, 0.03]  # 单笔交易风险1%-3%
    }
    optimizer = ParameterOptimizer(base_config, param_space)

    try:
        # 使用BacktestEngine加载预处理数据
        # 通过配置初始化引擎
        from .config import load_base_config
        config = load_base_config()
        engine = BacktestEngine(config)
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
