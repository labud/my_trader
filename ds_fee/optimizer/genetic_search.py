import numpy as np
from .base import BaseOptimizer
from .params import OptimizationParams
from .shared_state import SharedState

class GeneticOptimizer(BaseOptimizer):
    def validate_params(self, params: OptimizationParams) -> None:
        if params.population_size is not None and params.population_size <= 0:
            raise ValueError("population_size必须大于0")
        if params.generations is not None and params.generations <= 0:
            raise ValueError("generations必须大于0")

    def optimize(self, data, params: OptimizationParams, shared_state: SharedState):
        # 验证参数
        self.validate_params(params)
        
        # 从base_config获取参数范围
        param_ranges = self.base_config.get('optimizer', {}).get('param_ranges', {})
        if not param_ranges:
            raise ValueError("未在配置中找到参数优化范围配置(optimizer.param_ranges)")
        
        # 初始化种群
        population = self._initialize_population(params.population_size or 50)
        
        for gen in range(params.generations or 20):
            # 评估适应度
            # 过滤无效评估结果
            evaluated_pop = []
            for ind in population:
                res = evaluate_params(ind, data, self.base_config, self.dashboard, params.verbose)
                if res and isinstance(res, dict):
                    if 'metrics' not in res:
                        res['metrics'] = {
                            'annualized_return': -np.inf,
                            'max_drawdown': 1.0,
                            'sharpe_ratio': -np.inf,
                            'profit_factor': 0,
                            'win_rate': 0,
                            'volatility': np.inf,
                            'funding_ratio': 0
                        }
                    evaluated_pop.append(res)
            
            # 选择
            selected = self._selection(evaluated_pop)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            population = self._mutation(offspring)
            
            # 显示进度
            if params.verbose:
                self.dashboard.print_best_result(evaluated_pop)
        
        return sorted(evaluated_pop, key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf), reverse=True)

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

    def _mutation(self, offspring):
        """高斯变异（更新参数范围）"""
        mutated = []
        for params in offspring:
            mutated_params = params.copy()
            # 价差阈值变异 (新范围0.002-0.006)
            if np.random.rand() < 0.15:  # 增加变异概率
                mutated_params['spread_threshold'] = np.clip(
                    np.random.normal(params['spread_threshold'], 0.0015),  # 增加变异幅度
                    0.002, 0.006
                ).round(4)
            # 杠杆倍数变异 (新范围2-5)
            if np.random.rand() < 0.15:
                mutated_params['leverage'] = np.random.choice([2, 3, 4, 5])  # 增加杠杆选项
            # 持仓时间变异 (新选项)
            if np.random.rand() < 0.15:
                mutated_params['max_hold_seconds'] = np.random.choice([7200, 14400, 28800])  # 调整持仓时间
            # 止盈参数变异 (扩大范围)
            if np.random.rand() < 0.15:
                mutated_params['take_profit'] = np.random.choice(np.arange(0.006, 0.02, 0.002))  # 扩大止盈范围
            # 止损参数变异 (扩大范围)
            if np.random.rand() < 0.15:
                mutated_params['stop_loss'] = np.random.choice(np.arange(0.004, 0.012, 0.001))  # 扩大止损范围
            mutated.append(mutated_params)
        return mutated