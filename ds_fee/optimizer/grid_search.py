from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from .base import BaseOptimizer, evaluate_params
from .params import OptimizationParams
from .shared_state import SharedState

class GridSearchOptimizer(BaseOptimizer):
    def validate_params(self, params: OptimizationParams) -> None:
        if params.n_jobs is not None and params.n_jobs == 0:
            raise ValueError("n_jobs不能为0")

    def optimize(self, data, params: OptimizationParams, shared_state: SharedState):
        # 验证参数
        self.validate_params(params)
        
        # 从base_config获取参数范围
        param_ranges = self.base_config.get('optimizer', {}).get('param_ranges', {})
        if not param_ranges:
            raise ValueError("未在配置中找到参数优化范围配置(optimizer.param_ranges)")
        
        # 生成参数网格
        param_grid = list(ParameterGrid(param_ranges))
        print(f"\n📊 开始网格搜索，参数组合数: {len(param_grid)}")
        
        # 并行执行回测，传递共享状态管理器实例
        results = Parallel(n_jobs=params.n_jobs, verbose=50)(
            delayed(evaluate_params)(
                param, 
                data, 
                self.base_config, 
                shared_state,  # 传递共享状态管理器实例
                params.verbose
            )
            for param in param_grid
        )
        
        # 过滤无效结果并排序
        valid_results = [r for r in results if r.get('metrics')]
        return sorted(valid_results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)