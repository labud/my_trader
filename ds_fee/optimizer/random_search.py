import numpy as np
from .base import BaseOptimizer, evaluate_params
from .params import OptimizationParams

class RandomSearchOptimizer(BaseOptimizer):
    def validate_params(self, params: OptimizationParams) -> None:
        if params.samples is not None and params.samples <= 0:
            raise ValueError("samples必须大于0")

    def optimize(self, data, params: OptimizationParams):
        # 验证参数
        self.validate_params(params)
        
        # 从base_config获取参数范围
        param_ranges = self.base_config.get('optimizer', {}).get('param_ranges', {})
        if not param_ranges:
            raise ValueError("未在配置中找到参数优化范围配置(optimizer.param_ranges)")
        
        print(f"\n🎲 开始随机采样，样本数: {params.samples or 500}")
        
        results = []
        for i in range(params.samples or 500):
            # 生成随机参数
            params_dict = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, list):
                    params_dict[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, dict) and 'min' in param_range and 'max' in param_range:
                    params_dict[param_name] = np.round(
                        np.random.uniform(param_range['min'], param_range['max']),
                        param_range.get('decimals', 4)
                    )
            
            # 评估参数
            result = evaluate_params(params_dict, data, self.base_config, self.dashboard, params.verbose)
            results.append(result)
            
            # 更新进度
            if (i+1) % 10 == 0 or (i+1) == (params.samples or 500):
                print(f"进度: {i+1}/{params.samples or 500} ({(i+1)/(params.samples or 500):.1%})")
                self.dashboard.print_best_result(results)
        
        # 过滤无效结果并排序
        valid_results = [r for r in results if r.get('metrics')]
        return sorted(valid_results, key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf), reverse=True)