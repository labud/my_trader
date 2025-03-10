from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizationParams:
    """优化器参数配置类
    
    封装所有优化器共用的输入参数，提供默认值
    
    Attributes:
        n_jobs: 并行任务数（用于网格搜索，-1表示使用全部CPU核心）
        samples: 采样数量（用于随机搜索）
        population_size: 种群大小（用于遗传算法）
        generations: 迭代代数（用于遗传算法）
        verbose: 是否打印详细日志
    """
    n_jobs: int = -1
    samples: Optional[int] = None
    population_size: Optional[int] = None
    generations: Optional[int] = None
    verbose: bool = False