import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from skopt import BayesSearchCV
from .BacktestEngine import BacktestEngine


class ParameterOptimizer:
    def __init__(self, base_config, param_space):
        self.base_config = base_config
        self.param_space = param_space

    def grid_search(self, data, n_jobs=-1):
        """网格搜索优化"""
        best_params = None
        best_sharpe = -np.inf

        grid = ParameterGrid(self.param_space)
        for params in grid:
            config = self.base_config.copy()
            config.update(params)

            engine = BacktestEngine(config)
            engine.load_data(data)
            engine.run_backtest()
            metrics = engine.analyze_performance()

            if metrics['夏普比率'] > best_sharpe:
                best_sharpe = metrics['夏普比率']
                best_params = params

        return best_params

    def bayesian_optimization(self, data, n_iter=50):
        """贝叶斯优化"""
        opt = BayesSearchCV(
            estimator=BacktestEngine(self.base_config),
            search_spaces=self.param_space,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3).split(data),
            scoring=self._sharpe_scorer,
            error_score='raise'
        )
        opt.fit(data.iloc[:, 0], data.iloc[:, 1])  # 假设数据格式为(features, target)
        return opt.best_params_

    def _sharpe_scorer(self, estimator, X, y):
        """自定义夏普比率评分函数"""
        estimator.run_backtest()
        metrics = estimator.analyze_performance()
        return metrics['sharpe_ratio']
