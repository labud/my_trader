from ds_fee.backtest.data_processor import DataProcessor
from ds_fee.backtest.visualizer import ResultVisualizer
from ds_fee.backtest.backtest_core import BacktestCore

class BacktestEngine:
    """回测引擎入口类，协调各模块工作"""
    
    def __init__(self, config):
        self.data_processor = DataProcessor(config)
        self.backtest_core = BacktestCore(config, self.data_processor)
        self.visualizer = ResultVisualizer(config)

    def run_full_backtest(self, params, verbose=True):
        """执行完整回测流程"""
        # 1. 数据预处理
        self.data_processor.preprocess_data(verbose=verbose)
        
        # 2. 执行回测
        results = self.backtest_core.run_backtest(params, verbose=verbose)
        results['final_equity'] = results['cash']  # 添加最终权益字段
        
        # 3. 保存结果
        self.visualizer._save_results(results, params)
        
        # 4. 打印结果
        self.visualizer.print_backtest_results(results, self.backtest_core.initial_balance)
        
        # 5. 可视化
        self.visualizer.visualize_results(results, self.data_processor.preprocessed_data, verbose=verbose)
        
        return results

if __name__ == "__main__":
    # 保留原有主函数逻辑
    def main():
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from ds_fee.config import load_base_config
            config = load_base_config()
            engine = BacktestEngine(config)
            
            params = {
                "spread_threshold": 0.003,  # 降低价差阈值至0.3%以增加交易机会
                "leverage": 2,  # 保持杠杆以控制风险
                "max_hold_seconds": 28800,  # 保持最大持仓时间为8小时
                "min_funding_rate": 0.0003,  # 降低最小资金费率要求至0.03%
                "risk_per_trade": 0.02,  # 保持单笔交易风险不变
                "take_profit": 0.012,  # 保持止盈不变
                "stop_loss": 0.008  # 保持止损不变
            }
            
            results = engine.run_full_backtest(params, False)
    
        except Exception as e:
            print(f"发生错误: {str(e)}")
            raise
    
    main()
