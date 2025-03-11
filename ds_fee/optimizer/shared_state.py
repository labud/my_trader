import numpy as np
import multiprocessing
import os

def increment_counter(shared_state):
    """增加计数器并返回当前值"""
    with shared_state.lock:
        shared_state.counter.value += 1
        pid = os.getpid()
        # print(f"cnt {shared_state.counter.value} 在进程 {pid} 中执行")
        return shared_state.counter.value

def update_best_result(shared_state, result):
    """更新最优结果"""
    with shared_state.lock:
        current_best = shared_state.state
        current_best_sharpe = current_best.get('sharpe_ratio', -np.inf)
        new_sharpe = result['metrics'].get('sharpe_ratio', -np.inf)
        
        if new_sharpe > current_best_sharpe:
            shared_state.state.update({
                'sharpe_ratio': new_sharpe,
                'annualized_return': result['metrics'].get('annualized_return', -np.inf),
                'metrics': result['metrics'],
                'params': result['params']
            })
            return True
        return False

def get_best_result(shared_state):
    """获取最优结果"""
    with shared_state.lock:
        if not shared_state.state:
            return {
                'sharpe_ratio': -np.inf,
                'annualized_return': -np.inf,
                'params': {},
                'metrics': {}
            }
        return dict(shared_state.state)

class SharedState:
    def __init__(self, manager):
        self.state = manager.dict()
        self.counter = manager.Value('i', 0)
        self.lock = manager.Lock()