import ccxt
import time
import threading
import numpy as np
from queue import Queue
from datetime import datetime


# ==================== 异步订单执行器 ====================
class AsyncOrderExecutor:
    def __init__(self, config):
        self.config = config
        self.order_queue = Queue()
        self.results = {}
        self.error_flag = False

        # 初始化交易所连接
        self.exchange_perp = ccxt.binance({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
        self.exchange_spot = ccxt.binance({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'enableRateLimit': True
        })

    def _worker(self):
        """工作线程执行函数"""
        while not self.order_queue.empty() and not self.error_flag:
            task_id, order_info = self.order_queue.get()
            try:
                exchange = getattr(self, f"exchange_{order_info['account_type']}")
                order = exchange.create_order(
                    symbol=order_info['symbol'],
                    type=order_info['type'],
                    side=order_info['side'],
                    amount=order_info['amount'],
                    params=order_info.get('params', {})
                )
                self.results[task_id] = {
                    'status': 'filled',
                    'order_id': order['id'],
                    'price': order['price'],
                    'timestamp': datetime.now().isoformat()
                }
                print(f"订单 {task_id} 执行成功: {order_info['side']} {order_info['amount']} {order_info['symbol']}")
            except Exception as e:
                self.error_flag = True
                self.results[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"订单 {task_id} 执行失败: {str(e)}")
            finally:
                self.order_queue.task_done()

    def execute_arbitrage(self, funding_rate):
        """执行套利策略"""
        print(f"\n当前资金费率: {funding_rate:.4%}")
        order_list = []

        # 生成订单列表
        if funding_rate > self.config['thresholds']['short_threshold']:
            print("执行正费率套利策略")
            order_list.extend([
                {
                    'account_type': 'perp',
                    'symbol': self.config['symbol'] + ':USDT',
                    'type': 'market',
                    'side': 'sell',
                    'amount': self.config['position_size'],
                    'params': {'leverage': self.config['leverage']}
                },
                {
                    'account_type': 'spot',
                    'symbol': self.config['symbol'],
                    'type': 'market',
                    'side': 'buy',
                    'amount': self.config['position_size']
                }
            ])
        elif funding_rate < self.config['thresholds']['long_threshold']:
            print("执行负费率套利策略")
            order_list.extend([
                {
                    'account_type': 'perp',
                    'symbol': self.config['symbol'] + ':USDT',
                    'type': 'market',
                    'side': 'buy',
                    'amount': self.config['position_size'],
                    'params': {'leverage': self.config['leverage']}
                },
                {
                    'account_type': 'spot',
                    'symbol': self.config['symbol'],
                    'type': 'market',
                    'side': 'sell',
                    'amount': self.config['position_size'],
                    'params': {'marginMode': 'isolated'}
                }
            ])
        else:
            print("资金费率在正常范围内，不执行操作")
            return False

        # 提交订单到队列
        for idx, order in enumerate(order_list):
            self.order_queue.put((f'task_{idx}', order))

        # 启动工作线程
        threads = []
        for _ in range(min(2, len(order_list))):  # 最多2个并发线程
            t = threading.Thread(target=self._worker)
            t.start()
            threads.append(t)

        # 等待执行完成
        start_time = time.time()
        while not self.order_queue.empty():
            if time.time() - start_time > self.config['timeout']:
                self.error_flag = True
                print("订单执行超时")
                break
            time.sleep(0.1)

        # 等待所有线程结束
        for t in threads:
            t.join()

        # 检查执行结果
        success = all(result['status'] == 'filled' for result in self.results.values())
        if not success:
            print("启动交易回滚...")
            self._rollback_orders(order_list)
            return False

        print("套利交易成功完成")
        return True

    def _rollback_orders(self, original_orders):
        """回滚已成功的订单"""
        rollback_orders = []
        for order in original_orders:
            if any(res['status'] == 'filled' for res in self.results.values()
                   if res.get('order_id') == order.get('order_id')):
                rollback_order = order.copy()
                rollback_order['side'] = 'buy' if order['side'] == 'sell' else 'sell'
                rollback_orders.append(rollback_order)

        if rollback_orders:
            print("执行回滚订单...")
            for idx, order in enumerate(rollback_orders):
                self.order_queue.put((f'rollback_{idx}', order))

            threads = []
            for _ in range(min(2, len(rollback_orders))):
                t = threading.Thread(target=self._worker)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            print(f"回滚完成，结果: {self.results}")


# ==================== 配置参数 ====================
CONFIG = {
    'api_key': 'YOUR_API_KEY',
    'api_secret': 'YOUR_API_SECRET',
    'symbol': 'BTC/USDT',
    'position_size': 0.001,  # BTC数量
    'leverage': 3,
    'timeout': 10,  # 秒
    'thresholds': {
        'long_threshold': -0.0005,  # -0.05%
        'short_threshold': 0.001  # +0.1%
    }
}

# ==================== 示例执行 ====================
if __name__ == "__main__":
    # 初始化套利执行器
    arbitrager = AsyncOrderExecutor(CONFIG)

    # 模拟获取资金费率（实际应从交易所API获取）
    test_funding_rate = 0.0012  # +0.12%

    # 执行套利
    success = arbitrager.execute_arbitrage(test_funding_rate)

    # 打印最终结果
    if success:
        print("\n=== 交易汇总 ===")
        for task_id, result in arbitrager.results.items():
            print(f"{task_id}: {result['side']} {result['amount']} @ {result['price']}")
    else:
        print("\n!!! 交易失败 !!!")
        for task_id, result in arbitrager.results.items():
            if result['status'] != 'filled':
                print(f"{task_id} 失败原因: {result.get('error', '未知错误')}")
