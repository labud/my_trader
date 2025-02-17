import ccxt
import numpy as np
import time
from datetime import datetime


class RiskMonitor:
    def __init__(self, exchange):
        self.exchange = exchange
        self.historical_data = {
            'funding_rates': [],
            'spreads': []
        }
        self.last_update = time.time()

    def check_all_risks(self):
        """执行完整风险检查"""
        return {
            'funding_risk': self.check_funding_rate_risk(),
            'liquidity_risk': self.check_liquidity(),
            'exchange_risk': self.check_exchange_status(),
            'margin_risk': self.check_margin_level(),
            'spread_risk': self.check_spread(),
            'latency_risk': self.check_latency()
        }

    # ---------- 核心风险检查方法 ----------
    def check_funding_rate_risk(self):
        """资金费率异常风险检测"""
        try:
            current_rate = self.exchange.fetch_funding_rate('BTC/USDT:USDT')['fundingRate']
            self._update_historical('funding_rates', current_rate)

            # 趋势逆转检测
            if len(self.historical_data['funding_rates']) >= 3:
                avg_prev = np.mean(self.historical_data['funding_rates'][-3:])
                if (current_rate > 0 and avg_prev < -0.0001) or (current_rate < 0 and avg_prev > 0.0001):
                    return self._format_risk('high', 'reduce_position',
                                             f"费率趋势逆转 (当前:{current_rate:.4%} vs 过去3次平均:{avg_prev:.4%})")

            # 波动率检测
            volatility = np.std(self.historical_data['funding_rates'])
            if abs(current_rate) > 3 * volatility:
                return self._format_risk('critical', 'emergency_hedge',
                                         f"费率异常波动 (当前:{current_rate:.4%} 3σ范围:{3 * volatility:.4%})")

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('critical', 'pause_trading', f"费率获取失败: {str(e)}")

    def check_liquidity(self):
        """市场流动性风险检测"""
        try:
            order_book = self.exchange.fetch_order_book('BTC/USDT')
            bid_depth = sum([bid[1] for bid in order_book['bids'][:3]])
            ask_depth = sum([ask[1] for ask in order_book['asks'][:3]])

            # 安全交易量计算
            safe_volume = min(bid_depth, ask_depth) * 0.05
            if safe_volume < 0.1:  # 至少0.1 BTC流动性
                return self._format_risk('critical', 'halt_trading',
                                         f"流动性不足 (安全交易量:{safe_volume:.2f} BTC)")

            # 滑点分析
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            slippage = abs(order_book['asks'][0][0] - mid_price) / mid_price
            if slippage > 0.002:
                return self._format_risk('high', 'adjust_order_size',
                                         f"高滑点风险 ({slippage:.2%})")
            elif slippage > 0.001:
                return self._format_risk('medium', 'limit_order_only')

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('critical', 'pause_trading', f"订单簿获取失败: {str(e)}")

    def check_exchange_status(self):
        """交易所健康状态检测"""
        try:
            status = self.exchange.fetch_status()
            if status['status'] != 'ok':
                return self._format_risk('critical', 'switch_backup',
                                         f"交易所状态异常: {status['message']}")

            # API错误率监控
            error_rate = self.exchange.error_count / max(self.exchange.request_count, 1)
            if error_rate > 0.1:
                return self._format_risk('critical', 'switch_api_node',
                                         f"API错误率过高 ({error_rate:.1%})")
            elif error_rate > 0.05:
                return self._format_risk('high', 'reduce_frequency')

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('critical', 'emergency_stop', f"状态检查失败: {str(e)}")

    def check_margin_level(self):
        """保证金风险检测"""
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions()

            # 计算保证金使用率
            free_margin = balance['USDT']['free']
            used_margin = sum([pos['initialMargin'] for pos in positions])
            margin_ratio = used_margin / (free_margin + 1e-6)  # 避免除零

            if margin_ratio > 0.8:
                return self._format_risk('critical', 'deposit_margin',
                                         f"保证金使用率过高 ({margin_ratio:.0%})")
            elif margin_ratio > 0.6:
                return self._format_risk('high', 'reduce_leverage')

            # 维持保证金检查
            for pos in positions:
                maint_ratio = pos['maintenanceMargin'] / pos['notional']
                if maint_ratio > 0.05:
                    return self._format_risk('medium', 'partial_close',
                                             f"{pos['symbol']} 维持保证金率过高 ({maint_ratio:.1%})")

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('critical', 'flat_positions', f"保证金数据获取失败: {str(e)}")

    def check_spread(self):
        """现货与合约价差风险"""
        try:
            # 获取双市场价格
            spot = self.exchange.fetch_ticker('BTC/USDT')['last']
            perp = self.exchange.fetch_ticker('BTC/USDT:USDT')['last']
            current_spread = (perp - spot) / spot

            self._update_historical('spreads', current_spread)
            hist_data = self.historical_data['spreads']

            # 动态阈值计算
            if len(hist_data) >= 24:  # 24小时数据
                mean = np.mean(hist_data)
                std = np.std(hist_data)
                if abs(current_spread - mean) > 3 * std:
                    return self._format_risk('critical', 'emergency_hedge',
                                             f"价差异常 (当前:{current_spread:.2%} 3σ范围:{3 * std:.2%})")

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('high', 'pause_trading', f"价差获取失败: {str(e)}")

    def check_latency(self):
        """网络延迟风险检测"""
        try:
            start = time.time()
            self.exchange.fetch_order_book('BTC/USDT', limit=5)
            latency = time.time() - start

            if latency > 2.0:
                return self._format_risk('critical', 'switch_network',
                                         f"严重延迟 ({latency:.2f}s)")
            elif latency > 0.5:
                return self._format_risk('medium', 'retry_order')

            return self._format_risk('low', None)

        except Exception as e:
            return self._format_risk('critical', 'emergency_stop', f"延迟检测失败: {str(e)}")

    # ---------- 辅助方法 ----------
    def _update_historical(self, data_type, value):
        """更新历史数据队列"""
        if len(self.historical_data[data_type]) >= 1000:  # 保持最多1000个数据点
            self.historical_data[data_type].pop(0)
        self.historical_data[data_type].append(value)
        self.last_update = time.time()

    def _format_risk(self, level, action=None, details=''):
        """统一风险响应格式"""
        return {
            'risk_level': level,
            'recommended_action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化交易所连接
    exchange = ccxt.binance({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET',
        'enableRateLimit': True
    })

    # 创建监控实例
    monitor = RiskMonitor(exchange)

    # 执行全面风险检查
    risks = monitor.check_all_risks()
    for risk_type, risk_info in risks.items():
        print(f"[{risk_type.upper()}] 风险等级: {risk_info['risk_level']}")
        print(f"建议操作: {risk_info['recommended_action']}")
        print(f"详细信息: {risk_info['details']}\n")
