import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import os

class ResultVisualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = getattr(config, 'output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_results(self, results, preprocessed_data, save_to_file=True, verbose = True):
        """可视化回测结果"""
        # 统一时区校验
        def validate_timezone(df, name):
            """统一时区校验方法"""
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{name}索引不是时间类型")
            if df.index.tz is None:
                raise ValueError(f"{name}时间索引缺少时区信息")
            return df.index.tz

        # 校验原始数据时区
        base_tz = validate_timezone(preprocessed_data, "preprocessed_data")
        
        # 校验结果数据时区
        results_df = pd.DataFrame(results['equity_curve']).set_index('timestamp')
        results_tz = validate_timezone(results_df, "results")
        if base_tz != results_tz:
            raise ValueError(f"时区不一致: preprocessed_data({base_tz}) vs results({results_tz})")
            
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['figure.dpi'] = 100

        # 可视化逻辑（保持原有代码结构）
        # 启用快速绘图模式
        plt.ioff()
        plt.style.use('fast')
        
        # 性能优化增强版
        import time
        start_time = time.time()
        
        # 使用更高效的agg后端
        plt.switch_backend('agg')
            
        # 优化时区处理和采样间隔（30分钟）
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp']).dt.tz_convert(base_tz)
        
        # 最终优化：按日采样
        equity_df = (equity_df.set_index('timestamp')
                     .resample('D').last()  # 改为每日采样
                     .ffill())
        
        # 保持preprocessed_data原始时区
        preprocessed_data = (preprocessed_data
                             .resample('D').last()
                             .ffill())
        
        # 最终优化合并字段
        merged_df = pd.merge_asof(
            preprocessed_data[['close_spot']].reset_index(),  # 仅保留必要字段
            equity_df[['value', 'position']].reset_index(),
            on='timestamp',
            direction='nearest'
        ).set_index('timestamp').astype({'position': 'float32'})  # 优化内存
        
        # 添加必要的计算列并验证数据
        merged_df['position'] = merged_df['position'].fillna(0)
        
        # 验证时间序列是否有序
        is_sorted = (merged_df.index == merged_df.index.sort_values()).all()
        if not is_sorted:
            print("警告: 数据时间序列无序，正在进行排序...")
            merged_df = merged_df.sort_index()
        
        # 创建组合图表（资金曲线、价格曲线和仓位） 
        plt.figure(figsize=(14, 18))  # 增加总高度适应三个子图
        
        # 资金曲线子图（占40%高度）
        ax1 = plt.subplot(311)
        plt.subplots_adjust(hspace=0.3)  # 增加子图间距
        # 打印交易数据信息
        if 'trades' in results and results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            if verbose:
                print(f"交易数据概览:\n{trades_df.describe()}\n")
                print(f"前5笔交易记录:\n{trades_df.head()}\n")
        ax1.plot(merged_df.index, merged_df['value'], label='资金曲线', linewidth=1.5, color='#1f77b4')
        
        # 添加买卖点标记
        if 'trades' in results and results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            if verbose and not trades_df.empty:
                print(f"发现{trades_df.shape[0]}笔交易记录")
            # 使用timestamp作为开仓时间，exit_time作为平仓时间
            buy_dates = trades_df['timestamp']  # 开仓时间点
            sell_dates = trades_df['exit_time']  # 平仓时间点
            
            # 打印交易时间点信息
            if verbose:
                print(f"开仓时间点数量: {len(buy_dates)}")
                print(f"平仓时间点数量: {len(sell_dates)}")
                print(f"时间范围: {buy_dates.min()} 至 {sell_dates.max()}")

                print(f"买入交易数: {len(buy_dates)}, 卖出交易数: {len(sell_dates)}")
            
            # 使用统一的时间序列处理函数
            from ds_fee.utils.time_series import process_single_time_series
            
            # 处理买入和卖出时间
            buy_dt = process_single_time_series(
                pd.to_datetime(buy_dates, errors='coerce'),
                target_tz=base_tz
            ).dropna()
            
            sell_dt = process_single_time_series(
                pd.to_datetime(sell_dates, errors='coerce'),
                target_tz=base_tz
            ).dropna()
            
            # 处理merged_df的索引
            merged_df.index = process_single_time_series(
                merged_df.index,
                target_tz=base_tz
            )
            
            # 检查时间戳精度
            if (not buy_dt.empty) and verbose:
                print("\n[DEBUG] 时间戳精度示例:")
                print(f"buy_dt 第一个时间: {buy_dt[0]}")
                nearest_idx = np.searchsorted(merged_df.index, buy_dt[0], side='right')-1
                if nearest_idx >= 0:
                    print(f"对应的merged_df时间: {merged_df.index[nearest_idx]}")
                    print(f"时间差: {merged_df.index[nearest_idx] - buy_dt[0]}")
            
            # 使用valid_mask的反向条件获取无效交易点
            valid_buy_mask = (buy_dt >= merged_df.index[0]) & (buy_dt <= merged_df.index[-1])
            valid_sell_mask = (sell_dt >= merged_df.index[0]) & (sell_dt <= merged_df.index[-1])
            buy_dt_valid = buy_dt[valid_buy_mask]
            sell_dt_valid = sell_dt[valid_sell_mask]
            buy_dt_invalid = buy_dt[~valid_buy_mask]
            sell_dt_invalid = sell_dt[~valid_sell_mask]

            # 使用索引搜索（兼容空值情况）
            valid_buy = merged_df.index[np.searchsorted(merged_df.index, buy_dt_valid, side='right')-1]

            
            # 输出部分匹配失败的案例
            if len(buy_dt_valid) > len(valid_buy):
                print("\n[DEBUG] 匹配失败案例分析:")
                failed_matches = set(buy_dt_valid) - set(valid_buy)
                print(f"匹配失败的时间点示例(前3个):")
                for dt in list(failed_matches)[:3]:
                    print(f"时间点: {dt}")
                    closest_idx = np.searchsorted(merged_df.index, dt, side='right')-1
                    if closest_idx >= 0:
                        print(f"最接近的merged_df时间点: {merged_df.index[closest_idx]}")
                        print(f"时间差: {merged_df.index[closest_idx] - dt}")

            valid_sell = merged_df.index[np.searchsorted(merged_df.index, sell_dt_valid, side='right')-1]
         
            # 转换为唯一索引并确保时区一致
            valid_buy = pd.DatetimeIndex(valid_buy).tz_localize(None).tz_localize(base_tz).unique()
            valid_sell = pd.DatetimeIndex(valid_sell).tz_localize(None).tz_localize(base_tz).unique()
            
            # 确保索引存在于merged_df中
            valid_buy = valid_buy[valid_buy.isin(merged_df.index)]
            valid_sell = valid_sell[valid_sell.isin(merged_df.index)]
            
            if not valid_buy.empty:
                buy_values = merged_df.loc[valid_buy, 'value']
                ax1.scatter(valid_buy, buy_values, 
                            marker='^', color='green', s=100, label='买入点',
                            edgecolors='black', linewidths=0.5)
            if not valid_sell.empty:
                sell_values = merged_df.loc[valid_sell, 'value']
                ax1.scatter(valid_sell, sell_values, 
                            marker='v', color='red', s=100, label='卖出点',
                            edgecolors='black', linewidths=0.5)
            # 打印调试信息
            if verbose:
                print(f"有效买入时间点数量: {len(valid_buy)}/{len(buy_dates)}")
                print(f"有效卖出时间点数量: {len(valid_sell)}/{len(sell_dates)}")
        
            ax1.set_title('资金曲线', fontsize=12)
            ax1.set_ylabel('净值', color='#1f77b4')
            ax1.tick_params(axis='y', labelcolor='#1f77b4')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left')
            
            # 价格走势子图
            ax2 = plt.subplot(312, sharex=ax1)

            # 仓位子图
            ax3 = plt.subplot(313, sharex=ax1)
            ax3.step(merged_df.index, merged_df['position'], 
                    where='post', linewidth=1, color='#7f7f7f')
            ax3.fill_between(merged_df.index, merged_df['position'], 
                            alpha=0.2, color='#d62728', step='post')
            ax3.set_title('仓位变化')
            ax3.set_ylabel('仓位')
            ax3.grid(False)
            ax2.plot(merged_df.index, merged_df['close_spot'],
                   label='现货价格', linewidth=1.5, color='#2ca02c')
            
            # 核心买卖点标记逻辑
            if 'trades' in results and not results['trades']:
                return
                
            trades_df = pd.DataFrame(results['trades'])
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.tz_convert(base_tz)
            
            # 确保时区一致性
            if trades_df['timestamp'].dt.tz != merged_df.index.tz:
                if verbose:
                    print(f"[WARNING] 时区不一致，正在统一时区到: {base_tz}")
                trades_df['timestamp'] = trades_df['timestamp'].dt.tz_convert(base_tz)
            
            merged_trades = pd.merge_asof(
                trades_df.sort_values('timestamp'),
                merged_df[['close_spot']].reset_index(),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('12h')
            )
            
            # 清除无效数据并记录
            valid_trades = merged_trades.dropna(subset=['timestamp', 'close_spot'])
            if verbose:
                print(f"有效交易数: {len(valid_trades)} (丢弃{len(merged_trades)-len(valid_trades)}条无效数据)")
            
            # 提取开平仓数据
            buy_data = merged_trades[['timestamp', 'close_spot']]  # 开仓点数据
            sell_data = merged_trades[['exit_time', 'close_spot']].rename(columns={'exit_time': 'timestamp'})  # 平仓点数据
            
            # 绘制标记前添加数据校验
            if len(buy_data['timestamp']) != len(buy_data['close_spot']):
                raise ValueError(f"买入点数据异常: 时间戳数量({len(buy_data['timestamp'])}) 与价格数量({len(buy_data['close_spot'])}) 不匹配")
            if len(sell_data['timestamp']) != len(sell_data['close_spot']):
                raise ValueError(f"卖出点数据异常: 时间戳数量({len(sell_data['timestamp'])}) 与价格数量({len(sell_data['close_spot'])}) 不匹配")
            
            # 绘制标记
            if not buy_data.empty:
                ax2.scatter(buy_data['timestamp'], buy_data['close_spot'], 
                          marker='^', color='green', s=80, label='买入点')
            if not sell_data.empty:
                ax2.scatter(sell_data['timestamp'], sell_data['close_spot'],
                          marker='v', color='red', s=80, label='卖出点')
            
            ax2.set_ylabel('价格', color='#2ca02c')
            ax2.tick_params(axis='y', labelcolor='#2ca02c')
            
            plt.tight_layout()
            
            plt.tight_layout()
            
            # 保存组合图表
            if save_to_file:
                combined_path = os.path.join(self.output_dir, 'backtest_result.png')
                plt.savefig(combined_path, bbox_inches='tight', dpi=150, facecolor='white')
                print(f"组合图表已保存至: {combined_path}")
                plt.close()
            
            # 添加图例并格式化日期
            ax1.legend(loc='upper left', frameon=False)
            ax2.legend(loc='upper right', frameon=False)
            
            # 格式化x轴日期显示（应用在仓位子图）
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax3.xaxis.set_major_locator(locator)
            ax3.xaxis.set_major_formatter(formatter)
            
            # 优化保存配置
            # 生成交互式可视化
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 创建Plotly交互式图表
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.05,
                                  specs=[[{"type": "scatter"}],
                                         [{"type": "scatter"}],
                                         [{"type": "bar"}]])
                
                # 添加资金曲线
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['value'],
                                       name='资金曲线', line=dict(color='#1f77b4')),
                            row=1, col=1)
                
                # 添加买卖点到资金曲线
                if not valid_buy.empty:
                    buy_values = merged_df.loc[valid_buy, 'value']
                    fig.add_trace(go.Scatter(x=valid_buy, y=buy_values,
                                           mode='markers',
                                           name='买入点',
                                           marker=dict(symbol='triangle-up',
                                                      size=12,
                                                      color='green',
                                                      line=dict(color='black', width=1))),
                                row=1, col=1)
                if not valid_sell.empty:
                    sell_values = merged_df.loc[valid_sell, 'value']
                    fig.add_trace(go.Scatter(x=valid_sell, y=sell_values,
                                           mode='markers',
                                           name='卖出点',
                                           marker=dict(symbol='triangle-down',
                                                      size=12,
                                                      color='red',
                                                      line=dict(color='black', width=1))),
                                row=1, col=1)
                
                # 添加价格曲线
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['close_spot'],
                                       name='现货价格', line=dict(color='#2ca02c')),
                            row=2, col=1)
                
                # 添加买卖点到价格曲线
                if not buy_data.empty:
                    fig.add_trace(go.Scatter(x=buy_data['timestamp'], y=buy_data['close_spot'],
                                           mode='markers',
                                           name='买入点',
                                           marker=dict(symbol='triangle-up',
                                                      size=12,
                                                      color='green',
                                                      line=dict(color='black', width=1))),
                                row=2, col=1)
                if not sell_data.empty:
                    fig.add_trace(go.Scatter(x=sell_data['timestamp'], y=sell_data['close_spot'],
                                           mode='markers',
                                           name='卖出点',
                                           marker=dict(symbol='triangle-down',
                                                      size=12,
                                                      color='red',
                                                      line=dict(color='black', width=1))),
                                row=2, col=1)
                
                # 添加仓位柱状图
                fig.add_trace(go.Bar(x=merged_df.index, y=merged_df['position'],
                                   name='仓位', marker_color=np.where(merged_df['position']>0, 
                                                                     '#d62728', '#17becf')),
                            row=3, col=1)
                
                # 更新布局
                fig.update_layout(height=900, title_text="回测结果交互式可视化",
                                hovermode="x unified",
                                showlegend=False)
                
                # 保存交互式HTML
                html_path = os.path.join(self.output_dir, 'backtest_interactive.html')
                fig.write_html(html_path)
                print(f"交互式图表已保存至: {html_path}")
                
            except ImportError:
                print("提示：安装plotly可获得交互式图表 → pip install plotly")
                plt.close()
            else:
                # 交互式显示
                plt.switch_backend('TkAgg')
                plt.show()
                plt.close()

    def _save_results(self, results, params):
        """统一结果存储方法"""
        # 保持原有_save_results逻辑
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(os.path.join(self.output_dir, 'trades.csv'), index=False)
        
        positions_df = pd.DataFrame([
            {'timestamp': e['timestamp'], 'position': e['value'] - results['final_equity'] + e['value']} 
            for e in results['equity_curve']
        ])
        positions_df.to_csv(os.path.join(self.output_dir, 'positions.csv'), index=False)
        
        pd.DataFrame([results['risk_metrics']]).to_csv(
            os.path.join(self.output_dir, 'risk_metrics.csv'), 
            index=False
        )

    def print_backtest_results(self, results, initial_balance):
        """统一结果打印格式"""
        print(f"\n========== 回测结果汇总 ==========="
            f"\n初始资金: {initial_balance:,.2f}"
            f"\n最终资金: {results['final_equity']:,.2f}"
            f"\n总交易次数: {results['total_trades']}"
            f"\n胜率: {results['win_rate']:.2%}"
            f"\n最大回撤: {results['max_drawdown']:.2%}"
            f"\n夏普比率: {results['sharpe_ratio']:.2f}"
            f"\n===================================")

        # 打印交易明细
        if 'trades' in results and results['trades']:
            print("\n========== 交易明细 ===========\n")
            for i, trade in enumerate(results['trades'], 1):
                if trade.get('type') == 'close':
                    direction_str = "做多期货" if trade['direction'] > 0 else "做空期货"
                    print(f"交易 #{i}:")
                    print(f"  开仓时间: {trade['timestamp']}")
                    print(f"  平仓时间: {trade['exit_time']}")
                    print(f"  持仓方向: {direction_str}")
                    print(f"  开仓数量: {trade['qty']:.4f}")
                    print(f"  开仓价格: 现货={trade['price_spot']:.2f}, 期货={trade['price_future']:.2f}")
                    print(f"  平仓价格: 现货={trade['exit_price_spot']:.2f}, 期货={trade['exit_price_future']:.2f}")
                    print(f"  价差收益: {trade['spread_pnl']:.2f} USDT")
                    print(f"  资金费收益: {trade.get('funding_pnl', 0):.2f} USDT")
                    print(f"  开仓手续费成本: {trade.get('open_fee_cost', 0):.2f} USDT")
                    print(f"  平仓手续费成本: {trade.get('close_fee_cost', 0):.2f} USDT")
                    print(f"  总收益: {trade['pnl']:.2f} USDT")
                    print(f"  持仓时间: {trade.get('holding_hours', 0):.2f} 小时")
                    print(f"  平仓原因: {trade.get('close_reason', '未知')}")
                    print()
