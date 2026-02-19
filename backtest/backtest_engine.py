# 回测引擎

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

class BacktestEngine:
    """
    回测引擎类，用于执行策略回测
    """
    
    def __init__(self, config=None):
        """
        初始化回测引擎
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认配置
        default_config = {
            'start_date': '2019-01-01',  # 回测开始日期
            'end_date': '2023-12-31',  # 回测结束日期
            'benchmark': '000300',  # 基准指数
            'account': 1000000,  # 初始资金
            'commission_rate': 0.0003,  # 手续费率
            'slippage_rate': 0.0001,  # 滑点率
            'trade_frequency': 'day',  # 交易频率
            'rebalance_dates': None,  # 再平衡日期
            'verbose': True,  # 是否打印详细信息
            'risk_management': {  # 风控配置
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.20,
                'trailing_stop_pct': 0.15,
                'max_drawdown_limit': 0.25
            }
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
        
        # 初始化回测结果
        self.results = None
    
    def run_backtest(self, strategy, data):
        """
        运行回测
        
        参数:
            strategy: 策略实例
            data: 市场数据
            
        返回:
            dict: 回测结果
        """
        # 确保数据是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("数据必须是pandas.DataFrame")
        
        # 获取配置参数
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        account = self.config['account']
        commission_rate = self.config['commission_rate']
        slippage_rate = self.config['slippage_rate']
        trade_frequency = self.config['trade_frequency']
        verbose = self.config['verbose']
        
        # 确保数据按日期排序
        data = data.sort_values('date')
        
        # 获取交易日期
        if trade_frequency == 'day':
            trade_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif trade_frequency == 'week':
            trade_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        elif trade_frequency == 'month':
            trade_dates = pd.date_range(start=start_date, end=end_date, freq='BM')
        else:
            raise ValueError(f"不支持的交易频率: {trade_frequency}")
        
        # 如果指定了再平衡日期，则使用指定的日期
        if self.config['rebalance_dates'] is not None:
            trade_dates = pd.to_datetime(self.config['rebalance_dates'])
        
        # 初始化回测结果
        results = {
            'dates': [],
            'portfolio_value': [],
            'returns': [],
            'benchmark_value': [],
            'benchmark_returns': [],
            'positions': [],
            'trades': []
        }
        
        # 初始化投资组合
        portfolio_value = account
        positions = {}
        cash = account
        
        # 获取基准指数数据
        benchmark = self.config['benchmark']
        benchmark_data = data[data['stock_id'] == benchmark].copy()
        if not benchmark_data.empty:
            benchmark_data = benchmark_data.set_index('date')['close']
            benchmark_data = benchmark_data.reindex(trade_dates, method='ffill')
            benchmark_value = benchmark_data / benchmark_data.iloc[0] * account
        else:
            # 如果没有基准数据，则使用等值线
            benchmark_value = pd.Series(account, index=trade_dates)
        
        # 重置策略状态
        strategy.reset()
        
        # 重置风控状态
        self.risk_manager.reset()
        
        # 执行回测
        for i, date in enumerate(trade_dates):
            if verbose and i % 20 == 0:
                print(f"回测进度: {i}/{len(trade_dates)} ({date.strftime('%Y-%m-%d')})")
            
            # 获取当前日期的市场数据
            current_data = data[data['date'] <= date].copy()
            daily_data = data[data['date'] == date].set_index('stock_id')['close'].to_dict()
            
            # 1. 更新风控状态（账户价值）
            is_liquidated = self.risk_manager.update_portfolio_value(portfolio_value)
            
            # 2. 检查风控平仓信号
            risk_close_list, risk_reasons = self.risk_manager.check_positions(daily_data)
            
            if verbose and risk_close_list:
                for stock in risk_close_list:
                    print(f"[{date.strftime('%Y-%m-%d')}] 风控触发: {stock} - {risk_reasons[stock]}")
            
            # 生成交易信号
            signals = strategy.generate_signals(current_data)
            
            # 生成交易决策
            trade_decision = strategy.generate_trade_decision(signals)
            
            # 风控干预：强制修改交易决策
            # 如果触发了账户级强平，清空所有目标权重
            if is_liquidated:
                if verbose:
                    print(f"[{date.strftime('%Y-%m-%d')}] 账户级风控触发，强制清仓")
                # 将所有当前持仓的目标权重设为0
                liquidation_decision = []
                for stock_id in positions.keys():
                    liquidation_decision.append({'stock_id': stock_id, 'weight': 0.0})
                trade_decision = pd.DataFrame(liquidation_decision)
            
            # 如果有个股风控触发，强制将其目标权重设为0
            elif risk_close_list:
                # 确保trade_decision包含这些股票的卖出指令
                # 如果trade_decision里已经有这些股票，修改权重为0
                # 如果没有，添加进去
                
                # 先把DataFrame转为字典方便操作
                decision_dict = {}
                if not trade_decision.empty:
                    decision_dict = trade_decision.set_index('stock_id')['weight'].to_dict()
                
                for stock_id in risk_close_list:
                    decision_dict[stock_id] = 0.0
                
                # 重建DataFrame
                trade_decision = pd.DataFrame([{'stock_id': k, 'weight': v} for k, v in decision_dict.items()])
            
            # 更新策略持仓
            strategy.update_positions(trade_decision, date)
            
            # 执行交易
            trades = []
            new_positions = {}
            
            # 处理所有涉及的股票（包括持仓的和决策中的）
            all_involved_stocks = set(positions.keys())
            if not trade_decision.empty:
                all_involved_stocks.update(trade_decision['stock_id'].tolist())
            
            # 将trade_decision转换为字典以便快速查找
            target_weights = {}
            if not trade_decision.empty:
                target_weights = trade_decision.set_index('stock_id')['weight'].to_dict()
            
            for stock_id in all_involved_stocks:
                target_weight = target_weights.get(stock_id, positions.get(stock_id, 0) * daily_data.get(stock_id, 0) / portfolio_value if portfolio_value > 0 else 0)
                
                # 获取股票价格
                if stock_id in daily_data:
                    price = daily_data[stock_id]
                else:
                    # 如果当天没有数据，跳过交易，保持持仓
                    if stock_id in positions:
                        new_positions[stock_id] = positions[stock_id]
                    continue
                
                # 计算目标持仓金额
                target_value = portfolio_value * target_weight
                
                # 计算当前持仓金额
                current_value = positions.get(stock_id, 0) * price
                
                # 计算交易金额
                trade_value = target_value - current_value
                
                # 简单的阈值判断，避免微小交易
                if abs(trade_value) > 100:  # 最小交易额
                    # 计算交易成本
                    commission = abs(trade_value) * commission_rate
                    slippage = abs(trade_value) * slippage_rate
                    cost = commission + slippage
                    
                    # 更新现金
                    cash -= trade_value + cost
                    
                    # 更新持仓
                    new_shares = trade_value / price
                    new_positions[stock_id] = positions.get(stock_id, 0) + new_shares
                    
                    # 记录交易
                    trades.append({
                        'date': date,
                        'stock_id': stock_id,
                        'price': price,
                        'shares': new_shares,
                        'value': trade_value,
                        'cost': cost
                    })
                    
                    # 通知风控管理器
                    if new_shares > 0: # 买入
                        # 如果是新开仓或加仓，更新成本价（这里简化为最新买入价，也可以做加权平均）
                        self.risk_manager.on_position_open(stock_id, price)
                    elif new_positions[stock_id] <= 0: # 完全卖出
                        self.risk_manager.on_position_close(stock_id)
                        if stock_id in new_positions:
                            del new_positions[stock_id]
                            
                elif stock_id in positions:
                    # 保持原有持仓
                    new_positions[stock_id] = positions[stock_id]
            
            # 更新持仓
            
            # 更新持仓
            positions = new_positions
            
            # 计算投资组合价值
            portfolio_value = cash
            for stock_id, shares in positions.items():
                stock_data = data[(data['stock_id'] == stock_id) & (data['date'] <= date)]
                if not stock_data.empty:
                    price = stock_data.iloc[-1]['close']
                    portfolio_value += shares * price
            
            # 记录结果
            results['dates'].append(date)
            results['portfolio_value'].append(portfolio_value)
            results['positions'].append(positions.copy())
            results['trades'].extend(trades)
        
        # 计算收益率
        portfolio_values = pd.Series(results['portfolio_value'], index=results['dates'])
        results['returns'] = portfolio_values.pct_change().fillna(0).values
        
        # 计算基准收益率
        benchmark_returns = benchmark_value.pct_change().fillna(0).values
        results['benchmark_value'] = benchmark_value.values
        results['benchmark_returns'] = benchmark_returns
        
        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(results['returns'], benchmark_returns)
        results.update(risk_metrics)
        
        # 计算额外的风险指标（统一键名并修复未定义变量）
        returns = np.array(results['returns'])
        annual_return = results['annual_return']
        # Calmar Ratio
        results['calmar_ratio'] = annual_return / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 0
        
        # 最大回撤持续时间（按净值序列计算）
        portfolio_values_series = pd.Series(results['portfolio_value'], index=results['dates'])
        running_max = portfolio_values_series.cummax()
        drawdown_flag = (portfolio_values_series < running_max)
        if drawdown_flag.any():
            # 统计最长连续处于水下的区间长度
            groups = drawdown_flag.astype(int).groupby(drawdown_flag.astype(int).diff().ne(0).cumsum()).sum()
            results['max_drawdown_duration'] = int(groups.max())
        else:
            results['max_drawdown_duration'] = 0
        
        # 偏度与峰度
        returns_series = pd.Series(returns)
        results['skewness'] = float(returns_series.skew())
        results['kurtosis'] = float(returns_series.kurtosis())
        
        # 捕获比率（上行期间的策略/基准收益比）
        up_mask = benchmark_returns > 0
        up_capture = (returns[up_mask].mean() / benchmark_returns[up_mask].mean()) if benchmark_returns[up_mask].mean() != 0 else 0
        results['capture_ratio'] = float(up_capture)
        
        # 欧米茄比率 (Omega Ratio) - 阈值0
        threshold = 0.0
        up_part = returns[returns > threshold] - threshold
        down_part = threshold - returns[returns < threshold]
        results['omega_ratio'] = float(np.sum(up_part) / np.sum(down_part)) if np.sum(down_part) != 0 else float('inf')
        
        # 特雷诺比率 (Treynor Ratio)
        risk_free_rate = 0.03
        results['treynor_ratio'] = float((annual_return - risk_free_rate) / results['beta']) if results['beta'] != 0 else 0
        
        # 尾部比率 (Tail Ratio)
        percentile = 5
        upper_percentile = float(np.percentile(returns, 100 - percentile))
        lower_percentile = float(np.percentile(returns, percentile))
        results['tail_ratio'] = abs(upper_percentile / lower_percentile) if lower_percentile != 0 else float('inf')
        
        # 风险价值 VaR/CVaR
        var_5 = float(np.percentile(returns, percentile))
        results['value_at_risk'] = var_5
        results['conditional_value_at_risk'] = float(np.mean(returns[returns <= var_5])) if np.any(returns <= var_5) else 0
        
        self.results = results
        return results
    
    def calculate_risk_metrics(self, returns, benchmark_returns):
        """
        计算风险指标
        
        参数:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        返回:
            dict: 风险指标
        """
        # 确保输入是numpy数组
        returns = np.array(returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # 计算累积收益率
        cum_returns = (1 + returns).cumprod() - 1
        cum_benchmark_returns = (1 + benchmark_returns).cumprod() - 1
        
        # 计算年化收益率
        n_years = len(returns) / 252  # 假设一年有252个交易日
        annual_return = (1 + cum_returns[-1]) ** (1 / n_years) - 1
        annual_benchmark_return = (1 + cum_benchmark_returns[-1]) ** (1 / n_years) - 1
        
        # 计算波动率
        volatility = returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
        benchmark_sharpe_ratio = (annual_benchmark_return - risk_free_rate) / benchmark_volatility if benchmark_volatility != 0 else 0
        
        # 计算最大回撤
        cum_returns_series = pd.Series((1 + returns).cumprod())
        running_max = cum_returns_series.cummax()
        drawdown = (cum_returns_series / running_max) - 1
        max_drawdown = drawdown.min()
        
        # 计算信息比率
        excess_returns = returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        # 计算胜率
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        # 计算索提诺比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # 计算贝塔系数
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance != 0 else 0
        
        # 计算阿尔法
        alpha = annual_return - (risk_free_rate + beta * (annual_benchmark_return - risk_free_rate))
        
        # 返回风险指标
        return {
            'annual_return': annual_return,
            'annual_benchmark_return': annual_benchmark_return,
            'volatility': volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'sortino_ratio': sortino_ratio,
            'beta': beta,
            'alpha': alpha
        }
    
    def get_results(self):
        """
        获取回测结果
        
        返回:
            dict: 回测结果
        """
        if self.results is None:
            raise ValueError("尚未运行回测，请先调用run_backtest方法")
        
        return self.results
    
    def get_summary(self):
        """
        获取回测摘要
        
        返回:
            pandas.DataFrame: 回测摘要
        """
        if self.results is None:
            raise ValueError("尚未运行回测，请先调用run_backtest方法")
        
        # 提取风险指标
        metrics = [
            'annual_return', 'annual_benchmark_return',
            'volatility', 'benchmark_volatility',
            'sharpe_ratio', 'benchmark_sharpe_ratio',
            'max_drawdown', 'information_ratio',
            'win_rate', 'sortino_ratio',
            'beta', 'alpha'
        ]
        
        summary = {}
        for metric in metrics:
            if metric in self.results:
                summary[metric] = self.results[metric]
        
        return pd.DataFrame(summary, index=['value']).T
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未运行回测，请先调用run_backtest方法")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
            
            # 创建图形
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # 绘制投资组合价值和基准价值
            dates = self.results['dates']
            portfolio_values = self.results['portfolio_value']
            benchmark_values = self.results['benchmark_value']
            
            ax1.plot(dates, portfolio_values, label='Portfolio')
            ax1.plot(dates, benchmark_values, label='Benchmark')
            ax1.set_title('Portfolio Value vs Benchmark')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True)
            
            # 设置x轴日期格式
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 绘制收益率
            returns = self.results['returns']
            benchmark_returns = self.results['benchmark_returns']
            
            ax2.plot(dates[1:], returns[1:], label='Strategy Returns')
            ax2.plot(dates[1:], benchmark_returns[1:], label='Benchmark Returns')
            ax2.set_title('Daily Returns')
            ax2.set_ylabel('Returns')
            ax2.legend()
            ax2.grid(True)
            
            # 设置x轴日期格式
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 绘制回撤
            portfolio_values_series = pd.Series(portfolio_values, index=dates)
            running_max = portfolio_values_series.cummax()
            drawdown = (portfolio_values_series / running_max) - 1
            
            benchmark_values_series = pd.Series(benchmark_values, index=dates)
            benchmark_running_max = benchmark_values_series.cummax()
            benchmark_drawdown = (benchmark_values_series / benchmark_running_max) - 1
            
            ax3.fill_between(dates, 0, drawdown, color='red', alpha=0.3, label='Strategy Drawdown')
            ax3.fill_between(dates, 0, benchmark_drawdown, color='blue', alpha=0.3, label='Benchmark Drawdown')
            ax3.set_title('Drawdown')
            ax3.set_ylabel('Drawdown')
            ax3.legend()
            ax3.grid(True)
            
            # 设置x轴日期格式
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # 设置y轴百分比格式
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            if save_path is not None:
                plt.savefig(save_path)
                print(f"图形已保存至 {save_path}")
            
            # 显示图形
            plt.show()
            
        except ImportError:
            print("绘图需要matplotlib库，请安装后再试")
    
    def save_results(self, path):
        """
        保存回测结果
        
        参数:
            path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未运行回测，请先调用run_backtest方法")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 提取可序列化的结果
        serializable_results = {}
        for key, value in self.results.items():
            if key == 'dates':
                serializable_results[key] = [d.strftime('%Y-%m-%d') for d in value]
            elif key == 'positions':
                # 将持仓转换为可序列化的格式
                serializable_positions = []
                for pos in value:
                    serializable_pos = {str(k): float(v) for k, v in pos.items()}
                    serializable_positions.append(serializable_pos)
                serializable_results[key] = serializable_positions
            elif key == 'trades':
                # 将交易记录转换为可序列化的格式
                serializable_trades = []
                for trade in value:
                    serializable_trade = {}
                    for k, v in trade.items():
                        if k == 'date':
                            serializable_trade[k] = v.strftime('%Y-%m-%d')
                        else:
                            serializable_trade[k] = float(v) if isinstance(v, (int, float, np.number)) else v
                    serializable_trades.append(serializable_trade)
                serializable_results[key] = serializable_trades
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        # 保存结果
        import json
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"回测结果已保存至 {path}")
