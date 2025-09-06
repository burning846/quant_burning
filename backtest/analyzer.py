# 回测结果分析器

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime

class BacktestAnalyzer:
    """
    回测结果分析器，用于分析回测结果并生成报告
    """
    
    def __init__(self, results=None):
        """
        初始化分析器
        
        参数:
            results: 回测结果字典
        """
        self.results = results
    
    def load_results(self, results):
        """
        加载回测结果
        
        参数:
            results: 回测结果字典
        """
        self.results = results
    
    def load_results_from_file(self, file_path):
        """
        从文件加载回测结果
        
        参数:
            file_path: 结果文件路径
        """
        import json
        with open(file_path, 'r') as f:
            self.results = json.load(f)
        
        # 将日期字符串转换为日期对象
        if 'dates' in self.results:
            self.results['dates'] = [datetime.strptime(d, '%Y-%m-%d') for d in self.results['dates']]
        
        # 将交易记录中的日期字符串转换为日期对象
        if 'trades' in self.results:
            for trade in self.results['trades']:
                if 'date' in trade:
                    trade['date'] = datetime.strptime(trade['date'], '%Y-%m-%d')
    
    def get_summary(self):
        """
        获取回测摘要
        
        返回:
            pandas.DataFrame: 回测摘要
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 提取风险指标
        metrics = [
            'annual_return', 'annual_benchmark_return',
            'volatility', 'benchmark_volatility',
            'sharpe_ratio', 'benchmark_sharpe_ratio',
            'max_drawdown', 'information_ratio',
            'win_rate', 'sortino_ratio',
            'beta', 'alpha',
            'calmar_ratio', 'omega_ratio',
            'treynor_ratio', 'tail_ratio',
            'value_at_risk', 'conditional_value_at_risk',
            'max_drawdown_duration', 'skewness',
            'kurtosis', 'capture_ratio'
        ]
        
        summary = {}
        for metric in metrics:
            if metric in self.results:
                summary[metric] = self.results[metric]
        
        # 创建摘要DataFrame
        summary_df = pd.DataFrame(summary, index=['value']).T
        
        # 格式化显示
        formatted_summary = summary_df.copy()
        for metric in formatted_summary.index:
            if 'return' in metric or 'ratio' in metric or 'drawdown' in metric or metric in ['win_rate', 'alpha', 'beta']:
                formatted_summary.loc[metric, 'value'] = f"{float(formatted_summary.loc[metric, 'value']):.4f}"
        
        return formatted_summary
    
    def plot_equity_curve(self, save_path=None):
        """
        绘制权益曲线
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制投资组合价值和基准价值
        dates = self.results['dates']
        portfolio_values = self.results['portfolio_value']
        benchmark_values = self.results['benchmark_value']
        
        plt.plot(dates, portfolio_values, label='策略')
        plt.plot(dates, benchmark_values, label='基准')
        plt.title('策略与基准权益曲线对比')
        plt.ylabel('价值')
        plt.grid(True)
        plt.legend()
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"权益曲线图已保存至 {save_path}")
        
        plt.show()
    
    def plot_returns(self, save_path=None):
        """
        绘制收益率曲线
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制收益率
        dates = self.results['dates'][1:]
        returns = self.results['returns'][1:]
        benchmark_returns = self.results['benchmark_returns'][1:]
        
        plt.plot(dates, returns, label='策略收益率')
        plt.plot(dates, benchmark_returns, label='基准收益率')
        plt.title('每日收益率对比')
        plt.ylabel('收益率')
        plt.grid(True)
        plt.legend()
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"收益率曲线图已保存至 {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, save_path=None):
        """
        绘制回撤曲线
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 计算回撤
        dates = self.results['dates']
        portfolio_values = pd.Series(self.results['portfolio_value'], index=dates)
        benchmark_values = pd.Series(self.results['benchmark_value'], index=dates)
        
        portfolio_running_max = portfolio_values.cummax()
        portfolio_drawdown = (portfolio_values / portfolio_running_max) - 1
        
        benchmark_running_max = benchmark_values.cummax()
        benchmark_drawdown = (benchmark_values / benchmark_running_max) - 1
        
        # 绘制回撤
        plt.fill_between(dates, 0, portfolio_drawdown, color='red', alpha=0.3, label='策略回撤')
        plt.fill_between(dates, 0, benchmark_drawdown, color='blue', alpha=0.3, label='基准回撤')
        plt.title('回撤分析')
        plt.ylabel('回撤')
        plt.grid(True)
        plt.legend()
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 设置y轴百分比格式
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"回撤曲线图已保存至 {save_path}")
        
        plt.show()
    
    def plot_monthly_returns(self, save_path=None):
        """
        绘制月度收益热力图
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建日期索引的收益率序列
        dates = self.results['dates'][1:]
        returns = self.results['returns'][1:]
        returns_series = pd.Series(returns, index=dates)
        
        # 计算月度收益率
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建月度收益率透视表
        monthly_returns_pivot = monthly_returns.unstack(level=0)
        monthly_returns_pivot = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })
        monthly_returns_pivot = monthly_returns_pivot.pivot('year', 'month', 'returns')
        
        # 设置月份列名
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot.columns = [month_names[i-1] for i in monthly_returns_pivot.columns]
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制热力图
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, linewidths=1)
        plt.title('月度收益率热力图')
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"月度收益热力图已保存至 {save_path}")
        
        plt.show()
    
    def plot_rolling_statistics(self, window=60, save_path=None):
        """
        绘制滚动统计指标
        
        参数:
            window: 滚动窗口大小
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建日期索引的收益率序列
        dates = self.results['dates'][1:]
        returns = self.results['returns'][1:]
        benchmark_returns = self.results['benchmark_returns'][1:]
        
        returns_series = pd.Series(returns, index=dates)
        benchmark_returns_series = pd.Series(benchmark_returns, index=dates)
        
        # 计算滚动统计指标
        rolling_sharpe = returns_series.rolling(window).apply(
            lambda x: (x.mean() - 0.03/252) / (x.std() * np.sqrt(252)) if x.std() != 0 else 0
        )
        
        rolling_volatility = returns_series.rolling(window).std() * np.sqrt(252)
        
        rolling_beta = returns_series.rolling(window).apply(
            lambda x: np.cov(x, benchmark_returns_series.loc[x.index])[0, 1] / np.var(benchmark_returns_series.loc[x.index]) 
            if np.var(benchmark_returns_series.loc[x.index]) != 0 else 0
        )
        
        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 绘制滚动夏普比率
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax1.set_title(f'滚动夏普比率 (窗口={window}天)')
        ax1.set_ylabel('夏普比率')
        ax1.grid(True)
        
        # 绘制滚动波动率
        ax2.plot(rolling_volatility.index, rolling_volatility.values)
        ax2.set_title(f'滚动波动率 (窗口={window}天)')
        ax2.set_ylabel('年化波动率')
        ax2.grid(True)
        
        # 绘制滚动贝塔
        ax3.plot(rolling_beta.index, rolling_beta.values)
        ax3.set_title(f'滚动贝塔系数 (窗口={window}天)')
        ax3.set_ylabel('贝塔')
        ax3.grid(True)
        
        # 设置x轴日期格式
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"滚动统计指标图已保存至 {save_path}")
        
        plt.show()
    
    def plot_distribution(self, save_path=None):
        """
        绘制收益率分布
        
        参数:
            save_path: 保存路径
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 获取收益率
        returns = self.results['returns'][1:]
        benchmark_returns = self.results['benchmark_returns'][1:]
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制收益率分布
        sns.histplot(returns, kde=True, stat='density', alpha=0.5, label='策略收益率')
        sns.histplot(benchmark_returns, kde=True, stat='density', alpha=0.5, label='基准收益率')
        
        # 添加正态分布曲线
        x = np.linspace(min(min(returns), min(benchmark_returns)), max(max(returns), max(benchmark_returns)), 100)
        plt.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), 'r-', lw=2, label='策略正态分布拟合')
        plt.plot(x, stats.norm.pdf(x, np.mean(benchmark_returns), np.std(benchmark_returns)), 'b-', lw=2, label='基准正态分布拟合')
        
        plt.title('收益率分布')
        plt.xlabel('收益率')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"收益率分布图已保存至 {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir='results'):
        """
        生成回测报告
        
        参数:
            output_dir: 输出目录
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成摘要
        summary = self.get_summary()
        summary.to_csv(f"{output_dir}/summary_{timestamp}.csv")
        
        # 生成图表
        self.plot_equity_curve(save_path=f"{output_dir}/equity_curve_{timestamp}.png")
        self.plot_returns(save_path=f"{output_dir}/returns_{timestamp}.png")
        self.plot_drawdown(save_path=f"{output_dir}/drawdown_{timestamp}.png")
        self.plot_monthly_returns(save_path=f"{output_dir}/monthly_returns_{timestamp}.png")
        self.plot_rolling_statistics(save_path=f"{output_dir}/rolling_stats_{timestamp}.png")
        
        try:
            self.plot_distribution(save_path=f"{output_dir}/distribution_{timestamp}.png")
        except:
            print("绘制分布图失败，可能缺少scipy库")
        
        # 生成HTML报告
        self._generate_html_report(output_dir, timestamp)
        
        print(f"回测报告已生成至 {output_dir} 目录")
    
    def _generate_html_report(self, output_dir, timestamp):
        """
        生成HTML报告
        
        参数:
            output_dir: 输出目录
            timestamp: 时间戳
        """
        # 获取摘要数据
        summary = self.get_summary()
        
        # 构建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>量化策略回测报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .summary-table th {{ background-color: #f2f2f2; }}
                .summary-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ margin-bottom: 30px; }}
                .chart {{ width: 100%; max-width: 1000px; margin-bottom: 10px; }}
                .footer {{ margin-top: 30px; text-align: center; font-size: 12px; color: #777; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>量化策略回测报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>回测摘要</h2>
                <table class="summary-table">
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                    </tr>
        """
        
        # 添加摘要表格行
        for index, row in summary.iterrows():
            metric_name = index
            if metric_name == 'annual_return':
                metric_name = '年化收益率'
            elif metric_name == 'annual_benchmark_return':
                metric_name = '基准年化收益率'
            elif metric_name == 'volatility':
                metric_name = '波动率'
            elif metric_name == 'benchmark_volatility':
                metric_name = '基准波动率'
            elif metric_name == 'sharpe_ratio':
                metric_name = '夏普比率'
            elif metric_name == 'benchmark_sharpe_ratio':
                metric_name = '基准夏普比率'
            elif metric_name == 'max_drawdown':
                metric_name = '最大回撤'
            elif metric_name == 'information_ratio':
                metric_name = '信息比率'
            elif metric_name == 'win_rate':
                metric_name = '胜率'
            elif metric_name == 'sortino_ratio':
                metric_name = '索提诺比率'
            elif metric_name == 'beta':
                metric_name = '贝塔系数'
            elif metric_name == 'alpha':
                metric_name = '阿尔法'
            
            html_content += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{row['value']}</td>
                    </tr>
            """
        
        # 添加图表
        html_content += f"""
                </table>
                
                <h2>回测图表</h2>
                
                <div class="chart-container">
                    <h3>权益曲线</h3>
                    <img class="chart" src="equity_curve_{timestamp}.png" alt="权益曲线">
                </div>
                
                <div class="chart-container">
                    <h3>收益率曲线</h3>
                    <img class="chart" src="returns_{timestamp}.png" alt="收益率曲线">
                </div>
                
                <div class="chart-container">
                    <h3>回撤分析</h3>
                    <img class="chart" src="drawdown_{timestamp}.png" alt="回撤分析">
                </div>
                
                <div class="chart-container">
                    <h3>月度收益热力图</h3>
                    <img class="chart" src="monthly_returns_{timestamp}.png" alt="月度收益热力图">
                </div>
                
                <div class="chart-container">
                    <h3>滚动统计指标</h3>
                    <img class="chart" src="rolling_stats_{timestamp}.png" alt="滚动统计指标">
                </div>
        """
        
        # 添加分布图（如果存在）
        if os.path.exists(f"{output_dir}/distribution_{timestamp}.png"):
            html_content += f"""
                <div class="chart-container">
                    <h3>收益率分布</h3>
                    <img class="chart" src="distribution_{timestamp}.png" alt="收益率分布">
                </div>
            """
        
        # 添加页脚和结束标签
        html_content += f"""
                <div class="footer">
                    <p>基于qlib的量化交易系统 - 回测报告</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 写入HTML文件
        with open(f"{output_dir}/report_{timestamp}.html", 'w', encoding='utf-8') as f:
            f.write(html_content)