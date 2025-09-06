# 可视化模块

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime

class Visualizer:
    """
    可视化类，用于绘制回测结果和策略性能图表
    """
    
    def __init__(self, results=None):
        """
        初始化可视化器
        
        参数:
            results: 回测结果字典
        """
        self.results = results
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('Set2')
    
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
    
    def plot_equity_curve(self, save_path=None, show=True):
        """
        绘制权益曲线
        
        参数:
            save_path: 保存路径
            show: 是否显示图表
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
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_returns(self, save_path=None, show=True):
        """
        绘制收益率曲线
        
        参数:
            save_path: 保存路径
            show: 是否显示图表
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
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_drawdown(self, save_path=None, show=True):
        """
        绘制回撤曲线
        
        参数:
            save_path: 保存路径
            show: 是否显示图表
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
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_monthly_returns(self, save_path=None, show=True):
        """
        绘制月度收益热力图
        
        参数:
            save_path: 保存路径
            show: 是否显示图表
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
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_rolling_statistics(self, window=60, save_path=None, show=True):
        """
        绘制滚动统计指标
        
        参数:
            window: 滚动窗口大小
            save_path: 保存路径
            show: 是否显示图表
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
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_performance_comparison(self, results_dict, metric='portfolio_value', title=None, save_path=None, show=True):
        """
        绘制多个策略的性能对比图
        
        参数:
            results_dict: 包含多个策略回测结果的字典，格式为 {策略名称: 回测结果}
            metric: 要比较的指标，默认为'portfolio_value'
            title: 图表标题
            save_path: 保存路径
            show: 是否显示图表
        """
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制每个策略的指标
        for strategy_name, results in results_dict.items():
            if metric in results:
                plt.plot(results['dates'], results[metric], label=strategy_name)
        
        # 设置图表标题和标签
        if title is None:
            title = f'策略{metric}对比'
        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"性能对比图已保存至 {save_path}")
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, metrics=None, save_path=None, show=True):
        """
        绘制多个策略的指标对比图
        
        参数:
            metrics_dict: 包含多个策略指标的字典，格式为 {策略名称: {指标名称: 指标值}}
            metrics: 要比较的指标列表，默认为None（比较所有指标）
            save_path: 保存路径
            show: 是否显示图表
        """
        # 如果未指定指标，则使用第一个策略的所有指标
        if metrics is None and metrics_dict:
            first_strategy = list(metrics_dict.keys())[0]
            metrics = list(metrics_dict[first_strategy].keys())
        
        # 创建指标对比DataFrame
        metrics_df = pd.DataFrame(metrics_dict).T
        
        # 如果指定了指标，则只选择这些指标
        if metrics is not None:
            metrics_df = metrics_df[metrics]
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制条形图
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('策略指标对比')
        plt.ylabel('指标值')
        plt.grid(True, axis='y')
        plt.legend(title='指标')
        
        # 调整x轴标签
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for i, strategy in enumerate(metrics_df.index):
            for j, metric in enumerate(metrics_df.columns):
                value = metrics_df.loc[strategy, metric]
                plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"指标对比图已保存至 {save_path}")
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_matrix(self, returns_dict, save_path=None, show=True):
        """
        绘制策略收益率相关性矩阵
        
        参数:
            returns_dict: 包含多个策略收益率的字典，格式为 {策略名称: 收益率序列}
            save_path: 保存路径
            show: 是否显示图表
        """
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # 计算相关性矩阵
        corr_matrix = returns_df.corr()
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
        plt.title('策略收益率相关性矩阵')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save_path is not None:
            plt.savefig(save_path)
            print(f"相关性矩阵图已保存至 {save_path}")
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_performance_dashboard(self, output_dir='results/dashboard'):
        """
        生成性能仪表盘
        
        参数:
            output_dir: 输出目录
        """
        if self.results is None:
            raise ValueError("尚未加载回测结果")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成各种图表
        self.plot_equity_curve(save_path=f"{output_dir}/equity_curve_{timestamp}.png", show=False)
        self.plot_returns(save_path=f"{output_dir}/returns_{timestamp}.png", show=False)
        self.plot_drawdown(save_path=f"{output_dir}/drawdown_{timestamp}.png", show=False)
        self.plot_monthly_returns(save_path=f"{output_dir}/monthly_returns_{timestamp}.png", show=False)
        self.plot_rolling_statistics(save_path=f"{output_dir}/rolling_stats_{timestamp}.png", show=False)
        
        # 提取关键指标
        metrics = {
            'annual_return': self.results.get('annual_return', 0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0),
            'max_drawdown': self.results.get('max_drawdown', 0),
            'win_rate': self.results.get('win_rate', 0),
            'volatility': self.results.get('volatility', 0),
            'beta': self.results.get('beta', 0),
            'alpha': self.results.get('alpha', 0)
        }
        
        # 生成指标摘要CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_dir}/metrics_summary_{timestamp}.csv", index=False)
        
        print(f"性能仪表盘已生成至 {output_dir} 目录")
        
        return f"{output_dir}/equity_curve_{timestamp}.png"