#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多因子策略示例脚本

本脚本展示如何构建和回测多因子选股策略
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer
from results.visualizer import Visualizer
from strategies.base_strategy import BaseStrategy


class MultiFactor(BaseStrategy):
    """多因子选股策略
    
    基于多个因子对股票进行评分和排序，选择得分最高的股票进行投资
    
    参数:
        factors (list): 因子列表，例如 ['pe', 'pb', 'momentum_20', 'volatility_20']
        weights (list): 因子权重列表，必须与factors长度相同
        top_n (int): 选择得分最高的前N只股票
        rebalance_days (int): 调仓周期（天数）
    """
    
    def __init__(self, factors, weights=None, top_n=5, rebalance_days=20):
        super().__init__()
        self.factors = factors
        self.weights = weights if weights else [1.0/len(factors)] * len(factors)
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.last_rebalance_date = None
        self.day_count = 0
        
        # 验证参数
        if len(self.factors) != len(self.weights):
            raise ValueError("因子列表和权重列表长度必须相同")
        if sum(self.weights) != 1.0:
            self.weights = [w/sum(self.weights) for w in self.weights]
            print(f"权重已归一化: {self.weights}")
    
    def generate_signals(self, data):
        """生成交易信号
        
        参数:
            data (DataFrame): 包含所有股票数据的DataFrame
            
        返回:
            DataFrame: 包含交易信号的DataFrame
        """
        # 获取当前日期
        current_date = data['date'].iloc[0]
        
        # 检查是否需要调仓
        if self.last_rebalance_date is None or self.day_count >= self.rebalance_days:
            self.day_count = 0
            self.last_rebalance_date = current_date
            
            # 获取当前日期的所有股票数据
            current_data = data[data['date'] == current_date].copy()
            
            # 计算每个因子的得分
            for factor in self.factors:
                if factor not in current_data.columns:
                    raise ValueError(f"数据中不存在因子: {factor}")
                
                # 对于不同类型的因子，可能需要不同的排序方向
                # 例如，对于PE、PB等，较小值更好；对于动量等，较大值更好
                reverse = False
                if factor in ['pe', 'pb', 'volatility_20']:
                    reverse = True  # 较小值更好
                
                # 处理缺失值
                current_data[f'{factor}_score'] = current_data[factor].fillna(current_data[factor].median())
                
                # 对因子进行排序打分
                current_data[f'{factor}_score'] = current_data[f'{factor}_score'].rank(ascending=reverse)
            
            # 计算综合得分
            current_data['total_score'] = 0
            for i, factor in enumerate(self.factors):
                current_data['total_score'] += current_data[f'{factor}_score'] * self.weights[i]
            
            # 选择得分最高的top_n只股票
            selected_stocks = current_data.nlargest(self.top_n, 'total_score')['stock_id'].tolist()
            
            # 生成信号
            signals = pd.DataFrame()
            signals['stock_id'] = current_data['stock_id']
            signals['date'] = current_date
            signals['signal'] = 0
            
            # 对选中的股票生成买入信号
            signals.loc[signals['stock_id'].isin(selected_stocks), 'signal'] = 1
            
            return signals
        else:
            # 不需要调仓，维持当前持仓
            self.day_count += 1
            signals = pd.DataFrame()
            signals['stock_id'] = data[data['date'] == current_date]['stock_id']
            signals['date'] = current_date
            signals['signal'] = 0
            
            # 对当前持仓的股票保持买入信号
            for stock_id in self.positions.keys():
                if stock_id in signals['stock_id'].values:
                    signals.loc[signals['stock_id'] == stock_id, 'signal'] = 1
            
            return signals
    
    def generate_trade_decision(self, signals, data):
        """根据信号生成交易决策
        
        参数:
            signals (DataFrame): 交易信号
            data (DataFrame): 市场数据
            
        返回:
            dict: 交易决策，格式为 {stock_id: {'action': 'buy'/'sell'/'hold', 'amount': amount}}
        """
        current_date = signals['date'].iloc[0]
        current_data = data[data['date'] == current_date]
        
        # 初始化交易决策
        decisions = {}
        
        # 处理每只股票
        for _, row in signals.iterrows():
            stock_id = row['stock_id']
            signal = row['signal']
            
            # 获取当前股票价格
            stock_price = current_data[current_data['stock_id'] == stock_id]['close'].values[0]
            
            # 当前是否持有该股票
            currently_holding = stock_id in self.positions and self.positions[stock_id] > 0
            
            if signal == 1 and not currently_holding:
                # 买入信号且当前未持有
                # 计算买入数量（假设每只股票分配相等资金）
                allocation = self.cash / self.top_n
                amount = int(allocation / stock_price / 100) * 100  # 买入整数手（假设每手100股）
                
                if amount > 0:
                    decisions[stock_id] = {'action': 'buy', 'amount': amount}
            
            elif signal == 0 and currently_holding:
                # 卖出信号且当前持有
                decisions[stock_id] = {'action': 'sell', 'amount': self.positions[stock_id]}
        
        return decisions


def load_config(config_path="../config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 创建数据目录
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['backtest']['results_dir'], exist_ok=True)
    os.makedirs(config['visualization']['plots_dir'], exist_ok=True)
    
    # 设置股票和时间范围
    symbols = config['data']['default_symbols']
    start_date = config['data']['default_start_date']
    end_date = config['data']['default_end_date']
    
    print(f"获取股票数据: {symbols}")
    print(f"时间范围: {start_date} 至 {end_date}")
    
    # 获取数据
    try:
        # 尝试加载本地数据
        data_path = os.path.join(config['data']['processed_dir'], 'processed_data.csv')
        data = pd.read_csv(data_path)
        print(f"已加载本地数据: {data_path}")
    except FileNotFoundError:
        # 如果本地数据不存在，则从网络获取并处理
        print("本地数据不存在，从网络获取数据...")
        fetcher = DataFetcher()
        data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source='yahoo')
        
        # 处理数据
        processor = DataProcessor()
        data = processor.process_stock_data(data)
        
        # 保存处理后的数据
        data.to_csv(data_path, index=False)
        print(f"数据已保存至: {data_path}")
    
    print(f"数据集大小: {len(data)} 条记录")
    
    # 创建多因子策略
    # 定义因子和权重
    factors = ['pe', 'pb', 'momentum_20', 'volatility_20', 'rsi_14']
    weights = [0.2, 0.2, 0.3, 0.15, 0.15]  # 权重之和为1
    
    # 创建策略实例
    multi_factor_strategy = MultiFactor(
        factors=factors,
        weights=weights,
        top_n=3,  # 选择得分最高的前3只股票
        rebalance_days=20  # 20天调仓一次
    )
    
    # 创建回测引擎
    backtest_config = {
        'start_date': '2019-01-01',  # 回测开始日期
        'end_date': '2021-12-31',  # 回测结束日期
        'benchmark': '000001.SZ',  # 基准指数
        'account': 1000000,  # 初始资金
        'commission_rate': 0.0003,  # 手续费率
        'slippage_rate': 0.0001,  # 滑点率
        'trade_frequency': 'day',  # 交易频率
        'verbose': True  # 是否打印详细信息
    }
    
    backtest_engine = BacktestEngine(backtest_config)
    
    # 运行回测
    print("\n开始回测多因子策略...")
    results = backtest_engine.run_backtest(multi_factor_strategy, data)
    
    # 保存回测结果
    result_path = os.path.join(
        config['backtest']['results_dir'], 
        f"multi_factor_backtest_{datetime.now().strftime('%Y%m%d')}.json"
    )
    backtest_engine.save_results(result_path)
    print(f"回测结果已保存至: {result_path}")
    
    # 分析回测结果
    analyzer = BacktestAnalyzer(results)
    summary = analyzer.get_summary()
    
    print("\n回测摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 绘制回测结果
    print("\n生成回测图表...")
    visualizer = Visualizer()
    visualizer.load_backtest_results(results)
    
    # 绘制权益曲线
    equity_curve_path = os.path.join(
        config['visualization']['plots_dir'], 
        f"multi_factor_equity_curve_{datetime.now().strftime('%Y%m%d')}.png"
    )
    visualizer.plot_equity_curve(save_path=equity_curve_path)
    print(f"权益曲线已保存至: {equity_curve_path}")
    
    # 绘制回撤曲线
    drawdown_path = os.path.join(
        config['visualization']['plots_dir'], 
        f"multi_factor_drawdown_{datetime.now().strftime('%Y%m%d')}.png"
    )
    visualizer.plot_drawdown(save_path=drawdown_path)
    print(f"回撤曲线已保存至: {drawdown_path}")
    
    # 绘制月度收益热力图
    monthly_returns_path = os.path.join(
        config['visualization']['plots_dir'], 
        f"multi_factor_monthly_returns_{datetime.now().strftime('%Y%m%d')}.png"
    )
    visualizer.plot_monthly_returns(save_path=monthly_returns_path)
    print(f"月度收益热力图已保存至: {monthly_returns_path}")
    
    # 因子权重敏感性分析
    print("\n进行因子权重敏感性分析...")
    
    # 定义不同的权重组合
    weight_scenarios = [
        [0.4, 0.2, 0.2, 0.1, 0.1],  # 更重视PE
        [0.2, 0.4, 0.2, 0.1, 0.1],  # 更重视PB
        [0.1, 0.1, 0.6, 0.1, 0.1],  # 更重视动量
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 更重视波动率
        [0.1, 0.1, 0.1, 0.1, 0.6]   # 更重视RSI
    ]
    
    scenario_results = {}
    scenario_metrics = {}
    
    for i, weights in enumerate(weight_scenarios):
        scenario_name = f"权重方案{i+1}"
        print(f"测试 {scenario_name}: {weights}")
        
        # 创建策略
        strategy = MultiFactor(
            factors=factors,
            weights=weights,
            top_n=3,
            rebalance_days=20
        )
        
        # 运行回测
        backtest_engine = BacktestEngine(backtest_config)
        results = backtest_engine.run_backtest(strategy, data)
        
        # 保存结果
        scenario_results[scenario_name] = results
        
        # 获取关键指标
        metrics = {
            'annual_return': results['annual_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate']
        }
        scenario_metrics[scenario_name] = metrics
    
    # 比较不同权重方案的性能
    metrics_df = pd.DataFrame(scenario_metrics).T
    print("\n不同权重方案性能比较:")
    print(metrics_df)
    
    # 保存比较结果
    comparison_path = os.path.join(
        config['backtest']['results_dir'], 
        f"weight_scenario_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    metrics_df.to_csv(comparison_path)
    print(f"权重方案比较结果已保存至: {comparison_path}")
    
    # 绘制权重方案对比图
    plt.figure(figsize=(12, 8))
    
    # 绘制年化收益率对比
    plt.subplot(2, 2, 1)
    metrics_df['annual_return'].plot(kind='bar')
    plt.title('年化收益率对比')
    plt.ylabel('年化收益率')
    plt.grid(axis='y')
    
    # 绘制夏普比率对比
    plt.subplot(2, 2, 2)
    metrics_df['sharpe_ratio'].plot(kind='bar')
    plt.title('夏普比率对比')
    plt.ylabel('夏普比率')
    plt.grid(axis='y')
    
    # 绘制最大回撤对比
    plt.subplot(2, 2, 3)
    metrics_df['max_drawdown'].plot(kind='bar')
    plt.title('最大回撤对比')
    plt.ylabel('最大回撤')
    plt.grid(axis='y')
    
    # 绘制胜率对比
    plt.subplot(2, 2, 4)
    metrics_df['win_rate'].plot(kind='bar')
    plt.title('胜率对比')
    plt.ylabel('胜率')
    plt.grid(axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    comparison_plot_path = os.path.join(
        config['visualization']['plots_dir'], 
        f"weight_scenario_comparison_{datetime.now().strftime('%Y%m%d')}.png"
    )
    plt.savefig(comparison_plot_path)
    print(f"权重方案对比图已保存至: {comparison_plot_path}")
    
    print("\n多因子策略示例完成")


if __name__ == "__main__":
    main()