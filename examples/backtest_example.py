# 回测示例脚本

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from strategies.simple_strategies import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossStrategy, RSIStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer

def run_backtest_example():
    """
    运行回测示例
    """
    print("=== 开始回测示例 ===")
    
    # 创建数据目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 获取数据
    print("\n1. 获取股票数据")
    symbols = ['000001.SZ', '600000.SH', '600519.SH', '000858.SZ', '601318.SH', '000063.SZ', '002594.SZ', '600438.SH', '603501.SH', '300750.SZ']
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    try:
        # 尝试使用本地数据
        data = pd.read_csv('data/raw/stock_data.csv')
        print(f"  已加载本地数据，共 {len(data)} 条记录")
    except FileNotFoundError:
        # 如果本地数据不存在，则从网络获取
        print("  本地数据不存在，从网络获取数据...")
        fetcher = DataFetcher()
        data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source='akshare')
        data.to_csv('data/raw/stock_data.csv', index=False)
        print(f"  数据获取完成，共 {len(data)} 条记录")
    
    # 处理数据
    print("\n2. 处理数据")
    processor = DataProcessor()
    processed_data = processor.process_stock_data(data)
    processed_data.to_csv('data/processed/processed_data.csv', index=False)
    print(f"  数据处理完成，共 {len(processed_data)} 条记录")
    
    # 创建回测引擎
    print("\n3. 创建回测引擎")
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
    
    # 创建策略
    print("\n4. 创建策略")
    print("  4.1 动量策略")
    momentum_strategy = MomentumStrategy(lookback_period=20, holding_period=5)
    
    print("  4.2 均值回归策略")
    mean_reversion_strategy = MeanReversionStrategy(lookback_period=10, threshold=1.5)
    
    print("  4.3 均线交叉策略")
    ma_cross_strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
    
    print("  4.4 RSI策略")
    rsi_strategy = RSIStrategy(rsi_period=14, overbought=70, oversold=30)
    
    # 运行回测
    print("\n5. 运行回测")
    
    # 选择一个策略进行回测
    strategy = ma_cross_strategy  # 可以替换为其他策略
    print(f"  使用策略: {strategy.__class__.__name__}")
    
    results = backtest_engine.run_backtest(strategy, processed_data)
    
    # 保存回测结果
    backtest_engine.save_results('results/backtest_results.json')
    
    # 分析回测结果
    print("\n6. 分析回测结果")
    analyzer = BacktestAnalyzer(results)
    
    # 打印摘要
    summary = analyzer.get_summary()
    print("\n回测摘要:")
    print(summary)
    
    # 生成报告
    print("\n7. 生成回测报告")
    analyzer.generate_report(output_dir='results')
    
    print("\n=== 回测示例完成 ===")

def compare_strategies():
    """
    比较不同策略的性能
    """
    print("=== 开始策略比较 ===")
    
    # 加载处理后的数据
    try:
        processed_data = pd.read_csv('data/processed/processed_data.csv')
        print(f"已加载处理后的数据，共 {len(processed_data)} 条记录")
    except FileNotFoundError:
        print("处理后的数据不存在，请先运行回测示例")
        return
    
    # 创建回测引擎
    backtest_config = {
        'start_date': '2019-01-01',
        'end_date': '2021-12-31',
        'benchmark': '000001.SZ',
        'account': 1000000,
        'commission_rate': 0.0003,
        'slippage_rate': 0.0001,
        'trade_frequency': 'day',
        'verbose': False  # 关闭详细输出
    }
    
    # 创建策略列表
    strategies = [
        MomentumStrategy(lookback_period=20, holding_period=5),
        MeanReversionStrategy(lookback_period=10, threshold=1.5),
        MovingAverageCrossStrategy(short_window=5, long_window=20),
        RSIStrategy(rsi_period=14, overbought=70, oversold=30)
    ]
    
    # 存储结果
    strategy_results = {}
    strategy_metrics = {}
    
    # 运行每个策略的回测
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        print(f"\n运行 {strategy_name} 回测...")
        
        backtest_engine = BacktestEngine(backtest_config)
        results = backtest_engine.run_backtest(strategy, processed_data)
        
        # 保存结果
        strategy_results[strategy_name] = results
        
        # 获取关键指标
        metrics = {
            'annual_return': results['annual_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate']
        }
        strategy_metrics[strategy_name] = metrics
    
    # 比较策略性能
    print("\n策略性能比较:")
    metrics_df = pd.DataFrame(strategy_metrics).T
    print(metrics_df)
    
    # 绘制权益曲线对比
    plt.figure(figsize=(12, 6))
    
    # 添加基准曲线
    benchmark_values = strategy_results[strategies[0].__class__.__name__]['benchmark_value']
    dates = strategy_results[strategies[0].__class__.__name__]['dates']
    plt.plot(dates, benchmark_values, label='基准', linestyle='--')
    
    # 添加每个策略的权益曲线
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        portfolio_values = strategy_results[strategy_name]['portfolio_value']
        plt.plot(dates, portfolio_values, label=strategy_name)
    
    plt.title('策略权益曲线对比')
    plt.xlabel('日期')
    plt.ylabel('价值')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/strategy_comparison.png')
    plt.show()
    
    print("\n=== 策略比较完成 ===")

if __name__ == "__main__":
    # 运行回测示例
    run_backtest_example()
    
    # 比较不同策略
    # compare_strategies()