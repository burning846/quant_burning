#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化示例

本示例展示如何使用可视化模块展示回测结果
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import Analyzer
from results.visualizer import Visualizer
from strategies.strategy_factory import StrategyFactory
from utils.config_loader import ConfigLoader

# 加载配置
config = ConfigLoader().load_config()

# 创建结果目录
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'output')
os.makedirs(results_dir, exist_ok=True)

# 设置股票和时间范围
symbols = config.get('default_symbols', ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'])
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

print(f"使用股票: {symbols}")
print(f"回测时间范围: {start_date} 至 {end_date}")

# 设置回测参数
backtest_config = {
    'start_date': start_date,
    'end_date': end_date,
    'benchmark': 'SPY',  # 使用SPY作为基准
    'initial_capital': 100000,
    'position_size': 0.1,  # 每个股票使用10%的资金
    'commission': 0.001,  # 0.1%手续费
}

# 创建回测引擎
backtest_engine = BacktestEngine(symbols=symbols, config=backtest_config)

# 创建策略
strategy_factory = StrategyFactory()
momentum_strategy = strategy_factory.create_strategy('momentum', lookback=20)
ma_cross_strategy = strategy_factory.create_strategy('ma_cross', short_window=20, long_window=50)
rsi_strategy = strategy_factory.create_strategy('rsi', lookback=14, overbought=70, oversold=30)

# 运行回测
print("\n运行均线交叉策略回测...")
results = backtest_engine.run_backtest(ma_cross_strategy)

# 保存回测结果
results_file = os.path.join(results_dir, 'ma_cross_results.pkl')
backtest_engine.save_results(results_file)
print(f"回测结果已保存至: {results_file}")

# 分析回测结果
analyzer = Analyzer(results)
summary = analyzer.get_summary()
print("\n回测结果摘要:")
print(summary)

# 可视化回测结果
visualizer = Visualizer(results)

# 绘制权益曲线
equity_curve_file = os.path.join(results_dir, 'equity_curve.png')
visualizer.plot_equity_curve(save_path=equity_curve_file, show=False)
print(f"权益曲线图已保存至: {equity_curve_file}")

# 绘制收益率分布
returns_dist_file = os.path.join(results_dir, 'returns_distribution.png')
visualizer.plot_returns_distribution(save_path=returns_dist_file, show=False)

# 绘制回撤图
drawdown_file = os.path.join(results_dir, 'drawdown.png')
visualizer.plot_drawdown(save_path=drawdown_file, show=False)

# 绘制月度收益热力图
monthly_returns_file = os.path.join(results_dir, 'monthly_returns.png')
visualizer.plot_monthly_returns(save_path=monthly_returns_file, show=False)

# 绘制滚动夏普比率
rolling_sharpe_file = os.path.join(results_dir, 'rolling_sharpe.png')
visualizer.plot_rolling_sharpe(window=60, save_path=rolling_sharpe_file, show=False)

print("\n所有可视化图表已生成完毕，请查看results/output目录")

# 运行多个策略并比较
print("\n运行多个策略并比较...")

# 运行动量策略
print("运行动量策略回测...")
momentum_results = backtest_engine.run_backtest(momentum_strategy)
momentum_analyzer = Analyzer(momentum_results)
momentum_summary = momentum_analyzer.get_summary()

# 运行RSI策略
print("运行RSI策略回测...")
rsi_results = backtest_engine.run_backtest(rsi_strategy)
rsi_analyzer = Analyzer(rsi_results)
rsi_summary = rsi_analyzer.get_summary()

# 比较策略性能
print("\n策略性能比较:")
print("\n1. 均线交叉策略:")
print(summary)
print("\n2. 动量策略:")
print(momentum_summary)
print("\n3. RSI策略:")
print(rsi_summary)

print("\n示例运行完成!")