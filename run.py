# 主运行脚本

import os
import sys
import yaml
import pandas as pd
import numpy as np
import qlib
from qlib.config import REG_CN, REG_US
from datetime import datetime

# 导入项目模块
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from strategies.simple_strategies import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossStrategy, RSIStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer

def load_config(config_path='config/config.yaml'):
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def init_qlib(config):
    """
    初始化qlib环境
    
    参数:
        config: 配置字典
    """
    # 获取qlib配置
    qlib_config = config.get('qlib', {})
    
    # 设置默认配置
    provider_uri = qlib_config.get('provider_uri', 'data/qlib_data')
    region = qlib_config.get('region', REG_US)
    
    # 初始化qlib
    qlib.init(provider_uri=provider_uri, region=region)
    print("qlib环境初始化完成")

def prepare_data(config):
    """
    准备数据
    
    参数:
        config: 配置字典
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    # 获取数据配置
    data_config = config.get('data', {})
    symbols = data_config.get('default_symbols', ['000001.SZ', '600000.SH'])
    start_date = data_config.get('default_start_date', '2018-01-01')
    end_date = data_config.get('default_end_date', '2023-12-31')
    data_source = data_config.get('default_source', 'akshare')
    
    # 创建数据目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 获取数据
    print("\n1. 获取股票数据")
    data_path = 'data/raw/stock_data.csv'
    
    try:
        # 尝试使用本地数据
        data = pd.read_csv(data_path)
        print(f"  已加载本地数据，共 {len(data)} 条记录")
    except FileNotFoundError:
        # 如果本地数据不存在，则从网络获取
        print(f"  本地数据不存在，从{data_source}获取数据...")
        fetcher = DataFetcher()
        data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source=data_source)
        data.to_csv(data_path, index=False)
        print(f"  数据获取完成，共 {len(data)} 条记录")
    
    # 处理数据
    print("\n2. 处理数据")
    processor = DataProcessor()
    processed_data = processor.process_stock_data(data)
    processed_data.to_csv('data/processed/processed_data.csv', index=False)
    print(f"  数据处理完成，共 {len(processed_data)} 条记录")
    
    return processed_data

def run_backtest(data, config):
    """
    运行回测
    
    参数:
        data: 处理后的数据
        config: 配置字典
        
    返回:
        dict: 回测结果
    """
    # 获取回测配置
    backtest_config = config.get('backtest', {})
    strategy_name = backtest_config.get('strategy', 'MovingAverageCross')
    
    print("\n3. 创建回测引擎")
    backtest_engine = BacktestEngine(backtest_config)
    
    # 创建策略
    print("\n4. 创建策略")
    if strategy_name == 'Momentum':
        lookback_period = backtest_config.get('lookback_period', 20)
        holding_period = backtest_config.get('holding_period', 5)
        strategy = MomentumStrategy(lookback_period=lookback_period, holding_period=holding_period)
    elif strategy_name == 'MeanReversion':
        lookback_period = backtest_config.get('lookback_period', 10)
        threshold = backtest_config.get('threshold', 1.5)
        strategy = MeanReversionStrategy(lookback_period=lookback_period, threshold=threshold)
    elif strategy_name == 'MovingAverageCross':
        short_window = backtest_config.get('short_window', 5)
        long_window = backtest_config.get('long_window', 20)
        strategy = MovingAverageCrossStrategy(short_window=short_window, long_window=long_window)
    elif strategy_name == 'RSI':
        rsi_period = backtest_config.get('rsi_period', 14)
        overbought = backtest_config.get('overbought', 70)
        oversold = backtest_config.get('oversold', 30)
        strategy = RSIStrategy(rsi_period=rsi_period, overbought=overbought, oversold=oversold)
    else:
        raise ValueError(f"不支持的策略: {strategy_name}")
    
    print(f"  使用策略: {strategy.__class__.__name__}")
    
    # 运行回测
    print("\n5. 运行回测")
    results = backtest_engine.run_backtest(strategy, data)
    
    # 保存回测结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = f"results/backtest_results_{strategy_name}_{timestamp}.json"
    backtest_engine.save_results(result_path)
    
    # 分析回测结果
    print("\n6. 分析回测结果")
    analyzer = BacktestAnalyzer(results)
    
    # 打印摘要
    summary = analyzer.get_summary()
    print("\n回测摘要:")
    print(summary)
    
    # 生成报告
    print("\n7. 生成回测报告")
    report_dir = f"results/{strategy_name}_{timestamp}"
    analyzer.generate_report(output_dir=report_dir)
    
    return results

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    print("=== 基于qlib的量化交易系统 ===")
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化qlib
    init_qlib(config)
    
    # 准备数据
    data = prepare_data(config)
    
    # 运行回测
    results = run_backtest(data, config)
    
    print("\n=== 量化交易系统运行完成 ===")

if __name__ == "__main__":
    main()
