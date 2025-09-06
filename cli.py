#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
命令行界面，用于方便用户使用量化交易系统
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from models.ml_model import MLModel
from models.dl_model import DLModel
from strategies.simple_strategies import (
    MomentumStrategy, 
    MeanReversionStrategy, 
    MovingAverageCrossStrategy, 
    RSIStrategy
)
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer
from results.visualizer import Visualizer


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def fetch_data(args, config):
    """获取股票数据"""
    print("开始获取股票数据...")
    
    # 获取参数
    symbols = args.symbols.split(',') if args.symbols else config['data']['default_symbols']
    start_date = args.start_date or config['data']['default_start_date']
    end_date = args.end_date or config['data']['default_end_date']
    source = args.source or config['data']['default_source']
    
    # 创建数据目录
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    # 获取数据
    fetcher = DataFetcher()
    data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source=source)
    
    # 保存数据
    output_path = os.path.join(config['data']['raw_dir'], f"stock_data_{datetime.now().strftime('%Y%m%d')}.csv")
    data.to_csv(output_path, index=False)
    
    print(f"数据获取完成，共 {len(data)} 条记录")
    print(f"数据已保存至: {output_path}")
    
    return data, output_path


def process_data(args, config, data=None):
    """处理股票数据"""
    print("开始处理股票数据...")
    
    # 如果没有提供数据，则从文件加载
    if data is None:
        input_path = args.input or os.path.join(
            config['data']['raw_dir'], 
            sorted([f for f in os.listdir(config['data']['raw_dir']) if f.startswith('stock_data_')])[-1]
        )
        print(f"从文件加载数据: {input_path}")
        data = pd.read_csv(input_path)
    
    # 处理数据
    processor = DataProcessor()
    processed_data = processor.process_stock_data(data)
    
    # 保存处理后的数据
    output_path = os.path.join(
        config['data']['processed_dir'], 
        f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    processed_data.to_csv(output_path, index=False)
    
    print(f"数据处理完成，共 {len(processed_data)} 条记录")
    print(f"处理后的数据已保存至: {output_path}")
    
    return processed_data, output_path


def train_model(args, config, data=None):
    """训练模型"""
    print("开始训练模型...")
    
    # 如果没有提供数据，则从文件加载
    if data is None:
        input_path = args.input or os.path.join(
            config['data']['processed_dir'], 
            sorted([f for f in os.listdir(config['data']['processed_dir']) if f.startswith('processed_data_')])[-1]
        )
        print(f"从文件加载数据: {input_path}")
        data = pd.read_csv(input_path)
    
    # 获取参数
    model_type = args.model_type or 'ml'
    model_name = args.model_name or (
        config['models']['default_ml_model'] if model_type == 'ml' else config['models']['default_dl_model']
    )
    
    # 创建模型目录
    os.makedirs(config['models']['model_dir'], exist_ok=True)
    
    # 准备特征和标签
    processor = DataProcessor()
    features, labels = processor.prepare_features_and_labels(data)
    
    # 训练模型
    if model_type == 'ml':
        model = MLModel(model_name=model_name)
    else:
        model = DLModel(model_name=model_name)
    
    model.train(features, labels)
    
    # 保存模型
    model_path = os.path.join(
        config['models']['model_dir'], 
        f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
    )
    model.save(model_path)
    
    print(f"模型训练完成")
    print(f"模型已保存至: {model_path}")
    
    return model, model_path


def run_backtest(args, config, data=None):
    """运行回测"""
    print("开始运行回测...")
    
    # 如果没有提供数据，则从文件加载
    if data is None:
        input_path = args.input or os.path.join(
            config['data']['processed_dir'], 
            sorted([f for f in os.listdir(config['data']['processed_dir']) if f.startswith('processed_data_')])[-1]
        )
        print(f"从文件加载数据: {input_path}")
        data = pd.read_csv(input_path)
    
    # 获取参数
    strategy_name = args.strategy or 'ma_cross'
    
    # 创建回测目录
    os.makedirs(config['backtest']['results_dir'], exist_ok=True)
    
    # 创建回测引擎
    backtest_config = config['backtest']['default_config']
    if args.start_date:
        backtest_config['start_date'] = args.start_date
    if args.end_date:
        backtest_config['end_date'] = args.end_date
    
    backtest_engine = BacktestEngine(backtest_config)
    
    # 创建策略
    if strategy_name == 'momentum':
        strategy_config = config['strategies']['momentum']
        strategy = MomentumStrategy(
            lookback_period=strategy_config['lookback_period'],
            holding_period=strategy_config['holding_period']
        )
    elif strategy_name == 'mean_reversion':
        strategy_config = config['strategies']['mean_reversion']
        strategy = MeanReversionStrategy(
            lookback_period=strategy_config['lookback_period'],
            threshold=strategy_config['threshold']
        )
    elif strategy_name == 'ma_cross':
        strategy_config = config['strategies']['ma_cross']
        strategy = MovingAverageCrossStrategy(
            short_window=strategy_config['short_window'],
            long_window=strategy_config['long_window']
        )
    elif strategy_name == 'rsi':
        strategy_config = config['strategies']['rsi']
        strategy = RSIStrategy(
            rsi_period=strategy_config['rsi_period'],
            overbought=strategy_config['overbought'],
            oversold=strategy_config['oversold']
        )
    else:
        raise ValueError(f"不支持的策略: {strategy_name}")
    
    # 运行回测
    results = backtest_engine.run_backtest(strategy, data)
    
    # 保存回测结果
    result_path = os.path.join(
        config['backtest']['results_dir'], 
        f"{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d')}.json"
    )
    backtest_engine.save_results(result_path)
    
    print(f"回测完成")
    print(f"回测结果已保存至: {result_path}")
    
    # 分析回测结果
    analyzer = BacktestAnalyzer(results)
    summary = analyzer.get_summary()
    
    print("\n回测摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 创建可视化目录
    os.makedirs(config['visualization']['plots_dir'], exist_ok=True)
    
    # 绘制回测结果
    if args.plot:
        print("\n生成回测图表...")
        visualizer = Visualizer()
        visualizer.load_backtest_results(results)
        
        # 绘制权益曲线
        equity_curve_path = os.path.join(
            config['visualization']['plots_dir'], 
            f"{strategy_name}_equity_curve_{datetime.now().strftime('%Y%m%d')}.png"
        )
        visualizer.plot_equity_curve(save_path=equity_curve_path)
        print(f"权益曲线已保存至: {equity_curve_path}")
        
        # 绘制回撤曲线
        drawdown_path = os.path.join(
            config['visualization']['plots_dir'], 
            f"{strategy_name}_drawdown_{datetime.now().strftime('%Y%m%d')}.png"
        )
        visualizer.plot_drawdown(save_path=drawdown_path)
        print(f"回撤曲线已保存至: {drawdown_path}")
        
        # 绘制月度收益热力图
        monthly_returns_path = os.path.join(
            config['visualization']['plots_dir'], 
            f"{strategy_name}_monthly_returns_{datetime.now().strftime('%Y%m%d')}.png"
        )
        visualizer.plot_monthly_returns(save_path=monthly_returns_path)
        print(f"月度收益热力图已保存至: {monthly_returns_path}")
    
    return results, result_path


def compare_strategies(args, config, data=None):
    """比较不同策略的性能"""
    print("开始比较不同策略的性能...")
    
    # 如果没有提供数据，则从文件加载
    if data is None:
        input_path = args.input or os.path.join(
            config['data']['processed_dir'], 
            sorted([f for f in os.listdir(config['data']['processed_dir']) if f.startswith('processed_data_')])[-1]
        )
        print(f"从文件加载数据: {input_path}")
        data = pd.read_csv(input_path)
    
    # 创建回测目录和可视化目录
    os.makedirs(config['backtest']['results_dir'], exist_ok=True)
    os.makedirs(config['visualization']['plots_dir'], exist_ok=True)
    
    # 创建回测引擎
    backtest_config = config['backtest']['default_config']
    if args.start_date:
        backtest_config['start_date'] = args.start_date
    if args.end_date:
        backtest_config['end_date'] = args.end_date
    
    # 创建策略列表
    strategies = [
        MomentumStrategy(
            lookback_period=config['strategies']['momentum']['lookback_period'],
            holding_period=config['strategies']['momentum']['holding_period']
        ),
        MeanReversionStrategy(
            lookback_period=config['strategies']['mean_reversion']['lookback_period'],
            threshold=config['strategies']['mean_reversion']['threshold']
        ),
        MovingAverageCrossStrategy(
            short_window=config['strategies']['ma_cross']['short_window'],
            long_window=config['strategies']['ma_cross']['long_window']
        ),
        RSIStrategy(
            rsi_period=config['strategies']['rsi']['rsi_period'],
            overbought=config['strategies']['rsi']['overbought'],
            oversold=config['strategies']['rsi']['oversold']
        )
    ]
    
    # 存储结果
    strategy_results = {}
    strategy_metrics = {}
    
    # 运行每个策略的回测
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        print(f"运行 {strategy_name} 回测...")
        
        backtest_engine = BacktestEngine(backtest_config)
        results = backtest_engine.run_backtest(strategy, data)
        
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
    metrics_df = pd.DataFrame(strategy_metrics).T
    print("\n策略性能比较:")
    print(metrics_df)
    
    # 保存比较结果
    comparison_path = os.path.join(
        config['backtest']['results_dir'], 
        f"strategy_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    metrics_df.to_csv(comparison_path)
    print(f"策略比较结果已保存至: {comparison_path}")
    
    # 绘制策略对比图
    if args.plot:
        print("\n生成策略对比图...")
        visualizer = Visualizer()
        
        # 绘制权益曲线对比
        equity_comparison_path = os.path.join(
            config['visualization']['plots_dir'], 
            f"equity_comparison_{datetime.now().strftime('%Y%m%d')}.png"
        )
        visualizer.plot_equity_comparison(strategy_results, save_path=equity_comparison_path)
        print(f"权益曲线对比图已保存至: {equity_comparison_path}")
        
        # 绘制指标对比图
        metrics_comparison_path = os.path.join(
            config['visualization']['plots_dir'], 
            f"metrics_comparison_{datetime.now().strftime('%Y%m%d')}.png"
        )
        visualizer.plot_metrics_comparison(
            strategy_metrics, 
            metrics=['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
            save_path=metrics_comparison_path
        )
        print(f"指标对比图已保存至: {metrics_comparison_path}")
    
    return strategy_results, strategy_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="量化交易系统命令行工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 获取数据命令
    fetch_parser = subparsers.add_parser("fetch", help="获取股票数据")
    fetch_parser.add_argument("--symbols", type=str, help="股票代码列表，以逗号分隔")
    fetch_parser.add_argument("--start-date", type=str, help="开始日期")
    fetch_parser.add_argument("--end-date", type=str, help="结束日期")
    fetch_parser.add_argument("--source", type=str, help="数据源")
    
    # 处理数据命令
    process_parser = subparsers.add_parser("process", help="处理股票数据")
    process_parser.add_argument("--input", type=str, help="输入数据文件路径")
    
    # 训练模型命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--input", type=str, help="输入数据文件路径")
    train_parser.add_argument("--model-type", type=str, choices=["ml", "dl"], help="模型类型")
    train_parser.add_argument("--model-name", type=str, help="模型名称")
    
    # 回测命令
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    backtest_parser.add_argument("--input", type=str, help="输入数据文件路径")
    backtest_parser.add_argument("--strategy", type=str, help="策略名称")
    backtest_parser.add_argument("--start-date", type=str, help="回测开始日期")
    backtest_parser.add_argument("--end-date", type=str, help="回测结束日期")
    backtest_parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    # 比较策略命令
    compare_parser = subparsers.add_parser("compare", help="比较不同策略的性能")
    compare_parser.add_argument("--input", type=str, help="输入数据文件路径")
    compare_parser.add_argument("--start-date", type=str, help="回测开始日期")
    compare_parser.add_argument("--end-date", type=str, help="回测结束日期")
    compare_parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    # 全流程命令
    pipeline_parser = subparsers.add_parser("pipeline", help="运行完整流程")
    pipeline_parser.add_argument("--symbols", type=str, help="股票代码列表，以逗号分隔")
    pipeline_parser.add_argument("--start-date", type=str, help="开始日期")
    pipeline_parser.add_argument("--end-date", type=str, help="结束日期")
    pipeline_parser.add_argument("--source", type=str, help="数据源")
    pipeline_parser.add_argument("--strategy", type=str, help="策略名称")
    pipeline_parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    # 解析参数
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 执行命令
    if args.command == "fetch":
        fetch_data(args, config)
    elif args.command == "process":
        process_data(args, config)
    elif args.command == "train":
        train_model(args, config)
    elif args.command == "backtest":
        run_backtest(args, config)
    elif args.command == "compare":
        compare_strategies(args, config)
    elif args.command == "pipeline":
        # 运行完整流程
        data, _ = fetch_data(args, config)
        processed_data, _ = process_data(args, config, data)
        run_backtest(args, config, processed_data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()