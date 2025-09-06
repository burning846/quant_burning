#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qlib集成示例脚本

本脚本展示如何将我们的量化系统与Qlib框架集成使用
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

# 尝试导入qlib
try:
    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.estimator.handler import Alpha158
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    print("警告: Qlib未安装或导入失败，请先安装qlib: pip install pyqlib")
    QLIB_AVAILABLE = False


def load_config(config_path="../config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_qlib_data(data, processor, output_dir):
    """准备Qlib格式的数据
    
    参数:
        data (DataFrame): 原始股票数据
        processor (DataProcessor): 数据处理器实例
        output_dir (str): 输出目录
        
    返回:
        str: 数据目录路径
    """
    print("准备Qlib格式数据...")
    
    # 创建qlib数据目录
    qlib_data_dir = os.path.join(output_dir, 'qlib_data')
    os.makedirs(qlib_data_dir, exist_ok=True)
    
    # 使用处理器生成qlib格式数据
    processor.prepare_qlib_dataset(data, qlib_data_dir)
    
    return qlib_data_dir


def run_qlib_workflow(qlib_data_dir, config):
    """运行Qlib工作流
    
    参数:
        qlib_data_dir (str): Qlib数据目录
        config (dict): 配置信息
        
    返回:
        dict: 实验结果
    """
    if not QLIB_AVAILABLE:
        print("错误: Qlib未安装，无法运行Qlib工作流")
        return None
    
    print("初始化Qlib环境...")
    qlib.init(provider_uri=qlib_data_dir, region=REG_CN)
    
    # 设置Qlib实验参数
    market = "csi300"  # 使用沪深300成分股
    benchmark = "000001.SZ"  # 基准指数
    
    # 设置时间范围
    train_start_date = config['backtest']['default_config']['start_date']
    train_end_date = "2020-12-31"  # 训练集结束日期
    test_start_date = "2021-01-01"  # 测试集开始日期
    test_end_date = config['backtest']['default_config']['end_date']
    
    # 设置任务信息
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": train_start_date,
                        "end_time": test_end_date,
                        "fit_start_time": train_start_date,
                        "fit_end_time": train_end_date,
                        "instruments": market,
                    },
                },
                "segments": {
                    "train": (train_start_date, train_end_date),
                    "valid": (test_start_date, test_end_date),
                    "test": (test_start_date, test_end_date),
                },
            },
        },
    }
    
    # 创建并训练模型
    print("创建并训练Qlib模型...")
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    
    # 训练模型
    train_df = dataset.prepare("train", col_set=["feature", "label"])
    x_train, y_train = train_df["feature"], train_df["label"]
    
    # 验证集
    valid_df = dataset.prepare("valid", col_set=["feature", "label"])
    x_valid, y_valid = valid_df["feature"], valid_df["label"]
    
    # 拟合模型
    print(f"训练数据形状: {x_train.shape}, 验证数据形状: {x_valid.shape}")
    model.fit(x_train, y_train, x_valid, y_valid)
    
    # 预测
    print("使用模型进行预测...")
    recorder = R.get_recorder()
    
    # 预测训练集
    pred_train = model.predict(x_train)
    train_score = model.score(x_train, y_train)
    print(f"训练集得分: {train_score}")
    
    # 预测测试集
    pred_valid = model.predict(x_valid)
    valid_score = model.score(x_valid, y_valid)
    print(f"测试集得分: {valid_score}")
    
    # 记录信号
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
    
    # 分析投资组合
    par = PortAnaRecord(recorder)
    par.generate()
    
    # 获取结果
    analysis_df = recorder.load_objects("portfolio_analysis/report.pkl")
    print("\nQlib投资组合分析结果:")
    print(analysis_df)
    
    # 获取特征重要性
    if hasattr(model, "get_feature_importance"):
        feature_importances = model.get_feature_importance()
        print("\n特征重要性 (前10):")
        for feature, importance in sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    return {
        "model": model,
        "dataset": dataset,
        "recorder": recorder,
        "analysis": analysis_df,
    }


def visualize_qlib_results(results, config):
    """可视化Qlib结果
    
    参数:
        results (dict): Qlib实验结果
        config (dict): 配置信息
    """
    if not QLIB_AVAILABLE or results is None:
        return
    
    print("\n可视化Qlib结果...")
    
    # 创建可视化目录
    plots_dir = config['visualization']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取分析结果
    analysis_df = results["analysis"]
    
    # 绘制累积收益曲线
    plt.figure(figsize=(12, 6))
    analysis_df.loc[:, ["cumulative_return", "cumulative_return_benchmark"]].plot()
    plt.title("Qlib模型累积收益曲线")
    plt.xlabel("日期")
    plt.ylabel("累积收益")
    plt.grid(True)
    plt.legend(["策略", "基准"])
    
    # 保存图表
    cumulative_return_path = os.path.join(
        plots_dir, f"qlib_cumulative_return_{datetime.now().strftime('%Y%m%d')}.png"
    )
    plt.savefig(cumulative_return_path)
    print(f"累积收益曲线已保存至: {cumulative_return_path}")
    
    # 绘制超额收益曲线
    plt.figure(figsize=(12, 6))
    excess_return = analysis_df["cumulative_return"] - analysis_df["cumulative_return_benchmark"]
    excess_return.plot()
    plt.title("Qlib模型超额收益曲线")
    plt.xlabel("日期")
    plt.ylabel("超额收益")
    plt.grid(True)
    
    # 保存图表
    excess_return_path = os.path.join(
        plots_dir, f"qlib_excess_return_{datetime.now().strftime('%Y%m%d')}.png"
    )
    plt.savefig(excess_return_path)
    print(f"超额收益曲线已保存至: {excess_return_path}")
    
    # 绘制回撤曲线
    plt.figure(figsize=(12, 6))
    analysis_df.loc[:, ["drawdown", "max_drawdown"]].plot()
    plt.title("Qlib模型回撤曲线")
    plt.xlabel("日期")
    plt.ylabel("回撤")
    plt.grid(True)
    plt.legend(["回撤", "最大回撤"])
    
    # 保存图表
    drawdown_path = os.path.join(
        plots_dir, f"qlib_drawdown_{datetime.now().strftime('%Y%m%d')}.png"
    )
    plt.savefig(drawdown_path)
    print(f"回撤曲线已保存至: {drawdown_path}")
    
    # 绘制性能指标条形图
    metrics = [
        "annual_return",
        "max_drawdown",
        "information_ratio",
        "alpha",
        "beta",
        "sharpe",
        "win_rate",
    ]
    
    # 提取最后一行的性能指标
    performance = analysis_df.iloc[-1][metrics].to_dict()
    
    plt.figure(figsize=(12, 6))
    pd.Series(performance).plot(kind="bar")
    plt.title("Qlib模型性能指标")
    plt.ylabel("指标值")
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    
    # 保存图表
    metrics_path = os.path.join(
        plots_dir, f"qlib_metrics_{datetime.now().strftime('%Y%m%d')}.png"
    )
    plt.savefig(metrics_path)
    print(f"性能指标图已保存至: {metrics_path}")


def main():
    """主函数"""
    # 检查qlib是否可用
    if not QLIB_AVAILABLE:
        print("错误: 请先安装qlib: pip install pyqlib")
        return
    
    # 加载配置
    config = load_config()
    
    # 创建数据目录
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
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
        raw_data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source='yahoo')
        
        # 处理数据
        processor = DataProcessor()
        data = processor.process_stock_data(raw_data)
        
        # 保存处理后的数据
        data.to_csv(data_path, index=False)
        print(f"数据已保存至: {data_path}")
    
    print(f"数据集大小: {len(data)} 条记录")
    
    # 创建数据处理器
    processor = DataProcessor()
    
    # 准备Qlib数据
    qlib_data_dir = prepare_qlib_data(data, processor, config['data']['data_dir'])
    
    # 运行Qlib工作流
    results = run_qlib_workflow(qlib_data_dir, config)
    
    # 可视化结果
    if results is not None:
        visualize_qlib_results(results, config)
    
    print("\nQlib集成示例完成")


if __name__ == "__main__":
    main()