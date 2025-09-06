#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型预测示例脚本

本脚本展示如何使用训练好的模型进行股票收益预测
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from models.ml_model import MLModel
from models.dl_model import DLModel


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
    os.makedirs(config['models']['model_dir'], exist_ok=True)
    
    # 设置股票和时间范围
    symbols = config['data']['default_symbols']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 两年数据
    
    print(f"获取股票数据: {symbols}")
    print(f"时间范围: {start_date} 至 {end_date}")
    
    # 获取数据
    fetcher = DataFetcher()
    data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source='yahoo')
    print(f"获取到 {len(data)} 条数据记录")
    
    # 处理数据
    processor = DataProcessor()
    processed_data = processor.process_stock_data(data)
    print(f"处理后的数据: {len(processed_data)} 条记录")
    
    # 准备特征和标签
    features, labels = processor.prepare_features_and_labels(processed_data)
    print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 划分训练集和测试集
    train_size = int(len(features) * 0.8)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    
    # 训练机器学习模型
    print("\n训练机器学习模型...")
    ml_models = {
        'random_forest': MLModel(model_name='random_forest'),
        'linear': MLModel(model_name='linear'),
        'xgboost': MLModel(model_name='xgboost')
    }
    
    ml_predictions = {}
    for name, model in ml_models.items():
        print(f"训练 {name} 模型...")
        model.train(X_train, y_train)
        
        # 预测
        predictions = model.predict(X_test)
        ml_predictions[name] = predictions
        
        # 评估
        metrics = model.evaluate(X_test, y_test)
        print(f"{name} 模型评估结果:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # 特征重要性
        if name != 'linear':
            importances = model.get_feature_importance()
            print(f"\n{name} 模型特征重要性 (前5):")
            for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feature}: {importance:.4f}")
    
    # 训练深度学习模型
    print("\n训练深度学习模型...")
    dl_models = {
        'lstm': DLModel(model_name='lstm'),
        'mlp': DLModel(model_name='mlp')
    }
    
    dl_predictions = {}
    for name, model in dl_models.items():
        print(f"训练 {name} 模型...")
        model.train(X_train, y_train)
        
        # 预测
        predictions = model.predict(X_test)
        dl_predictions[name] = predictions
        
        # 评估
        metrics = model.evaluate(X_test, y_test)
        print(f"{name} 模型评估结果:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # 可视化预测结果
    print("\n可视化预测结果...")
    
    # 获取测试集对应的日期和股票
    test_data = processed_data.iloc[train_size:].reset_index(drop=True)
    
    # 选择一只股票进行可视化
    stock_id = symbols[0]
    stock_mask = test_data['stock_id'] == stock_id
    stock_dates = pd.to_datetime(test_data[stock_mask]['date'])
    actual_returns = y_test[stock_mask]
    
    # 绘制预测结果对比图
    plt.figure(figsize=(12, 8))
    
    # 绘制实际收益率
    plt.plot(stock_dates, actual_returns, 'k-', label='实际收益率')
    
    # 绘制机器学习模型预测
    for name, preds in ml_predictions.items():
        plt.plot(stock_dates, preds[stock_mask], '--', label=f'{name} 预测')
    
    # 绘制深度学习模型预测
    for name, preds in dl_predictions.items():
        plt.plot(stock_dates, preds[stock_mask], ':', label=f'{name} 预测')
    
    plt.title(f'{stock_id} 收益率预测对比')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    os.makedirs('../results/plots', exist_ok=True)
    plt.savefig(f'../results/plots/model_prediction_comparison_{datetime.now().strftime("%Y%m%d")}.png')
    plt.show()
    
    # 模型预测性能对比
    print("\n模型预测性能对比:")
    all_metrics = {}
    
    # 收集所有模型的MSE和R2
    for name, model in {**ml_models, **dl_models}.items():
        metrics = model.evaluate(X_test, y_test)
        all_metrics[name] = {
            'MSE': metrics.get('mse', 0),
            'R2': metrics.get('r2', 0)
        }
    
    # 转换为DataFrame并打印
    metrics_df = pd.DataFrame(all_metrics).T
    print(metrics_df)
    
    # 绘制性能对比条形图
    plt.figure(figsize=(12, 6))
    
    # MSE对比（越低越好）
    plt.subplot(1, 2, 1)
    metrics_df['MSE'].plot(kind='bar')
    plt.title('均方误差 (MSE) 对比')
    plt.ylabel('MSE (越低越好)')
    plt.grid(axis='y')
    
    # R2对比（越高越好）
    plt.subplot(1, 2, 2)
    metrics_df['R2'].plot(kind='bar')
    plt.title('决定系数 (R²) 对比')
    plt.ylabel('R² (越高越好)')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f'../results/plots/model_performance_comparison_{datetime.now().strftime("%Y%m%d")}.png')
    plt.show()
    
    print("\n预测示例完成")


if __name__ == "__main__":
    main()