# 基础模型类

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from qlib.model.base import Model
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

class BaseModel(Model, ABC):
    """
    基础模型类，所有策略模型的父类
    """
    
    def __init__(self, config=None):
        """
        初始化基础模型
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, dataset):
        """
        训练模型
        
        参数:
            dataset: 训练数据集
            
        返回:
            self: 训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, dataset):
        """
        使用模型进行预测
        
        参数:
            dataset: 测试数据集
            
        返回:
            预测结果
        """
        pass
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存模型（具体实现由子类完成）
        self._save_model(path)
        print(f"模型已保存至 {path}")
    
    def load(self, path):
        """
        加载模型
        
        参数:
            path: 模型路径
            
        返回:
            self: 加载后的模型实例
        """
        self._load_model(path)
        self.fitted = True
        print(f"模型已从 {path} 加载")
        return self
    
    @abstractmethod
    def _save_model(self, path):
        """
        保存模型的具体实现
        
        参数:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def _load_model(self, path):
        """
        加载模型的具体实现
        
        参数:
            path: 模型路径
        """
        pass
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
            dict: 特征名称到重要性的映射
        """
        # 默认实现，子类可以覆盖
        return {}
    
    def evaluate(self, dataset, metrics=None):
        """
        评估模型性能
        
        参数:
            dataset: 评估数据集
            metrics: 评估指标列表
            
        返回:
            dict: 评估结果
        """
        # 默认评估指标
        if metrics is None:
            metrics = ['mse', 'mae', 'ic', 'rank_ic']
        
        # 获取预测结果
        pred = self.predict(dataset)
        
        # 获取真实标签
        label = dataset.get('label', None)
        if label is None:
            raise ValueError("数据集中没有标签，无法评估模型性能")
        
        # 计算评估指标
        results = {}
        for metric in metrics:
            if metric == 'mse':
                results[metric] = np.mean((pred - label) ** 2)
            elif metric == 'mae':
                results[metric] = np.mean(np.abs(pred - label))
            elif metric == 'ic':
                # 计算信息系数（Information Coefficient）
                ic_values = []
                for i in range(pred.shape[1]):
                    ic = np.corrcoef(pred[:, i], label[:, i])[0, 1]
                    ic_values.append(ic)
                results[metric] = np.nanmean(ic_values)
            elif metric == 'rank_ic':
                # 计算排名信息系数
                rank_ic_values = []
                for i in range(pred.shape[1]):
                    pred_rank = pd.Series(pred[:, i]).rank()
                    label_rank = pd.Series(label[:, i]).rank()
                    rank_ic = np.corrcoef(pred_rank, label_rank)[0, 1]
                    rank_ic_values.append(rank_ic)
                results[metric] = np.nanmean(rank_ic_values)
        
        return results