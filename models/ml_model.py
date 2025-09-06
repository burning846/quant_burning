# 机器学习模型

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class MLModel(BaseModel):
    """
    机器学习模型类，支持多种常用的机器学习算法
    """
    
    def __init__(self, config=None):
        """
        初始化机器学习模型
        
        参数:
            config: 配置字典，包含模型类型、参数等
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'model_type': 'random_forest',  # 默认使用随机森林
            'model_kwargs': {},  # 模型参数
            'feature_cols': None,  # 特征列
            'label_col': 'label_1d'  # 标签列
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """
        初始化具体的机器学习模型
        """
        model_type = self.config['model_type']
        model_kwargs = self.config['model_kwargs']
        
        # 根据模型类型选择具体的模型
        if model_type == 'linear':
            self.model = LinearRegression(**model_kwargs)
        elif model_type == 'logistic':
            self.model = LogisticRegression(**model_kwargs)
        elif model_type == 'ridge':
            self.model = Ridge(**model_kwargs)
        elif model_type == 'lasso':
            self.model = Lasso(**model_kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**model_kwargs)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**model_kwargs)
        elif model_type == 'svr':
            self.model = SVR(**model_kwargs)
        elif model_type == 'mlp':
            self.model = MLPRegressor(**model_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def fit(self, dataset):
        """
        训练模型
        
        参数:
            dataset: 训练数据集，可以是字典或DataFrame
            
        返回:
            self: 训练后的模型实例
        """
        # 提取特征和标签
        X, y = self._extract_features_and_label(dataset)
        
        # 训练模型
        self.model.fit(X, y)
        self.fitted = True
        
        return self
    
    def predict(self, dataset):
        """
        使用模型进行预测
        
        参数:
            dataset: 测试数据集，可以是字典或DataFrame
            
        返回:
            numpy.ndarray: 预测结果
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 提取特征
        X = self._extract_features(dataset)
        
        # 进行预测
        pred = self.model.predict(X)
        
        # 确保预测结果是二维数组
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)
        
        return pred
    
    def _extract_features_and_label(self, dataset):
        """
        从数据集中提取特征和标签
        
        参数:
            dataset: 数据集，可以是字典或DataFrame
            
        返回:
            tuple: (特征数组, 标签数组)
        """
        # 提取特征
        X = self._extract_features(dataset)
        
        # 提取标签
        if isinstance(dataset, dict):
            if 'label' in dataset:
                y = dataset['label']
            else:
                raise ValueError("数据集中没有标签")
        elif isinstance(dataset, pd.DataFrame):
            label_col = self.config['label_col']
            if label_col in dataset.columns:
                y = dataset[label_col].values
            else:
                raise ValueError(f"数据集中没有标签列 {label_col}")
        else:
            raise ValueError("不支持的数据集类型")
        
        # 确保标签是一维数组
        if len(y.shape) > 1 and y.shape[1] > 1:
            # 如果有多个标签列，只使用第一个
            y = y[:, 0]
        
        return X, y
    
    def _extract_features(self, dataset):
        """
        从数据集中提取特征
        
        参数:
            dataset: 数据集，可以是字典或DataFrame
            
        返回:
            numpy.ndarray: 特征数组
        """
        if isinstance(dataset, dict):
            if 'feature' in dataset:
                X = dataset['feature']
            else:
                raise ValueError("数据集中没有特征")
        elif isinstance(dataset, pd.DataFrame):
            feature_cols = self.config['feature_cols']
            if feature_cols is None:
                # 如果未指定特征列，则使用所有非标签列作为特征
                label_col = self.config['label_col']
                feature_cols = [col for col in dataset.columns if col != label_col]
            X = dataset[feature_cols].values
        else:
            raise ValueError("不支持的数据集类型")
        
        return X
    
    def _save_model(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def _load_model(self, path):
        """
        加载模型
        
        参数:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
            dict: 特征名称到重要性的映射
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 检查模型是否支持特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # 获取特征名称
            feature_cols = self.config['feature_cols']
            if feature_cols is None:
                # 如果未指定特征列，则使用索引作为名称
                feature_cols = [f'feature_{i}' for i in range(len(importances))]
            
            # 创建特征重要性字典
            importance_dict = {}
            for i, col in enumerate(feature_cols):
                if i < len(importances):
                    importance_dict[col] = importances[i]
            
            return importance_dict
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型，使用系数作为特征重要性
            coefs = self.model.coef_
            
            # 确保系数是一维数组
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            
            # 获取特征名称
            feature_cols = self.config['feature_cols']
            if feature_cols is None:
                # 如果未指定特征列，则使用索引作为名称
                feature_cols = [f'feature_{i}' for i in range(len(coefs))]
            
            # 创建特征重要性字典
            importance_dict = {}
            for i, col in enumerate(feature_cols):
                if i < len(coefs):
                    importance_dict[col] = abs(coefs[i])  # 使用系数的绝对值
            
            return importance_dict
        else:
            # 模型不支持特征重要性
            return {}