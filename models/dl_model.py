# 深度学习模型

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base_model import BaseModel

class LSTMModel(nn.Module):
    """
    LSTM神经网络模型
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out

class MLPModel(nn.Module):
    """
    多层感知机模型
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        super(MLPModel, self).__init__()
        
        # 构建MLP层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class DLModel(BaseModel):
    """
    深度学习模型类，支持LSTM和MLP等神经网络
    """
    
    def __init__(self, config=None):
        """
        初始化深度学习模型
        
        参数:
            config: 配置字典，包含模型类型、参数等
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'model_type': 'lstm',  # 默认使用LSTM
            'input_size': 10,  # 输入特征维度
            'hidden_size': 64,  # 隐藏层大小
            'num_layers': 2,  # LSTM层数
            'hidden_sizes': [64, 32],  # MLP隐藏层大小
            'output_size': 1,  # 输出维度
            'dropout': 0.2,  # Dropout比例
            'batch_size': 64,  # 批次大小
            'epochs': 100,  # 训练轮数
            'learning_rate': 0.001,  # 学习率
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
            'feature_cols': None,  # 特征列
            'label_col': 'label_1d',  # 标签列
            'sequence_length': 10  # 序列长度（用于LSTM）
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
        初始化具体的深度学习模型
        """
        model_type = self.config['model_type']
        
        if model_type == 'lstm':
            self.model = LSTMModel(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=self.config['output_size'],
                dropout=self.config['dropout']
            )
        elif model_type == 'mlp':
            self.model = MLPModel(
                input_size=self.config['input_size'],
                hidden_sizes=self.config['hidden_sizes'],
                output_size=self.config['output_size'],
                dropout=self.config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 将模型移动到指定设备
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
    
    def fit(self, dataset):
        """
        训练模型
        
        参数:
            dataset: 训练数据集，可以是字典或DataFrame
            
        返回:
            self: 训练后的模型实例
        """
        # 准备数据
        train_loader = self._prepare_data(dataset)
        
        # 训练模型
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for inputs, targets in train_loader:
                # 将数据移动到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Loss: {total_loss/len(train_loader):.4f}')
        
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
        
        # 准备数据
        test_loader = self._prepare_data(dataset, is_train=False)
        
        # 预测
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        # 合并预测结果
        pred = np.vstack(predictions) if predictions else np.array([])
        
        return pred
    
    def _prepare_data(self, dataset, is_train=True):
        """
        准备数据加载器
        
        参数:
            dataset: 数据集，可以是字典或DataFrame
            is_train: 是否是训练数据
            
        返回:
            torch.utils.data.DataLoader: 数据加载器
        """
        # 提取特征和标签
        if is_train:
            X, y = self._extract_features_and_label(dataset)
        else:
            X = self._extract_features(dataset)
            # 对于测试数据，创建一个虚拟标签
            y = np.zeros((X.shape[0], self.config['output_size']))
        
        # 对于LSTM模型，需要重塑特征为序列形式
        if self.config['model_type'] == 'lstm':
            X = self._reshape_to_sequences(X)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=is_train
        )
        
        return dataloader
    
    def _reshape_to_sequences(self, X):
        """
        将特征重塑为序列形式（用于LSTM）
        
        参数:
            X: 特征数组
            
        返回:
            numpy.ndarray: 重塑后的特征数组
        """
        # 获取序列长度
        seq_length = self.config['sequence_length']
        
        # 如果样本数小于序列长度，则填充
        if X.shape[0] < seq_length:
            padding = np.zeros((seq_length - X.shape[0], X.shape[1]))
            X = np.vstack([padding, X])
        
        # 创建序列
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i+seq_length])
        
        return np.array(sequences)
    
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
            if isinstance(label_col, list):
                y = dataset[label_col].values
            elif label_col in dataset.columns:
                y = dataset[label_col].values.reshape(-1, 1)
            else:
                raise ValueError(f"数据集中没有标签列 {label_col}")
        else:
            raise ValueError("不支持的数据集类型")
        
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
                if isinstance(label_col, list):
                    feature_cols = [col for col in dataset.columns if col not in label_col]
                else:
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
        # 保存模型参数和配置
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def _load_model(self, path):
        """
        加载模型
        
        参数:
            path: 模型路径
        """
        # 加载模型参数和配置
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新配置
        self.config.update(checkpoint['config'])
        
        # 重新初始化模型
        self._init_model()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])