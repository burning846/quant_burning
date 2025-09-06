# 数据处理模块

import os
import pandas as pd
import numpy as np
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

class DataProcessor:
    """
    数据处理类，用于清洗数据、生成特征等
    """
    
    def __init__(self, config=None):
        """
        初始化数据处理器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
    
    def clean_stock_data(self, df):
        """
        清洗股票数据
        
        参数:
            df: pandas.DataFrame, 原始股票数据
            
        返回:
            pandas.DataFrame: 清洗后的数据
        """
        if df is None or df.empty:
            return None
        
        # 复制数据，避免修改原始数据
        df_clean = df.copy()
        
        # 处理缺失值
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # 处理异常值（例如使用3倍标准差法）
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean[col] = df_clean[col].clip(mean - 3 * std, mean + 3 * std)
        
        return df_clean
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        参数:
            df: pandas.DataFrame, 股票数据
            
        返回:
            pandas.DataFrame: 添加了技术指标的数据
        """
        if df is None or df.empty:
            return None
        
        # 复制数据，避免修改原始数据
        df_tech = df.copy()
        
        # 确保索引是日期类型
        if not isinstance(df_tech.index, pd.DatetimeIndex):
            df_tech.index = pd.to_datetime(df_tech.index)
        
        # 计算移动平均线
        df_tech['ma5'] = df_tech['close'].rolling(window=5).mean()
        df_tech['ma10'] = df_tech['close'].rolling(window=10).mean()
        df_tech['ma20'] = df_tech['close'].rolling(window=20).mean()
        df_tech['ma30'] = df_tech['close'].rolling(window=30).mean()
        df_tech['ma60'] = df_tech['close'].rolling(window=60).mean()
        
        # 计算MACD
        df_tech['ema12'] = df_tech['close'].ewm(span=12, adjust=False).mean()
        df_tech['ema26'] = df_tech['close'].ewm(span=26, adjust=False).mean()
        df_tech['dif'] = df_tech['ema12'] - df_tech['ema26']
        df_tech['dea'] = df_tech['dif'].ewm(span=9, adjust=False).mean()
        df_tech['macd'] = 2 * (df_tech['dif'] - df_tech['dea'])
        
        # 计算RSI
        delta = df_tech['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_tech['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df_tech['ma20_std'] = df_tech['close'].rolling(window=20).std()
        df_tech['upper_band'] = df_tech['ma20'] + 2 * df_tech['ma20_std']
        df_tech['lower_band'] = df_tech['ma20'] - 2 * df_tech['ma20_std']
        
        # 计算成交量变化
        if 'volume' in df_tech.columns:
            df_tech['volume_ma5'] = df_tech['volume'].rolling(window=5).mean()
            df_tech['volume_ma10'] = df_tech['volume'].rolling(window=10).mean()
            df_tech['volume_ratio'] = df_tech['volume'] / df_tech['volume_ma5']
        
        # 计算价格变化率
        df_tech['price_change'] = df_tech['close'].pct_change()
        df_tech['price_change_1d'] = df_tech['close'].pct_change(1)
        df_tech['price_change_5d'] = df_tech['close'].pct_change(5)
        df_tech['price_change_10d'] = df_tech['close'].pct_change(10)
        df_tech['price_change_20d'] = df_tech['close'].pct_change(20)
        
        # 处理计算过程中产生的缺失值
        df_tech = df_tech.fillna(method='bfill')
        
        return df_tech
    
    def normalize_features(self, df, method='standard'):
        """
        特征归一化
        
        参数:
            df: pandas.DataFrame, 包含特征的数据
            method: 归一化方法，'standard'或'minmax'
            
        返回:
            pandas.DataFrame: 归一化后的数据
        """
        if df is None or df.empty:
            return None
        
        # 复制数据，避免修改原始数据
        df_norm = df.copy()
        
        # 选择需要归一化的列（通常是数值型特征）
        numeric_cols = df_norm.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 排除不需要归一化的列，如日期、ID等
        exclude_cols = ['date', 'ts_code', 'code', 'symbol']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 选择归一化方法
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 对特征进行归一化
        if feature_cols:
            df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
        
        return df_norm
    
    def generate_qlib_features(self, df, label_col='close'):
        """
        生成qlib兼容的特征和标签
        
        参数:
            df: pandas.DataFrame, 股票数据
            label_col: 用于生成标签的列名
            
        返回:
            pandas.DataFrame: 包含特征和标签的数据
        """
        if df is None or df.empty:
            return None
        
        # 复制数据，避免修改原始数据
        df_qlib = df.copy()
        
        # 确保索引是日期类型
        if not isinstance(df_qlib.index, pd.DatetimeIndex):
            df_qlib.index = pd.to_datetime(df_qlib.index)
        
        # 生成未来收益率作为标签（例如，未来1天、5天、10天的收益率）
        df_qlib['label_1d'] = df_qlib[label_col].pct_change(-1)  # 未来1天收益率
        df_qlib['label_5d'] = df_qlib[label_col].pct_change(-5)  # 未来5天收益率
        df_qlib['label_10d'] = df_qlib[label_col].pct_change(-10)  # 未来10天收益率
        
        # 删除最后10行，因为它们的未来收益率标签是NaN
        df_qlib = df_qlib.iloc[:-10]
        
        # 处理计算过程中产生的缺失值
        df_qlib = df_qlib.fillna(method='bfill')
        
        return df_qlib
    
    def prepare_dataset_for_qlib(self, df, time_col='date', asset_col='symbol', 
                                feature_cols=None, label_cols=None):
        """
        准备qlib格式的数据集
        
        参数:
            df: pandas.DataFrame, 股票数据
            time_col: 时间列名
            asset_col: 资产列名
            feature_cols: 特征列名列表
            label_cols: 标签列名列表
            
        返回:
            pandas.DataFrame: qlib格式的数据集
        """
        if df is None or df.empty:
            return None
        
        # 复制数据，避免修改原始数据
        df_dataset = df.copy()
        
        # 如果未指定特征列和标签列，则使用默认值
        if feature_cols is None:
            # 排除标签列和非特征列
            exclude_cols = ['label_1d', 'label_5d', 'label_10d', 'date', 'ts_code', 'code', 'symbol']
            feature_cols = [col for col in df_dataset.columns if col not in exclude_cols]
        
        if label_cols is None:
            label_cols = ['label_1d', 'label_5d', 'label_10d']
        
        # 确保时间列是日期类型
        if time_col in df_dataset.columns:
            df_dataset[time_col] = pd.to_datetime(df_dataset[time_col])
        
        # 创建qlib格式的数据集
        dataset = {
            'feature': df_dataset[feature_cols].values,
            'label': df_dataset[label_cols].values if label_cols else None,
            'time': df_dataset[time_col].values if time_col in df_dataset.columns else None,
            'asset': df_dataset[asset_col].values if asset_col in df_dataset.columns else None
        }
        
        return dataset
    
    def batch_process_stock_data(self, data_dir, output_dir, process_type='all'):
        """
        批量处理股票数据
        
        参数:
            data_dir: 原始数据目录
            output_dir: 输出目录
            process_type: 处理类型，可选'clean', 'technical', 'normalize', 'qlib', 'all'
            
        返回:
            dict: 股票代码到处理后数据的映射
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        result = {}
        for file in tqdm(csv_files, desc=f"处理股票数据({process_type})"):
            try:
                # 读取数据
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # 根据处理类型选择处理方法
                if process_type == 'clean' or process_type == 'all':
                    df = self.clean_stock_data(df)
                
                if process_type == 'technical' or process_type == 'all':
                    df = self.calculate_technical_indicators(df)
                
                if process_type == 'normalize' or process_type == 'all':
                    df = self.normalize_features(df, method='standard')
                
                if process_type == 'qlib' or process_type == 'all':
                    df = self.generate_qlib_features(df)
                
                if df is not None and not df.empty:
                    # 保存处理后的数据
                    output_file = os.path.join(output_dir, file)
                    df.to_csv(output_file)
                    
                    # 提取股票代码
                    stock_code = file.replace('.csv', '').replace('_', '.')
                    result[stock_code] = df
            except Exception as e:
                print(f"处理{file}失败: {e}")
        
        return result