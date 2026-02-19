# 数据处理模块

import os
import pandas as pd
import numpy as np
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
    
    def process_stock_data(self, data):
        """
        处理股票数据（整合流程）
        
        参数:
            data: pandas.DataFrame, 包含多只股票的原始数据
            
        返回:
            pandas.DataFrame: 处理后的数据
        """
        if data is None or data.empty:
            return None
            
        # 确保有stock_id列
        if 'stock_id' not in data.columns:
            # 如果没有stock_id，假设是单只股票，添加虚拟ID
            data = data.copy()
            data['stock_id'] = 'unknown'
            
        # 1. 清洗
        df = self.clean_stock_data(data)
        
        # 2. 计算指标
        df = self.calculate_technical_indicators(df)
        
        return df

    def clean_stock_data(self, df):
        """
        清洗股票数据
        """
        if df is None or df.empty:
            return None
        
        df_clean = df.copy()
        
        # 按股票分组处理
        def _clean_group(group):
            # 排序
            group = group.sort_values('date')
            
            # 处理缺失值
            group = group.fillna(method='ffill').fillna(method='bfill')
            
            # 移除全局价格截断逻辑，因为它对于非平稳的时间序列（股价）是不合理的
            # 如果需要去极值，应该针对收益率进行，而不是原始价格
            
            return group

        # Apply to each group
        df_clean = df_clean.groupby('stock_id', group_keys=False).apply(_clean_group)
        
        return df_clean
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        """
        if df is None or df.empty:
            return None
        
        df_tech = df.copy()
        
        # 确保索引是日期类型（如果是索引）
        # 这里假设date是列
        if 'date' in df_tech.columns:
            df_tech['date'] = pd.to_datetime(df_tech['date'])
        
        def _calc_indicators(group):
            group = group.sort_values('date')
            
            # 计算移动平均线
            group['ma5'] = group['close'].rolling(window=5).mean()
            group['ma10'] = group['close'].rolling(window=10).mean()
            group['ma20'] = group['close'].rolling(window=20).mean()
            group['ma30'] = group['close'].rolling(window=30).mean()
            group['ma60'] = group['close'].rolling(window=60).mean()
            
            # 计算MACD
            group['ema12'] = group['close'].ewm(span=12, adjust=False).mean()
            group['ema26'] = group['close'].ewm(span=26, adjust=False).mean()
            group['dif'] = group['ema12'] - group['ema26']
            group['dea'] = group['dif'].ewm(span=9, adjust=False).mean()
            group['macd'] = 2 * (group['dif'] - group['dea'])
            
            # 计算RSI
            delta = group['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            group['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            group['ma20_std'] = group['close'].rolling(window=20).std()
            group['upper_band'] = group['ma20'] + 2 * group['ma20_std']
            group['lower_band'] = group['ma20'] - 2 * group['ma20_std']
            
            # 计算成交量变化
            if 'volume' in group.columns:
                group['volume_ma5'] = group['volume'].rolling(window=5).mean()
                group['volume_ma10'] = group['volume'].rolling(window=10).mean()
                # 避免除以0
                group['volume_ratio'] = np.where(group['volume_ma5'] != 0, group['volume'] / group['volume_ma5'], 0)
            
            # 计算价格变化率
            group['price_change'] = group['close'].pct_change()
            group['price_change_1d'] = group['close'].pct_change(1)
            group['price_change_5d'] = group['close'].pct_change(5)
            group['price_change_10d'] = group['close'].pct_change(10)
            group['price_change_20d'] = group['close'].pct_change(20)
            
            # 处理计算过程中产生的缺失值
            group = group.fillna(method='bfill')
            return group

        df_tech = df_tech.groupby('stock_id', group_keys=False).apply(_calc_indicators)
        
        return df_tech
