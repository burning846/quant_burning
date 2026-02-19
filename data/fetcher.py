# 数据获取模块

import os
import pandas as pd
import numpy as np
import akshare as ak
import tushare as ts
import yfinance as yf
from datetime import datetime
from tqdm import tqdm

class DataFetcher:
    """
    数据获取类，支持从多种来源获取股票数据
    """
    
    def __init__(self, config=None):
        """
        初始化数据获取器
        
        参数:
            config: 配置字典，包含API密钥等信息
        """
        self.config = config or {}
        # 如果有Tushare token，则设置
        if 'tushare_token' in self.config:
            ts.set_token(self.config['tushare_token'])
            self.ts_pro = ts.pro_api()
        else:
            self.ts_pro = None
    
    def fetch_stock_daily_from_tushare(self, code, start_date, end_date):
        """
        从Tushare获取股票日线数据
        """
        if self.ts_pro is None:
            raise ValueError("Tushare API未初始化，请提供有效的token")
        
        df = self.ts_pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            # 按日期排序
            df = df.sort_values('trade_date')
            # 重命名列以符合qlib格式
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            })
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    def fetch_stock_daily_from_akshare(self, code, start_date, end_date):
        """
        从AKShare获取股票日线数据
        """
        # AKShare的股票代码格式转换
        if code.endswith('.SZ'):
            ak_code = code.split('.')[0]
        elif code.endswith('.SH'):
            ak_code = code.split('.')[0]
        else:
            ak_code = code
            
        try:
            # 使用AKShare获取股票日线数据
            df = ak.stock_zh_a_hist(symbol=ak_code, start_date=start_date, end_date=end_date, adjust="qfq")
            if df is not None and not df.empty:
                # 重命名列以符合qlib格式
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"从AKShare获取数据失败: {e}")
            return None
    
    def fetch_stock_daily_from_yfinance(self, code, start_date, end_date):
        """
        从Yahoo Finance获取股票日线数据
        """
        # 转换为Yahoo Finance格式的股票代码
        if code.endswith('.SZ'):
            yf_code = code.replace('.SZ', '.SZ')
        elif code.endswith('.SH'):
            yf_code = code.replace('.SH', '.SS')
        else:
            # 对于美股代码，直接使用原代码
            yf_code = code
            
        try:
            # 将日期格式转换为yfinance所需的格式
            start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
            # 使用yfinance获取数据
            try:
                # 尝试使用新版参数 auto_adjust
                df = yf.download(yf_code, start=start, end=end, progress=False, auto_adjust=True)
            except TypeError:
                # 兼容旧版本
                print(f"Warning: yfinance version might be incompatible with auto_adjust, falling back to default.")
                df = yf.download(yf_code, start=start, end=end, progress=False)
                
            if df is not None and not df.empty:
                # 如果是多级索引（yfinance >= 0.2.x 可能会返回多级索引），降级处理
                if isinstance(df.columns, pd.MultiIndex):
                    # 尝试找到 Close 列所在的层级
                    try:
                        # 这种情况下通常列是 (Price, Ticker)
                        df.columns = df.columns.get_level_values(0)
                    except:
                        pass
                    
                # 重命名列以符合qlib格式
                # 转换为小写并映射
                df.columns = [c.lower() for c in df.columns]
                rename_map = {
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'adj close': 'close' # 如果没有 auto_adjust，优先用 adj close
                }
                
                # 如果存在 'adj close' 且我们想要复权数据，优先使用它作为 close
                if 'adj close' in df.columns and 'close' in df.columns:
                    # 如果刚才没用 auto_adjust，这里手动替换
                    df['close'] = df['adj close']
                    
                df = df.rename(columns=rename_map)
                
                # 确保包含所需的列
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    # 尝试根据现有列进行修复
                    pass
                # 将index转为column
                df.reset_index(inplace=True)
                df = df.rename(columns={'Date': 'date'})
                # 确保日期类型
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"从Yahoo Finance获取数据失败: {e}")
            return None
    
    def save_data_to_csv(self, df, file_path):
        """
        将数据保存为CSV文件
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存数据
        df.to_csv(file_path, index=False)
        print(f"数据已保存至 {file_path}")
    
    def batch_fetch_stock_data(self, stock_list, start_date, end_date, source='tushare', 
                              save_dir='data/raw/stocks'):
        """
        批量获取股票数据
        
        返回:
            pandas.DataFrame: 合并后的所有股票数据
        """
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 如果stock_list是DataFrame，提取股票代码列
        if isinstance(stock_list, pd.DataFrame) and 'ts_code' in stock_list.columns:
            codes = stock_list['ts_code'].tolist()
        else:
            codes = stock_list
        
        all_data = []
        for code in tqdm(codes, desc=f"从{source}获取股票数据"):
            try:
                # 根据数据源选择获取方法
                if source.lower() == 'tushare':
                    df = self.fetch_stock_daily_from_tushare(code, start_date, end_date)
                elif source.lower() == 'akshare':
                    df = self.fetch_stock_daily_from_akshare(code, start_date, end_date)
                elif source.lower() == 'yfinance':
                    df = self.fetch_stock_daily_from_yfinance(code, start_date, end_date)
                else:
                    raise ValueError(f"不支持的数据源: {source}")
                
                if df is not None and not df.empty:
                    df['stock_id'] = code
                    # 保存数据
                    file_name = f"{code.replace('.', '_')}.csv"
                    file_path = os.path.join(save_dir, file_name)
                    self.save_data_to_csv(df, file_path)
                    all_data.append(df)
            except Exception as e:
                print(f"获取{code}数据失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)
