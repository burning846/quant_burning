# 数据获取模块

import os
import pandas as pd
import numpy as np
import akshare as ak
import tushare as ts
import yfinance as yf
from datetime import datetime
from qlib.data import D
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
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
        
        参数:
            code: 股票代码，如'000001.SZ'
            start_date: 开始日期，如'20180101'
            end_date: 结束日期，如'20211231'
            
        返回:
            pandas.DataFrame: 股票日线数据
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
            df.set_index('date', inplace=True)
        return df
    
    def fetch_stock_daily_from_akshare(self, code, start_date, end_date):
        """
        从AKShare获取股票日线数据
        
        参数:
            code: 股票代码，如'000001'（不带交易所后缀）
            start_date: 开始日期，如'20180101'
            end_date: 结束日期，如'20211231'
            
        返回:
            pandas.DataFrame: 股票日线数据
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
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"从AKShare获取数据失败: {e}")
            return None
    
    def fetch_stock_daily_from_yfinance(self, code, start_date, end_date):
        """
        从Yahoo Finance获取股票日线数据
        
        参数:
            code: 股票代码，如'000001.SS'（上交所）或'000001.SZ'（深交所）
            start_date: 开始日期，如'2018-01-01'
            end_date: 结束日期，如'2021-12-31'
            
        返回:
            pandas.DataFrame: 股票日线数据
        """
        # 转换为Yahoo Finance格式的股票代码
        if code.endswith('.SZ'):
            yf_code = code.replace('.SZ', '.SZ')
        elif code.endswith('.SH'):
            yf_code = code.replace('.SH', '.SS')
        else:
            yf_code = code
            
        try:
            # 将日期格式转换为yfinance所需的格式
            start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            
            # 使用yfinance获取数据
            df = yf.download(yf_code, start=start, end=end)
            if df is not None and not df.empty:
                # 重命名列以符合qlib格式
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                # 确保索引是日期类型
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"从Yahoo Finance获取数据失败: {e}")
            return None
    
    def save_data_to_csv(self, df, file_path):
        """
        将数据保存为CSV文件
        
        参数:
            df: pandas.DataFrame, 要保存的数据
            file_path: str, 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存数据
        df.to_csv(file_path)
        print(f"数据已保存至 {file_path}")
    
    def fetch_and_save_stock_list(self, save_path='data/raw/stock_list.csv'):
        """
        获取并保存股票列表
        
        参数:
            save_path: 保存路径
            
        返回:
            pandas.DataFrame: 股票列表数据
        """
        if self.ts_pro is None:
            raise ValueError("Tushare API未初始化，请提供有效的token")
        
        # 获取股票列表
        stock_list = self.ts_pro.stock_basic(exchange='', list_status='L', 
                                           fields='ts_code,symbol,name,area,industry,list_date')
        
        # 保存数据
        self.save_data_to_csv(stock_list, save_path)
        
        return stock_list
    
    def fetch_and_save_index_data(self, index_code='000300.SH', start_date='20180101', 
                                 end_date='20211231', save_path='data/raw/index_data.csv'):
        """
        获取并保存指数数据
        
        参数:
            index_code: 指数代码，默认为沪深300
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径
            
        返回:
            pandas.DataFrame: 指数数据
        """
        if self.ts_pro is None:
            raise ValueError("Tushare API未初始化，请提供有效的token")
        
        # 获取指数数据
        index_data = self.ts_pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        
        # 保存数据
        self.save_data_to_csv(index_data, save_path)
        
        return index_data
    
    def batch_fetch_stock_data(self, stock_list, start_date, end_date, source='tushare', 
                              save_dir='data/raw/stocks'):
        """
        批量获取股票数据
        
        参数:
            stock_list: 股票代码列表或DataFrame
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源，可选'tushare', 'akshare', 'yfinance'
            save_dir: 保存目录
            
        返回:
            dict: 股票代码到数据的映射
        """
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 如果stock_list是DataFrame，提取股票代码列
        if isinstance(stock_list, pd.DataFrame) and 'ts_code' in stock_list.columns:
            codes = stock_list['ts_code'].tolist()
        else:
            codes = stock_list
        
        result = {}
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
                    # 保存数据
                    file_name = f"{code.replace('.', '_')}.csv"
                    file_path = os.path.join(save_dir, file_name)
                    self.save_data_to_csv(df, file_path)
                    result[code] = df
            except Exception as e:
                print(f"获取{code}数据失败: {e}")
        
        return result