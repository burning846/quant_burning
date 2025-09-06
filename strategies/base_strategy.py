# 基础策略类

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis
from qlib.utils import init_instance_by_config

class BaseStrategy(ABC):
    """
    基础策略类，所有交易策略的父类
    """
    
    def __init__(self, config=None):
        """
        初始化基础策略
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.positions = {}  # 当前持仓
        self.history = []  # 交易历史
    
    @abstractmethod
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
            data: 市场数据
            
        返回:
            pandas.DataFrame: 交易信号
        """
        pass
    
    @abstractmethod
    def generate_trade_decision(self, signals):
        """
        根据信号生成交易决策
        
        参数:
            signals: 交易信号
            
        返回:
            pandas.DataFrame: 交易决策
        """
        pass
    
    def update_positions(self, trade_decision, current_date):
        """
        更新持仓
        
        参数:
            trade_decision: 交易决策
            current_date: 当前日期
            
        返回:
            dict: 更新后的持仓
        """
        # 记录交易历史
        self.history.append({
            'date': current_date,
            'trade_decision': trade_decision
        })
        
        # 更新持仓
        for _, row in trade_decision.iterrows():
            stock_id = row['stock_id']
            weight = row['weight']
            
            if weight > 0:
                self.positions[stock_id] = weight
            else:
                # 如果权重为0，则清仓
                if stock_id in self.positions:
                    del self.positions[stock_id]
        
        return self.positions
    
    def get_positions(self):
        """
        获取当前持仓
        
        返回:
            dict: 当前持仓
        """
        return self.positions
    
    def get_history(self):
        """
        获取交易历史
        
        返回:
            list: 交易历史
        """
        return self.history
    
    def reset(self):
        """
        重置策略状态
        """
        self.positions = {}
        self.history = []