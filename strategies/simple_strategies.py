# 简单策略实现

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    动量策略：买入过去表现最好的股票
    """
    
    def __init__(self, config=None):
        """
        初始化动量策略
        
        参数:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'lookback_period': 20,  # 回看期
            'holding_period': 5,  # 持有期
            'topk': 10,  # 买入前k只股票
            'price_col': 'close'  # 价格列名
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
            data: 市场数据，包含多只股票的价格数据
            
        返回:
            pandas.DataFrame: 交易信号
        """
        # 确保数据是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("数据必须是pandas.DataFrame")
        
        # 获取配置参数
        lookback_period = self.config['lookback_period']
        price_col = self.config['price_col']
        
        # 计算每只股票的回报率
        returns = {}
        for stock_id, stock_data in data.groupby('stock_id'):
            # 确保数据按日期排序
            stock_data = stock_data.sort_values('date')
            
            # 计算过去lookback_period的回报率
            if len(stock_data) > lookback_period:
                past_price = stock_data[price_col].iloc[-lookback_period-1]
                current_price = stock_data[price_col].iloc[-1]
                returns[stock_id] = (current_price / past_price) - 1
        
        # 创建信号DataFrame
        signals = pd.DataFrame({
            'stock_id': list(returns.keys()),
            'signal': list(returns.values())
        })
        
        return signals
    
    def generate_trade_decision(self, signals):
        """
        根据信号生成交易决策
        
        参数:
            signals: 交易信号
            
        返回:
            pandas.DataFrame: 交易决策
        """
        # 获取配置参数
        topk = self.config['topk']
        
        # 选择信号最强的topk只股票
        top_signals = signals.sort_values('signal', ascending=False).head(topk)
        
        # 计算每只股票的权重（等权重）
        weight = 1.0 / len(top_signals) if len(top_signals) > 0 else 0
        
        # 创建交易决策DataFrame
        trade_decision = pd.DataFrame({
            'stock_id': top_signals['stock_id'],
            'weight': weight
        })
        
        return trade_decision

class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略：买入过去表现最差的股票
    """
    
    def __init__(self, config=None):
        """
        初始化均值回归策略
        
        参数:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'lookback_period': 20,  # 回看期
            'holding_period': 5,  # 持有期
            'topk': 10,  # 买入前k只股票
            'price_col': 'close'  # 价格列名
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
            data: 市场数据，包含多只股票的价格数据
            
        返回:
            pandas.DataFrame: 交易信号
        """
        # 确保数据是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("数据必须是pandas.DataFrame")
        
        # 获取配置参数
        lookback_period = self.config['lookback_period']
        price_col = self.config['price_col']
        
        # 计算每只股票的回报率
        returns = {}
        for stock_id, stock_data in data.groupby('stock_id'):
            # 确保数据按日期排序
            stock_data = stock_data.sort_values('date')
            
            # 计算过去lookback_period的回报率
            if len(stock_data) > lookback_period:
                past_price = stock_data[price_col].iloc[-lookback_period-1]
                current_price = stock_data[price_col].iloc[-1]
                returns[stock_id] = (current_price / past_price) - 1
        
        # 创建信号DataFrame
        signals = pd.DataFrame({
            'stock_id': list(returns.keys()),
            'signal': list(returns.values())
        })
        
        return signals
    
    def generate_trade_decision(self, signals):
        """
        根据信号生成交易决策
        
        参数:
            signals: 交易信号
            
        返回:
            pandas.DataFrame: 交易决策
        """
        # 获取配置参数
        topk = self.config['topk']
        
        # 选择信号最弱的topk只股票（均值回归策略买入表现最差的股票）
        bottom_signals = signals.sort_values('signal', ascending=True).head(topk)
        
        # 计算每只股票的权重（等权重）
        weight = 1.0 / len(bottom_signals) if len(bottom_signals) > 0 else 0
        
        # 创建交易决策DataFrame
        trade_decision = pd.DataFrame({
            'stock_id': bottom_signals['stock_id'],
            'weight': weight
        })
        
        return trade_decision

class MovingAverageCrossStrategy(BaseStrategy):
    """
    均线交叉策略：当短期均线上穿长期均线时买入，下穿时卖出
    """
    
    def __init__(self, config=None):
        """
        初始化均线交叉策略
        
        参数:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'short_window': 5,  # 短期窗口
            'long_window': 20,  # 长期窗口
            'price_col': 'close'  # 价格列名
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
            data: 市场数据，包含多只股票的价格数据
            
        返回:
            pandas.DataFrame: 交易信号
        """
        # 确保数据是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("数据必须是pandas.DataFrame")
        
        # 获取配置参数
        short_window = self.config['short_window']
        long_window = self.config['long_window']
        price_col = self.config['price_col']
        
        # 计算每只股票的均线交叉信号
        signals_list = []
        for stock_id, stock_data in data.groupby('stock_id'):
            # 确保数据按日期排序
            stock_data = stock_data.sort_values('date')
            
            # 计算短期和长期移动平均线
            if len(stock_data) > long_window:
                short_ma = stock_data[price_col].rolling(window=short_window).mean()
                long_ma = stock_data[price_col].rolling(window=long_window).mean()
                
                # 计算当前和前一个交易日的均线差值
                current_diff = short_ma.iloc[-1] - long_ma.iloc[-1]
                prev_diff = short_ma.iloc[-2] - long_ma.iloc[-2]
                
                # 判断是否发生上穿或下穿
                if prev_diff <= 0 and current_diff > 0:  # 上穿
                    signal = 1  # 买入信号
                elif prev_diff >= 0 and current_diff < 0:  # 下穿
                    signal = -1  # 卖出信号
                else:
                    signal = 0  # 无信号
                
                signals_list.append({
                    'stock_id': stock_id,
                    'signal': signal,
                    'date': stock_data['date'].iloc[-1]
                })
        
        # 创建信号DataFrame
        signals = pd.DataFrame(signals_list)
        
        return signals
    
    def generate_trade_decision(self, signals):
        """
        根据信号生成交易决策
        
        参数:
            signals: 交易信号
            
        返回:
            pandas.DataFrame: 交易决策
        """
        # 选择有买入信号的股票
        buy_signals = signals[signals['signal'] == 1]
        
        # 选择有卖出信号的股票
        sell_signals = signals[signals['signal'] == -1]
        
        # 获取当前持仓
        current_positions = set(self.positions.keys())
        
        # 需要买入的股票
        buy_stocks = set(buy_signals['stock_id']) - current_positions
        
        # 需要卖出的股票
        sell_stocks = set(sell_signals['stock_id']) & current_positions
        
        # 保持持有的股票
        hold_stocks = current_positions - sell_stocks
        
        # 计算新的持仓权重
        total_stocks = len(buy_stocks) + len(hold_stocks)
        weight = 1.0 / total_stocks if total_stocks > 0 else 0
        
        # 创建交易决策DataFrame
        trade_decisions = []
        
        # 添加买入决策
        for stock_id in buy_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': weight
            })
        
        # 添加持有决策
        for stock_id in hold_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': weight
            })
        
        # 添加卖出决策
        for stock_id in sell_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': 0  # 权重为0表示卖出
            })
        
        return pd.DataFrame(trade_decisions)

class RSIStrategy(BaseStrategy):
    """
    RSI策略：当RSI低于超卖线时买入，高于超买线时卖出
    """
    
    def __init__(self, config=None):
        """
        初始化RSI策略
        
        参数:
            config: 配置字典
        """
        super().__init__(config)
        
        # 默认配置
        default_config = {
            'rsi_period': 14,  # RSI计算周期
            'oversold_threshold': 30,  # 超卖阈值
            'overbought_threshold': 70,  # 超买阈值
            'price_col': 'close'  # 价格列名
        }
        
        # 更新配置
        if self.config is None:
            self.config = default_config
        else:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
    
    def calculate_rsi(self, prices, period=14):
        """
        计算RSI指标
        
        参数:
            prices: 价格序列
            period: RSI计算周期
            
        返回:
            numpy.ndarray: RSI值
        """
        # 计算价格变化
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        # 计算RSI
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
            data: 市场数据，包含多只股票的价格数据
            
        返回:
            pandas.DataFrame: 交易信号
        """
        # 确保数据是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("数据必须是pandas.DataFrame")
        
        # 获取配置参数
        rsi_period = self.config['rsi_period']
        oversold_threshold = self.config['oversold_threshold']
        overbought_threshold = self.config['overbought_threshold']
        price_col = self.config['price_col']
        
        # 计算每只股票的RSI信号
        signals_list = []
        for stock_id, stock_data in data.groupby('stock_id'):
            # 确保数据按日期排序
            stock_data = stock_data.sort_values('date')
            
            # 计算RSI
            if len(stock_data) > rsi_period:
                prices = stock_data[price_col].values
                rsi = self.calculate_rsi(prices, rsi_period)
                
                # 获取当前和前一个交易日的RSI值
                current_rsi = rsi[-1]
                prev_rsi = rsi[-2]
                
                # 判断是否发生超买或超卖
                if prev_rsi >= oversold_threshold and current_rsi < oversold_threshold:  # 进入超卖区域
                    signal = 1  # 买入信号
                elif prev_rsi <= overbought_threshold and current_rsi > overbought_threshold:  # 进入超买区域
                    signal = -1  # 卖出信号
                else:
                    signal = 0  # 无信号
                
                signals_list.append({
                    'stock_id': stock_id,
                    'signal': signal,
                    'rsi': current_rsi,
                    'date': stock_data['date'].iloc[-1]
                })
        
        # 创建信号DataFrame
        signals = pd.DataFrame(signals_list)
        
        return signals
    
    def generate_trade_decision(self, signals):
        """
        根据信号生成交易决策
        
        参数:
            signals: 交易信号
            
        返回:
            pandas.DataFrame: 交易决策
        """
        # 选择有买入信号的股票
        buy_signals = signals[signals['signal'] == 1]
        
        # 选择有卖出信号的股票
        sell_signals = signals[signals['signal'] == -1]
        
        # 获取当前持仓
        current_positions = set(self.positions.keys())
        
        # 需要买入的股票
        buy_stocks = set(buy_signals['stock_id']) - current_positions
        
        # 需要卖出的股票
        sell_stocks = set(sell_signals['stock_id']) & current_positions
        
        # 保持持有的股票
        hold_stocks = current_positions - sell_stocks
        
        # 计算新的持仓权重
        total_stocks = len(buy_stocks) + len(hold_stocks)
        weight = 1.0 / total_stocks if total_stocks > 0 else 0
        
        # 创建交易决策DataFrame
        trade_decisions = []
        
        # 添加买入决策
        for stock_id in buy_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': weight
            })
        
        # 添加持有决策
        for stock_id in hold_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': weight
            })
        
        # 添加卖出决策
        for stock_id in sell_stocks:
            trade_decisions.append({
                'stock_id': stock_id,
                'weight': 0  # 权重为0表示卖出
            })
        
        return pd.DataFrame(trade_decisions)