import pandas as pd
import numpy as np

class RiskManager:
    """
    风控管理器
    负责监控持仓风险，触发止损、移动止盈和账户级风控
    """
    
    def __init__(self, config=None):
        """
        初始化风控管理器
        
        参数:
            config: 配置字典，包含风控参数
        """
        self.config = config or {}
        
        # 获取风控参数
        risk_config = self.config.get('risk_management', {})
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.10)      # 默认10%止损
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.15) # 默认15%移动止盈
        self.max_drawdown_limit = risk_config.get('max_drawdown_limit', 0.25) # 默认25%账户最大回撤
        
        # 状态记录
        self.entry_prices = {}       # 持仓成本价 {stock_id: price}
        self.highest_prices = {}     # 持仓期间最高价 {stock_id: price}
        self.peak_portfolio_value = 0 # 账户历史最高净值
        self.triggered_liquidation = False # 是否触发了账户级强平
        
    def reset(self):
        """重置状态"""
        self.entry_prices = {}
        self.highest_prices = {}
        self.peak_portfolio_value = 0
        self.triggered_liquidation = False
        
    def update_portfolio_value(self, current_value):
        """
        更新账户净值，检查账户级风控
        
        参数:
            current_value: 当前账户总价值
            
        返回:
            bool: 是否触发账户级强平
        """
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            if drawdown > self.max_drawdown_limit:
                self.triggered_liquidation = True
                return True
                
        return False
        
    def on_position_open(self, stock_id, price):
        """
        记录开仓信息
        """
        self.entry_prices[stock_id] = price
        self.highest_prices[stock_id] = price
        
    def on_position_close(self, stock_id):
        """
        清理平仓信息
        """
        if stock_id in self.entry_prices:
            del self.entry_prices[stock_id]
        if stock_id in self.highest_prices:
            del self.highest_prices[stock_id]
            
    def check_positions(self, current_prices):
        """
        检查所有持仓，返回需要平仓的股票列表
        
        参数:
            current_prices: 当前价格字典 {stock_id: price} or DataFrame
            
        返回:
            list: 需要平仓的股票代码列表
            dict: 平仓原因 {stock_id: reason}
        """
        close_list = []
        reasons = {}
        
        # 如果触发了账户级强平，清空所有持仓
        if self.triggered_liquidation:
            for stock_id in self.entry_prices.keys():
                close_list.append(stock_id)
                reasons[stock_id] = "Account Max Drawdown Limit Triggered"
            return close_list, reasons
            
        for stock_id, entry_price in self.entry_prices.items():
            if stock_id not in current_prices:
                continue
                
            current_price = current_prices[stock_id]
            
            # 更新最高价
            if current_price > self.highest_prices.get(stock_id, 0):
                self.highest_prices[stock_id] = current_price
                
            # 1. 检查固定止损 (Stop Loss)
            # 亏损超过设定比例
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct > self.stop_loss_pct:
                close_list.append(stock_id)
                reasons[stock_id] = f"Stop Loss Triggered: -{loss_pct:.2%}"
                continue
                
            # 2. 检查固定止盈 (Take Profit)
            # 盈利超过设定比例
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > self.take_profit_pct:
                close_list.append(stock_id)
                reasons[stock_id] = f"Take Profit Triggered: +{profit_pct:.2%}"
                continue

            # 3. 检查移动止盈 (Trailing Stop)
            # 从最高点回撤超过设定比例
            highest = self.highest_prices[stock_id]
            if highest > 0:
                drawdown = (highest - current_price) / highest
                if drawdown > self.trailing_stop_pct:
                    # 只有在盈利的情况下才触发移动止盈（或者允许在亏损时也触发以保护本金，这里假设保护最高点）
                    # 通常移动止盈是为了保住利润，但也可用作动态止损
                    close_list.append(stock_id)
                    reasons[stock_id] = f"Trailing Stop Triggered: -{drawdown:.2%} from high"
                    
        return close_list, reasons
