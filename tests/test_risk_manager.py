
import unittest
import pandas as pd
from backtest.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'risk_management': {
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.20,
                'trailing_stop_pct': 0.15,
                'max_drawdown_limit': 0.20
            }
        }
        self.risk_manager = RiskManager(self.config)
        
    def test_initialization(self):
        self.assertEqual(self.risk_manager.stop_loss_pct, 0.10)
        self.assertEqual(self.risk_manager.take_profit_pct, 0.20)
        self.assertEqual(self.risk_manager.trailing_stop_pct, 0.15)
        self.assertEqual(self.risk_manager.max_drawdown_limit, 0.20)
        
    def test_stop_loss(self):
        # 模拟开仓: 100元买入 AAPL
        self.risk_manager.on_position_open('AAPL', 100.0)
        
        # 价格跌至 95 (跌5%) -> 不应触发
        close_list, reasons = self.risk_manager.check_positions({'AAPL': 95.0})
        self.assertNotIn('AAPL', close_list)
        
        # 价格跌至 89 (跌11%) -> 应触发止损 (阈值10%)
        close_list, reasons = self.risk_manager.check_positions({'AAPL': 89.0})
        self.assertIn('AAPL', close_list)
        self.assertIn('Stop Loss', reasons['AAPL'])
        
    def test_take_profit(self):
        # 模拟开仓: 100元买入 MSFT
        self.risk_manager.on_position_open('MSFT', 100.0)
        
        # 价格涨至 115 (涨15%) -> 不应触发 (阈值20%)
        close_list, reasons = self.risk_manager.check_positions({'MSFT': 115.0})
        self.assertNotIn('MSFT', close_list)
        
        # 价格涨至 125 (涨25%) -> 应触发固定止盈
        close_list, reasons = self.risk_manager.check_positions({'MSFT': 125.0})
        self.assertIn('MSFT', close_list)
        self.assertIn('Take Profit', reasons['MSFT'])

    def test_trailing_stop(self):
        # 模拟开仓: 100元买入 NVDA
        self.risk_manager.on_position_open('NVDA', 100.0)
        
        # 价格涨至 150
        self.risk_manager.check_positions({'NVDA': 150.0})
        # 此时最高价应为 150
        self.assertEqual(self.risk_manager.highest_prices['NVDA'], 150.0)
        
        # 价格回撤至 130 (回撤 (150-130)/150 = 13.3%) -> 不应触发 (阈值15%)
        close_list, reasons = self.risk_manager.check_positions({'NVDA': 130.0})
        self.assertNotIn('NVDA', close_list)
        
        # 价格回撤至 120 (回撤 (150-120)/150 = 20%) -> 应触发移动止盈
        close_list, reasons = self.risk_manager.check_positions({'NVDA': 120.0})
        self.assertIn('NVDA', close_list)
        self.assertIn('Trailing Stop', reasons['NVDA'])
        
    def test_account_max_drawdown(self):
        # 初始净值 100万
        self.risk_manager.update_portfolio_value(1000000)
        
        # 净值涨至 120万
        self.risk_manager.update_portfolio_value(1200000)
        self.assertEqual(self.risk_manager.peak_portfolio_value, 1200000)
        
        # 净值跌至 100万 (回撤 16.6%) -> 不触发 (阈值20%)
        is_liquidated = self.risk_manager.update_portfolio_value(1000000)
        self.assertFalse(is_liquidated)
        
        # 净值跌至 90万 (回撤 25%) -> 应触发账户熔断
        is_liquidated = self.risk_manager.update_portfolio_value(900000)
        self.assertTrue(is_liquidated)
        self.assertTrue(self.risk_manager.triggered_liquidation)
        
        # 触发熔断后，check_positions 应返回所有持仓
        self.risk_manager.on_position_open('TSLA', 200)
        close_list, reasons = self.risk_manager.check_positions({'TSLA': 200})
        self.assertIn('TSLA', close_list)
        self.assertIn('Account Max Drawdown', reasons['TSLA'])

if __name__ == '__main__':
    unittest.main()
