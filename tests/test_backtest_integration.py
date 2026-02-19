
import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

from backtest.backtest_engine import BacktestEngine
from backtest.risk_manager import RiskManager
from strategies.simple_strategies import MomentumStrategy

class MockStrategy(MomentumStrategy):
    """
    Mock策略，生成固定信号
    """
    def generate_signals(self, data):
        # 简单返回所有股票信号为1（买入）
        stock_ids = data['stock_id'].unique()
        return pd.DataFrame({
            'stock_id': stock_ids,
            'signal': 1.0
        })
        
    def generate_trade_decision(self, signals):
        # 简单均分权重
        if signals.empty:
            return pd.DataFrame()
        weight = 1.0 / len(signals)
        return pd.DataFrame({
            'stock_id': signals['stock_id'],
            'weight': weight
        })

class TestBacktestIntegration(unittest.TestCase):
    
    def setUp(self):
        # 构造Mock数据
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='B')
        self.data = []
        
        # 构造一只股票 AAPL，价格从100涨到150，再跌回120
        # 1-10日: 100
        # 11-20日: 100 -> 150 (涨)
        # 21-30日: 150 -> 120 (跌)
        
        for i, date in enumerate(dates):
            price = 100.0
            if i >= 10 and i < 20:
                price = 100.0 + (i - 10) * 5.0 # 100 -> 145
            elif i >= 20:
                price = 150.0 - (i - 20) * 3.0 # 150 -> 120
                
            self.data.append({
                'date': date,
                'stock_id': 'AAPL',
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1000
            })
            
            # 添加基准 SPY
            self.data.append({
                'date': date,
                'stock_id': 'SPY',
                'open': 100.0,
                'high': 100.0,
                'low': 100.0,
                'close': 100.0, # 基准不动
                'volume': 1000
            })
            
        self.df = pd.DataFrame(self.data)
        
        self.config = {
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'benchmark': 'SPY',
            'account': 100000,
            'commission_rate': 0.0, # 简化计算
            'slippage_rate': 0.0,
            'trade_frequency': 'day',
            'verbose': False,
            'risk_management': {
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.20,
                'trailing_stop_pct': 0.10,
                'max_drawdown_limit': 0.50
            }
        }
        
        self.engine = BacktestEngine(self.config)
        self.strategy = MockStrategy()
        
    def test_run_backtest(self):
        # 运行回测
        results = self.engine.run_backtest(self.strategy, self.df)
        
        # 验证结果结构
        self.assertIn('portfolio_value', results)
        self.assertIn('returns', results)
        self.assertIn('positions', results)
        self.assertIn('trades', results)
        
        # 验证交易逻辑
        # 我们期待策略在开始时买入 AAPL
        # 价格从100涨到125时（>20%），应该触发 Take Profit
        # 检查是否有卖出 AAPL 的交易
        
        trades = results['trades']
        has_sell = False
        for trade in trades:
            if trade['shares'] < 0 and trade['stock_id'] == 'AAPL':
                has_sell = True
                # 检查卖出价格是否触发了止盈 (大约在120-125之间)
                # 我们的模拟数据是线性增长的，具体哪一天触发取决于回测步进
                print(f"Sell triggered at {trade['date']} price {trade['price']}")
                break
                
        self.assertTrue(has_sell, "Should have triggered a sell signal due to take profit or trailing stop")
        
    def tearDown(self):
        # 清理可能生成的文件
        if os.path.exists('results'):
            # shutil.rmtree('results') # 暂时保留方便查看
            pass

if __name__ == '__main__':
    unittest.main()
