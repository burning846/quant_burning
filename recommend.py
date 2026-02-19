
import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import yaml

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from strategies.simple_strategies import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossStrategy, RSIStrategy
from backtest.risk_manager import RiskManager

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_strategy(strategy_name, config):
    backtest_config = config.get('backtest', {})
    
    if strategy_name == 'Momentum':
        lookback_period = backtest_config.get('lookback_period', 20)
        holding_period = backtest_config.get('holding_period', 5)
        return MomentumStrategy(lookback_period=lookback_period, holding_period=holding_period)
    elif strategy_name == 'MeanReversion':
        lookback_period = backtest_config.get('lookback_period', 10)
        threshold = backtest_config.get('threshold', 1.5)
        return MeanReversionStrategy(lookback_period=lookback_period, threshold=threshold)
    elif strategy_name == 'MovingAverageCross':
        short_window = backtest_config.get('short_window', 5)
        long_window = backtest_config.get('long_window', 20)
        return MovingAverageCrossStrategy(short_window=short_window, long_window=long_window)
    elif strategy_name == 'RSI':
        rsi_period = backtest_config.get('rsi_period', 14)
        overbought = backtest_config.get('overbought', 70)
        oversold = backtest_config.get('oversold', 30)
        return RSIStrategy(rsi_period=rsi_period, overbought=overbought, oversold=oversold)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def main():
    parser = argparse.ArgumentParser(description='美股量化交易决策推荐系统')
    parser.add_argument('--config', type=str, default='config/config_us.yaml', help='配置文件路径')
    parser.add_argument('--lookback', type=int, default=100, help='获取历史数据天数')
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # 1. 获取数据
    data_config = config.get('data', {})
    symbols = data_config.get('default_symbols', [])
    source = data_config.get('default_source', 'yfinance')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.lookback)).strftime('%Y-%m-%d')
    
    print(f"Fetching data for {len(symbols)} symbols from {source} ({start_date} to {end_date})...")
    
    fetcher = DataFetcher()
    # 强制使用网络获取最新数据
    try:
        raw_data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source=source)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if raw_data.empty:
        print("No data fetched.")
        return

    print("Processing data...")
    processor = DataProcessor()
    data = processor.process_stock_data(raw_data)
    
    # 2. 初始化策略
    strategy_name = config.get('backtest', {}).get('strategy', 'Momentum')
    print(f"Running strategy: {strategy_name}")
    strategy = get_strategy(strategy_name, config)
    
    # 3. 生成信号
    # 我们只关心最近的信号，但策略可能需要历史窗口，所以传入所有数据
    print("Generating signals...")
    signals = strategy.generate_signals(data)
    trade_decisions = strategy.generate_trade_decision(signals)
    
    # 4. 分析最新一天的决策
    latest_date = data['date'].max()
    print(f"\n=== 交易推荐报告 ({latest_date.strftime('%Y-%m-%d')}) ===")
    print(f"策略: {strategy_name}")
    
    # 获取建议持仓的股票列表
    target_portfolio = []
    if not trade_decisions.empty:
        target_portfolio = trade_decisions[trade_decisions['weight'] > 0]['stock_id'].tolist()
    
    recommendations = []
    
    for symbol in symbols:
        symbol_data = data[data['stock_id'] == symbol].sort_values('date')
        if symbol_data.empty:
            continue
            
        latest_row = symbol_data.iloc[-1]
        latest_price = latest_row['close']
        
        # 判断建议动作
        if symbol in target_portfolio:
            action = "BUY / HOLD"
        else:
            action = "SELL / AVOID"
            
        # 获取一些技术指标供参考
        indicators = []
        # 计算简单的移动平均线
        ma5 = symbol_data['close'].rolling(5).mean().iloc[-1]
        ma20 = symbol_data['close'].rolling(20).mean().iloc[-1]
        indicators.append(f"MA5: {ma5:.2f}")
        indicators.append(f"MA20: {ma20:.2f}")
        
        # 如果策略计算了特定指标，尝试获取
        if strategy_name == 'RSI' and 'rsi' in signals.columns:
            # 尝试获取RSI值
            if 'date' in signals.columns:
                sig_row = signals[(signals['stock_id'] == symbol) & (signals['date'] == latest_date)]
            else:
                # 假设Momentum等策略的信号对应最新日期
                sig_row = signals[signals['stock_id'] == symbol]
                
            if not sig_row.empty and 'rsi' in sig_row.columns:
                indicators.append(f"RSI: {sig_row.iloc[0]['rsi']:.2f}")
                
        elif strategy_name == 'Momentum' and 'signal' in signals.columns:
             sig_row = signals[signals['stock_id'] == symbol]
             if not sig_row.empty:
                 indicators.append(f"Momentum: {sig_row.iloc[0]['signal']:.2%}")

        recommendations.append({
            'Symbol': symbol,
            'Price': latest_price,
            'Action': action,
            'Indicators': ", ".join(indicators)
        })
        
    # 5. 输出表格
    rec_df = pd.DataFrame(recommendations)
    # 按Action排序，推荐买入的排前面
    rec_df['SortKey'] = rec_df['Action'].apply(lambda x: 0 if 'BUY' in x else 1)
    rec_df = rec_df.sort_values(['SortKey', 'Symbol'])
    del rec_df['SortKey']
    
    print(rec_df.to_string(index=False))

    
    print("\n=== 风控提示 ===")
    risk_manager = RiskManager(config)
    print(f"当前策略止损设置: -{risk_manager.stop_loss_pct:.1%}")
    print(f"当前策略移动止盈: -{risk_manager.trailing_stop_pct:.1%}")
    print("建议：请根据个人风险承受能力，在下单时设置相应的止损单。")

if __name__ == "__main__":
    main()
