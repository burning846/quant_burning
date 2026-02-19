
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
import os
from datetime import datetime, timedelta

from data.fetcher import DataFetcher
from data.processor import DataProcessor
from strategies.simple_strategies import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossStrategy, RSIStrategy

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_strategy(strategy_name, config):
    # 复用 recommend.py 中的逻辑
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

def plot_tracking_analysis(data, signals, symbol, strategy_name, output_file=None):
    """
    绘制跟踪分析图表
    """
    # 设置绘图风格
    plt.style.use('bmh') # 使用内置样式，避免seaborn依赖
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 准备数据
    dates = data['date']
    prices = data['close']
    
    # 主图：价格与均线
    ax1.plot(dates, prices, label='Close Price', color='black', alpha=0.7)
    
    # 如果是均线策略，画出均线
    if strategy_name == 'MovingAverageCross':
        if 'ma5' in data.columns:
            ax1.plot(dates, data['ma5'], label='MA5', linestyle='--', alpha=0.6)
        if 'ma20' in data.columns:
            ax1.plot(dates, data['ma20'], label='MA20', linestyle='--', alpha=0.6)
            
    # 标记买卖信号
    # signals DataFrame 应该包含 'date', 'signal' (1, -1, 0)
    # 需要将signals对齐到data
    
    # 提取买入和卖出点
    buy_signals = []
    sell_signals = []
    
    # 简单的对齐逻辑：假设signals包含所有日期的信号或者按日期对应
    # 在 simple_strategies.py 中，generate_signals 返回的 DataFrame 结构不尽相同
    # MovingAverageCross 和 RSI 返回包含 'date' 的 list -> DataFrame
    # Momentum 和 MeanReversion 返回只包含 stock_id 和 signal 的 snapshot (这就尴尬了，不支持历史信号)
    
    # 我们需要修改/适配 strategies 使得它们能返回历史信号序列，或者在这里重新计算
    # 幸运的是，MovingAverageCross 和 RSI 在 simple_strategies.py 的实现里是返回了时间序列信号的
    # 而 Momentum 目前的实现只返回了基于最新数据的 snapshot (returns dict)。
    
    # 针对 Momentum/MeanReversion，我们需要模拟历史信号生成（Rolling）
    # 这里为了演示，我们主要支持 MA 和 RSI 的完整可视化
    
    if 'date' in signals.columns:
        # 合并信号
        merged = pd.merge(data, signals[['date', 'stock_id', 'signal']], on=['date', 'stock_id'], how='left').fillna(0)
        
        buys = merged[merged['signal'] == 1]
        sells = merged[merged['signal'] == -1]
        
        ax1.scatter(buys['date'], buys['close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sells['date'], sells['close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title(f'{symbol} Tracking Analysis - Strategy: {strategy_name}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 副图：指标
    if strategy_name == 'RSI' and 'rsi' in signals.columns:
        # 需要把RSI数据合并进来
        merged_rsi = pd.merge(data, signals[['date', 'stock_id', 'rsi']], on=['date', 'stock_id'], how='left')
        ax2.plot(merged_rsi['date'], merged_rsi['rsi'], label='RSI', color='purple')
        ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
    else:
        # 默认画成交量
        ax2.bar(dates, data['volume'], color='gray', alpha=0.5, label='Volume')
        ax2.set_ylabel('Volume')
        
    ax2.legend()
    ax2.grid(True)
    
    # 格式化日期轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='个股跟踪分析工具')
    parser.add_argument('symbol', type=str, help='股票代码 (e.g., AAPL)')
    parser.add_argument('--config', type=str, default='config/config_us.yaml', help='配置文件路径')
    parser.add_argument('--days', type=int, default=365, help='分析天数')
    parser.add_argument('--output', type=str, default=None, help='保存图表路径')
    args = parser.parse_args()
    
    print(f"Analyzing {args.symbol}...")
    config = load_config(args.config)
    
    # 获取数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    fetcher = DataFetcher()
    # 强制单只股票获取
    # 这里我们借用 batch_fetch 但只传一个
    try:
        raw_data = fetcher.batch_fetch_stock_data([args.symbol], start_date, end_date, source=config['data']['default_source'])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
        
    if raw_data.empty:
        print(f"No data found for {args.symbol}")
        return
        
    # 处理数据
    processor = DataProcessor()
    data = processor.process_stock_data(raw_data)
    
    # 运行策略
    strategy_name = config['backtest']['strategy']
    print(f"Applying Strategy: {strategy_name}")
    
    strategy = get_strategy(strategy_name, config)
    
    # 滚动生成信号 (Rolling Signal Generation)
    # 由于部分策略只支持生成快照信号，我们需要模拟回测过程逐日生成
    print("Generating rolling signals...")
    signals_list = []
    
    # 获取需要回看的最小窗口期
    # 为了简化，我们假设至少需要50天数据
    min_window = 50
    
    if len(data) > min_window:
        for i in range(min_window, len(data)):
            # 获取截止到当日的数据切片
            # 注意：这在Python循环中可能较慢，但对于单只股票分析是可以接受的
            current_data = data.iloc[:i+1]
            current_date = current_data.iloc[-1]['date']
            
            # 生成信号
            try:
                # 某些策略可能需要重新初始化或reset状态，但简单策略通常是无状态的或者只依赖传入的数据
                # 这里假设 generate_signals 是无副作用的
                daily_signals = strategy.generate_signals(current_data)
                
                # 提取针对当前股票的信号
                # daily_signals 可能是包含多只股票的 DataFrame
                if not daily_signals.empty:
                    if 'date' in daily_signals.columns:
                        # 如果策略返回带日期的信号，取最后一行
                        latest_signal = daily_signals[daily_signals['date'] == current_date]
                    else:
                        # 如果策略只返回快照（如 Momentum），则只有一行，即为最新信号
                        latest_signal = daily_signals[daily_signals['stock_id'] == args.symbol]
                    
                    if not latest_signal.empty:
                        sig_val = 0
                        if 'signal' in latest_signal.columns:
                            sig_val = latest_signal.iloc[0]['signal']
                        
                        # 如果有信号 (1 或 -1)，或者是 0 但我们想记录状态
                        if sig_val != 0:
                            signals_list.append({
                                'date': current_date,
                                'stock_id': args.symbol,
                                'signal': sig_val,
                                # 如果有其他指标（如RSI），也可以提取
                                'rsi': latest_signal.iloc[0].get('rsi', None)
                            })
                            
            except Exception as e:
                # 忽略某些日期的错误
                pass
    
    signals = pd.DataFrame(signals_list)
    
    # 绘制图表
    output_file = args.output if args.output else f"tracking_{args.symbol}_{strategy_name}.png"
    plot_tracking_analysis(data, signals, args.symbol, strategy_name, output_file)

if __name__ == "__main__":
    main()
