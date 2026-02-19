
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from datetime import datetime, timedelta

# 导入项目模块
from data.fetcher import DataFetcher
from data.processor import DataProcessor
from strategies.simple_strategies import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossStrategy, RSIStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer

# 设置页面配置
st.set_page_config(page_title="美股量化分析仪表盘", layout="wide")

# 加载配置
@st.cache_data
def load_config(config_path='config/config_us.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()

# 侧边栏
st.sidebar.title("控制面板")
selected_page = st.sidebar.radio("选择功能", ["行情概览", "个股分析", "策略回测"])

# 数据获取函数
@st.cache_data
def get_data(symbols, days=365):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    fetcher = DataFetcher()
    # 尝试从yfinance获取
    try:
        raw_data = fetcher.batch_fetch_stock_data(symbols, start_date, end_date, source='yfinance')
        processor = DataProcessor()
        data = processor.process_stock_data(raw_data)
        return data
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

if selected_page == "行情概览":
    st.title("美股行情概览")
    
    # 获取默认股票池
    symbols = config['data']['default_symbols']
    
    if st.button("刷新数据"):
        st.cache_data.clear()
        
    data = get_data(symbols, days=30)
    
    if not data.empty:
        # 展示最新行情
        latest_date = data['date'].max()
        st.subheader(f"最新行情 ({latest_date.strftime('%Y-%m-%d')})")
        
        latest_data = data[data['date'] == latest_date].copy()
        
        # 计算涨跌幅
        # 需要前一天的数据
        prev_date = data[data['date'] < latest_date]['date'].max()
        prev_data = data[data['date'] == prev_date].set_index('stock_id')['close']
        
        latest_data['prev_close'] = latest_data['stock_id'].map(prev_data)
        latest_data['pct_change'] = (latest_data['close'] - latest_data['prev_close']) / latest_data['prev_close']
        
        # 格式化展示
        cols = st.columns(len(symbols))
        for i, symbol in enumerate(symbols):
            row = latest_data[latest_data['stock_id'] == symbol]
            if not row.empty:
                price = row.iloc[0]['close']
                change = row.iloc[0]['pct_change']
                
                with cols[i % 4]: # 每行4个
                    st.metric(
                        label=symbol,
                        value=f"${price:.2f}",
                        delta=f"{change:.2%}"
                    )

elif selected_page == "个股分析":
    st.title("个股深度分析")
    
    symbol = st.sidebar.text_input("输入股票代码", value="AAPL").upper()
    days = st.sidebar.slider("分析天数", 30, 1000, 365)
    
    data = get_data([symbol], days=days)
    
    if not data.empty:
        # K线图与均线
        st.subheader("股价走势")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['date'], data['close'], label='Close')
        if 'ma20' in data.columns:
            ax.plot(data['date'], data['ma20'], label='MA20', linestyle='--')
        if 'ma60' in data.columns:
            ax.plot(data['date'], data['ma60'], label='MA60', linestyle='--')
            
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # 技术指标
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI (14)")
            if 'rsi' in data.columns:
                fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
                ax_rsi.plot(data['date'], data['rsi'], color='purple')
                ax_rsi.axhline(70, color='red', linestyle='--')
                ax_rsi.axhline(30, color='green', linestyle='--')
                ax_rsi.set_ylim(0, 100)
                st.pyplot(fig_rsi)
                
        with col2:
            st.subheader("MACD")
            if 'macd' in data.columns:
                fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
                ax_macd.plot(data['date'], data['dif'], label='DIF')
                ax_macd.plot(data['date'], data['dea'], label='DEA')
                ax_macd.bar(data['date'], data['macd'], label='MACD', alpha=0.3)
                ax_macd.legend()
                st.pyplot(fig_macd)

elif selected_page == "策略回测":
    st.title("策略回测实验室")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_name = st.selectbox("选择策略", ["Momentum", "MeanReversion", "MovingAverageCross", "RSI"])
        initial_capital = st.number_input("初始资金", value=100000)
        
    with col2:
        start_date = st.date_input("开始日期", value=datetime(2020, 1, 1))
        end_date = st.date_input("结束日期", value=datetime.now())
        
    # 策略参数配置
    st.subheader("策略参数")
    params = {}
    if strategy_name == "Momentum":
        params['lookback_period'] = st.slider("回看周期", 5, 60, 20)
        params['holding_period'] = st.slider("持有周期", 1, 20, 5)
    elif strategy_name == "RSI":
        params['rsi_period'] = st.slider("RSI周期", 5, 30, 14)
        params['overbought'] = st.slider("超买阈值", 50, 90, 70)
        params['oversold'] = st.slider("超卖阈值", 10, 50, 30)
    elif strategy_name == "MovingAverageCross":
        params['short_window'] = st.slider("短期均线", 2, 20, 5)
        params['long_window'] = st.slider("长期均线", 10, 100, 20)
        
    if st.button("开始回测"):
        with st.spinner("正在回测中..."):
            # 准备配置
            bt_config = config.copy()
            bt_config['backtest']['strategy'] = strategy_name
            bt_config['backtest']['start_date'] = start_date.strftime('%Y-%m-%d')
            bt_config['backtest']['end_date'] = end_date.strftime('%Y-%m-%d')
            bt_config['backtest']['account'] = initial_capital
            bt_config['backtest'].update(params)
            
            # 获取数据
            symbols = config['data']['default_symbols']
            fetcher = DataFetcher()
            raw_data = fetcher.batch_fetch_stock_data(symbols, bt_config['backtest']['start_date'], bt_config['backtest']['end_date'], source='yfinance')
            processor = DataProcessor()
            data = processor.process_stock_data(raw_data)
            
            # 初始化引擎
            engine = BacktestEngine(bt_config['backtest'])
            
            # 初始化策略
            if strategy_name == 'Momentum':
                strategy = MomentumStrategy(params)
            elif strategy_name == 'RSI':
                strategy = RSIStrategy(params)
            elif strategy_name == 'MovingAverageCross':
                strategy = MovingAverageCrossStrategy(params)
            else:
                strategy = MeanReversionStrategy(params)
                
            # 运行回测
            results = engine.run_backtest(strategy, data)
            analyzer = BacktestAnalyzer(results)
            summary = analyzer.get_summary()
            
            # 展示结果
            st.success("回测完成！")
            
            # 核心指标
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("年化收益率", f"{summary.loc['annual_return', 'value']:.2%}")
            m2.metric("夏普比率", f"{float(summary.loc['sharpe_ratio', 'value']):.2f}")
            m3.metric("最大回撤", f"{summary.loc['max_drawdown', 'value']:.2%}")
            m4.metric("胜率", f"{summary.loc['win_rate', 'value']:.2%}")
            
            # 权益曲线
            st.subheader("权益曲线")
            equity_df = pd.DataFrame({
                'Portfolio': results['portfolio_value'],
                'Benchmark': results['benchmark_value']
            }, index=results['dates'])
            st.line_chart(equity_df)
            
            # 详细数据
            with st.expander("查看详细交易记录"):
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df)
