# 美股量化分析与风控系统使用指南

本项目已升级支持美股市场分析、交易决策推荐及主动风控管理。

## 1. 环境准备

确保已安装 Python (3.8+) 并安装以下依赖：

```bash
pip install -r requirements.txt
```

**注意**: `qlib` 可能需要 Microsoft C++ Build Tools 才能在 Windows 上正确安装。如果安装困难，可以尝试使用预编译的 whl 包或在 WSL/Docker 环境中运行。

## 2. 配置文件

美股专用配置文件位于 `config/config_us.yaml`。
主要配置项：
- **数据源**: `yfinance` (自动复权)
- **股票池**: AAPL, MSFT, NVDA, TSLA, SPY, QQQ 等
- **风控参数**:
  - `stop_loss_pct`: 止损阈值 (默认 7%)
  - `trailing_stop_pct`: 移动止盈 (默认 10%)
  - `max_drawdown_limit`: 账户熔断 (默认 20%)

## 3. 功能模块

### A. 可视化仪表盘 (`dashboard.py`) **[NEW]**

提供交互式的 Web 界面，整合行情概览、个股深度分析和策略回测实验室。

**运行方式**:
```bash
streamlit run dashboard.py
```

功能包括：
- **行情概览**: 实时查看关注股票的涨跌幅。
- **个股分析**: 交互式 K 线图，叠加 RSI、MACD 等技术指标。
- **策略回测**: 在线调整策略参数（如均线周期、RSI阈值）并即时查看回测结果。

### B. 每日交易推荐 (`recommend.py`)

获取最新行情并生成“买入/持有/卖出”建议。

**运行方式**:
```bash
python recommend.py --config config/config_us.yaml
```

**输出示例**:
```text
=== 交易推荐报告 (2024-05-20) ===
策略: Momentum
Symbol  Price   Action      Indicators
NVDA    947.80  BUY / HOLD  MA5: 940.12, Momentum: 15.2%
AAPL    191.04  SELL / AVOID MA5: 189.50, Momentum: -2.1%
```

### B. 个股跟踪分析 (`track.py`)

对指定股票进行深度分析，生成包含价格、技术指标和策略买卖点的可视化图表。

**运行方式**:
```bash
# 分析 AAPL 过去一年的表现
python track.py AAPL --config config/config_us.yaml --days 365
```

图表将保存为 `tracking_AAPL_StrategyName.png`。

### C. 历史回测 (`run.py`)

使用历史数据验证策略和风控效果。

**运行方式**:
```bash
python run.py --config config/config_us.yaml
```

回测结果将保存在 `results/` 目录下，包含详细的交易记录和性能指标。

### D. 风控模块 (`backtest/risk_manager.py`)

集成在回测引擎中，自动执行：
1. **个股止损**: 亏损超过设定比例自动平仓。
2. **移动止盈**: 从高点回撤超过设定比例自动止盈。
3. **账户熔断**: 账户总净值回撤过大时强制清仓保护本金。

## 4. 测试与验证

本项目包含单元测试以确保风控模块的可靠性。

**运行测试**:
```bash
python -m unittest tests/test_risk_manager.py
```

## 5. 常见问题

- **数据获取失败**: 确保网络能访问 Yahoo Finance。如果 `yfinance` 下载失败，脚本会尝试使用本地缓存。
- **Qlib 报错**: 确保 `qlib` 已正确初始化。如果是数据问题，尝试删除 `data/qlib_data_us` 目录重新生成。
