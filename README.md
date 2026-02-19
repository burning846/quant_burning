
<div align="center">

# 🔥 Quant Burning | 量化燃烧

### 企业级多市场量化交易与分析系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Market](https://img.shields.io/badge/Market-US%20%26%20CN-orange?style=for-the-badge)](https://finance.yahoo.com/)

[功能特性](#-核心功能) • [快速开始](#-快速开始) • [项目结构](#-项目结构) • [可视化](#-可视化展示)

</div>

---

## 📖 项目简介

**Quant Burning** 是一个现代化的量化交易研究平台，旨在为宽客提供从数据获取、策略开发、回测验证到实盘模拟的一站式解决方案。

系统核心基于 Microsoft Qlib 思想构建，但进行了轻量化重构，特别针对**美股市场 (US Stock)** 进行了深度优化，集成了 **Yahoo Finance** 自动复权数据、**Streamlit** 交互式仪表盘以及**专业级风控模块**。

无论你是量化初学者还是资深交易员，Quant Burning 都能助你燃烧数据，提炼阿尔法！

## 🚀 核心功能

### 1. 🌍 全球市场支持
- **美股 (US)**: 深度集成 `yfinance`，支持自动复权、拆股调整。内置 AAPL, NVDA, TSLA, SPY 等热门标的池。
- **A股 (CN)**: 支持 Tushare/AKShare 数据源（模块化预留）。

### 2. 🛡️ 机构级风控体系
内置 `RiskManager` 模块，为你的资金保驾护航：
- **🛑 固定止损 (Stop Loss)**: 单笔亏损超过阈值（如 7%）自动平仓。
- **💰 固定止盈 (Take Profit)**: 盈利达到目标（如 20%）自动落袋为安。
- **📉 移动止盈 (Trailing Stop)**: 利润回撤超过设定比例（如 10%）自动离场，保住胜利果实。
- **💥 账户熔断 (Circuit Breaker)**: 净值回撤触及警戒线（如 20%）强制清仓。

### 3. 📊 交互式分析仪表盘
基于 Streamlit 打造的现代化 Web UI (`dashboard.py`)：
- **行情概览**: 实时监控核心股票池涨跌幅。
- **深度分析**: 交互式 K 线图，叠加 MACD, RSI, Bollinger Bands 等技术指标。
- **回测实验室**: 无需写代码，通过滑块调整参数，即时查看策略表现。

### 4. 🧠 智能策略库
内置多种经典策略实现：
- **Momentum**: 动量策略，追涨杀跌。
- **Mean Reversion**: 均值回归，捕捉超跌反弹。
- **Moving Average Cross**: 均线交叉，趋势跟踪。
- **RSI**: 超买超卖反转策略。

### 5. 🛠️ 实用工具箱
- **`recommend.py`**: 每日交易决策助手，生成“买入/持有/卖出”信号日报。
- **`track.py`**: 个股历史回溯工具，可视化复盘策略买卖点。

## 📦 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/quant_burning.git
cd quant_burning

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 配置 Python 环境
# 推荐使用 Python 3.8+
```

## 🎮 快速开始

### 方式一：可视化仪表盘 (推荐)
无需敲代码，直接启动 Web 界面：
```bash
streamlit run dashboard.py
```
*浏览器将自动打开，尽情探索行情与回测！*

### 方式二：命令行工具

**1. 获取明日交易建议**
```bash
python recommend.py --config config/config_us.yaml
```

**2. 个股深度复盘**
```bash
# 分析 NVDA 过去一年的策略表现
python track.py NVDA --config config/config_us.yaml --days 365
```

**3. 批量策略回测**
```bash
python run.py --config config/config_us.yaml
```

## 📂 项目结构

```text
quant_burning/
├── backtest/               # 回测核心模块
│   ├── backtest_engine.py  # 回测引擎
│   ├── risk_manager.py     # 风控管理器 (止损/止盈/熔断)
│   └── analyzer.py         # 绩效分析与绘图
├── config/                 # 配置文件
│   ├── config.yaml         # 默认配置
│   └── config_us.yaml      # 美股专用配置
├── data/                   # 数据层
│   ├── fetcher.py          # 多源数据获取 (yfinance/tushare)
│   └── processor.py        # 数据清洗与特征计算 (MACD/RSI...)
├── strategies/             # 策略库
│   ├── base_strategy.py    # 策略基类
│   └── simple_strategies.py # 预置策略实现
├── tests/                  # 测试套件
│   ├── test_risk_manager.py
│   └── test_backtest_integration.py
├── dashboard.py            # Streamlit 可视化仪表盘
├── recommend.py            # 每日推荐脚本
├── track.py                # 个股跟踪脚本
├── run.py                  # 回测入口脚本
└── requirements.txt        # 项目依赖
```

## 📈 可视化展示

> *这里可以放置 dashboard 的截图，展示 K 线图、回测权益曲线等*

## ⚙️ 配置说明

在 `config/config_us.yaml` 中自定义你的交易世界：

```yaml
# 风控参数自定义
risk_management:
  stop_loss_pct: 0.07       # 7% 止损
  take_profit_pct: 0.20     # 20% 止盈
  trailing_stop_pct: 0.10   # 10% 移动止盈
  max_drawdown_limit: 0.20  # 20% 账户熔断

# 回测参数
backtest:
  strategy: "Momentum"      # 默认策略
  commission_rate: 0.0005   # 佣金费率
```

## ⚠️ 免责声明

本项目仅供量化交易学习与研究使用。实盘交易存在巨大风险，作者不对任何投资损失负责。代码中的策略与参数仅作演示，不构成投资建议。

---

<div align="center">
  <p>Made with ❤️ by Quant Burning Team</p>
  <p>
    <a href="https://github.com/yourusername/quant_burning/stargazers">
      <img src="https://img.shields.io/github/stars/yourusername/quant_burning?style=social" alt="GitHub stars">
    </a>
  </p>
</div>
