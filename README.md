# 基于qlib的量化交易系统

## 项目简介

这是一个基于Microsoft qlib框架构建的简单量化交易系统，提供了从数据获取、处理、模型训练到策略回测的完整流程。系统设计模块化，易于扩展和定制。现已支持中国A股和美股市场数据。

## 项目结构

```
.
├── backtest/               # 回测模块
│   ├── __init__.py
│   ├── backtest_engine.py  # 回测引擎
│   └── analyzer.py         # 回测结果分析器
├── config/                 # 配置文件
│   └── config.yaml         # 系统配置
├── data/                   # 数据模块
│   ├── __init__.py
│   ├── fetcher.py          # 数据获取
│   ├── processor.py        # 数据处理
│   ├── raw/                # 原始数据存储
│   └── processed/          # 处理后数据存储
├── examples/               # 示例代码
│   └── backtest_example.py # 回测示例
├── models/                 # 模型模块
│   ├── __init__.py
│   ├── base_model.py       # 基础模型类
│   ├── ml_model.py         # 机器学习模型
│   ├── dl_model.py         # 深度学习模型
│   └── saved/              # 保存的模型
├── notebooks/              # Jupyter笔记本
├── results/                # 结果输出
│   ├── __init__.py
│   └── visualizer.py       # 可视化工具
├── strategies/             # 策略模块
│   ├── __init__.py
│   ├── base_strategy.py    # 基础策略类
│   └── simple_strategies.py # 简单策略实现
├── __init__.py             # 包初始化
├── requirements.txt        # 依赖项
├── run.py                  # 主运行脚本
└── README.md               # 项目说明
```

## 功能特点

1. **数据获取与处理**
   - 支持从多种数据源获取股票数据（Yahoo Finance等）
   - 支持中国A股和美股市场数据
   - 数据清洗、特征工程和标准化
   - 生成qlib格式数据集

2. **模型构建**
   - 支持多种机器学习模型（随机森林、XGBoost、线性回归等）
   - 支持深度学习模型（LSTM、MLP等）
   - 模型训练、评估和保存

3. **策略实现**
   - 动量策略
   - 均值回归策略
   - 移动平均线交叉策略
   - RSI策略
   - 易于扩展自定义策略

4. **回测系统**
   - 灵活的回测引擎
   - 支持多种交易频率
   - 考虑交易成本和滑点
   - 详细的回测结果分析

5. **性能评估**
   - 年化收益率 (Annual Return)
   - 夏普比率 (Sharpe Ratio)
   - 最大回撤 (Maximum Drawdown)
   - 贝塔系数 (Beta)
   - 阿尔法 (Alpha)
   - 信息比率 (Information Ratio)
   - 索提诺比率 (Sortino Ratio)
   - 卡玛比率 (Calmar Ratio)
   - 欧米茄比率 (Omega Ratio)
   - 特雷诺比率 (Treynor Ratio)
   - 尾部比率 (Tail Ratio)
   - 风险价值 (Value at Risk)
   - 条件风险价值 (Conditional Value at Risk)
   - 最大回撤持续时间 (Max Drawdown Duration)
   - 偏度 (Skewness)
   - 峰度 (Kurtosis)
   - 捕获比率 (Capture Ratio)

6. **结果分析与可视化**
   - 权益曲线
   - 收益率分布图
   - 回撤分析
   - 月度收益热力图
   - 滚动夏普比率
   - 多策略性能比较

## 安装与配置

### 环境要求

- Python 3.7+
- 依赖包见requirements.txt

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/quant_burning.git
cd quant_burning
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置系统

编辑 `config.yaml` 文件，设置数据源和股票列表。可以选择中国A股市场(REG_CN)或美股市场(REG_US)。

```yaml
# 示例配置
qlib:
  provider_uri: "~/.qlib/qlib_data/cn_data"  # 或使用美股数据路径
  region: REG_US  # 使用美股，改为REG_CN使用A股

data_source: "yahoo"  # 数据源

default_symbols:  # 默认股票列表
  - AAPL
  - MSFT
  - AMZN
  - GOOGL
  - META
```

编辑`config.yaml`文件，根据需要修改配置参数。

## 使用示例

### 运行回测示例

```bash
python examples/backtest_example.py
```

### 运行可视化示例

```bash
python examples/visualization_example.py
```

### 自定义策略

1. 在 `strategies` 目录下创建新的策略类，继承 `BaseStrategy`
2. 实现 `generate_signals` 方法
3. 在 `strategy_factory.py` 中注册新策略
4. 在回测脚本中使用新策略

## 使用指南

### 运行示例回测

```bash
python examples/backtest_example.py
```

### 使用主程序运行

```bash
python run.py
```

### 自定义策略

1. 在`strategies`目录下创建新的策略文件
2. 继承`BaseStrategy`类并实现必要的方法
3. 在配置文件中指定使用新策略

### 自定义模型

1. 在`models`目录下创建新的模型文件
2. 继承`BaseModel`类并实现必要的方法
3. 在配置文件中指定使用新模型

## 示例代码

### 数据获取

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_stock_data('000001.SZ', '2020-01-01', '2021-01-01', source='yahoo')
print(data.head())
```

### 策略回测

```python
from data.processor import DataProcessor
from strategies.simple_strategies import MovingAverageCrossStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.analyzer import BacktestAnalyzer

# 处理数据
processor = DataProcessor()
processed_data = processor.process_stock_data(data)

# 创建策略
strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)

# 创建回测引擎
backtest_config = {
    'start_date': '2020-01-01',
    'end_date': '2021-01-01',
    'benchmark': '000001.SZ',
    'account': 1000000
}
backtest_engine = BacktestEngine(backtest_config)

# 运行回测
results = backtest_engine.run_backtest(strategy, processed_data)

# 分析结果
analyzer = BacktestAnalyzer(results)
analyzer.generate_report()
```

## 注意事项

- 本系统仅用于学习和研究目的，不构成投资建议
- 实际交易中需考虑更多因素，如流动性、市场冲击等
- 回测结果不代表未来表现

## 扩展计划

- 添加更多数据源支持
- 实现更多策略和模型
- 添加实时交易接口
- 优化回测性能
- 添加投资组合优化功能

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情见 [LICENSE](LICENSE) 文件