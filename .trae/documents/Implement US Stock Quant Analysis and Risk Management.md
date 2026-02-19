# Implementation Plan: US Stock Quant Analysis & Risk Management

I will enhance the current framework to support US stock analysis, tracking, trading recommendations, and active risk control.

## 1. Configuration Updates (`config/config.yaml`)
- **Add Risk Control Settings**: Define thresholds for active risk management.
  - `max_drawdown_limit`: 20% (Force liquidation if portfolio drops this much).
  - `stop_loss_limit`: 10% (Sell specific stock if it drops this much from entry).
- **Add Tracking Settings**: Define default US stocks to track.
  - `tracking_symbols`: ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'META'].
  - `data_source`: 'yfinance'.

## 2. Risk Management Module (`backtest/risk_manager.py`)
- **Create `RiskManager` class**:
  - **State**: Tracks `peak_value` (for drawdown) and `entry_prices` (for stop-loss).
  - **Logic**:
    - `check_global_risk(current_portfolio_value)`: Checks for max drawdown.
    - `check_position_risk(positions, current_prices)`: Checks individual stop-loss.
    - Returns override signals (e.g., "FORCE_SELL") to the engine.

## 3. Backtest Engine Integration (`backtest/backtest_engine.py`)
- **Integrate `RiskManager`**:
  - Initialize `RiskManager` with config parameters.
  - **In Daily Loop**:
    1. Update Risk Manager with current portfolio value and prices.
    2. Check for Risk Triggers (Drawdown/Stop-loss).
    3. **Priority Execution**: If risk triggers exist, execute "Risk Control Trades" immediately.
    4. Proceed with Strategy Signals only if no global risk lockout.

## 4. Tracking & Analysis Script (`run_analysis.py`)
- **Create new script** to support "Tracking Analysis" and "Decision Recommendations".
- **Workflow**:
  1. **Fetch Data**: Get latest data (up to today) for `tracking_symbols` using `yfinance`.
  2. **Run Strategy**: Apply the configured strategy (e.g., Momentum or MovingAverage) on the latest data window.
  3. **Risk Assessment**: Calculate current drawdown and risk metrics based on a hypothetical entry or recent history.
  4. **Report Generation**: Output a clear console table:
     - **Ticker**: Symbol (e.g., AAPL).
     - **Price**: Latest Close.
     - **Signal**: BUY / SELL / HOLD (from Strategy).
     - **Risk Status**: Safe / Warning / Stop-Loss.
     - **Recommendation**: Final action suggestion.

## 5. Execution & Verification
- **Install Dependencies**: Ensure `yfinance` is available.
- **Test Backtest**: Run `run.py` to verify the backtest engine still works with the new Risk Manager (passive verification).
- **Test Analysis**: Run `run_analysis.py` to generate a live report for US stocks.
