# Python for Quant Trading — Learning Journey

This repository documents my hands-on journey from finance management into quantitative trading & research, leveraging Python, statistical modeling, and strategy backtesting.

---

## 📅 Day 3 — First Momentum Backtest
- **Goal:** Build a simple cross-sectional momentum strategy using historical price data.
- **What I learned:**
  - Download & preprocess stock data
  - Calculate momentum signals
  - Equal-weight portfolio construction
  - Backtest performance metrics
- **Key output:** Equity curve showing momentum vs. benchmark

---

## 📅 Day 4 — Cross-Sectional Momentum Refinement
- **Goal:** Improve the Day 3 strategy by refining the ranking & portfolio weighting logic.
- **What I learned:**
  - Handle submodules & repository structure in Git
  - More robust signal ranking and portfolio rebalancing
  - How to avoid Git pitfalls when working across multiple project folders
- **Key output:** Cleaner, more accurate equity curve with improved rebalance logic

---

## 📅 Day 5 — Equal-Weight Portfolio vs Buy & Hold
- **Goal:** Compare a regularly rebalanced equal-weight portfolio against a Buy & Hold strategy (net of costs & cash).
- **What I learned:**
  - Side-by-side performance analysis
  - Implementing transaction cost assumptions
  - Measuring impact of rebalancing on returns
- **Key output:** Overlaid equity curves — rebalanced portfolio vs buy & hold

---

## 📅 Day 6 — Grid Search + Walk-Forward Optimization
- **Goal:** Automate hyperparameter tuning for strategy lookback, skip periods, and top-N asset selection.
- **What I learned:**
  - Grid search to find best parameter combinations
  - Walk-forward validation to reduce overfitting risk
  - How to interpret performance tables from multiple training periods
- **Key output:** Optimal strategy parameters for each walk-forward segment

---

## 📅 Day 7 — Multi-Factor Strategy (Momentum + Volatility Filter)
- **Goal:** Combine multiple factors into a composite signal and test its predictive power.
- **What I learned:**
  - Calculate an **Information Coefficient (IC)** between signals & forward returns
  - Filter stocks using both momentum rank and volatility thresholds
  - Build a multi-factor equity curve
- **Performance Metrics:**
  - **CAGR:** 39.48%
  - **Volatility:** 53.53%
  - **Sharpe Ratio:** 0.70
  - **Max Drawdown:** -61.55%
  - **Hit Rate:** 53.31%
- **Key output:**  
  - IC time series chart (signal quality over time)  
  - Multi-factor strategy equity curve

---

## 📂 Outputs
- **Charts:** `/charts/`
- **Data:** `/outputs/`
- **Scripts:** `/dayXX-*` folders
- **Dependencies:** See `requirements.txt`

---

## 🚀 Next Steps
- Factor weighting optimization
- Regime switching logic
- Position sizing based on volatility
- Integration of additional alpha signals

---

**Disclaimer:** Educational purposes only. Not financial advice.
