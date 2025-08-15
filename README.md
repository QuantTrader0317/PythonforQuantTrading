# Python for Quant Trading — Days 3–8

A progressive learning and build-out of Python-based quantitative trading strategies, moving from simple backtests to more advanced multi-factor and volatility targeting approaches.  
Each day builds on previous work, with clean, GitHub-ready code, reproducible results, and charts.

---

## 📅 Day 3 — Equal-Weight Multi-Asset Backtest

**Strategy:**  
- Equal-weighted portfolio of multiple ETFs  
- Monthly or quarterly rebalancing  
- Includes transaction costs & cash drag  
- Outputs equity curve and CSV logs  

**Key Learnings:**  
- Portfolio rebalancing logic  
- Handling OHLCV data with `yfinance`  
- Equity curve plotting  

**Artifacts:**  
- `outputs/daily_returns.csv`  
- `outputs/equity_curves.csv`  
- `outputs/weights_daily.csv`  
- `charts/portfolio_eqcurve.png`  

---

## 📅 Day 4 — Cross-Sectional Momentum (12-1)

**Strategy:**  
- Rank ETFs by 12-month momentum, skip most recent month (12-1)  
- Go long top-N ranked assets  
- Monthly/quarterly rebalancing  
- Transaction costs & cash drag  

**Key Learnings:**  
- Ranking & sorting assets by factor values  
- Avoiding look-ahead bias  
- Outputting picks by rebalance date  

**Artifacts:**  
- `outputs/picks_by_rebalance.csv`  
- `charts/xs_mom_equity.png`

---

## 📅 Day 5 — Walk-Forward Optimization

**Strategy:**  
- Train on rolling windows, test on next period  
- Optimizes `top_n` assets based on train period performance  
- Applies best params to next test period  
- Produces walk-forward equity curve  

**Key Learnings:**  
- Rolling train/test split  
- Param optimization in live-like fashion  
- Handling rebalance logic with changing parameters  

**Artifacts:**  
- `outputs/wf_equity_curves.csv`  
- `charts/wf_portfolio_eqcurve.png`

---

## 📅 Day 6 — Grid Search + Walk-Forward

**Strategy:**  
- Expands Day 5 to test a **grid of parameters** (lookback days, skip days, top_n)  
- Picks combination with highest Sharpe in training set  
- Applies to walk-forward test period  

**Key Learnings:**  
- Grid search over multiple hyperparameters  
- Comparing model performance via metrics  
- Outputting parameter tables  

**Artifacts:**  
- `outputs/gridsearch_results.csv`  
- `charts/gridsearch_eqcurve.png`

---

## 📅 Day 7 — Multi-Factor Strategy

**Strategy:**  
- Combine **momentum** + **value** factors  
- Rank assets by composite score  
- Go long top-N ranked assets each rebalance  

**Key Learnings:**  
- Building and combining multiple factors  
- Information Coefficient (IC) analysis for predictive power  
- Factor performance visualization  

**Results:**  
- CAGR: ~39%  
- Sharpe: ~0.70  
- Max Drawdown: ~-61% (aggressive profile)  

**Artifacts:**  
- `outputs/day07_daily_returns.csv`  
- `charts/day07_equity.png`  
- `charts/day07_ic_series.png`

---

## 📅 Day 8 — Volatility Targeting Overlay

**Strategy:**  
- Take any strategy (e.g., Day 4 momentum)  
- Scale position sizes to target a constant annualized volatility  
- Use rolling standard deviation to adjust leverage up or down  
- Reduces drawdowns, smooths equity curve  

**Key Learnings:**  
- Volatility targeting math  
- Rolling-window volatility estimation  
- Overlaying risk management on existing strategies  

**Artifacts:**  
- `outputs/day08_vol_target_daily_returns.csv`  
- `charts/day08_vol_target_equity.png`  

**Equity Curve:**  
![Vol Target Equity](charts/day08_vol_target_equity.png)

