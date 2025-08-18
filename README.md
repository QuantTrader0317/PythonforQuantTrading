# Python for Quant — Days 1–15 (Core Portfolio Stack)

Build a **rules-based, diversified portfolio** step-by-step. By Day 15 you have:
- A **risk-parity core** (ERC/Inverse-Vol)
- **Volatility targeting**
- **Execution controls** (no-trade band, throttle, turnover caps, costs)
- **Walk-forward validation**
- **Robustness tests** (sensitivity, regimes, jackknife, cost stress, bootstrap CIs)
- A simple **macro/regime switch** using **rates (TNX)** and **vol (VIX)**

> This is educational code, not investment advice. Expect live results to be lower than backtests.

---

## TL;DR Results (what a “healthy” run looks like)
- **Net Sharpe**: ~0.4–0.7 range depending on costs/period
- **Max Drawdown**: materially lower with vol-target + regime switch vs. vanilla EW
- **Turnover**: target ≤ 1–2%/month after bands/throttle
- **Robustness**: Sharpe stays positive under realistic fee/param stress; not reliant on any single ETF

---

## Repo Layout

