# Day 10 — Factor Timing via Information Coefficient (IC)

**Idea**  
Use the **Information Coefficient (IC)** of our cross-sectional momentum signal to **dial exposure up/down**. When IC is positive and rising, stay risk-on; when IC turns negative, cut or go to cash.

## Features
- Signal: **12-1 momentum (1-month skip)**, rank across the ETF universe
- Timing metric: **monthly IC** = Spearman rank-corr between **signal ranks** and **next-month returns**
- Smoothing: **EMA(ic, span=k)**
- Exposure map (default):  
  - `IC > +0.05` → **1.0**  
  - `−0.05 ≤ IC ≤ +0.05` → **0.5**  
  - `IC < −0.05` → **0.0**
- Costs: transaction costs (bps of turnover) + cash drag (annual RF)
- Artifacts: CSVs for returns/IC/exposure, equity & IC charts

## Quickstart (Windows / PowerShell)

python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python day10_factor_timing.py --top 3 --reb M --lookback 252 --skip 21 --fwd 21 --ic_span 6 --upper 0.05 --lower -0.05

# Files

outputs/day10_returns_base_vs_timed.csv — daily returns (Base vs Timed)

outputs/day10_metrics_base_vs_timed.csv — metric table

outputs/day10_ic_series.csv — monthly IC

outputs/day10_exposure_series.csv — exposure per rebalance month

charts/day10_eq_base_vs_timed.png — equity curves

charts/day10_ic_series.png — IC + EMA and thresholds

# How it works (short)

Compute momentum(12-1) → cross-sectional ranks at each rebalance.

Compute forward 1-month returns; take Spearman corr across names → IC per month.

Smooth IC with EMA → map to exposure {1.0, 0.5, 0.0}.

Multiply exposure by daily strategy PnL to get Timed returns; compare to Base.

# Tuning tips

More time “on”: --upper 0.02 --lower -0.02

Faster reaction: --ic_span 3

Binary risk: change mapping to {1, 0} if half-exposure drags

Combine overlays: apply Day 8 Vol Targeting on top of Timed returns