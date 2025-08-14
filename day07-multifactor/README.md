# Day 7 — Multi-Factor Cross-Sectional Strategy (Momentum + Volatility Filter)

**Goal**  
Blend multiple signals into one **composite rank** and test it on a cross-sectional ETF universe:
- Factor 1: **12-1 momentum** = (P[t−21] / P[t−252]) − 1
- Factor 2: **3-month volatility** = rolling std(returns, 63d) — lower is better
- Composite: `Score = w_mom * rank(mom) + w_vol * rank(-vol)`

**Why**  
Multi-factor models often beat single-factor by improving robustness and stability across regimes.

---

## Quickstart (Windows / PowerShell)
```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python day07_multifactor.py

Custom Example

python day07_multifactor.py ^
  --top 3 ^
  --w_mom 0.7 --w_vol 0.3 ^
  --weighting inv_vol ^
  --reb M --rf 0.02 --tc_bps 10
Parameters

--tickers : universe (default: sectors + TLT, GLD)

--start : data start date (default 2005-01-01)

--reb : M (monthly) or Q (quarterly) → safely mapped to ME/QE

--rf : annual cash drag (e.g., 0.02)

--tc_bps : transaction cost in bps of turnover (e.g., 10)

--lookback : momentum lookback days (default 252)

--skip : days to skip (default 21)

--vol_window : days for vol (default 63)

--top : number of assets to hold (default 3)

--w_mom, --w_vol : factor weights (auto-normalized)

--weighting : equal or inv_vol

Outputs

outputs/day07_daily_returns.csv — net daily returns

outputs/day07_equity_curve.csv — equity curve

outputs/day07_weights_daily.csv — daily portfolio weights

outputs/day07_picks_by_rebalance.csv — top-N selections per rebalance

outputs/day07_ic_series.csv — Information Coefficient time series

charts/day07_equity.png — equity curve

charts/day07_ic_series.png — IC over time

IC (Information Coefficient)
At each rebalance, we compute the Spearman rank correlation between the composite score and the forward returns to the next rebalance.

Positive IC ⇒ higher scores tended to outperform next period.

IC mean and stability matter more than any one value.

Notes

Uses safe_freq so 'M'→'ME', 'Q'→'QE' (no pandas warnings).

inv_vol weighting uses 63-day rolling vol to scale weights.

Keep factor weights simple (e.g., 0.7 / 0.3). Complexity = overfit risk.

Disclaimer

Research/education only. Not investment advice.