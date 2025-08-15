🧠 Day 9 (v2) — Risk Parity Portfolio: Inverse-Vol & Equal Risk Contribution (ERC)
🚀 What’s New in v2

✅ Two modes: ivol (inverse volatility) and erc (equal risk contribution)

🧠 ERC mode uses Ledoit–Wolf shrinkage covariance for more stable estimates (requires scikit-learn)

🛡️ Weight caps (--w_cap) to limit overconcentration (e.g., max 25% in any asset)

🧱 Volatility floors (--vol_floor) to avoid divide-by-zero

🌀 Rebalance band (--band): avoids minor turnover when weight changes are small

📉 Choice of vol estimation method: rolling or EWMA in ivol mode

✅ Example Usage
# Inverse-volatility mode with EWMA vol and weight cap
python day09_risk_parity_v2.py --mode ivol --window 63 --reb M --vol_method ewma --vol_floor 0.01 --w_cap 0.25 --band 0.05

# Equal Risk Contribution with Ledoit-Wolf shrinkage, weight cap, rebalance band
python day09_risk_parity_v2.py --mode erc --window 126 --reb M --w_cap 0.25 --band 0.05

💼 Metrics Output

=== Day 9 — Risk Parity Portfolio (ivol or erc) ===
CAGR: 0.12
Vol: 0.09
Sharpe: 0.84
MaxDD: -0.07
Hit%: 0.54

🧾 Files Generated

outputs/day09_daily_returns.csv

outputs/day09_weights_daily.csv

charts/day09_equity.png

# Day 9 — Risk Parity vs Equal Weight

**What**  
Risk parity allocates more weight to low-volatility assets and less to high-volatility assets so each contributes similar risk. We compare it to a simple equal-weight baseline.

**How**  
- Daily returns from `yfinance` adjusted close  
- Rolling volatility (e.g., 63 trading days)  
- Rebalance monthly/quarterly  
- Weights at each rebalance: `w_i ∝ 1 / vol_i` (normalized to sum=1)  
- Transaction costs (bps on turnover) + cash drag (rf)  
- Charts + CSV outputs

---

## Quickstart (Windows / PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python day09_risk_parity.py

---

## Custom example

python day09_risk_parity.py ^
  --tickers XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD ^
  --start 2005-01-01 ^
  --reb M ^
  --window 63 ^
  --rf 0.02 ^
  --tc_bps 10

Outputs

outputs/day09_daily_returns_ew_vs_rp.csv — daily returns (EW vs RP)

outputs/day09_metrics_ew_vs_rp.csv — metrics table

charts/day09_eq_ew_vs_rp.png — equity curves

charts/day09_rp_weights_last12m.png — RP weight stack (last 12 months)

Notes

'M' / 'Q' safely mapped to 'ME' / 'QE' to avoid pandas warnings.

If a ticker has near-zero recent volatility, it will naturally get a tiny weight.

Risk parity smooths volatility; CAGR may be lower than EW in strong bull markets.

Disclaimer: Research/education only. Not investment advice.