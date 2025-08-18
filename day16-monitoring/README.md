# Day 16 — Live Monitoring & Paper-Trading Harness

Turn the Day 15 strategy into a daily routine that **pulls prices**, computes **target weights** (inverse-vol + regime-aware vol targeting), generates **orders**, **paper-fills** them (spread + simple impact), updates a **positions ledger**, and exports a **daily report** (CSV + PNG).

## Quickstart

python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt

# From inside day16-monitoring/
python day16_monitor.py --tickers XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD --start 2018-01-01
Outputs per run:

orders/orders_YYYY-MM-DD.csv – what you would have traded

live/positions.csv – paper book (shares per ticker + cash)

outputs/day16_metrics_YYYY-MM-DD.csv – one-page metrics (CAGR, Vol, Sharpe, MaxDD, NAV)

charts/day16_report_YYYY-MM-DD.png – paper NAV + exposure scaler + hostile regime timeline

logs/day16_run_YYYY-MM-DD.txt – run notes

How it works
Weights: Monthly inverse-volatility (risk-parity-lite) with optional --w_cap and --alloc_band.

Regime overlay: compute hostile when ^VIX or ^TNX is above its EMA (spans configurable).
Reduce the volatility target by --reduce in hostile periods.
Exposure scaler s = clip(target / realized_vol, vt_min, vt_max) (capped at 1.0, no leverage in paper).

Orders: Compare current weights (from live/positions.csv) to targets.
Skip if L1 drift < --no_trade_band. Cap daily L1 turnover with --turnover_cap.

Paper fills: Fill at close ± half-spread (--spread_bps) ± impact (--impact_coef * ATR).

NAV & report: Append positions for NAV history, compute metrics, write a PNG dashboard.

Common flags
--reb M|Q – rebalance calendar for weights (default: M)

--vt_target 0.10 – annualized vol target

--vt_span 63 – span for realized vol

--vt_min 0.5 --vt_max 1.0 – bounds for exposure scaler (1.0 = no leverage in paper)

--use_tnx/--use_vix --tnx_span --vix_span --host_or – regime detection toggles

--no_trade_band 0.01 – skip small drifts

--turnover_cap 0.20 – cap daily L1 weight change

--spread_bps 2 --impact_coef 0.10 – paper microstructure

Notes
This is a paper harness. When you’re happy with stability, you can map orders_*.csv to a broker API (IBKR, Alpaca, TradeStation, etc.) in a later day (execution layer).

If ^VIX or ^TNX isn’t available, the script falls back to a benign regime (no reduction).