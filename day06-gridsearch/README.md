# Day 6 — Grid Search + Walk-Forward (XS Momentum)

**Goal**  
Test **multiple parameter combos** (lookback, skip, top-N) and validate with **walk-forward** splits:
- Train window: sweep the grid, pick best by in-sample Sharpe
- Test window: apply that combo out-of-sample
- Stitch all OOS periods and report aggregate performance

**Why**  
Robust strategies don’t win because of one lucky parameter; they hold up **across time**.

---

## Quickstart
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python day06_gridsearch_walkforward.py

Custom grid example:

python day06_gridsearch_walkforward.py ^
  --grid_lookbacks 126,252,378 ^
  --grid_skips 0,21 ^
  --grid_tops 2,3,5 ^
  --reb Q --rf 0.03 --tc_bps 15 --train_years 6 --test_years 2 --debug

Parameters
--grid_lookbacks: momentum lookback days (e.g., 126,189,252)

--grid_skips: skip days to avoid short-term reversal (e.g., 0,21)

--grid_tops: number of assets to hold (e.g., 2,3,4)

Other flags match Day 5 (tickers, start, rf, tc_bps, reb, train/test years)

Outputs
outputs/wf_grid_period_results.csv — per-period OOS metrics + chosen combo

outputs/wf_grid_train_sweeps.csv — full in-sample grid results per period

outputs/wf_grid_oos_daily_returns.csv — stitched OOS returns

charts/wf_grid_oos_equity.png — stitched OOS equity (best-per-period combo)

Notes
Resampling is warning-proof ('M'→'ME', 'Q'→'QE').

Keep grids small and sane to avoid overfitting.

Use --debug to print the top in-sample combos each period.