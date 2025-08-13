# Python for Quant — Learning in Public (Days 3–5)

Straightforward, reproducible quant projects. Clean code, charts, CSV outputs. Built as part of my doctoral work in Finance and my transition into quantitative trading.

## Repo Structure
python-for-quant/
├─ day03_backtest/
│ ├─ backtest.py
│ ├─ README.md
│ ├─ requirements.txt
│ └─ charts/, outputs/
├─ day04_xs_momentum/
│ ├─ xs_momentum.py
│ ├─ README.md
│ ├─ requirements.txt
│ └─ charts/, outputs/
├─ day05_walkforward/
│ ├─ day05_walkforward.py
│ ├─ README.md
│ ├─ requirements.txt
│ └─ charts/, outputs/
└─ README.md ← (this file)


## Quickstart (Windows / PowerShell)
> Each day has its **own** `requirements.txt`. Activate a venv per day or reuse one and install the day’s requirements before running.

```bash
# Example: run Day 4
cd day04_xs_momentum
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python xs_momentum.py

Projects
Day 3 — Multi-Asset Equal-Weight Backtest
What: Equal-weight portfolio with monthly/quarterly rebalance, transaction costs, cash drag, optional vol targeting.

Run:
cd day03_backtest
. .venv\Scripts\Activate.ps1   # or create one as above
pip install -r requirements.txt
python backtest.py
Outputs: charts/portfolio_eqcurve.png, outputs/*.csv

Docs: Day 3 README

Day 4 — Cross-Sectional Momentum (12–1)
What: Rank universe by 12-1 momentum (skip 1 month), long top-N, equal weight. Costs + cash drag.

Run:
cd day04_xs_momentum
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python xs_momentum.py
Outputs: charts/xs_mom_equity.png, outputs/*.csv

Docs: Day 4 README

Day 5 — Walk-Forward Testing (Out-of-Sample)
What: Rolling train → test → roll. Pick best top-N in-sample by Sharpe, apply OOS. Stitches all OOS periods and reports aggregate results.

Run (default 5y/1y, monthly):
cd day05_walkforward
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python day05_walkforward.py

Custom example:

python day05_walkforward.py --train_years 6 --test_years 2 --top_grid 3,5,7 --reb Q --rf 0.03 --tc_bps 15

Debug mode (see picks & contributions per rebalance):
python day05_walkforward.py --debug

Outputs:

charts/wf_oos_equity.png (stitched OOS equity)

outputs/wf_period_results.csv (per-period metrics + chosen top-N)

outputs/wf_oos_daily_returns.csv (stitched OOS returns)

Docs: Day 5 README

Tips
Charts in README: After running a day once, commit the charts/*.png so GitHub renders previews inside each day’s README.

No pandas warnings: Code maps M→ME, Q→QE under the hood (safe_freq).

Repro: Data via yfinance (adjusted close). For papers, pin versions or cache data.

Roadmap (Next)
Day 6: Parameter stability heatmaps (across time + universes)

Day 7: Volatility filter / regime switch

Day 8: Transaction-cost sensitivity & slippage models

Day 9: Cross-asset combo (momentum + carry)

Day 10: Simple risk parity wrapper

Research Context
This series supports my doctoral research in Finance on how systematic methods and disciplined validation (e.g., walk-forward) improve portfolio robustness. It’s also my public build toward quant trading and research roles.
