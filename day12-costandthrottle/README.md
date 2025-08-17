## Day 12 — Transaction Costs & Execution Throttling

**Goal**: Make backtests closer to reality. We compare:
- **Instant / simple bps** — jump to target on rebalance; cost = `tc_bps × turnover`.
- **Throttled / advanced** — partial-to-target with **no-trade band** + **daily turnover cap**; cost = **spread + impact**:
  - Spread/slippage: `spread_bps × turnover`
  - Impact (square-root): `impact_coef × vol_proxy × sqrt(turnover)` where `vol_proxy` is EWMA daily cross-asset vol.

**Execution knobs**
- `--exec_speed` (0–1): fraction of gap traded at each rebalance day (e.g., 0.33).
- `--no_trade_band`: per-asset band (abs weight) to ignore micro changes (e.g., 0.002).
- `--turnover_cap`: daily L1 turnover cap (e.g., 0.20 = 20% of book).
- Costs: `--spread_bps`, `--impact_coef`, `--impact_span` (EWMA window).

**Quickstart**

python day12_costs_and_throttle.py --mode erc --window 126 --reb M --tc_bps 10
python day12_costs_and_throttle.py --mode erc --window 126 --reb M \
  --exec_speed 0.33 --no_trade_band 0.002 --turnover_cap 0.20 \
  --spread_bps 2 --impact_coef 0.10 --impact_span 63
Outputs

outputs/day12_returns.csv — daily returns (instant vs throttled)

outputs/day12_turnover.csv — daily turnover (instant vs throttled)

outputs/day12_weights_targets.csv — allocator targets (monthly, ffilled)

outputs/day12_weights_exec.csv — executed daily weights (throttled)

outputs/day12_metrics.csv — CAGR, Vol, Sharpe, MaxDD, Hit%

charts/day12_equity.png — equity curves

charts/day12_turnover_*.png — 21-day avg turnover

Interpretation

Expect throttled to show lower turnover, lower costs, smoother equity; CAGR can be lower or higher depending on how aggressive your instant trading was and how realistic your cost settings are. The point is control.


