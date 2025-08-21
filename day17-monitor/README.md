## Day 17 — Monitoring & Safety Rails
Purpose: catch operational issues before they become losses
- Checks: file freshness, NaNs, negative weights, weight caps, target sum≈1, turnover spikes, drift (held but no target).
- Inputs: `positions.csv` (from Day 16) and the latest `orders_YYYY-MM-DD.csv` in `orders/`.
- Output: `reports/day17_monitor_YYYY-MM-DD.txt` and non-zero exit code on alerts.

Run:
python day17_monitor.py --positions positions.csv --orders_dir orders --outdir reports \
  --w_cap 0.30 --turnover_cap 0.25 --sum_tol 0.02 --stale_days 2
