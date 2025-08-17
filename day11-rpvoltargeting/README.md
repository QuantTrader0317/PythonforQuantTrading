# Day 11 — Risk Parity + Volatility Targeting
Acronyms (expanded)

IVOL — Inverse Volatility: weights ∝ 1/volatility (risk-parity-lite).

ERC — Equal Risk Contribution: choose weights so each asset contributes ~equal portfolio risk (uses Ledoit–Wolf shrinkage covariance).

EWMA — Exponentially Weighted Moving Average: fast-decay average; here used to estimate realized vol.

Vol Targeting: scale portfolio risk to a chosen annualized volatility (e.g., 10%).

What this module does

Build monthly risk-balanced weights with IVOL or ERC.

Turn weights into daily returns (includes transaction costs and cash drag).

Apply Vol Targeting on the whole portfolio using EWMA vol, with clamps to avoid silly leverage.

# Quickstart
create & activate venv
python -m venv .venv
. .venv\Scripts\Activate.ps1   # (Windows PowerShell)

# install
pip install -r requirements.txt

# run (ERC allocator + 10% target vol)
python day11_rp_voltarget.py --mode erc --window 126 --reb M \
  --vt_target 0.10 --vt_span 63 --vt_min 0.5 --vt_max 2.0 \
  --tc_bps 10 --rf 0.02 --w_cap 0.25 --band 0.05

# optional: more realistic sizing (no look-ahead) + regime-aware targets
python day11_rp_voltarget.py --mode erc --window 126 --reb M \
  --vt_forecast --vt_regime --vt_thr 0.18 --vt_target_lo 0.08 --vt_target_hi 0.12 \
  --vt_span 63 --vt_min 0.5 --vt_max 2.0 --tc_bps 10 --rf 0.02 --w_cap 0.25 --band 0.05

Your results (from outputs/day11_metrics.csv)
Portfolio	CAGR	Vol	Sharpe	MaxDD	Hit%
EqualWeight	9.25%	15.76%	0.4600	−30.31%	53.42%
RiskParity – ERC	8.51%	15.41%	0.4225	−30.21%	52.69%
RP-ERC + VolTarget	6.42%	10.59%	0.4176	−18.93%	52.58%

Interpretation (blunt + useful):

Risk down hard: RP+VT cut realized vol by ~31% vs RP and reduced worst drawdown by ~37%.

Return down too: CAGR −2.09 pp vs RP (tradeoff for a smoother ride).

Sharpe ~flat: −0.005 vs RP (basically unchanged).

Target accuracy: realized vol = 10.59% (only +0.59% above the 10% target).

# Artifacts

outputs/day11_daily_returns.csv — EqualWeight, RP, RP+VT

outputs/day11_metrics.csv — table above

outputs/day11_weights.csv — allocator weights (monthly, ffilled)

outputs/day11_scaler_sigma.csv — daily scale factor s and EWMA sigma (plus target if enabled)

charts/day11_equity_curves.png — EW vs RP vs RP+VT

charts/day11_drawdown_*.png — drawdowns per variant

charts/day11_scaler.png — daily scale factor (how much we lever/de-risk)

charts/day11_target_series.png — daily target level when regime option is on

# Tuning playbook (fast knobs)

More juice: raise vol target → --vt_target 0.12 (or 0.14). Expect higher CAGR; a bit more DD.

React faster/slower:

Faster sizing: --vt_span 42–63

Smoother sizing: --vt_span 84–126

Clamp smarter: --vt_min 0.7 --vt_max 1.8 to avoid over-de-risking bottoms.

Regime-aware: --vt_regime --vt_thr 0.18 --vt_target_lo 0.08 --vt_target_hi 0.12 (cut risk more in stress, run a touch hotter in calm).

Cost realism (next pass): scale weights before turnover so costs reflect the levered/de-levered book.

# Why this matters

This is the grown-up portfolio shape: a risk-balanced allocator (ERC/IVOL) inside a book-level risk control (Vol Targeting). Lower shocks, steadier risk — something you can actually live with (and present in a PM interview).