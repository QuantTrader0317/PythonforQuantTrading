"""
Day 10 — Factor Timing via Information Coefficient (IC)

Idea:
- Build a cross-sectional momentum strategy (12-1, top-N).
- For each rebalance month, compute IC = Spearman rank corr between signal ranks and
  forward 1-month returns across the universe.
- Smooth IC with an EMA; map to exposure (1.0 / 0.5 / 0.0) using thresholds.
- Apply exposure to daily returns; compare Base vs Timed.

Run:
  python day10_factor_timing.py --top 3 --reb M --lookback 252 --skip 21 --fwd 21 \
    --ic_span 6 --upper 0.05 --lower -0.05 --tc_bps 10 --rf 0.02
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

TRADING_DAYS = 252

# ---------------- Basics / Utils ----------------
def safe_freq(freq: str) -> str:
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def fetch_prices(tickers, start="2005-01-01") -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()

def cagr(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if not len(x): return np.nan
    growth = float((1 + x).prod())
    years = len(x) / periods
    return growth ** (1 / years) - 1 if years > 0 else np.nan

def ann_vol(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    return float(x.std() * np.sqrt(periods)) if len(x) else np.nan

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if not np.isfinite(v) or v == 0: return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    if eq.empty: return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x > 0).mean()) if len(x) else np.nan

def turnover_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)

def xsection_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)

def build_equal_weights(sig_rank: pd.DataFrame, r: pd.DataFrame, top_n: int, freq: str) -> pd.DataFrame:
    """Pick top-N by rank at each rebalance; return daily weights (ffilled)."""
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    for dt in rebs:
        if dt not in sig_rank.index: continue
        s = sig_rank.loc[dt].dropna().sort_values(ascending=False)
        chosen = list(s.index[:top_n])
        if not chosen: continue
        w_t.loc[dt, chosen] = 1.0 / len(chosen)
    return w_t.ffill().fillna(0.0)

# ---------------- Core Day 10 ----------------
def compute_momentum(px: pd.DataFrame, lookback=252, skip=21) -> pd.DataFrame:
    p1 = px.shift(skip)     # price at t - skip
    p0 = px.shift(lookback) # price at t - lookback
    return (p1 / p0) - 1.0

def forward_returns(px: pd.DataFrame, fwd: int) -> pd.DataFrame:
    """Forward cumulative return over next fwd days, aligned to t."""
    return (px.shift(-fwd) / px - 1.0)

def compute_monthly_ic(sig_rank: pd.DataFrame, fwd_ret: pd.DataFrame, freq: str) -> pd.Series:
    """
    Spearman rank correlation between signal ranks (cross-section) and forward returns
    at each rebalance date. Uses pandas' corr(method='spearman') to avoid SciPy.
    """
    dates = rebalance_dates(sig_rank.index, freq)
    vals = []
    for dt in dates:
        if dt not in sig_rank.index or dt not in fwd_ret.index:
            vals.append((dt, np.nan)); continue
        s = sig_rank.loc[dt]
        fr = fwd_ret.loc[dt]
        df = pd.concat([s.rename("sig"), fr.rename("fwd")], axis=1).dropna()
        if len(df) < 3:
            vals.append((dt, np.nan)); continue
        rho = df["sig"].corr(df["fwd"], method="spearman")
        vals.append((dt, float(rho) if pd.notnull(rho) else np.nan))
    return pd.Series({d:v for d,v in vals}).sort_index()

def exposure_from_ic(ic: pd.Series, span: int = 6, upper=0.05, lower=-0.05) -> pd.Series:
    """Smooth IC with EMA; map to exposure: 1.0 / 0.5 / 0.0 via thresholds."""
    ic_ema = ic.ewm(span=span, min_periods=max(2, span//2)).mean()
    expo = pd.Series(index=ic_ema.index, dtype=float)
    for dt, val in ic_ema.items():
        if not np.isfinite(val):
            expo.loc[dt] = 1.0  # default on
        elif val > upper:
            expo.loc[dt] = 1.0
        elif val < lower:
            expo.loc[dt] = 0.0
        else:
            expo.loc[dt] = 0.5
    return expo.rename("exposure")

def base_strategy_returns(px: pd.DataFrame, reb="M", top=3, lookback=252, skip=21, tc_bps=10, rf_annual=0.02) -> tuple[pd.Series, pd.DataFrame]:
    r = daily_returns(px)
    sig = compute_momentum(px, lookback, skip)
    sr = xsection_rank(sig)
    w = build_equal_weights(sr, r, top_n=top, freq=reb)
    costs = turnover_costs(w, tc_bps)
    rf_daily = rf_annual / TRADING_DAYS
    ret = (w * r).sum(axis=1) - costs - rf_daily
    return ret.rename("ret_base"), w

def apply_timing(ret_base: pd.Series, exposure: pd.Series, reb: str) -> pd.Series:
    """Map monthly exposure to daily; multiply returns."""
    daily_expo = pd.Series(index=ret_base.index, dtype=float)
    reb_days = rebalance_dates(ret_base.index, reb)
    expo_reb = exposure.reindex(reb_days).ffill()
    for dt in reb_days:
        daily_expo.loc[dt] = expo_reb.loc[dt]
    daily_expo = daily_expo.ffill().fillna(1.0)
    return (daily_expo * ret_base).rename("ret_timed")

def metrics_table(rA: pd.Series, rB: pd.Series, names=("Base","Timed"), rf=0.02) -> pd.DataFrame:
    out = {}
    for nm, r in zip(names, (rA, rB)):
        eq = (1 + r).cumprod()
        out[nm] = {
            "CAGR": cagr(r),
            "Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf_annual=rf),
            "MaxDD": max_dd(eq),
            "Hit%": hit_rate(r),
        }
    return pd.DataFrame(out)

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Day 10 — Factor Timing via IC")
    ap.add_argument("--tickers", type=str, default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    ap.add_argument("--start", type=str, default="2005-01-01")
    ap.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--fwd", type=int, default=21, help="Forward window (days) for IC target returns")
    ap.add_argument("--ic_span", type=int, default=6, help="EMA span for IC smoothing (rebalance periods)")
    ap.add_argument("--upper", type=float, default=0.05, help="Upper IC threshold (go full risk)")
    ap.add_argument("--lower", type=float, default=-0.05, help="Lower IC threshold (go to cash)")
    ap.add_argument("--tc_bps", type=int, default=10)
    ap.add_argument("--rf", type=float, default=0.02)
    ap.add_argument("--outdir", type=str, default=".")
    return ap.parse_args()

def main():
    a = parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)

    # Base strategy
    ret_base, _ = base_strategy_returns(px, reb=a.reb, top=a.top, lookback=a.lookback, skip=a.skip,
                                        tc_bps=a.tc_bps, rf_annual=a.rf)

    # IC timing inputs
    sig = compute_momentum(px, a.lookback, a.skip)
    sr = xsection_rank(sig)
    fwd = forward_returns(px, a.fwd)

    # Monthly IC and exposure
    ic = compute_monthly_ic(sr, fwd, a.reb)
    expo = exposure_from_ic(ic, span=a.ic_span, upper=a.upper, lower=a.lower)

    # Apply exposure
    ret_timed = apply_timing(ret_base, expo, a.reb)

    # Outputs
    out_csv = os.path.join(a.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_charts = os.path.join(a.outdir, "charts"); os.makedirs(out_charts, exist_ok=True)

    # Save returns + IC
    pd.concat([ret_base, ret_timed], axis=1).to_csv(os.path.join(out_csv, "day10_returns_base_vs_timed.csv"))
    ic.to_csv(os.path.join(out_csv, "day10_ic_series.csv"))
    expo.to_csv(os.path.join(out_csv, "day10_exposure_series.csv"))

    # Metrics
    comp = metrics_table(ret_base, ret_timed, names=("Base","Timed"), rf=a.rf)
    comp.to_csv(os.path.join(out_csv, "day10_metrics_base_vs_timed.csv"))

    # Charts
    eq_base = (1 + ret_base).cumprod().rename("Base")
    eq_timed = (1 + ret_timed).cumprod().rename("Timed")

    # Equity comparison
    fig, ax = plt.subplots(figsize=(10,5))
    eq_base.plot(ax=ax); eq_timed.plot(ax=ax)
    ax.set_title(f"Day 10 — Factor Timing via IC (upper={a.upper:+.2f}, lower={a.lower:+.2f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_charts, "day10_eq_base_vs_timed.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # IC chart
    fig, ax = plt.subplots(figsize=(10,3.2))
    ic.plot(ax=ax)
    ic.ewm(span=a.ic_span, min_periods=max(2, a.ic_span//2)).mean().plot(ax=ax, linestyle="--")
    ax.axhline(a.upper, linestyle=":", lw=1); ax.axhline(a.lower, linestyle=":", lw=1)
    ax.set_title("Day 10 — Information Coefficient (monthly) & EMA")
    fig.tight_layout()
    fig.savefig(os.path.join(out_charts, "day10_ic_series.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\n=== Day 10 — Factor Timing (net) ===")
    print(comp.round(4).to_string())
    print("\nArtifacts:")
    print(f"- returns: {os.path.join(out_csv, 'day10_returns_base_vs_timed.csv')}")
    print(f"- metrics: {os.path.join(out_csv, 'day10_metrics_base_vs_timed.csv')}")
    print(f"- ic:      {os.path.join(out_csv, 'day10_ic_series.csv')}")
    print(f"- exposure:{os.path.join(out_csv, 'day10_exposure_series.csv')}")
    print(f"- chart_eq:{os.path.join(out_charts, 'day10_eq_base_vs_timed.png')}")
    print(f"- chart_ic:{os.path.join(out_charts, 'day10_ic_series.png')}")

if __name__ == "__main__":
    main()