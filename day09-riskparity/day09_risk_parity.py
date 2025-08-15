"""
Day 9 — Risk Parity vs Equal Weight

Idea
----
Allocate more capital to low-vol assets and less to high-vol assets so that
each asset contributes similarly to total portfolio risk.

Implementation
--------------
- Compute rolling volatility from daily returns.
- At each rebalance, set weights ∝ 1 / vol (normalize to sum=1).
- Compare to equal-weight baseline.
- Apply transaction costs (bps on turnover) and cash drag (rf).
- Save CSVs and charts.

Run
---
python day09_risk_parity.py
python day09_risk_parity.py --window 63 --reb M --rf 0.02 --tc_bps 10
"""

from __future__ import annotations
import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ===== If you already have quant_utils.py (from Day 8), import from there =====
# from quant_utils import (
#     TRADING_DAYS, fetch_prices, daily_returns, ann_vol, cagr, sharpe, max_dd, hit_rate,
#     turnover_costs, safe_freq, rebalance_dates
# )

# ===== Lightweight built-ins (kept local to avoid import collisions) =====
TRADING_DAYS = 252

def safe_freq(freq: str) -> str:
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()

def ann_vol(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    return float(x.std() * np.sqrt(periods)) if len(x) else np.nan

def cagr(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if not len(x): return np.nan
    growth = float((1 + x).prod())
    years = len(x) / periods
    return growth ** (1 / years) - 1 if years > 0 else np.nan

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if not np.isfinite(v) or v == 0: return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna()
    if not len(eq): return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x > 0).mean()) if len(x) else np.nan

def turnover_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)

# ===== Strategy bits =====

def risk_parity_weights(r: pd.DataFrame, window: int, freq: str) -> pd.DataFrame:
    """
    Compute inverse-volatility weights at each rebalance date, then ffill daily.
    w_i(t_reb) ∝ 1 / vol_i(window), normalized to sum=1.
    """
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)

    # rolling vol (daily std, annualized)
    roll_vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)

    for dt in rebs:
        if dt not in roll_vol.index:
            continue
        v = roll_vol.loc[dt].replace(0, np.nan).dropna()
        if v.empty:
            continue
        inv = 1.0 / v
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            continue
        w = inv / inv.sum()
        w_t.loc[dt, w.index] = w.values

    return w_t.ffill().fillna(0.0)

def equal_weight_weights(r: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Equal weights across all available assets at each rebalance date."""
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    cols = list(r.columns)
    if not cols:
        return w_t.fillna(0.0)
    ew = 1.0 / len(cols)
    for dt in rebs:
        w_t.loc[dt, cols] = ew
    return w_t.ffill().fillna(0.0)

def run_portfolio(px: pd.DataFrame,
                  reb: str = "M",
                  window: int = 63,
                  rf_annual: float = 0.02,
                  tc_bps: int = 10) -> Dict[str, pd.Series]:
    """
    Build EW and RP daily return streams on same universe and compare.
    """
    r = daily_returns(px)
    if r.empty:
        return {"ew": pd.Series(dtype=float), "rp": pd.Series(dtype=float)}

    # weights
    w_ew = equal_weight_weights(r, reb)
    w_rp = risk_parity_weights(r, window, reb)

    # costs & cash
    rf_daily = rf_annual / TRADING_DAYS
    c_ew = turnover_costs(w_ew, tc_bps)
    c_rp = turnover_costs(w_rp, tc_bps)

    # returns
    ret_ew = (w_ew * r).sum(axis=1) - c_ew - rf_daily
    ret_rp = (w_rp * r).sum(axis=1) - c_rp - rf_daily

    return {"ew": ret_ew.rename("ret_ew"), "rp": ret_rp.rename("ret_rp")}

def metrics(ret: pd.Series, rf_annual: float = 0.02) -> Dict[str, float]:
    eq = (1 + ret).cumprod()
    return {
        "CAGR": cagr(ret),
        "Vol": ann_vol(ret),
        "Sharpe": sharpe(ret, rf_annual=rf_annual),
        "MaxDD": max_dd(eq),
        "Hit%": hit_rate(ret),
    }

# ===== CLI =====

def parse_args():
    p = argparse.ArgumentParser(description="Day 9 — Risk Parity vs Equal Weight")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"], help="Rebalance freq")
    p.add_argument("--window", type=int, default=63, help="Rolling vol window (days)")
    p.add_argument("--rf", type=float, default=0.02, help="Annual risk-free for cash drag")
    p.add_argument("--tc_bps", type=int, default=10, help="Transaction cost (bps of turnover)")
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)

    out = run_portfolio(px, reb=a.reb, window=a.window, rf_annual=a.rf, tc_bps=a.tc_bps)
    ret_ew, ret_rp = out["ew"], out["rp"]

    # Metrics
    perf_ew = metrics(ret_ew, rf_annual=a.rf)
    perf_rp = metrics(ret_rp, rf_annual=a.rf)
    perf_df = pd.DataFrame({"EqualWeight": perf_ew, "RiskParity": perf_rp})

    # Save artifacts
    out_csv = os.path.join(a.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_charts = os.path.join(a.outdir, "charts"); os.makedirs(out_charts, exist_ok=True)

    pd.concat([ret_ew, ret_rp], axis=1).to_csv(os.path.join(out_csv, "day09_daily_returns_ew_vs_rp.csv"))
    perf_df.to_csv(os.path.join(out_csv, "day09_metrics_ew_vs_rp.csv"))

    # Equity curves
    eq_ew = (1 + ret_ew).cumprod().rename("EqualWeight")
    eq_rp = (1 + ret_rp).cumprod().rename("RiskParity")

    fig, ax = plt.subplots(figsize=(10, 5))
    eq_ew.plot(ax=ax); eq_rp.plot(ax=ax)
    ax.set_title(f"Day 9 — Risk Parity vs Equal Weight (window={a.window}d, reb={a.reb})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_charts, "day09_eq_ew_vs_rp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Weight stack (last year) — optional quick viz
    # (Compute RP weights again on full index for plotting stack)
    r = daily_returns(px)
    w_rp = risk_parity_weights(r, a.window, a.reb)
    if not w_rp.empty:
        last_year = w_rp.index.max() - pd.DateOffset(years=1)
        w_tail = w_rp.loc[last_year:].clip(lower=0)
        if not w_tail.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            w_tail.plot.area(ax=ax)
            ax.set_title("Day 9 — Risk Parity Weights (last 12m)")
            fig.tight_layout()
            fig.savefig(os.path.join(out_charts, "day09_rp_weights_last12m.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Print summary
    print("\n=== Day 9 — Metrics (net) ===")
    print(perf_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and pd.notnull(x) else x).to_string())

    print("\nArtifacts:")
    print(f"- returns: {os.path.join(out_csv, 'day09_daily_returns_ew_vs_rp.csv')}")
    print(f"- metrics: {os.path.join(out_csv, 'day09_metrics_ew_vs_rp.csv')}")
    print(f"- chart_eq: {os.path.join(out_charts, 'day09_eq_ew_vs_rp.png')}")
    print(f"- chart_weights: {os.path.join(out_charts, 'day09_rp_weights_last12m.png')}")

if __name__ == "__main__":
    main()