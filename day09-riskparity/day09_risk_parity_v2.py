"""
Day 9 (v2) — Risk Parity: Inverse-Vol OR Equal Risk Contribution (ERC)

Upgrades vs v1
--------------
- mode 'ivol': inverse-volatility weights (simple, fast)
- mode 'erc' : Equal Risk Contribution using Ledoit–Wolf shrinkage covariance
- Weight caps / vol floors to avoid concentration & divide-by-zero
- Rebalance band to skip tiny changes and cut turnover
- Rolling or EWMA vol choice for 'ivol' mode

Run
---
python day09_risk_parity_v2.py --mode ivol --window 63 --reb M --rf 0.02 --tc_bps 10
python day09_risk_parity_v2.py --mode erc  --window 126 --reb M --rf 0.02 --tc_bps 10 --w_cap 0.25 --band 0.05
"""

from __future__ import annotations
import argparse
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional: scikit-learn for Ledoit–Wolf shrinkage (required for 'erc' mode)
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

TRADING_DAYS = 252

# ---------- Basics ----------
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

# ---------- Metrics ----------
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

# ---------- Helpers ----------
def normalize_weights(w: pd.Series, w_cap: Optional[float]) -> pd.Series:
    w = w.clip(lower=0.0)
    if w_cap is not None and w_cap > 0:
        w = w.clip(upper=w_cap)
    s = w.sum()
    return w / s if s > 0 else w

def erc_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    """
    Simple projected gradient to solve for Equal Risk Contribution.
    Minimize deviation of risk contributions from their mean under simplex constraint.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 0.01
    for _ in range(max_iter):
        m = cov @ w                     # marginal risks
        rc = w * m                      # risk contributions
        target = rc.mean()
        grad = m - (target / np.maximum(w, 1e-12))
        w = w - step * grad
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 0:
            w = w / s
        if np.linalg.norm(rc - target) < tol:
            break
    return w

def rolling_vol_matrix(r: pd.DataFrame, window: int, method: str) -> pd.DataFrame:
    """
    For ivol mode we only need per-asset vol, not full cov.
    method='rolling' -> r.rolling(window).std()
    method='ewma'    -> r.ewm(span=window).std()
    """
    if method == "ewma":
        vol = r.ewm(span=window).std() * np.sqrt(TRADING_DAYS)
    else:
        vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol

# ---------- Weight engines ----------
def weights_inverse_vol(r: pd.DataFrame,
                        window: int,
                        freq: str,
                        vol_method: str = "rolling",
                        vol_floor: float = 0.0,
                        w_cap: Optional[float] = None,
                        band: float = 0.0) -> pd.DataFrame:
    """
    Inverse-vol weights at each rebalance; optional caps & rebalance band.
    """
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    vol = rolling_vol_matrix(r, window, vol_method)

    prev_w = None
    for dt in rebs:
        if dt not in vol.index:
            continue
        v = vol.loc[dt].replace(0, np.nan)
        if vol_floor > 0:
            v = v.fillna(v.median())
            v = v.clip(lower=vol_floor)
        else:
            v = v.dropna()
        if v.empty:
            continue

        inv = 1.0 / v
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            continue
        w = normalize_weights(inv, w_cap)

        # Rebalance band: skip if small change
        if prev_w is not None:
            # align indexes
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if band > 0 and drift < band:
                w = prev_w.copy()

        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()

    return w_t.ffill().fillna(0.0)

def weights_erc(r: pd.DataFrame,
                window: int,
                freq: str,
                w_cap: Optional[float] = None,
                band: float = 0.0,
                min_obs: int = 40) -> pd.DataFrame:
    """
    ERC weights: use Ledoit–Wolf covariance on rolling window up to each rebalance.
    """
    if not _HAS_SK:
        raise ImportError("scikit-learn is required for ERC mode (pip install scikit-learn).")

    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)

    prev_w = None
    for dt in rebs:
        # rolling window ending at dt
        r_window = r.loc[:dt].tail(window)
        if r_window.shape[0] < min_obs:
            continue

        tickers = r_window.columns
        # shrinkage covariance
        lw = LedoitWolf().fit(r_window.values)
        cov = lw.covariance_

        w_np = erc_weights(cov)  # np array
        w = pd.Series(w_np, index=tickers)
        w = normalize_weights(w, w_cap)

        # Rebalance band
        if prev_w is not None:
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if band > 0 and drift < band:
                w = prev_w.copy()

        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()

    return w_t.ffill().fillna(0.0)

# ---------- Top-level run ----------
def run_portfolios(px: pd.DataFrame,
                   mode: str = "ivol",
                   window: int = 63,
                   reb: str = "M",
                   rf_annual: float = 0.02,
                   tc_bps: int = 10,
                   vol_method: str = "rolling",
                   vol_floor: float = 0.0,
                   w_cap: Optional[float] = None,
                   band: float = 0.0) -> Dict[str, pd.Series]:
    r = daily_returns(px)
    if r.empty:
        return {"ret": pd.Series(dtype=float)}

    if mode == "ivol":
        w = weights_inverse_vol(r, window, reb, vol_method, vol_floor, w_cap, band)
    elif mode == "erc":
        w = weights_erc(r, window, reb, w_cap, band)
    else:
        raise ValueError("mode must be 'ivol' or 'erc'")

    rf_daily = rf_annual / TRADING_DAYS
    costs = turnover_costs(w, tc_bps)
    ret = (w * r).sum(axis=1) - costs - rf_daily
    return {"ret": ret.rename(f"ret_{mode}"), "weights": w}

def metrics_table(ret: pd.Series, rf_annual: float) -> pd.DataFrame:
    eq = (1 + ret).cumprod()
    m = {
        "CAGR": cagr(ret),
        "Vol": ann_vol(ret),
        "Sharpe": sharpe(ret, rf_annual),
        "MaxDD": max_dd(eq),
        "Hit%": hit_rate(ret)
    }
    return pd.DataFrame(m, index=[0]).T

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 9 (v2) — Risk Parity (ivol or ERC)")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"], help="Rebalance freq")
    p.add_argument("--mode", type=str, default="ivol", choices=["ivol", "erc"])
    p.add_argument("--window", type=int, default=63, help="Rolling window (days)")
    p.add_argument("--rf", type=float, default=0.02, help="Annual risk-free (cash drag)")
    p.add_argument("--tc_bps", type=int, default=10, help="Transaction cost in bps of turnover")
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling", "ewma"], help="ivol volatility estimator")
    p.add_argument("--vol_floor", type=float, default=0.0, help="Floor on vol (ivol mode)")
    p.add_argument("--w_cap", type=float, default=None, help="Cap on any single weight (e.g., 0.25)")
    p.add_argument("--band", type=float, default=0.0, help="Rebalance L1 band (skip if change < band)")
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    if a.mode == "erc" and not _HAS_SK:
        raise SystemExit("ERC mode requires scikit-learn. Install with: pip install scikit-learn")

    px = fetch_prices(tickers, a.start)
    out = run_portfolios(
        px,
        mode=a.mode,
        window=a.window,
        reb=a.reb,
        rf_annual=a.rf,
        tc_bps=a.tc_bps,
        vol_method=a.vol_method,
        vol_floor=a.vol_floor,
        w_cap=a.w_cap,
        band=a.band
    )
    ret = out["ret"]
    w = out["weights"]
    eq = (1 + ret).cumprod()

    # Save artifacts
    os.makedirs(a.outdir, exist_ok=True)
    ret.to_csv(f"{a.outdir}/day09_returns_{a.mode}.csv")
    w.to_csv(f"{a.outdir}/day09_weights_{a.mode}.csv")

    # Plot equity curve
    plt.figure(figsize=(10, 4))
    eq.plot(color="black", lw=2, label="Equity Curve")
    plt.title(f"Day 9 — Risk Parity ({a.mode.upper()})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{a.outdir}/day09_eqcurve_{a.mode}.png")

    # Print summary
    print("=== Day 9 — Risk Parity Performance ===")
    print(metrics_table(ret, rf_annual=a.rf))
    print(f"\nArtifacts:")
    print(f"- daily_returns: {a.outdir}/day09_returns_{a.mode}.csv")
    print(f"- weights_daily: {a.outdir}/day09_weights_{a.mode}.csv")
    print(f"- chart_eq:      {a.outdir}/day09_eqcurve_{a.mode}.png")

if __name__ == "__main__":
    main()