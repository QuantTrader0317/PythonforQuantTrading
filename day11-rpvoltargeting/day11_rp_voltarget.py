"""
Day 11 — Risk Parity + Volatility Targeting

What this does
--------------
1) Build a monthly-risk allocator:
   - IVOL = Inverse-Volatility weights (simple risk parity-lite)
   - ERC  = Equal Risk Contribution (uses Ledoit–Wolf shrinkage covariance)
2) Turn monthly weights into daily portfolio returns (with costs & cash drag).
3) Apply VOL TARGETING (scale daily returns to a target annualized volatility).
   - Vol estimate = EWMA (Exponentially Weighted Moving Average) of daily returns
   - Scale clamp to avoid silly leverage (e.g., 0.5x to 2.0x)

Run examples
------------
python day11_rp_voltarget.py --mode ivol --window 63  --reb M --vt_target 0.10 --vt_span 63 --vt_min 0.5 --vt_max 2.0
python day11_rp_voltarget.py --mode erc  --window 126 --reb M --vt_target 0.10 --vt_span 63 --w_cap 0.25 --band 0.05
"""

from __future__ import annotations
import argparse
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional (needed for ERC = Equal Risk Contribution)
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

TRADING_DAYS = 252

# ---------- Basics ----------
def safe_freq(freq: str) -> str:
    """Map pandas-deprecated codes to safe ones: 'M'->'ME', 'Q'->'QE'."""
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
    if eq.empty: return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x > 0).mean()) if len(x) else np.nan

def turnover_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    """bps of 1-way turnover."""
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
    Equal Risk Contribution via simple projected gradient.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 0.01
    for _ in range(max_iter):
        m = cov @ w                 # marginal risk
        rc = w * m                  # risk contribution
        target = rc.mean()
        grad = m - (target / np.maximum(w, 1e-12))
        w = np.clip(w - step * grad, 0.0, None)
        s = w.sum()
        if s > 0: w = w / s
        if np.linalg.norm(rc - target) < tol:
            break
    return w

def rolling_vol_matrix(r: pd.DataFrame, window: int, method: str) -> pd.DataFrame:
    if method == "ewma":
        return r.ewm(span=window).std() * np.sqrt(TRADING_DAYS)
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)

# ---------- Allocators ----------
def weights_equal_weight(r: pd.DataFrame, freq: str) -> pd.DataFrame:
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    for dt in rebs:
        cols = r.columns[r.loc[:dt].tail(1).notna().values[0]]
        if len(cols) == 0: continue
        w_t.loc[dt, cols] = 1.0 / len(cols)
    return w_t.ffill().fillna(0.0)

def weights_inverse_vol(r: pd.DataFrame, window: int, freq: str,
                        vol_method: str = "rolling", vol_floor: float = 0.0,
                        w_cap: Optional[float] = None, band: float = 0.0) -> pd.DataFrame:
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    vol = rolling_vol_matrix(r, window, vol_method)
    prev_w = None
    for dt in rebs:
        if dt not in vol.index: continue
        v = vol.loc[dt].replace(0, np.nan)
        v = (v.fillna(v.median()).clip(lower=vol_floor) if vol_floor > 0 else v.dropna())
        if v.empty: continue
        inv = 1.0 / v
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty: continue
        w = normalize_weights(inv, w_cap)

        if prev_w is not None and band > 0:
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if drift < band:
                w = prev_w.copy()

        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()
    return w_t.ffill().fillna(0.0)

def weights_erc(r: pd.DataFrame, window: int, freq: str,
                w_cap: Optional[float] = None, band: float = 0.0,
                min_obs: int = 40) -> pd.DataFrame:
    if not _HAS_SK:
        raise ImportError("ERC requires scikit-learn. Install with: pip install scikit-learn")
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    prev_w = None
    for dt in rebs:
        r_win = r.loc[:dt].tail(window)
        if r_win.shape[0] < min_obs: continue
        tickers = r_win.columns
        lw = LedoitWolf().fit(r_win.values)
        cov = lw.covariance_
        w = pd.Series(erc_weights(cov), index=tickers)
        w = normalize_weights(w, w_cap)

        if prev_w is not None and band > 0:
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if drift < band:
                w = prev_w.copy()

        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()
    return w_t.ffill().fillna(0.0)

# ---------- Vol Targeting ----------
def vol_target_series(
    ret: pd.Series,
    target: float = 0.10,
    span: int = 63,
    s_min: float = 0.5,
    s_max: float = 2.0,
    use_forecast: bool = True,
    regime: bool = False,
    regime_thr: float = 0.18,         # threshold on realized vol (annualized)
    target_lo: float = 0.08,          # target when vol > threshold
    target_hi: float = 0.12           # target when vol <= threshold
) -> Dict[str, pd.Series]:
    """
    Vol targeting with options:
      - `use_forecast`: use only *past* returns to size today (no look-ahead)
        sigma_t = sqrt(EWMA(r_{t-1}^2)) * sqrt(252)
      - `regime`: choose daily target based on current vol regime
        target_t = target_lo if sigma_t > regime_thr else target_hi
    """
    # EWMA of squared returns (adjust=False gives recursive EWMA). Shift(1) for forecast.
    sigma = (ret.pow(2)
                .ewm(span=span, adjust=False)
                .mean()
                .shift(1 if use_forecast else 0)
                .pow(0.5)) * np.sqrt(TRADING_DAYS)

    if regime:
        # piecewise daily target based on sigma_t
        target_series = pd.Series(target_hi, index=ret.index)
        target_series[sigma > regime_thr] = target_lo
    else:
        target_series = pd.Series(target, index=ret.index)

    scaler = (target_series / sigma).clip(lower=s_min, upper=s_max).fillna(1.0)
    ret_vt = scaler * ret
    return {
        "ret_vt": ret_vt.rename("ret_vt"),
        "scaler": scaler.rename("scaler"),
        "sigma": sigma.rename("sigma"),
        "target_series": target_series.rename("target")
    }


# ---------- Top-level ----------
def run(px: pd.DataFrame, mode: str, window: int, reb: str,
        rf_annual: float, tc_bps: int, vol_method: str,
        vol_floor: float, w_cap: Optional[float], band: float,
        vt_target: float, vt_span: int, vt_min: float, vt_max: float,
        vt_forecast: bool, vt_regime: bool, vt_thr: float,
        vt_target_lo: float, vt_target_hi: float) -> Dict[str, pd.Series]:

    r = daily_returns(px)

    # Baselines
    w_eq = weights_equal_weight(r, reb)
    w_rp = (weights_inverse_vol(r, window, reb, vol_method, vol_floor, w_cap, band)
            if mode == "ivol" else
            weights_erc(r, window, reb, w_cap, band))

    rf_daily = rf_annual / TRADING_DAYS
    costs_eq = turnover_costs(w_eq, tc_bps)
    costs_rp = turnover_costs(w_rp, tc_bps)

    ret_eq = (w_eq * r).sum(axis=1) - costs_eq - rf_daily
    ret_rp = (w_rp * r).sum(axis=1) - costs_rp - rf_daily

    # ---- Vol targeting with the new flags ----
    vt = vol_target_series(
        (w_rp * r).sum(axis=1),
        target=vt_target, span=vt_span, s_min=vt_min, s_max=vt_max,
        use_forecast=vt_forecast, regime=vt_regime, regime_thr=vt_thr,
        target_lo=vt_target_lo, target_hi=vt_target_hi
    )
    ret_rp_vt = vt["ret_vt"] - costs_rp - rf_daily

    out = {
        "ret_eq": ret_eq.rename("EqualWeight"),
        "ret_rp": ret_rp.rename(f"RiskParity_{mode}"),
        "ret_rp_vt": ret_rp_vt.rename(f"RiskParity_{mode}_VT"),
        "weights": w_rp,
        "scaler": vt["scaler"],
        "sigma": vt["sigma"]
    }
    if "target_series" in vt:
        out["target"] = vt["target_series"]
    return out


def metrics_table(rets: Dict[str, pd.Series], rf_annual: float) -> pd.DataFrame:
    out = {}
    for name, r in rets.items():
        if not name.startswith("ret_"): continue
        eq = (1 + r).cumprod()
        out[name.replace("ret_","")] = {
            "CAGR": cagr(r),
            "Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf_annual),
            "MaxDD": max_dd(eq),
            "Hit%": hit_rate(r)
        }
    return pd.DataFrame(out).T

# ---------- CLI / Main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 11 — Risk Parity + Vol Targeting")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"], help="Rebalance frequency")
    p.add_argument("--mode", type=str, default="erc", choices=["ivol","erc"], help="Allocator: IVOL or ERC")
    p.add_argument("--window", type=int, default=126, help="Rolling window (days) for allocator")
    p.add_argument("--rf", type=float, default=0.02, help="Annual risk-free for cash drag")
    p.add_argument("--tc_bps", type=int, default=10, help="Transaction cost in bps of turnover")
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling","ewma"], help="IVOL vol estimator")
    p.add_argument("--vol_floor", type=float, default=0.0, help="IVOL vol floor (avoid div/0)")
    p.add_argument("--w_cap", type=float, default=0.25, help="Max single-name weight cap")
    p.add_argument("--band", type=float, default=0.05, help="Rebalance band (skip if L1 change < band)")
    p.add_argument("--vt_target", type=float, default=0.10, help="Target annualized volatility")
    p.add_argument("--vt_span", type=int, default=63, help="EWMA span for vol estimate")
    p.add_argument("--vt_min", type=float, default=0.5, help="Min scale")
    p.add_argument("--vt_max", type=float, default=2.0, help="Max scale")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--vt_forecast", action="store_true",
                   help="Use forecasted (lagged) vol for sizing to avoid look-ahead.")
    p.add_argument("--vt_regime", action="store_true",
                   help="Enable regime-aware targets (target_hi/target_lo by vol threshold).")
    p.add_argument("--vt_thr", type=float, default=0.18,
                   help="Annualized vol threshold for regime split.")
    p.add_argument("--vt_target_lo", type=float, default=0.08,
                   help="Target vol when regime is HIGH (sigma > thr).")
    p.add_argument("--vt_target_hi", type=float, default=0.12,
                   help="Target vol when regime is LOW  (sigma <= thr).")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode == "erc" and not _HAS_SK:
        raise SystemExit("ERC mode needs scikit-learn. Install: pip install scikit-learn "
                         "or rerun with --mode ivol")

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)

    # RUN
    res = run(
        px, a.mode, a.window, a.reb,
        a.rf, a.tc_bps, a.vol_method, a.vol_floor, a.w_cap, a.band,
        a.vt_target, a.vt_span, a.vt_min, a.vt_max,
        a.vt_forecast, a.vt_regime, a.vt_thr, a.vt_target_lo, a.vt_target_hi
    )

    # SAVE ARTIFACTS
    out_csv = os.path.join(a.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_ch  = os.path.join(a.outdir, "charts");  os.makedirs(out_ch,  exist_ok=True)

    pd.concat([res["ret_eq"], res["ret_rp"], res["ret_rp_vt"]], axis=1)\
      .to_csv(os.path.join(out_csv, "day11_daily_returns.csv"))
    res["weights"].to_csv(os.path.join(out_csv, "day11_weights.csv"))

    # scaler/sigma and optional target series
    scaler_sigma = pd.concat([res["scaler"], res["sigma"], res.get("target")], axis=1)
    scaler_sigma.to_csv(os.path.join(out_csv, "day11_scaler_sigma.csv"))

    # METRICS
    mt = metrics_table({k: v for k, v in res.items() if isinstance(v, pd.Series)}, a.rf).round(4)
    mt.to_csv(os.path.join(out_csv, "day11_metrics.csv"))

    # CHARTS: equity curves
    eq_eq = (1 + res["ret_eq"]).cumprod()
    eq_rp = (1 + res["ret_rp"]).cumprod()
    eq_vt = (1 + res["ret_rp_vt"]).cumprod()

    plt.figure(figsize=(12, 6))
    eq_eq.plot(label="EqualWeight")
    eq_rp.plot(label=f"RiskParity-{a.mode.upper()}")
    eq_vt.plot(label=f"RP-{a.mode.upper()} + VolTarget")
    plt.title(f"Day 11 — RP + Vol Targeting (target={a.vt_target:.0%}, span={a.vt_span})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_ch, "day11_equity_curves.png"), dpi=150)
    plt.close()

    # Drawdowns
    for nm, eq in [("EqualWeight", eq_eq), (f"RP-{a.mode.upper()}", eq_rp), (f"RP-{a.mode.upper()}+VT", eq_vt)]:
        dd = eq / eq.cummax() - 1.0
        dd.plot(figsize=(12, 3), title=f"Drawdown — {nm}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_ch, f"day11_drawdown_{nm.replace('+','_')}.png"), dpi=150)
        plt.close()

    # Scaler plot
    res["scaler"].plot(figsize=(12, 3), title="Vol Target — Daily Scale Factor (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_ch, "day11_scaler.png"), dpi=150)
    plt.close()

    # Target series plot (only if regime/forecast branch returned it)
    if "target" in res:
        res["target"].plot(figsize=(12, 3), title="Vol Target — Daily Target (regime-aware)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_ch, "day11_target_series.png"), dpi=150)
        plt.close()

    # SUMMARY
    print("\n=== Day 11 — RP + Vol Targeting (net) ===")
    print(mt.to_string())
    print("\nArtifacts:")
    print(f"- daily_returns: {os.path.join(out_csv, 'day11_daily_returns.csv')}")
    print(f"- weights:       {os.path.join(out_csv, 'day11_weights.csv')}")
    print(f"- scaler/sigma:  {os.path.join(out_csv, 'day11_scaler_sigma.csv')}")
    print(f"- equity:        {os.path.join(out_ch,  'day11_equity_curves.png')}")
    print(f"- scaler_chart:  {os.path.join(out_ch,  'day11_scaler.png')}")
    if 'target' in res:
        print(f"- target_chart:  {os.path.join(out_ch,  'day11_target_series.png')}")
    print(f"- drawdowns:     {os.path.join(out_ch,  'day11_drawdown_*.png')}")
