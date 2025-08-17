"""
Day 12 — Costs & Execution Throttling

What this teaches:
- Naive costs (bps of turnover) vs more realistic costs (spread + impact)
- Execution throttling:
  * per-asset no-trade band around current weights
  * partial moves (execution speed)
  * daily turnover cap

We compare two variants:
  1) instant_simplebps  : jump to target on rebalance; flat bps costs
  2) throttled_advanced : partial to target with band + cap; spread+impact costs

Run examples:
  python day12_costs_and_throttle.py --mode erc --window 126 --reb M --tc_bps 10
  python day12_costs_and_throttle.py --mode erc --window 126 --reb M \
    --exec_speed 0.33 --no_trade_band 0.002 --turnover_cap 0.20 \
    --spread_bps 2 --impact_coef 0.10 --impact_span 63
"""

from __future__ import annotations
import argparse, os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional for ERC allocator (Equal Risk Contribution)
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
    if eq.empty: return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x > 0).mean()) if len(x) else np.nan

# ---------- Allocators (IVOL / ERC / EW) ----------
def normalize_weights(w: pd.Series, w_cap: Optional[float]) -> pd.Series:
    w = w.clip(lower=0.0)
    if w_cap is not None and w_cap > 0:
        w = w.clip(upper=w_cap)
    s = w.sum()
    return w / s if s > 0 else w

def erc_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 0.01
    for _ in range(max_iter):
        m = cov @ w
        rc = w * m
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
            if float((w_al - prev_al).abs().sum()) < band:
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
        lw = LedoitWolf().fit(r_win.values)
        cov = lw.covariance_
        w = pd.Series(erc_weights(cov), index=r_win.columns)
        w = normalize_weights(w, w_cap)
        if prev_w is not None and band > 0:
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            if float((w_al - prev_al).abs().sum()) < band:
                w = prev_w.copy()
        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()
    return w_t.ffill().fillna(0.0)

# ---------- Cost models ----------
def naive_cost(turnover: pd.Series, tc_bps: int) -> pd.Series:
    """Flat bps of 1-way turnover."""
    return turnover * (tc_bps / 10000.0)

def advanced_cost(turnover: pd.Series,
                  spread_bps: float,
                  vol_proxy: pd.Series,
                  impact_coef: float = 0.10) -> pd.Series:
    """
    Spread/slippage + square-root impact.
    - spread part: turnover * spread_bps
    - impact part: impact_coef * vol_proxy * sqrt(turnover)
    vol_proxy is a daily series (e.g., cross-asset EWMA vol of returns).
    """
    spread = turnover * (spread_bps / 10000.0)
    impact = impact_coef * vol_proxy * np.sqrt(turnover.clip(lower=0.0))
    return (spread.fillna(0.0) + impact.fillna(0.0)).rename("cost_adv")

# ---------- Execution engines ----------
def to_daily_turnover(w: pd.DataFrame) -> pd.Series:
    """One-way turnover per day from a daily weight series."""
    return (w.diff().abs().sum(axis=1)).fillna(0.0).rename("turnover")

def instant_execution_returns(r: pd.DataFrame, w_target: pd.DataFrame,
                              tc_bps: int, rf_annual: float) -> Tuple[pd.Series, pd.Series]:
    """
    Jump to target on each rebalance; flat bps costs.
    """
    w = w_target.ffill().fillna(0.0)
    turn = to_daily_turnover(w)
    costs = naive_cost(turn, tc_bps)
    rf_daily = rf_annual / TRADING_DAYS
    ret = (w * r).sum(axis=1) - costs - rf_daily
    return ret.rename("ret_instant_simplebps"), turn.rename("turnover_instant")

def throttled_execution_returns(r: pd.DataFrame, w_targets: pd.DataFrame,
                                exec_speed: float = 0.33,
                                no_trade_band: float = 0.002,
                                turnover_cap: float = 0.20,
                                spread_bps: float = 2.0,
                                impact_coef: float = 0.10,
                                impact_span: int = 63,
                                rf_annual: float = 0.02) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Move partially toward targets at each rebalance, with per-asset band and daily turnover cap.
    Costs = spread + square-root impact.
    """
    dates = r.index
    cols  = r.columns
    rebs = w_targets.dropna(how="all").index.intersection(dates)

    # EWMA daily vol proxy across assets (for impact cost)
    vol_proxy = r.ewm(span=impact_span).std().mean(axis=1).fillna(0.0)

    w_exec = pd.DataFrame(0.0, index=dates, columns=cols)
    w_curr = pd.Series(0.0, index=cols)
    turns  = pd.Series(0.0, index=dates)

    for d in dates:
        # If today is a rebalance date, compute a trade toward target
        if d in rebs:
            tgt = w_targets.loc[d].fillna(0.0)
            gap = tgt - w_curr

            # No-trade band per asset
            mask_move = gap.abs() >= no_trade_band
            gap = gap.where(mask_move, 0.0)

            # Partial move
            dv = exec_speed * gap

            # Daily turnover (before cap)
            turn = float(dv.abs().sum())

            # Cap daily turnover
            if turnover_cap is not None and turn > turnover_cap and turn > 0:
                dv *= (turnover_cap / turn)
                turn = turnover_cap

            w_curr = (w_curr + dv).clip(lower=0.0)
            s = w_curr.sum()
            if s > 0:
                w_curr = w_curr / s  # renormalize if drifted
            turns.loc[d] = turn

        w_exec.loc[d] = w_curr.values

    # Costs and returns
    costs = advanced_cost(turns.rename("turnover"), spread_bps, vol_proxy, impact_coef)
    rf_daily = rf_annual / TRADING_DAYS
    ret = (w_exec * r).sum(axis=1) - costs - rf_daily

    return ret.rename("ret_throttled_advanced"), turns.rename("turnover_throttled"), w_exec

# ---------- Top driver ----------
def run(px: pd.DataFrame, mode: str, window: int, reb: str,
        rf_annual: float, tc_bps: int, vol_method: str,
        vol_floor: float, w_cap: Optional[float], band: float,
        exec_speed: float, no_trade_band: float, turnover_cap: float,
        spread_bps: float, impact_coef: float, impact_span: int) -> Dict[str, object]:

    r = daily_returns(px)

    # Build monthly allocator (targets)
    if mode == "ivol":
        w_targets = weights_inverse_vol(r, window, reb, vol_method, vol_floor, w_cap, band)
    elif mode == "erc":
        w_targets = weights_erc(r, window, reb, w_cap, band)
    else:
        w_targets = weights_equal_weight(r, reb)

    # Variant 1: instant → simple bps cost
    ret_instant, turn_instant = instant_execution_returns(r, w_targets, tc_bps, rf_annual)

    # Variant 2: throttled → spread + impact
    ret_throt, turn_throt, w_exec = throttled_execution_returns(
        r, w_targets, exec_speed, no_trade_band, turnover_cap,
        spread_bps, impact_coef, impact_span, rf_annual
    )

    return {
        "ret_instant": ret_instant,
        "turn_instant": turn_instant,
        "ret_throttled": ret_throt,
        "turn_throttled": turn_throt,
        "weights_targets": w_targets,
        "weights_exec": w_exec
    }

def metrics_table(rets: Dict[str, pd.Series], rf_annual: float) -> pd.DataFrame:
    out = {}
    for nm, r in rets.items():
        eq = (1 + r).cumprod()
        out[nm] = {
            "CAGR": cagr(r),
            "Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf_annual),
            "MaxDD": max_dd(eq),
            "Hit%": hit_rate(r)
        }
    return pd.DataFrame(out).T

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 12 — Costs & Execution Throttling")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    p.add_argument("--mode", type=str, default="erc", choices=["erc","ivol","ew"])
    p.add_argument("--window", type=int, default=126)

    # naive cost
    p.add_argument("--tc_bps", type=int, default=10, help="Flat bps of turnover for 'instant' variant")
    p.add_argument("--rf", type=float, default=0.02)

    # allocator knobs
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling","ewma"])
    p.add_argument("--vol_floor", type=float, default=0.0)
    p.add_argument("--w_cap", type=float, default=0.25)
    p.add_argument("--band", type=float, default=0.05)

    # execution throttle
    p.add_argument("--exec_speed", type=float, default=0.33, help="Fraction of gap traded at each rebalance day")
    p.add_argument("--no_trade_band", type=float, default=0.002, help="Per-asset no-trade band (abs weight)")
    p.add_argument("--turnover_cap", type=float, default=0.20, help="Daily turnover cap (L1 sum of abs weight changes)")

    # advanced costs
    p.add_argument("--spread_bps", type=float, default=2.0, help="Bid-ask/ slippage bps per 1-way turnover")
    p.add_argument("--impact_coef", type=float, default=0.10, help="Impact coefficient in cost = k * vol_proxy * sqrt(turnover)")
    p.add_argument("--impact_span", type=int, default=63, help="EWMA span for vol_proxy")
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode == "erc" and not _HAS_SK:
        raise SystemExit("ERC mode requires scikit-learn. Install with: pip install scikit-learn "
                         "or rerun with --mode ivol/ew")

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)

    res = run(px, a.mode, a.window, a.reb, a.rf, a.tc_bps, a.vol_method, a.vol_floor,
              a.w_cap, a.band, a.exec_speed, a.no_trade_band, a.turnover_cap,
              a.spread_bps, a.impact_coef, a.impact_span)

    # Save
    out_csv = os.path.join(a.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_ch  = os.path.join(a.outdir, "charts");  os.makedirs(out_ch,  exist_ok=True)

    pd.concat([res["ret_instant"], res["ret_throttled"]], axis=1)\
      .to_csv(os.path.join(out_csv, "day12_returns.csv"))
    pd.concat([res["turn_instant"], res["turn_throttled"]], axis=1)\
      .to_csv(os.path.join(out_csv, "day12_turnover.csv"))
    res["weights_targets"].to_csv(os.path.join(out_csv, "day12_weights_targets.csv"))
    res["weights_exec"].to_csv(os.path.join(out_csv, "day12_weights_exec.csv"))

    mt = metrics_table({"Instant_simplebps": res["ret_instant"],
                        "Throttled_advanced": res["ret_throttled"]}, a.rf).round(4)
    mt.to_csv(os.path.join(out_csv, "day12_metrics.csv"))

    # Charts
    eq_i = (1 + res["ret_instant"]).cumprod().rename("Instant_simplebps")
    eq_t = (1 + res["ret_throttled"]).cumprod().rename("Throttled_advanced")
    plt.figure(figsize=(12,6))
    eq_i.plot(); eq_t.plot()
    plt.title("Day 12 — Equity (Instant vs Throttled)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_ch, "day12_equity.png"), dpi=150); plt.close()

    res["turn_instant"].rolling(21).mean().plot(figsize=(12,3), title="Turnover (21d avg) — Instant")
    plt.tight_layout(); plt.savefig(os.path.join(out_ch, "day12_turnover_instant.png"), dpi=150); plt.close()
    res["turn_throttled"].rolling(21).mean().plot(figsize=(12,3), title="Turnover (21d avg) — Throttled")
    plt.tight_layout(); plt.savefig(os.path.join(out_ch, "day12_turnover_throttled.png"), dpi=150); plt.close()

    print("\n=== Day 12 — Costs & Throttle (net) ===")
    print(mt.to_string())
    print("\nArtifacts:")
    print(f"- returns:   {os.path.join(out_csv, 'day12_returns.csv')}")
    print(f"- turnover:  {os.path.join(out_csv, 'day12_turnover.csv')}")
    print(f"- weights_*: {os.path.join(out_csv, 'day12_weights_targets.csv')} , {os.path.join(out_csv, 'day12_weights_exec.csv')}")
    print(f"- equity:    {os.path.join(out_ch,  'day12_equity.png')}")
    print(f"- turnover charts: {os.path.join(out_ch,  'day12_turnover_instant.png')} , {os.path.join(out_ch,  'day12_turnover_throttled.png')}")

if __name__ == "__main__":
    main()