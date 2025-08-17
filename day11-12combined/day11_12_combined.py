"""
Day 11 + 12 — Risk Parity + Vol Targeting + Execution (Instant vs Throttled)

Pipeline:
1) Allocator weights (IVOL or ERC) at rebalance dates, ffill daily.
2) Pre-VT portfolio returns; EWMA vol (optionally lagged = no look-ahead).
3) Daily VT scaler 's', scale whole book: w_vt = s * w_alloc (no renorm).
4) Execution:
   - Instant_simplebps: jump to VT target on rebalance dates; costs = bps * turnover.
   - Throttled_advanced: partial move + no-trade band + daily turnover cap;
                         costs = spread + sqrt-impact tied to EWMA cross-asset vol.
"""

from __future__ import annotations
import argparse, os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional for ERC allocator
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
    """
    Return the LAST **TRADING DAY** of each month/quarter in `index`.
    This avoids calendar month-end labels (e.g., 2018-06-30) that aren't in the trading index.
    """
    if freq.upper().startswith("Q"):
        grp = index.to_period("Q")
    else:  # default monthly
        grp = index.to_period("M")
    s = pd.Series(index=index, data=index)
    rebs = s.groupby(grp).max()
    return pd.DatetimeIndex(rebs.values)

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
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
    g = float((1 + x).prod()); yrs = len(x)/periods
    return g**(1/yrs) - 1 if yrs>0 else np.nan

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if not np.isfinite(v) or v == 0: return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    if eq.empty: return np.nan
    return float((eq/eq.cummax()-1).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x>0).mean()) if len(x) else np.nan

# ---------- Allocators ----------
def normalize_weights(w: pd.Series, w_cap: Optional[float]) -> pd.Series:
    w = w.clip(lower=0.0)
    if w_cap is not None and w_cap>0: w = w.clip(upper=w_cap)
    s = w.sum()
    return w/s if s>0 else w

def erc_weights(cov: np.ndarray, tol: float=1e-8, max_iter:int=10000) -> np.ndarray:
    n = cov.shape[0]; w = np.ones(n)/n; step=0.01
    for _ in range(max_iter):
        m = cov@w; rc = w*m; tgt = rc.mean()
        grad = m - (tgt/np.maximum(w,1e-12))
        w = np.clip(w - step*grad, 0.0, None); s = w.sum()
        if s>0: w/=s
        if np.linalg.norm(rc - tgt) < tol: break
    return w

def rolling_vol_matrix(r: pd.DataFrame, window:int, method:str) -> pd.DataFrame:
    if method=="ewma": return r.ewm(span=window).std()*np.sqrt(TRADING_DAYS)
    return r.rolling(window).std()*np.sqrt(TRADING_DAYS)

def weights_equal_weight(r: pd.DataFrame, freq:str) -> pd.DataFrame:
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for dt in rebs:
        cols = r.columns[r.loc[:dt].tail(1).notna().values[0]]
        if len(cols)==0: continue
        w_t.loc[dt, cols] = 1/len(cols)
    return w_t.ffill().fillna(0.0)

def weights_inverse_vol(r: pd.DataFrame, window:int, freq:str,
                        vol_method:str="rolling", vol_floor:float=0.0,
                        w_cap:Optional[float]=None, band:float=0.0) -> pd.DataFrame:
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    vol = rolling_vol_matrix(r, window, vol_method); prev=None
    for dt in rebs:
        if dt not in vol.index: continue
        v = vol.loc[dt].replace(0,np.nan)
        v = (v.fillna(v.median()).clip(lower=vol_floor) if vol_floor>0 else v.dropna())
        if v.empty: continue
        inv = (1.0/v).replace([np.inf,-np.inf], np.nan).dropna()
        if inv.empty: continue
        w = normalize_weights(inv, w_cap)
        if prev is not None and band>0:
            if float((w.reindex(prev.index).fillna(0)-prev.reindex(w.index).fillna(0)).abs().sum())<band:
                w = prev.copy()
        w_t.loc[dt, w.index] = w.values; prev = w.copy()
    return w_t.ffill().fillna(0.0)

def weights_erc(r: pd.DataFrame, window:int, freq:str,
                w_cap:Optional[float]=None, band:float=0.0, min_obs:int=40) -> pd.DataFrame:
    if not _HAS_SK:
        raise ImportError("ERC requires scikit-learn. pip install scikit-learn")
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(np.nan, index=r.index, columns=r.columns); prev=None
    for dt in rebs:
        rw = r.loc[:dt].tail(window)
        if rw.shape[0]<min_obs: continue
        cov = LedoitWolf().fit(rw.values).covariance_
        w = pd.Series(erc_weights(cov), index=rw.columns)
        w = normalize_weights(w, w_cap)
        if prev is not None and band>0:
            if float((w.reindex(prev.index).fillna(0)-prev.reindex(w.index).fillna(0)).abs().sum())<band:
                w = prev.copy()
        w_t.loc[dt, w.index] = w.values; prev = w.copy()
    return w_t.ffill().fillna(0.0)

# ---------- Vol targeting (weight-scaling) ----------
def vol_target_from_portfolio(ret_pre: pd.Series, target:float=0.10, span:int=63,
                              s_min:float=0.5, s_max:float=2.0, use_forecast:bool=True,
                              regime:bool=False, regime_thr:float=0.18,
                              target_lo:float=0.08, target_hi:float=0.12) -> Dict[str,pd.Series]:
    sigma = (ret_pre.pow(2).ewm(span=span, adjust=False).mean()
             .shift(1 if use_forecast else 0).pow(0.5))*np.sqrt(TRADING_DAYS)
    if regime:
        tgt = pd.Series(target_hi, index=ret_pre.index)
        tgt[sigma > regime_thr] = target_lo
    else:
        tgt = pd.Series(target, index=ret_pre.index)
    s = (tgt / sigma).clip(lower=s_min, upper=s_max).fillna(1.0)
    return {"s": s.rename("scaler"), "sigma": sigma.rename("sigma"), "tgt": tgt.rename("target")}

# ---------- Costs ----------
def naive_cost(turnover: pd.Series, tc_bps:int) -> pd.Series:
    return turnover * (tc_bps/10000.0)

def advanced_cost(turnover: pd.Series, spread_bps:float, vol_proxy:pd.Series, impact_coef:float=0.10) -> pd.Series:
    spread = turnover * (spread_bps/10000.0)
    impact = impact_coef * vol_proxy * np.sqrt(turnover.clip(lower=0.0))
    return (spread.fillna(0.0)+impact.fillna(0.0)).rename("cost_adv")

# ---------- Execution ----------
def to_daily_turnover(w: pd.DataFrame) -> pd.Series:
    return (w.diff().abs().sum(axis=1)).fillna(0.0).rename("turnover")

def instant_exec_returns(r: pd.DataFrame, w_targets_daily: pd.DataFrame, tc_bps:int, rf_annual:float) -> Tuple[pd.Series,pd.Series]:
    w = w_targets_daily.ffill().fillna(0.0)
    turn = to_daily_turnover(w)
    costs = naive_cost(turn, tc_bps)
    rf_d = rf_annual/TRADING_DAYS
    ret = (w*r).sum(axis=1) - costs - rf_d
    return ret.rename("ret_instant_simplebps"), turn.rename("turnover_instant")

def throttled_exec_returns(r: pd.DataFrame, w_targets_reb: pd.DataFrame, dates: pd.DatetimeIndex,
                           exec_speed:float=0.33, no_trade_band:float=0.002, turnover_cap:float=0.20,
                           spread_bps:float=2.0, impact_coef:float=0.10, impact_span:int=63,
                           rf_annual:float=0.02) -> Tuple[pd.Series,pd.Series,pd.DataFrame]:
    rebs = w_targets_reb.index
    cols = w_targets_reb.columns
    w_exec = pd.DataFrame(0.0, index=dates, columns=cols)
    w_curr = pd.Series(0.0, index=cols); turns = pd.Series(0.0, index=dates)
    vol_proxy = r.ewm(span=impact_span).std().mean(axis=1).fillna(0.0)
    for d in dates:
        if d in rebs:
            tgt = w_targets_reb.loc[d].fillna(0.0)
            gap = tgt - w_curr
            gap = gap.where(gap.abs()>=no_trade_band, 0.0)
            dv = exec_speed*gap
            turn = float(dv.abs().sum())
            if turnover_cap is not None and turn>turnover_cap and turn>0:
                dv *= (turnover_cap/turn); turn = turnover_cap
            w_curr = (w_curr + dv).clip(lower=0.0); s = w_curr.sum()
            if s>0: w_curr/=s
            turns.loc[d] = turn
        w_exec.loc[d] = w_curr.values
    costs = advanced_cost(turns.rename("turnover"), spread_bps, vol_proxy, impact_coef)
    rf_d = rf_annual/TRADING_DAYS
    ret = (w_exec*r).sum(axis=1) - costs - rf_d
    return ret.rename("ret_throttled_advanced"), turns.rename("turnover_throttled"), w_exec

# ---------- Driver ----------
def run(px: pd.DataFrame, mode:str, window:int, reb:str, rf_annual:float,
        vol_method:str, vol_floor:float, w_cap:Optional[float], band:float,
        vt_target:float, vt_span:int, vt_min:float, vt_max:float,
        vt_forecast:bool, vt_regime:bool, vt_thr:float, vt_lo:float, vt_hi:float,
        tc_bps:int, exec_speed:float, no_trade_band:float, turnover_cap:float,
        spread_bps:float, impact_coef:float, impact_span:int) -> Dict[str,object]:

    r = daily_returns(px)

    # Allocator targets (monthly/quarterly then ffilled)
    if mode=="ivol":
        w_alloc = weights_inverse_vol(r, window, reb, vol_method, vol_floor, w_cap, band)
    elif mode=="erc":
        w_alloc = weights_erc(r, window, reb, w_cap, band)
    else:
        w_alloc = weights_equal_weight(r, reb)

    # Pre-VT returns and scaler
    ret_pre = (w_alloc * r).sum(axis=1)
    vt = vol_target_from_portfolio(ret_pre, target=vt_target, span=vt_span,
                                   s_min=vt_min, s_max=vt_max, use_forecast=vt_forecast,
                                   regime=vt_regime, regime_thr=vt_thr,
                                   target_lo=vt_lo, target_hi=vt_hi)

    # Weight-scaled VT book
    s = vt["s"]
    w_vt_daily = w_alloc.mul(s, axis=0)  # keep leverage info; no renorm

    # Build VT targets at trading-day rebalances
    rebs = rebalance_dates(r.index, reb)
    w_vt_reb = w_vt_daily.loc[rebs].dropna(how="all")
    w_vt_daily_ff = w_vt_reb.reindex(r.index).ffill().fillna(0.0)

    # Instant vs Throttled
    ret_inst, turn_inst = instant_exec_returns(r, w_vt_daily_ff, tc_bps, rf_annual)
    ret_th, turn_th, w_exec = throttled_exec_returns(r, w_vt_reb, r.index,
                                                     exec_speed, no_trade_band, turnover_cap,
                                                     spread_bps, impact_coef, impact_span, rf_annual)

    return {
        "ret_instant": ret_inst, "turn_instant": turn_inst,
        "ret_throttled": ret_th, "turn_throttled": turn_th,
        "w_targets_reb": w_vt_reb, "w_instant_daily": w_vt_daily_ff, "w_exec": w_exec,
        "scaler": s, "sigma": vt["sigma"], "target": vt["tgt"]
    }

def metrics_table(rets: Dict[str,pd.Series], rf_annual:float) -> pd.DataFrame:
    out={}
    for nm, x in rets.items():
        eq=(1+x).cumprod()
        out[nm]={"CAGR":cagr(x),"Vol":ann_vol(x),"Sharpe":sharpe(x,rf_annual),
                 "MaxDD":max_dd(eq),"Hit%":hit_rate(x)}
    return pd.DataFrame(out).T

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 11+12 Combined")
    p.add_argument("--tickers", type=str, default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    p.add_argument("--mode", type=str, default="erc", choices=["erc","ivol","ew"])
    p.add_argument("--window", type=int, default=126)
    p.add_argument("--rf", type=float, default=0.02)
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling","ewma"])
    p.add_argument("--vol_floor", type=float, default=0.0)
    p.add_argument("--w_cap", type=float, default=0.25)
    p.add_argument("--band", type=float, default=0.05)

    # VT
    p.add_argument("--vt_target", type=float, default=0.10)
    p.add_argument("--vt_span", type=int, default=63)
    p.add_argument("--vt_min", type=float, default=0.5)
    p.add_argument("--vt_max", type=float, default=2.0)
    p.add_argument("--vt_forecast", action="store_true")
    p.add_argument("--vt_regime", action="store_true")
    p.add_argument("--vt_thr", type=float, default=0.18)
    p.add_argument("--vt_target_lo", type=float, default=0.08)
    p.add_argument("--vt_target_hi", type=float, default=0.12)

    # Execution + costs
    p.add_argument("--tc_bps", type=int, default=10)
    p.add_argument("--exec_speed", type=float, default=0.33)
    p.add_argument("--no_trade_band", type=float, default=0.002)
    p.add_argument("--turnover_cap", type=float, default=0.20)
    p.add_argument("--spread_bps", type=float, default=2.0)
    p.add_argument("--impact_coef", type=float, default=0.10)
    p.add_argument("--impact_span", type=int, default=63)

    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode=="erc" and not _HAS_SK:
        raise SystemExit("ERC needs scikit-learn (pip install scikit-learn) or use --mode ivol/ew")
    tickers=[t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px=fetch_prices(tickers, a.start)

    res = run(px, a.mode, a.window, a.reb, a.rf,
              a.vol_method, a.vol_floor, a.w_cap, a.band,
              a.vt_target, a.vt_span, a.vt_min, a.vt_max,
              a.vt_forecast, a.vt_regime, a.vt_thr, a.vt_target_lo, a.vt_target_hi,
              a.tc_bps, a.exec_speed, a.no_trade_band, a.turnover_cap,
              a.spread_bps, a.impact_coef, a.impact_span)

    # Save
    out_csv=os.path.join(a.outdir,"outputs"); os.makedirs(out_csv,exist_ok=True)
    out_ch =os.path.join(a.outdir,"charts");  os.makedirs(out_ch, exist_ok=True)

    pd.concat([res["ret_instant"], res["ret_throttled"]], axis=1)\
      .to_csv(os.path.join(out_csv,"day11_12_returns.csv"))
    pd.concat([res["turn_instant"], res["turn_throttled"]], axis=1)\
      .to_csv(os.path.join(out_csv,"day11_12_turnover.csv"))
    res["w_targets_reb"].to_csv(os.path.join(out_csv,"day11_12_weights_targets_reb.csv"))
    res["w_instant_daily"].to_csv(os.path.join(out_csv,"day11_12_weights_instant_daily.csv"))
    res["w_exec"].to_csv(os.path.join(out_csv,"day11_12_weights_exec.csv"))
    pd.concat([res["scaler"], res["sigma"], res["target"]], axis=1)\
      .to_csv(os.path.join(out_csv,"day11_12_scaler_sigma_target.csv"))

    mt = metrics_table({"Instant_simplebps":res["ret_instant"],
                        "Throttled_advanced":res["ret_throttled"]}, a.rf).round(4)
    mt.to_csv(os.path.join(out_csv,"day11_12_metrics.csv"))

    # Chart
    eq_i=(1+res["ret_instant"]).cumprod().rename("Instant_simplebps")
    eq_t=(1+res["ret_throttled"]).cumprod().rename("Throttled_advanced")
    plt.figure(figsize=(12,6)); eq_i.plot(); eq_t.plot()
    plt.title("Day 11+12 — RP+VT: Instant vs Throttled (net)"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_ch,"day11_12_equity.png"), dpi=150); plt.close()

    print("\n=== Day 11+12 — Metrics (net) ===")
    print(mt.to_string())
    print("\nArtifacts written to:")
    for f in ["day11_12_returns.csv","day11_12_turnover.csv","day11_12_weights_targets_reb.csv",
              "day11_12_weights_instant_daily.csv","day11_12_weights_exec.csv",
              "day11_12_scaler_sigma_target.csv"]:
        print(" -", os.path.join(out_csv,f))
    print(" -", os.path.join(out_ch,"day11_12_equity.png"))

if __name__=="__main__":
    main()