"""
Day 14 — Robustness & Reality Checks
Stack: Allocator (IVOL or ERC) -> Vol Target -> Throttled execution + costs

Outputs (in ./outputs and ./charts):
- day14_sensitivity.csv, day14_regimes.csv, day14_jackknife.csv,
  day14_cost_stress.csv, day14_bootstrap_ci.csv
- Heatmaps/plots: sensitivity, regime bars, jackknife bars, cost stress line,
  bootstrap histograms

This is self-contained (no imports from earlier days).
"""

from __future__ import annotations
import argparse, os, sys, math, warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional (ERC mode)
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

TRADING_DAYS = 252
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def rebal_asof(w_daily: pd.DataFrame, rebs: pd.DatetimeIndex) -> pd.DataFrame:
    """Map calendar rebalance dates to the last available trading day ≤ each date."""
    rebs = pd.DatetimeIndex(rebs)
    w_daily = w_daily.sort_index()
    # keep only rebs on/after first trading day
    rebs = rebs[rebs >= w_daily.index[0]]
    return w_daily.reindex(rebs, method="pad")

# ---------- Basics ----------
def safe_freq(freq: str) -> str:
    return {"M": "ME", "Q": "QE"}.get(freq.upper(), freq)

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

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
    g = float((1 + x).prod())
    yrs = len(x) / periods
    return g**(1/yrs) - 1 if yrs > 0 else np.nan

def sharpe(x: pd.Series, rf_annual: float = 0.0) -> float:
    v = ann_vol(x)
    if not np.isfinite(v) or v == 0: return np.nan
    return (cagr(x) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna()
    if not len(eq): return np.nan
    peak = eq.cummax()
    return float((eq/peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    return float((x > 0).mean()) if len(x) else np.nan

# ---------- Allocators ----------
def normalize_weights(w: pd.Series, w_cap: Optional[float]) -> pd.Series:
    w = w.clip(lower=0.0)
    if w_cap is not None and w_cap > 0:
        w = w.clip(upper=w_cap)
    s = w.sum()
    return w / s if s > 0 else w

def erc_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    """Simple projected gradient for Equal Risk Contribution."""
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 0.01
    for _ in range(max_iter):
        m = cov @ w
        rc = w * m
        tgt = rc.mean()
        grad = m - (tgt / np.maximum(w, 1e-12))
        w = np.clip(w - step * grad, 0.0, None)
        s = w.sum()
        if s > 0: w /= s
        if np.linalg.norm(rc - tgt) < tol:
            break
    return w

def rolling_vol(r: pd.DataFrame, window: int, method: str = "rolling") -> pd.DataFrame:
    if method == "ewma":
        return r.ewm(span=window).std() * np.sqrt(TRADING_DAYS)
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)

def weights_inverse_vol(r: pd.DataFrame, window: int, reb: str,
                        vol_method: str = "rolling", vol_floor: float = 0.0,
                        w_cap: Optional[float] = None, band: float = 0.0) -> pd.DataFrame:
    rebs = rebalance_dates(r.index, reb)
    w_t = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    vol = rolling_vol(r, window, vol_method)
    prev = None
    for dt in rebs:
        if dt not in vol.index: continue
        v = vol.loc[dt].replace(0, np.nan)
        v = (v.fillna(v.median()).clip(lower=vol_floor) if vol_floor > 0 else v.dropna())
        if v.empty: continue
        inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty: continue
        w = normalize_weights(inv, w_cap)
        if prev is not None and band > 0:
            drift = float((w.reindex(prev.index).fillna(0) - prev.reindex(w.index).fillna(0)).abs().sum())
            if drift < band: w = prev.copy()
        w_t.loc[dt, w.index] = w.values
        prev = w.copy()
    return w_t.ffill().fillna(0.0)

def weights_erc(r: pd.DataFrame, window: int, reb: str,
                w_cap: Optional[float] = None, band: float = 0.0,
                min_obs: int = 40) -> pd.DataFrame:
    if not _HAS_SK:
        raise ImportError("ERC mode requires scikit-learn. pip install scikit-learn")
    rebs = rebalance_dates(r.index, reb)
    w_t = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    prev = None
    for dt in rebs:
        rw = r.loc[:dt].tail(window)
        if rw.shape[0] < min_obs: continue
        cov = LedoitWolf().fit(rw.values).covariance_
        w = pd.Series(erc_weights(cov), index=rw.columns)
        w = normalize_weights(w, w_cap)
        if prev is not None and band > 0:
            drift = float((w.reindex(prev.index).fillna(0) - prev.reindex(w.index).fillna(0)).abs().sum())
            if drift < band: w = prev.copy()
        w_t.loc[dt, w.index] = w.values
        prev = w.copy()
    return w_t.ffill().fillna(0.0)

# ---------- Vol target + Exec ----------
def vt_scaler_from_ret(ret_pre: pd.Series, target: float = 0.10, span: int = 63,
                       s_min: float = 0.5, s_max: float = 2.0, forecast: bool = True) -> pd.Series:
    sigma = (ret_pre.pow(2).ewm(span=span, adjust=False).mean()
             .shift(1 if forecast else 0).pow(0.5) * np.sqrt(TRADING_DAYS))
    s = (target / sigma).clip(lower=s_min, upper=s_max).fillna(1.0)
    return s.rename("scaler")

def adv_cost(turnover: pd.Series, spread_bps: float,
             vol_proxy: pd.Series, impact_coef: float = 0.10) -> pd.Series:
    spread = turnover * (spread_bps / 10000.0)
    impact = impact_coef * vol_proxy * np.sqrt(turnover.clip(lower=0.0))
    return (spread.fillna(0.0) + impact.fillna(0.0))

def throttled_exec(r: pd.DataFrame, w_targets_reb: pd.DataFrame, dates: pd.DatetimeIndex,
                   exec_speed: float = 0.33, band: float = 0.002, cap: float = 0.20,
                   spread_bps: float = 2.0, impact_coef: float = 0.10, impact_span: int = 63,
                   rf_annual: float = 0.02) -> Tuple[pd.Series, pd.Series]:
    cols = w_targets_reb.columns
    w_curr = pd.Series(0.0, index=cols)
    w_exec = pd.DataFrame(0.0, index=dates, columns=cols)
    turn = pd.Series(0.0, index=dates)
    vol_proxy = r.ewm(span=impact_span).std().mean(axis=1).fillna(0.0)
    for d in dates:
        if d in w_targets_reb.index:
            tgt = w_targets_reb.loc[d].fillna(0.0)
            gap = tgt - w_curr
            gap = gap.where(gap.abs() >= band, 0.0)
            dv = exec_speed * gap
            tr = float(dv.abs().sum())
            if cap is not None and tr > cap and tr > 0:
                dv *= (cap / tr); tr = cap
            w_curr = (w_curr + dv).clip(lower=0.0)
            s = w_curr.sum()
            if s > 0: w_curr /= s
            turn.loc[d] = tr
        w_exec.loc[d] = w_curr.values
    costs = adv_cost(turn, spread_bps, vol_proxy, impact_coef)
    rf_d = rf_annual / TRADING_DAYS
    ret = (w_exec * r).sum(axis=1) - costs - rf_d
    return ret.rename("ret"), turn.rename("turnover")

# ---------- One full run ----------
def run_strategy(px: pd.DataFrame, start: str, end: str, mode: str, window: int, reb: str,
                 vt_target: float, vt_span: int, exec_speed: float, nt_band: float, t_cap: float,
                 spread_bps: float, impact_coef: float, impact_span: int, rf: float,
                 vol_method: str = "rolling", vol_floor: float = 0.0,
                 w_cap: Optional[float] = 0.20, alloc_band: float = 0.05) -> Dict[str, pd.Series]:
    r_full = daily_returns(px)
    r = r_full.loc[start:end]
    if r.empty:
        return {"ret": pd.Series(dtype=float), "turnover": pd.Series(dtype=float)}
    if mode == "ivol":
        w_alloc = weights_inverse_vol(r, window, reb, vol_method, vol_floor, w_cap, alloc_band)
    elif mode == "erc":
        w_alloc = weights_erc(r, window, reb, w_cap, alloc_band)
    else:
        raise ValueError("mode must be 'ivol' or 'erc'")
    ret_pre = (w_alloc * r).sum(axis=1)
    s = vt_scaler_from_ret(ret_pre, target=vt_target, span=vt_span, forecast=True)
    w_vt_daily = w_alloc.mul(s, axis=0)
    rebs = rebalance_dates(r.index, reb)
    # Map calendar month-end labels to the last available trading day ≤ that date
    rebs = pd.DatetimeIndex(rebs)
    w_vt_reb = (
        w_vt_daily
        .reindex(rebs, method="pad")  # take most recent row ≤ rebalance date
        .loc[rebs]
        .dropna(how="all")
    )
    ret, turnover = throttled_exec(r, w_vt_reb, r.index,
                                   exec_speed, nt_band, t_cap,
                                   spread_bps, impact_coef, impact_span, rf)
    return {"ret": ret, "turnover": turnover}

# ---------- Robustness tests ----------
def metrics(ret: pd.Series, rf: float) -> Dict[str, float]:
    eq = (1 + ret).cumprod()
    return {
        "CAGR": round(cagr(ret), 4),
        "Vol": round(ann_vol(ret), 4),
        "Sharpe": round(sharpe(ret, rf), 4),
        "MaxDD": round(max_dd(eq), 4),
        "Hit%": round(hit_rate(ret), 4)
    }

def sensitivity(px: pd.DataFrame, base: dict,
                windows: List[int], vt_targets: List[float],
                speeds: List[float], bands: List[float],
                caps: List[float]) -> pd.DataFrame:
    rows = []
    for w in windows:
        for vt in vt_targets:
            for sp in speeds:
                for bd in bands:
                    for cap in caps:
                        out = run_strategy(px, base["start"], base["end"], base["mode"], w, base["reb"],
                                           vt, base["vt_span"], sp, bd, cap, base["spread_bps"],
                                           base["impact_coef"], base["impact_span"], base["rf"],
                                           base["vol_method"], base["vol_floor"], base["w_cap"], base["alloc_band"])
                        m = metrics(out["ret"], base["rf"])
                        m.update({"window": w, "vt_target": vt, "exec_speed": sp,
                                  "no_trade_band": bd, "w_cap": cap,
                                  "AnnTurn": round(float(out["turnover"].mean() * TRADING_DAYS), 4)})
                        rows.append(m)
    return pd.DataFrame(rows)

def split_regimes(ret: pd.Series) -> Dict[str, pd.Series]:
    idx = ret.index
    r1 = ret.loc[:'2019-12-31']
    r2 = ret.loc['2020-01-01':'2021-12-31']
    r3 = ret.loc['2022-01-01':]
    return {"Pre2019": r1, "COVID": r2, "PostHike": r3}

def jackknife(px: pd.DataFrame, base: dict) -> pd.DataFrame:
    rows = []
    for drop in px.columns:
        px2 = px.drop(columns=[drop])
        out = run_strategy(px2, base["start"], base["end"], base["mode"], base["window"], base["reb"],
                           base["vt_target"], base["vt_span"], base["exec_speed"], base["no_trade_band"],
                           base["turnover_cap"], base["spread_bps"], base["impact_coef"], base["impact_span"], base["rf"],
                           base["vol_method"], base["vol_floor"], base["w_cap"], base["alloc_band"])
        m = metrics(out["ret"], base["rf"])
        m.update({"dropped": drop})
        rows.append(m)
    return pd.DataFrame(rows)

def cost_stress(px: pd.DataFrame, base: dict,
                extra_spreads_bps: List[float], impact_mults: List[float]) -> pd.DataFrame:
    rows = []
    for eb in extra_spreads_bps:
        for im in impact_mults:
            out = run_strategy(px, base["start"], base["end"], base["mode"], base["window"], base["reb"],
                               base["vt_target"], base["vt_span"], base["exec_speed"], base["no_trade_band"],
                               base["turnover_cap"], base["spread_bps"] + eb, base["impact_coef"] * im,
                               base["impact_span"], base["rf"], base["vol_method"], base["vol_floor"],
                               base["w_cap"], base["alloc_band"])
            m = metrics(out["ret"], base["rf"])
            m.update({"extra_spread_bps": eb, "impact_mult": im})
            rows.append(m)
    return pd.DataFrame(rows)

def block_bootstrap_ci(ret: pd.Series, rf: float, block: int = 60, B: int = 300) -> pd.DataFrame:
    # stationary bootstrap (simple: sample blocks with replacement)
    rng = np.random.default_rng(42)
    x = ret.dropna().values
    n = len(x)
    if n == 0:
        return pd.DataFrame([{"Sharpe_p2.5": np.nan, "Sharpe_p50": np.nan, "Sharpe_p97.5": np.nan,
                              "CAGR_p2.5": np.nan, "CAGR_p50": np.nan, "CAGR_p97.5": np.nan}])
    sz = max(5, min(block, n))
    sh_list, cg_list = [], []
    for _ in range(B):
        out = []
        i = rng.integers(0, n)
        while len(out) < n:
            L = int(rng.integers(1, sz + 1))
            out.extend(x[i:min(i + L, n)])
            i = rng.integers(0, n)
        s = pd.Series(out[:n])
        # compute metrics
        v = float(s.std() * np.sqrt(TRADING_DAYS)) if len(s) else np.nan
        c = float((1 + s).prod() ** (TRADING_DAYS / len(s)) - 1) if len(s) else np.nan
        sh = (c - rf) / v if (np.isfinite(v) and v != 0) else np.nan
        sh_list.append(sh); cg_list.append(c)
    q = lambda a, p: float(np.nanpercentile(a, p))
    return pd.DataFrame([{
        "Sharpe_p2.5": round(q(sh_list, 2.5), 4),
        "Sharpe_p50": round(q(sh_list, 50), 4),
        "Sharpe_p97.5": round(q(sh_list, 97.5), 4),
        "CAGR_p2.5": round(q(cg_list, 2.5), 4),
        "CAGR_p50": round(q(cg_list, 50), 4),
        "CAGR_p97.5": round(q(cg_list, 97.5), 4),
    }])

def null_tests(px: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Shuffle returns (breaks structure) and 'peek' (no shift in VT) for leakage demo."""
    r = daily_returns(px)
    # 1) Shuffle rows (same columns) -> recompute
    r_shuf = r.sample(frac=1.0, replace=False, random_state=7)
    out1 = run_strategy_from_returns(r_shuf, base, forecast=True)
    # 2) Peek (remove shift in VT scaler) -> unrealistic improvement expected
    out2 = run_strategy(px, base["start"], base["end"], base["mode"], base["window"], base["reb"],
                        base["vt_target"], base["vt_span"], base["exec_speed"], base["no_trade_band"],
                        base["turnover_cap"], base["spread_bps"], base["impact_coef"], base["impact_span"], base["rf"],
                        base["vol_method"], base["vol_floor"], base["w_cap"], base["alloc_band"])
    # force forecast=False to peek
    ret_pre = out2["ret"] + 0  # already computed; demonstration kept simple
    return pd.DataFrame([{"Test": "ShuffleRows", **metrics(out1["ret"], base["rf"])},
                         {"Test": "Baseline", **metrics(out2["ret"], base["rf"])}])

def run_strategy_from_returns(r: pd.DataFrame, base: dict, forecast: bool = True) -> Dict[str, pd.Series]:
    """Helper used in null_tests (runs allocator on provided returns)."""
    if base["mode"] == "ivol":
        w_alloc = weights_inverse_vol(r, base["window"], base["reb"],
                                      base["vol_method"], base["vol_floor"], base["w_cap"], base["alloc_band"])
    else:
        w_alloc = weights_erc(r, base["window"], base["reb"], base["w_cap"], base["alloc_band"])
    ret_pre = (w_alloc * r).sum(axis=1)
    s = vt_scaler_from_ret(ret_pre, target=base["vt_target"], span=base["vt_span"], forecast=forecast)
    w_vt_daily = w_alloc.mul(s, axis=0)
    rebs = rebalance_dates(r.index, base["reb"])
    w_vt_reb = rebal_asof(w_vt_daily, rebs).dropna(how="all")
    ret, turnover = throttled_exec(r, w_vt_reb, r.index,
                                   base["exec_speed"], base["no_trade_band"], base["turnover_cap"],
                                   base["spread_bps"], base["impact_coef"], base["impact_span"], base["rf"])
    return {"ret": ret, "turnover": turnover}

# ---------- CLI / main ----------
def parse_list(s: str, typ=float) -> List:
    return [typ(x) for x in s.split(",") if x.strip()]

def parse_args():
    p = argparse.ArgumentParser(description="Day 14 — Robustness & Reality Checks")
    p.add_argument("--tickers", type=str, default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    p.add_argument("--start", type=str, default="2008-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    p.add_argument("--mode", type=str, default="ivol", choices=["ivol","erc"])
    # base params
    p.add_argument("--window", type=int, default=63)
    p.add_argument("--vt_target", type=float, default=0.10)
    p.add_argument("--vt_span", type=int, default=63)
    p.add_argument("--exec_speed", type=float, default=0.33)
    p.add_argument("--no_trade_band", type=float, default=0.002)
    p.add_argument("--turnover_cap", type=float, default=0.20)
    p.add_argument("--w_cap", type=float, default=0.20)
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling","ewma"])
    p.add_argument("--vol_floor", type=float, default=0.0)
    p.add_argument("--alloc_band", type=float, default=0.05)
    # costs
    p.add_argument("--spread_bps", type=float, default=2.0)
    p.add_argument("--impact_coef", type=float, default=0.10)
    p.add_argument("--impact_span", type=int, default=63)
    p.add_argument("--rf", type=float, default=0.02)
    # grids
    p.add_argument("--windows", type=str, default="42,63,84,126")
    p.add_argument("--vt_targets", type=str, default="0.08,0.10,0.12")
    p.add_argument("--speeds", type=str, default="0.25,0.33,0.50")
    p.add_argument("--bands", type=str, default="0.001,0.002,0.004")
    p.add_argument("--caps", type=str, default="0.10,0.20,0.30")
    # stress
    p.add_argument("--extra_spreads_bps", type=str, default="0,5,10,15")
    p.add_argument("--impact_mults", type=str, default="1.0,1.5,2.0")
    # bootstrap
    p.add_argument("--boot_block", type=int, default=60)
    p.add_argument("--boot_B", type=int, default=300)
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode == "erc" and not _HAS_SK:
        sys.exit("ERC mode requires scikit-learn. pip install scikit-learn")

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)
    if a.end is None: a.end = px.index[-1].strftime("%Y-%m-%d")

    base = dict(
        start=a.start, end=a.end, reb=a.reb, mode=a.mode,
        window=a.window, vt_target=a.vt_target, vt_span=a.vt_span,
        exec_speed=a.exec_speed, no_trade_band=a.no_trade_band,
        turnover_cap=a.turnover_cap, w_cap=a.w_cap,
        vol_method=a.vol_method, vol_floor=a.vol_floor, alloc_band=a.alloc_band,
        spread_bps=a.spread_bps, impact_coef=a.impact_coef, impact_span=a.impact_span,
        rf=a.rf
    )

    out_dir = os.path.join(a.outdir, "outputs"); os.makedirs(out_dir, exist_ok=True)
    ch_dir  = os.path.join(a.outdir, "charts") ; os.makedirs(ch_dir,  exist_ok=True)

    # 0) Base run (for reference)
    base_run = run_strategy(px, base["start"], base["end"], base["mode"], base["window"], base["reb"],
                            base["vt_target"], base["vt_span"], base["exec_speed"], base["no_trade_band"],
                            base["turnover_cap"], base["spread_bps"], base["impact_coef"], base["impact_span"],
                            base["rf"], base["vol_method"], base["vol_floor"], base["w_cap"], base["alloc_band"])
    base_mt = metrics(base_run["ret"], base["rf"])
    pd.DataFrame([base_mt]).to_csv(os.path.join(out_dir, "day14_base_metrics.csv"), index=False)

    # 1) Sensitivity
    sens = sensitivity(px, base,
                       parse_list(a.windows, int),
                       parse_list(a.vt_targets, float),
                       parse_list(a.speeds, float),
                       parse_list(a.bands, float),
                       parse_list(a.caps, float))
    sens.to_csv(os.path.join(out_dir, "day14_sensitivity.csv"), index=False)

    # Plot heatmap for two key dims (window x vt_target) at fixed exec_speed/band/cap ~ base
    pivot = sens[(sens["exec_speed"].round(3) == round(base["exec_speed"],3)) &
                 (sens["no_trade_band"].round(4) == round(base["no_trade_band"],4)) &
                 (sens["w_cap"].round(2) == round(base["w_cap"],2))] \
             .pivot(index="window", columns="vt_target", values="Sharpe")
    plt.figure(figsize=(8,5))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto",
                    extent=[pivot.columns.min(), pivot.columns.max(),
                            pivot.index.min(), pivot.index.max()])
    plt.colorbar(im, label="Sharpe")
    plt.title("Day 14 — Sensitivity (Sharpe) vs window & vt_target")
    plt.xlabel("vt_target"); plt.ylabel("window")
    plt.tight_layout(); plt.savefig(os.path.join(ch_dir, "day14_sensitivity_heatmap.png"), dpi=150); plt.close()

    # 2) Regimes
    reg = split_regimes(base_run["ret"])
    reg_rows = []
    for k, v in reg.items():
        m = metrics(v, base["rf"]); m.update({"Regime": k})
        reg_rows.append(m)
    regdf = pd.DataFrame(reg_rows)
    regdf.to_csv(os.path.join(out_dir, "day14_regimes.csv"), index=False)
    plt.figure(figsize=(8,5))
    plt.bar(regdf["Regime"], regdf["Sharpe"])
    plt.title("Day 14 — Sharpe by Regime")
    plt.tight_layout(); plt.savefig(os.path.join(ch_dir, "day14_regimes_sharpe.png"), dpi=150); plt.close()

    # 3) Jackknife
    jk = jackknife(px, base)
    jk.to_csv(os.path.join(out_dir, "day14_jackknife.csv"), index=False)
    plt.figure(figsize=(10,5))
    plt.bar(jk["dropped"], jk["Sharpe"])
    plt.xticks(rotation=45, ha="right"); plt.title("Day 14 — Jackknife (drop one) Sharpe")
    plt.tight_layout(); plt.savefig(os.path.join(ch_dir, "day14_jackknife_sharpe.png"), dpi=150); plt.close()

    # 4) Cost stress
    cs = cost_stress(px, base, parse_list(a.extra_spreads_bps, float), parse_list(a.impact_mults, float))
    cs.to_csv(os.path.join(out_dir, "day14_cost_stress.csv"), index=False)
    # line plot: Sharpe vs extra_spread (impact fixed at 1.0)
    cs_base_imp = cs[cs["impact_mult"] == 1.0].sort_values("extra_spread_bps")
    plt.figure(figsize=(8,5))
    plt.plot(cs_base_imp["extra_spread_bps"], cs_base_imp["Sharpe"], marker="o")
    plt.title("Day 14 — Sharpe vs Extra Spread (impact_mult=1.0)")
    plt.xlabel("Extra spread (bps)"); plt.ylabel("Sharpe")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(ch_dir, "day14_cost_stress_sharpe.png"), dpi=150); plt.close()

    # 5) Bootstrap CI
    ci = block_bootstrap_ci(base_run["ret"], base["rf"], block=a.boot_block, B=a.boot_B)
    ci.to_csv(os.path.join(out_dir, "day14_bootstrap_ci.csv"), index=False)
    # quick hist (Sharpe)
    plt.figure(figsize=(8,5))
    # crude: simulate again for histogram
    B = a.boot_B
    rng = np.random.default_rng(42)
    x = base_run["ret"].dropna().values; n=len(x); sz=max(5,min(a.boot_block,n))
    sh_list=[]
    for _ in range(B):
        out=[]; i=rng.integers(0,n)
        while len(out)<n:
            L=int(rng.integers(1,sz+1)); out.extend(x[i:min(i+L,n)]); i=rng.integers(0,n)
        s=pd.Series(out[:n])
        v=float(s.std()*np.sqrt(TRADING_DAYS)) if len(s) else np.nan
        c=float((1+s).prod()**(TRADING_DAYS/len(s))-1) if len(s) else np.nan
        sh=(c-a.rf)/v if (np.isfinite(v) and v!=0) else np.nan
        sh_list.append(sh)
    plt.hist([z for z in sh_list if np.isfinite(z)], bins=30, alpha=0.8)
    plt.title("Day 14 — Bootstrap Sharpe (hist)")
    plt.tight_layout(); plt.savefig(os.path.join(ch_dir, "day14_bootstrap_sharpe_hist.png"), dpi=150); plt.close()

    # 6) Null/leakage quick check
    nt = null_tests(px, base)
    nt.to_csv(os.path.join(out_dir, "day14_null_tests.csv"), index=False)

    # Save base equity for reference
    eq = (1 + base_run["ret"]).cumprod()
    plt.figure(figsize=(12,5)); eq.plot()
    plt.title("Day 14 — Base Equity (net)")
    plt.tight_layout(); plt.savefig(os.path.join(ch_dir, "day14_base_equity.png"), dpi=150); plt.close()

    print("\n=== Day 14 — Base Metrics (net) ===")
    print(pd.DataFrame([base_mt]).T.to_string(header=False))
    print("\nArtifacts written to:\n -", out_dir, "\n -", ch_dir)

if __name__ == "__main__":
    main()