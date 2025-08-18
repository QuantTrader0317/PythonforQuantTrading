# Day 15 — Macro/Regime Switch: Rates + Volatility (hardened)
from __future__ import annotations
import argparse, os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

TRADING_DAYS = 252

# Ledoit–Wolf for ERC
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ----------------- Utils -----------------
def safe_freq(freq: str) -> str:
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    # Expect DataFrame with 'Close' (and possibly multiindex if many tickers)
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            df = df["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()

def fetch_single_close(symbol: str, start: str) -> pd.Series:
    """Robustly fetch a single symbol's Close as a 1-D Series, sorted, float, business frequency."""
    df = yf.download(symbol, start=start, progress=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    # If it's already a Series
    if isinstance(df, pd.Series):
        s = df.dropna()
    else:
        # DataFrame cases
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            # MultiIndex or odd return shapes
            try:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
            except Exception:
                s = df.iloc[:, 0]

        # If still 2-D, squeeze safely
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()

    # Final cleanups
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().asfreq("B").ffill()

    # Guarantee 1-D Series
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s.values, index=s.index, dtype=float)

# --------------- Metrics -----------------
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

def normalize_weights(w: pd.Series, w_cap: Optional[float]) -> pd.Series:
    w = w.clip(lower=0.0)
    if w_cap is not None and w_cap > 0:
        w = w.clip(upper=w_cap)
    s = w.sum()
    return w / s if s > 0 else w

# --------------- ERC weights ---------------
def erc_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    step = 0.01
    for _ in range(max_iter):
        m = cov @ w
        rc = w * m
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

def weights_erc(r: pd.DataFrame,
                window: int,
                reb: str,
                w_cap: Optional[float] = None,
                band: float = 0.0,
                min_obs: int = 40) -> pd.DataFrame:
    if not _HAS_SK:
        raise ImportError("scikit-learn is required for ERC mode (pip install scikit-learn).")
    rebs = rebalance_dates(r.index, reb)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    prev_w = None
    for dt in rebs:
        r_window = r.loc[:dt].tail(window)
        if r_window.shape[0] < min_obs:
            continue
        tickers = r_window.columns
        lw = LedoitWolf().fit(r_window.values)
        cov = lw.covariance_
        w_np = erc_weights(cov)
        w = pd.Series(w_np, index=tickers)
        w = normalize_weights(w, w_cap)
        if prev_w is not None:
            w_al = w.reindex(prev_w.index).fillna(0.0)
            prev_al = prev_w.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if band > 0 and drift < band:
                w = prev_w.copy()
        w_t.loc[dt, w.index] = w.values
        prev_w = w.copy()
    return w_t.ffill().fillna(0.0)

# --------------- Vol targeting ---------------
def ewma_vol(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).std() * np.sqrt(TRADING_DAYS)

def vol_target_scaler(ret: pd.Series,
                      target: float,
                      span: int,
                      s_min: float = 0.5,
                      s_max: float = 1.5) -> pd.Series:
    rv = ewma_vol(ret.fillna(0.0), span=span)
    s = target / rv.replace(0.0, np.nan)
    s = s.clip(lower=s_min, upper=s_max).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s

# --------------- Regime signals ---------------
def get_regime(index: pd.DatetimeIndex,
               use_tnx: bool = True, tnx_span: int = 126,
               use_vix: bool = True, vix_span: int = 90,
               host_or: bool = True) -> pd.Series:
    """Return boolean Series (True = hostile), aligned to 'index'."""
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.Series(False, index=index)

    # Rates
    if use_tnx:
        tnx = fetch_single_close("^TNX", start=str(index[0].date()))
        tnx_sma = tnx.rolling(tnx_span, min_periods=max(5, tnx_span // 2)).mean()
        rates_host = (tnx > tnx_sma).reindex(index, method="pad").fillna(False).astype(bool)
    else:
        rates_host = pd.Series(False, index=index)

    # Vol
    if use_vix:
        vix = fetch_single_close("^VIX", start=str(index[0].date()))
        vix_sma = vix.rolling(vix_span, min_periods=max(5, vix_span // 2)).mean()
        vol_host = (vix > vix_sma).reindex(index, method="pad").fillna(False).astype(bool)
    else:
        vol_host = pd.Series(False, index=index)

    # Combine robustly as 1-D Series
    host = (rates_host | vol_host) if host_or else (rates_host & vol_host)
    if isinstance(host, pd.DataFrame):
        host = host.any(axis=1)
    return pd.Series(host.values.astype(bool), index=index)

# --------------- Runner ---------------
def run(px: pd.DataFrame,
        reb: str,
        window: int,
        rf_annual: float,
        tc_bps: int,
        w_cap: Optional[float],
        band: float,
        vt_target: float,
        vt_span: int,
        vt_min: float,
        vt_max: float,
        # regime knobs
        use_tnx: bool,
        tnx_span: int,
        use_vix: bool,
        vix_span: int,
        host_or: bool,
        hostile_reduce: float,
        outdir: str) -> Dict[str, pd.Series]:

    r = daily_returns(px).sort_index()
    if r.empty:
        raise ValueError("No returns computed from prices.")

    # ERC weights (rebalance level), then ffill daily
    w = weights_erc(r, window, reb, w_cap=w_cap, band=band)

    # base daily returns (before scaling)
    rf_daily = rf_annual / TRADING_DAYS
    costs = turnover_costs(w, tc_bps).reindex(r.index).fillna(0.0)
    ret_raw = (w * r).sum(axis=1).reindex(r.index).fillna(0.0) - costs - rf_daily

    # BASE constant target
    s_base = vol_target_scaler(ret_raw, vt_target, vt_span, vt_min, vt_max)
    ret_base = ret_raw * s_base
    eq_base = (1 + ret_base).cumprod()

    # Regime series aligned to ret_raw index
    hostile = get_regime(r.index, use_tnx, tnx_span, use_vix, vix_span, host_or)
    hostile = hostile.reindex(ret_raw.index, method="pad").fillna(False).astype(bool)

    # Dynamic target (multiplier, no .loc assignment)
    mult = pd.Series(1.0, index=ret_raw.index)
    mult = mult.where(~hostile, hostile_reduce)
    dyn_target = vt_target * mult

    # Realized vol & scaler (aligned)
    rv = ewma_vol(ret_raw, span=vt_span).reindex(ret_raw.index)
    s_reg = (dyn_target / rv.replace(0.0, np.nan)) \
                .clip(lower=vt_min, upper=vt_max) \
                .replace([np.inf, -np.inf], np.nan) \
                .fillna(0.0)

    ret_reg = ret_raw * s_reg
    eq_reg = (1 + ret_reg).cumprod()

    # ---- outputs ----
    os.makedirs(os.path.join(outdir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "outputs"), exist_ok=True)

    pd.DataFrame({"ret_base": ret_base, "ret_regime": ret_reg}).to_csv(
        os.path.join(outdir, "outputs", "day15_daily_returns.csv"))
    pd.DataFrame({"eq_base": eq_base, "eq_regime": eq_reg}).to_csv(
        os.path.join(outdir, "outputs", "day15_equity_curves.csv"))
    w.to_csv(os.path.join(outdir, "outputs", "day15_weights_daily.csv"))
    pd.DataFrame({"hostile": hostile.astype(int)}).to_csv(
        os.path.join(outdir, "outputs", "day15_regime_series.csv"))
    pd.DataFrame({"s_base": s_base, "s_regime": s_reg}).to_csv(
        os.path.join(outdir, "outputs", "day15_scalers.csv"))

    # charts
    plt.figure(figsize=(12,5))
    plt.plot(eq_base.index, eq_base, label="Base (const target)")
    plt.plot(eq_reg.index, eq_reg, label="Regime-aware")
    plt.title("Day 15 — Equity (Base vs Regime-aware)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "day15_equity.png"), dpi=150)

    def dd(e):
        p = e.cummax(); return e/p - 1.0

    plt.figure(figsize=(12,3))
    plt.plot(eq_base.index, dd(eq_base), label="Base")
    plt.plot(eq_reg.index, dd(eq_reg), label="Regime")
    plt.title("Day 15 — Drawdown (Base vs Regime)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "day15_drawdown.png"), dpi=150)

    plt.figure(figsize=(12,2.5))
    plt.plot(hostile.index, hostile.astype(int))
    plt.ylim(-0.1, 1.1); plt.title("Day 15 — Hostile Regime (1 = yes)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "day15_regime_timeline.png"), dpi=150)

    def table(name, ret):
        eq = (1+ret).cumprod()
        return pd.Series({
            "CAGR": cagr(ret),
            "Vol": ann_vol(ret),
            "Sharpe": sharpe(ret, rf_annual),
            "MaxDD": max_dd(eq),
            "Hit%": hit_rate(ret)
        }, name=name)
    mt = pd.concat([table("Base", ret_base), table("Regime", ret_reg)], axis=1)
    mt.to_csv(os.path.join(outdir, "outputs", "day15_metrics.csv"))

    print("\n=== Day 15 Metrics (net) ===")
    print(mt.round(4))
    print("\nArtifacts written to ./outputs and ./charts")
    return {"ret_base": ret_base, "ret_reg": ret_reg, "hostile": hostile}

# --------------- CLI ---------------
def parse_args():
    p = argparse.ArgumentParser(description="Day 15 — Macro/Regime Switch for RP+VolTarget")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    p.add_argument("--window", type=int, default=126)
    p.add_argument("--rf", type=float, default=0.02)
    p.add_argument("--tc_bps", type=int, default=10)
    p.add_argument("--w_cap", type=float, default=0.25)
    p.add_argument("--band", type=float, default=0.002)
    # vol targeting
    p.add_argument("--vt_target", type=float, default=0.10)
    p.add_argument("--vt_span", type=int, default=126)
    p.add_argument("--vt_min", type=float, default=0.5)
    p.add_argument("--vt_max", type=float, default=1.5)
    # regime
    p.add_argument("--use_tnx", action="store_true", default=True)
    p.add_argument("--tnx_span", type=int, default=126)
    p.add_argument("--use_vix", action="store_true", default=True)
    p.add_argument("--vix_span", type=int, default=90)
    p.add_argument("--host_or", action="store_true", default=True)
    p.add_argument("--reduce", type=float, default=0.6, help="hostile target = base_target * reduce")
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    if not _HAS_SK:
        raise SystemExit("scikit-learn is required (ERC). pip install scikit-learn")
    px = fetch_prices(tickers, a.start)
    run(px, a.reb, a.window, a.rf, a.tc_bps, a.w_cap, a.band,
        a.vt_target, a.vt_span, a.vt_min, a.vt_max,
        a.use_tnx, a.tnx_span, a.use_vix, a.vix_span, a.host_or, a.reduce, a.outdir)

if __name__ == "__main__":
    main()
