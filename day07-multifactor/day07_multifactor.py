"""
Day 7 — Multi-Factor Cross-Sectional Strategy (Momentum + Volatility Filter)

What this does
--------------
- Builds two factors on a universe of ETFs:
  * 12-1 Momentum: (P[t-21] / P[t-252]) - 1   (skip last month)
  * 3M Volatility: rolling std of daily returns over 63 trading days
- Cross-sectionally ranks factors at each rebalance date and combines into a composite score:
    Score = w_mom * rank(mom) + w_vol * rank(-vol)  (lower vol = better)
- Selects top-N by composite score and allocates either equal-weight or inverse-vol.
- Applies transaction costs (bps on turnover) and cash drag (rf).
- Outputs CSVs + equity curve and an Information Coefficient (IC) series that
  measures cross-sectional predictive power of the composite each rebalance.

Usage example
-------------
  python day07_multifactor.py
  python day07_multifactor.py --top 3 --w_mom 0.7 --w_vol 0.3 --weighting inv_vol --reb M --rf 0.02 --tc_bps 10
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

TRADING_DAYS = 252

# ---------- Helpers / Metrics ----------

def safe_freq(freq: str) -> str:
    """Map deprecated pandas resample codes to replacements."""
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

def ann_vol(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(x.std() * np.sqrt(periods))

def cagr(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    gr = float((1 + x).prod())
    yrs = len(x) / periods
    if yrs <= 0:
        return np.nan
    return gr ** (1 / yrs) - 1

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if v == 0 or np.isnan(v):
        return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return float((x > 0).mean())


# ---------- Data / Factor Construction ----------

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def compute_factors(px: pd.DataFrame, lookback_mom: int, skip_days: int, vol_window: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      r  : daily returns
      mom: momentum signal (12-1 style)
      vol: rolling 3M volatility (std of daily returns)
    """
    r = px.pct_change().dropna()
    mom = (px.shift(skip_days) / px.shift(lookback_mom) - 1.0)
    vol = r.rolling(vol_window).std()
    return r, mom, vol

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def xsection_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0,1] per date (rows)."""
    return df.rank(axis=1, pct=True)


# ---------- Portfolio Construction ----------

def build_weights_from_scores(scores: pd.DataFrame,
                              r: pd.DataFrame,
                              top_n: int,
                              freq: str,
                              weighting: str = "equal",
                              inv_vol_window: int = 63) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given cross-sectional composite scores, pick top-N at each rebalance and build daily weights.

    Returns:
      w     : daily weights (ffilled between rebalances)
      picks : DataFrame indexed by rebalance date with columns rank1, rank2, ...
    """
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    picks_rows = []

    # Precompute inverse-vol weights base
    if weighting == "inv_vol":
        vol_tr = r.rolling(inv_vol_window).std().replace(0, np.nan)

    for dt in rebs:
        if dt not in scores.index:
            continue

        s = scores.loc[dt].dropna()
        if s.empty:
            continue

        # Sort desc by composite; deterministic tie-breaker by ticker
        s = s.sort_values(ascending=False)
        s = s.groupby(s.values, group_keys=False).apply(lambda x: x.sort_index())
        chosen = list(s.index[:top_n])
        if not chosen:
            continue

        if weighting == "equal":
            w_sel = pd.Series(1.0 / len(chosen), index=chosen)
        elif weighting == "inv_vol":
            # smaller vol => higher weight
            vol_slice = vol_tr.loc[dt, chosen]
            vol_slice = vol_slice.replace(0, np.nan)
            inv = 1.0 / vol_slice
            inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
            if inv.empty:
                w_sel = pd.Series(1.0 / len(chosen), index=chosen)
            else:
                inv = inv / inv.sum()
                # ensure all chosen get something (fallback equal if any NaNs)
                w_sel = inv.reindex(chosen).fillna(0.0)
                if w_sel.sum() <= 0:
                    w_sel = pd.Series(1.0 / len(chosen), index=chosen)
        else:
            raise ValueError("weighting must be 'equal' or 'inv_vol'")

        # Record picks
        picks_rows.append({"date": dt, **{f"rank{i}": t for i, t in enumerate(chosen, 1)}})
        w_t.loc[dt, w_sel.index] = w_sel.values

    w = w_t.ffill().fillna(0.0)
    picks = pd.DataFrame(picks_rows).set_index("date").sort_index() if picks_rows else pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return w, picks

def turnover_and_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)


# ---------- Information Coefficient (IC) ----------

def compute_ic(scores: pd.DataFrame, r: pd.DataFrame, freq: str) -> pd.Series:
    """
    Spearman rank IC at each rebalance date between composite scores at time t
    and forward returns until next rebalance (exclusive of allocation day).
    """
    rebs = rebalance_dates(r.index, freq)
    out = []
    for i, dt in enumerate(rebs):
        if dt not in scores.index:
            continue
        # next segment end (exclusive)
        if i + 1 < len(rebs):
            end = rebs[i + 1]
        else:
            end = r.index[-1]

        seg = r.loc[dt:end]
        if len(seg) <= 1:
            continue
        seg = seg.iloc[1:]  # start next day after allocation
        fwd = (1 + seg).prod() - 1.0  # cumulative forward return per asset

        s = scores.loc[dt].dropna()
        fwd = fwd.reindex(s.index).dropna()
        s = s.reindex(fwd.index).dropna()
        if len(s) < 3:
            continue

        ic, _ = spearmanr(s.values, fwd.values)
        out.append((dt, ic))

    if not out:
        return pd.Series(dtype=float, name="IC")
    ic_series = pd.Series({dt: ic for dt, ic in out}).sort_index()
    ic_series.name = "IC"
    return ic_series


# ---------- Strategy Runner ----------

def run_multifactor(px: pd.DataFrame,
                    start_dt: pd.Timestamp,
                    end_dt: pd.Timestamp,
                    lookback_mom: int,
                    skip_days: int,
                    vol_window: int,
                    top_n: int,
                    w_mom: float,
                    w_vol: float,
                    reb: str,
                    rf_annual: float,
                    tc_bps: int,
                    weighting: str) -> Dict[str, object]:
    """
    Build factors -> composite scores -> weights -> daily returns; compute metrics + IC.
    """
    px_win = px.loc[start_dt:end_dt]
    r, mom, vol = compute_factors(px_win, lookback_mom, skip_days, vol_window)
    if r.empty:
        return {"ret_net": pd.Series(dtype=float), "eq": pd.Series(dtype=float)}

    # Cross-sectional ranks
    mom_rank = xsection_rank(mom.reindex(r.index))
    vol_rank = xsection_rank(-vol.reindex(r.index))  # negative vol => higher rank for low vol

    # Composite score at each date (align indexes)
    composite = (w_mom * mom_rank + w_vol * vol_rank).dropna(how="all")

    # Rebalance on chosen schedule using composite scores at rebalance dates
    scores_at_reb = composite.copy()
    w, picks = build_weights_from_scores(scores_at_reb, r, top_n, reb, weighting=weighting, inv_vol_window=vol_window)

    costs = turnover_and_costs(w, tc_bps)
    rf_daily = rf_annual / TRADING_DAYS

    strat_r_gross = (w * r).sum(axis=1)
    strat_r_net = strat_r_gross - costs - rf_daily
    eq = (1 + strat_r_net).cumprod().rename("eq")

    # Compute IC on composite scores
    ic = compute_ic(scores_at_reb, r, reb)

    return {"ret_net": strat_r_net, "eq": eq, "weights": w, "picks": picks, "ic": ic}


# ---------- CLI / Main ----------

@dataclass
class MFConfig:
    tickers: List[str]
    start: str = "2005-01-01"
    rf_annual: float = 0.02
    rebalance: str = "M"           # 'M' or 'Q' → mapped to 'ME'/'QE'
    tc_bps: int = 10
    lookback_mom: int = 252
    skip_days: int = 21
    vol_window: int = 63
    top_n: int = 3
    w_mom: float = 0.7
    w_vol: float = 0.3
    weighting: str = "equal"       # 'equal' or 'inv_vol'
    outdir: str = "."

def parse_args() -> MFConfig:
    p = argparse.ArgumentParser(description="Day 7 — Multi-Factor XS (Momentum + Vol Filter)")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--rf", type=float, default=0.02)
    p.add_argument("--reb", type=str, default="M", choices=["M", "Q"])
    p.add_argument("--tc_bps", type=int, default=10)
    p.add_argument("--lookback", type=int, default=252, help="Momentum lookback days")
    p.add_argument("--skip", type=int, default=21, help="Skip days (12-1 uses 21)")
    p.add_argument("--vol_window", type=int, default=63, help="Volatility window (trading days)")
    p.add_argument("--top", type=int, default=3, help="Top-N assets to hold")
    p.add_argument("--w_mom", type=float, default=0.7, help="Weight for momentum rank")
    p.add_argument("--w_vol", type=float, default=0.3, help="Weight for (negative) vol rank")
    p.add_argument("--weighting", type=str, default="equal", choices=["equal", "inv_vol"])
    p.add_argument("--outdir", type=str, default=".")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    # normalize weights
    total = a.w_mom + a.w_vol
    w_mom = a.w_mom / total
    w_vol = a.w_vol / total

    return MFConfig(
        tickers=tickers,
        start=a.start,
        rf_annual=a.rf,
        rebalance=a.reb,
        tc_bps=a.tc_bps,
        lookback_mom=a.lookback,
        skip_days=a.skip,
        vol_window=a.vol_window,
        top_n=a.top,
        w_mom=w_mom,
        w_vol=w_vol,
        weighting=a.weighting,
        outdir=a.outdir,
    )

def main() -> None:
    cfg = parse_args()
    px = fetch_prices(cfg.tickers, cfg.start)

    out = run_multifactor(
        px,
        start_dt=px.index[0],
        end_dt=px.index[-1],
        lookback_mom=cfg.lookback_mom,
        skip_days=cfg.skip_days,
        vol_window=cfg.vol_window,
        top_n=cfg.top_n,
        w_mom=cfg.w_mom,
        w_vol=cfg.w_vol,
        reb=cfg.rebalance,
        rf_annual=cfg.rf_annual,
        tc_bps=cfg.tc_bps,
        weighting=cfg.weighting,
    )

    # Metrics
    ret = out["ret_net"]
    eq = out["eq"]
    rf = cfg.rf_annual
    perf = {
        "CAGR": cagr(ret),
        "Vol": ann_vol(ret),
        "Sharpe": sharpe(ret, rf_annual=rf),
        "MaxDD": max_dd(eq),
        "Hit%": hit_rate(ret),
    }

    # Save artifacts
    out_csv = os.path.join(cfg.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_charts = os.path.join(cfg.outdir, "charts"); os.makedirs(out_charts, exist_ok=True)

    ret.rename("ret").to_csv(os.path.join(out_csv, "day07_daily_returns.csv"))
    eq.rename("eq").to_csv(os.path.join(out_csv, "day07_equity_curve.csv"))
    out["weights"].to_csv(os.path.join(out_csv, "day07_weights_daily.csv"))
    if not out["picks"].empty:
        out["picks"].to_csv(os.path.join(out_csv, "day07_picks_by_rebalance.csv"))
    out["ic"].to_csv(os.path.join(out_csv, "day07_ic_series.csv"))

    # Charts
    fig, ax = plt.subplots(figsize=(10, 5))
    eq.plot(ax=ax, label="Multi-Factor Equity")
    ax.set_title("Day 7 — Multi-Factor (Momentum + Vol Filter) Equity Curve")
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(out_charts, "day07_equity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not out["ic"].empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        out["ic"].plot(ax=ax)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Day 7 — Information Coefficient (Composite vs Forward Returns)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_charts, "day07_ic_series.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Console print
    print("\n=== Day 7 — Multi-Factor Performance (net) ===")
    for k, v in perf.items():
        try:
            print(f"{k}: {v:.4f}")
        except Exception:
            print(f"{k}: {v}")

    print("\nArtifacts:")
    print(f"- daily_returns: {os.path.join(out_csv, 'day07_daily_returns.csv')}")
    print(f"- equity_curve:  {os.path.join(out_csv, 'day07_equity_curve.csv')}")
    print(f"- weights_daily: {os.path.join(out_csv, 'day07_weights_daily.csv')}")
    print(f"- picks_by_reb:  {os.path.join(out_csv, 'day07_picks_by_rebalance.csv')}")
    print(f"- ic_series:     {os.path.join(out_csv, 'day07_ic_series.csv')}")
    print(f"- chart_eq:      {os.path.join(out_charts, 'day07_equity.png')}")
    print(f"- chart_ic:      {os.path.join(out_charts, 'day07_ic_series.png')}")

if __name__ == "__main__":
    main()
