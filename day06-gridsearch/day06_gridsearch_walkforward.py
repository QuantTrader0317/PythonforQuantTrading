"""
Day 6 — Parameter Grid Search + Walk-Forward (XS Momentum 12-1 style)

What this does
--------------
- Define a small parameter grid over: lookback_days, skip_days, top_n
- For each walk-forward TRAIN window: sweep the grid, rank by in-sample Sharpe, pick best combo
- Apply the chosen combo to the following TEST window (OOS)
- Stitch all OOS returns, compute aggregate metrics, save CSVs + chart
- --debug prints train/test windows and the top grid combos each period
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# ---------- Helpers / Metrics ----------

def safe_freq(freq: str) -> str:
    """Map deprecated pandas resample codes to replacements."""
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

def ann_vol(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if x.empty: return np.nan
    return float(x.std() * np.sqrt(periods))

def cagr(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if x.empty: return np.nan
    gr = float((1 + x).prod())
    yrs = len(x) / periods
    if yrs <= 0: return np.nan
    return gr ** (1 / yrs) - 1

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if v == 0 or np.isnan(v): return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna()
    if eq.empty: return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty: return np.nan
    return float((x > 0).mean())

# ---------- Strategy Components (XS momentum) ----------

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def compute_signal(px: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """
    Momentum signal = (P[t-skip] / P[t-lookback]) - 1
    For 12-1 momentum, lookback≈252, skip≈21.
    """
    p1 = px.shift(skip)
    p0 = px.shift(lookback)
    return (p1 / p0) - 1.0

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def build_weights(sig: pd.DataFrame, r: pd.DataFrame, top_n: int, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    picks_rows = []

    for dt in rebs:
        if dt not in sig.index: continue
        s = sig.loc[dt].dropna()
        if s.empty: continue
        s = s.sort_values(ascending=False)
        # deterministic tie-break
        s = s.groupby(s.values, group_keys=False).apply(lambda x: x.sort_index())
        chosen = list(s.index[:top_n])
        if not chosen: continue

        picks_rows.append({"date": dt, **{f"rank{i}": t for i, t in enumerate(chosen, 1)}})
        w_t.loc[dt, chosen] = 1.0 / len(chosen)

    w = w_t.ffill().fillna(0.0)
    picks = pd.DataFrame(picks_rows).set_index("date").sort_index() if picks_rows else pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return w, picks

def turnover_and_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)

def run_strategy(px: pd.DataFrame,
                 start_dt: pd.Timestamp,
                 end_dt: pd.Timestamp,
                 lookback: int,
                 skip: int,
                 top_n: int,
                 reb: str,
                 rf_annual: float,
                 tc_bps: int) -> Dict[str, pd.Series]:
    """Run XS momentum with given parameters over [start_dt, end_dt]."""
    px_win = px.loc[start_dt:end_dt]
    r = px_win.pct_change().dropna()
    if r.empty:
        return {"ret_net": pd.Series(dtype=float), "eq": pd.Series(dtype=float)}

    rf_daily = rf_annual / TRADING_DAYS
    sig = compute_signal(px_win, lookback, skip)
    w, _ = build_weights(sig, r, top_n, reb)
    costs = turnover_and_costs(w, tc_bps)

    strat_r_gross = (w * r).sum(axis=1)
    strat_r_net = strat_r_gross - costs - rf_daily
    eq = (1 + strat_r_net).cumprod().rename("eq")
    return {"ret_net": strat_r_net, "eq": eq}

# ---------- Walk-Forward Engine ----------

@dataclass
class WFConfig:
    tickers: List[str]
    start: str = "2005-01-01"
    rf_annual: float = 0.02
    rebalance: str = "M"       # 'M' or 'Q' (mapped via safe_freq)
    tc_bps: int = 10
    train_years: int = 5
    test_years: int = 1
    grid_lookbacks: List[int] = None
    grid_skips: List[int] = None
    grid_tops: List[int] = None
    outdir: str = "."
    debug: bool = False

def make_periods(idx: pd.DatetimeIndex, train_years: int, test_years: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    if len(idx) == 0: return []
    start = pd.Timestamp(idx[0]); end = pd.Timestamp(idx[-1])
    periods = []
    t0 = pd.Timestamp(year=start.year, month=start.month, day=1)
    while True:
        tr_s = t0
        tr_e = tr_s + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        te_s = tr_e + pd.Timedelta(days=1)
        te_e = te_s + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if te_s > end: break
        periods.append((tr_s, min(tr_e, end), te_s, min(te_e, end)))
        t0 = t0 + pd.DateOffset(years=test_years)
        if t0 > end: break
    return periods

def param_grid(lookbacks: Iterable[int], skips: Iterable[int], tops: Iterable[int]) -> List[Tuple[int,int,int]]:
    out = []
    for L in lookbacks:
        for S in skips:
            for N in tops:
                out.append((int(L), int(S), int(N)))
    return out

def sweep_train(px: pd.DataFrame,
                tr_start: pd.Timestamp,
                tr_end: pd.Timestamp,
                grid: List[Tuple[int,int,int]],
                reb: str,
                rf_annual: float,
                tc_bps: int) -> Tuple[Tuple[int,int,int], pd.DataFrame]:
    """Return best (lookback, skip, top) by in-sample Sharpe and the full results table."""
    rows = []
    for lookback, skip, top_n in grid:
        out = run_strategy(px, tr_start, tr_end, lookback, skip, top_n, reb, rf_annual, tc_bps)
        ret = out["ret_net"]
        rows.append({
            "lookback": lookback,
            "skip": skip,
            "top": top_n,
            "Sharpe": sharpe(ret, rf_annual=rf_annual),
            "CAGR": cagr(ret),
            "Vol": ann_vol(ret)
        })
    df = pd.DataFrame(rows).sort_values(["Sharpe","CAGR"], ascending=False, na_position="last")
    best_row = df.dropna(subset=["Sharpe"]).head(1)
    if best_row.empty:
        # fallback to first combo if everything NaN
        best = grid[0]
    else:
        best = (int(best_row.iloc[0]["lookback"]), int(best_row.iloc[0]["skip"]), int(best_row.iloc[0]["top"]))
    return best, df

def run_grid_walkforward(cfg: WFConfig) -> Dict[str, object]:
    px = fetch_prices(cfg.tickers, cfg.start)

    # Defaults for grids if not provided
    if not cfg.grid_lookbacks: cfg.grid_lookbacks = [126, 189, 252]   # ~6m, 9m, 12m
    if not cfg.grid_skips:     cfg.grid_skips     = [0, 21]           # no skip, 1-month skip
    if not cfg.grid_tops:      cfg.grid_tops      = [2, 3, 4]

    # sanity: enough data
    min_required = max(cfg.grid_lookbacks) + max(cfg.grid_skips) + 5
    if len(px) < min_required:
        raise ValueError("Not enough data for the largest lookback/skip. Use earlier --start or smaller grid.")

    r_all = px.pct_change().dropna()
    periods = make_periods(r_all.index, cfg.train_years, cfg.test_years)
    if not periods:
        raise ValueError("No walk-forward periods constructed. Check date ranges.")

    out_csv = os.path.join(cfg.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_charts = os.path.join(cfg.outdir, "charts"); os.makedirs(out_charts, exist_ok=True)

    grid = param_grid(cfg.grid_lookbacks, cfg.grid_skips, cfg.grid_tops)

    all_oos = []
    per_period_rows = []
    all_train_tables = []  # keep each period's sweep

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(periods, 1):
        # pad train start to ensure signal availability
        tr_s_pad = tr_s - pd.Timedelta(days=max(cfg.grid_lookbacks) + max(cfg.grid_skips) + 5)
        if tr_s_pad < px.index[0]: tr_s_pad = px.index[0]

        best, train_table = sweep_train(px, tr_s_pad, tr_e, grid, cfg.rebalance, cfg.rf_annual, cfg.tc_bps)
        L, S, N = best

        # TEST with chosen combo
        test = run_strategy(px, te_s, te_e, L, S, N, cfg.rebalance, cfg.rf_annual, cfg.tc_bps)
        te_ret = test["ret_net"]

        # period metrics
        if te_ret.empty:
            m = {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Vol": np.nan, "Hit%": np.nan}
        else:
            eq = (1 + te_ret).cumprod()
            m = {"CAGR": cagr(te_ret), "Sharpe": sharpe(te_ret, cfg.rf_annual),
                 "MaxDD": max_dd(eq), "Vol": ann_vol(te_ret), "Hit%": hit_rate(te_ret)}

        per_period_rows.append({
            "period": i,
            "train_start": tr_s.date(), "train_end": tr_e.date(),
            "test_start": te_s.date(), "test_end": te_e.date(),
            "chosen_lookback": L, "chosen_skip": S, "chosen_top": N,
            **m
        })
        all_oos.append(te_ret)

        # keep train sweep with period tag
        tt = train_table.copy()
        tt["period"] = i
        all_train_tables.append(tt)

        if cfg.debug:
            print(f"\n[DEBUG] P{i} Train {tr_s.date()}→{tr_e.date()} | Test {te_s.date()}→{te_e.date()}")
            print(f"[DEBUG] Best combo: lookback={L}, skip={S}, top={N}")
            print("[DEBUG] Train top combos:")
            print(train_table.head(5).to_string(index=False))

    # Aggregate OOS
    if all_oos:
        oos_all = pd.concat(all_oos).sort_index()
        oos_all = oos_all[~oos_all.index.duplicated(keep="last")]
        eq_all = (1 + oos_all).cumprod()
        agg = {
            "CAGR_OOS": cagr(oos_all),
            "Sharpe_OOS": sharpe(oos_all, cfg.rf_annual),
            "MaxDD_OOS": max_dd(eq_all),
            "Vol_OOS": ann_vol(oos_all),
            "Hit%_OOS": hit_rate(oos_all),
        }
    else:
        oos_all = pd.Series(dtype=float)
        agg = {k: np.nan for k in ["CAGR_OOS","Sharpe_OOS","MaxDD_OOS","Vol_OOS","Hit%_OOS"]}

    # Save artifacts
    per_period_df = pd.DataFrame(per_period_rows)
    per_period_df.to_csv(os.path.join(out_csv, "wf_grid_period_results.csv"), index=False)

    if all_train_tables:
        full_grid = pd.concat(all_train_tables, ignore_index=True)
        full_grid.to_csv(os.path.join(out_csv, "wf_grid_train_sweeps.csv"), index=False)
    else:
        full_grid = pd.DataFrame()

    oos_all.rename("ret_oos").to_csv(os.path.join(out_csv, "wf_grid_oos_daily_returns.csv"))

    if not oos_all.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        eq_all.plot(ax=ax, label="Walk-Forward OOS Equity (best-per-period)")
        ax.set_title("Day 6 — Grid Search + Walk-Forward (stitched OOS)")
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(out_charts, "wf_grid_oos_equity.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {"period_table": per_period_df, "grid_table": full_grid, "oos": oos_all, "agg": agg,
            "paths": {
                "period_results": os.path.join(out_csv, "wf_grid_period_results.csv"),
                "train_sweeps": os.path.join(out_csv, "wf_grid_train_sweeps.csv"),
                "oos_returns": os.path.join(out_csv, "wf_grid_oos_daily_returns.csv"),
                "oos_chart": os.path.join(out_charts, "wf_grid_oos_equity.png"),
            }}

# ---------- CLI ----------

def parse_args() -> 'WFConfig':
    p = argparse.ArgumentParser(description="Day 6 — Grid Search Walk-Forward (XS Momentum)")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--rf", type=float, default=0.02, help="Annual cash rate")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"], help="Rebalance frequency")
    p.add_argument("--tc_bps", type=int, default=10, help="Transaction cost in bps")
    p.add_argument("--train_years", type=int, default=5)
    p.add_argument("--test_years", type=int, default=1)
    p.add_argument("--grid_lookbacks", type=str, default="126,189,252", help="Comma ints (e.g., 126,189,252)")
    p.add_argument("--grid_skips", type=str, default="0,21", help="Comma ints (e.g., 0,21)")
    p.add_argument("--grid_tops", type=str, default="2,3,4", help="Comma ints (e.g., 2,3,4)")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--debug", action="store_true", help="Verbose per-period info")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    Ls = [int(x) for x in a.grid_lookbacks.split(",") if x.strip()]
    Ss = [int(x) for x in a.grid_skips.split(",") if x.strip()]
    Ns = [int(x) for x in a.grid_tops.split(",") if x.strip()]

    return WFConfig(
        tickers=tickers,
        start=a.start,
        rf_annual=a.rf,
        rebalance=a.reb,
        tc_bps=a.tc_bps,
        train_years=a.train_years,
        test_years=a.test_years,
        grid_lookbacks=Ls,
        grid_skips=Ss,
        grid_tops=Ns,
        outdir=a.outdir,
        debug=a.debug,
    )

def main() -> None:
    cfg = parse_args()
    out = run_grid_walkforward(cfg)

    print("\n=== Day 6 — Per-Period Results (OOS) ===")
    print(out["period_table"].to_string(index=False))

    print("\n=== Aggregate OOS (stitched) ===")
    for k, v in out["agg"].items():
        try: print(f"{k}: {v:.4f}")
        except Exception: print(f"{k}: {v}")

    print("\nArtifacts:")
    for k, v in out["paths"].items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()