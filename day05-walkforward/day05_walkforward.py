#Quanttrader0317 day05 walk-forward analysis
"""
Day 5 — Walk-Forward Analysis for Cross-Sectional Momentum (12-1)

- Trains on a rolling window (e.g., 5y), picks best top-N by in-sample Sharpe
- Tests that top-N on the next period (true out-of-sample)
- Stitches OOS periods into one series; saves CSVs + OOS equity chart
- Uses safe_freq() so 'M'→'ME' and 'Q'→'QE' (no pandas FutureWarning)
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


# ---------- Strategy Components ----------

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def compute_signal(px: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """
    12-1 momentum = (P[t-skip] / P[t-lookback]) - 1
    """
    p1 = px.shift(skip)
    p0 = px.shift(lookback)
    return (p1 / p0) - 1.0

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index

def build_weights(sig: pd.DataFrame, r: pd.DataFrame, top_n: int, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create daily weights by picking top-N at each rebalance.
    Robust to periods where no picks are formed (returns empty 'picks' cleanly).
    """
    rebs = rebalance_dates(r.index, freq)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    picks_rows = []

    for dt in rebs:
        if dt not in sig.index:
            continue
        s = sig.loc[dt].dropna()
        if s.empty:
            continue
        # rank desc; deterministic tie-break by ticker
        s = s.sort_values(ascending=False)
        s = s.groupby(s.values, group_keys=False).apply(lambda x: x.sort_index())
        chosen = list(s.index[:top_n])

        if len(chosen) == 0:
            continue

        picks_rows.append(
            {"date": dt, **{f"rank{i}": t for i, t in enumerate(chosen, 1)}}
        )
        w_t.loc[dt, chosen] = 1.0 / len(chosen)

    # weights forward-filled between rebalances
    w = w_t.ffill().fillna(0.0)

    # build 'picks' safely (may be empty)
    if picks_rows:
        picks = pd.DataFrame(picks_rows).set_index("date").sort_index()
    else:
        picks = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    return w, picks

def turnover_and_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)

def run_xs_momentum(px: pd.DataFrame,
                    start_dt: pd.Timestamp,
                    end_dt: pd.Timestamp,
                    top_n: int,
                    reb: str,
                    rf_annual: float,
                    tc_bps: int,
                    lookback_days: int,
                    skip_days: int) -> Dict[str, pd.Series]:
    """
    Run the strategy on a date range [start_dt, end_dt] inclusive.
    Returns dict with net daily returns and equity curve.
    """
    px_win = px.loc[start_dt:end_dt]
    r = px_win.pct_change().dropna()
    if r.empty:
        return {"ret_net": pd.Series(dtype=float), "eq": pd.Series(dtype=float)}

    rf_daily = rf_annual / TRADING_DAYS
    sig = compute_signal(px_win, lookback_days, skip_days)

    w, _ = build_weights(sig, r, top_n, reb)
    costs = turnover_and_costs(w, tc_bps)

    strat_r_gross = (w * r).sum(axis=1)
    strat_r_net = strat_r_gross - costs - rf_daily
    eq = (1 + strat_r_net).cumprod().rename("eq")

    return {"ret_net": strat_r_net, "eq": eq}

def param_sweep_train(px: pd.DataFrame,
                      start_dt: pd.Timestamp,
                      end_dt: pd.Timestamp,
                      top_grid: List[int],
                      reb: str,
                      rf_annual: float,
                      tc_bps: int,
                      lookback_days: int,
                      skip_days: int) -> Tuple[int, pd.DataFrame]:
    """
    Train on [start_dt, end_dt] by sweeping top_n over top_grid.
    Select best Sharpe (in-sample). Returns (best_top_n, results_df).
    """
    res = []
    for top_n in top_grid:
        out = run_xs_momentum(px, start_dt, end_dt, top_n, reb, rf_annual, tc_bps, lookback_days, skip_days)
        ret = out["ret_net"]
        if ret.empty:
            res.append({"top": top_n, "Sharpe": np.nan, "CAGR": np.nan, "Vol": np.nan})
            continue
        res.append({
            "top": top_n,
            "Sharpe": sharpe(ret, rf_annual=rf_annual),
            "CAGR": cagr(ret),
            "Vol": ann_vol(ret)
        })
    df = pd.DataFrame(res).sort_values("Sharpe", ascending=False)
    if df["Sharpe"].notna().any():
        best_top = int(df.dropna(subset=["Sharpe"]).iloc[0]["top"])
    else:
        best_top = top_grid[0]
    return best_top, df


# ---------- Walk-Forward Engine ----------

@dataclass
class WFConfig:
    tickers: List[str]
    start: str = "2005-01-01"
    rf_annual: float = 0.02
    rebalance: str = "M"     # 'M' or 'Q' (mapped via safe_freq)
    tc_bps: int = 10
    lookback_days: int = 252
    skip_days: int = 21
    train_years: int = 5
    test_years: int = 1
    top_grid: List[int] = None
    outdir: str = "."
    debug: bool = False

def make_periods(idx: pd.DatetimeIndex, train_years: int, test_years: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build rolling train/test periods:
      [train_start, train_end], [test_start, test_end]
    Step forward by test_years each iteration.
    """
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    if len(idx) == 0:
        return []

    start = pd.Timestamp(idx[0])
    end = pd.Timestamp(idx[-1])

    periods = []
    t0 = pd.Timestamp(year=start.year, month=start.month, day=1)
    while True:
        train_start = t0
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)

        if test_start > end:
            break

        # clip to available data range
        train_end_clipped = min(train_end, end)
        test_end_clipped = min(test_end, end)

        periods.append((train_start, train_end_clipped, test_start, test_end_clipped))

        t0 = t0 + pd.DateOffset(years=test_years)
        if t0 > end:
            break

    return periods

def run_walk_forward(cfg: WFConfig) -> Dict[str, object]:
    px = fetch_prices(cfg.tickers, cfg.start)
    if cfg.top_grid is None or len(cfg.top_grid) == 0:
        cfg.top_grid = [2, 3, 4]  # small, sane grid

    # Ensure enough history to form signals
    min_required_days = cfg.lookback_days + cfg.skip_days + 5
    if len(px) < min_required_days:
        raise ValueError("Not enough data to form 12-1 signals. Try an earlier start or fewer tickers.")

    r_all = px.pct_change().dropna()
    periods = make_periods(r_all.index, cfg.train_years, cfg.test_years)
    if not periods:
        raise ValueError("No walk-forward periods could be constructed. Check dates and windows.")

    out_csv = os.path.join(cfg.outdir, "outputs")
    out_charts = os.path.join(cfg.outdir, "charts")
    os.makedirs(out_csv, exist_ok=True)
    os.makedirs(out_charts, exist_ok=True)

    oos_rets = []
    rows = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(periods, 1):
        # pad train start to ensure signal availability during training
        tr_s_pad = tr_s - pd.Timedelta(days=cfg.lookback_days + cfg.skip_days + 5)
        tr_s_pad = max(tr_s_pad, r_all.index[0])

        # TRAIN: choose best top-N by Sharpe
        best_top, train_table = param_sweep_train(
            px, tr_s_pad, tr_e, cfg.top_grid, cfg.rebalance, cfg.rf_annual,
            cfg.tc_bps, cfg.lookback_days, cfg.skip_days
        )

        # TEST: run with chosen top-N
        test_out = run_xs_momentum(
            px, te_s, te_e, best_top, cfg.rebalance, cfg.rf_annual,
            cfg.tc_bps, cfg.lookback_days, cfg.skip_days
        )
        te_ret = test_out["ret_net"]

        # Period metrics (OOS)
        if te_ret.empty:
            per_metrics = {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Vol": np.nan, "Hit%": np.nan}
        else:
            eq = (1 + te_ret).cumprod()
            per_metrics = {
                "CAGR": cagr(te_ret),
                "Sharpe": sharpe(te_ret, rf_annual=cfg.rf_annual),
                "MaxDD": max_dd(eq),
                "Vol": ann_vol(te_ret),
                "Hit%": hit_rate(te_ret),
            }

        rows.append({
            "period": i,
            "train_start": tr_s.date(), "train_end": tr_e.date(),
            "test_start": te_s.date(), "test_end": te_e.date(),
            "chosen_top": best_top,
            **per_metrics
        })
        oos_rets.append(te_ret)
        # --- DEBUG LOGGING (train/test window + picks + segment contributions) ---
        if cfg.debug:
            print(f"\n[DEBUG] Train: {tr_s.date()} → {tr_e.date()} | Test: {te_s.date()} → {te_e.date()}")
            # Show training sweep results (top 3 rows)
            try:
                tt = train_table.sort_values("Sharpe", ascending=False).head(3)
                print("[DEBUG] Train sweep (top):")
                print(tt.to_string(index=False))
            except Exception:
                pass
            print(f"[DEBUG] Chosen top-N for test: {best_top}")

            # Build picks inside TEST window and compute per-rebalance contributions
            px_test = px.loc[te_s:te_e]
            r_test = px_test.pct_change().dropna()
            if not r_test.empty:
                sig_test = compute_signal(px_test, cfg.lookback_days, cfg.skip_days)
                w_test, picks_test = build_weights(sig_test, r_test, best_top, cfg.rebalance)

                if not picks_test.empty:
                    reb_list = list(picks_test.index)
                    for j, dt_reb in enumerate(reb_list):
                        # figure segment end = next rebalance start (exclusive); or test end
                        if j + 1 < len(reb_list):
                            seg_end = reb_list[j + 1]
                        else:
                            seg_end = te_e

                        # slice (exclude allocation day’s first row of returns)
                        seg = r_test.loc[dt_reb:seg_end]
                        if len(seg) > 1:
                            seg = seg.iloc[1:]  # start from next day

                        # get chosen tickers on this rebalance
                        row = picks_test.loc[dt_reb]
                        chosen = [row[c] for c in row.index if str(c).startswith("rank") and pd.notna(row[c])]
                        print(f"[DEBUG] Test rebalance {dt_reb.date()}: Picks = {chosen}")

                        # per-asset cumulative return over the segment
                        contrib = {}
                        for tkr in chosen:
                            if tkr in seg.columns and not seg[tkr].empty:
                                contrib[tkr] = float((1 + seg[tkr]).prod() - 1)
                        if contrib:
                            print(f"[DEBUG] Segment returns thru {pd.to_datetime(seg.index[-1]).date()}: {contrib}")
                else:
                    print("[DEBUG] No valid picks formed inside test window (likely early NaNs).")
            else:
                print("[DEBUG] No returns in test window (empty r_test).")

        # Optional: per-period chart
        if not te_ret.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            (1 + te_ret).cumprod().plot(ax=ax, label=f"OOS eq (period {i})")
            ax.set_title(f"Walk-Forward OOS Equity – Period {i} ({te_s.date()} to {te_e.date()}) | top={best_top}")
            ax.legend(); plt.tight_layout()
            fig.savefig(os.path.join(out_charts, f"wf_oos_period_{i}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

    # Aggregate OOS (dedupe index just in case)
    if oos_rets:
        oos_all = pd.concat(oos_rets).sort_index()
        oos_all = oos_all[~oos_all.index.duplicated(keep="last")]
        eq_all = (1 + oos_all).cumprod()
        agg = {
            "CAGR_OOS": cagr(oos_all),
            "Sharpe_OOS": sharpe(oos_all, rf_annual=cfg.rf_annual),
            "MaxDD_OOS": max_dd(eq_all),
            "Vol_OOS": ann_vol(oos_all),
            "Hit%_OOS": hit_rate(oos_all),
        }
    else:
        oos_all = pd.Series(dtype=float)
        agg = {k: np.nan for k in ["CAGR_OOS", "Sharpe_OOS", "MaxDD_OOS", "Vol_OOS", "Hit%_OOS"]}

    # Save artifacts
    period_df = pd.DataFrame(rows)
    period_df.to_csv(os.path.join(out_csv, "wf_period_results.csv"), index=False)
    oos_all.rename("ret_oos").to_csv(os.path.join(out_csv, "wf_oos_daily_returns.csv"))

    if not oos_all.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        (1 + oos_all).cumprod().plot(ax=ax, label="Walk-Forward OOS Equity")
        ax.set_title("Walk-Forward Out-of-Sample Equity (stitched)")
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(out_charts, "wf_oos_equity.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "period_table": period_df,
        "oos": oos_all,
        "agg": agg,
        "paths": {
            "period_table": os.path.join(out_csv, "wf_period_results.csv"),
            "oos_returns": os.path.join(out_csv, "wf_oos_daily_returns.csv"),
            "oos_chart": os.path.join(out_charts, "wf_oos_equity.png"),
        },
    }


# ---------- CLI ----------

def parse_args() -> 'WFConfig':
    p = argparse.ArgumentParser(description="Day 5 — Walk-Forward XS Momentum (12-1)")
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--rf", type=float, default=0.02, help="Annual cash rate")
    p.add_argument("--reb", type=str, default="M", choices=["M", "Q"], help="Rebalance frequency")
    p.add_argument("--tc_bps", type=int, default=10, help="Transaction costs in bps of turnover")
    p.add_argument("--lookback", type=int, default=252, help="Lookback days for momentum")
    p.add_argument("--skip", type=int, default=21, help="Skip days to avoid short-term reversal")
    p.add_argument("--train_years", type=int, default=5, help="Training window length (years)")
    p.add_argument("--test_years", type=int, default=1, help="Test window length (years)")
    p.add_argument("--top_grid", type=str, default="2,3,4", help="Comma-separated candidate N (top picks)")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    top_grid = [int(x) for x in a.top_grid.split(",") if x.strip()]

    return WFConfig(
        tickers=tickers,
        start=a.start,
        rf_annual=a.rf,
        rebalance=a.reb,
        tc_bps=a.tc_bps,
        lookback_days=a.lookback,
        skip_days=a.skip,
        train_years=a.train_years,
        test_years=a.test_years,
        top_grid=top_grid,
        outdir=a.outdir,
        debug=a.debug,
    )

def main() -> None:
    cfg = parse_args()
    out = run_walk_forward(cfg)

    print("\n=== Walk-Forward Period Results ===")
    print(out["period_table"].to_string(index=False))

    agg = out["agg"]
    print("\n=== Aggregate Out-of-Sample (stitched) ===")
    for k, v in agg.items():
        try:
            print(f"{k}: {v:.4f}")
        except Exception:
            print(f"{k}: {v}")

    print("\nArtifacts:")
    for k, v in out["paths"].items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()

