'''

dAY 8 - Volatility Targeting Overlay

Idea:
- Start with a base daily return series (e.g., Day 4 top-N equal weight).
- Compute a rolling realized vol.
- Scale returns by L_t = target_vol / rolling_vol_t (with caps).
- Compare before/after: drawdowns, Sharpe, vol, CAGR.

Run:
 python day08_volatility_targeting.py --target 0.10 -- window 63 --reb M --top 3
'''

from __future__ import annotations
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from utils import(
    TRADING_DAYS, fetch_prices, daily_returns, xsection_rank, build_equal_weights,
    turnover_costs, cagr, ann_vol, sharpe, max_dd, hit_rate, scale_to_target_vol
)

def compute_momentum(px: pd.DataFrame, lookback=252, skip=21) -> pd.DataFrame:
    p1 = px.shift(skip); p0 = px.shift(lookback)
    return (p1 / p0) - 1.0

def base_strategy_returns(px: pd.DataFrame, reb="M", top=3, lookback=252, skip=21, tc_bps=10, rf_annual=0.02) -> pd.Series:
    r = daily_returns(px)
    sig = compute_momentum(px, lookback, skip)
    sig_rank = xsection_rank(sig)
    w = build_equal_weights(sig_rank, r, top_n=top, freq=reb)
    # Costs + cash
    costs = turnover_costs(w, tc_bps)
    rf_daily = rf_annual / TRADING_DAYS
    ret = (w * r).sum(axis=1) - costs - rf_daily
    return ret.rename("ret_base")

def apply_vol_target(ret_base: pd.Series, target=0.10, window=63) -> pd.Series:
    L = scale_to_target_vol(ret_base, target_ann_vol=target, window=window)
    ret_vt = (L * ret_base).rename("ret_vt")
    return ret_vt

def compare_metrics(ret_a: pd.Series, ret_b: pd.Series, names=("Base", "VolTarget"), rf=0.02):
    out = {}
    for nm, r in zip(names, (ret_a, ret_b)):
        eq = (1 + r).cumprod()
        out[nm] = {
            "CAGR": cagr(r),
            "Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf_annual=rf),
            "MaxDD": max_dd(eq),
            "Hit%": hit_rate(r),
        }
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    ap.add_argument("--start", type=str, default="2005-01-01")
    ap.add_argument("--reb", type=str, default="M", choices=["M", "Q"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--tc_bps", type=int, default=10)
    ap.add_argument("--rf", type=float, default=0.02)
    ap.add_argument("--target", type=float, default=0.10, help="Target annualized vol")
    ap.add_argument("--window", type=int, default=63, help="Rolling window (days)")
    ap.add_argument("--outdir", type=str, default=".")
    a = ap.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px = fetch_prices(tickers, a.start)

    ret_base = base_strategy_returns(px, reb=a.reb, top=a.top, lookback=a.lookback, skip=a.skip, tc_bps=a.tc_bps, rf_annual=a.rf)
    ret_vt = apply_vol_target(ret_base, target=a.target, window=a.window)

    # Metrics + save
    out_csv = os.path.join(a.outdir, "outputs"); os.makedirs(out_csv, exist_ok=True)
    out_charts = os.path.join(a.outdir, "charts"); os.makedirs(out_charts, exist_ok=True)

    comp = compare_metrics(ret_base, ret_vt, names=("Base","VolTarget"), rf=a.rf)
    comp.to_csv(os.path.join(out_csv, "day08_metrics_comparison.csv"))

    # Curves
    eq_base = (1 +ret_base).cumprod().rename("Base")
    eq_vt = (1 + ret_vt).cumprod().rename("VolTarget")

    fig, ax = plt.subplots(figsize=(10,5))
    eq_base.plot(ax=ax); eq_vt.plot(ax=ax)
    ax.set_title(f"Day 8 - Volatility Targeting Overlay (target={a.target:.0%}, window={a.window}d)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_charts, "day08_vol_target_equity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n== Day 8 - Metrics ===")
    print(comp.round(4).to_string())

    # Save returns
    pd.concat([ret_base, ret_vt], axis=1).to_csv(os.path.join(out_csv, "day08_returns_base_vs_vt.csv"))
    print("\nArtifacts:")
    print(f"- metrics: {os.path.join(out_csv, 'day08_metrics_comparison.csv')}")
    print(f"- returns: {os.path.join(out_csv, 'day08_returns_base_vs.csv')}")
    print(f"- chart: {os.path.join(out_charts, 'day08_vol_target_equity.png')}")

if __name__ == "__main__":
    main()