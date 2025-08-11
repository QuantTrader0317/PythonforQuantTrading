#!/usr/bin/env python3
"""
Multi-asset equal-weight portfolio backtest (warning-proof resample)
- Data: yfinance (adjusted close), daily returns
- Rebalance: monthly or quarterly (EOM/EoQ) using a safe frequency mapper
- Costs: turnover-based (bps)
- Cash: subtract annual rf as daily drag
- Vol targeting: optional, scale weights to target annualized vol (cap leverage)
- Outputs: CSVs + equity curve chart + console metrics

Usage examples:
  python backtest.py
  python backtest.py --tickers SPY,QQQ,TLT,GLD --start 2005-01-01 --rf 0.02 --reb ME --tc_bps 5 \
                     --target_vol 0.10 --leverage_cap 2.0

Notes:
- Learning scaffold, not investment advice.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# -----------------------------
# Safe frequency mapper (kills pandas FutureWarning for 'M'/'Q')
# -----------------------------

def safe_freq(freq: str) -> str:
    """Map deprecated pandas resample codes to their replacements.
    'M' -> 'ME' (month-end), 'Q' -> 'QE' (quarter-end). Others returned unchanged.
    """
    return {"M": "ME", "Q": "QE"}.get(freq, freq)

# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    tickers: List[str]
    start: str = "2005-01-01"
    rf_annual: float = 0.02           # annual cash rate
    rebalance: str = "ME"              # 'ME' monthly, 'QE' quarterly (safe_freq handles mapping)
    tc_bps: int = 5                   # transaction cost per turnover unit (bps)
    target_vol: Optional[float] = None  # e.g., 0.10 for 10% annual; None disables
    leverage_cap: float = 2.0
    outdir: str = "."

# -----------------------------
# Metrics
# -----------------------------

def ann_vol(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(x.std() * np.sqrt(periods))

def cagr(x: pd.Series, periods: int = TRADING_DAYS) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    gr = float((1 + x).prod())
    yrs = len(x) / periods
    return gr ** (1 / yrs) - 1

def sharpe(x: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    v = ann_vol(x, periods)
    if v == 0 or np.isnan(v):
        return np.nan
    return (cagr(x, periods) - rf_annual) / v

def max_dd(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())

def hit_rate(x: pd.Series) -> float:
    return float((x > 0).mean())

# -----------------------------
# Core logic
# -----------------------------

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    """Download adjusted close prices for tickers from Yahoo Finance."""
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index().dropna(how="all").ffill()
    return df


def equal_weight_targets(r: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Equal weights set at each rebalance date; held constant between rebalances."""
    reb_dates = r.resample(freq).last().index
    mask = r.index.isin(reb_dates)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)
    w_t.loc[mask] = 1.0 / r.shape[1]
    return w_t.ffill().fillna(0.0)


def apply_vol_targeting(r: pd.DataFrame, w_t: pd.DataFrame, target_vol: float, cap: float) -> pd.DataFrame:
    """Scale target weights at rebalance dates so portfolio annual vol ≈ target.
    Uses 60-day RMS of individual vols as a conservative proxy; caps leverage.
    """
    lookback = 60  # days
    daily_vol = r.rolling(lookback).std()
    rms = np.sqrt((daily_vol ** 2).mean(axis=1))  # scalar time series
    scale = target_vol / (rms * np.sqrt(TRADING_DAYS))
    scale = scale.clip(upper=cap).fillna(0.0)

    # apply scale only on rows where weights are explicitly set (rebalance rows)
    reb_rows = w_t.ne(0).any(axis=1) & w_t.notna().any(axis=1)
    w_t.loc[reb_rows] = w_t.loc[reb_rows].mul(scale.loc[reb_rows], axis=0)
    return w_t.ffill().fillna(0.0)


def turnover_and_costs(w: pd.DataFrame, tc_bps: int) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    return turnover * (tc_bps / 10000.0)


def run_backtest(cfg: Config) -> dict:
    # Data
    px = fetch_prices(cfg.tickers, cfg.start)
    r = px.pct_change().dropna()
    rf_daily = cfg.rf_annual / TRADING_DAYS

    # Targets & (optional) vol targeting
    w_target = equal_weight_targets(r, cfg.rebalance)
    if cfg.target_vol is not None:
        w_target = apply_vol_targeting(r, w_target, cfg.target_vol, cfg.leverage_cap)

    # Final weights held between rebalances
    w = w_target.ffill().fillna(0.0)

    # Costs
    costs = turnover_and_costs(w, cfg.tc_bps)

    # Portfolio returns (gross, then net of costs & cash)
    port_r_gross = (w * r).sum(axis=1)
    port_r_net = port_r_gross - costs - rf_daily

    # Buy & hold equal-weight benchmark (no rebalance)
    w_bh = pd.Series(1.0 / r.shape[1], index=r.columns)
    bh_r = (w_bh * r).sum(axis=1) - rf_daily

    # Equity curves
    eq_port = (1 + port_r_net).cumprod().rename("eq_port")
    eq_bh = (1 + bh_r).cumprod().rename("eq_bh")

    # Metrics
    report = {
        "CAGR_port": cagr(port_r_net),
        "Vol_port": ann_vol(port_r_net),
        "Sharpe_port": sharpe(port_r_net, rf_annual=cfg.rf_annual),
        "MaxDD_port": max_dd(eq_port),
        "Hit%_port": hit_rate(port_r_net),
        "CAGR_bh": cagr(bh_r),
        "Vol_bh": ann_vol(bh_r),
        "Sharpe_bh": sharpe(bh_r, rf_annual=cfg.rf_annual),
        "MaxDD_bh": max_dd(eq_bh),
    }

    # Outputs
    out_csv = os.path.join(cfg.outdir, "outputs")
    out_charts = os.path.join(cfg.outdir, "charts")
    os.makedirs(out_csv, exist_ok=True)
    os.makedirs(out_charts, exist_ok=True)

    pd.DataFrame({"port_r_net": port_r_net, "bh_r": bh_r}).to_csv(
        os.path.join(out_csv, "daily_returns.csv"), index=True
    )
    eq_port.to_frame().join(eq_bh).to_csv(os.path.join(out_csv, "equity_curves.csv"))
    w.to_csv(os.path.join(out_csv, "weights_daily.csv"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    eq_bh.plot(ax=ax, label="BH eq-weight")
    eq_port.plot(ax=ax, label="Reb eq-weight (net)")
    ax.set_title("Equity Curve: Eq-Weight Portfolio vs Buy&Hold (net of costs & cash)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_charts, "portfolio_eqcurve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "report": report,
        "paths": {
            "daily_returns": os.path.join(out_csv, "daily_returns.csv"),
            "equity_curves": os.path.join(out_csv, "equity_curves.csv"),
            "weights_daily": os.path.join(out_csv, "weights_daily.csv"),
            "chart": os.path.join(out_charts, "portfolio_eqcurve.png"),
        },
    }

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Multi-asset portfolio backtest (eq-weight)")
    p.add_argument("--tickers", type=str, default="SPY,QQQ,TLT,GLD",
                   help="Comma-separated tickers, e.g., SPY,QQQ,TLT,GLD")
    p.add_argument("--start", type=str, default="2005-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--rf", type=float, default=0.02, help="Annual cash rate, e.g., 0.02 for 2%")
    p.add_argument("--reb", type=str, default="ME", choices=["ME", "QE"], help="Rebalance freq: ME or QE")
    p.add_argument("--tc_bps", type=int, default=5, help="Transaction cost in bps per turnover unit")
    p.add_argument("--target_vol", type=float, default=None, nargs="?",
                   help="Target annual vol (e.g., 0.10). Omit for None.")
    p.add_argument("--leverage_cap", type=float, default=2.0, help="Max gross leverage when targeting vol")
    p.add_argument("--outdir", type=str, default=".", help="Output base directory")
    a = p.parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    return Config(
        tickers=tickers,
        start=a.start,
        rf_annual=a.rf,
        rebalance=a.reb,
        tc_bps=a.tc_bps,
        target_vol=a.target_vol,
        leverage_cap=a.leverage_cap,
        outdir=a.outdir,
    )


def main() -> None:
    cfg = parse_args()
    out = run_backtest(cfg)
    print("\n=== Performance (net) ===")
    for k, v in out["report"].items():
        try:
            print(f"{k}: {v:.4f}")
        except Exception:
            print(f"{k}: {v}")
    print("\nArtifacts:")
    for k, v in out["paths"].items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
