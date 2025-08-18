"""
Day 16 — Live Monitoring & Paper-Trading Harness

What it does (single daily run):
1) Pull latest prices (yfinance) for your universe
2) Compute target weights:
   - inverse-vol (risk-parity-lite) monthly
   - regime-aware volatility targeting using ^TNX (10y yield) and ^VIX
3) Compare to current paper positions, apply band/turnover caps
4) Emit orders (CSV), paper-fill them (spread + simple impact), update positions/cash
5) Write a one-page CSV metrics snapshot and a simple PNG chart (equity + scaler)

NO broker connectivity. Safe dress rehearsal for live.
"""

from __future__ import annotations
import argparse, os, math, json, datetime as dt
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

TRADING_DAYS = 252


# ---------- Utils / IO ----------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def today_str() -> str:
    return dt.date.today().isoformat()


def safe_freq(freq: str) -> str:
    # pandas changed 'M'/'Q' to 'ME'/'QE'
    return {"M": "ME", "Q": "QE"}.get(freq, freq)


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index.to_series().resample(safe_freq(freq)).last().index


def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index().dropna(how="all").ffill()
    # keep business days only (helps with index alignment)
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_single_close(symbol: str, start: str) -> pd.Series:
    y = yf.download(symbol, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    return pd.Series(y).dropna().sort_index()


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


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
    if not np.isfinite(v) or v == 0:
        return np.nan
    return (cagr(x, periods) - rf_annual) / v


def max_dd(eq: pd.Series) -> float:
    eq = eq.dropna()
    if not len(eq): return np.nan
    peak = eq.cummax()
    return float((eq / peak - 1).min())


# ---------- Strategy bricks ----------

def rolling_vol_df(r: pd.DataFrame, window: int, method: str = "rolling") -> pd.DataFrame:
    if method == "ewma":
        return r.ewm(span=window).std() * np.sqrt(TRADING_DAYS)
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)


def weights_inverse_vol(r: pd.DataFrame,
                        window: int,
                        reb: str,
                        vol_method: str = "rolling",
                        w_cap: Optional[float] = None,
                        band: float = 0.0) -> pd.DataFrame:
    """
    Monthly inverse-vol weights with optional weight-cap and rebalance-band.
    """
    rebs = rebalance_dates(r.index, reb)
    vol = rolling_vol_df(r, window, vol_method)
    w_t = pd.DataFrame(index=r.index, columns=r.columns, data=np.nan)

    prev = None
    for dt_i in rebs:
        if dt_i not in vol.index:
            continue
        v = vol.loc[dt_i].replace(0, np.nan).dropna()
        if v.empty:
            continue
        w = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
        w = w.clip(lower=0)
        if w_cap is not None and w_cap > 0:
            w = w.clip(upper=w_cap)
        s = w.sum()
        if s > 0:
            w = w / s
        if prev is not None:
            # L1 drift for banding
            w_al = w.reindex(prev.index).fillna(0.0)
            prev_al = prev.reindex(w.index).fillna(0.0)
            drift = float((w_al - prev_al).abs().sum())
            if band > 0 and drift < band:
                w = prev.copy()
        w_t.loc[dt_i, w.index] = w.values
        prev = w.copy()

    return w_t.ffill().fillna(0.0)


def get_hostile_series(index: pd.DatetimeIndex,
                       use_tnx: bool = True, tnx_span: int = 63,
                       use_vix: bool = True, vix_span: int = 63,
                       host_or: bool = True) -> pd.Series:
    """
    Hostile regime when:
      - ^VIX > EMA(vix, span)  (risk-off),
      - ^TNX > EMA(tnx, span)  (yields rising; often risk-off for bonds/long-duration equities)
    Combine with OR (default) or AND.
    """
    start = str(index[0].date())
    host_bits = []

    if use_vix:
        try:
            vix = fetch_single_close("^VIX", start).reindex(index).ffill()
            ema_vix = vix.ewm(span=vix_span).mean()
            host_bits.append((vix > ema_vix).astype(int))
        except Exception:
            host_bits.append(pd.Series(0, index=index))
    if use_tnx:
        try:
            tnx = fetch_single_close("^TNX", start).reindex(index).ffill()
            ema_tnx = tnx.ewm(span=tnx_span).mean()
            host_bits.append((tnx > ema_tnx).astype(int))
        except Exception:
            host_bits.append(pd.Series(0, index=index))

    if not host_bits:
        return pd.Series(0, index=index)

    host = pd.concat(host_bits, axis=1).fillna(0).astype(int)
    if host_or:
        out = (host.max(axis=1) > 0).astype(int)
    else:
        out = (host.sum(axis=1) == host.shape[1]).astype(int)
    return out.rename("hostile")


def scaler_vol_target(r: pd.Series,
                      vt_target: float = 0.10,
                      vt_span: int = 63,
                      vt_min: float = 0.5,
                      vt_max: float = 1.0,
                      hostile: Optional[pd.Series] = None,
                      hostile_reduce: float = 0.7) -> pd.Series:
    """
    Build a daily exposure scaler s_t in [vt_min, vt_max].
    If hostile==1 reduce the target by hostile_reduce (e.g., 0.7).
    NOTE: we cap at 1.0 to avoid leverage in paper-trading.
    """
    rv = r.rolling(vt_span).std() * np.sqrt(TRADING_DAYS)
    rv = rv.replace(0.0, np.nan).ffill()
    base_target = pd.Series(vt_target, index=rv.index)
    if hostile is not None:
        base_target = base_target.where(hostile == 0, vt_target * hostile_reduce)
    s = (base_target / rv).clip(lower=vt_min, upper=vt_max).fillna(vt_min)
    return s.rename("scaler")


# ---------- Paper trading ----------

def load_positions(path: str, tickers: List[str], initial_cash: float) -> pd.Series:
    """
    Positions file: a single row with index ['cash', *tickers]
    Shares are integers, cash is float USD.
    """
    if os.path.exists(path):
        pos = pd.read_csv(path, index_col=0).iloc[-1]
        # ensure all tickers exist
        for t in tickers:
            if t not in pos.index:
                pos[t] = 0
        if "cash" not in pos.index:
            pos["cash"] = initial_cash
        return pos[tickers + ["cash"]]
    # initialize
    pos = pd.Series({t: 0 for t in tickers})
    pos["cash"] = float(initial_cash)
    save_positions(path, pos)
    return pos


def save_positions(path: str, pos: pd.Series) -> None:
    # append-friendly small ledger
    df = pd.DataFrame(pos).T
    df.index = [today_str()]
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header)


def estimate_atr(px: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    r = px.pct_change()
    atr = (r.rolling(span).std() * px).fillna(method="bfill").fillna(0.0)
    return atr


def generate_orders(today: pd.Timestamp,
                    px_today: pd.Series,
                    w_target: pd.Series,
                    pos: pd.Series,
                    no_trade_band: float,
                    turnover_cap: float) -> pd.DataFrame:
    """
    Compare current weights vs target; return desired share changes by ticker.
    """
    tickers = [c for c in w_target.index if w_target[c] > 0 or c in pos.index]
    tickers = [t for t in tickers if t in px_today.index]

    # Current NAV
    nav = float(pos["cash"] + sum(pos[t] * px_today[t] for t in tickers))

    # Current weights
    curr_w = pd.Series({t: (pos[t] * px_today[t]) / nav if nav > 0 else 0.0 for t in tickers})

    # L1 drift band
    drift = float((w_target.reindex(curr_w.index).fillna(0.0) - curr_w).abs().sum())
    if drift < no_trade_band:
        return pd.DataFrame(columns=["ticker", "side", "qty", "target_shares",
                                     "price_est", "notional_est"])

    # Target shares (round to integers)
    tgt_shares = pd.Series({
        t: int(round((w_target[t] * nav) / max(px_today[t], 1e-8)))
        for t in tickers
    })

    delta = tgt_shares - pd.Series({t: int(pos.get(t, 0)) for t in tickers})
    # turnover cap on weights -> cap on notional delta
    if turnover_cap > 0:
        # translate to weight change approximation
        w_delta = (delta * px_today).abs().sum() / max(nav, 1e-8)
        if w_delta > turnover_cap:
            scale = turnover_cap / w_delta
            delta = (delta.astype(float) * scale).round().astype(int)

    orders = []
    for t in tickers:
        q = int(delta[t])
        if q == 0:
            continue
        side = "BUY" if q > 0 else "SELL"
        orders.append([t, side, q, int(tgt_shares[t]), float(px_today[t]), float(abs(q) * px_today[t])])

    odf = pd.DataFrame(orders, columns=["ticker", "side", "qty", "target_shares", "price_est", "notional_est"])
    odf.insert(0, "date", pd.Timestamp(today).date())
    return odf


def paper_fill_and_update(orders: pd.DataFrame,
                          pos: pd.Series,
                          px_today: pd.Series,
                          atr_today: pd.Series,
                          spread_bps: float,
                          impact_coef: float) -> pd.Series:
    """
    Simple microstructure: fill at close +/- half-spread and +/- impact * ATR.
    Impact sign follows trade direction.
    """
    for _, row in orders.iterrows():
        t = row["ticker"]
        q = int(row["qty"])
        side = 1 if q > 0 else -1
        price = float(px_today[t])
        half_spread = price * (spread_bps / 20000.0)  # half the spread in price units
        impact = float(atr_today.get(t, 0.0)) * impact_coef
        fill = price + side * (half_spread + impact)

        # Update shares & cash
        pos[t] = int(pos.get(t, 0)) + q
        pos["cash"] = float(pos["cash"]) - q * fill
    return pos


# ---------- Reporting ----------

def write_orders(orders_dir: str, odf: pd.DataFrame) -> str:
    ensure_dir(orders_dir)
    path = os.path.join(orders_dir, f"orders_{today_str()}.csv")
    odf.to_csv(path, index=False)
    return path


def write_metrics(outdir: str, nav_series: pd.Series, ret_series: pd.Series, extra: Dict) -> str:
    ensure_dir(outdir)
    eq = nav_series / nav_series.iloc[0]
    metrics = {
        "CAGR": cagr(ret_series),
        "Vol": ann_vol(ret_series),
        "Sharpe": sharpe(ret_series),
        "MaxDD": max_dd(eq),
        "LastNAV": float(nav_series.iloc[-1])
    }
    metrics.update(extra)
    path = os.path.join(outdir, f"day16_metrics_{today_str()}.csv")
    pd.DataFrame(metrics, index=[0]).to_csv(path, index=False)
    return path


def plot_report(chartdir: str,
                eq: pd.Series,
                scaler: pd.Series,
                hostile: pd.Series) -> str:
    ensure_dir(chartdir)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)

    axes[0].plot(eq.index, eq.values, label="Paper NAV (normalized)")
    axes[0].set_title("Day 16 — Paper NAV")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scaler.index, scaler.values, label="Exposure scaler (s)", lw=1.8)
    axes[1].plot(hostile.index, hostile.values, label="Hostile (1=yes)", lw=1.2)
    axes[1].set_ylim(-0.1, max(1.1, float(scaler.max()) + 0.1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    path = os.path.join(chartdir, f"day16_report_{today_str()}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------- Orchestration ----------

def run_once(a) -> None:
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    ensure_dir(a.outdir)
    ensure_dir(a.orders_dir)
    ensure_dir(a.live_dir)
    ensure_dir(a.logs_dir)
    ensure_dir(a.charts_dir)

    # 1) Data
    px = fetch_prices(tickers, a.start)
    r = daily_returns(px)
    if r.empty:
        raise SystemExit("No return data. Check tickers / start date.")

    # 2) Weights (inverse-vol monthly)
    w_rp = weights_inverse_vol(r, a.window, a.reb, a.vol_method, a.w_cap, a.alloc_band)
    w_today = w_rp.iloc[-1].copy()

    # 3) Regime-aware scaler
    hostile = get_hostile_series(r.index, a.use_tnx, a.tnx_span, a.use_vix, a.vix_span, a.host_or)
    scaler = scaler_vol_target(r.sum(axis=1), a.vt_target, a.vt_span, a.vt_min, a.vt_max,
                               hostile=hostile, hostile_reduce=a.reduce)
    s_today = float(scaler.iloc[-1])

    # Cap at 1.0 (no leverage in paper) and assign residual to cash
    s_capped = min(s_today, 1.0)
    w_target = (w_today * s_capped).clip(lower=0.0)
    # normalize for safety (sum <= 1.0)
    if w_target.sum() > 1.0:
        w_target = w_target / w_target.sum()
    # implicit cash = 1 - s_capped (not traded)

    # 4) Build orders
    pos_path = os.path.join(a.live_dir, "positions.csv")
    pos = load_positions(pos_path, tickers, a.initial_cash)

    today = px.index[-1]
    px_today = px.loc[today]
    atr = estimate_atr(px).loc[today]

    orders = generate_orders(today, px_today, w_target, pos, a.no_trade_band, a.turnover_cap)

    if not orders.empty:
        # 5) Paper fill & update
        pos = paper_fill_and_update(orders, pos, px_today, atr, a.spread_bps, a.impact_coef)
        save_positions(pos_path, pos)
        orders_path = write_orders(a.orders_dir, orders)
        log = f"[{today_str()}] {len(orders)} orders written to {orders_path}"
    else:
        # still append current pos to keep a NAV history
        save_positions(pos_path, pos)
        log = f"[{today_str()}] No trade (band/turnover). Positions updated for NAV history."

    # 6) NAV history & metrics/report
    pos_hist = pd.read_csv(pos_path, index_col=0, parse_dates=True)
    nav_series = (pos_hist[tickers] * px.reindex(pos_hist.index).ffill()).sum(axis=1) + pos_hist["cash"]
    nav_series = nav_series.dropna()
    eq = nav_series / nav_series.iloc[0]
    ret_series = nav_series.pct_change().dropna()

    metrics_path = write_metrics(a.outdir, nav_series, ret_series, {
        "scaler_today": s_today,
        "hostile_today": int(hostile.iloc[-1]),
        "orders_count": 0 if orders.empty else int(len(orders)),
    })
    chart_path = plot_report(a.charts_dir, eq, scaler.reindex(eq.index).ffill(), hostile.reindex(eq.index).ffill())

    # 7) Log
    with open(os.path.join(a.logs_dir, f"day16_run_{today_str()}.txt"), "a", encoding="utf-8") as f:
        f.write(log + "\n")
        f.write(f"metrics: {metrics_path}\n")
        f.write(f"chart:   {chart_path}\n")
        f.write(json.dumps({"tickers": tickers}, indent=2) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Day 16 — Live Monitoring & Paper Trading")
    # Universe & data
    p.add_argument("--tickers", type=str,
                   default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", type=str, default="2018-01-01")
    # Strategy knobs
    p.add_argument("--reb", type=str, default="M", choices=["M", "Q"])
    p.add_argument("--window", type=int, default=63)
    p.add_argument("--vol_method", type=str, default="rolling", choices=["rolling", "ewma"])
    p.add_argument("--w_cap", type=float, default=None, help="cap any single weight (e.g., 0.25)")
    p.add_argument("--alloc_band", type=float, default=0.0, help="rebalance L1 band on weights")

    # Vol targeting + regime
    p.add_argument("--vt_target", type=float, default=0.10)
    p.add_argument("--vt_span", type=int, default=63)
    p.add_argument("--vt_min", type=float, default=0.5)
    p.add_argument("--vt_max", type=float, default=1.0)  # <=1 to avoid leverage in paper
    p.add_argument("--use_tnx", action="store_true", default=True)
    p.add_argument("--tnx_span", type=int, default=63)
    p.add_argument("--use_vix", action="store_true", default=True)
    p.add_argument("--vix_span", type=int, default=63)
    p.add_argument("--host_or", action="store_true", default=True,
                   help="combine signals with OR (else AND if flag not used)")
    p.add_argument("--reduce", type=float, default=0.7, help="hostile regime reduces target by this factor")

    # Execution / paper fills
    p.add_argument("--no_trade_band", type=float, default=0.01, help="skip if L1 weight drift < band")
    p.add_argument("--turnover_cap", type=float, default=0.20, help="cap L1 weight change per day")
    p.add_argument("--spread_bps", type=float, default=2.0)
    p.add_argument("--impact_coef", type=float, default=0.10, help="fraction of ATR added to price")
    p.add_argument("--initial_cash", type=float, default=100000.0)

    # Folders
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--orders_dir", type=str, default="orders")
    p.add_argument("--live_dir", type=str, default="live")
    p.add_argument("--logs_dir", type=str, default="logs")
    p.add_argument("--charts_dir", type=str, default="charts")
    return p.parse_args()


def main():
    a = parse_args()
    run_once(a)
    print("Day 16 run complete. Check:", a.orders_dir, a.live_dir, a.outdir, a.charts_dir, a.logs_dir)


if __name__ == "__main__":
    main()
