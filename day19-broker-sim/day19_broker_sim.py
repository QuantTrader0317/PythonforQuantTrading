"""
Day 19 — Broker Sim + Reconciliation

Reads the latest orders, simulates fills (slippage/commission/partial),
updates positions ledger (shares + cash), and writes a reconciliation report.

Run (PowerShell):
py .\day19_broker_sim.py ^
  --orders_dir "..\day16-monitoring\orders" ^
  --positions "..\day16-monitoring\live\positions.csv" ^
  --prices "..\day16-monitoring\outputs\latest_close.csv" ^
  --fills_dir ".\fills" --outdir ".\reports" ^
  --slippage_bps 2 --commission_bps 0 --fill_rate 1.0
"""

from __future__ import annotations
import argparse, os, glob, datetime as dt
from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

TRADING_DAYS = 252

# ---------- I/O ----------
def latest_orders_path(orders_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(orders_dir, "orders_*.csv")))
    if not files:
        raise SystemExit(f"No orders_*.csv found in {orders_dir}")
    return files[-1]

def read_orders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ticker", "side", "qty"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Orders must have {need}. Got {df.columns.tolist()}")
    # Optional columns for pricing
    if "price_est" not in df.columns and "notional_est" not in df.columns:
        df["price_est"] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper()
    # normalize side to +1/-1
    side_map = {"BUY": 1, "SELL": -1, "Long": 1, "Short": -1}
    df["side_norm"] = df["side"].map(side_map).fillna(0).astype(int)
    df["qty"] = df["qty"].astype(float)
    return df

def read_positions_ledger(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        last = df.iloc[-1]
        return df, last
    # seed if missing
    cols = ["cash"]  # shares cols appear as we trade
    df = pd.DataFrame(columns=cols)
    last = pd.Series({"cash": 100000.0})  # default starting cash
    return df, last

def read_prices(tickers, prices_csv: Optional[str], orders_df: pd.DataFrame) -> pd.Series:
    # 1) explicit CSV
    if prices_csv and os.path.exists(prices_csv):
        px = pd.read_csv(prices_csv)
        px = px.rename(columns={c: c.lower() for c in px.columns})
        if {"symbol","close"}.issubset(px.columns):
            s = px.set_index("symbol")["close"].reindex(tickers)
            if s.isna().any():
                # fall back to price_est from orders for missing tickers
                est = (orders_df.groupby("ticker")["price_est"].mean()
                                 .reindex(tickers))
                s = s.fillna(est)
            return s.astype(float)
    # 2) yfinance fallback
    if _HAS_YF:
        data = yf.download(list(tickers), period="5d", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        s = data.ffill().iloc[-1].reindex(tickers)
        # final fallback to price_est if still missing
        est = (orders_df.groupby("ticker")["price_est"].mean()
                         .reindex(tickers))
        return s.fillna(est).astype(float)
    # 3) only orders have prices
    est = (orders_df.groupby("ticker")["price_est"].mean()
                     .reindex(tickers))
    if est.isna().any():
        raise SystemExit("No prices available; provide --prices or install yfinance.")
    return est.astype(float)

# ---------- math ----------
def weights_from_positions(shares: pd.Series, prices: pd.Series, cash: float) -> pd.Series:
    shares = shares.reindex(prices.index).fillna(0.0)
    notional = (shares * prices).fillna(0.0)
    nav = float(notional.sum() + cash)
    if nav <= 0:
        return pd.Series(0.0, index=prices.index)
    w = notional / nav
    return w

# ---------- sim ----------
def simulate_fills(orders: pd.DataFrame,
                   prices: pd.Series,
                   *,
                   slippage_bps: float,
                   commission_bps: float,
                   fill_rate: float) -> pd.DataFrame:
    """
    Returns a fills table: ticker, qty_fill, fill_px, cash_delta, commission
    qty_fill = qty * side_norm * fill_rate
    fill_px  = price_est or market px, with slippage applied
    """
    df = orders.copy()
    # prefer price_est per order; fallback to px map
    px = df["price_est"].copy()
    miss = px.isna()
    if miss.any():
        px.loc[miss] = df.loc[miss, "ticker"].map(prices)
    # slippage: move price against you
    slip = slippage_bps / 10000.0
    df["fill_px"] = px * (1.0 + np.sign(df["side_norm"]) * slip)
    df["qty_fill"] = df["qty"] * df["side_norm"] * float(fill_rate)
    # gross cash delta
    df["cash_delta_gross"] = -(df["qty_fill"] * df["fill_px"])
    # commission
    comm = commission_bps / 10000.0
    df["commission"] = np.abs(df["cash_delta_gross"]) * comm
    df["cash_delta_net"] = df["cash_delta_gross"] - df["commission"]
    return df[["ticker","qty_fill","fill_px","cash_delta_net","commission"]]

def apply_fills_to_positions(last_row: pd.Series,
                             fills: pd.DataFrame) -> pd.Series:
    # update cash
    cash = float(last_row.get("cash", 0.0)) + float(fills["cash_delta_net"].sum())
    # update shares
    new_row = last_row.copy()
    new_row["cash"] = cash
    for t, q in fills[["ticker","qty_fill"]].itertuples(index=False):
        prev = float(new_row.get(t, 0.0))
        new_row[t] = prev + float(q)
    return new_row

# ---------- outputs ----------
def write_fills(fills_dir: str, fills: pd.DataFrame) -> str:
    os.makedirs(fills_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(fills_dir, f"fills_{ts}.csv")
    fills.to_csv(path, index=False)
    return path

def append_positions(positions_path: str, df_hist: pd.DataFrame, new_row: pd.Series) -> str:
    date = dt.datetime.now().strftime("%Y-%m-%d")
    new_row.name = pd.to_datetime(date)
    df_out = pd.concat([df_hist, new_row.to_frame().T], axis=0)
    df_out = df_out.sort_index()
    os.makedirs(os.path.dirname(positions_path), exist_ok=True)
    df_out.to_csv(positions_path)
    return positions_path

def write_recon(outdir: str,
                tickers: list[str],
                w_pre: pd.Series,
                w_tgt: pd.Series,
                w_post: pd.Series) -> str:
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({
        "ticker": tickers,
        "exec_w_before": w_pre.reindex(tickers).fillna(0.0).values,
        "target_w":      w_tgt.reindex(tickers).fillna(0.0).values,
        "exec_w_after":  w_post.reindex(tickers).fillna(0.0).values,
    })
    df["abs_gap_after"] = (df["target_w"] - df["exec_w_after"]).abs()
    txt = os.path.join(outdir, f"day19_recon_{dt.date.today()}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("=== Day 19 — Reconciliation ===\n")
        f.write(f"Date: {dt.date.today()}\n\n")
        f.write(df.sort_values("abs_gap_after", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n")
        f.write(f"\nSum |target - after|: {float((df['target_w']-df['exec_w_after']).abs().sum()):.4f}\n")
    # also save csv
    df.to_csv(txt.replace(".txt",".csv"), index=False)
    return txt

# ---------- target weights from orders ----------
def target_weights_from_orders(orders: pd.DataFrame, prices: pd.Series) -> pd.Series:
    # if explicit target_w exists, use it; else derive from target notional if present
    if "target_w" in orders.columns:
        s = (orders[["ticker","target_w"]]
             .groupby("ticker")["target_w"].sum())
        if np.isclose(s.sum(), 0.0):
            # derive from qty*price if zero sum
            notional = (orders["qty"].abs() * prices.reindex(orders["ticker"]).values)
            s = pd.Series(notional, index=orders["ticker"]).groupby(level=0).sum()
            s = s / s.sum()
        return s
    # derive from qty * price
    notional = (orders["qty"].abs() * prices.reindex(orders["ticker"]).values)
    s = pd.Series(notional, index=orders["ticker"]).groupby(level=0).sum()
    return s / s.sum()

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 19 — Broker Sim + Reconciliation")
    p.add_argument("--orders_dir", type=str, default="../day16-monitoring/orders")
    p.add_argument("--positions", type=str, default="../day16-monitoring/live/positions.csv")
    p.add_argument("--prices", type=str, default="../day16-monitoring/outputs/latest_close.csv",
                   help="CSV with columns symbol,close (optional; falls back to yfinance)")
    p.add_argument("--fills_dir", type=str, default="./fills")
    p.add_argument("--outdir", type=str, default="./reports")
    p.add_argument("--slippage_bps", type=float, default=2.0)
    p.add_argument("--commission_bps", type=float, default=0.0)
    p.add_argument("--fill_rate", type=float, default=1.0, help="0..1 partial fill rate")
    return p.parse_args()

def main():
    a = parse_args()
    orders_path = latest_orders_path(a.orders_dir)
    ords = read_orders(orders_path)

    tickers = sorted(ords["ticker"].unique().tolist())
    px = read_prices(tickers, a.prices, ords)

    # positions before
    pos_hist, last = read_positions_ledger(a.positions)
    shares_before = last.drop(labels=["cash"], errors="ignore")
    cash_before = float(last.get("cash", 0.0))
    w_pre = weights_from_positions(shares_before, px.reindex(shares_before.index).fillna(px.mean()), cash_before)

    # simulate fills
    fills = simulate_fills(ords, px,
                           slippage_bps=a.slippage_bps,
                           commission_bps=a.commission_bps,
                           fill_rate=a.fill_rate)
    fills_path = write_fills(a.fills_dir, fills)

    # apply fills
    new_last = apply_fills_to_positions(last, fills)
    # positions after
    shares_after = new_last.drop(labels=["cash"], errors="ignore")
    cash_after = float(new_last.get("cash", 0.0))
    # mark any leftover tickers (not in today's orders) at their known prices if available
    all_marks = px.copy()
    for t in shares_after.index:
        if t not in all_marks.index:
            all_marks.loc[t] = px.mean()
    w_post = weights_from_positions(shares_after, all_marks.reindex(shares_after.index), cash_after)

    # targets from orders
    w_tgt = target_weights_from_orders(ords, px)

    # outputs
    pos_path = append_positions(a.positions, pos_hist, new_last)
    recon_path = write_recon(a.outdir, sorted(set(w_pre.index) | set(w_tgt.index) | set(w_post.index)),
                             w_pre, w_tgt, w_post)

    print(f"Filled orders:   {orders_path}")
    print(f"Fills written:   {fills_path}")
    print(f"Positions saved: {pos_path}")
    print(f"Recon report:    {recon_path}")

if __name__ == "__main__":
    main()
