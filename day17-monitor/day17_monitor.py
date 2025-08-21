"""
Day 17 — Monitor & Safety Rails
Now tolerant to Day 16 orders schema (date,ticker,side,qty,target_shares,price_est,notional_est)
by deriving target_w from target_shares*price_est (or notional_est).

Usage
-----
python day17_monitor.py --positions positions_weights.csv --orders_dir ../day16-monitoring/orders --outdir reports \
  --w_cap 0.30 --turnover_cap 0.25 --sum_tol 0.02 --stale_days 2
"""

from __future__ import annotations
import argparse, os, sys, glob, re, datetime as dt
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

# ---------- Helpers ----------
def _latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    dated = []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if m:
            try:
                d = dt.date.fromisoformat(m.group(1))
            except Exception:
                d = None
        else:
            d = None
        dated.append((f, d, os.path.getmtime(f)))
    dated.sort(key=lambda x: (x[1] or dt.date.min, x[2]))
    return dated[-1][0]

def _pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _read_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = _pick_col(df, ["ticker","symbol","asset"])
    wcol = _pick_col(df, ["weight","w","exec_w","current_w"])
    if tcol is None or wcol is None:
        raise ValueError(f"positions file must have ticker & weight-like columns. Got: {df.columns.tolist()}")
    df = df[[tcol, wcol]].rename(columns={tcol:"ticker", wcol:"exec_w"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.groupby("ticker", as_index=False)["exec_w"].sum()
    return df

def _read_orders(path: str) -> pd.DataFrame:
    """
    Accepts any of:
      - columns: ticker + target_w (preferred)
      - OR ticker + notional_est (sum normalize to weights)
      - OR ticker + target_shares + price_est (shares*price -> notional -> weights)
    """
    df = pd.read_csv(path)
    tcol = _pick_col(df, ["ticker","symbol","asset"])
    if tcol is None:
        raise ValueError(f"orders file missing ticker/symbol column. Got: {df.columns.tolist()}")

    tgtw = _pick_col(df, ["target_w","weight_target","target","w_target","target_weight"])
    if tgtw:
        out = df[[tcol, tgtw]].rename(columns={tcol:"ticker", tgtw:"target_w"})
        out["ticker"] = out["ticker"].astype(str).str.upper()
        out = out.groupby("ticker", as_index=False)["target_w"].sum()
        return out

    # derive weights from notional
    not_col = _pick_col(df, ["notional_est","target_notional","notional"])
    if not_col:
        tmp = df[[tcol, not_col]].rename(columns={tcol:"ticker", not_col:"_not"})
        tmp["ticker"] = tmp["ticker"].astype(str).str.upper()
        tmp = tmp.groupby("ticker", as_index=False)["_not"].sum()
        total = float(tmp["_not"].sum())
        if total <= 0:
            raise ValueError("Derived total notional <= 0 from orders file.")
        tmp["target_w"] = tmp["_not"] / total
        return tmp[["ticker","target_w"]]

    # derive from target_shares * price_est
    sh_col = _pick_col(df, ["target_shares","tgt_shares","shares_target"])
    px_col = _pick_col(df, ["price_est","price","px","close"])
    if sh_col and px_col:
        tmp = df[[tcol, sh_col, px_col]].rename(columns={tcol:"ticker", sh_col:"_sh", px_col:"_px"})
        tmp["ticker"] = tmp["ticker"].astype(str).str.upper()
        tmp["_not"] = tmp["_sh"].astype(float) * tmp["_px"].astype(float)
        tmp = tmp.groupby("ticker", as_index=False)["_not"].sum()
        total = float(tmp["_not"].sum())
        if total <= 0:
            raise ValueError("Derived total notional <= 0 from target_shares*price_est.")
        tmp["target_w"] = tmp["_not"] / total
        return tmp[["ticker","target_w"]]

    raise ValueError(f"orders file must contain ticker and target weight columns, or enough to derive them "
                     f"(notional_est OR target_shares+price_est). Got: {df.columns.tolist()}")

def _today() -> dt.date:
    return dt.datetime.now().date()

# ---------- Checks ----------
def run_checks(positions: pd.DataFrame,
               orders: pd.DataFrame,
               *,
               w_cap: float,
               turnover_cap: float,
               sum_tol: float,
               stale_days: int,
               positions_mtime: float,
               orders_mtime: float) -> Tuple[List[str], dict]:
    alerts = []
    facts = {}

    now = dt.datetime.now().timestamp()
    stale_pos_days = (now - positions_mtime) / 86400.0
    stale_ord_days = (now - orders_mtime) / 86400.0
    facts["positions_age_days"] = round(stale_pos_days, 2)
    facts["orders_age_days"]    = round(stale_ord_days, 2)
    if stale_pos_days > stale_days:
        alerts.append(f"[STALE] positions file is {stale_pos_days:.1f} days old (> {stale_days})")
    if stale_ord_days > stale_days:
        alerts.append(f"[STALE] latest orders file is {stale_ord_days:.1f} days old (> {stale_days})")

    all_tickers = sorted(set(positions["ticker"]) | set(orders["ticker"]))
    p = positions.set_index("ticker").reindex(all_tickers).fillna(0.0)
    o = orders.set_index("ticker").reindex(all_tickers).fillna(0.0)

    # turnover (approx): sum |target_w - exec_w|
    turnover = float((o["target_w"] - p["exec_w"]).abs().sum())
    facts["turnover"] = round(turnover, 4)
    if turnover > turnover_cap:
        alerts.append(f"[TURNOVER] turnover {turnover:.3f} > cap {turnover_cap:.3f}")

    # weights sanity
    neg = o["target_w"][o["target_w"] < -1e-9]
    big = o["target_w"][o["target_w"] > w_cap + 1e-12]
    s = float(o["target_w"].sum())
    facts["target_sum"] = round(s, 6)
    facts["target_min"] = round(float(o["target_w"].min()), 6)
    facts["target_max"] = round(float(o["target_w"].max()), 6)

    if len(neg) > 0:
        alerts.append(f"[WEIGHT] {len(neg)} targets are negative (long-only expected)")
    if len(big) > 0:
        alerts.append(f"[WEIGHT] {len(big)} targets exceed cap {w_cap:.2f}")

    # sum tolerance: allow <=1 if you keep cash outside orders; treat 0.98–1.02 as OK by default
    if abs(s - 1.0) > sum_tol and abs(s) > 1e-9:
        alerts.append(f"[SUM] target weights sum {s:.4f} not within 1±{sum_tol}")

    if o["target_w"].isna().any():
        n = int(o["target_w"].isna().sum())
        alerts.append(f"[NAN] {n} NaN target weights found")

    missing = p.index[p["exec_w"].abs() > 1e-9].difference(o.index[o["target_w"].abs() > 1e-12])
    if len(missing) > 0:
        alerts.append(f"[DRIFT] {len(missing)} tickers held but no target provided: {', '.join(list(missing)[:10])}...")

    return alerts, facts

def write_report(outdir: str, alerts: List[str], facts: dict, positions_path: str, orders_path: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"day17_monitor_{_today()}.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("=== Day 17 — Monitor Report ===\n")
        f.write(f"Date:      {_today()}\n")
        f.write(f"Positions: {positions_path}\n")
        f.write(f"Orders:    {orders_path}\n\n")
        f.write("-- Facts --\n")
        for k, v in facts.items():
            f.write(f"{k}: {v}\n")
        f.write("\n-- Alerts --\n")
        if alerts:
            for a in alerts:
                f.write(a + "\n")
        else:
            f.write("None. ✅\n")
    return fn

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 17 — Monitoring & Safety Rails")
    p.add_argument("--positions", type=str, default="positions_weights.csv", help="Ticker/exec_w CSV")
    p.add_argument("--orders_dir", type=str, default="orders", help="Folder containing orders_YYYY-MM-DD.csv")
    p.add_argument("--outdir", type=str, default="reports", help="Where to write the monitor report")
    p.add_argument("--w_cap", type=float, default=0.30, help="Max single weight")
    p.add_argument("--turnover_cap", type=float, default=0.25, help="Daily turnover cap (L1)")
    p.add_argument("--sum_tol", type=float, default=0.02, help="Allowable deviation of target weights sum from 1")
    p.add_argument("--stale_days", type=int, default=2, help="Max age (days) for positions/orders before flagging")
    return p.parse_args()

def main():
    a = parse_args()

    if not os.path.exists(a.positions):
        sys.exit(f"ERROR: positions file not found: {a.positions}")
    latest_orders = _latest_file(os.path.join(a.orders_dir, "orders_*.csv"))
    if not latest_orders:
        sys.exit(f"ERROR: no orders files found matching {a.orders_dir}/orders_*.csv")

    pos = _read_positions(a.positions)
    ords = _read_orders(latest_orders)

    alerts, facts = run_checks(
        pos, ords,
        w_cap=a.w_cap,
        turnover_cap=a.turnover_cap,
        sum_tol=a.sum_tol,
        stale_days=a.stale_days,
        positions_mtime=os.path.getmtime(a.positions),
        orders_mtime=os.path.getmtime(latest_orders),
    )
    report_path = write_report(a.outdir, alerts, facts, a.positions, latest_orders)

    print(f"\nReport written: {report_path}")
    if alerts:
        print("\nALERTS:")
        for a_ in alerts:
            print(" -", a_)
        sys.exit(2)
    else:
        print("All checks passed. ✅")

if __name__ == "__main__":
    main()
