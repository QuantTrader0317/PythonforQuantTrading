"""
Day 18 — Alerts + Mini Dashboard

Reads positions + latest orders, runs safety checks (like Day 17),
emits an HTML report, and optionally sends Slack/email alerts on failure.

Usage (PowerShell):
py .\day18_alerts.py ^
  --positions "..\day17-monitor\positions_weights.csv" ^
  --orders_dir "..\day16-monitoring\orders" ^
  --outdir ".\reports" ^
  --w_cap 0.30 --turnover_cap 0.25 --sum_tol 0.02 --stale_days 2 ^
  --slack_webhook "https://hooks.slack.com/services/XXX/YYY/ZZZ"

Env (optional):
SLACK_WEBHOOK_URL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
"""

from __future__ import annotations
import argparse, os, sys, glob, re, datetime as dt, base64, io, smtplib, ssl
from email.mime.text import MIMEText
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from jinja2 import Template

TRADING_DAYS = 252

# ---------- utils ----------
def _latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    dated = []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        d = None
        if m:
            try:
                d = dt.date.fromisoformat(m.group(1))
            except Exception:
                d = None
        dated.append((f, d, os.path.getmtime(f)))
    dated.sort(key=lambda x: (x[1] or dt.date.min, x[2]))
    return dated[-1][0]

def _pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def _read_positions_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = _pick_col(df, ["ticker","symbol","asset"])
    wcol = _pick_col(df, ["exec_w","weight","w","current_w"])
    if tcol is None or wcol is None:
        raise ValueError(f"positions file must have ticker & weight-like columns. Got: {df.columns.tolist()}")
    df = df[[tcol, wcol]].rename(columns={tcol:"ticker", wcol:"exec_w"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.groupby("ticker", as_index=False)["exec_w"].sum()

def _read_orders_any(path: str) -> pd.DataFrame:
    """
    Accepts any of:
      - ticker + target_w
      - ticker + notional_est
      - ticker + target_shares + price_est
    Returns: DataFrame[ticker, target_w]
    """
    df = pd.read_csv(path)
    tcol = _pick_col(df, ["ticker","symbol","asset"])
    if tcol is None:
        raise ValueError(f"orders file missing ticker/symbol column. Got: {df.columns.tolist()}")
    # 1) direct weights
    tgtw = _pick_col(df, ["target_w","weight_target","target","w_target","target_weight"])
    if tgtw:
        out = df[[tcol, tgtw]].rename(columns={tcol:"ticker", tgtw:"target_w"})
        out["ticker"] = out["ticker"].astype(str).str.upper()
        return out.groupby("ticker", as_index=False)["target_w"].sum()
    # 2) notional
    not_col = _pick_col(df, ["notional_est","target_notional","notional"])
    if not_col:
        tmp = df[[tcol, not_col]].rename(columns={tcol:"ticker", not_col:"_not"})
        tmp["ticker"] = tmp["ticker"].astype(str).str.upper()
        tmp = tmp.groupby("ticker", as_index=False)["_not"].sum()
        total = float(tmp["_not"].sum())
        if total <= 0: raise ValueError("Total notional <= 0.")
        tmp["target_w"] = tmp["_not"] / total
        return tmp[["ticker","target_w"]]
    # 3) shares * price
    sh_col = _pick_col(df, ["target_shares","tgt_shares","shares_target"])
    px_col = _pick_col(df, ["price_est","price","px","close"])
    if sh_col and px_col:
        tmp = df[[tcol, sh_col, px_col]].rename(columns={tcol:"ticker", sh_col:"_sh", px_col:"_px"})
        tmp["ticker"] = tmp["ticker"].astype(str).str.upper()
        tmp["_not"] = tmp["_sh"].astype(float) * tmp["_px"].astype(float)
        tmp = tmp.groupby("ticker", as_index=False)["_not"].sum()
        total = float(tmp["_not"].sum())
        if total <= 0: raise ValueError("Total notional <= 0 from shares*price.")
        tmp["target_w"] = tmp["_not"] / total
        return tmp[["ticker","target_w"]]
    raise ValueError(f"orders file must have ticker and target_w, or enough to derive it. Got: {df.columns.tolist()}")

def _today() -> dt.date:
    return dt.datetime.now().date()

# ---------- checks ----------
def run_checks(positions: pd.DataFrame,
               orders: pd.DataFrame,
               *,
               w_cap: float,
               turnover_cap: float,
               sum_tol: float,
               stale_days: int,
               positions_mtime: float,
               orders_mtime: float) -> Tuple[List[str], dict]:
    alerts, facts = [], {}
    now = dt.datetime.now().timestamp()
    pos_age = (now - positions_mtime)/86400.0
    ord_age = (now - orders_mtime)/86400.0
    facts["positions_age_days"] = round(pos_age, 2)
    facts["orders_age_days"] = round(ord_age, 2)
    if pos_age > stale_days:
        alerts.append(f"[STALE] positions {pos_age:.1f}d old (> {stale_days})")
    if ord_age > stale_days:
        alerts.append(f"[STALE] orders {ord_age:.1f}d old (> {stale_days})")
    # align
    all_tk = sorted(set(positions["ticker"]) | set(orders["ticker"]))
    p = positions.set_index("ticker").reindex(all_tk).fillna(0.0)
    o = orders.set_index("ticker").reindex(all_tk).fillna(0.0)
    # turnover
    turnover = float((o["target_w"] - p["exec_w"]).abs().sum())
    facts["turnover"] = round(turnover, 4)
    if turnover > turnover_cap:
        alerts.append(f"[TURNOVER] {turnover:.3f} > cap {turnover_cap:.3f}")
    # sanity
    s = float(o["target_w"].sum())
    facts["target_sum"] = round(s, 6)
    facts["target_min"] = round(float(o["target_w"].min()), 6) if len(o)>0 else 0.0
    facts["target_max"] = round(float(o["target_w"].max()), 6) if len(o)>0 else 0.0
    neg = int((o["target_w"] < -1e-9).sum())
    big = int((o["target_w"] > w_cap + 1e-12).sum())
    if neg: alerts.append(f"[WEIGHT] {neg} negative targets (long-only expected)")
    if big: alerts.append(f"[WEIGHT] {big} targets exceed cap {w_cap:.2f}")
    if abs(s - 1.0) > sum_tol and abs(s) > 1e-9:
        alerts.append(f"[SUM] target sum {s:.4f} not within 1±{sum_tol}")
    if o["target_w"].isna().any():
        alerts.append(f"[NAN] NaN target weights found")
    missing = p.index[p["exec_w"].abs() > 1e-9].difference(o.index[o["target_w"].abs() > 1e-12])
    if len(missing) > 0:
        alerts.append(f"[DRIFT] held but no target: {', '.join(list(missing)[:10])}...")
    return alerts, facts

# ---------- small NAV sparkline (optional) ----------
def _load_nav_series(metrics_dir: str) -> Optional[pd.Series]:
    # Look for day16_metrics_*.csv with columns date, nav (best effort)
    files = sorted(glob.glob(os.path.join(metrics_dir, "day16_metrics_*.csv")))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    dcol = _pick_col(df, ["date","asof"])
    ncol = _pick_col(df, ["nav","equity","portfolio_value"])
    if not dcol or not ncol: return None
    s = pd.Series(df[ncol].values, index=pd.to_datetime(df[dcol]))
    return s.dropna()

def _sparkline_png_b64(s: pd.Series) -> Optional[str]:
    if s is None or len(s) < 5: return None
    fig = plt.figure(figsize=(4, 0.8), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------- alerts ----------
def send_slack(webhook_url: str, text: str) -> None:
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] Slack send failed: {e}")

def send_email_smtp(host: str, port: int, user: str, pwd: str, to_addr: str, subject: str, body_html: str):
    msg = MIMEText(body_html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=ctx)
        server.login(user, pwd)
        server.sendmail(user, [to_addr], msg.as_string())

# ---------- HTML ----------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Day 18 — Daily Report {{ date }}</title>
<style>
 body{font-family:Inter,system-ui,Arial,sans-serif;margin:20px}
 h1{margin:0 0 8px 0} .muted{color:#666}
 .pill{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px}
 .ok{background:#e6ffed;color:#006d32} .bad{background:#ffecec;color:#b00020}
 table{border-collapse:collapse;width:100%;margin-top:12px}
 th,td{border-bottom:1px solid #eee;padding:6px;text-align:left}
 .small{font-size:12px;color:#666}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .card{border:1px solid #eee;border-radius:12px;padding:12px}
 .spark{height:80px}
</style>
</head>
<body>
<h1>Day 18 — Daily Report <span class="muted">{{ date }}</span></h1>

<div class="grid">
  <div class="card">
    <h3>Facts</h3>
    <table>
      <tr><td>positions_age_days</td><td>{{ facts.positions_age_days }}</td></tr>
      <tr><td>orders_age_days</td><td>{{ facts.orders_age_days }}</td></tr>
      <tr><td>turnover</td><td>{{ facts.turnover }}</td></tr>
      <tr><td>target_sum</td><td>{{ facts.target_sum }}</td></tr>
      <tr><td>target_min</td><td>{{ facts.target_min }}</td></tr>
      <tr><td>target_max</td><td>{{ facts.target_max }}</td></tr>
    </table>
  </div>
  <div class="card">
    <h3>Alerts</h3>
    {% if alerts|length == 0 %}
      <span class="pill ok">None</span>
    {% else %}
      {% for a in alerts %}
        <div><span class="pill bad">Alert</span> {{ a }}</div>
      {% endfor %}
    {% endif %}
  </div>
</div>

<div class="card">
  <h3>Top target weights</h3>
  <table>
    <tr><th>Ticker</th><th>Target w</th></tr>
    {% for row in top_targets %}
      <tr><td>{{ row.ticker }}</td><td>{{ "%.3f"|format(row.target_w) }}</td></tr>
    {% endfor %}
  </table>
  <div class="small">Showing top 10 by target weight.</div>
</div>

{% if spark %}
<div class="card">
  <h3>NAV sparkline (paper)</h3>
  <img class="spark" src="data:image/png;base64,{{ spark }}" />
</div>
{% endif %}

<div class="small muted" style="margin-top:10px">
  Generated {{ date }} — educational use only, not investment advice.
</div>
</body></html>
"""

def write_html(outdir: str, facts: dict, alerts: List[str], targets: pd.DataFrame, nav_series: Optional[pd.Series]) -> str:
    os.makedirs(outdir, exist_ok=True)
    top_targets = targets.sort_values("target_w", ascending=False).head(10).to_dict(orient="records")
    spark_b64 = _sparkline_png_b64(nav_series) if nav_series is not None else None
    html = Template(HTML_TEMPLATE).render(
        date=str(_today()),
        facts=facts,
        alerts=alerts,
        top_targets=top_targets,
        spark=spark_b64
    )
    fn = os.path.join(outdir, f"day18_report_{_today()}.html")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(html)
    return fn

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Day 18 — Alerts + Mini Dashboard")
    p.add_argument("--positions", type=str, default="../day17-monitor/positions_weights.csv")
    p.add_argument("--orders_dir", type=str, default="../day16-monitoring/orders")
    p.add_argument("--metrics_dir", type=str, default="../day16-monitoring/outputs", help="where day16_metrics_*.csv may live")
    p.add_argument("--outdir", type=str, default="./reports")
    p.add_argument("--w_cap", type=float, default=0.30)
    p.add_argument("--turnover_cap", type=float, default=0.25)
    p.add_argument("--sum_tol", type=float, default=0.02)
    p.add_argument("--stale_days", type=int, default=2)
    # alerts
    p.add_argument("--slack_webhook", type=str, default=os.getenv("SLACK_WEBHOOK_URL"))
    p.add_argument("--email_to", type=str, default=os.getenv("EMAIL_TO"))
    p.add_argument("--smtp_host", type=str, default=os.getenv("SMTP_HOST"))
    p.add_argument("--smtp_port", type=int, default=int(os.getenv("SMTP_PORT", "587")))
    p.add_argument("--smtp_user", type=str, default=os.getenv("SMTP_USER"))
    p.add_argument("--smtp_pass", type=str, default=os.getenv("SMTP_PASS"))
    return p.parse_args()

def main():
    a = parse_args()

    if not os.path.exists(a.positions):
        sys.exit(f"ERROR: positions file not found: {a.positions}")
    latest_orders = _latest_file(os.path.join(a.orders_dir, "orders_*.csv"))
    if not latest_orders:
        sys.exit(f"ERROR: no orders files found in {a.orders_dir}")

    pos = _read_positions_weights(a.positions)
    ords = _read_orders_any(latest_orders)

    alerts, facts = run_checks(
        pos, ords,
        w_cap=a.w_cap,
        turnover_cap=a.turnover_cap,
        sum_tol=a.sum_tol,
        stale_days=a.stale_days,
        positions_mtime=os.path.getmtime(a.positions),
        orders_mtime=os.path.getmtime(latest_orders),
    )

    nav = _load_nav_series(a.metrics_dir)
    html_path = write_html(a.outdir, facts, alerts, ords, nav)
    print(f"Report: {html_path}")

    # Alerts
    if alerts:
        msg = "*Day 18 Alerts*\n" + "\n".join(f"- {x}" for x in alerts)
        if a.slack_webhook:
            send_slack(a.slack_webhook, msg)
        if a.email_to and a.smtp_host and a.smtp_user and a.smtp_pass:
            try:
                send_email_smtp(a.smtp_host, a.smtp_port, a.smtp_user, a.smtp_pass,
                                a.email_to, f"Day 18 Alerts — {_today()}", f"<pre>{msg}</pre>")
            except Exception as e:
                print(f"[WARN] Email send failed: {e}")
        # non-zero exit so CI/Task Scheduler can flag
        sys.exit(2)
    else:
        print("No alerts. ✅")
        sys.exit(0)

if __name__ == "__main__":
    main()
