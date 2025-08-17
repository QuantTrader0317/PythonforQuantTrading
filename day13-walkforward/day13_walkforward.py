"""
Day 13 — Walk-Forward Validation (WFV)
Stack: Allocator (IVOL/optional ERC) -> Vol Target -> Throttled execution + costs
We grid-search on the TRAIN window using: score = Sharpe - penalty * annualized_turnover,
pick best params, then apply out-of-sample on the TEST window. Repeat and stitch returns.
"""

from __future__ import annotations
import argparse, os, sys
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Optional (only if you use --mode erc)
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

TRADING_DAYS = 252

# ---------------- Basics ----------------
def last_trading_reb_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return LAST trading day of each month/quarter present in `index`."""
    grp = index.to_period("Q" if freq.upper().startswith("Q") else "M")
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby(grp).max().values)

def fetch_prices(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.sort_index().dropna(how="all").ffill()

def r_daily(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()

# ---------------- Metrics ----------------
def ann_vol(x: pd.Series, periods:int=TRADING_DAYS) -> float:
    x=x.dropna(); return float(x.std()*np.sqrt(periods)) if len(x) else np.nan

def cagr(x: pd.Series, periods:int=TRADING_DAYS) -> float:
    x=x.dropna();
    if not len(x): return np.nan
    g=float((1+x).prod()); yrs=len(x)/periods
    return g**(1/yrs)-1 if yrs>0 else np.nan

def sharpe(x: pd.Series, rf_annual:float=0.0) -> float:
    v=ann_vol(x);
    if not np.isfinite(v) or v==0: return np.nan
    return (cagr(x)-rf_annual)/v

def max_dd(eq: pd.Series) -> float:
    if eq.empty: return np.nan
    return float((eq/eq.cummax()-1).min())

def hit(x: pd.Series) -> float:
    x=x.dropna(); return float((x>0).mean()) if len(x) else np.nan

# ---------------- Allocators ----------------
def normalize_w(w: pd.Series, cap: Optional[float]) -> pd.Series:
    w=w.clip(lower=0.0)
    if cap is not None and cap>0: w=w.clip(upper=cap)
    s=w.sum(); return w/s if s>0 else w

def erc_weights(cov: np.ndarray, tol:float=1e-8, it:int=10000) -> np.ndarray:
    n=cov.shape[0]; w=np.ones(n)/n; step=0.01
    for _ in range(it):
        m=cov@w; rc=w*m; tgt=rc.mean()
        grad = m - (tgt/np.maximum(w,1e-12))
        w=np.clip(w-step*grad,0.0,None); s=w.sum()
        if s>0: w/=s
        if np.linalg.norm(rc-tgt)<tol: break
    return w

def rolling_vol(r: pd.DataFrame, window:int, method:str="rolling")->pd.DataFrame:
    if method=="ewma": return r.ewm(span=window).std()*np.sqrt(TRADING_DAYS)
    return r.rolling(window).std()*np.sqrt(TRADING_DAYS)

def w_ivol(r: pd.DataFrame, window:int, reb:str, vol_method:str="rolling",
           vol_floor:float=0.0, cap:Optional[float]=None, band:float=0.0)->pd.DataFrame:
    rebs=last_trading_reb_dates(r.index, reb)
    w_t=pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    vol=rolling_vol(r, window, vol_method); prev=None
    for dt in rebs:
        if dt not in vol.index: continue
        v=vol.loc[dt].replace(0,np.nan)
        v=(v.fillna(v.median()).clip(lower=vol_floor) if vol_floor>0 else v.dropna())
        if v.empty: continue
        inv=(1.0/v).replace([np.inf,-np.inf],np.nan).dropna()
        if inv.empty: continue
        w=normalize_w(inv, cap)
        if prev is not None and band>0:
            drift=float((w.reindex(prev.index).fillna(0)-prev.reindex(w.index).fillna(0)).abs().sum())
            if drift<band: w=prev.copy()
        w_t.loc[dt, w.index]=w.values; prev=w.copy()
    return w_t.ffill().fillna(0.0)

def w_erc(r: pd.DataFrame, window:int, reb:str,
          cap:Optional[float]=None, band:float=0.0, min_obs:int=40)->pd.DataFrame:
    if not _HAS_SK: raise ImportError("ERC requires scikit-learn")
    rebs=last_trading_reb_dates(r.index, reb)
    w_t=pd.DataFrame(np.nan, index=r.index, columns=r.columns); prev=None
    for dt in rebs:
        rw=r.loc[:dt].tail(window)
        if rw.shape[0]<min_obs: continue
        cov=LedoitWolf().fit(rw.values).covariance_
        w=pd.Series(erc_weights(cov), index=rw.columns)
        w=normalize_w(w, cap)
        if prev is not None and band>0:
            drift=float((w.reindex(prev.index).fillna(0)-prev.reindex(w.index).fillna(0)).abs().sum())
            if drift<band: w=prev.copy()
        w_t.loc[dt, w.index]=w.values; prev=w.copy()
    return w_t.ffill().fillna(0.0)

# ---------------- Vol targeting ----------------
def vt_scaler_from_ret(ret_pre: pd.Series, target:float=0.10, span:int=63,
                       s_min:float=0.5, s_max:float=2.0, forecast:bool=True)->pd.Series:
    sigma=(ret_pre.pow(2).ewm(span=span, adjust=False).mean().shift(1 if forecast else 0).pow(0.5)
           *np.sqrt(TRADING_DAYS))
    s=(target/sigma).clip(lower=s_min, upper=s_max).fillna(1.0)
    return s.rename("scaler")

# ---------------- Costs & execution ----------------
def adv_cost(turnover: pd.Series, spread_bps:float, vol_proxy:pd.Series, impact_coef:float=0.10)->pd.Series:
    spread = turnover*(spread_bps/10000.0)
    impact = impact_coef*vol_proxy*np.sqrt(turnover.clip(lower=0.0))
    return (spread.fillna(0.0)+impact.fillna(0.0))

def throttled_exec(r: pd.DataFrame, w_targets_reb: pd.DataFrame, dates: pd.DatetimeIndex,
                   exec_speed:float=0.33, band:float=0.002, cap:float=0.20,
                   spread_bps:float=2.0, impact_coef:float=0.10, impact_span:int=63,
                   rf_annual:float=0.02)->Tuple[pd.Series,pd.Series]:
    cols=w_targets_reb.columns; w_curr=pd.Series(0.0, index=cols)
    w_exec=pd.DataFrame(0.0, index=dates, columns=cols)
    turn=pd.Series(0.0, index=dates)
    vol_proxy=r.ewm(span=impact_span).std().mean(axis=1).fillna(0.0)
    for d in dates:
        if d in w_targets_reb.index:
            tgt=w_targets_reb.loc[d].fillna(0.0)
            gap=tgt-w_curr; gap=gap.where(gap.abs()>=band,0.0)
            dv=exec_speed*gap
            tr=float(dv.abs().sum())
            if cap is not None and tr>cap and tr>0:
                dv*= (cap/tr); tr=cap
            w_curr=(w_curr+dv).clip(lower=0.0); s=w_curr.sum()
            if s>0: w_curr/=s
            turn.loc[d]=tr
        w_exec.loc[d]=w_curr.values
    costs=adv_cost(turn, spread_bps, vol_proxy, impact_coef)
    rf_d=rf_annual/TRADING_DAYS
    ret=(w_exec*r).sum(axis=1) - costs - rf_d
    return ret.rename("ret"), turn.rename("turnover")

# ---------------- Single run ----------------
def run_once(px: pd.DataFrame, start:str, end:str, mode:str, window:int, reb:str,
             vt_target:float, vt_span:int, exec_speed:float, nt_band:float, t_cap:float,
             spread_bps:float, impact_coef:float, impact_span:int, rf:float,
             vol_method:str="rolling", vol_floor:float=0.0, w_cap:Optional[float]=0.25,
             alloc_band:float=0.05) -> Dict[str,object]:
    r_full=r_daily(px)
    r=r_full.loc[start:end]
    if r.empty: return {"ret":pd.Series(dtype=float),"turnover":pd.Series(dtype=float)}

    if mode=="ivol":
        w_alloc=w_ivol(r, window, reb, vol_method, vol_floor, w_cap, alloc_band)
    elif mode=="erc":
        w_alloc=w_erc(r, window, reb, w_cap, alloc_band)
    else:
        raise ValueError("mode must be 'ivol' or 'erc'")

    ret_pre=(w_alloc*r).sum(axis=1)
    s=vt_scaler_from_ret(ret_pre, target=vt_target, span=vt_span, forecast=True)
    w_vt_daily = w_alloc.mul(s, axis=0)

    rebs=last_trading_reb_dates(r.index, reb)
    w_vt_reb = w_vt_daily.loc[rebs].dropna(how="all")

    ret, turnover = throttled_exec(r, w_vt_reb, r.index,
                                   exec_speed, nt_band, t_cap,
                                   spread_bps, impact_coef, impact_span, rf)
    return {"ret":ret, "turnover":turnover}

# ---------------- Objective ----------------
def objective(ret: pd.Series, turnover: pd.Series, rf:float, penalty:float) -> float:
    sh=sharpe(ret, rf)
    ann_turn = float(turnover.mean()*TRADING_DAYS)
    if not np.isfinite(sh): return -np.inf
    return sh - penalty*ann_turn

# ---------------- Walk-forward ----------------
def walk_forward(px: pd.DataFrame, start:str, mode:str, reb:str,
                 train_years:int, test_months:int, embargo_days:int,
                 windows:List[int], vt_targets:List[float],
                 speeds:List[float], bands:List[float], caps:List[float],
                 spread_bps:float, impact_coef:float, impact_span:int,
                 rf:float, vt_span:int, penalty:float,
                 vol_method:str="rolling") -> Dict[str,object]:

    r_index = r_daily(px).index
    if start is None: start=r_index[0].strftime("%Y-%m-%d")
    t0 = pd.Timestamp(start)

    out_returns=[]; rows=[]; train_scores=[]; fold=0
    while True:
        tr_start = t0
        tr_end   = tr_start + pd.DateOffset(years=train_years)
        emb_end  = tr_end + pd.Timedelta(days=embargo_days)
        te_end   = emb_end + pd.DateOffset(months=test_months)
        if te_end > r_index[-1]: break

        best=None; best_row=None
        for w in windows:
            for vt in vt_targets:
                for sp in speeds:
                    for bd in bands:
                        for cap in caps:
                            run = run_once(px, tr_start.strftime("%Y-%m-%d"),
                                           tr_end.strftime("%Y-%m-%d"),
                                           mode, w, reb, vt, vt_span,
                                           sp, bd, cap, spread_bps, impact_coef, impact_span, rf,
                                           vol_method)
                            score = objective(run["ret"], run["turnover"], rf, penalty)
                            train_scores.append({
                                "train_start":tr_start, "train_end":tr_end,
                                "window":w, "vt_target":vt, "exec_speed":sp,
                                "no_trade_band":bd, "turnover_cap":cap,
                                "score":score, "Sharpe":sharpe(run["ret"],rf),
                                "AnnTurn":float(run["turnover"].mean()*TRADING_DAYS)
                            })
                            if (best is None) or (score>best):
                                best=score; best_row=(w, vt, sp, bd, cap)

        hist_start = tr_start.strftime("%Y-%m-%d")
        test_start = emb_end.strftime("%Y-%m-%d")
        test_end   = te_end.strftime("%Y-%m-%d")
        run_hist = run_once(px, hist_start, test_end, mode, best_row[0], reb,
                            best_row[1], vt_span, best_row[2], best_row[3], best_row[4],
                            spread_bps, impact_coef, impact_span, rf, vol_method)
        ret_test = run_hist["ret"].loc[test_start:test_end]
        out_returns.append(ret_test)

        rows.append({
            "train_start":tr_start, "train_end":tr_end,
            "test_start":pd.Timestamp(test_start), "test_end":pd.Timestamp(test_end),
            "window":best_row[0], "vt_target":best_row[1],
            "exec_speed":best_row[2], "no_trade_band":best_row[3],
            "turnover_cap":best_row[4]
        })

        fold += 1
        print(f"[WFV] Completed fold {fold}: train {tr_start.date()}→{tr_end.date()} | test {test_start}→{test_end} | best {best_row}", flush=True)
        t0 = te_end

    wf_ret = pd.concat(out_returns).sort_index()
    return {"ret":wf_ret, "choices":pd.DataFrame(rows), "train_scores":pd.DataFrame(train_scores)}

# ---------------- CLI ----------------
def parse_list(s: str, typ=float) -> List:
    return [typ(x) for x in s.split(",") if x.strip()]

def parse_args():
    p = argparse.ArgumentParser(description="Day 13 — Walk-Forward Validation")
    p.add_argument("--tickers", type=str, default="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,XLC,TLT,GLD")
    p.add_argument("--start", type=str, default="2008-01-01")
    p.add_argument("--reb", type=str, default="M", choices=["M","Q"])
    p.add_argument("--mode", type=str, default="ivol", choices=["ivol","erc"])  # default ivol to avoid sklearn
    p.add_argument("--train_years", type=int, default=5)
    p.add_argument("--test_months", type=int, default=6)
    p.add_argument("--embargo_days", type=int, default=21)
    p.add_argument("--windows", type=str, default="63,126")
    p.add_argument("--vt_targets", type=str, default="0.08,0.10,0.12")
    p.add_argument("--speeds", type=str, default="0.25,0.33,0.50")
    p.add_argument("--bands", type=str, default="0.001,0.002,0.004")
    p.add_argument("--caps", type=str, default="0.10,0.20")
    p.add_argument("--vt_span", type=int, default=63)
    p.add_argument("--spread_bps", type=float, default=2.0)
    p.add_argument("--impact_coef", type=float, default=0.10)
    p.add_argument("--impact_span", type=int, default=63)
    p.add_argument("--rf", type=float, default=0.02)
    p.add_argument("--penalty", type=float, default=0.5, help="penalty * annualized_turnover")
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode=="erc" and not _HAS_SK:
        sys.exit("ERC mode requires scikit-learn. Install with: pip install scikit-learn")

    tickers=[t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    px=fetch_prices(tickers, a.start)

    res = walk_forward(px, a.start, a.mode, a.reb, a.train_years, a.test_months,
                       a.embargo_days, parse_list(a.windows,int), parse_list(a.vt_targets,float),
                       parse_list(a.speeds,float), parse_list(a.bands,float),
                       parse_list(a.caps,float), a.spread_bps, a.impact_coef, a.impact_span,
                       a.rf, a.vt_span, a.penalty)

    # Save artifacts
    out_csv=os.path.join(a.outdir,"outputs"); os.makedirs(out_csv, exist_ok=True)
    out_ch =os.path.join(a.outdir,"charts");  os.makedirs(out_ch,  exist_ok=True)

    res["choices"].to_csv(os.path.join(out_csv,"day13_wf_params.csv"), index=False)
    res["train_scores"].to_csv(os.path.join(out_csv,"day13_wf_train_scores.csv"), index=False)
    res["ret"].rename("ret").to_csv(os.path.join(out_csv,"day13_wf_returns.csv"))

    ret=res["ret"]; eq=(1+ret).cumprod()
    mt = pd.DataFrame({
        "CAGR":[cagr(ret)], "Vol":[ann_vol(ret)], "Sharpe":[sharpe(ret,a.rf)],
        "MaxDD":[max_dd(eq)], "Hit%":[hit(ret)]
    }).T.round(4)
    mt.to_csv(os.path.join(out_csv,"day13_wf_metrics.csv"))

    plt.figure(figsize=(12,6))
    eq.plot(); plt.title("Day 13 — Walk-Forward Equity (net)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_ch,"day13_wf_equity.png"), dpi=150); plt.close()

    ch = res["choices"].copy(); ch["t"]=pd.to_datetime(ch["test_start"])
    plt.figure(figsize=(12,6))
    for col in ["window","vt_target","exec_speed","no_trade_band","turnover_cap"]:
        plt.step(ch["t"], ch[col], where="post", label=col)
    plt.legend(); plt.title("Day 13 — Chosen Params per Test Window")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_ch,"day13_wf_param_choices.png"), dpi=150); plt.close()

    print("\n=== Day 13 — Walk-Forward (net) ===")
    print(mt.to_string(header=False))
    print("\nArtifacts:")
    for f in ["day13_wf_params.csv","day13_wf_train_scores.csv","day13_wf_returns.csv","day13_wf_metrics.csv"]:
        print(" -", os.path.join(out_csv,f))
    for f in ["day13_wf_equity.png","day13_wf_param_choices.png"]:
        print(" -", os.path.join(out_ch,f))

if __name__=="__main__":
    main()