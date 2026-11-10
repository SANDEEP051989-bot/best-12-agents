#!/usr/bin/env python3
"""compare_all.py - v2 vs v7.0 vs v7.1 on ~2 years of hourly data (BTC, Gold, US Oil).
Code rules only (no AI, no votes, no room rule) - same test for all three. Result goes to Telegram.
Needs agents_v7.py in the same folder."""
import os, sys, math
import numpy as np, pandas as pd, requests
import agents_v7 as v7

TOK = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_TOKEN") or ""
CHAT = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID") or ""

def send(t):
    print(t)
    if TOK and CHAT:
        for i in range(0, len(t), 3800):
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage", data={"chat_id": CHAT, "text": t[i:i+3800]}, timeout=30)

def daily_view_per_bar(h, mkt, f):
    m = v7.MARKETS[mkt]; slow, fast, wk = m["daily"]
    daily = v7.daily_from_hourly(h, mkt)
    out = {}
    for D in daily.index:
        dfin = daily[daily.index < D]
        if len(dfin) < slow + 3:
            out[D] = "NEUTRAL"; continue
        wkk = v7.week_keys(dfin.index)
        wc = dfin["Close"].groupby(wkk).last()
        cw = v7.week_keys(pd.DatetimeIndex([D]))[0]
        out[D] = v7.daily_view_state(dfin["Close"], wc[wc.index < cw], slow, fast, wk)[0]
    return pd.Series([out.get(k, "NEUTRAL") for k in f["dkey"]], index=f.index)

def v2_side(h, f):
    d = h["Close"].resample("1D").last().dropna()
    e50, e200 = v7.ema(d, 50), v7.ema(d, 200)
    w = d.resample("W").last()
    w4 = (w > w.shift(4)).reindex(d.index, method="ffill")
    sc = np.sign(d - e50) + np.sign(d - e200) + np.sign(e50 - e200) + np.where(w4, 1, -1)
    sc = pd.Series(sc, index=d.index).shift(1)            # yesterday's finished day only
    s = sc.reindex(f.index.normalize()).values
    side = np.where(s >= 2, "BUY", np.where(s <= -2, "SELL", None))
    r, hist = f["rsi"], f["hist"]
    mom_s = (r < 50) & (r < r.shift(6)) & (hist < hist.shift(3))
    mom_b = (r > 50) & (r > r.shift(6)) & (hist > hist.shift(3))
    return side, mom_b.values, mom_s.values

def run_sys(name, mkt, f, dv, v2):
    O, H, L, C = (f[k].values for k in ("o", "h", "l", "c"))
    A, slo, shi = f["atr"].values, f["stop_lo"].values, f["stop_hi"].values
    tr, t4 = f["trend"].values, f["trend4"].values
    fb = v7.MARKETS[mkt]["fill_bars"]; n = len(f); i = 300; rs = []
    while i < n - 1:
        d = None
        if name == "v2":
            s = v2[0][i]
            if s == "BUY" and v2[1][i]: d = "BUY"
            if s == "SELL" and v2[2][i]: d = "SELL"
            if d and v7.finite(A[i]):
                dist = 1.5 * A[i]; sg = 1 if d == "BUY" else -1
                cost = v7.MARKETS[mkt]["cost_pct"] / 100 * C[i] / dist
                out = v7.sim_fixed(O, H, L, C, i + fb, d, C[i] - sg * dist, C[i] + sg * 2.0 * A[i], dist, cost, 48)
            else: out = None
        else:
            d = {"UP": "BUY", "DOWN": "SELL"}.get(tr[i])
            if d:
                gate = dv.iloc[i] if name == "v7.0" else {"UP": "UP", "DOWN": "DOWN"}.get(t4[i], "NEUTRAL")
                if (d == "BUY" and gate == "DOWN") or (d == "SELL" and gate == "UP"): d = None
            out = None
            if d:
                v7.CFG["STOP_MAX_ATR"] = 2.0 if name == "v7.0" else 3.0
                p = v7.plan_at(d, C[i], A[i], slo[i], shi[i], mkt)
                if p["valid"] and p["cost_r"] <= v7.CFG["MAX_COST_R"]:
                    out = v7.sim_fixed(O, H, L, C, i + fb, d, p["stop"], p["t"][1], p["dist"], p["cost_r"])
        if out and not out.get("skip"):
            rs.append(out["r"]); i += fb + out["bars"] + 1
        else:
            i += 1
    if not rs: return f"{name}: 0 trades"
    rs = np.array(rs); win = (rs > 0).mean() * 100
    return f"{name}: {len(rs)} trades, won {win:.0f}%, average {rs.mean():+.2f}R, total {rs.sum():+.1f}R"

def main():
    now = v7.now_utc()
    lines = ["COMPARE v2 vs v7.0 vs v7.1 (about 2 years, hourly, code rules only, fees included)", ""]
    for mkt in v7.MARKETS:
        try:
            h = v7.closed_hourly(v7.get_hourly(mkt), now, v7.MARKETS[mkt]["data_delay_min"])
            if mkt == "OIL": h, _ = v7.wti_back_adjust(h)
            f = v7.build_features(h, mkt)
            dv = daily_view_per_bar(h, mkt, f); v2 = v2_side(h, f)
            lines.append(f"== {v7.MARKETS[mkt]['name']} ==")
            for s in ("v2", "v7.0", "v7.1"):
                lines.append(run_sys(s, mkt, f, dv, v2))
            lines.append("")
        except Exception as e:
            lines.append(f"{mkt}: error {type(e).__name__}: {e}")
    lines.append("Note: AI parts, votes and room rule are not in this test. Best = higher total R with enough trades.")
    send("\n".join(lines))

if __name__ == "__main__":
    main()
