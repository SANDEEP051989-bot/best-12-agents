#!/usr/bin/env python3
"""renko_test.py - tests "Ankit Renko Mahakaal" rules on BTC, Gold, US Oil (15m, 1h, 4h).
Same logic as the TradingView indicator: ATR bricks, Supertrend on the brick price, Renko RSI,
4-hour trend filter. Entry at the next candle's open when the trend flips (and filters agree).
Stop = the trend line at entry. Exit = trend flips back, or the stop is hit. Fees included.
Shows each version: no filter, + RSI, + RSI + 4h filter."""
import os, numpy as np, pandas as pd, requests, yfinance as yf

MKTS = {"Bitcoin": ("BTC-USD", 0.12), "Gold": ("GC=F", 0.05), "US Oil": ("CL=F", 0.10)}
CAPITAL = 20000; RISK_RS = CAPITAL * 0.005      # Rs 100 risk per trade (0.5%)
BRICKS = [0.5, 0.75, 1.0, 1.5]                  # brick sizes tested (x ATR)
TOK = os.getenv("TG_TOKEN", ""); CHAT = os.getenv("TG_CHAT", "")

def send(t):
    print(t)
    if TOK and CHAT:
        for i in range(0, len(t), 3800):
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage", data={"chat_id": CHAT, "text": t[i:i+3800]}, timeout=30)

def load(tk, tf):
    df = yf.Ticker(tk).history(period="60d" if tf == "15m" else "729d", interval="15m" if tf == "15m" else "1h")
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_convert("UTC").tz_localize(None)
    return df

def resample(df, rule):
    g = df.resample(rule, origin="epoch")
    return pd.DataFrame({"Open": g.Open.first(), "High": g.High.max(), "Low": g.Low.min(), "Close": g.Close.last()}).dropna()

def renko(df, aLen=14, bX=1.0, sX=2.0, rLen=14):
    C = df.Close.values
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    A = tr.rolling(aLen).mean().values          # Pine ta.atr = RMA; small difference
    A = tr.ewm(alpha=1/aLen, adjust=False).mean().values
    n = len(C); trend = np.zeros(n, int); line = np.full(n, np.nan); rsi = np.full(n, 50.0)
    rc = bsz = gain = loss = upL = dnL = None; rdir = 0; t = 1; rv = 50.0
    for i in range(n):
        a = A[i]
        if rc is None:
            if i >= aLen: rc, bsz = C[i], a * bX
            trend[i] = t; continue
        c = C[i]; nUp = nDn = 0
        if rdir >= 0 and c >= rc + bsz: nUp = int((c - rc) // bsz)
        elif rdir <= 0 and c <= rc - bsz: nDn = int((rc - c) // bsz)
        elif rdir == 1 and c <= rc - 2 * bsz: nDn = int((rc - c) // bsz)
        elif rdir == -1 and c >= rc + 2 * bsz: nUp = int((c - rc) // bsz)
        for _ in range(nUp):
            gain = bsz if gain is None else (gain * (rLen - 1) + bsz) / rLen
            loss = 0.0 if loss is None else loss * (rLen - 1) / rLen
        for _ in range(nDn):
            gain = 0.0 if gain is None else gain * (rLen - 1) / rLen
            loss = bsz if loss is None else (loss * (rLen - 1) + bsz) / rLen
        if nUp: rc += nUp * bsz; rdir = 1; bsz = a * bX
        if nDn: rc -= nDn * bsz; rdir = -1; bsz = a * bX
        if gain is not None and loss is not None:
            rv = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
        lo, hi = rc - sX * bsz, rc + sX * bsz
        pu, pd_ = upL, dnL
        upL = lo if pu is None else max(lo, pu)
        dnL = hi if pd_ is None else min(hi, pd_)
        if t == -1 and pd_ is not None and rc > pd_: t = 1; upL = lo
        elif t == 1 and pu is not None and rc < pu: t = -1; dnL = hi
        trend[i] = t; line[i] = upL if t == 1 else dnL; rsi[i] = rv
    return trend, line, rsi

def htf_trend(df, base_index):
    h = resample(df, "4h")
    t, _, _ = renko(h)
    s = pd.Series(t, index=h.index + pd.Timedelta(hours=4))      # known only after the 4h candle closes
    return s.reindex(base_index, method="ffill").fillna(0).astype(int).values

def backtest(df, fee, use_rsi, use_htf, t4, bX=1.0):
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    tr, line, rsi = renko(df, bX=bX)
    n = len(C); rs = []; i = 30
    while i < n - 2:
        up = tr[i] == 1 and tr[i-1] == -1; dn = tr[i] == -1 and tr[i-1] == 1
        side = None
        if up and (not use_rsi or rsi[i] >= 50) and (not use_htf or t4[i] == 1): side = 1
        if dn and (not use_rsi or rsi[i] <= 50) and (not use_htf or t4[i] == -1): side = -1
        if side is None or np.isnan(line[i]): i += 1; continue
        entry = O[i+1]; stop = line[i]; dist = (entry - stop) * side
        if dist <= 0: i += 1; continue
        cost = fee / 100 * entry / dist; r = None
        for j in range(i + 1, n):
            if (L[j] <= stop) if side == 1 else (H[j] >= stop):
                ex = min(O[j], stop) if side == 1 else max(O[j], stop); r = (ex - entry) * side / dist - cost; break
            if tr[j] == -side and j + 1 < n:
                r = (O[j+1] - entry) * side / dist - cost; break
        if r is None: break
        rs.append(r); i = j + 1
    return rs

def fmt(rs):
    if not rs: return "0 trades"
    rs = np.array(rs)
    return (f"{len(rs)} trades, won {(rs > 0).mean()*100:.0f}%, total {rs.sum():+.1f}R = "
            f"Rs {rs.sum() * RISK_RS:+,.0f}")

def main():
    out = ["RENKO MAHAKAAL TEST v2 (fees included)",
           f"Money: Rs {CAPITAL:,} capital, Rs {RISK_RS:,.0f} risk per trade.", ""]
    best = None
    for name, (tk, fee) in MKTS.items():
        out.append(f"== {name} ==")
        for tf in ("15m", "1h", "4h"):
            try:
                raw = load(tk, tf)
                df = resample(raw, "4h") if tf == "4h" else raw
                t4 = htf_trend(raw, df.index)
                days = (df.index[-1] - df.index[0]).days
                out.append(f"{tf} ({days} days):")
                for b in BRICKS:
                    vers = [("trend", False, False), ("+RSI", True, False)]
                    if tf != "4h":
                        vers.append(("+RSI+4h", True, True))
                    for label, ur, uh in vers:
                        rs = backtest(df, fee, ur, uh, t4, b)
                        out.append(f"  brick {b} {label}: {fmt(rs)}")
                        if len(rs) >= 30 and (best is None or sum(rs) > best[0]):
                            best = (sum(rs), f"{name} {tf} brick {b} {label}", len(rs))
            except Exception as e:
                out.append(f"{tf}: error {type(e).__name__}: {e}")
        out.append("")
    if best:
        out.append(f"BEST (30+ trades): {best[1]} - {best[2]} trades, {best[0]:+.1f}R = Rs {best[0] * RISK_RS:+,.0f}")
    else:
        out.append("BEST: no setting had 30+ trades.")
    out.append("15m has only ~60 days of data, so its numbers are weak.")
    send("\n".join(out))


if __name__ == "__main__":
    main()
