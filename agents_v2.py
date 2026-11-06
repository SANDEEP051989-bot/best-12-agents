# agents_v2.py  -  version 6  "Veteran Desk"
# 12 AI agents, each thinking like a 30-year veteran trader, for BTC, Gold and US Oil.
# Brain: Gemini.  Reports: Telegram.  Server: GitHub Actions.
#
# New in version 4 (fixes found in the first live reports):
#  1. ONE market-phase definition (trend up / down / mixed) shared by every agent.
#  2. Support/resistance: tiny levels removed; the code measures room UP and room DOWN in ATR.
#  3. Time machine votes when a proven rule has a live signal (not only a brand-new one).
#  4. Time machine proves rules on older data and checks them on newer data (no fooling itself).
#  5. Agents that Gemini skipped are asked again; if still no answer a simple rule backup votes.
#  6. (yml) report card is saved safely: pull before push.
#  7. Momentum rules are equal for BUY and SELL.
#  8. Bear prompt rewritten (no copy-able sentence).
#  9. Bull checks room UP, Bear checks room DOWN.
# 10. Every agent follows a 5-step veteran way of thinking and says what would prove it wrong.
# 11. More news sources for Gold.
# Plus: a quality check that flags (and overrules) votes that contradict the data.
#
# New in version 5 (AI quality inspector made safer):
#  1. The inspector sees each member's OWN data (dollar, VIX, news, futures...), not only the chart.
#  2. A trade may rest on at most 1 vote that the inspector changed.
#  3. The inspector itself is on the report card (were its changes right?).
#  4. The inspector may change News and Mood only with clear evidence (numbers or a headline).
#  5. The inspector must use the exact short member names, so no change is silently lost.
#  6. The inspector only runs when at least 5 members already agree on one side (saves Gemini questions).
#
# New in version 6 (from the 16 live screenshots of 6 Nov):
#  1. CODE RULES decide the 5 chart members (trend, momentum, levels, volatility, big picture):
#     a vote against the rule becomes WAIT, a WAIT where the rule is clear becomes the rule's vote.
#  2. The AI inspector now only reviews the 6 judgement members (news, mood, macro, specialist, bull, bear)
#     and may only REMOVE one bad vote (make it WAIT) with proof from that member's own data. It can no longer
#     cancel good votes or create new ones, so it cannot block a good trade or contradict itself.
#  3. Faster market phase: price vs EMA200 and a run of falling/rising weekly closes now count too.
#  4. Bitcoin specialist: price down + open interest up (new shorts) can no longer give a BUY.
#  5. Weak levels (a single hourly swing) no longer block a trade; only strong levels count for room.
#  6. Time machine: one extra rule and fairer test sizes, so it votes more often.
#  7. More oil news sources.
#
# Paper trading only. Not financial advice.

import os, json, time, datetime as dt
import requests
import numpy as np
import pandas as pd
import yfinance as yf

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

MARKETS = {"BTC": "BTC-USD", "GOLD": "GC=F", "OIL": "CL=F"}
NEWS_TICKERS = {"BTC": ["BTC-USD", "IBIT", "MSTR"], "GOLD": ["GC=F", "GLD", "IAU", "NEM"], "OIL": ["CL=F", "BZ=F", "USO", "XLE", "XOM"]}
ROUND_STEP = {"BTC": 5000, "GOLD": 100, "OIL": 5}     # only big round numbers count as levels

# ----- trade rule (change these numbers if you want more or fewer trades) -----
MIN_SIDE = 6           # at least this many agents on one side
MAX_OPPOSITE = 1       # at most this many agents on the other side
MIN_CONF = 6.0         # average confidence of the agreeing agents (1-10)

# ----- trade plan -----
STOP_ATR = 1.5         # stop-loss = 1.5 x hourly ATR
TARGET1_ATR = 2.0      # target 1 = 2 x hourly ATR
TARGET2_ATR = 3.5      # target 2 = 3.5 x hourly ATR
MAX_HOLD_BARS = 48     # a trade or a vote is judged within 48 hourly candles
WIN_R = TARGET1_ATR / STOP_ATR
MIN_LEVEL_GAP_ATR = 0.5   # levels closer than this to price are "at the level", not room-blockers

# ----- safety -----
MAX_DAILY_LOSS_R = 3.0          # stop new trades after losing 3 x risk in one day
MAX_RUN_SECONDS = 12 * 60       # stop asking Gemini after 12 minutes
PAUSE_BETWEEN_CALLS = 2

STATE_FILE = "state.json"
START = time.time()
STATS = {"asked": 0, "answered": 0, "busy": 0, "errors": []}
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


# =====================================================================
# small helpers
# =====================================================================
def log(msg):
    print(f"[{int(time.time() - START)}s] {msg}", flush=True)


def time_left():
    return time.time() - START < MAX_RUN_SECONDS


def num(x, d=2):
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), d)
    except Exception:
        return None


def pct(a, b):
    try:
        if b in (0, None) or pd.isna(a) or pd.isna(b):
            return None
        return round((float(a) / float(b) - 1) * 100, 2)
    except Exception:
        return None


def jdump(obj):
    return json.dumps(obj, default=str, ensure_ascii=False)


def load_state():
    s = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
        except Exception:
            s = {}
    for k, v in {"votes_log": [], "trades": [], "weights": {}, "memory": {}, "last_oi": {}}.items():
        s.setdefault(k, v)
    s.pop("signals", None)
    return s


def save_state(s):
    s["votes_log"] = s["votes_log"][-500:]
    s["trades"] = s["trades"][-300:]
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1, default=str)


# =====================================================================
# market data
# =====================================================================
def to_utc(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def get_data(ticker, period="60d", interval="1h"):
    for attempt in range(2):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df):
                return to_utc(df)
        except Exception as e:
            log(f"download {ticker} failed: {e}")
        time.sleep(3)
    return pd.DataFrame()


def last_change(ticker, days=5):
    df = get_data(ticker, "3mo", "1d")
    if len(df) <= days:
        return None, None
    c = df.Close
    return num(c.iloc[-1]), pct(c.iloc[-1], c.iloc[-1 - days])


def get_news(tickers, n=8):
    out, seen = [], set()
    for t in tickers:
        try:
            for it in (yf.Ticker(t).news or [])[:n]:
                title = it.get("title") or (it.get("content") or {}).get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    out.append(title)
        except Exception:
            pass
    return out[:14]


def get_macro():
    out = {}
    for name, t in {"dollar_index": "DX-Y.NYB", "us10y_yield": "^TNX",
                    "sp500": "^GSPC", "vix_fear_index": "^VIX"}.items():
        last, ch = last_change(t, 5)
        if last is not None:
            out[name] = {"last": last, "change_5d_pct": ch}
    return out


def get_btc_derivatives(state):
    out = {"source": None}
    try:
        f = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=10)
        o = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT", timeout=10)
        if f.status_code == 200 and o.status_code == 200:
            out.update(source="binance", funding_rate_pct_8h=num(float(f.json()["lastFundingRate"]) * 100, 4),
                       open_interest_btc=num(o.json()["openInterest"], 0))
    except Exception:
        pass
    if not out["source"]:
        try:
            f = requests.get("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP", timeout=10)
            o = requests.get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP", timeout=10)
            fd, od = f.json().get("data") or [], o.json().get("data") or []
            if fd and od:
                out.update(source="okx", funding_rate_pct_8h=num(float(fd[0]["fundingRate"]) * 100, 4),
                           open_interest_btc=num(od[0].get("oiCcy"), 0))
        except Exception:
            pass
    if not out["source"]:
        try:
            r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"
                             "?instrument_name=BTC-PERPETUAL", timeout=10).json()
            res = (r.get("result") or [{}])[0]
            if res:
                price = res.get("mark_price") or res.get("last") or 0
                oi_usd = res.get("open_interest")
                out.update(source="deribit", funding_rate_pct_8h=num((res.get("funding_8h") or 0) * 100, 4),
                           open_interest_btc=num(oi_usd / price, 0) if oi_usd and price else None)
        except Exception:
            pass
    if out["source"] and out.get("open_interest_btc"):
        prev = state["last_oi"].get(out["source"])
        out["open_interest_change_since_last_run_pct"] = pct(out["open_interest_btc"], prev) if prev else None
        state["last_oi"][out["source"]] = out["open_interest_btc"]
    if not out["source"]:
        out["note"] = "futures data not available this hour"
    return out


def get_fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10).json()["data"]
        return {"today": int(d[0]["value"]), "label": d[0]["value_classification"],
                "yesterday": int(d[1]["value"]) if len(d) > 1 else None}
    except Exception:
        return None


def volume_ratio_daily(ticker):
    df = get_data(ticker, "3mo", "1d")
    if len(df) < 25 or df.Volume.iloc[-21:-1].mean() == 0:
        return None
    return num(df.Volume.iloc[-1] / df.Volume.iloc[-21:-1].mean())


# =====================================================================
# indicators
# =====================================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(df):
    return pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(),
                      (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).rolling(n).mean()


def adx(df, n=14):
    up, dn = df.High.diff(), -df.Low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / tr
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def macd(close):
    m = ema(close, 12) - ema(close, 26)
    s = ema(m, 9)
    return m, s, m - s


def swing_points(df, k=5, count=6):
    hi = df.High[(df.High == df.High.rolling(2 * k + 1, center=True).max())]
    lo = df.Low[(df.Low == df.Low.rolling(2 * k + 1, center=True).min())]
    return [float(x) for x in hi.tail(count)], [float(x) for x in lo.tail(count)]


def volume_poc(df, bars=200, bins=30):
    sub = df.tail(bars)
    if len(sub) < 20 or sub.Volume.sum() <= 0:
        return None
    hist, edges = np.histogram(sub.Close, bins=bins, weights=sub.Volume)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def percentile_rank(series, window=500):
    s = series.dropna().tail(window)
    if len(s) < 20:
        return None
    return num((s < s.iloc[-1]).mean() * 100, 0)


def divergence(close, r, look=20):
    if len(close) < 2 * look + 1:
        return "none"
    p_now, p_prev = close.iloc[-look:], close.iloc[-2 * look:-look]
    r_now, r_prev = r.iloc[-look:], r.iloc[-2 * look:-look]
    if p_now.max() > p_prev.max() and r_now.max() < r_prev.max():
        return "bearish (price higher high, RSI lower high)"
    if p_now.min() < p_prev.min() and r_now.min() > r_prev.min():
        return "bullish (price lower low, RSI higher low)"
    return "none"


# =====================================================================
# ONE market phase for the whole desk
# =====================================================================
def market_phase(h1, d1):
    c, dc = h1.Close, d1.Close
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    de50, de200 = ema(dc, 50), ema(dc, 200)
    weekly = dc.resample("W").last().dropna()
    score = 0
    score += 1 if dc.iloc[-1] > de50.iloc[-1] else -1            # price vs 50-day line
    score += 1 if dc.iloc[-1] > de200.iloc[-1] else -1           # price vs 200-day line
    score += 1 if de50.iloc[-1] > de200.iloc[-1] else -1         # 50-day line vs 200-day line
    if len(weekly) >= 5:
        score += 1 if weekly.iloc[-1] > weekly.iloc[-5] else -1  # 4-week direction
    if len(weekly) >= 7:
        steps = np.sign(np.diff(weekly.iloc[-7:].values))         # last 6 weekly moves
        if (steps < 0).sum() >= 4:
            score -= 1                                            # a run of falling weeks
        elif (steps > 0).sum() >= 4:
            score += 1                                            # a run of rising weeks
    daily = "up" if score >= 2 else "down" if score <= -2 else "mixed"
    if e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]:
        hourly = "up"
    elif e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]:
        hourly = "down"
    elif c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
        hourly = "turning up"
    elif c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
        hourly = "turning down"
    else:
        hourly = "mixed"
    adx_d = float(adx(d1)[0].iloc[-1])
    adx_h = float(adx(h1)[0].iloc[-1])
    atr_pct = percentile_rank(atr(h1)) or 50
    if atr_pct >= 95:
        regime = "extreme volatility"
    elif adx_d >= 20 or adx_h >= 25:
        regime = "trending"
    else:
        regime = "ranging"
    words = {"up": "UP-trend", "down": "DOWN-trend", "mixed": "no clear trend"}
    text = (f"Daily: {words[daily]} (score {score:+d} of 5). Hourly: {hourly}. "
            f"Regime: {regime} (daily ADX {adx_d:.0f}, hourly ADX {adx_h:.0f}).")
    return {"daily": daily, "hourly": hourly, "regime": regime, "score": score,
            "adx_daily": num(adx_d, 1), "adx_hourly": num(adx_h, 1), "text": text}


# =====================================================================
# clean support / resistance with room measured in ATR
# =====================================================================
def key_levels(name, h1, d1, price, a):
    sh_h, sl_h = swing_points(h1, 5, 8)
    sh_d, sl_d = swing_points(d1, 3, 6)
    prev = d1.iloc[-2] if len(d1) > 1 else d1.iloc[-1]
    pivot = (prev.High + prev.Low + prev.Close) / 3
    year = d1.tail(252)
    step = ROUND_STEP.get(name, 1)
    cand = [(x, "hourly swing high") for x in sh_h] + [(x, "hourly swing low") for x in sl_h] + \
           [(x, "daily swing high") for x in sh_d] + [(x, "daily swing low") for x in sl_d] + \
           [(float(prev.High), "previous day high"), (float(prev.Low), "previous day low"),
            (float(2 * pivot - prev.Low), "pivot R1"), (float(2 * pivot - prev.High), "pivot S1"),
            (float(year.High.max()), "52-week high"), (float(year.Low.min()), "52-week low"),
            (float(ema(d1.Close, 200).iloc[-1]), "daily EMA200"),
            (float((price // step) * step), "big round number"), (float((price // step + 1) * step), "big round number")]
    poc = volume_poc(h1)
    if poc:
        cand.append((poc, "volume point of control"))
    cand = [(p, s) for p, s in cand if p and not np.isnan(p)]
    # merge levels that sit within 0.3 ATR of each other into one zone
    cand.sort()
    zones = []
    for p, s in cand:
        if zones and abs(p - zones[-1]["price"]) <= 0.3 * a:
            z = zones[-1]
            z["sources"].add(s)
            z["price"] = (z["price"] * z["n"] + p) / (z["n"] + 1)
            z["n"] += 1
        else:
            zones.append({"price": p, "sources": {s}, "n": 1})
    minor_only = {"hourly swing high", "hourly swing low", "pivot R1", "pivot S1"}
    for z in zones:     # strong = 2+ levels together, or one important level (daily, previous day, 52-week, round...)
        z["strong"] = z["n"] >= 2 or bool(z["sources"] - minor_only)
    at_level = [z for z in zones if abs(z["price"] - price) < MIN_LEVEL_GAP_ATR * a]
    above = sorted([z for z in zones if z["price"] - price >= MIN_LEVEL_GAP_ATR * a], key=lambda z: z["price"])
    below = sorted([z for z in zones if price - z["price"] >= MIN_LEVEL_GAP_ATR * a], key=lambda z: -z["price"])
    strong_above = [z for z in above if z["strong"]]
    strong_below = [z for z in below if z["strong"]]
    fmt = lambda z: {"price": num(z["price"]), "type": ", ".join(sorted(z["sources"])), "strength": z["n"],
                     "strong": z["strong"]}
    room_up = (strong_above[0]["price"] - price) / a if strong_above else 99.0
    room_down = (price - strong_below[0]["price"]) / a if strong_below else 99.0
    return {
        "price_is_at_level": [fmt(z) for z in at_level],
        "strong_resistances_above": [fmt(z) for z in strong_above[:3]],
        "strong_supports_below": [fmt(z) for z in strong_below[:3]],
        "weak_levels_nearby (do not block a trade)": [fmt(z) for z in (above + below) if not z["strong"]][:4],
        "room_up_to_next_resistance_in_ATR": num(min(room_up, 99), 2),
        "room_down_to_next_support_in_ATR": num(min(room_down, 99), 2),
        "buy_has_room": bool(room_up >= TARGET1_ATR),
        "sell_has_room": bool(room_down >= TARGET1_ATR),
        "broke_above_previous_day_high": bool(price > prev.High),
        "broke_below_previous_day_low": bool(price < prev.Low),
        "hourly_atr": num(a),
    }


# =====================================================================
# facts: each agent gets only its own slice (plus the shared market phase)
# =====================================================================
def build_facts(name, h1, d1, macro, extra, news, phase):
    c, dc = h1.Close, d1.Close
    price = float(c.iloc[-1])
    a_h1, a_d1 = atr(h1), atr(d1)
    a = float(a_h1.iloc[-1])
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    de20, de50, de200 = ema(dc, 20), ema(dc, 50), ema(dc, 200)
    adx_h, pdi_h, mdi_h = adx(h1)
    adx_d, pdi_d, mdi_d = adx(d1)
    m_h, s_h, hist_h = macd(c)
    _, _, hist_d = macd(dc)
    r_h, r_d = rsi(c), rsi(dc)
    sh_h, sl_h = swing_points(h1, 5, 4)
    year = d1.tail(252)
    vol_avg = h1.Volume.rolling(20).mean().iloc[-1]
    vol_ratio = num(h1.Volume.iloc[-1] / vol_avg) if vol_avg and vol_avg > 0 else None
    bb_mid = c.rolling(20).mean()
    bb_width = (4 * c.rolling(20).std()) / bb_mid * 100
    range_24h = float(h1.High.tail(24).max() - h1.Low.tail(24).min())
    weekly = dc.resample("W").last().dropna().tail(6)
    levels = key_levels(name, h1, d1, price, a)

    base = {"market": name, "price": num(price),
            "change_24h_pct": pct(price, c.iloc[-25]) if len(c) > 25 else None,
            "MARKET_PHASE (same for whole desk)": phase["text"]}

    facts = {
        "trend": {**base,
                  "hourly": {"ema20": num(e20.iloc[-1]), "ema50": num(e50.iloc[-1]), "ema200": num(e200.iloc[-1]),
                             "ema20_slope_10bars_pct": pct(e20.iloc[-1], e20.iloc[-11]),
                             "adx": num(adx_h.iloc[-1], 1), "plus_di": num(pdi_h.iloc[-1], 1), "minus_di": num(mdi_h.iloc[-1], 1),
                             "macd_above_signal": bool(m_h.iloc[-1] > s_h.iloc[-1]),
                             "last_swing_highs": [num(x) for x in sh_h], "last_swing_lows": [num(x) for x in sl_h]},
                  "daily": {"close": num(dc.iloc[-1]), "ema50": num(de50.iloc[-1]), "ema200": num(de200.iloc[-1]),
                            "adx": num(adx_d.iloc[-1], 1), "plus_di": num(pdi_d.iloc[-1], 1), "minus_di": num(mdi_d.iloc[-1], 1)}},
        "momentum": {**base,
                     "rsi_hourly_now": num(r_h.iloc[-1], 1), "rsi_hourly_6_bars_ago": num(r_h.iloc[-7], 1),
                     "rsi_daily": num(r_d.iloc[-1], 1),
                     "change_6h_pct": pct(price, c.iloc[-7]),
                     "change_5d_pct": pct(dc.iloc[-1], dc.iloc[-6]) if len(dc) > 6 else None,
                     "macd_histogram_now": num(hist_h.iloc[-1], 4), "macd_histogram_3_bars_ago": num(hist_h.iloc[-4], 4),
                     "macd_histogram_rising": bool(hist_h.iloc[-1] > hist_h.iloc[-4]),
                     "volume_vs_20bar_average": vol_ratio, "rsi_divergence_hourly": divergence(c, r_h)},
        "levels": {**base, **levels},
        "volatility": {**base,
                       "hourly_atr": num(a), "hourly_atr_percentile_500bars": percentile_rank(a_h1),
                       "daily_atr": num(a_d1.iloc[-1]), "range_last_24h": num(range_24h),
                       "range_24h_vs_daily_atr": num(range_24h / a_d1.iloc[-1]) if a_d1.iloc[-1] else None,
                       "bollinger_width_percentile": percentile_rank(bb_width),
                       "atr_rising_vs_5_bars_ago": bool(a_h1.iloc[-1] > a_h1.iloc[-6]),
                       "last_24h_move_direction": "up" if price > c.iloc[-25] else "down"},
        "daily_view": {**base,
                       "daily_close": num(dc.iloc[-1]), "daily_ema20": num(de20.iloc[-1]), "daily_ema50": num(de50.iloc[-1]),
                       "daily_ema200": num(de200.iloc[-1]), "daily_rsi": num(r_d.iloc[-1], 1), "daily_adx": num(adx_d.iloc[-1], 1),
                       "daily_macd_histogram": num(hist_d.iloc[-1], 4),
                       "change_20d_pct": pct(dc.iloc[-1], dc.iloc[-21]) if len(dc) > 21 else None,
                       "change_60d_pct": pct(dc.iloc[-1], dc.iloc[-61]) if len(dc) > 61 else None,
                       "position_in_52w_range_pct": num((price - year.Low.min()) / (year.High.max() - year.Low.min()) * 100, 0)
                       if year.High.max() > year.Low.min() else None,
                       "last_6_weekly_closes": [num(x) for x in weekly]},
        "news": {**base, "headlines": news},
        "mood": {**base, "vix_fear_index": macro.get("vix_fear_index"),
                 "rsi_daily": num(r_d.iloc[-1], 1), "volume_vs_20bar_average": vol_ratio,
                 "change_5d_pct": pct(dc.iloc[-1], dc.iloc[-6]) if len(dc) > 6 else None,
                 **({"crypto_fear_greed": extra.get("fear_greed"),
                     "funding_rate_pct_8h": extra.get("derivatives", {}).get("funding_rate_pct_8h")} if name == "BTC" else {})},
        "macro": {**base, **macro},
        "special": {**base, **extra},
    }
    summary = {**base, "hourly_atr": num(a), "daily_trend": phase["daily"], "hourly_trend": phase["hourly"],
               "rsi_hourly": num(r_h.iloc[-1], 1), "rsi_daily": num(r_d.iloc[-1], 1),
               "room_up_in_ATR": levels["room_up_to_next_resistance_in_ATR"],
               "room_down_in_ATR": levels["room_down_to_next_support_in_ATR"],
               "nearest_strong_resistance": (levels["strong_resistances_above"] or [{}])[0].get("price"),
               "nearest_strong_support": (levels["strong_supports_below"] or [{}])[0].get("price"),
               "change_5d_pct": pct(dc.iloc[-1], dc.iloc[-6]) if len(dc) > 6 else None}
    raw = {"price": price, "atr": a, "rsi_h": float(r_h.iloc[-1]), "rsi_h_6": float(r_h.iloc[-7]),
           "hist_rising": bool(hist_h.iloc[-1] > hist_h.iloc[-4]), "hist_h": float(hist_h.iloc[-1]),
           "room_up": levels["room_up_to_next_resistance_in_ATR"], "room_down": levels["room_down_to_next_support_in_ATR"],
           "atr_pct": percentile_rank(a_h1) or 50, "range_vs_atr": (range_24h / a_d1.iloc[-1]) if a_d1.iloc[-1] else 1,
           "chg24": pct(price, c.iloc[-25]) if len(c) > 25 else 0,
           "oi_chg": (extra.get("derivatives") or {}).get("open_interest_change_since_last_run_pct"),
           "funding": (extra.get("derivatives") or {}).get("funding_rate_pct_8h")}
    return facts, summary, raw


def get_extra(name, state, macro):
    if name == "BTC":
        _, eth_5d = last_change("ETH-USD", 5)
        _, btc_5d = last_change("BTC-USD", 5)
        return {"derivatives": get_btc_derivatives(state), "fear_greed": get_fear_greed(),
                "eth_5d_change_pct": eth_5d, "btc_5d_change_pct": btc_5d,
                "eth_minus_btc_5d_pct": num(eth_5d - btc_5d) if eth_5d is not None and btc_5d is not None else None,
                "bitcoin_etf_IBIT_volume_vs_20day_avg": volume_ratio_daily("IBIT")}
    if name == "GOLD":
        si_last, si_5d = last_change("SI=F", 5)
        gc_last, gc_5d = last_change("GC=F", 5)
        _, tip_5d = last_change("TIP", 5)
        return {"silver_5d_change_pct": si_5d, "gold_5d_change_pct": gc_5d,
                "gold_silver_ratio": num(gc_last / si_last) if gc_last and si_last else None,
                "TIP_bond_5d_change_pct (up = real yields falling = good for gold)": tip_5d,
                "gold_etf_GLD_volume_vs_20day_avg": volume_ratio_daily("GLD"),
                "dollar_index": macro.get("dollar_index"), "us10y_yield": macro.get("us10y_yield")}
    if name == "OIL":
        bz_last, bz_5d = last_change("BZ=F", 5)
        cl_last, cl_5d = last_change("CL=F", 5)
        _, xle_5d = last_change("XLE", 5)
        now = dt.datetime.now(dt.timezone.utc)
        heads = [h for h in get_news(["CL=F", "USO", "BZ=F"], 10)
                 if any(k in h.lower() for k in ("opec", "eia", "inventor", "stockpile", "supply", "sanction",
                                                  "iran", "russia", "saudi", "barrel", "refin", "hurricane"))]
        return {"brent_price": bz_last, "brent_minus_wti_spread": num(bz_last - cl_last) if bz_last and cl_last else None,
                "wti_5d_change_pct": cl_5d, "brent_5d_change_pct": bz_5d,
                "energy_stocks_XLE_5d_change_pct": xle_5d,
                "xle_minus_wti_5d_pct": num(xle_5d - cl_5d) if xle_5d is not None and cl_5d is not None else None,
                "today_utc_weekday": now.strftime("%A"),
                "us_eia_weekly_oil_report": "TODAY (Wednesday 14:30 UTC)" if now.weekday() == 2 else "on Wednesday",
                "oil_supply_headlines": heads[:8], "dollar_index": macro.get("dollar_index")}
    return {}


# =====================================================================
# the 30-year veterans: one way of thinking for all, own expertise for each
# =====================================================================
DESK_RULES = """
HOW A 30-YEAR VETERAN THINKS (follow these 5 steps in your head, in this order):
Step 1 - Phase. Read MARKET_PHASE. It is measured once for the whole desk; accept it, do not re-measure it.
         In a trending phase you trade WITH the daily trend. In a ranging phase you buy support and sell resistance.
         In extreme volatility you lower your confidence.
Step 2 - Top-down. Bigger timeframe first (weekly, daily), hourly only for timing.
Step 3 - Both sides. Apply your rules for BUY and for SELL with equal care, then pick the side your data favours.
         Pick BUY or SELL whenever your evidence leans at least 55/45. Confidence: 5 = slight lean, 6-7 = clear edge,
         8-9 = strong, 10 = almost never.
         WAIT only if your evidence is truly balanced or your data is missing. "Uncertain", "be careful",
         "no confirmation yet" are NOT reasons to wait - a veteran states his lean and sizes it with confidence.
Step 4 - Proof of being wrong. Name the price or condition that would prove your vote wrong.
Step 5 - Vote. Your reason: 1-2 short simple-English sentences (max 45 words) naming the exact numbers you used.

You are one of 12 desk members. Judge ONLY from YOUR data; the others cover the other areas.
Trades last hours to 2 days. Stop-loss = 1.5 x hourly ATR. Target 1 = 2 x hourly ATR.

Two examples of good answers (different markets and numbers each time):
{"vote": "SELL", "confidence": 7, "reason": "Daily down-trend and hourly RSI fell from 49 to 37 with the MACD histogram still falling; sellers are in control.", "wrong_if": "hourly RSI climbs back above 50"}
{"vote": "BUY", "confidence": 6, "reason": "Price is holding the 2400 daily support with 2.6 ATR of room up to resistance at 2460 in a daily up-trend.", "wrong_if": "an hourly close below 2390"}

Reply ONLY as JSON: {"vote": "BUY" or "SELL" or "WAIT", "confidence": 1-10, "reason": "...", "wrong_if": "..."}
"""

EXPERTS = {
    "trend": """You are a trend-following trader with 30 years on futures and currency desks, trained in the Turtle school.
What you check: daily close vs EMA50 and EMA200 and their order; hourly EMA20/50/200 order and the EMA20 slope;
ADX strength (above 25 real trend, below 20 weak) and which DI line is on top; the last swing highs and lows
(higher highs and lows = up, lower highs and lows = down); MACD vs its signal line.
Your rules: trade with the daily trend and use the hourly to time it. Daily down + hourly down = SELL rallies.
Daily up + hourly up = BUY dips. Daily trend clear but hourly against it = still vote the DAILY direction with confidence 5-6
(the hourly pullback is your entry, not a reason to wait). Never buy a down-trend because it looks cheap.""",

    "momentum": """You are a momentum specialist with 30 years of trading, a student of Wilder and Murphy.
What you check: hourly RSI now vs 6 bars ago, daily RSI, the MACD histogram now vs 3 bars ago, 6-hour and 5-day change,
volume vs average, divergence.
Your rules - they are equal for both sides:
- RSI below 50 and falling, MACD histogram falling = SELL (bearish momentum building).
- RSI above 50 and rising, MACD histogram rising = BUY (bullish momentum building).
- RSI below 30 in a down-trend is NOT a buy by itself; it only becomes a BUY when RSI and the histogram both turn up.
- RSI above 70 in an up-trend is NOT a sell by itself; it only becomes a SELL when RSI and the histogram both turn down.
- Divergence is an early warning that lowers confidence in the current move; it needs the histogram to turn to flip your vote.""",

    "levels": """You are a price-action trader with 30 years of experience, starting on the trading floor. You trade support,
resistance and institutional zones (previous day high and low, pivots, big round numbers, volume point of control,
52-week extremes, daily EMA200). The code has already removed tiny levels, marked which levels are STRONG,
and measured the room to the next STRONG level in ATR for you. Weak levels (a single hourly swing) do not block a trade.
Your rules:
- BUY needs room UP of at least 2 ATR (buy_has_room true). SELL needs room DOWN of at least 2 ATR (sell_has_room true).
- Room ABOVE does not matter for a SELL, and room BELOW does not matter for a BUY.
- Price at support (price_is_at_level, a low-type level) in an up or mixed phase with room up = BUY.
- Price at resistance (a high-type level) in a down or mixed phase with room down = SELL.
- A break below the previous day low in a down-trend with room down = SELL; a break above the previous day high in an
  up-trend with room up = BUY.
- If the trend side has room, vote the trend side. WAIT only when NEITHER side has room.""",

    "volatility": """You are a volatility and risk-regime specialist with 30 years on options desks.
What you check: hourly ATR percentile, whether ATR is rising, Bollinger width percentile (low = squeeze),
24h range vs daily ATR, the direction of the last 24h move, and MARKET_PHASE.
Your rules: rising volatility in the direction of the daily trend continues - vote that direction.
A squeeze (Bollinger percentile below 20) breaks in the direction of the daily trend - vote it.
After a stretched move (range above 2x daily ATR) against the daily trend, a snap-back toward the trend is likely - vote the trend side.
WAIT only if ATR percentile is above 95 AND the range is above 2.5x daily ATR, or MARKET_PHASE shows no clear daily trend
and volatility gives no direction.""",

    "daily_view": """You are a position trader with 30 years of experience who reads only weekly and daily charts.
What you check: the last 6 weekly closes, daily close vs EMA20/50/200, position in the 52-week range, 20 and 60-day change,
daily RSI, daily ADX and MACD histogram.
Your rules: the higher timeframe sets the direction. Buy dips in daily up-trends, sell rallies in daily down-trends.
A market in the lower 20% of its 52-week range in a down-trend usually goes lower first. A market making new highs in an
up-trend usually goes higher. You ignore the hourly chart completely.""",

    "news": """You are a market news analyst with 30 years of experience, formerly running a Reuters and Bloomberg
commodities and crypto desk.
What you do: mark each headline bullish, bearish or neutral for THIS market; weigh it (central banks, wars, supply shocks,
regulation, ETF flows matter most; opinion pieces little); compare with the 24h price change.
Your rules: bad news that price ignores is bullish, good news that price ignores is bearish. Price falling on bearish news
in a down-trend = SELL. If there are no relevant headlines, vote WAIT with confidence 3 and say so.""",

    "mood": """You are a sentiment and positioning specialist with 30 years of experience, a disciplined contrarian at extremes
and a trend follower in between.
What you check: VIX level and its 5-day change (above 25 = fear); for Bitcoin also Fear and Greed (below 20 extreme fear,
above 80 extreme greed) and the funding rate; daily RSI, 5-day change and volume.
Your rules: EXTREME fear plus a big fall plus selling volume drying up = contrarian BUY. EXTREME greed after a big rise = SELL.
Between the extremes, vote WITH the daily trend: rising fear in a down-trend = SELL for Bitcoin and Oil;
rising fear helps Gold, so it is BUY for Gold when the daily trend is not down.""",

    "macro": """You are a global macro strategist with 30 years at a macro hedge fund.
How each market reacts: Bitcoin trades like a high-risk tech stock (falling dollar, falling yields, rising stocks help it).
Gold likes a falling dollar, falling real yields and fear. Oil likes strong growth and rising stocks; recession fear hurts it.
What you check: 5-day change of the dollar index, US 10-year yield, S&P 500 and VIX.
Decide whether the macro wind is a tailwind (BUY) or a headwind (SELL) for THIS market and how strong it is.""",

    "special_BTC": """You are a crypto derivatives trader with 30 years in derivatives, the last 14 in Bitcoin.
What you check: funding rate (above 0.03% per 8h = longs crowded, below 0 = shorts crowded); open interest change vs price
(price down + OI up = new shorts, bearish; price down + OI down = longs giving up; price up + OI up = new longs, bullish);
Fear and Greed today vs yesterday; ETH vs BTC over 5 days (ETH stronger = risk appetite); IBIT ETF volume vs average.
Your rules: crowded positioning gets squeezed the other way. A big ETF volume spike on a down day is usually selling,
on an up day usually buying - read it with the price direction. If futures data is missing, judge from the rest.""",

    "special_GOLD": """You are a bullion dealer with 30 years at a London bullion bank.
What you check: dollar index and US 10-year yield over 5 days (both falling = tailwind); the TIP bond ETF (up = real yields
falling = good for gold); silver vs gold over 5 days (silver confirming = genuine move); gold/silver ratio; GLD volume.
Your rules: gold hates rising real yields and a strong dollar; when the dollar and yields send opposite signals, the real-yield
signal (TIP) wins.""",

    "special_OIL": """You are a crude oil trader with 30 years in physical and futures oil.
What you check: supply headlines (OPEC+, sanctions, wars, hurricanes, refineries), the weekly US EIA report (Wednesday;
draws bullish, builds bearish), Brent minus WTI spread, energy stocks (XLE) vs crude over 5 days, the dollar.
Your rules: supply shocks drive the big moves. Energy stocks falling LESS than crude is only a mild positive; it does not
beat a clear daily down-trend plus bearish supply news.""",
}

BULL_PROMPT = """You are a senior long-only portfolio manager with 30 years of experience. Your job: find the best honest BUY
case for {name} right now, using the desk's opinions and the key facts.
Check: does the daily phase allow buying? Is there at least 2 ATR of room UP to the next resistance (room_up_in_ATR)?
If the long case is genuinely good, vote BUY with a confidence. If it is weak, vote WAIT and name what is missing.
You never vote SELL. Reason: 1-2 short simple-English sentences (max 45 words) with exact numbers.
Desk opinions: {votes}
Key facts: {summary}
Reply ONLY as JSON: {{"vote": "BUY" or "WAIT", "confidence": 1-10, "reason": "...", "wrong_if": "..."}}"""

BEAR_PROMPT = """You are a senior short-seller with 30 years of experience. Your job: find the best honest SELL case for {name}
right now, using the desk's opinions and the key facts.
Check: is the daily phase down or weakening? Are momentum and the economy against this market?
Is there at least 2 ATR of room DOWN to the next support (room_down_in_ATR)? Room UP to resistance does not matter for a short.
If the short case is genuinely good, vote SELL with a confidence. If it is weak, vote WAIT and name what is missing.
You never vote BUY. Reason: 1-2 short simple-English sentences (max 45 words) in your own words, with exact numbers.
Desk opinions: {votes}
Key facts: {summary}
Reply ONLY as JSON: {{"vote": "SELL" or "WAIT", "confidence": 1-10, "reason": "...", "wrong_if": "..."}}"""

RISK_PROMPT = """You are the chief risk officer of a trading desk, 30 years of experience.
Proposed paper trade on {name}: {plan}
Key facts: {summary}
Headlines: {news}
Open paper trades: {open_trades}
Approve only if ALL are true:
1) For a BUY there is at least 2 ATR of room UP; for a SELL at least 2 ATR of room DOWN (see key facts).
2) The stop-loss is not sitting right on top of an obvious level that will be hunted.
3) No major event in the next few hours (central bank decision, EIA report day for oil, big US data).
4) It does not double a bet that is already open.
Reason: 1-2 short simple-English sentences (max 45 words).
Reply ONLY as JSON: {{"approve": true or false, "reason": "..."}}"""

PM_PROMPT = """You are the head of the trading desk, 30 years of experience. You give the final yes or no.
Trade on {name}: {plan}
Market phase: {phase}
Desk vote count: {count}
Bull view: {bull}
Bear view: {bear}
Risk officer: {risk}
Take the trade if it goes with the daily trend, the desk clearly agrees and the risk officer approved.
Reason: 1-2 short simple-English sentences (max 45 words).
Reply ONLY as JSON: {{"final": "TAKE" or "SKIP", "reason": "..."}}"""

INSPECTOR_PROMPT = """You are the desk's QUALITY INSPECTOR: 30 years as head of trading performance review at a large bank.
You do not trade. The chart members are already checked by fixed code rules. You check the JUDGEMENT members on {name}:
news, mood, macro, special (market specialist), bull and bear.

Shared facts (measured by code, always correct):
{facts}

The judgement members' votes, reasons, and the exact data EACH was given:
{members}

What each must do:
- news: judge only its headlines. Price falling on bearish headlines = SELL is correct; no relevant headlines = WAIT.
- mood: contrarian at EXTREME fear or greed; otherwise vote with the daily trend (for Gold, rising fear is good
  unless the daily trend is down).
- macro: judge the dollar, yields, stocks and VIX for THIS market.
- special: follow its own market rules (Bitcoin: price down + open interest up = new shorts = bearish).
- bull: BUY or WAIT only, needs room UP. bear: SELL or WAIT only, needs room DOWN.

Your only power: you may REMOVE a vote (turn a BUY or SELL into WAIT) when the member clearly broke its own rule or its
reason contradicts ITS OWN data. You may NOT remove a vote just because the shared market phase is mixed - news, macro
and special judge their own data, not the chart. You may not create a BUY or SELL.
Remove at most the single worst vote. Quote the exact number or headline from that member's data as proof.
Use ONLY these member names: {names}
Reply ONLY as JSON: {{"remove": {{"agent": "<one of the names above, or none>", "problem": "<one short sentence>",
"evidence": "<exact number or headline from its data>"}}, "warnings": ["<short note about any other doubtful vote>"],
"overall": "<one short sentence about the desk's work>"}}"""

MAX_INSPECTOR_FIXES = 1     # the inspector may remove at most 1 vote per market
INSPECT_MIN_SIDE = 5        # the inspector only runs when at least this many agree on one side
MAX_CHANGED_IN_TRADE = 1    # a trade may rest on at most this many inspector-changed votes

ANALYSTS = ["trend", "momentum", "levels", "volatility", "daily_view", "news", "mood", "macro", "special"]
JUDGEMENT = ["news", "mood", "macro", "special", "bull", "bear"]
ALL_VOTERS = ANALYSTS + ["bull", "bear", "time_machine"]


# =====================================================================
# Gemini: never gets stuck
# =====================================================================
def ask_gemini(prompt):
    if not time_left():
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}}
    STATS["asked"] += 1
    for attempt in range(2):
        try:
            r = requests.post(url, json=body, timeout=30)
            if r.status_code == 200:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if txt.startswith("```"):
                    txt = txt.strip("`").replace("json", "", 1).strip()
                ans = json.loads(txt)
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if isinstance(ans, dict):
                    STATS["answered"] += 1
                    return ans
                return None
            if r.status_code in (429, 500, 503):
                STATS["busy"] += 1
                log(f"Gemini busy ({r.status_code}), short wait")
                time.sleep(6)
                continue
            msg = f"Gemini error {r.status_code}: {r.text[:150]}"
            log(msg)
            if msg not in STATS["errors"]:
                STATS["errors"].append(msg)
            return None
        except Exception as e:
            log(f"Gemini call failed: {e}")
    return None


def clean_vote(ans, allowed=("BUY", "SELL", "WAIT")):
    if not isinstance(ans, dict):
        return None
    v = str(ans.get("vote", "WAIT")).upper().strip()
    if v not in allowed:
        v = "WAIT"
    try:
        c = int(round(float(ans.get("confidence", 5))))
    except Exception:
        c = 5
    return {"vote": v, "orig": v, "confidence": max(1, min(10, c)), "reason": str(ans.get("reason", "")).strip(),
            "wrong_if": str(ans.get("wrong_if", "")).strip()}


def memory_note(state, name, agent):
    hist = state["memory"].get(f"{name}:{agent}", [])[-10:]
    if not hist:
        return "No checked calls yet."
    right = sum(1 for h in hist if h["right"])
    last = ", ".join(f"{h['said']} {'right' if h['right'] else 'wrong'}" for h in hist[-5:])
    return (f"Your last {len(hist)} checked calls: {right} right, {len(hist) - right} wrong "
            f"(target hit before stop-loss = right). Most recent: {last}. Learn from it.")


def analyst_prompt(name, agent, facts, state):
    expert = EXPERTS[f"special_{name}"] if agent == "special" else EXPERTS[agent]
    return (f"{expert}\n{DESK_RULES}\nMarket: {name}\n"
            f"Your track record: {memory_note(state, name, agent)}\n"
            f"YOUR DATA: {jdump(facts[agent])}")


def debate_prompts(name, summary, votes):
    short = {a: f"{v['vote']} {v['confidence']}/10: {v['reason']}" for a, v in votes.items() if a in ANALYSTS}
    return (BULL_PROMPT.format(name=name, votes=jdump(short), summary=jdump(summary)),
            BEAR_PROMPT.format(name=name, votes=jdump(short), summary=jdump(summary)))


# =====================================================================
# rule backup (used only when Gemini gave no answer even after a retry)
# =====================================================================
def trend_side(phase):
    if phase["daily"] in ("up", "down"):
        return "BUY" if phase["daily"] == "up" else "SELL"
    if phase["hourly"] in ("up", "down"):
        return "BUY" if phase["hourly"] == "up" else "SELL"
    return None


def code_rule(agent, phase, raw):
    """The fixed rule for each chart member. Returns (vote, why) or (None, why) when the rule gives no clear answer."""
    d_side = "BUY" if phase["daily"] == "up" else "SELL" if phase["daily"] == "down" else None
    t_side = trend_side(phase)
    if agent == "trend":
        return (t_side, f"daily {phase['daily']}, hourly {phase['hourly']}") if t_side else (None, "no clear trend")
    if agent == "momentum":
        if raw["rsi_h"] < 50 and raw["rsi_h"] < raw["rsi_h_6"] and not raw["hist_rising"]:
            return "SELL", f"RSI {raw['rsi_h']:.1f} below 50 and falling, MACD histogram falling"
        if raw["rsi_h"] > 50 and raw["rsi_h"] > raw["rsi_h_6"] and raw["hist_rising"]:
            return "BUY", f"RSI {raw['rsi_h']:.1f} above 50 and rising, MACD histogram rising"
        return None, "momentum mixed"
    if agent == "levels":
        if t_side == "BUY" and raw["room_up"] >= TARGET1_ATR:
            return "BUY", f"trend up with {raw['room_up']} ATR room up to a strong level"
        if t_side == "SELL" and raw["room_down"] >= TARGET1_ATR:
            return "SELL", f"trend down with {raw['room_down']} ATR room down to a strong level"
        if raw["room_up"] < TARGET1_ATR and raw["room_down"] < TARGET1_ATR:
            return "WAIT", f"no room either way (up {raw['room_up']}, down {raw['room_down']} ATR)"
        return None, "room only against the trend"
    if agent == "volatility":
        if raw["atr_pct"] > 95 and raw["range_vs_atr"] > 2.5:
            return "WAIT", "volatility extreme"
        return (d_side, f"daily {phase['daily']}-trend") if d_side else (None, "no daily trend")
    if agent == "daily_view":
        return (d_side, f"daily {phase['daily']}-trend (score {phase['score']:+d})") if d_side else (None, "no daily trend")
    return None, ""


def rule_backup(agent, name, phase, raw, macro):
    v, why = code_rule(agent, phase, raw)
    if agent == "macro":
        dxy = (macro.get("dollar_index") or {}).get("change_5d_pct") or 0
        spx = (macro.get("sp500") or {}).get("change_5d_pct") or 0
        yld = (macro.get("us10y_yield") or {}).get("change_5d_pct") or 0
        score = (-1 if dxy > 0.3 else 1 if dxy < -0.3 else 0) + \
                ((-1 if yld > 1.5 else 1 if yld < -1.5 else 0) if name == "GOLD" else (1 if spx > 0.5 else -1 if spx < -0.5 else 0))
        v = ("BUY" if score > 0 else "SELL") if score else None
        why = f"dollar {dxy:+.2f}%, stocks {spx:+.2f}%, yields {yld:+.2f}% over 5 days"
    v = v or "WAIT"
    return {"vote": v, "orig": v, "confidence": 5 if v != "WAIT" else 3,
            "reason": f"[rule backup - Gemini did not answer] {why}.", "wrong_if": "", "backup": True}


# =====================================================================
# quality check: catches votes that contradict the data
# =====================================================================
def quality_check(name, votes, phase, raw):
    """The code has the final word on the 5 chart members and on clear Bitcoin futures signals."""
    flags = []
    for agent in ("trend", "momentum", "levels", "volatility", "daily_view"):
        v = votes.get(agent)
        if not v or v.get("backup"):
            continue
        rule, why = code_rule(agent, phase, raw)
        if rule is None or v["vote"] == rule:
            continue
        old, new = v["vote"], rule                       # the fixed rule decides chart members
        v["vote"] = new
        v["confidence"] = 5 if new != "WAIT" else 3
        v["reason"] = f"[code rule changed {old} to {new}: {why}] " + v["reason"]
        flags.append(f"{LABELS[agent]} {old} -> {new} ({why})")
    sp = votes.get("special")
    if name == "BTC" and sp and sp["vote"] == "BUY" and (raw.get("chg24") or 0) < 0 and (raw.get("oi_chg") or 0) > 0 \
            and (raw.get("funding") or 0) >= 0:
        sp["vote"], sp["confidence"] = "WAIT", 3
        sp["reason"] = (f"[code rule changed BUY to WAIT: price down {raw['chg24']}% with open interest up "
                        f"{raw['oi_chg']}% = new shorts, which is bearish] " + sp["reason"])
        flags.append("Market specialist BUY -> WAIT (price down + open interest up = new shorts)")
    return flags


def norm_agent_name(text):
    t = str(text).strip().lower()
    for key, label in LABELS.items():
        if t == key or t == label.lower():
            return key
    t = t.split("(")[0].strip().replace(" ", "_").replace("/", "_")
    aliases = {"support_resistance": "levels", "levels": "levels", "big_picture": "daily_view", "daily": "daily_view",
               "market_mood": "mood", "economy": "macro", "market_specialist": "special", "specialist": "special",
               "bull": "bull", "bear": "bear"}
    return aliases.get(t, t)


def ai_inspector(name, votes, phase, raw, summary, facts):
    """Gemini inspector: reviews the 6 judgement members; may only remove ONE bad vote, with proof from its own data."""
    count = {"BUY": 0, "SELL": 0}
    for v in votes.values():
        if v["vote"] in count:
            count[v["vote"]] += 1
    if max(count.values()) < INSPECT_MIN_SIDE:
        return [], [], f"not needed (no side has {INSPECT_MIN_SIDE} votes yet)"
    if not time_left():
        return [], [], "skipped (time up)"
    shared = {**summary, "buy_has_room_2ATR": raw["room_up"] >= TARGET1_ATR, "sell_has_room_2ATR": raw["room_down"] >= TARGET1_ATR}
    members = {}
    for a in JUDGEMENT:
        v = votes.get(a)
        if not v or v.get("backup"):
            continue
        data_for = facts.get(a, {}) if a in ANALYSTS else {"see": "shared facts"}
        data_for = {k: val for k, val in data_for.items() if k not in ("market", "MARKET_PHASE (same for whole desk)")}
        members[a] = {"vote": v["vote"], "confidence": v["confidence"], "reason": v["reason"], "its_data": data_for}
    if not members:
        return [], [], "nothing to check"
    ans = ask_gemini(INSPECTOR_PROMPT.format(name=name, facts=jdump(shared), members=jdump(members),
                                             names=", ".join(members)))
    if not isinstance(ans, dict):
        return [], [], "did not answer"
    fixes, notes = [], [str(w).strip() for w in (ans.get("warnings") or []) if str(w).strip()][:3]
    rm = ans.get("remove") or {}
    if isinstance(rm, list):
        rm = rm[0] if rm and isinstance(rm[0], dict) else {}
    agent = norm_agent_name(rm.get("agent", "")) if isinstance(rm, dict) else ""
    if agent in members and votes[agent]["vote"] in ("BUY", "SELL"):
        problem = str(rm.get("problem", "")).strip()
        evidence = str(rm.get("evidence", "")).strip()
        own = jdump(members[agent]["its_data"]).lower() + " " + members[agent]["reason"].lower()
        has_proof = len(evidence) >= 4 and (any(ch.isdigit() for ch in evidence) or evidence.lower()[:25] in own)
        phase_only = agent in ("news", "macro", "special") and any(w in (problem + evidence).lower()
                                                                     for w in ("phase", "mixed", "no clear trend"))
        if has_proof and not phase_only:
            old = votes[agent]["vote"]
            votes[agent].update(vote="WAIT", confidence=3, inspected=True, removed_side=old)
            votes[agent]["reason"] = f"[AI inspector removed {old}: {problem}] " + votes[agent]["reason"]
            fixes.append(f"{LABELS.get(agent, agent)} {old} -> WAIT: {problem} (proof: {evidence})")
        elif problem:
            notes.append(f"{LABELS.get(agent, agent)}: {problem} (not applied: "
                         f"{'reason was the shared phase' if phase_only else 'no proof from its own data'})")
    return fixes, notes, str(ans.get("overall", "")).strip()


# =====================================================================
# judging trades: what was hit first, stop-loss or target?
# =====================================================================
def first_touch(h1, side, entry, stop, target, start_ts, max_bars=MAX_HOLD_BARS):
    bars = h1[h1.index > start_ts].head(max_bars)
    risk = abs(entry - stop)
    for _, b in bars.iterrows():
        if side == "BUY":
            if b.Low <= stop:
                return "loss", -1.0
            if b.High >= target:
                return "win", round(abs(target - entry) / risk, 2)
        else:
            if b.High >= stop:
                return "loss", -1.0
            if b.Low <= target:
                return "win", round(abs(target - entry) / risk, 2)
    if len(bars) >= max_bars:
        last = float(bars.Close.iloc[-1])
        move = (last - entry) if side == "BUY" else (entry - last)
        return "expired", round(move / risk, 2) if risk else 0.0
    return "open", None


def levels_for(side, entry, a):
    if side == "BUY":
        return entry - STOP_ATR * a, entry + TARGET1_ATR * a, entry + TARGET2_ATR * a
    return entry + STOP_ATR * a, entry - TARGET1_ATR * a, entry - TARGET2_ATR * a


def update_report_card(state, data):
    for rec in state["votes_log"]:
        if rec.get("checked"):
            continue
        h1 = data.get(rec["market"], {}).get("h1")
        if h1 is None or not len(h1) or not rec.get("atr"):
            continue
        start = pd.Timestamp(rec["time"], unit="s", tz="UTC")
        results = {}
        for side in ("BUY", "SELL"):
            stop, t1, _ = levels_for(side, rec["price"], rec["atr"])
            results[side] = first_touch(h1, side, rec["price"], stop, t1, start)[0]
        if "open" in results.values() and time.time() - rec["time"] < (MAX_HOLD_BARS + 24) * 3600:
            continue
        rec["checked"] = True
        judged = [(a, said, said) for a, said in rec["votes"].items()] + \
                 [("inspector", "REMOVED " + side, side) for side in rec.get("inspector_removed", {}).values()]
        for agent, said, side in judged:
            if side not in ("BUY", "SELL") or results[side] not in ("win", "loss"):
                continue
            key = f"{rec['market']}:{agent}"
            if agent == "inspector":                       # the inspector was right if the removed vote would have lost
                mem = state["memory"].setdefault(key, [])
                mem.append({"said": said, "right": results[side] == "loss", "price": rec["price"]})
                state["memory"][key] = mem[-20:]
                continue
            mem = state["memory"].setdefault(key, [])
            mem.append({"said": said, "right": results[said] == "win", "price": rec["price"]})
            state["memory"][key] = mem[-20:]
            if len(state["memory"][key]) >= 5:
                acc = sum(h["right"] for h in state["memory"][key]) / len(state["memory"][key])
                state["weights"][key] = round(0.5 + acc, 2)


def update_trades(state, data):
    for t in state["trades"]:
        if t["status"] != "open":
            continue
        h1 = data.get(t["market"], {}).get("h1")
        if h1 is None or not len(h1):
            continue
        res, r = first_touch(h1, t["side"], t["entry"], t["stop"], t["target1"],
                             pd.Timestamp(t["time"], unit="s", tz="UTC"))
        if res != "open":
            t["status"], t["r"], t["closed"] = res, r, time.time()


def open_trade_r(t, price):
    risk = abs(t["entry"] - t["stop"])
    if not risk or price is None:
        return 0.0
    move = (price - t["entry"]) if t["side"] == "BUY" else (t["entry"] - price)
    return round(move / risk, 2)


def todays_result_r(state, prices):
    today = dt.datetime.now(dt.timezone.utc).date()
    closed = sum(t.get("r") or 0 for t in state["trades"]
                 if t["status"] != "open" and t.get("closed")
                 and dt.datetime.fromtimestamp(t["closed"], dt.timezone.utc).date() == today)
    opened = sum(open_trade_r(t, prices.get(t["market"])) for t in state["trades"] if t["status"] == "open")
    return round(closed + opened, 2)


# =====================================================================
# time machine: proves 4 rules on older data, checks them on newer data
# =====================================================================
def rule_signals(h1):
    c = h1.Close
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c)
    m, s, _ = macd(c)
    hh, ll = h1.High.rolling(20).max().shift(1), h1.Low.rolling(20).min().shift(1)
    va = h1.Volume.rolling(20).mean()
    has_vol = bool(h1.Volume.tail(100).sum() > 0)
    vol_ok = (h1.Volume > 1.3 * va) if has_vol else pd.Series(True, index=h1.index)
    return {
        "trend pullback": ((e50 > e200) & (c > e20) & (c.shift(1) <= e20.shift(1)) & (r > 45),
                           (e50 < e200) & (c < e20) & (c.shift(1) >= e20.shift(1)) & (r < 55)),
        "20-bar breakout": ((c > hh) & vol_ok & (e50 > e200), (c < ll) & vol_ok & (e50 < e200)),
        "dip in uptrend": ((r < 32) & (c > e200), (r > 68) & (c < e200)),
        "MACD cross with trend": ((m > s) & (m.shift(1) <= s.shift(1)) & (c > e50),
                                  (m < s) & (m.shift(1) >= s.shift(1)) & (c < e50)),
        "RSI 50 cross with trend": ((r > 50) & (r.shift(1) <= 50) & (e20 > e50) & (c > e200),
                                    (r < 50) & (r.shift(1) >= 50) & (e20 < e50) & (c < e200)),
    }


def simulate(h1, buy, sell, a, start=200, end=None):
    """Plays the rule one trade at a time. Returns list of (signal_index, side, r_multiple or None if still running)."""
    hi, lo, cl = h1.High.values, h1.Low.values, h1.Close.values
    b, s, av = buy.fillna(False).values, sell.fillna(False).values, a.values
    n = len(cl) if end is None else end
    i, res = start, []
    while i < n:
        side = "BUY" if b[i] else "SELL" if s[i] else None
        if side is None or np.isnan(av[i]) or not av[i]:
            i += 1
            continue
        entry = cl[i]
        stop, t1, _ = levels_for(side, entry, av[i])
        risk = abs(entry - stop)
        j, out = i + 1, None
        while j < min(len(cl), i + 1 + MAX_HOLD_BARS):
            if side == "BUY":
                if lo[j] <= stop:
                    out = -1.0; break
                if hi[j] >= t1:
                    out = WIN_R; break
            else:
                if hi[j] >= stop:
                    out = -1.0; break
                if lo[j] <= t1:
                    out = WIN_R; break
            j += 1
        if out is None and j >= i + 1 + MAX_HOLD_BARS:
            move = (cl[j - 1] - entry) if side == "BUY" else (entry - cl[j - 1])
            out = move / risk
        res.append((i, side, out))
        if out is None:
            break                                 # still running at the end of the data
        i = j + 1
    return res


def stats(results):
    done = [r for _, _, r in results if r is not None]
    if not done:
        return {"trades": 0, "win_pct": 0.0, "avg_r": 0.0}
    return {"trades": len(done), "win_pct": round(sum(1 for x in done if x > 0) / len(done) * 100, 1),
            "avg_r": round(float(np.mean(done)), 2)}


def time_machine(h1):
    a = atr(h1)
    n = len(h1)
    split = int(n * 2 / 3)
    rules = rule_signals(h1)
    report = {}
    best_live = None
    for name, (buy, sell) in rules.items():
        old = stats(simulate(h1, buy, sell, a, 200, split))
        new_res = simulate(h1, buy, sell, a, split, None)
        new = stats(new_res)
        proven = old["trades"] >= 8 and old["avg_r"] > 0.05 and new["trades"] >= 4 and new["avg_r"] > 0
        report[name] = {"older": old, "newer": new, "proven": proven}
        if not proven:
            continue
        # a "live" signal = the last trade is still running, or a new signal in the last 24 hours
        live = None
        if new_res and new_res[-1][2] is None and n - 1 - new_res[-1][0] <= 24:
            live = new_res[-1][1]
        else:
            recent_b = bool(buy.iloc[-24:].fillna(False).any())
            recent_s = bool(sell.iloc[-24:].fillna(False).any())
            if recent_b != recent_s:
                live = "BUY" if recent_b else "SELL"
        if live:
            score = (old["avg_r"] + new["avg_r"]) / 2
            if best_live is None or score > best_live[0]:
                best_live = (score, name, live, old, new)
    if best_live:
        score, name, side, old, new = best_live
        conf = int(max(5, min(8, round(5 + score * 6))))
        return {"vote": side, "confidence": conf,
                "reason": (f"Proven rule '{name}' has a live {side} signal. Older data: {old['trades']} trades, "
                           f"won {old['win_pct']}%, {old['avg_r']:+} x risk each. Newer data: {new['trades']} trades, "
                           f"won {new['win_pct']}%, {new['avg_r']:+} x risk each."),
                "wrong_if": "price hits the 1.5 ATR stop-loss"}, report
    proven_names = [k for k, v in report.items() if v["proven"]]
    return {"vote": "WAIT", "confidence": 3,
            "reason": (f"Proven rules: {', '.join(proven_names) if proven_names else 'none'}; "
                       f"none has a live signal now." if proven_names else
                       "No rule passed the test on both older and newer data, so I do not vote."),
            "wrong_if": ""}, report


# =====================================================================
# boss: the counting rule
# =====================================================================
def boss(name, votes, state):
    count = {"BUY": 0, "SELL": 0, "WAIT": 0}
    for v in votes.values():
        count[v["vote"]] += 1
    side = "BUY" if count["BUY"] > count["SELL"] else "SELL" if count["SELL"] > count["BUY"] else "MIXED"
    team = [a for a, v in votes.items() if v["vote"] == side]
    opposite = count["SELL"] if side == "BUY" else count["BUY"] if side == "SELL" else 0
    w = lambda a: state["weights"].get(f"{name}:{a}", 1.0)
    total_w = sum(w(a) for a in team)
    conf = sum(votes[a]["confidence"] * w(a) for a in team) / total_w if total_w else 0.0
    changed = [a for a in team if votes[a].get("inspected")]
    core_ok = len(team) - len(changed) >= MIN_SIDE - MAX_CHANGED_IN_TRADE
    go = side in ("BUY", "SELL") and len(team) >= MIN_SIDE and opposite <= MAX_OPPOSITE and conf >= MIN_CONF and core_ok
    if side == "MIXED":
        why = "BUY and SELL votes are equal"
    elif len(team) < MIN_SIDE:
        why = f"only {len(team)} agents say {side}, need {MIN_SIDE}"
    elif opposite > MAX_OPPOSITE:
        why = f"{opposite} agents say the opposite, at most {MAX_OPPOSITE} allowed"
    elif conf < MIN_CONF:
        why = f"average confidence {conf:.1f} is below {MIN_CONF}"
    elif not core_ok:
        why = (f"the {side} count rests on {len(changed)} votes changed by the AI inspector, "
               f"at most {MAX_CHANGED_IN_TRADE} allowed")
    else:
        why = "rule passed"
    return {"side": side, "count": count, "agree": len(team), "opposite": opposite,
            "weighted_conf": round(conf, 1), "go": go, "why": why}


# =====================================================================
# Telegram
# =====================================================================
def send(text):
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > 3800:
            chunks.append(line[:3800]); line = line[3800:]
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    for ch in chunks:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": ch}, timeout=20)
            time.sleep(1)
        except Exception as e:
            log(f"Telegram failed: {e}")


def weekly_report(state):
    lines = ["WEEKLY AGENT REPORT CARD", "(right = target hit before stop-loss)"]
    for key, hist in sorted(state["memory"].items()):
        acc = round(sum(h["right"] for h in hist) / len(hist) * 100)
        lines.append(f"{key}: right {acc}% of last {len(hist)} calls, weight {state['weights'].get(key, 1.0)}")
    if len(lines) == 2:
        lines.append("Not enough checked calls yet.")
    closed = [t for t in state["trades"] if t["status"] != "open"]
    if closed:
        wins = sum(1 for t in closed if (t.get("r") or 0) > 0)
        total = round(sum(t.get("r") or 0 for t in closed), 2)
        lines.append(f"\nPaper trades closed: {len(closed)}, winners {wins}, total {total:+} x risk")
    send("\n".join(lines))


# =====================================================================
# main
# =====================================================================
LABELS = {"trend": "Trend", "momentum": "Momentum", "levels": "Support/Resistance", "volatility": "Volatility",
          "daily_view": "Big picture (daily)", "news": "News", "mood": "Market mood", "macro": "Economy (macro)",
          "special": "Market specialist", "bull": "Bull (buy case)", "bear": "Bear (sell case)",
          "time_machine": "Time machine"}


def main():
    log("start")
    state = load_state()
    macro = get_macro()

    data, prices = {}, {}
    for name, t in MARKETS.items():
        data[name] = {"h1": get_data(t, "60d", "1h"), "d1": get_data(t, "2y", "1d")}
        if len(data[name]["h1"]):
            prices[name] = float(data[name]["h1"].Close.iloc[-1])

    update_report_card(state, data)
    update_trades(state, data)
    day_r = todays_result_r(state, prices)
    open_trades = [t for t in state["trades"] if t["status"] == "open"]

    # ---- step 1: gather everything and ask every agent once ----
    work = {}
    for name, t in MARKETS.items():
        try:
            h1, d1 = data[name]["h1"], data[name]["d1"]
            if len(h1) < 300 or len(d1) < 210:
                work[name] = {"error": "Not enough price data this hour."}
                continue
            log(f"--- {name} ---")
            phase = market_phase(h1, d1)
            extra = get_extra(name, state, macro)
            news = get_news(NEWS_TICKERS[name])
            facts, summary, raw = build_facts(name, h1, d1, macro, extra, news, phase)
            votes = {}
            for agent in ANALYSTS:
                v = clean_vote(ask_gemini(analyst_prompt(name, agent, facts, state)))
                log(f"{name} {agent}: {(v or {}).get('vote', 'no answer')}")
                if v:
                    votes[agent] = v
                time.sleep(PAUSE_BETWEEN_CALLS)
            work[name] = {"phase": phase, "extra": extra, "news": news, "facts": facts, "summary": summary,
                          "raw": raw, "votes": votes, "h1": h1}
        except Exception as e:
            log(f"{name} error: {e}")
            work[name] = {"error": f"Error: {e}"}

    # ---- step 2: ask skipped analysts once more ----
    for name, w in work.items():
        if "error" in w:
            continue
        for agent in ANALYSTS:
            if agent not in w["votes"] and time_left():
                v = clean_vote(ask_gemini(analyst_prompt(name, agent, w["facts"], state)))
                if v:
                    w["votes"][agent] = v
                    log(f"{name} {agent}: answered on retry")
                time.sleep(PAUSE_BETWEEN_CALLS)

    # ---- step 3: bull, bear (with one retry each) ----
    for name, w in work.items():
        if "error" in w:
            continue
        for role, allowed in (("bull", ("BUY", "WAIT")), ("bear", ("SELL", "WAIT"))):
            for attempt in range(2):
                if role in w["votes"] or not time_left():
                    break
                bull_p, bear_p = debate_prompts(name, w["summary"], w["votes"])
                v = clean_vote(ask_gemini(bull_p if role == "bull" else bear_p), allowed=allowed)
                if v:
                    w["votes"][role] = v
                time.sleep(PAUSE_BETWEEN_CALLS)

    # ---- step 4: backups, quality check, time machine, decision, report ----
    now_ist = dt.datetime.now(IST)
    report = [f"AI TEAM REPORT {now_ist:%d %b %I:%M %p} India time",
              f"Open paper trades: {len(open_trades)} | Today's paper result: {day_r:+} x risk"]
    for t in open_trades:
        report.append(f"  open: {t['market']} {t['side']} from {t['entry']}, stop {t['stop']}, "
                      f"target {t['target1']}, now {open_trade_r(t, prices.get(t['market'])):+} x risk")

    for name, w in work.items():
        if "error" in w:
            report.append(f"\n===== {name} =====\n{w['error']}")
            continue
        try:
            votes, phase, raw, summary = w["votes"], w["phase"], w["raw"], w["summary"]
            for agent in ANALYSTS:
                if agent not in votes and agent not in ("news", "mood", "special"):
                    votes[agent] = rule_backup(agent, name, phase, raw, macro)
            votes["time_machine"], _ = time_machine(w["h1"])
            flags = quality_check(name, votes, phase, raw)                                   # code rules first
            ai_fixes, ai_notes, ai_overall = ai_inspector(name, votes, phase, raw, summary, w["facts"])
            time.sleep(PAUSE_BETWEEN_CALLS)

            b = boss(name, votes, state)
            price, a_now = raw["price"], raw["atr"]
            c = b["count"]
            block = [f"\n===== {name} {num(price):,} =====",
                     f"Phase: {phase['text']}",
                     f"Room: up {raw['room_up']} ATR | down {raw['room_down']} ATR (a trade needs {TARGET1_ATR})",
                     f"Votes: BUY {c['BUY']} | SELL {c['SELL']} | WAIT {c['WAIT']} (out of {len(votes)} agents)"]

            if b["go"]:
                stop, t1, t2 = levels_for(b["side"], price, a_now)
                plan = {"side": b["side"], "entry": num(price), "stop": num(stop),
                        "target1": num(t1), "target2": num(t2), "hourly_atr": num(a_now)}
                same = [x for x in open_trades if x["market"] == name]
                room = raw["room_up"] if b["side"] == "BUY" else raw["room_down"]
                if room < TARGET1_ATR:
                    block.append(f"Decision: NO TRADE - {b['side']} passed the vote, but there is only {room} ATR of room "
                                 f"{'up' if b['side'] == 'BUY' else 'down'} to the next level (target needs {TARGET1_ATR}).")
                elif same:
                    block.append(f"Decision: NO NEW TRADE - already in a {same[0]['side']} paper trade on {name}.")
                elif day_r <= -MAX_DAILY_LOSS_R:
                    block.append(f"Decision: NO NEW TRADE - daily loss limit reached ({day_r} x risk).")
                else:
                    risk = ask_gemini(RISK_PROMPT.format(name=name, plan=jdump(plan), summary=jdump(summary),
                                                         news=jdump(w["news"][:6]), open_trades=jdump(open_trades))) \
                        or {"approve": False, "reason": "risk officer did not answer"}
                    time.sleep(PAUSE_BETWEEN_CALLS)
                    pm = ask_gemini(PM_PROMPT.format(name=name, plan=jdump(plan), phase=phase["text"], count=jdump(c),
                                                     bull=jdump(votes.get("bull")), bear=jdump(votes.get("bear")),
                                                     risk=jdump(risk))) \
                        or {"final": "SKIP", "reason": "desk head did not answer"}
                    approved = str(risk.get("approve")).strip().lower() in ("true", "yes", "1")
                    if approved and str(pm.get("final", "")).strip().upper() == "TAKE":
                        trade = {"market": name, "time": time.time(), "status": "open", **plan}
                        state["trades"].append(trade)
                        open_trades.append(trade)
                        block += [f"SIGNAL: {plan['side']} (paper trade)",
                                  f"Entry {plan['entry']} | Stop-loss {plan['stop']} | Target 1 {plan['target1']} | Target 2 {plan['target2']}",
                                  f"Risk officer: {risk.get('reason')}", f"Desk head: {pm.get('reason')}"]
                    else:
                        block += [f"Decision: NO TRADE - {b['side']} passed the vote but was blocked.",
                                  f"Risk officer: {risk.get('reason')}", f"Desk head: {pm.get('reason')}"]
            else:
                block.append(f"Decision: NO TRADE - {b['why']}.")

            block.append("What each agent said:")
            for agent in ALL_VOTERS:
                if agent in votes:
                    v = votes[agent]
                    wt = state["weights"].get(f"{name}:{agent}")
                    line = f"- {LABELS[agent]}: {v['vote']} ({v['confidence']}/10){f' [weight {wt}]' if wt else ''} {v['reason']}"
                    if v.get("wrong_if") and v["vote"] != "WAIT":
                        line += f" | Wrong if: {v['wrong_if']}"
                    block.append(line)
                else:
                    block.append(f"- {LABELS[agent]}: no answer (Gemini busy)")
            block.append("Code rules: " + ("; ".join(flags) if flags else "all chart votes follow their rules."))
            block.append("AI inspector: " + ("; ".join(ai_fixes) if ai_fixes else "no vote removed.")
                         + (f" Warnings: {'; '.join(ai_notes)}." if ai_notes else "")
                         + (f" Overall: {ai_overall}" if ai_overall else ""))
            if name == "BTC" and not w["extra"].get("derivatives", {}).get("source"):
                block.append("Note: BTC futures data was not available this hour.")
            report += block

            state["votes_log"].append({"time": time.time(), "market": name, "price": num(price),
                                       "atr": num(a_now, 4),
                                       "votes": {a: v.get("orig", v["vote"]) for a, v in votes.items() if not v.get("backup")},
                                       "inspector_removed": {a: v["removed_side"] for a, v in votes.items() if v.get("removed_side")}})
        except Exception as e:
            log(f"{name} error: {e}")
            report.append(f"\n===== {name} =====\nError: {e}")

    report.append(f"\nGemini: {STATS['answered']} of {STATS['asked']} questions answered, "
                  f"busy {STATS['busy']} times. Run time {int(time.time() - START)}s")
    for err in STATS["errors"][:2]:
        report.append(err)
    report.append("\nPaper trades only. Not financial advice.")
    send("\n".join(report))

    now = dt.datetime.now(dt.timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if now.weekday() == 0 and now.hour >= 2 and state.get("last_weekly") != today:
        weekly_report(state)
        state["last_weekly"] = today
    save_state(state)
    log("done")


if __name__ == "__main__":
    main()
