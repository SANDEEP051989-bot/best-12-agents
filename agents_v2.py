# agents_v2.py  -  version 3  "Veteran Desk"
# 12 AI agents, each written as a 30-year veteran trader, for BTC, Gold and US Oil.
# Brain: Gemini.  Reports: Telegram.  Server: GitHub Actions.
#
# What is new in version 3 (the 8 fixes):
#  1. Full reasons in the report (nothing cut off).
#  2. New trade rule: at least 6 agents on one side and at most 1 on the other side.
#  3. Every agent gets ONLY the data for its own job, so they stop repeating each other.
#  4. Agents must pick BUY or SELL when evidence leans 55/45; WAIT only when truly balanced.
#  5. Bull only argues BUY, Bear only argues SELL (each may say WAIT if its case is weak).
#  6. BTC futures data (funding rate, open interest) from 3 sources: Binance, OKX, Deribit.
#  7. Time machine tests 4 real trading rules with stop-loss and target, uses the best one.
#  8. Report card checks what was hit first: stop-loss or target.
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
NEWS_EXTRA = {"BTC": "IBIT", "GOLD": "GLD", "OIL": "USO"}
ROUND_STEP = {"BTC": 1000, "GOLD": 50, "OIL": 1}

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
    """Turn numpy / pandas numbers into plain rounded numbers (or None)."""
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
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
        except Exception:
            s = {}
    else:
        s = {}
    s.setdefault("votes_log", [])
    s.setdefault("trades", [])
    s.setdefault("weights", {})
    s.setdefault("memory", {})
    s.setdefault("last_oi", {})
    s.pop("signals", None)          # old version 2 data is not used any more
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
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
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
    return out[:12]


def get_macro():
    out = {}
    for name, t in {"dollar_index": "DX-Y.NYB", "us10y_yield": "^TNX",
                    "sp500": "^GSPC", "vix_fear_index": "^VIX"}.items():
        last, ch = last_change(t, 5)
        if last is not None:
            out[name] = {"last": last, "change_5d_pct": ch}
    return out


def get_btc_derivatives(state):
    """Funding rate and open interest. Binance often blocks GitHub (USA) servers,
    so OKX and Deribit are tried as well."""
    out = {"source": None}
    # 1) Binance
    try:
        f = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=10)
        o = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT", timeout=10)
        if f.status_code == 200 and o.status_code == 200:
            out.update(source="binance",
                       funding_rate_pct_8h=num(float(f.json()["lastFundingRate"]) * 100, 4),
                       open_interest_btc=num(o.json()["openInterest"], 0))
    except Exception:
        pass
    # 2) OKX
    if not out["source"]:
        try:
            f = requests.get("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP", timeout=10)
            o = requests.get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP", timeout=10)
            fd, od = f.json().get("data") or [], o.json().get("data") or []
            if fd and od:
                out.update(source="okx",
                           funding_rate_pct_8h=num(float(fd[0]["fundingRate"]) * 100, 4),
                           open_interest_btc=num(od[0].get("oiCcy"), 0))
        except Exception:
            pass
    # 3) Deribit
    if not out["source"]:
        try:
            r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"
                             "?instrument_name=BTC-PERPETUAL", timeout=10).json()
            res = (r.get("result") or [{}])[0]
            if res:
                price = res.get("mark_price") or res.get("last") or 0
                oi_usd = res.get("open_interest")
                out.update(source="deribit",
                           funding_rate_pct_8h=num((res.get("funding_8h") or 0) * 100, 4),
                           open_interest_btc=num(oi_usd / price, 0) if oi_usd and price else None)
        except Exception:
            pass
    if out["source"] and out.get("open_interest_btc"):
        prev = state["last_oi"].get(out["source"])
        out["open_interest_change_since_last_hour_pct"] = pct(out["open_interest_btc"], prev) if prev else None
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
    if len(df) < 25 or df.Volume.tail(20).mean() == 0:
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
    return pd.concat([df.High - df.Low,
                      (df.High - df.Close.shift()).abs(),
                      (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).rolling(n).mean()


def adx(df, n=14):
    up = df.High.diff()
    dn = -df.Low.diff()
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


def swings(df, k=5, count=5):
    """Recent swing highs and lows (turning points)."""
    hi = df.High[(df.High == df.High.rolling(2 * k + 1, center=True).max())]
    lo = df.Low[(df.Low == df.Low.rolling(2 * k + 1, center=True).min())]
    return [num(x) for x in hi.tail(count)], [num(x) for x in lo.tail(count)]


def volume_poc(df, bars=200, bins=30):
    sub = df.tail(bars)
    if len(sub) < 20 or sub.Volume.sum() <= 0:
        return None
    hist, edges = np.histogram(sub.Close, bins=bins, weights=sub.Volume)
    i = int(np.argmax(hist))
    return num((edges[i] + edges[i + 1]) / 2)


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
# facts: each agent gets only its own slice
# =====================================================================
def build_facts(name, h1, d1, macro, extra, news):
    c, dc = h1.Close, d1.Close
    price = float(c.iloc[-1])
    a_h1, a_d1 = atr(h1), atr(d1)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    de20, de50, de200 = ema(dc, 20), ema(dc, 50), ema(dc, 200)
    adx_h, pdi_h, mdi_h = adx(h1)
    adx_d, pdi_d, mdi_d = adx(d1)
    m_h, s_h, hist_h = macd(c)
    m_d, s_d, hist_d = macd(dc)
    r_h, r_d = rsi(c), rsi(dc)
    sh_h, sl_h = swings(h1)
    sh_d, sl_d = swings(d1, k=3, count=4)
    prev_day = d1.iloc[-2] if len(d1) > 1 else d1.iloc[-1]
    pivot = (prev_day.High + prev_day.Low + prev_day.Close) / 3
    step = ROUND_STEP.get(name, 1)
    year = d1.tail(252)
    vol_avg = h1.Volume.rolling(20).mean().iloc[-1]
    vol_ratio = num(h1.Volume.iloc[-1] / vol_avg) if vol_avg and vol_avg > 0 else None
    bb_mid = c.rolling(20).mean()
    bb_width = (4 * c.rolling(20).std()) / bb_mid * 100
    range_24h = float(h1.High.tail(24).max() - h1.Low.tail(24).min())
    weekly = dc.resample("W").last().dropna().tail(6)

    levels_above = sorted([x for x in sh_h + sh_d if x and x > price])
    levels_below = sorted([x for x in sl_h + sl_d if x and x < price], reverse=True)

    base = {"market": name, "price": num(price), "change_24h_pct": pct(price, c.iloc[-25]) if len(c) > 25 else None}

    facts = {
        "trend": {**base,
                  "hourly": {"ema20": num(e20.iloc[-1]), "ema50": num(e50.iloc[-1]), "ema200": num(e200.iloc[-1]),
                             "ema20_slope_10bars_pct": pct(e20.iloc[-1], e20.iloc[-11]),
                             "adx": num(adx_h.iloc[-1], 1), "plus_di": num(pdi_h.iloc[-1], 1), "minus_di": num(mdi_h.iloc[-1], 1),
                             "macd": num(m_h.iloc[-1], 3), "macd_signal": num(s_h.iloc[-1], 3),
                             "recent_swing_highs": sh_h, "recent_swing_lows": sl_h},
                  "daily": {"ema20": num(de20.iloc[-1]), "ema50": num(de50.iloc[-1]), "ema200": num(de200.iloc[-1]),
                            "adx": num(adx_d.iloc[-1], 1), "plus_di": num(pdi_d.iloc[-1], 1), "minus_di": num(mdi_d.iloc[-1], 1)}},
        "momentum": {**base,
                     "rsi_hourly": num(r_h.iloc[-1], 1), "rsi_hourly_6_bars_ago": num(r_h.iloc[-7], 1),
                     "rsi_daily": num(r_d.iloc[-1], 1),
                     "change_6h_pct": pct(price, c.iloc[-7]), "change_5d_pct": pct(dc.iloc[-1], dc.iloc[-6]) if len(dc) > 6 else None,
                     "macd_histogram_now": num(hist_h.iloc[-1], 3), "macd_histogram_3_bars_ago": num(hist_h.iloc[-4], 3),
                     "volume_vs_20bar_average": vol_ratio,
                     "rsi_divergence_hourly": divergence(c, r_h)},
        "levels": {**base,
                   "nearest_resistances_above": levels_above[:3], "nearest_supports_below": levels_below[:3],
                   "previous_day_high": num(prev_day.High), "previous_day_low": num(prev_day.Low),
                   "pivot": num(pivot), "pivot_r1": num(2 * pivot - prev_day.Low), "pivot_s1": num(2 * pivot - prev_day.High),
                   "hourly_50bar_high": num(h1.High.tail(50).max()), "hourly_50bar_low": num(h1.Low.tail(50).min()),
                   "daily_20day_high": num(d1.High.tail(20).max()), "daily_20day_low": num(d1.Low.tail(20).min()),
                   "high_52_week": num(year.High.max()), "low_52_week": num(year.Low.min()),
                   "daily_ema200": num(de200.iloc[-1]), "volume_point_of_control_200h": volume_poc(h1),
                   "round_number_below": num((price // step) * step), "round_number_above": num((price // step + 1) * step),
                   "hourly_atr": num(a_h1.iloc[-1])},
        "volatility": {**base,
                       "hourly_atr": num(a_h1.iloc[-1]), "hourly_atr_pct_of_price": pct(price + a_h1.iloc[-1], price),
                       "hourly_atr_percentile_500bars": percentile_rank(a_h1),
                       "daily_atr": num(a_d1.iloc[-1]), "range_last_24h": num(range_24h),
                       "range_24h_vs_daily_atr": num(range_24h / a_d1.iloc[-1]) if a_d1.iloc[-1] else None,
                       "bollinger_width_pct": num(bb_width.iloc[-1]), "bollinger_width_percentile": percentile_rank(bb_width),
                       "hourly_trend_direction": "up" if e20.iloc[-1] > e50.iloc[-1] else "down",
                       "daily_trend_direction": "up" if de50.iloc[-1] > de200.iloc[-1] else "down"},
        "daily_view": {**base,
                       "daily_close": num(dc.iloc[-1]), "daily_ema20": num(de20.iloc[-1]), "daily_ema50": num(de50.iloc[-1]),
                       "daily_ema200": num(de200.iloc[-1]), "daily_rsi": num(r_d.iloc[-1], 1), "daily_adx": num(adx_d.iloc[-1], 1),
                       "daily_macd_histogram": num(hist_d.iloc[-1], 3),
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
    summary = {**base, "hourly_atr": num(a_h1.iloc[-1]),
               "hourly_trend": "up" if e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1] else
                               "down" if e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1] else "mixed",
               "daily_trend": "up" if dc.iloc[-1] > de50.iloc[-1] > de200.iloc[-1] else
                              "down" if dc.iloc[-1] < de50.iloc[-1] < de200.iloc[-1] else "mixed",
               "rsi_hourly": num(r_h.iloc[-1], 1), "rsi_daily": num(r_d.iloc[-1], 1),
               "nearest_resistance": levels_above[0] if levels_above else None,
               "nearest_support": levels_below[0] if levels_below else None,
               "change_5d_pct": pct(dc.iloc[-1], dc.iloc[-6]) if len(dc) > 6 else None}
    return facts, summary


def get_extra(name, state, macro):
    if name == "BTC":
        eth_last, eth_5d = last_change("ETH-USD", 5)
        _, btc_5d = last_change("BTC-USD", 5)
        return {"derivatives": get_btc_derivatives(state), "fear_greed": get_fear_greed(),
                "eth_5d_change_pct": eth_5d, "btc_5d_change_pct": btc_5d,
                "eth_vs_btc_5d_pct": num(eth_5d - btc_5d) if eth_5d is not None and btc_5d is not None else None,
                "bitcoin_etf_IBIT_volume_vs_20day_avg": volume_ratio_daily("IBIT")}
    if name == "GOLD":
        si_last, si_5d = last_change("SI=F", 5)
        gc_last, gc_5d = last_change("GC=F", 5)
        _, tip_5d = last_change("TIP", 5)
        return {"silver_5d_change_pct": si_5d, "gold_5d_change_pct": gc_5d,
                "gold_silver_ratio": num(gc_last / si_last) if gc_last and si_last else None,
                "TIP_inflation_bond_5d_change_pct (up = real yields falling)": tip_5d,
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
                "today_utc_weekday": now.strftime("%A"),
                "us_eia_weekly_oil_report": "today (Wednesday 14:30 UTC)" if now.weekday() == 2 else "on Wednesday",
                "oil_supply_headlines": heads[:8], "dollar_index": macro.get("dollar_index")}
    return {}


# =====================================================================
# the 30-year veterans
# =====================================================================
DESK_RULES = """
You sit on a 12-member professional trading desk. Trades are held from a few hours to 2 days.
Stop-loss = 1.5 x hourly ATR, first target = 2 x hourly ATR.

How you vote:
- Judge ONLY from YOUR data below. Other desk members cover the other areas; do not do their job.
- Vote BUY or SELL whenever your evidence leans at least 55/45 one way. Show strength with confidence:
  5 = slight lean, 6-7 = clear edge, 8-9 = strong, 10 = almost never.
- Vote WAIT only if your evidence is truly balanced or your data is missing.
  "The market is uncertain" or "be careful" is NOT a reason to wait - markets are always uncertain.
- A big fall is not automatically a BUY and a big rise is not automatically a SELL.
  Go with the trend unless YOUR data shows real exhaustion or a turn.
- Your reason: 1 to 2 short sentences in simple English (max 45 words) that name the exact numbers you used.

Reply ONLY as JSON: {"vote": "BUY" or "SELL" or "WAIT", "confidence": 1-10, "reason": "..."}
"""

EXPERTS = {
    "trend": """You are a trend-following trader with 30 years on futures and currency desks, trained in the Turtle-trading school.
Your checklist:
1) Daily trend: price vs daily EMA50 and EMA200, and EMA50 vs EMA200.
2) Hourly trend: is EMA20 > EMA50 > EMA200 (up) or the reverse (down)? Is EMA20 rising or falling?
3) Trend strength: ADX above 25 = real trend, below 20 = no trend. +DI above -DI = buyers in control.
4) Structure: higher swing highs and lows = uptrend; lower highs and lows = downtrend.
5) MACD above or below its signal line.
Your rules from experience: the trend is your friend until it ends. Trade with the daily trend and use the hourly to time it.
When daily and hourly disagree, the daily wins but your confidence drops. In a strong downtrend (ADX > 25, -DI on top)
you SELL rallies; you never buy just because the price looks cheap. In a strong uptrend you BUY dips.""",

    "momentum": """You are a momentum and oscillator specialist with 30 years of trading, a student of Wilder and Murphy.
Your checklist:
1) RSI regime: in uptrends RSI holds 40-80, in downtrends 20-60. RSI above 50 and rising = buyers gaining.
2) RSI now vs 6 bars ago: is momentum speeding up or slowing down?
3) Price change over 6 hours and 5 days: accelerating or fading?
4) MACD histogram now vs 3 bars ago: growing in the move's direction = momentum building.
5) Volume vs average: real moves come with volume.
6) Divergence: an early warning only, it needs confirmation from a turning histogram.
Your rules from experience: oversold can stay oversold in a crash, so RSI under 30 in a downtrend is NOT a buy by itself.
Buy when momentum turns up from low levels with the histogram rising; sell when it rolls over from high levels.
Momentum usually turns before price does.""",

    "levels": """You are a price-action trader with 30 years of experience, starting on the trading floor, expert in support,
resistance and the zones where big institutions place orders: previous day high and low, pivot points, round numbers,
the volume point of control, 52-week extremes and the daily EMA200.
Your checklist:
1) Where is price vs the nearest support below and nearest resistance above?
2) How much room is there to the next level, measured in hourly ATR? A trade needs at least 2 ATR of room.
3) Is price bouncing from a level, rejecting it, or breaking through it?
4) A broken support becomes resistance and a broken resistance becomes support.
Your rules from experience: buy near strong support with room above; sell near strong resistance with room below.
Never buy just under resistance and never sell just above support. A clean break and hold beyond a level flips the trade.""",

    "volatility": """You are a volatility and risk-regime specialist with 30 years on options and volatility desks.
Your checklist:
1) Hourly ATR percentile: above 80 = very volatile, below 20 = very quiet.
2) Bollinger width percentile: very low = squeeze, a big move is coming, usually in the direction of the higher trend.
3) Last 24h range vs daily ATR: above 1.5 = stretched move, follow-through is less likely; above 2.5 = extreme.
4) Hourly and daily trend direction.
Your rules from experience: volatility expanding in the direction of the trend tends to continue - vote that direction.
After an extreme range day, a snap-back against the move is more likely - vote against the move with modest confidence.
In a squeeze, vote the direction of the daily trend. Say WAIT only when volatility is so extreme
(ATR percentile above 95 and range above 2.5x daily ATR) that a 1.5 ATR stop would be hit by noise.""",

    "daily_view": """You are a position trader with 30 years of experience who only reads weekly and daily charts,
in the style of the great swing traders. You ignore hourly noise.
Your checklist:
1) Last 6 weekly closes: rising, falling or flat?
2) Daily close vs EMA20, EMA50, EMA200.
3) Position in the 52-week range: near the top or near the bottom?
4) Performance over 20 and 60 days, daily RSI, daily ADX and MACD histogram.
Your rules from experience: the higher timeframe sets the direction. Buy dips in daily uptrends, sell rallies in daily
downtrends. A market near its 52-week low in a downtrend usually goes lower before it turns.
A market making new highs in an uptrend usually goes higher.""",

    "news": """You are a market news analyst with 30 years of experience, formerly running a Reuters and Bloomberg
commodities and crypto news desk.
Your checklist:
1) Mark each headline bullish, bearish or neutral for THIS market.
2) Weigh importance: central banks, wars, supply shocks, regulation and ETF flows matter most; opinion pieces matter little.
3) Has price already reacted? Compare with the 24h change.
Your rules from experience: buy the rumour, sell the news. Bad news that price ignores is bullish; good news that price
ignores is bearish. If there are no headlines or none are relevant, vote WAIT with confidence 3.""",

    "mood": """You are a sentiment and positioning specialist with 30 years of experience, a disciplined contrarian.
Your checklist:
1) VIX fear index: above 25 = fear, below 15 = complacency; and is it rising or falling?
2) For Bitcoin: Crypto Fear and Greed index (below 25 extreme fear, above 75 extreme greed) and funding rate
   (above 0.03% per 8h = crowded longs, below 0 = crowded shorts).
3) Daily RSI, 5-day change and volume: is the crowd panicking or chasing?
Your rules from experience: extreme fear after a big fall, with selling volume drying up, is a contrarian BUY.
Extreme greed after a big rise is a contrarian SELL. When sentiment is in the middle, go with the crowd's direction.""",

    "macro": """You are a global macro strategist with 30 years at a large macro hedge fund.
How each market reacts:
- Bitcoin trades like a high-risk tech stock: a falling dollar, falling yields and rising stocks help it; the opposite hurts.
- Gold likes a falling dollar, falling real yields and fear; it dislikes rising yields and a strong dollar.
- Oil likes strong growth and rising stocks; recession fear hurts it; a strong dollar hurts slightly.
Your checklist: 5-day change of the dollar index, US 10-year yield, S&P 500 and VIX. Decide whether the macro wind is
a tailwind or a headwind for THIS market, and how strong.""",

    "special_BTC": """You are a crypto derivatives trader with 30 years in derivatives, the last 14 of them in Bitcoin.
Your checklist:
1) Funding rate: above 0.03% per 8h = longs crowded (risk of a drop), negative = shorts crowded (risk of a squeeze up).
2) Open interest change vs price change: price down + OI up = new shorts (bearish); price down + OI down = longs
   giving up, selling may be ending; price up + OI up = new longs (bullish); price up + OI down = short covering (weaker).
3) Fear and Greed index today vs yesterday.
4) ETH vs BTC over 5 days: ETH stronger = risk appetite returning.
5) Bitcoin ETF (IBIT) volume vs average: a spike shows big money acting.
If futures data is missing, judge from the other items; do not wait just because one item is missing.""",

    "special_GOLD": """You are a bullion dealer with 30 years at a London bullion bank.
Your checklist:
1) Dollar index and US 10-year yield over 5 days: both falling = gold tailwind.
2) TIP inflation-bond ETF: rising = real yields falling = good for gold.
3) Silver vs gold over 5 days: silver confirming gold = a genuine move; silver lagging = weak move.
4) Gold/silver ratio: rising = fear and defence.
5) GLD ETF volume vs average: a spike shows investors moving.
Your rules from experience: gold hates rising real yields and a strong dollar. Central-bank buying supports dips.""",

    "special_OIL": """You are a crude oil trader with 30 years in physical and futures oil markets.
Your checklist:
1) Supply headlines: OPEC+ cuts or increases, sanctions, wars, hurricanes, refinery problems.
2) US EIA weekly report (Wednesday): stock draws are bullish, builds are bearish. Big event risk on report day.
3) Brent minus WTI spread: widening = tighter global supply.
4) Energy stocks (XLE) over 5 days: they often lead crude; disagreement is a warning.
5) Dollar index.
Your rules from experience: supply shocks drive the big moves. When energy stocks and crude disagree, trust the stocks.""",
}

BULL_PROMPT = """You are a senior long-only portfolio manager with 30 years of experience.
Build the strongest HONEST case to BUY {name} right now, using the desk's opinions and the key facts.
If the long case is genuinely good (you would put your own money in), vote BUY and give confidence.
If it is weak, vote WAIT and say what is missing. You never vote SELL.
Reason: 1-2 short simple-English sentences (max 45 words) with exact numbers.
Desk opinions: {votes}
Key facts: {summary}
Reply ONLY as JSON: {{"vote": "BUY" or "WAIT", "confidence": 1-10, "reason": "..."}}"""

BEAR_PROMPT = """You are a senior short-seller with 30 years of experience.
Build the strongest HONEST case to SELL {name} right now, using the desk's opinions and the key facts.
A market below its major moving averages with falling momentum usually gives a strong short case - say so clearly.
If the short case is genuinely good, vote SELL and give confidence. If it is weak, vote WAIT and say what is missing.
You never vote BUY.
Reason: 1-2 short simple-English sentences (max 45 words) with exact numbers.
Desk opinions: {votes}
Key facts: {summary}
Reply ONLY as JSON: {{"vote": "SELL" or "WAIT", "confidence": 1-10, "reason": "..."}}"""

RISK_PROMPT = """You are the chief risk officer of a trading desk, 30 years of experience.
Proposed paper trade on {name}: {plan}
Key facts: {summary}
Headlines: {news}
Open paper trades: {open_trades}
Approve only if ALL are true:
1) The stop-loss is not sitting inside obvious noise (it is beyond, or not right next to, the nearest support for a BUY /
   resistance for a SELL).
2) There is room to the first target: the nearest opposing level is not closer than the first target.
3) Volatility is not extreme and there is no major event in the next few hours (central bank, EIA report, big data).
4) It does not double the same bet already open.
Reason: 1-2 short simple-English sentences (max 45 words).
Reply ONLY as JSON: {{"approve": true or false, "reason": "..."}}"""

PM_PROMPT = """You are the head of the trading desk, 30 years of experience. You give the final yes or no.
Trade on {name}: {plan}
Desk vote count: {count}
Bull view: {bull}
Bear view: {bear}
Risk officer: {risk}
Take the trade only if you would show it proudly to your investors.
Reason: 1-2 short simple-English sentences (max 45 words).
Reply ONLY as JSON: {{"final": "TAKE" or "SKIP", "reason": "..."}}"""

ANALYSTS = ["trend", "momentum", "levels", "volatility", "daily_view", "news", "mood", "macro", "special"]
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
    return {"vote": v, "confidence": max(1, min(10, c)), "reason": str(ans.get("reason", "")).strip()}


def memory_note(state, name, agent):
    hist = state["memory"].get(f"{name}:{agent}", [])[-10:]
    if not hist:
        return "No checked calls yet."
    right = sum(1 for h in hist if h["right"])
    last = ", ".join(f"{h['said']} {'right' if h['right'] else 'wrong'}" for h in hist[-5:])
    return (f"Your last {len(hist)} checked calls: {right} right, {len(hist) - right} wrong "
            f"(judged by whether target or stop-loss was hit first). Most recent: {last}. "
            f"Learn from this: if you are often wrong, look harder at what you missed.")


def run_analysts(name, facts, state):
    votes = {}
    for agent in ANALYSTS:
        if not time_left():
            break
        expert = EXPERTS[f"special_{name}"] if agent == "special" else EXPERTS[agent]
        prompt = (f"{expert}\n{DESK_RULES}\nMarket: {name}\n"
                  f"Your track record: {memory_note(state, name, agent)}\n"
                  f"YOUR DATA: {jdump(facts[agent])}")
        v = clean_vote(ask_gemini(prompt))
        log(f"{name} {agent}: {(v or {}).get('vote', 'no answer')}")
        if v:
            votes[agent] = v
        time.sleep(PAUSE_BETWEEN_CALLS)
    return votes


def debate(name, summary, votes):
    short = {a: f"{v['vote']} {v['confidence']}/10: {v['reason']}" for a, v in votes.items()}
    bull = clean_vote(ask_gemini(BULL_PROMPT.format(name=name, votes=jdump(short), summary=jdump(summary))),
                      allowed=("BUY", "WAIT"))
    time.sleep(PAUSE_BETWEEN_CALLS)
    bear = clean_vote(ask_gemini(BEAR_PROMPT.format(name=name, votes=jdump(short), summary=jdump(summary))),
                      allowed=("SELL", "WAIT"))
    time.sleep(PAUSE_BETWEEN_CALLS)
    return bull, bear


# =====================================================================
# judging trades: what was hit first, stop-loss or target?
# =====================================================================
def first_touch(h1, side, entry, stop, target, start_ts, max_bars=MAX_HOLD_BARS):
    """Returns ("win" | "loss" | "expired" | "open", r_multiple)."""
    bars = h1[h1.index > start_ts].head(max_bars)
    risk = abs(entry - stop)
    for _, b in bars.iterrows():
        if side == "BUY":
            if b.Low <= stop:            # if both happen in one candle we assume the stop came first
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
    """Checks old votes: for every BUY or SELL vote, was the target or the stop-loss hit first?"""
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
        if "open" in results.values():
            if time.time() - rec["time"] < (MAX_HOLD_BARS + 24) * 3600:
                continue                      # not decided yet, check again next hour
        rec["checked"] = True
        for agent, said in rec["votes"].items():
            if said not in ("BUY", "SELL") or results[said] not in ("win", "loss"):
                continue                      # WAIT votes and undecided trades are not scored
            key = f"{rec['market']}:{agent}"
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
# time machine: tests 4 real trading rules on the last 60 days
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
    rules = {
        "trend pullback": ((e50 > e200) & (c > e20) & (c.shift(1) <= e20.shift(1)) & (r > 45),
                           (e50 < e200) & (c < e20) & (c.shift(1) >= e20.shift(1)) & (r < 55)),
        "20-bar breakout": ((c > hh) & vol_ok & (e50 > e200), (c < ll) & vol_ok & (e50 < e200)),
        "dip in uptrend": ((r < 32) & (c > e200), (r > 68) & (c < e200)),
        "MACD cross with trend": ((m > s) & (m.shift(1) <= s.shift(1)) & (c > e50),
                                  (m < s) & (m.shift(1) >= s.shift(1)) & (c < e50)),
    }
    return rules


def test_rule(h1, buy, sell, a):
    hi, lo, cl = h1.High.values, h1.Low.values, h1.Close.values
    b, s, av = buy.fillna(False).values, sell.fillna(False).values, a.values
    n, i, res = len(cl), 200, []
    while i < n - 1:
        side = "BUY" if b[i] else "SELL" if s[i] else None
        if side is None or not av[i] or np.isnan(av[i]):
            i += 1
            continue
        entry = cl[i]
        stop, t1, _ = levels_for(side, entry, av[i])
        risk = abs(entry - stop)
        j, out = i + 1, None
        while j < min(n, i + 1 + MAX_HOLD_BARS):
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
        if out is None:
            if j >= n:
                break                         # trade not finished yet, stop testing
            move = (cl[j - 1] - entry) if side == "BUY" else (entry - cl[j - 1])
            out = move / risk
        res.append(out)
        i = j + 1                              # one trade at a time
    if not res:
        return {"trades": 0, "win_pct": 0, "avg_r": 0}
    return {"trades": len(res), "win_pct": round(sum(1 for x in res if x > 0) / len(res) * 100, 1),
            "avg_r": round(float(np.mean(res)), 2)}


def time_machine(h1):
    a = atr(h1)
    rules = rule_signals(h1)
    tested = {name: test_rule(h1, buy, sell, a) for name, (buy, sell) in rules.items()}
    firing = []
    for name, (buy, sell) in rules.items():
        st = tested[name]
        side = "BUY" if bool(buy.iloc[-2:].fillna(False).any()) else "SELL" if bool(sell.iloc[-2:].fillna(False).any()) else None
        if side and st["trades"] >= 15 and st["avg_r"] > 0.05:
            firing.append((st["avg_r"], name, side, st))
    best_name = max(tested, key=lambda k: tested[k]["avg_r"] if tested[k]["trades"] >= 15 else -9)
    best = tested[best_name]
    if firing:
        avg_r, name, side, st = max(firing)
        conf = int(max(5, min(8, round(5 + avg_r * 6))))
        return {"vote": side, "confidence": conf,
                "reason": (f"Rule '{name}' is firing now. In the last 60 days it made {st['trades']} test trades, "
                           f"won {st['win_pct']}%, and earned {st['avg_r']:+} x risk per trade on average.")}, tested
    return {"vote": "WAIT", "confidence": 3,
            "reason": (f"No proven rule is firing now. Best rule '{best_name}': {best['trades']} test trades, "
                       f"won {best['win_pct']}%, {best['avg_r']:+} x risk per trade.")}, tested


# =====================================================================
# boss: the counting rule
# =====================================================================
def boss(name, votes, state):
    count = {"BUY": 0, "SELL": 0, "WAIT": 0}
    for v in votes.values():
        count[v["vote"]] += 1
    if count["BUY"] > count["SELL"]:
        side = "BUY"
    elif count["SELL"] > count["BUY"]:
        side = "SELL"
    else:
        side = "MIXED"
    team = [a for a, v in votes.items() if v["vote"] == side]
    opposite = count["SELL"] if side == "BUY" else count["BUY"] if side == "SELL" else 0
    w = lambda a: state["weights"].get(f"{name}:{a}", 1.0)
    total_w = sum(w(a) for a in team)
    conf = sum(votes[a]["confidence"] * w(a) for a in team) / total_w if total_w else 0.0
    go = side in ("BUY", "SELL") and len(team) >= MIN_SIDE and opposite <= MAX_OPPOSITE and conf >= MIN_CONF
    if side == "MIXED":
        why = "BUY and SELL votes are equal"
    elif len(team) < MIN_SIDE:
        why = f"only {len(team)} agents say {side}, need {MIN_SIDE}"
    elif opposite > MAX_OPPOSITE:
        why = f"{opposite} agents say the opposite, at most {MAX_OPPOSITE} allowed"
    elif conf < MIN_CONF:
        why = f"average confidence {conf:.1f} is below {MIN_CONF}"
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
    lines = ["WEEKLY AGENT REPORT CARD",
             "(right = target hit before stop-loss)"]
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
        h1, d1 = get_data(t, "60d", "1h"), get_data(t, "2y", "1d")
        data[name] = {"h1": h1, "d1": d1}
        if len(h1):
            prices[name] = float(h1.Close.iloc[-1])

    update_report_card(state, data)
    update_trades(state, data)
    day_r = todays_result_r(state, prices)
    open_trades = [t for t in state["trades"] if t["status"] == "open"]

    now_ist = dt.datetime.now(IST)
    report = [f"AI TEAM REPORT {now_ist:%d %b %I:%M %p} India time",
              f"Open paper trades: {len(open_trades)} | Today's paper result: {day_r:+} x risk"]
    for t in open_trades:
        report.append(f"  open: {t['market']} {t['side']} from {t['entry']}, stop {t['stop']}, "
                      f"target {t['target1']}, now {open_trade_r(t, prices.get(t['market'])):+} x risk")

    for name, t in MARKETS.items():
        try:
            log(f"--- {name} ---")
            h1, d1 = data[name]["h1"], data[name]["d1"]
            if len(h1) < 250 or len(d1) < 60:
                report.append(f"\n===== {name} =====\nNot enough price data this hour.")
                continue
            extra = get_extra(name, state, macro)
            news = get_news([t, NEWS_EXTRA[name]])
            facts, summary = build_facts(name, h1, d1, macro, extra, news)

            votes = run_analysts(name, facts, state)
            bull, bear = debate(name, summary, votes) if time_left() else (None, None)
            if bull:
                votes["bull"] = bull
            if bear:
                votes["bear"] = bear
            tm_vote, _ = time_machine(h1)
            votes["time_machine"] = tm_vote

            b = boss(name, votes, state)
            price = float(h1.Close.iloc[-1])
            a_now = float(atr(h1).iloc[-1])
            c = b["count"]
            block = [f"\n===== {name} {num(price):,} =====",
                     f"Votes: BUY {c['BUY']} | SELL {c['SELL']} | WAIT {c['WAIT']} (out of {len(votes)} agents)"]

            if b["go"]:
                stop, t1, t2 = levels_for(b["side"], price, a_now)
                plan = {"side": b["side"], "entry": num(price), "stop": num(stop),
                        "target1": num(t1), "target2": num(t2), "hourly_atr": num(a_now)}
                same_market_open = [x for x in open_trades if x["market"] == name]
                if same_market_open:
                    block.append(f"Decision: NO NEW TRADE - already in a {same_market_open[0]['side']} paper trade on {name}.")
                elif day_r <= -MAX_DAILY_LOSS_R:
                    block.append(f"Decision: NO NEW TRADE - daily loss limit reached ({day_r} x risk).")
                else:
                    risk = ask_gemini(RISK_PROMPT.format(name=name, plan=jdump(plan), summary=jdump(summary),
                                                         news=jdump(news[:6]), open_trades=jdump(open_trades))) \
                        or {"approve": False, "reason": "risk officer did not answer"}
                    time.sleep(PAUSE_BETWEEN_CALLS)
                    pm = ask_gemini(PM_PROMPT.format(name=name, plan=jdump(plan), count=jdump(c), bull=jdump(bull),
                                                     bear=jdump(bear), risk=jdump(risk))) \
                        or {"final": "SKIP", "reason": "desk head did not answer"}
                    approved = str(risk.get("approve")).strip().lower() in ("true", "yes", "1")
                    if approved and str(pm.get("final", "")).strip().upper() == "TAKE":
                        trade = {"market": name, "time": time.time(), "status": "open", **plan}
                        state["trades"].append(trade)
                        open_trades.append(trade)
                        block += [f"SIGNAL: {plan['side']} (paper trade)",
                                  f"Entry {plan['entry']} | Stop-loss {plan['stop']} | Target 1 {plan['target1']} | Target 2 {plan['target2']}",
                                  f"Risk officer: {risk.get('reason')}",
                                  f"Desk head: {pm.get('reason')}"]
                    else:
                        block += [f"Decision: NO TRADE - {b['side']} passed the vote but was blocked.",
                                  f"Risk officer: {risk.get('reason')}",
                                  f"Desk head: {pm.get('reason')}"]
            else:
                block.append(f"Decision: NO TRADE - {b['why']}.")

            block.append("What each agent said:")
            for agent in ALL_VOTERS:
                if agent in votes:
                    v = votes[agent]
                    w = state["weights"].get(f"{name}:{agent}")
                    wt = f" [weight {w}]" if w else ""
                    block.append(f"- {LABELS[agent]}: {v['vote']} ({v['confidence']}/10){wt} {v['reason']}")
                else:
                    block.append(f"- {LABELS[agent]}: no answer (Gemini busy or time up)")
            if name == "BTC" and not extra.get("derivatives", {}).get("source"):
                block.append("Note: BTC futures data (funding, open interest) was not available this hour.")
            report += block

            state["votes_log"].append({"time": time.time(), "market": name, "price": num(price),
                                       "atr": num(a_now, 4), "votes": {a: v["vote"] for a, v in votes.items()}})
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
