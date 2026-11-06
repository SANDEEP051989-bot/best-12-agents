"""
Best 12 AI Agents - BTC and Gold
================================
Every hour it makes one report for BTC and one for Gold and sends it to Telegram.

ROUND 1 - nine specialists, each looks at something different:
  1. Trend & Higher Timeframe   - direction on 1h, 4h and 1 day
  2. Momentum                   - RSI, MACD, speed
  3. Institutional Zones        - big-volume zones, support and resistance
  4. Smart Money                - order blocks, fair value gaps, liquidity (stop-loss pools)
  5. Volume Profile             - prices where most trading happened
  6. Options & Funding          - options open interest, put/call ratio, futures funding (BTC only)
  7. News                       - latest headlines
  8. Sentiment                  - Fear & Greed index + mood of the headlines
  9. Macro                      - US dollar, interest rates, stock market

ROUND 2 - debate and risk:
 10. Bull Researcher            - builds the strongest case for BUY, then gives an honest vote
 11. Bear Researcher            - builds the strongest case for SELL, then gives an honest vote
 12. Risk Manager               - can stop (veto) any trade; gives stop-loss and target

BOSS - final decision:
  BUY or SELL only if  (a) at least MIN_AGREE of the 12 agents vote the same way,
                       (b) their average confidence is at least MIN_CONF,
                       (c) the Risk Manager did not veto.
  Otherwise WAIT. The Boss (AI) can still turn a BUY/SELL into WAIT, never the other way round.

Signals only. It never places a trade.

Setup (own computer / server):
  pip install ccxt pandas google-genai requests
  export GEMINI_API_KEY=your_key
  export TG_TOKEN=your_bot_token
  export TG_CHAT=your_chat_id
  python agents12_best.py            # runs forever, one report per hour
  RUN_ONCE=1 python agents12_best.py # one report and stop (used by GitHub)
"""
import os, time, json, re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import ccxt
import pandas as pd
import requests
from google import genai

# ------------------------------------------------------------------ settings
MARKETS     = [m.strip().upper() for m in os.getenv("MARKETS", "BTC,GOLD").split(",") if m.strip()]
EXCHANGE    = os.getenv("EXCHANGE", "binance")      # use kraken on GitHub (Binance blocks US servers)
MODEL       = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
MIN_AGREE   = int(os.getenv("MIN_AGREE", "8"))       # out of 12
MIN_CONF    = float(os.getenv("MIN_CONF", "6"))      # average confidence, 0 to 10
CALL_GAP    = float(os.getenv("CALL_GAP_SECONDS", "7"))  # pause between AI calls (free limit ~10 a minute)
SLEEP_SECS  = int(os.getenv("SLEEP_SECONDS", "3600"))
TG_TOKEN    = os.getenv("TG_TOKEN")
TG_CHAT     = os.getenv("TG_CHAT")

SYMBOLS = {
    "binance": {"BTC": "BTC/USDT", "GOLD": "PAXG/USDT"},
    "kraken":  {"BTC": "BTC/USD",  "GOLD": "PAXG/USD"},
}
NEWS_QUERY = {"BTC": "bitcoin", "GOLD": "gold price"}
NAMES = {"BTC": "Bitcoin (BTC)", "GOLD": "Gold (via PAXG, a token that follows the gold price)"}
UA = {"User-Agent": "Mozilla/5.0 (twelve-agents bot)"}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
ex = getattr(ccxt, EXCHANGE)()


# ------------------------------------------------------------------ market data
def candles(symbol, tf, n=300):
    d = ex.fetch_ohlcv(symbol, tf, limit=n)
    df = pd.DataFrame(d, columns=["t", "o", "h", "l", "c", "v"])
    return df.tail(n).reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def r(x):
    return round(float(x), 2)


def trend_block(df, name):
    c = df.c
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    return (f"{name}: price {r(c.iloc[-1])}, EMA20 {r(e20.iloc[-1])}, EMA50 {r(e50.iloc[-1])}, "
            f"EMA200 {r(e200.iloc[-1])}, EMA50 five candles ago {r(e50.iloc[-6])}, "
            f"high/low of last 50 candles {r(df.h.tail(50).max())}/{r(df.l.tail(50).min())}")


def momentum_block(df):
    c = df.c
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    rr = rsi(c)
    return (f"1h RSI now {r(rr.iloc[-1])}, 3 candles ago {r(rr.iloc[-4])}; "
            f"MACD {r(macd.iloc[-1])} vs signal {r(sig.iloc[-1])}; histogram last 5: "
            f"{[r(x) for x in hist.tail(5)]}; price change last 10 candles {r((c.iloc[-1] / c.iloc[-11] - 1) * 100)}%; "
            f"last 24 candles change {r((c.iloc[-1] / c.iloc[-25] - 1) * 100)}%")


def swing_points(df, k=3, look=150):
    d = df.tail(look).reset_index(drop=True)
    highs, lows = [], []
    for i in range(k, len(d) - k):
        if d.h[i] == d.h[i - k:i + k + 1].max():
            highs.append(r(d.h[i]))
        if d.l[i] == d.l[i - k:i + k + 1].min():
            lows.append(r(d.l[i]))
    return highs, lows


def institutional_block(df):
    d = df.tail(200).reset_index(drop=True)
    avgv = d.v.rolling(20).mean()
    body = (d.c - d.o).abs()
    zones = []
    for i in range(20, len(d)):
        if d.v[i] > 2 * avgv[i] and body[i] > 0.5 * (d.h[i] - d.l[i]):
            side = "buying" if d.c[i] > d.o[i] else "selling"
            zones.append(f"{side} zone {r(d.l[i])}-{r(d.h[i])} ({len(d) - i} candles ago)")
    highs, lows = swing_points(df)
    price = df.c.iloc[-1]
    above = sorted([x for x in highs if x > price])[:4]
    below = sorted([x for x in lows if x < price], reverse=True)[:4]
    return (f"Price {r(price)}. Heavy-volume candles (2x normal volume) on 1h: {zones[-6:] or 'none'}. "
            f"Swing highs above price (resistance): {above or 'none'}. Swing lows below price (support): {below or 'none'}")


def smart_money_block(df):
    d = df.tail(120).reset_index(drop=True)
    a = atr(d)
    price = d.c.iloc[-1]
    fvg_up, fvg_dn = [], []
    for i in range(2, len(d)):
        if d.l[i] > d.h[i - 2]:
            top, bot = d.l[i], d.h[i - 2]
            if d.l[i + 1:].min() > bot if i + 1 < len(d) else True:
                fvg_up.append(f"{r(bot)}-{r(top)}")
        if d.h[i] < d.l[i - 2]:
            top, bot = d.l[i - 2], d.h[i]
            if d.h[i + 1:].max() < top if i + 1 < len(d) else True:
                fvg_dn.append(f"{r(bot)}-{r(top)}")
    ob_bull, ob_bear = [], []
    for i in range(1, len(d) - 3):
        move = d.c[i + 3] - d.c[i]
        if d.c[i] < d.o[i] and move > 1.5 * a[i]:
            ob_bull.append(f"{r(d.l[i])}-{r(d.h[i])}")
        if d.c[i] > d.o[i] and -move > 1.5 * a[i]:
            ob_bear.append(f"{r(d.l[i])}-{r(d.h[i])}")
    highs, lows = swing_points(df, k=2, look=120)
    tol = price * 0.0015
    eq_highs = sorted({h for h in highs for h2 in highs if h != h2 and abs(h - h2) < tol})
    eq_lows = sorted({l for l in lows for l2 in lows if l != l2 and abs(l - l2) < tol})
    return (f"Price {r(price)}. Unfilled bullish fair value gaps: {fvg_up[-3:] or 'none'}. "
            f"Unfilled bearish fair value gaps: {fvg_dn[-3:] or 'none'}. "
            f"Bullish order blocks (last down candle before a strong rise): {ob_bull[-3:] or 'none'}. "
            f"Bearish order blocks: {ob_bear[-3:] or 'none'}. "
            f"Equal highs (stop-loss pools above): {eq_highs[-4:] or 'none'}. "
            f"Equal lows (stop-loss pools below): {eq_lows[:4] or 'none'}. "
            f"Last 6 candles high/low: {[(r(x), r(y)) for x, y in zip(d.h.tail(6), d.l.tail(6))]}")


def volume_profile_block(df, bins=30):
    d = df.tail(200)
    lo, hi = d.l.min(), d.h.max()
    step = (hi - lo) / bins or 1
    vol = [0.0] * bins
    for _, row in d.iterrows():
        idx = min(int((row.c - lo) / step), bins - 1)
        vol[idx] += row.v
    poc_i = max(range(bins), key=lambda i: vol[i])
    order = sorted(range(bins), key=lambda i: vol[i], reverse=True)
    total, acc, area = sum(vol), 0, []
    for i in order:
        acc += vol[i]
        area.append(i)
        if acc >= 0.7 * total:
            break
    vah, val = lo + (max(area) + 1) * step, lo + min(area) * step
    avgv = d.v.tail(50).mean()
    return (f"Last 200 hourly candles. Price {r(df.c.iloc[-1])}. Point of control (most traded price) "
            f"{r(lo + (poc_i + 0.5) * step)}. Value area (70% of volume) {r(val)}-{r(vah)}. "
            f"Last 5 candle volumes vs 50-candle average {r(avgv)}: {[r(x) for x in d.v.tail(5)]}; "
            f"last 5 candle colors: {['green' if c > o else 'red' for c, o in zip(d.c.tail(5), d.o.tail(5))]}")


# ------------------------------------------------------------------ outside data (all free, no key)
def get_json(url, **kw):
    return requests.get(url, headers=UA, timeout=15, **kw).json()


def derivatives_block(market):
    if market != "BTC":
        return "No free options or funding data for gold. Judge only from price action and say so."
    out = []
    try:
        t = get_json("https://www.deribit.com/api/v2/public/ticker", params={"instrument_name": "BTC-PERPETUAL"})["result"]
        out.append(f"Perpetual futures: funding (8h) {t.get('funding_8h')}, open interest {t.get('open_interest')}, "
                   f"mark price {t.get('mark_price')}")
    except Exception as e:
        out.append(f"funding not available ({e.__class__.__name__})")
    try:
        rows = get_json("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                        params={"currency": "BTC", "kind": "option"})["result"]
        calls = [x for x in rows if x["instrument_name"].endswith("-C")]
        puts = [x for x in rows if x["instrument_name"].endswith("-P")]
        coi = sum(x.get("open_interest") or 0 for x in calls)
        poi = sum(x.get("open_interest") or 0 for x in puts)

        def top(lst):
            by = {}
            for x in lst:
                strike = int(x["instrument_name"].split("-")[2])
                by[strike] = by.get(strike, 0) + (x.get("open_interest") or 0)
            return sorted(by, key=by.get, reverse=True)[:4]
        out.append(f"Options: put/call open interest ratio {r(poi / coi) if coi else 'n/a'}; "
                   f"strikes with most call money {top(calls)}; strikes with most put money {top(puts)}")
    except Exception as e:
        out.append(f"options not available ({e.__class__.__name__})")
    return ". ".join(out)


def news_block(market):
    try:
        q = quote_plus(NEWS_QUERY[market] + " when:1d")
        xml = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                           headers=UA, timeout=15).text
        items = ET.fromstring(xml).findall(".//item")[:12]
        heads = [i.findtext("title") for i in items]
        return "Headlines from the last 24 hours:\n- " + "\n- ".join(heads) if heads else "No headlines found."
    except Exception as e:
        return f"News not available ({e.__class__.__name__})."


def fear_greed():
    try:
        d = get_json("https://api.alternative.me/fng/", params={"limit": 7})["data"]
        return "Crypto Fear & Greed (0 = extreme fear, 100 = extreme greed), today first: " + \
               ", ".join(f"{x['value']} {x['value_classification']}" for x in d)
    except Exception as e:
        return f"Fear & Greed not available ({e.__class__.__name__})."


def yahoo(sym, label):
    try:
        d = get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(sym)}",
                     params={"range": "2mo", "interval": "1d"})
        closes = [x for x in d["chart"]["result"][0]["indicators"]["quote"][0]["close"] if x is not None]
        now = closes[-1]
        return (f"{label}: {r(now)}, 5-day change {r((now / closes[-6] - 1) * 100)}%, "
                f"20-day change {r((now / closes[-21] - 1) * 100)}%")
    except Exception as e:
        return f"{label}: not available ({e.__class__.__name__})"


def macro_block():
    return "; ".join([yahoo("DX-Y.NYB", "US Dollar index"), yahoo("^TNX", "US 10-year yield (x10)"),
                      yahoo("^GSPC", "S&P 500"), yahoo("GC=F", "Gold futures")])


# ------------------------------------------------------------------ AI
VOTE_JSON = '{"vote":"BUY or SELL or WAIT","confidence":0 to 10,"reason":"one short line in simple English"}'


def ask(prompt, fallback):
    for attempt in range(3):
        time.sleep(CALL_GAP)
        try:
            t = client.models.generate_content(model=MODEL, contents=prompt).text or ""
            j = json.loads(t[t.find("{"): t.rfind("}") + 1])
            if "vote" in j:
                j["vote"] = str(j["vote"]).upper().strip()
                if j["vote"] not in ("BUY", "SELL", "WAIT"):
                    j["vote"] = "WAIT"
                j["confidence"] = max(0.0, min(10.0, float(j.get("confidence") or 0)))
            return j
        except Exception as e:
            print(f"  AI retry {attempt + 1}: {e.__class__.__name__}: {str(e)[:120]}")
            time.sleep(15 * (attempt + 1))
    return dict(fallback)


SPECIALISTS = [
    ("Trend & Higher Timeframe",
     "You are a trend trader with 30 years of experience. Trade only with the main trend. "
     "Check if 1h, 4h and 1 day agree. If they disagree, vote WAIT.", "trend"),
    ("Momentum",
     "You are a momentum trader with 30 years of experience. Judge if buying or selling pressure is "
     "getting stronger or weaker. Watch for RSI above 70 or below 30.", "momentum"),
    ("Institutional Zones",
     "You are a former bank trader. Find where big institutions bought or sold heavily, and whether price "
     "is near strong support or resistance. Do not buy right under resistance or sell right above support.", "inst"),
    ("Smart Money",
     "You are a smart money (ICT) trader with 30 years of experience. Use order blocks, fair value gaps "
     "and liquidity. Big players often push price to the stop-loss pools first, then reverse.", "smc"),
    ("Volume Profile",
     "You are a volume profile expert. Price above the point of control and value area is strong, below is weak. "
     "Check if volume confirms the move.", "vp"),
    ("Options & Funding",
     "You are a derivatives trader. High positive funding means too many buyers (risk of drop); negative "
     "funding means too many sellers (risk of squeeze up). Strikes with big open interest act like magnets "
     "and walls. If data is missing, vote WAIT with low confidence.", "deriv"),
    ("News",
     "You are a news trader with 30 years of experience. Decide if today's news is good, bad or neutral "
     "for the price in the next few hours to a day. Ignore old or unimportant news.", "news"),
    ("Sentiment",
     "You are a market psychology expert. Extreme greed is a warning for buyers, extreme fear can be a "
     "buying chance. Judge the crowd's mood from the index and the headlines.", "sent"),
    ("Macro",
     "You are a macro trader with 30 years of experience. A rising US dollar and rising interest rates are "
     "usually bad for gold and bitcoin; falling ones are good. Rising stocks often help bitcoin.", "macro"),
]


def run_market(market):
    symbol = SYMBOLS.get(EXCHANGE, SYMBOLS["kraken"])[market]
    h1, h4, d1 = candles(symbol, "1h"), candles(symbol, "4h"), candles(symbol, "1d", 250)
    price = h1.c.iloc[-1]
    a = atr(h1).iloc[-1]
    head = f"Market: {NAMES[market]}. Current price {r(price)}. 1h ATR (normal candle size) {r(a)}.\n"
    data = {
        "trend": "\n".join([trend_block(h1, "1 hour"), trend_block(h4, "4 hour"), trend_block(d1, "1 day")]),
        "momentum": momentum_block(h1) + " | 4h RSI " + str(r(rsi(h4.c).iloc[-1])),
        "inst": institutional_block(h1),
        "smc": smart_money_block(h1),
        "vp": volume_profile_block(h1),
        "deriv": derivatives_block(market),
        "news": news_block(market),
        "sent": (fear_greed() if market == "BTC" else "No fear & greed index for gold.") + "\n" + news_block(market),
        "macro": macro_block(),
    }

    opinions = []
    for name, role, key in SPECIALISTS:
        p = (f"{role}\n\n{head}Your data:\n{data[key]}\n\n"
             f"Look ONLY at your own area. Reply ONLY in JSON: {VOTE_JSON}")
        o = ask(p, {"vote": "WAIT", "confidence": 0, "reason": "no answer from AI"})
        o["agent"] = name
        opinions.append(o)
        print(f"  {name}: {o.get('vote')} ({o.get('confidence')})")

    board = json.dumps([{k: o.get(k) for k in ("agent", "vote", "confidence", "reason")} for o in opinions])
    short = head + data["trend"] + "\n" + data["inst"]

    bull = ask(f"You are the Bull Researcher. Build the strongest honest case for BUY using the specialists' "
               f"views below. Then give your honest final vote (it can be WAIT or SELL if the case is weak).\n\n"
               f"{short}\nSpecialists: {board}\n\nReply ONLY in JSON: {VOTE_JSON}",
               {"vote": "WAIT", "confidence": 0, "reason": "no answer from AI"})
    bull["agent"] = "Bull Researcher"
    bear = ask(f"You are the Bear Researcher. Build the strongest honest case for SELL using the specialists' "
               f"views below. Then give your honest final vote (it can be WAIT or BUY if the case is weak).\n\n"
               f"{short}\nSpecialists: {board}\nBull case: {bull.get('reason')}\n\nReply ONLY in JSON: {VOTE_JSON}",
               {"vote": "WAIT", "confidence": 0, "reason": "no answer from AI"})
    bear["agent"] = "Bear Researcher"

    risk = ask(f"You are the Risk Manager with 30 years of experience. Protecting money comes first. "
               f"Read everything below. Set veto=true if the trade is too risky (big news soon, price stuck "
               f"between support and resistance, agents strongly split, stop-loss would be too far). "
               f"Give a stop-loss beyond the nearest strong level (about 1.5 to 3 ATR away) and a target at "
               f"least 1.5 times the stop distance.\n\n{short}\nSpecialists: {board}\n"
               f"Bull: {bull.get('vote')} - {bull.get('reason')}\nBear: {bear.get('vote')} - {bear.get('reason')}\n\n"
               'Reply ONLY in JSON: {"vote":"BUY or SELL or WAIT","confidence":0 to 10,"veto":true or false,'
               '"stop_loss":number,"target":number,"reason":"one short line"}',
               {"vote": "WAIT", "confidence": 0, "veto": True, "reason": "no answer from AI"})
    risk["agent"] = "Risk Manager"
    opinions += [bull, bear, risk]

    # ---- Boss: fixed rule first
    buys = [o for o in opinions if o.get("vote") == "BUY"]
    sells = [o for o in opinions if o.get("vote") == "SELL"]
    waits = 12 - len(buys) - len(sells)
    side, group = (("BUY", buys) if len(buys) >= len(sells) else ("SELL", sells))
    avg_conf = sum(float(o.get("confidence") or 0) for o in group) / len(group) if group else 0
    vetoed = bool(risk.get("veto"))
    rule_ok = len(group) >= MIN_AGREE and avg_conf >= MIN_CONF and not vetoed
    action = side if rule_ok else "WAIT"

    if not rule_ok:
        why = []
        if len(group) < MIN_AGREE:
            why.append(f"only {len(group)} of 12 agree (need {MIN_AGREE})")
        if avg_conf < MIN_CONF:
            why.append(f"average confidence {avg_conf:.1f} (need {MIN_CONF:g})")
        if vetoed:
            why.append("Risk Manager stopped it")
        boss_reason = "Rule not met: " + ", ".join(why) + "."
    else:
        boss = ask(f"You are the Boss of a trading team. The rule allows a {side}. You may confirm it or change it "
                   f"to WAIT if something important looks wrong. You can NOT change it to the other side.\n\n"
                   f"{short}\nAll 12 agents: {json.dumps(opinions, default=str)}\n\n"
                   'Reply ONLY in JSON: {"action":"' + side + ' or WAIT","reason":"two short lines"}',
                   {"action": "WAIT", "reason": "no answer from AI"})
        if str(boss.get("action", "")).upper().strip() != side:
            action = "WAIT"
        boss_reason = boss.get("reason", "")

    sl, tp = risk.get("stop_loss"), risk.get("target")
    try:
        sl, tp = float(sl), float(tp)
        bad = (action == "BUY" and not (sl < price < tp)) or (action == "SELL" and not (tp < price < sl))
        bad = bad or not (0.5 * a <= abs(price - sl) <= 6 * a) or abs(tp - price) > 15 * a
    except (TypeError, ValueError):
        bad = True
    if bad and action != "WAIT":   # fall back to ATR levels
        sl = price - 2 * a if action == "BUY" else price + 2 * a
        tp = price + 3 * a if action == "BUY" else price - 3 * a

    icon = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "WAIT": "⏸ WAIT"}[action]
    lines = [f"{NAMES[market]} | price {r(price)}",
             f"Votes: Buy {len(buys)} | Sell {len(sells)} | Wait {waits}",
             ""]
    for o in opinions:
        lines.append(f"• {o['agent']}: {o.get('vote')} ({o.get('confidence', 0):g}) - {o.get('reason', '')}")
    lines += ["", f"BOSS DECISION: {icon}"]
    if action != "WAIT":
        lines.append(f"Stop-loss {r(sl)} | Target {r(tp)}")
    lines.append(boss_reason)
    return "\n".join(lines)


def tell(msg):
    print(msg)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": msg[:4000]}, timeout=15)
        except Exception as e:
            print("Telegram error:", e)


def run_once():
    for m in MARKETS:
        print(f"--- {m} ---")
        try:
            tell(run_market(m))
        except Exception as e:
            tell(f"{m}: report failed - {e.__class__.__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    if os.getenv("RUN_ONCE") == "1":
        run_once()
    else:
        while True:
            run_once()
            time.sleep(SLEEP_SECS)
