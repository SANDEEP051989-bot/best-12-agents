# agents_v2.py FAST - 12 AI agents + time machine + report card + portfolio manager
# Markets: BTC, Gold, US Oil. Brain: Gemini. Reports: Telegram.
import os, json, time, datetime as dt
import requests
import pandas as pd
import yfinance as yf

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

MARKETS = {"BTC": "BTC-USD", "GOLD": "GC=F", "OIL": "CL=F"}
MIN_AGREE = 8          # change to 7 if too few signals
MIN_CONF = 6           # average confidence needed (1-10)
MAX_OPEN_LOSS_PCT = 6  # total loss limit across open paper trades
MAX_RUN_SECONDS = 12 * 60   # stop asking Gemini after 12 minutes
STATE_FILE = "state.json"
START = time.time()
STATS = {"asked": 0, "answered": 0, "busy": 0, "errors": []}


def log(msg):
    print(f"[{int(time.time() - START)}s] {msg}", flush=True)


def time_left():
    return time.time() - START < MAX_RUN_SECONDS


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"signals": [], "weights": {}, "memory": {}}


def save_state(s):
    s["signals"] = s["signals"][-500:]
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1)


def get_data(ticker, period="60d", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    tr = pd.concat([df.High - df.Low,
                    (df.High - df.Close.shift()).abs(),
                    (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def chart_facts(df):
    c = df.Close
    e20, e50, e200 = c.ewm(span=20).mean(), c.ewm(span=50).mean(), c.ewm(span=200).mean()
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    vol_ratio = df.Volume.iloc[-1] / max(df.Volume.rolling(20).mean().iloc[-1], 1)
    return {
        "price": round(float(c.iloc[-1]), 2),
        "ema20": round(float(e20.iloc[-1]), 2),
        "ema50": round(float(e50.iloc[-1]), 2),
        "ema200": round(float(e200.iloc[-1]), 2),
        "rsi": round(float(rsi(c).iloc[-1]), 1),
        "macd_above_signal": bool(macd.iloc[-1] > sig.iloc[-1]),
        "atr": round(float(atr(df).iloc[-1]), 2),
        "high_20": round(float(df.High.tail(20).max()), 2),
        "low_20": round(float(df.Low.tail(20).min()), 2),
        "volume_vs_avg": round(float(vol_ratio), 2),
        "change_24h_pct": round(float(c.iloc[-1] / c.iloc[-25] * 100 - 100), 2),
    }


def get_news(ticker, n=8):
    try:
        out = []
        for it in yf.Ticker(ticker).news[:n]:
            t = it.get("title") or it.get("content", {}).get("title", "")
            if t:
                out.append(t)
        return out
    except Exception:
        return []


def get_macro():
    out = {}
    for name, t in {"dollar_index": "DX-Y.NYB", "us10y_yield": "^TNX", "sp500": "^GSPC"}.items():
        try:
            d = get_data(t, "10d", "1d").Close
            out[name] = {"last": round(float(d.iloc[-1]), 2),
                         "5d_change_pct": round(float(d.iloc[-1] / d.iloc[-6] * 100 - 100), 2)}
        except Exception:
            pass
    return out


def get_btc_extra():
    out = {}
    try:
        out["fear_greed"] = requests.get("https://api.alternative.me/fng/", timeout=10).json()["data"][0]
    except Exception:
        pass
    try:
        f = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=10).json()
        out["funding_rate"] = f.get("lastFundingRate")
        oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT", timeout=10).json()
        out["open_interest"] = oi.get("openInterest")
    except Exception:
        pass
    return out


# ---------- Gemini: never gets stuck ----------
def ask_gemini(prompt):
    if not time_left():
        return None
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:generateContent?key={GEMINI_KEY}")
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3,
                                 "responseMimeType": "application/json"}}
    STATS["asked"] += 1
    for attempt in range(2):          # at most 2 tries, short waits
        try:
            r = requests.post(url, json=body, timeout=25)
            if r.status_code == 200:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                STATS["answered"] += 1
                return json.loads(txt)
            if r.status_code in (429, 503):
                STATS["busy"] += 1
                log(f"Gemini busy ({r.status_code}), short wait")
                time.sleep(6)
                continue
            msg = f"Gemini error {r.status_code}: {r.text[:150]}"
            log(msg)
            if msg not in STATS["errors"]:
                STATS["errors"].append(msg)
            return None               # wrong key / wrong model: do not retry
        except Exception as e:
            log(f"Gemini call failed: {e}")
    return None                       # skip this agent, move on


ANSWER_RULE = ('Reply ONLY as JSON: {"vote": "BUY" or "SELL" or "WAIT", '
               '"confidence": 1-10, "reason": "one short line"}')

AGENTS = {
    "trend":      "You are a trend analyst. Use the EMAs and MACD to judge the main trend.",
    "momentum":   "You are a momentum analyst. Use RSI, 24h change and volume.",
    "levels":     "You find support, resistance and institutional zones from the 20-bar high/low and price.",
    "volatility": "You judge if volatility (ATR) makes a trade safe or dangerous now.",
    "daily_view": "You look only at the DAILY chart facts and judge the bigger picture.",
    "news":       "You read the headlines and judge if news is good or bad for price.",
    "mood":       "You judge crowd mood: fear, greed, funding and open interest if given.",
    "macro":      "You judge dollar index, US 10 year yield and S&P 500 effect on this market.",
    "special":    "",
}

SPECIAL = {
    "BTC":  "You are a crypto derivatives expert. Judge funding rate and open interest.",
    "GOLD": "You are a gold expert. Gold usually rises when the dollar and yields fall and fear rises.",
    "OIL":  ("You are an oil expert. Watch OPEC news, weekly US oil stock (EIA) news, "
             "wars and supply cuts in the headlines."),
}


def run_analysts(name, facts, state):
    votes = {}
    for agent, role in AGENTS.items():
        if agent == "special":
            role = SPECIAL[name]
        past = state["memory"].get(f"{name}:{agent}", [])[-5:]
        prompt = (f"{role}\nMarket: {name}\nData: {json.dumps(facts)}\n"
                  f"Your last calls and how they turned out: {json.dumps(past)}\n"
                  f"Learn from mistakes. {ANSWER_RULE}")
        ans = ask_gemini(prompt)
        log(f"{name} {agent}: {(ans or {}).get('vote', 'skipped')}")
        if ans:
            votes[agent] = ans
        time.sleep(2)
    return votes


def debate(name, facts, votes):
    bull = ask_gemini(f"You are the BULL researcher for {name}. Make the strongest case to BUY "
                      f"using: {json.dumps(votes)} {json.dumps(facts)}. {ANSWER_RULE}")
    time.sleep(2)
    bear = ask_gemini(f"You are the BEAR researcher for {name}. Make the strongest case to SELL "
                      f"or stay out using: {json.dumps(votes)} {json.dumps(facts)}. {ANSWER_RULE}")
    return bull, bear


def risk_manager(name, facts, plan):
    prompt = (f"You are the risk manager for {name}. Proposed trade: {json.dumps(plan)}. "
              f"Facts: {json.dumps(facts)}. Is the stop-loss sensible, is reward at least 1.5x risk, "
              'is it too dangerous now? Reply ONLY as JSON: {"approve": true or false, "reason": "short"}')
    return ask_gemini(prompt) or {"approve": False, "reason": "risk agent did not answer"}


def portfolio_manager(name, plan, bull, bear, risk, open_trades):
    prompt = (f"You are the portfolio manager. Final say on this {name} trade: {json.dumps(plan)}. "
              f"Bull: {json.dumps(bull)} Bear: {json.dumps(bear)} Risk: {json.dumps(risk)}. "
              f"Open paper trades: {json.dumps(open_trades)}. Avoid too many trades in the same "
              'direction at once. Reply ONLY as JSON: {"final": "TAKE" or "SKIP", "reason": "short"}')
    return ask_gemini(prompt) or {"final": "SKIP", "reason": "manager did not answer"}


# ---------- time machine, report card, loss limit, boss ----------
def time_machine(df):
    c = df.Close
    e20, e50, r = c.ewm(span=20).mean(), c.ewm(span=50).mean(), rsi(c)
    wins = total = 0
    for i in range(60, len(c) - 24):
        buy = e20.iloc[i] > e50.iloc[i] and r.iloc[i] > 55
        sell = e20.iloc[i] < e50.iloc[i] and r.iloc[i] < 45
        if buy or sell:
            total += 1
            up = c.iloc[i + 24] > c.iloc[i]
            wins += int((buy and up) or (sell and not up))
    rate = round(wins / total * 100, 1) if total else 0
    now_buy = e20.iloc[-1] > e50.iloc[-1] and r.iloc[-1] > 55
    now_sell = e20.iloc[-1] < e50.iloc[-1] and r.iloc[-1] < 45
    vote = "BUY" if now_buy else "SELL" if now_sell else "WAIT"
    if rate < 52:
        vote = "WAIT"
    return {"vote": vote, "confidence": min(10, max(1, int(rate / 10))),
            "reason": f"rule won {rate}% of {total} past signals"}


def update_report_card(state, prices):
    now = time.time()
    for s in state["signals"]:
        if s.get("checked") or now - s["time"] < 24 * 3600:
            continue
        p = prices.get(s["market"])
        if not p:
            continue
        went_up = p > s["price"]
        s["checked"] = True
        for agent, v in s["votes"].items():
            if v not in ("BUY", "SELL"):
                continue
            right = (v == "BUY") == went_up
            key = f"{s['market']}:{agent}"
            state["memory"].setdefault(key, []).append(
                {"said": v, "right": right, "price": s["price"]})
            state["memory"][key] = state["memory"][key][-20:]
            hist = state["memory"][key]
            acc = sum(h["right"] for h in hist) / len(hist)
            state["weights"][key] = round(0.5 + acc, 2)


def open_loss_pct(state, prices):
    total = 0.0
    for s in state["signals"]:
        if s.get("taken") and not s.get("checked") and prices.get(s["market"]):
            move = (prices[s["market"]] / s["price"] - 1) * 100
            total += move if s["side"] == "BUY" else -move
    return round(total, 2)


def boss(name, votes, state):
    buy = [a for a, v in votes.items() if v and v.get("vote") == "BUY"]
    sell = [a for a, v in votes.items() if v and v.get("vote") == "SELL"]
    side, team = ("BUY", buy) if len(buy) >= len(sell) else ("SELL", sell)
    w = lambda a: state["weights"].get(f"{name}:{a}", 1.0)
    score = sum(w(a) for a in team)
    conf = sum(votes[a].get("confidence", 0) * w(a) for a in team) / max(score, 0.01)
    ok = len(team) >= MIN_AGREE and conf >= MIN_CONF
    return {"side": side, "agree": len(team), "of": len(votes),
            "weighted_conf": round(conf, 1), "go": ok}


def make_plan(side, facts):
    p, a = facts["price"], facts["atr"]
    if side == "BUY":
        return {"side": "BUY", "entry": p, "stop": round(p - 1.5 * a, 2),
                "target1": round(p + 2 * a, 2), "target2": round(p + 3.5 * a, 2)}
    return {"side": "SELL", "entry": p, "stop": round(p + 1.5 * a, 2),
            "target1": round(p - 2 * a, 2), "target2": round(p - 3.5 * a, 2)}


# ---------- Telegram, weekly report, main ----------
def send(text):
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": chunk}, timeout=20)
        except Exception as e:
            log(f"Telegram failed: {e}")


def weekly_report(state):
    lines = ["WEEKLY AGENT REPORT CARD"]
    for key, hist in sorted(state["memory"].items()):
        acc = round(sum(h["right"] for h in hist) / len(hist) * 100)
        lines.append(f"{key}: right {acc}% of last {len(hist)} calls, weight {state['weights'].get(key, 1.0)}")
    if len(lines) == 1:
        lines.append("Not enough checked signals yet.")
    send("\n".join(lines))


def main():
    log("start")
    state = load_state()
    prices = {}
    for name, t in MARKETS.items():
        try:
            prices[name] = float(get_data(t, "5d", "1h").Close.iloc[-1])
        except Exception as e:
            log(f"price {name} failed: {e}")
    update_report_card(state, prices)
    loss = open_loss_pct(state, prices)
    macro = get_macro()
    report = [f"AI TEAM REPORT {dt.datetime.utcnow():%d %b %H:%M} UTC",
              f"Open paper trades total: {loss}%"]

    for name, t in MARKETS.items():
        try:
            log(f"--- {name} ---")
            h1 = get_data(t, "60d", "1h")
            d1 = get_data(t, "1y", "1d")
            facts = {"hourly": chart_facts(h1), "daily": chart_facts(d1),
                     "news": get_news(t), "macro": macro}
            if name == "BTC":
                facts["btc_extra"] = get_btc_extra()
            votes = run_analysts(name, facts, state)
            bull, bear = debate(name, facts, votes)
            if bull:
                votes["bull"] = bull
            if bear:
                votes["bear"] = bear
            votes["time_machine"] = time_machine(h1)
            b = boss(name, votes, state)
            plan = make_plan(b["side"], facts["hourly"])
            line = (f"\n{name} {facts['hourly']['price']} | {b['side']} "
                    f"{b['agree']}/{b['of']} agree, conf {b['weighted_conf']}")
            taken = False
            if b["go"] and loss > -MAX_OPEN_LOSS_PCT:
                risk = risk_manager(name, facts, plan)
                open_tr = [s for s in state["signals"] if s.get("taken") and not s.get("checked")]
                pm = portfolio_manager(name, plan, bull, bear, risk, open_tr)
                if risk.get("approve") and pm.get("final") == "TAKE":
                    taken = True
                    line += (f"\nSIGNAL {plan['side']} entry {plan['entry']} stop {plan['stop']} "
                             f"T1 {plan['target1']} T2 {plan['target2']}\nWhy: {pm.get('reason')}")
                else:
                    line += f"\nBlocked: {risk.get('reason')} / {pm.get('reason')}"
            elif loss <= -MAX_OPEN_LOSS_PCT:
                line += "\nNo new trades: total loss limit reached"
            else:
                line += "\nNo trade: not enough agents agree"
            report.append(line)
            state["signals"].append({"time": time.time(), "market": name,
                "price": facts["hourly"]["price"], "side": b["side"], "taken": taken,
                "votes": {a: (v or {}).get("vote", "WAIT") for a, v in votes.items()}})
        except Exception as e:
            log(f"{name} error: {e}")
            report.append(f"\n{name}: error {e}")

    report.append(f"\nGemini: {STATS['answered']} of {STATS['asked']} questions answered, "
                  f"busy {STATS['busy']} times. Run time {int(time.time() - START)}s")
    for err in STATS["errors"][:2]:
        report.append(err)
    send("\n".join(report) + "\n\nPaper trades only. Not financial advice.")

    now = dt.datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    if now.weekday() == 0 and now.hour >= 2 and state.get("last_weekly") != today:
        weekly_report(state)
        state["last_weekly"] = today
    save_state(state)
    log("done")


if __name__ == "__main__":
    main()
