#!/usr/bin/env python3
"""
FULL RULES TEST - v7.0 vs v7.1 with every code filter, on Yahoo's last 2 years of hourly prices.

It replays the live "rules-only twin" hour by hour, using the real functions inside agents_v7.py:
  12 agents' code rules -> vote rule (6 on the trend side, at most 1 against, 10 of 12 answering)
  -> blocks (direction gate, room under 1R, Time Machine, volatility, shock / burst candles, costs, Friday/holiday)
  -> risk officer (one trade at a time, 1 per market per day, loss limits, half size when the gate is NEUTRAL)
  -> the live trade management (half off at 2R and stop to +1R, trailing stop, 4R, 24 hours, Friday close).
Four rule sets: v7.0 and v7.1, each with 6 votes (now) and with 5 votes. Plus the trend-only twin (no filters)
as the base line.

What cannot be replayed (same as the live rules-only twin, which has no AI either):
  the AI brains (news = WAIT), live Bitcoin derivatives and gold/oil fund-flow data (the Special agent has
  no data), and the economic-event calendar.

Usage:  python compare_full.py BTC      (or GOLD, OIL, ALL)
It changes nothing in the repo. The result goes to Telegram.
"""
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests

import agents_v7 as v7

PASSES = [  # (label, gate, stop limit in ATR, votes needed)
    ("v7.0 now (daily view, stop 2.0 ATR, 6 votes)", "DAILY", 2.0, 6),
    ("v7.0 with 5 votes", "DAILY", 2.0, 5),
    ("v7.1 (4-hour view, stop 3.0 ATR, 6 votes)", "4H", 3.0, 6),
    ("v7.1 with 5 votes", "4H", 3.0, 5),
]
BASELINES = [("trend only, no filters (stop 2.0 ATR)", 2.0), ("trend only, no filters (stop 3.0 ATR)", 3.0)]
MACRO_DAYS = 120          # the live code reads 120 days of macro prices
BLOCK_NAMES = {"daily_view": "direction gate against the trend", "room": "strong level under 1R away",
               "stop": "no valid stop", "costs": "costs above 0.2R", "time_machine": "Time Machine",
               "calendar": "Friday / holiday / market break", "burst": "burst candle", "shock": "shock candle",
               "funding_shock": "funding shock", "cascade": "cascade cooldown", "event": "event window",
               "vol_extreme": "extreme volatility", "vol.extreme": "extreme volatility",
               "macro_risk_off": "macro risk-off", "flows": "fund flows against", "crowded_longs": "crowded longs",
               "crowded_shorts": "crowded shorts", "long_flush": "long flush", "squeeze_done": "squeeze done"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ the two direction gates
def view_4h_state(h4, lags4):
    """v7.1 gate: last finished 4-hour close above 3 of the 4 earlier 4-hour closes = UP, below 3 of 4 = DOWN."""
    c = h4["Close"].dropna() if h4 is not None and len(h4) else pd.Series(dtype=float)
    if len(c) <= max(lags4):
        return "NEUTRAL", "not enough 4-hour history", {}
    now_c = float(c.iloc[-1])
    past = [float(c.iloc[-1 - l]) for l in lags4]
    ups = sum(1 for v in past if now_c > v)
    downs = sum(1 for v in past if now_c < v)
    ans = "UP" if ups >= 3 else ("DOWN" if downs >= 3 else "NEUTRAL")
    data = {"close_4h": now_c, "checks_higher": ups, "checks_lower": downs}
    return ans, f"4-hour trend {ans}", data


def gate_agent(x):
    """Replaces agent_daily_view during the replay: the gate comes from x['gate_state'] (daily or 4-hour)."""
    ans, why, data = x["gate_state"]
    vote = {"UP": "BUY", "DOWN": "SELL"}.get(ans, "WAIT")
    return v7.A("daily_view", vote, f"gate.{ans.lower()}", f"{ans}: {why}", data, gate=ans)


# ------------------------------------------------------------------ outside data, once per run
def macro_series():
    out = {}
    for k, tk in v7.MACRO_TICKERS.items():
        h = None
        for _ in range(3):
            try:
                h = v7.fetch_yahoo(tk, "729d", "1h")
                if h is not None and len(h):
                    break
            except Exception as e:
                log("macro download", tk, e)
            time.sleep(3)
        out[k] = h
        log("macro", k, 0 if h is None else len(h), "candles")
    return out


def macro_at(series, now):
    nowt = pd.Timestamp(now)
    cut = {}
    for k, h in series.items():
        if h is None or not len(h):
            cut[k] = None
            continue
        lim = nowt - pd.Timedelta(hours=1, minutes=v7.MACRO_DELAY.get(k, 0))
        pos = int(h.index.searchsorted(lim, side="right"))
        start = int(h.index.searchsorted(lim - pd.Timedelta(days=MACRO_DAYS), side="left"))
        cut[k] = h.iloc[start:pos] if pos > start else None
    return v7.macro_from(cut, now)


def fng_history():
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 0}, timeout=30)
        rows = r.json()["data"]
        s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s"): float(d["value"]) for d in rows}).sort_index()
        log("fear & greed history:", len(s), "days")
        return s
    except Exception as e:
        log("fear & greed history not reachable:", e)
        return None


def fng_at(s, now):
    if s is None:
        return None
    past = s[s.index <= pd.Timestamp(now)]
    if not len(past):
        return None
    return {"value": float(past.iloc[-1]), "avg30": float(past.iloc[-30:].mean())}


class Desk:
    """The small part of the live state that the decision code reads."""
    misc = {"blocks_until": {}, "last_bar": {}}
    cache = {}


# ------------------------------------------------------------------ one paper book
def new_book(mkt):
    return {"open": {mkt: None}, "closed": [], "exit_i": None, "skipped": 0, "why": Counter(), "hours": 0}


def close_if_done(book, mkt, i):
    t = book["open"].get(mkt)
    if t is None or book["exit_i"] is None or book["exit_i"] > i:
        return
    res = t.get("_res")
    if res is None:                        # still running when the data ends: not counted
        return
    t.update({"status": "closed", "r": float(res["r"]), "why": res["why"], "exit": res["exit"],
              "exit_ts": res["exit_ts"]})
    ex = v7.to_ist(res["exit_ts"])
    t["exit_day"] = ex.strftime("%Y-%m-%d")
    t["exit_week"] = v7.iso_week(ex.tz_localize(None))
    t["pnl_rs"] = round(t["r"] * t["risk_rs"], 2)
    book["closed"].append(t)
    book["open"][mkt] = None
    book["exit_i"] = None


def open_paper(book, mkt, d, plan, x, rp, kind, now, fm):
    t = v7.open_trade(book, mkt, d, plan, x, rp, kind, now)
    res = v7.manage(t, fm, mkt)
    if res["status"] == "skipped":
        book["open"][mkt] = None
        book["skipped"] += 1
        return
    if res["status"] != "closed":          # still running at the end of the data: not counted
        book["exit_i"] = 10 ** 9
        t["_res"] = None
        return
    t["_res"] = res
    book["exit_i"] = int(fm.index.get_loc(pd.Timestamp(res["exit_ts"])))


def reasons(twin):
    out = []
    side, opp, ans = twin.get("side", 0), twin.get("opp", 0), twin.get("answered", 12)
    if side < v7.CFG["MIN_SIDE"]:
        out.append(f"fewer than {v7.CFG['MIN_SIDE']} votes on the trend side")
    if opp > v7.CFG["MAX_OPPOSITE"]:
        out.append("more than 1 vote against")
    if ans < v7.CFG["MIN_ANSWERED"]:
        out.append("fewer than 10 agents with data")
    for code, _txt in twin.get("blocks", []):
        out.append(BLOCK_NAMES.get(code, str(code).replace("_", " ")))
    return out


# ------------------------------------------------------------------ the replay
def replay(mkt, macro, fng):
    m = v7.MARKETS[mkt]
    now0 = v7.now_utc()
    h = v7.get_hourly(mkt)
    h = v7.closed_hourly(h, now0, m["data_delay_min"]) if h is not None else None
    if h is None or len(h) < 3000:
        return None, "not enough hourly prices from Yahoo"
    if mkt == "OIL":
        h, _ = v7.wti_back_adjust(h)
    f = v7.build_features(h, mkt)
    fm = f.drop(columns=[c for c in ("adj",) if c in f.columns])   # prices are already back-adjusted
    n = len(f)
    # start only when the daily view has its full history (as it always has live)
    daily = v7.daily_from_hourly(h, mkt)
    need_days = m["daily"][0] + 5
    start_day = daily.index[min(len(daily) - 1, need_days)]
    start = max(1000, int(np.searchsorted(f.index.values, np.datetime64(pd.Timestamp(start_day)))))
    log(f"{mkt}: {n} hourly candles, test starts at candle {start} ({f.index[start]:%d %b %Y})")
    books = {p[0]: new_book(mkt) for p in PASSES}
    bases = {b[0]: new_book(mkt) for b in BASELINES}
    trend = f["trend"].values
    stop_lo, stop_hi = f["stop_lo"].values, f["stop_hi"].values
    orig_agent = v7.agent_daily_view
    keep = {k: v7.CFG[k] for k in ("STOP_MAX_ATR", "MIN_SIDE")}
    weights = {a: 1.0 for a in v7.VOTERS}
    t_start, evaluated = time.time(), 0
    try:
        v7.agent_daily_view = gate_agent
        for i in range(start, n - 1):
            for bk in list(books.values()) + list(bases.values()):
                close_if_done(bk, mkt, i)
            d = {"UP": "BUY", "DOWN": "SELL"}.get(trend[i])
            if not d:
                continue
            flat_p = [p for p in PASSES if books[p[0]]["open"][mkt] is None]
            flat_b = [b for b in BASELINES if bases[b[0]]["open"][mkt] is None]
            if not flat_p and not flat_b:
                continue
            now = f.index[i] + pd.Timedelta(hours=1, minutes=m["data_delay_min"] + 7)
            shared = {"macro": macro_at(macro, now), "fng": fng_at(fng, now) if mkt == "BTC" else None,
                      "deriv": None, "flows": None, "events": [], "usdinr": v7.CFG["USDINR_DEFAULT"]}
            v7.CFG["STOP_MAX_ATR"] = 2.0
            x, why = v7.market_snapshot(mkt, h.iloc[:i + 1], now, Desk, shared, f.iloc[:i + 1])
            if x is None:
                continue
            evaluated += 1
            x["usdinr"] = shared["usdinr"]
            x["calendar_block"] = v7.calendar_block(mkt, x, now)
            x["view4"] = view_4h_state(x["h4"], m["lags4"])
            c, a = x["c"], x["atr"]
            # ---- trend-only base lines (the live trend twin: trend + valid stop + costs, no other filter)
            for label, cap in flat_b:
                v7.CFG["STOP_MAX_ATR"] = cap
                p = v7.plan_at(d, c, a, float(stop_lo[i]), float(stop_hi[i]), mkt)
                bk = bases[label]
                bk["hours"] += 1
                if not p.get("valid") or p["cost_r"] > v7.CFG["MAX_COST_R"]:
                    continue
                ok, _w, rp = v7.risk_check(bk, mkt, d, now, "trend_twin")
                if ok:
                    open_paper(bk, mkt, d, p, x, rp, "trend", now, fm)
            # ---- the four rule sets (the live rules-only twin)
            for label, gate, cap, min_side in flat_p:
                v7.CFG["STOP_MAX_ATR"] = cap
                v7.CFG["MIN_SIDE"] = min_side
                x["plan_buy"] = v7.plan_at("BUY", c, a, float(stop_lo[i]), float(stop_hi[i]), mkt)
                x["plan_sell"] = v7.plan_at("SELL", c, a, float(stop_lo[i]), float(stop_hi[i]), mkt)
                x["gate_state"] = x["daily_view"] if gate == "DAILY" else x["view4"]
                rules = v7.rule_answers(x)
                lv = rules["levels"]
                if lv.get("struct") is not None and lv["vote"] in ("BUY", "SELL"):    # same as the live run
                    key = "plan_buy" if lv["vote"] == "BUY" else "plan_sell"
                    lvl_plan = v7.plan_at(lv["vote"], c, a, lv["struct"], lv["struct"], mkt)
                    if lvl_plan.get("valid"):
                        x[key] = dict(lvl_plan, structure="zone")
                        if not x["shock"]:
                            facts = v7.fact_sheet(x, rules)
                            x["facts"] = facts
                            rules["bull"] = v7.agent_debater("bull", x, facts)
                            rules["bear"] = v7.agent_debater("bear", x, facts)
                twin = v7.decide(x, rules, weights, Desk, mkt, None)
                bk = books[label]
                bk["hours"] += 1
                if not twin["trade"]:
                    for r in reasons(twin):
                        bk["why"][r] += 1
                    continue
                half = x["gate_state"][0] == "NEUTRAL"
                ok, why_r, rp = v7.risk_check(bk, mkt, twin["d"], now, "rules_twin", half)
                if not ok:
                    bk["why"]["risk officer: " + why_r] += 1
                    continue
                kind = "level" if rules["levels"].get("signal") in v7.LEVEL_SIGNALS and \
                    rules["levels"]["vote"] == twin["d"] else "trend"
                open_paper(bk, mkt, twin["d"], twin["plan"], x, rp, kind, now, fm)
            if evaluated % 500 == 0:
                done = (i - start) / max(1, n - 1 - start)
                log(f"{mkt}: {done * 100:.0f}% done, {evaluated} trend hours checked, "
                    f"{time.time() - t_start:.0f}s")
    finally:
        v7.agent_daily_view = orig_agent
        v7.CFG.update(keep)
    for bk in list(books.values()) + list(bases.values()):
        close_if_done(bk, mkt, n + 10 ** 9)
    days = (f.index[-1] - f.index[start]).days
    return {"books": books, "bases": bases, "days": days, "from": f.index[start], "to": f.index[-1]}, ""


def line(label, bk, weeks):
    rs = [t["r"] for t in bk["closed"]]
    k = len(rs)
    if not k:
        return f"- {label}: no trades", 0, 0.0
    won = sum(1 for r in rs if r > 0)
    tot = float(np.sum(rs))
    money = float(np.sum([t.get("pnl_rs", 0.0) for t in bk["closed"]]))
    return (f"- {label}: {k} trades ({k / weeks:.1f} a week), won {won / k * 100:.0f}%, total {tot:+.1f}R, "
            f"average {tot / k:+.2f}R, money Rs {money:+,.0f}"), k, tot


def report(mkt, res):
    m = v7.MARKETS[mkt]
    weeks = max(1.0, res["days"] / 7.0)
    out = [f"FULL RULES TEST - {m['name']} ({res['days']} days: {res['from']:%d %b %Y} to {res['to']:%d %b %Y})",
           "Hourly replay of the rules-only twin with every code filter, the live trade management and fees. "
           f"Money on Rs {v7.CFG['CAPITAL']:,.0f} capital as the code sizes it.", ""]
    best = None
    for label, *_ in PASSES:
        txt, k, tot = line(label, res["books"][label], weeks)
        out.append(txt)
        if k >= 10 and (best is None or tot > best[1]):
            best = (label, tot, k)
    out.append("")
    out.append("Base line (no votes, no room rule, no Time Machine):")
    for label, _cap in BASELINES:
        out.append(line(label, res["bases"][label], weeks)[0])
    out.append("")
    for label in (PASSES[0][0], PASSES[2][0]):
        bk = res["books"][label]
        hrs = max(1, bk["hours"])
        top = bk["why"].most_common(6)
        if top:
            out.append(f"What stopped {label.split(' (')[0]} most (share of trend hours with no trade):")
            out.append("  " + " | ".join(f"{w} {c / hrs * 100:.0f}%" for w, c in top))
    out.append("")
    if best is None:
        out.append("Verdict: no rule set had 10 or more trades - too few to judge.")
    elif best[1] <= 0:
        out.append("Verdict: none of the rule sets made money here, even with every filter on.")
    else:
        out.append(f"Verdict: best here = {best[0]} ({best[2]} trades, {best[1]:+.1f}R).")
    out.append("Not in this test: the AI (news = WAIT, exactly like the live rules-only twin), live derivatives / "
               "fund-flow data (the Special agent has none), the economic-event calendar.")
    out.append("v2 is not here: its votes are made by Gemini and cannot be replayed on old data.")
    return "\n".join(out)


def main():
    arg = (sys.argv[1] if len(sys.argv) > 1 else "ALL").upper()
    markets = list(v7.MARKETS) if arg == "ALL" else [x.strip() for x in arg.split(",") if x.strip() in v7.MARKETS]
    if not markets:
        print("Use: python compare_full.py BTC | GOLD | OIL | ALL")
        return
    log("downloading macro prices ...")
    macro = macro_series()
    fng = fng_history() if "BTC" in markets else None
    for mkt in markets:
        t0 = time.time()
        try:
            res, why = replay(mkt, macro, fng)
            text = report(mkt, res) if res else f"FULL RULES TEST - {v7.MARKETS[mkt]['name']}: {why}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            text = f"FULL RULES TEST - {v7.MARKETS[mkt]['name']}: stopped with an error: {str(e)[:300]}"
        log(f"{mkt} finished in {time.time() - t0:.0f}s")
        print(text)
        v7.tg_send(text)


if __name__ == "__main__":
    main()
