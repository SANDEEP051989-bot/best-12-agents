#!/usr/bin/env python3
"""
AI TRADING TEAM v7 - Bitcoin, Gold and US Oil (WTI), hourly
============================================================
Built from the research report "BTC Gold Oil agent rules" (9 Nov 2026).

Main idea: CODE makes every number decision. The AI brains explain, read the news and argue.

Every hour (GitHub runs Bitcoin at minute 7, and Gold + US Oil at minute 37 because Yahoo shows their prices
30 minutes late):
  1. fresh closed hourly candles for BTC, Gold and US Oil
  2. the trend switch (one shared rule) - no trend = no trade and no AI calls
  3. gates: event windows, extreme volatility, shock / burst candles, cooldowns, open trade
  4. 12 agents: each rule is worked out in code, then the agent's own AI brain explains it.
     The code checker corrects any vote that goes against the agent's own rule.
  5. vote: 6 on one side, at most 1 against, 10 of 12 answered, trend + daily view agree, no blocker
  6. bull vs bear rebuttal (candidate trades only) -> risk officer (code)
     -> inspector (AI, can only BLOCK, test mode on Gold and Oil) -> desk head (code)
  7. Telegram: a full report for each market every 2 hours (numbered points, all 12 agents with their reasons,
     the code check and the AI inspector), a special card the moment a trade forms, short updates for open
     trades, a PDF of the whole day's hourly reports just after midnight India time, and the report card PDF
     on Sunday.

A "rules-only twin" (no AI at all) and a "trend-only twin" trade on paper beside the team,
so after about 100 trades per market you can see if the AI layer earns its place.

Commands:
  python agents_v7.py run          the hourly run for all markets (default)
  python agents_v7.py run BTC      only Bitcoin   (python agents_v7.py run GOLD,OIL = only gold and oil)
  python agents_v7.py test         checks Telegram, every data source and every AI brain
  python agents_v7.py report       sends the latest full report of each market again, now
  python agents_v7.py daily-pdf    sends today's PDF now
  python agents_v7.py weekly-pdf   sends the report card PDF now

Paper trades only by default. Not financial advice.
"""
import base64
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


def _clean_secret(name):
    """Pasted keys often carry a space, a line break, quotes, 'Bearer ' or 'NAME=' in front. Remove them."""
    v = os.environ.get(name)
    if v is None:
        return
    v = v.strip().strip('"').strip("'").strip()
    if v.startswith("export "):
        v = v[7:].strip()
    for pre in (name + "=", name + ":", "Bearer ", "bearer "):
        if v.startswith(pre):
            v = v[len(pre):].strip().strip('"').strip("'").strip()
    os.environ[name] = v


for _k in ("TG_TOKEN", "TG_CHAT", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "SHEET_ID", "SHEET_NAME"):
    _clean_secret(_k)

VERSION = "v7.0"
IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")
STATE_DIR = os.getenv("STATE_DIR", "state_v7")
PDF_DIR = os.getenv("PDF_DIR") or os.path.join(tempfile.gettempdir(), "ai_team_v7_pdf")   # not saved in the repo
DRY_RUN = os.getenv("DRY_RUN") == "1"


def _env_num(name, default):
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return float(default)


CFG = {
    "CAPITAL": _env_num("CAPITAL", 20000),     # rupees
    "RISK_PCT_START": 0.5,                     # % per trade for the first 50 graded trades
    "RISK_PCT_MAX": min(1.0, _env_num("RISK_PCT", 1.0)),
    "MIN_SIDE": _env_num("MIN_SIDE", 6),       # 6 or more agents on one side
    "MAX_OPPOSITE": 1,                         # 1 or fewer agents against (counted, not weighted)
    "MIN_ANSWERED": 10,                        # at least 10 of 12 answered
    "STOP_MIN_ATR": 1.5, "STOP_MAX_ATR": 2.0, "STOP_BUFFER_ATR": 0.25, "ROUND_EXTRA_ATR": 0.1,
    # The swing point a trade's stop sits behind. The research's notes say "a swing low/high from the last 12
    # bars"; its levels table defines 1-hour swing points with 10 candles each side. In tests the last 10-candle
    # swing point was a median 4-5 ATR away, so the 2.0 ATR cap rejected about 90% of setups. Default 3 candles
    # each side; set SWING_K=10 for the strict version.
    "SWING_K": int(_env_num("SWING_K", 3)),
    "SWING_MAX_AGE": 72,                       # the swing point used for the stop is at most 72 candles old
    "SWING_BARS": 12,                          # 12-candle extreme, used only for the zone-trade wick check
    "TARGET_R": 2.0, "MAX_BARS": 24, "MAX_COST_R": 0.2,
    "SHOCK_ATR": 2.5, "BURST_ATR": 2.0,
    "DATA_MAX_AGE_MIN": 70, "MACRO_MAX_AGE_MIN": 90,
    "AI_MAX_CALLS": int(_env_num("AI_MAX_CALLS", 90)),
    "AI_TIMEOUT": 45, "AI_GAP_S": float(_env_num("AI_GAP_S", 2.5)),
    "RUN_DEADLINE_MIN": _env_num("RUN_DEADLINE_MIN", 22),  # no new AI calls after this many minutes (job limit 40)
    "FILL_LAG_MIN": 10,                        # paper fill = first candle that starts after (decision time - 10 min)
    # Telegram: "2h" = full report of each market every 2 hours + a special card for each trade (default),
    # "all" = full report every hour, "trades" = only the trade cards and trade updates
    "TG_MODE": {"all": "all", "1h": "all", "hourly": "all", "every hour": "all", "trades": "trades",
                "trade": "trades", "only trades": "trades", "off": "trades"}.get(
        (os.getenv("TG_MODE") or "2h").strip().lower(), "2h"),          # anything else = "2h"
    "REAL_MONEY": {m.strip().upper() for m in (os.getenv("REAL_MONEY") or "").split(",") if m.strip()},
    "USDINR_DEFAULT": 88.0,
    "BTC_CONTRACT": 0.001,                     # Delta Exchange India: 0.001 BTC per contract
}

MARKETS = {
    "BTC": {"name": "Bitcoin", "ticker": "BTC-USD", "is_247": True, "lags": [24, 72, 168, 336],
            "lags4": [6, 18, 42, 84], "daily": (100, 20, 10), "round_big": 5000, "round_bonus": 10000,
            "round_pts": 2.0, "round_small": 1000, "zone_pct": (0.3, 1.0), "fade_days": 10, "trail_atr": 3.0,
            # Delta Exchange India: limit (maker) entry 0.02% + stop (taker) exit 0.05%, +18% GST on fees = 0.083%,
            # plus about 0.035% for spread and exit slippage = 0.12% round trip. Market orders both ways would be
            # about 0.17%, and with this market's normal 1-hour move that is more than the 0.2R cost limit
            # (research: "use limit orders for entries where possible"). The trade card says: use a limit order.
            "cost_pct": 0.12, "decimals": 0, "signals": ("break", "retest"), "paper_only": False,
            "fill_bars": 1, "data_delay_min": 0},
    "GOLD": {"name": "Gold", "ticker": "GC=F", "is_247": False, "lags": [24, 72, 120, 240],
             "lags4": [6, 18, 30, 60], "daily": (150, 40, 20), "round_big": 100, "round_bonus": 1000,
             "round_pts": 2.0, "round_small": 50, "zone_pct": (0.15, 0.5), "fade_days": 10, "trail_atr": 2.5,
             "cost_pct": 0.05, "decimals": 1, "signals": ("break", "sweep"), "paper_only": False,
             "fill_bars": 2, "data_delay_min": 30},     # Yahoo shows COMEX 30 min late: decided at :37
    "OIL": {"name": "US Oil (WTI)", "ticker": "CL=F", "is_247": False, "lags": [24, 72, 120, 240],
            "lags4": [6, 18, 30, 60], "daily": (120, 20, 10), "round_big": 10, "round_bonus": None,
            "round_pts": 1.0, "round_small": 1, "zone_pct": (0.3, 1.0), "fade_days": 7, "trail_atr": 3.0,
            "cost_pct": 0.10, "decimals": 2, "signals": ("sweep", "eia_fade"), "paper_only": True,
            "fill_bars": 2, "data_delay_min": 30},      # Yahoo shows NYMEX 30 min late: decided at :37
}

AGENTS = ["trend", "daily_view", "momentum", "volatility", "levels", "special", "macro", "news", "mood",
          "bull", "bear", "time_machine"]
VOTERS = [a for a in AGENTS if a != "daily_view"]          # the daily view is a gate, not a voter
OWN_JUDGEMENT_OK = {a for a in VOTERS if a != "time_machine"}   # a BUY/SELL while the rule is neutral is kept
# A WAIT that only means "nothing is happening" (neutral) - the AI may add its own vote. Every other WAIT is a
# deliberate block (gold no-chase, divergence, inside a zone, burst candle, ...) and must stay WAIT.
NEUTRAL_RULES = {"mom.neutral", "vol.normal", "lvl.no_signal", "lvl.none_near", "spc.neutral", "spc.flows_neutral",
                 "mac.gold_neutral", "mac.btc_wait", "mac.oil_wait", "mood.middle"}
AI_NATIVE = {"news", "bull", "bear"}
LEVEL_SIGNALS = ("break", "retest", "sweep", "eia_fade")    # trades on these close everything at 2R (research)
NO_AI = {"time_machine"}
PRETTY = {"trend": "Trend", "daily_view": "Daily view", "momentum": "Momentum", "volatility": "Volatility",
          "levels": "Levels", "special": "Special", "macro": "Macro", "news": "News", "mood": "Mood",
          "bull": "Bull", "bear": "Bear", "time_machine": "Time machine"}

# preferred AI models (names change often - the program checks which ones exist once a day)
PREFS = {
    ("gemini", "lite"): ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"],
    ("gemini", "flash"): ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"],
    ("groq", "big"): ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    ("groq", "mid"): ["qwen/qwen3.8-27b", "qwen/qwen3-32b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
    ("groq", "small"): ["openai/gpt-oss-20b", "llama-3.1-8b-instant"],
    ("groq", "alt"): ["llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct",
                      "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"],
    ("groq", "tiny"): ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
    ("openrouter", "big"): ["nvidia/nemotron-3-ultra-550b:free", "deepseek/deepseek-chat-v3.1:free",
                            "meta-llama/llama-3.3-70b-instruct:free"],
    ("openrouter", "a"): ["inclusionai/ling-3.0-flash:free", "google/gemma-4-31b-it:free",
                          "meta-llama/llama-3.3-70b-instruct:free"],
    ("openrouter", "b"): ["google/gemma-4-31b-it:free", "poolside/laguna-s-2.1:free",
                          "meta-llama/llama-3.3-70b-instruct:free"],
    ("openrouter", "c"): ["poolside/laguna-s-2.1:free", "meta-llama/llama-3.3-70b-instruct:free",
                          "google/gemma-4-31b-it:free"],
}
# Main brain first, then backups on other providers. Free limits (late 2026): Groq about 1,000 requests a day per
# model, OpenRouter free models 50 a day (1,000 after a one-time $10 credit), Gemini free tier very small.
# So most roles start on Groq models, a few on OpenRouter and Gemini; bull and bear are on different providers.
ROLE_MODELS = {
    "trend": [("groq", "small"), ("gemini", "lite"), ("openrouter", "a")],
    "daily_view": [("groq", "tiny"), ("gemini", "lite"), ("openrouter", "b")],
    "momentum": [("gemini", "lite"), ("groq", "alt"), ("openrouter", "c")],
    "volatility": [("groq", "alt"), ("gemini", "lite"), ("openrouter", "a")],
    "levels": [("groq", "mid"), ("gemini", "lite"), ("openrouter", "b")],
    "special": [("groq", "small"), ("gemini", "lite"), ("openrouter", "c")],
    "macro": [("openrouter", "a"), ("groq", "alt"), ("gemini", "lite")],
    "mood": [("openrouter", "b"), ("groq", "tiny"), ("gemini", "lite")],
    "bull": [("openrouter", "c"), ("groq", "alt"), ("gemini", "lite")],
    "bear": [("groq", "mid"), ("gemini", "lite"), ("openrouter", "a")],
    "news": [("groq", "big"), ("openrouter", "big"), ("gemini", "flash"), ("gemini", "lite")],
    "inspector": [("groq", "big"), ("gemini", "flash"), ("openrouter", "big"), ("gemini", "lite")],
}
BRAIN_NAME = {"gemini": "Gemini", "groq": "Groq", "openrouter": "OpenRouter"}


# =============================================================== small helpers
def log(*a):
    print(dt.datetime.now(IST).strftime("%H:%M:%S"), *a, flush=True)


def now_utc():
    """Current time in UTC without timezone. NOW_UTC=2026-11-09T12:07 can fake it for tests."""
    v = os.getenv("NOW_UTC")
    if v:
        return pd.Timestamp(v).to_pydatetime().replace(tzinfo=None, microsecond=0)
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)   # whole seconds: saved times parse


def utc_ts(x):
    t = pd.Timestamp(x)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def to_ist(ts):
    return utc_ts(ts).tz_localize("UTC").tz_convert(IST)


def to_ny(ts):
    return utc_ts(ts).tz_localize("UTC").tz_convert(NY)


def ist_str(ts, fmt="%d %b %H:%M"):
    return to_ist(ts).strftime(fmt) + " IST"


def fnum(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:,.{d}f}"


def pxm(x, mkt):
    return fnum(x, MARKETS[mkt]["decimals"])


def finite(*xs):
    try:
        return all(x is not None and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def safe_str(s):
    """Text without broken characters (a lone half of an emoji from a feed or an AI breaks encoding)."""
    return str(s).encode("utf-8", "replace").decode("utf-8")


def clip(text, n):
    text = " ".join(safe_str(text or "").split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def r3(x):
    return float(round(float(x), 3))


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", str(text))
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        pass
    try:                                         # valid JSON followed by stray text such as another '}'
        obj, _ = json.JSONDecoder().raw_decode(t[s:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def wilson(k, n, z=1.96):
    """95% likely range for a hit rate."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def opposite(d):
    return {"BUY": "SELL", "SELL": "BUY"}.get(d, "WAIT")


def iso_week(ts):
    y, w, _ = utc_ts(ts).isocalendar()
    return f"{y}-W{w:02d}"


def latin1(text):
    rep = {"₹": "Rs ", "×": "x", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "’": "'", "‘": "'",
           "“": '"', "”": '"', "…": "...", "•": "-", "±": "+/-"}
    for k, v in rep.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


# =============================================================== saved state
def _p(*parts):
    path = os.path.join(STATE_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_json(name, default):
    path = _p(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:
        raise RuntimeError(f"saved file {STATE_DIR}/{name} is damaged ({e}) - restore it from the repo history") from e


def save_json(name, obj):
    path = _p(name)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)


class State:
    FILES = ("books", "grading", "inspector", "models", "deriv", "cache", "misc")

    def __init__(self):
        for f in self.FILES:
            setattr(self, f, load_json(f + ".json", {}))
        b = self.books
        for name in ("system", "rules_twin", "trend_twin"):
            bk = b.setdefault(name, {})
            bk.setdefault("open", {})
            bk.setdefault("closed", [])
            bk.setdefault("risk", {})
        self.grading.setdefault("pending", [])
        self.grading.setdefault("graded", [])
        self.grading.setdefault("vote_log", [])
        self.inspector.setdefault("status", {"BTC": "live", "GOLD": "test", "OIL": "test"})
        self.inspector.setdefault("pending", [])
        self.inspector.setdefault("graded", [])
        for k in ("skip_until", "fail_streak", "swap", "contra", "errors", "discovered", "weights"):
            self.models.setdefault(k, {})
        self.deriv.setdefault("snaps", [])
        self.deriv.setdefault("fund_hist", {})
        self.misc.setdefault("blocks_until", {})
        self.misc.setdefault("weights", {})

    def save(self):
        for f in self.FILES:
            save_json(f + ".json", getattr(self, f))

    def add_report(self, rec):
        day = to_ist(rec["time"]).strftime("%Y-%m-%d")
        path = _p("reports", f"{day}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def reports_for(self, day):
        path = _p("reports", f"{day}.jsonl")
        out = []
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        return out

    def prune(self, now):
        g = self.grading
        g["graded"] = g["graded"][-6000:]
        g["vote_log"] = g["vote_log"][-900:]
        cut = (now - dt.timedelta(days=6)).isoformat()          # weekends and holidays can stretch 24 bars
        g["pending"] = [p for p in g["pending"] if p["signal_ts"] >= cut]
        for name in self.books:
            self.books[name]["closed"] = self.books[name]["closed"][-600:]
        self.inspector["graded"] = self.inspector["graded"][-500:]
        cut15 = (now - dt.timedelta(days=15)).isoformat()
        self.deriv["snaps"] = [s for s in self.deriv["snaps"] if s["ts"] >= cut15]
        for v, rows in self.deriv["fund_hist"].items():
            self.deriv["fund_hist"][v] = [r for r in rows if r[0] >= (now - dt.timedelta(days=35)).isoformat()]
        old = (now - dt.timedelta(days=21)).strftime("%Y-%m-%d")
        for k in list(self.models["errors"]):
            if k.split("|")[0] < old:
                del self.models["errors"][k]
        keep_wk = iso_week(now - dt.timedelta(days=56))
        for k in list(self.models["contra"]):
            if k.split("|")[0] < keep_wk:
                del self.models["contra"][k]
        seen = self.cache.get("news_seen", {})
        cut2 = (now - dt.timedelta(days=2)).isoformat()
        self.cache["news_seen"] = {k: v for k, v in seen.items() if v >= cut2}
        rep_dir = os.path.join(STATE_DIR, "reports")
        if os.path.isdir(rep_dir):
            keep = (to_ist(now) - pd.Timedelta(days=9)).strftime("%Y-%m-%d")
            for fn in os.listdir(rep_dir):
                if fn.endswith(".jsonl") and fn[:10] < keep:
                    os.remove(os.path.join(rep_dir, fn))


# =============================================================== web helpers
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"}
SOURCES = {}   # name -> "ok" / "error ..." (shown in the test message and the PDF)


def note_source(name, ok, detail=""):
    SOURCES[name] = "ok" if ok else f"error {detail}".strip()


def http_get(url, params=None, headers=None, timeout=20, tries=2):
    """Returns (status, text). status 0 = network error."""
    last = (0, "")
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
            if r.status_code == 200:
                return 200, r.text
            last = (r.status_code, r.text[:300])
            if r.status_code in (400, 401, 403, 404):
                break
        except Exception as e:
            last = (0, str(e)[:200])
        if k + 1 < tries:
            time.sleep(1.5 * (k + 1))
    return last


def http_post_json(url, body, headers=None, timeout=20):
    try:
        r = requests.post(url, json=body, headers={**UA, **(headers or {})}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def http_json(url, params=None, headers=None, timeout=20):
    status, text = http_get(url, params=params, headers=headers, timeout=timeout)
    if status != 200:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def fetch_yahoo(ticker, period, interval):
    """Yahoo Finance candles as a DataFrame (UTC index without timezone)."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    for k in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, actions=False)
            if df is not None and not df.empty:
                time.sleep(0.3)                     # be gentle with Yahoo
                return clean_ohlc(df)
        except Exception as e:
            log(f"Yahoo {ticker} retry {k + 1}: {clip(e, 120)}")
        if k < 2:
            time.sleep(3 * (k + 1))
    return None


def clean_ohlc(df):
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({k.title(): df[cols[k]] for k in ("open", "high", "low", "close") if k in cols})
    out["Volume"] = df[cols["volume"]] if "volume" in cols else 0.0
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    out = out[(out["Low"] >= 0.5 * out["Close"]) & (out["High"] <= 2.0 * out["Close"])]   # broken bars (Low 0 ...)
    o, h, l, c = out["Open"], out["High"], out["Low"], out["Close"]
    h = np.maximum(h, c)                                # the close is trusted: High/Low must contain it
    l = np.minimum(l, c)
    pc = c.shift(1)
    bad_open = (o > h) | (o < l)                        # an Open outside the candle is a bad print:
    o = o.where(~bad_open, pc.fillna(o).clip(lower=l, upper=h))      # use the previous close instead
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    l = l.where(l >= 0.8 * body_lo, body_lo)            # a wick more than 20% beyond the body is a bad tick
    h = h.where(h <= 1.25 * body_hi, body_hi)           # (no real hourly candle of BTC, gold or WTI does that)
    out["Open"], out["High"], out["Low"] = o, h, l
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)
    return out.astype(float)


def fetch_kraken_btc():
    """Backup BTC hourly candles (last 720 hours) if Yahoo fails."""
    j = http_json("https://api.kraken.com/0/public/OHLC", params={"pair": "XBTUSD", "interval": 60})
    try:
        res = j["result"]
        key = [k for k in res if k != "last"][0]
        rows = res[key]
        df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "vwap", "Volume", "n"])
        df.index = pd.to_datetime(df["t"].astype(int), unit="s")
        return clean_ohlc(df[["Open", "High", "Low", "Close", "Volume"]])
    except Exception:
        return None


def closed_hourly(df, now, delay_min=0):
    """Only finished hourly candles that start on a full hour. delay_min = how late Yahoo shows this market:
    a candle is complete only once the delayed data has passed its end."""
    if df is None or df.empty:
        return df
    df = df[(df.index.minute == 0) & (df.index.second == 0)]
    return df[df.index + pd.Timedelta(hours=1, minutes=delay_min) <= pd.Timestamp(now)]


def closed_any(df, now, delay_min=0):
    """Finished hourly candles at any minute (US indexes such as ^VIX and ^TNX start their candles at :30)."""
    if df is None or df.empty:
        return df
    return df[df.index + pd.Timedelta(hours=1, minutes=delay_min) <= pd.Timestamp(now)]


CL_CODES = "FGHJKMNQUVXZ"


def cl_expiry(year, month):
    """Last trading day of the WTI (CL) contract for delivery month year/month: 3 business days before the
    25th of the month before (4 if the 25th is not a business day). US holidays are ignored."""
    py, pm = (year, month - 1) if month > 1 else (year - 1, 12)
    d = dt.date(py, pm, 25)
    need = 3 if d.weekday() < 5 else 4
    n = 0
    while n < need:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def cl_front_months(day, skip_days=2):
    """(front, second) delivery months trading on 'day'; the front rolls skip_days before its expiry."""
    y, m = day.year, day.month
    while cl_expiry(y, m) < day + dt.timedelta(days=skip_days):
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    y2, m2 = (y, m + 1) if m < 12 else (y + 1, 1)
    return (y, m), (y2, m2)


def wti_back_adjust(h):
    """Yahoo's CL=F jumps when it rolls to the next contract. Around each monthly expiry, the biggest jump
    between one candle's close and the next candle's open (if 0.5x ATR or more) is treated as the roll gap
    and added to all earlier candles, so the jump does not create fake trends or fake levels."""
    if h is None or len(h) < 50:
        return h, []
    a = atr(h, 14).values
    idx = h.index
    O, C = h["Open"].values, h["Close"].values
    adj = np.zeros(len(h))
    rolls = []
    y, m = idx[0].year, idx[0].month
    end = (idx[-1] + pd.Timedelta(days=40)).date()
    while dt.date(y, m, 1) <= end:
        exp = pd.Timestamp(cl_expiry(y, m))
        lo = int(np.searchsorted(idx.values, np.datetime64(exp - pd.Timedelta(days=3))))
        hi = int(np.searchsorted(idx.values, np.datetime64(exp + pd.Timedelta(days=3))))
        best, best_i = 0.0, None
        for i in range(max(lo, 15), min(hi, len(h))):
            if finite(a[i - 1]) and a[i - 1] > 0:
                size = abs(O[i] - C[i - 1]) / a[i - 1]
                if size > best:
                    best, best_i = size, i
        if best_i is not None and best >= 0.5:
            jump = float(O[best_i] - C[best_i - 1])
            adj[:best_i] += jump
            rolls.append({"time": idx[best_i].isoformat(), "jump": round(jump, 3), "atr_x": round(best, 2)})
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    out = h.copy()
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col] + adj
    out["Adj"] = adj          # how much each candle was shifted (saved trades are shifted the same way)
    return out, rolls


# =============================================================== indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(dn != 0, 100.0)


def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd_hist(c):
    m = ema(c, 12) - ema(c, 26)
    return m - m.ewm(span=9, adjust=False, min_periods=9).mean()


def trend_series(close, lags):
    """The shared trend switch: price now vs 4 earlier closes. 3 of 4 higher = UP, 3 of 4 lower = DOWN."""
    ups = sum((close > close.shift(l)).astype(int) for l in lags)
    downs = sum((close < close.shift(l)).astype(int) for l in lags)
    valid = close.shift(max(lags)).notna()
    s = np.where(~valid, "NONE", np.where(ups >= 3, "UP", np.where(downs >= 3, "DOWN", "NONE")))
    return pd.Series(s, index=close.index), ups, downs


def pivots(H, L, k):
    """Swing highs / lows with k candles on each side. Returns lists of (index, price)."""
    hi, lo = [], []
    n = len(H)
    for i in range(k, n - k):
        if H[i] >= H[i - k:i + k + 1].max():
            hi.append((i, float(H[i])))
        if L[i] <= L[i - k:i + k + 1].min():
            lo.append((i, float(L[i])))
    return hi, lo


def day_keys(idx, mkt):
    """Trading-day label of each candle. BTC: UTC day. Gold and Oil: the day starts at 17:00 New York."""
    idx = pd.DatetimeIndex(idx)
    if MARKETS[mkt]["is_247"]:
        return idx.normalize()
    ny = idx.tz_localize("UTC").tz_convert(NY) + pd.Timedelta(hours=7)
    return pd.DatetimeIndex(ny.tz_localize(None)).normalize()


def week_keys(dkeys):
    dk = pd.DatetimeIndex(dkeys)
    return dk - pd.to_timedelta(dk.weekday, unit="D")


def resample_4h(h):
    g = h.resample("4h", origin="epoch", label="left", closed="left")
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()}).dropna()
    if not len(h):
        return out
    last_end = h.index[-1] + pd.Timedelta(hours=1)
    return out[out.index + pd.Timedelta(hours=4) <= last_end]


def daily_from_hourly(h, mkt):
    dk = day_keys(h.index, mkt)
    g = h.groupby(dk)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                         "Close": g["Close"].last()})


# =============================================================== the market picture (all code)
def build_features(h, mkt):
    """Hourly features for every closed candle (used live and by the Time Machine)."""
    m = MARKETS[mkt]
    O, H, L, C, V = h["Open"], h["High"], h["Low"], h["Close"], h["Volume"]
    f = pd.DataFrame(index=h.index)
    f["o"], f["h"], f["l"], f["c"], f["v"] = O, H, L, C, V
    f["adj"] = h["Adj"] if "Adj" in h.columns else 0.0     # WTI roll shift of each candle (0 elsewhere)
    f["atr"] = atr(h, 14)
    f["atr_prev"] = f["atr"].shift(1)
    f["atr22"] = atr(h, 22)
    f["atr24"] = atr(h, 24)
    f["rsi"] = rsi(C)
    f["hist"] = macd_hist(C)
    mid = C.rolling(20).mean()
    sd = C.rolling(20).std(ddof=0)
    f["bb_up"], f["bb_lo"], f["bb_w"] = mid + 2 * sd, mid - 2 * sd, (4 * sd) / mid
    a20 = atr(h, 20)
    f["kc_up"], f["kc_lo"] = mid + 1.5 * a20, mid - 1.5 * a20
    f["squeeze"] = (f["bb_up"] < f["kc_up"]) & (f["bb_lo"] > f["kc_lo"])
    f["range"] = H - L
    f["shock"] = f["range"] >= CFG["SHOCK_ATR"] * f["atr_prev"]
    tr, ups, downs = trend_series(C, m["lags"])
    f["trend"], f["t_ups"], f["t_downs"] = tr, ups, downs
    f["swing_lo"] = L.rolling(CFG["SWING_BARS"]).min()
    f["swing_hi"] = H.rolling(CFG["SWING_BARS"]).max()
    # The swing point a trade depends on: the most recent confirmed 1-hour swing low below the price (for a BUY)
    # or swing high above it (for a SELL), at most 72 candles old. A swing point with k candles on each side is
    # known k candles later. None found = no valid stop (no trade) - the stop is never widened.
    f["stop_lo"], f["stop_hi"] = swing_stops(L.values, H.values, C.values, CFG["SWING_K"], CFG["SWING_MAX_AGE"])
    # 4-hour trend state (latest finished 4H candle)
    h4 = resample_4h(h)
    if len(h4) > max(m["lags4"]) + 1:
        t4, _, _ = trend_series(h4["Close"], m["lags4"])
        ends = pd.DataFrame({"end": h4.index + pd.Timedelta(hours=4), "t4": t4.values})
        q = pd.DataFrame({"end": h.index + pd.Timedelta(hours=1)})
        mm = pd.merge_asof(q, ends, on="end", direction="backward")
        f["trend4"] = mm["t4"].fillna("NONE").values
    else:
        f["trend4"] = "NONE"
    # yesterday's high / low and distance groups (Time Machine)
    dk = day_keys(h.index, mkt)
    daily = daily_from_hourly(h, mkt)
    prev = daily.shift(1)
    f["pd_high"] = prev["High"].reindex(dk).values
    f["pd_low"] = prev["Low"].reindex(dk).values
    dist = np.minimum((C - f["pd_high"]).abs(), (C - f["pd_low"]).abs()) / f["atr"]
    f["dist_g"] = np.where(dist < 1, 0, np.where(dist <= 3, 1, 2))
    f["atr_t"] = (f["atr"].rolling(250, min_periods=50).rank(pct=True).fillna(0.5) * 3).clip(0, 2.999).astype(int)
    f["rsi_g"] = np.where(f["rsi"] < 40, 0, np.where(f["rsi"] <= 60, 1, 2))
    hr = f.index.hour
    f["session"] = np.where((hr >= 7) & (hr < 13), 1, np.where((hr >= 13) & (hr < 21), 2, 0))
    f["wday"] = pd.DatetimeIndex(dk).weekday
    f["dkey"] = dk
    return f


def swing_stops(L, H, C, k, max_age):
    """For every candle: the latest confirmed swing low under the close and swing high over the close."""
    n = len(C)
    lo_out, hi_out = np.full(n, np.nan), np.full(n, np.nan)
    piv_lo, piv_hi = [], []                     # (index, price) of confirmed swing points, newest last
    for i in range(n):
        j = i - k                               # candle j is confirmed as a swing point at candle i
        if j - k >= 0:
            if L[j] == L[j - k:j + k + 1].min():
                piv_lo.append((j, L[j]))
            if H[j] == H[j - k:j + k + 1].max():
                piv_hi.append((j, H[j]))
        while piv_lo and piv_lo[0][0] < i - max_age:
            piv_lo.pop(0)
        while piv_hi and piv_hi[0][0] < i - max_age:
            piv_hi.pop(0)
        for _, p in reversed(piv_lo):
            if p < C[i]:
                lo_out[i] = p
                break
        for _, p in reversed(piv_hi):
            if p > C[i]:
                hi_out[i] = p
                break
    return lo_out, hi_out


def round_levels(price, step, lo, hi):
    if not step or not finite(price):
        return []
    a = math.floor(lo / step) * step
    out = []
    x = a
    while x <= hi + 1e-9:
        if x > 0:
            out.append(round(x, 6))
        x += step
    return out


def plan_at(d, entry, atr_v, swing_lo, swing_hi, mkt):
    """swing_lo / swing_hi = the structure the setup depends on (swing point, or zone edge for level trades)."""
    """Stop 0.25 ATR beyond the 12-bar swing, +0.1 ATR past a round number just inside it,
    at least 1.5 ATR away; if the structure needs more than 2.0 ATR the trade is not valid.
    Target 2x the stop; T1-T4 = 1R-4R."""
    if not finite(entry, atr_v) or atr_v <= 0 or d not in ("BUY", "SELL"):
        return {"valid": False, "why": "no data for a stop"}
    m = MARKETS[mkt]
    b = CFG["STOP_BUFFER_ATR"] * atr_v
    steps = [s for s in (m["round_small"], m["round_big"]) if s]
    if d == "BUY":
        if not finite(swing_lo):
            return {"valid": False, "why": "no 1-hour swing low below the price in the last 3 days"}
        stop = swing_lo - b
        if any(stop < r <= stop + b for s in steps for r in round_levels(stop, s, stop, stop + b)):
            stop -= CFG["ROUND_EXTRA_ATR"] * atr_v
        need = entry - stop
    else:
        if not finite(swing_hi):
            return {"valid": False, "why": "no 1-hour swing high above the price in the last 3 days"}
        stop = swing_hi + b
        if any(stop - b <= r < stop for s in steps for r in round_levels(stop, s, stop - b, stop)):
            stop += CFG["ROUND_EXTRA_ATR"] * atr_v
        need = stop - entry
    if need > CFG["STOP_MAX_ATR"] * atr_v:
        return {"valid": False, "why": f"the stop would need {need / atr_v:.1f}x ATR (more than 2.0x)"}
    dist = max(need, CFG["STOP_MIN_ATR"] * atr_v)
    sgn = 1 if d == "BUY" else -1
    stop = entry - sgn * dist
    cost_r = m["cost_pct"] / 100.0 * entry / dist
    return {"valid": True, "d": d, "entry": float(entry), "stop": float(stop), "dist": float(dist),
            "atr": float(atr_v), "cost_r": float(cost_r),
            "t": [float(entry + sgn * k * dist) for k in (1, 2, 3, 4)]}


def fill_index(idx, pos, fill_after=None):
    """Index of the candle whose open is the paper fill: the first candle after the signal candle that starts
    at or after 'fill_after' (decision time minus 10 minutes). Gold/oil are decided at :37 -> next hour's open."""
    start = pos + 1
    if fill_after:
        start = max(start, int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(fill_after)))))
    return start


def sim_fixed(O, H, L, C, start, d, stop, target, dist, cost_r, max_bars=None):
    """The standard test trade: buy at the next candle's open, fixed stop, 2R target, 24 candles,
    costs included, stop and target in the same candle = loss.
    Returns None if not finished yet, {"skip": True} if the open jumped past the stop or target."""
    max_bars = max_bars or CFG["MAX_BARS"]
    n = len(O)
    if start >= n:
        return None
    fill = float(O[start])
    if d == "BUY":
        if fill <= stop or fill >= target:
            return {"skip": True}
    else:
        if fill >= stop or fill <= target:
            return {"skip": True}
    for j in range(start, min(n, start + max_bars)):
        if d == "BUY":
            hs, ht = L[j] <= stop, H[j] >= target
            gap = O[j] < stop
        else:
            hs, ht = H[j] >= stop, L[j] <= target
            gap = O[j] > stop
        if hs:
            ex = float(O[j]) if gap else stop          # a gap through the stop fills at the open (worse)
            return {"r": (ex - fill) / dist * (1 if d == "BUY" else -1) - cost_r, "hit": False, "bars": j - start + 1,
                    "exit": float(ex), "why": "stop"}
        if ht:
            ex = target
            return {"r": (ex - fill) / dist * (1 if d == "BUY" else -1) - cost_r, "hit": True, "bars": j - start + 1,
                    "exit": float(ex), "why": "target"}
    if start + max_bars > n:
        return None
    ex = float(C[start + max_bars - 1])
    return {"r": (ex - fill) / dist * (1 if d == "BUY" else -1) - cost_r, "hit": False, "bars": max_bars,
            "exit": ex, "why": "time"}


# =============================================================== agents (rules in code)
def A(agent, vote, rule_id, reason, data=None, **kw):
    out = {"agent": agent, "vote": vote, "rule_id": rule_id, "reason": reason, "data": data or {},
           "blocks": [], "support": None, "no_data": False, "notes": []}
    out.update(kw)
    return out


def agent_trend(x):
    m = MARKETS[x["mkt"]]
    t, ups, downs = x["trend"], x["t_ups"], x["t_downs"]
    vote = {"UP": "BUY", "DOWN": "SELL"}.get(t, "WAIT")
    rid = {"UP": "trend.up", "DOWN": "trend.down"}.get(t, "trend.none")
    reason = f"price is higher than {ups} and lower than {downs} of the closes {m['lags']} candles ago"
    if x["burst"]:
        vote, rid = "WAIT", "trend.burst_wait"
        reason += "; the last candle was a burst of 2x ATR in the trend direction - wait one candle"
    data = {"close": x["c"], "checks_higher": ups, "checks_lower": downs}
    for lag, val in zip(m["lags"], x["lag_closes"]):
        data[f"close_{lag}_bars_ago"] = val
    return A("trend", vote, rid, reason, data)


def daily_view_state(dcl, wcl, slow_n, fast_n, wk_n):
    """Fast + slow daily checks (4 states) plus the weekly check. Uses finished days and weeks only."""
    c = dcl.dropna()
    if len(c) < slow_n + 3 or len(wcl.dropna()) < wk_n + 1:
        return "NEUTRAL", "not enough daily history", {}
    slow_up = c.iloc[-1] > c.iloc[-1 - slow_n]
    sign = (c > c.shift(fast_n)).iloc[fast_n:]
    win = sign.iloc[-60:]
    state = bool(win.iloc[0])
    vals = list(win.values)
    for i in range(1, len(vals)):
        if vals[i] != state and vals[i - 1] == vals[i]:
            state = bool(vals[i])                       # flipped for 2 closes in a row
    fast_up = state
    phase = {(True, True): "Bull", (True, False): "Correction", (False, False): "Bear",
             (False, True): "Rebound"}[(bool(slow_up), fast_up)]
    w = wcl.dropna()
    wma = w.rolling(wk_n).mean().iloc[-1]
    wc = w.iloc[-1]
    if phase in ("Bull", "Rebound") and wc > wma:
        ans = "UP"
    elif phase in ("Bear", "Correction") and wc < wma:
        ans = "DOWN"
    else:
        ans = "NEUTRAL"
    data = {"daily_close": float(c.iloc[-1]), f"close_{slow_n}_days_ago": float(c.iloc[-1 - slow_n]),
            f"close_{fast_n}_days_ago": float(c.iloc[-1 - fast_n]), "weekly_close": float(wc),
            f"weekly_avg_{wk_n}": float(wma)}
    why = f"{phase} (slow {'up' if slow_up else 'down'}, fast {'up' if fast_up else 'down'}), weekly close " \
          f"{'above' if wc > wma else 'below'} its {wk_n}-week average"
    return ans, why, data


def agent_daily_view(x):
    ans, why, data = x["daily_view"]
    vote = {"UP": "BUY", "DOWN": "SELL"}.get(ans, "WAIT")
    return A("daily_view", vote, f"daily.{ans.lower()}", f"{ans}: {why}", data, gate=ans)


def momentum_state(x):
    r, rp, hst, hp = x["rsi"], x["rsi_prev"], x["hist"], x["hist_prev"]
    if not finite(r, rp, hst, hp, x["close_24"]):
        return "neutral"
    if r >= 80 or r <= 20 or x["shock"]:
        return "overstretched"
    r8 = x["rsi_last8"]
    if 35 <= min(r8) <= 50 and rp < 50 <= r and hst > hp:
        return "pullback_up"
    if 50 <= max(r8) <= 65 and rp > 50 >= r and hst < hp:
        return "pullback_down"
    if r >= 55 and hst > 0 and hst > hp and x["c"] > x["close_24"]:
        return "strong_up"
    if r <= 45 and hst < 0 and hst < hp and x["c"] < x["close_24"]:
        return "strong_down"
    return "neutral"


def agent_momentum(x):
    mkt, t = x["mkt"], x["trend"]
    s = momentum_state(x)
    data = {"rsi": x["rsi"], "rsi_prev": x["rsi_prev"], "macd_hist": x["hist"], "macd_hist_prev": x["hist_prev"],
            "close": x["c"], "close_24_bars_ago": x["close_24"]}
    vote, rid, why = "WAIT", f"mom.{s}", ""
    up_ok = (t == "UP" and s in ("strong_up", "pullback_up"))
    dn_ok = (t == "DOWN" and s in ("strong_down", "pullback_down"))
    if up_ok:
        vote, why = "BUY", f"{s.replace('_', ' ')} in an up trend"
        if mkt == "GOLD" and s == "strong_up" and x["rsi"] > 70:
            vote, rid, why = "WAIT", "mom.gold_no_chase", "gold strong up but RSI above 70 - do not chase gold"
        if mkt == "OIL" and s == "strong_up":
            if not x["wti_window"]:
                vote, rid, why = "WAIT", "mom.wti_time", "WTI strong moves count only 10:00-14:30 New York"
            elif x["wti_bias"] == "DOWN":
                vote, rid, why = "WAIT", "mom.wti_bias", "WTI session bias is down"
        if vote == "BUY" and x["bear_div"]:
            vote, rid, why = "WAIT", "mom.divergence", "bearish RSI divergence at a new 24-candle high"
    elif dn_ok:
        vote, why = "SELL", f"{s.replace('_', ' ')} in a down trend"
        if mkt == "GOLD" and x["gold_top_quarter"]:
            vote, rid, why = "WAIT", "mom.gold_top_quarter", "gold price is in the top quarter of its 20-day range"
        if mkt == "OIL" and s == "strong_down":
            if not x["wti_window"]:
                vote, rid, why = "WAIT", "mom.wti_time", "WTI strong moves count only 10:00-14:30 New York"
            elif x["wti_bias"] == "UP":
                vote, rid, why = "WAIT", "mom.wti_bias", "WTI session bias is up"
    else:
        why = {"overstretched": "overstretched (RSI 80/20 or a shock candle)",
               "neutral": "no clear momentum state"}.get(s, f"{s.replace('_', ' ')} is against the trend")
        if s not in ("overstretched", "neutral"):
            rid = "mom.against_trend"
    reason = f"{why}; RSI {fnum(x['rsi'], 1)}, MACD histogram {fnum(x['hist'], 4)}"
    return A("momentum", vote, rid, reason, data, state=s)


def agent_volatility(x):
    mkt, t, f = x["mkt"], x["trend"], x["f"]
    data = {"atr": x["atr"], "atr_95pct_of_500": x["atr_q95"], "bb_upper": x["bb_up"], "bb_lower": x["bb_lo"],
            "bb_width": x["bb_w"], "bb_width_15pct_of_500": x["bbw_q15"], "close": x["c"]}
    extreme = x["shock"] or (finite(x["atr_q95"]) and x["atr"] >= x["atr_q95"])
    if extreme:
        return A("volatility", "WAIT", "vol.extreme", "extreme volatility (ATR in its top 5% or a shock candle) - "
                 "no new trade for the whole system", data, blocks=["BUY", "SELL"], block_code="vol_extreme")
    sq = f["squeeze"].values
    was = int(sq[-13:-1].sum()) if len(sq) >= 13 else 0
    if was >= 6 and not sq[-1] and (x["c"] > x["bb_up"] or x["c"] < x["bb_lo"]):
        d = "BUY" if x["c"] > x["bb_up"] else "SELL"
        ok = (d == "BUY" and t == "UP") or (d == "SELL" and t == "DOWN")
        vol_need = mkt == "BTC" and x["btc_quiet_time"]
        vol_ok = (not vol_need) or (x["vol_ratio"] is not None and x["vol_ratio"] > 1.5)
        if ok and vol_ok:
            return A("volatility", d, "vol.release", f"squeeze released ({was} of the last 12 candles) with a close "
                     f"outside the band, in the trend direction", data)
        why = "against the trend" if not ok else "volume not above 1.5x its 20-candle average in quiet hours"
        return A("volatility", "WAIT", "vol.release_no", f"squeeze released but {why}", data)
    if sq[-1] or (finite(x["bbw_q15"]) and x["bb_w"] <= x["bbw_q15"]):
        return A("volatility", "WAIT", "vol.compressed", "compressed (squeeze on or very narrow bands) - "
                 "a move is coming, direction unknown", data)
    hi24, lo24 = x["hi24_prev"], x["lo24_prev"]
    if finite(x["atr_5ago"]) and x["atr"] > x["atr_5ago"]:
        if t == "UP" and x["c"] > hi24:
            return A("volatility", "BUY", "vol.expanding_up", "volatility rising and close above the 24-candle high",
                     data)
        if t == "DOWN" and x["c"] < lo24:
            return A("volatility", "SELL", "vol.expanding_down", "volatility rising and close below the 24-candle "
                     "low", data)
    if mkt == "GOLD" and t == "UP" and x["c"] < x["bb_lo"]:
        return A("volatility", "BUY", "vol.gold_dip", "gold up trend and the close is below the lower band - buy "
                 "the dip", data)
    return A("volatility", "WAIT", "vol.normal", "normal volatility, no breakout in the trend direction", data)


# ----------------------------------------------------------------- levels
def build_zones(x):
    """Levels -> zones with points. Strong zone = 3+ points."""
    mkt, f, a, price = x["mkt"], x["f"], x["atr"], x["c"]
    m = MARKETS[mkt]
    lv = []                        # (price, points, source)
    if x["last_week"] is not None:
        lv += [(x["last_week"][0], 3.0, "week_high"), (x["last_week"][1], 3.0, "week_low")]
    if x["yesterday"] is not None:
        lv += [(x["yesterday"][0], 2.0, "day_high"), (x["yesterday"][1], 2.0, "day_low")]
    dd = x["daily_bars"].iloc[-120:]
    if len(dd) >= 7:
        hi, lo = pivots(dd["High"].values, dd["Low"].values, 3)
        lv += [(p, 2.0, "daily_swing") for _, p in hi + lo]
    h4 = x["h4"]
    if len(h4) >= 11:
        h4w = h4[h4.index >= h4.index[-1] - pd.Timedelta(days=60)]
        hi, lo = pivots(h4w["High"].values, h4w["Low"].values, 5)
        lv += [(p, 1.0, "4h_swing") for _, p in hi + lo]
    win = max(8 * a, 0.02 * price)
    for r in round_levels(price, m["round_big"], price - win, price + win):
        pts = m["round_pts"] + (1.0 if m["round_bonus"] and abs(r / m["round_bonus"] - round(r / m["round_bonus"])) < 1e-9
                                else 0.0)
        lv.append((r, pts, "round"))
    lv = [l for l in lv if finite(l[0]) and abs(l[0] - price) <= 12 * a]
    lv.sort(key=lambda l: l[0])
    clusters = []
    for l in lv:
        if clusters and l[0] - clusters[-1][-1][0] <= 0.5 * a:
            clusters[-1].append(l)
        else:
            clusters.append([l])
    zmin, zmax = m["zone_pct"]
    width = min(max(1.0 * a, price * zmin / 100), price * zmax / 100)
    zones = []
    for cl in clusters:
        w = sum(p for _, p, _ in cl)
        center = sum(pr * p for pr, p, _ in cl) / w
        zones.append({"mid": center, "lo": center - width / 2, "hi": center + width / 2, "src_pts": w,
                      "sources": sorted({s for _, _, s in cl})})
    # touches on 4H candles (60 days, 24h apart), reset after a 4H close through the zone, fading with age
    h4w = h4[h4.index >= (h4.index[-1] - pd.Timedelta(days=60))] if len(h4) else h4
    now = f.index[-1]
    for z in zones:
        touches, last_t, last_touch, side = 0, None, None, None
        for ts, row in h4w.iterrows():
            s = "below" if row["Close"] < z["lo"] else ("above" if row["Close"] > z["hi"] else side)
            if side is not None and s is not None and s != side:
                touches, last_t = 0, None          # broke through: touches reset
            if s is not None:
                side = s
            if row["High"] >= z["lo"] and row["Low"] <= z["hi"]:
                if last_t is None or (ts - last_t) >= pd.Timedelta(hours=24):
                    touches += 1
                    last_t = ts
                    last_touch = ts
        tp = min(touches, 5) * 0.5
        if last_touch is not None and tp > 0:
            days = (now - last_touch).days if m["is_247"] else int(np.busday_count(last_touch.date(), now.date()))
            tp *= 0.5 ** (days // m["fade_days"])
        z["touch_pts"] = tp
        z["touches"] = touches
        z["pts"] = z["src_pts"] + tp
        z["strong"] = z["pts"] >= 3.0
    return zones


def room_check(zones, d, entry, dist):
    """Distance to the first strong zone in the trade's path, in R. Under 1R blocks the trade."""
    if not finite(dist) or dist <= 0:
        return 0.0, None
    strong = [z for z in zones if z["strong"]]
    if d == "BUY":
        ahead = [z for z in strong if z["hi"] > entry]
        if not ahead:
            return None, None
        z = min(ahead, key=lambda z: z["lo"])
        gap = max(0.0, z["lo"] - entry)
    else:
        ahead = [z for z in strong if z["lo"] < entry]
        if not ahead:
            return None, None
        z = max(ahead, key=lambda z: z["hi"])
        gap = max(0.0, entry - z["hi"])
    return gap / dist, z


def agent_levels(x):
    mkt, f, a = x["mkt"], x["f"], x["atr"]
    zones = x["zones"]
    strong = [z for z in zones if z["strong"]]
    C, H, L, O = f["c"].values, f["h"].values, f["l"].values, f["o"].values
    c = C[-1]
    sup = [z for z in strong if z["hi"] < c]
    res = [z for z in strong if z["lo"] > c]
    ns = max(sup, key=lambda z: z["hi"]) if sup else None
    nr = min(res, key=lambda z: z["lo"]) if res else None
    data = {"close": float(c), "atr": float(a),
            "nearest_support": float(ns["mid"]) if ns else None,
            "nearest_resistance": float(nr["mid"]) if nr else None,
            "support_points": float(ns["pts"]) if ns else None,
            "resistance_points": float(nr["pts"]) if nr else None}
    data = {k: v for k, v in data.items() if v is not None}
    allowed = MARKETS[mkt]["signals"]
    if any(z["lo"] <= c <= z["hi"] for z in strong):
        return A("levels", "WAIT", "lvl.inside", "price is inside a strong zone", data)
    if not any(abs(z["mid"] - c) <= 2 * a or z["lo"] <= c + 2 * a and z["hi"] >= c - 2 * a for z in strong):
        if "eia_fade" not in allowed or not x["eia_fade"]:
            return A("levels", "WAIT", "lvl.none_near", "no strong zone within 2x ATR", data)
    signals = []        # (direction, kind, zone, candles ago)
    old_sweeps = []
    rng = H - L
    n = len(C)
    for z in strong:
        dayweek = any(s in z["sources"] for s in ("day_high", "day_low", "week_high", "week_low"))
        roundz = "round" in z["sources"]
        for k in (1, 2, 3):
            i = n - k
            if i < 1:
                continue
            up = C[i] >= z["hi"] + 0.25 * a and C[i - 1] <= z["hi"] and rng[i] >= 1.0 * a
            dn = C[i] <= z["lo"] - 0.25 * a and C[i - 1] >= z["lo"] and rng[i] >= 1.0 * a
            if up and c > z["hi"]:
                signals.append(("BUY", "break", z, k))
            if dn and c < z["lo"]:
                signals.append(("SELL", "break", z, k))
        for b in range(2, 8):
            i = n - b
            if i < 1:
                continue
            if C[i] >= z["hi"] + 0.25 * a and C[i - 1] <= z["hi"] and rng[i] >= 1.0 * a:
                after = slice(i + 1, n)
                if L[after].min() <= z["hi"] and C[after].min() >= z["lo"] and c > z["hi"] and \
                        (L[-1] <= z["hi"] or C[-2] <= z["hi"]):
                    signals.append(("BUY", "retest", z, b))
            if C[i] <= z["lo"] - 0.25 * a and C[i - 1] >= z["lo"] and rng[i] >= 1.0 * a:
                after = slice(i + 1, n)
                if H[after].max() >= z["lo"] and C[after].max() <= z["hi"] and c < z["lo"] and \
                        (H[-1] >= z["lo"] or C[-2] >= z["lo"]):
                    signals.append(("SELL", "retest", z, b))
        for k in (1, 2, 3):                 # sweep-and-reclaim; only the newest candle can vote, older ones
            i = n - k                       # are kept to spot a break and a sweep in opposite directions
            if i < 1:
                continue
            if C[i - 1] < z["lo"] and H[i] >= z["hi"] + 0.1 * a and C[i] < z["lo"]:
                (signals if k == 1 else old_sweeps).append(("SELL", "sweep", z, k))
            if C[i - 1] > z["hi"] and L[i] <= z["lo"] - 0.1 * a and C[i] > z["hi"]:
                (signals if k == 1 else old_sweeps).append(("BUY", "sweep", z, k))
        z["_dayweek"], z["_round"] = dayweek, roundz
    if x["eia_fade"]:
        signals.append((x["eia_fade"], "eia_fade", None, 1))
    breaks = {s[0] for s in signals if s[1] == "break"}
    sweeps = {s[0] for s in signals + old_sweeps if s[1] == "sweep"}
    if breaks and sweeps and (breaks | sweeps) == {"BUY", "SELL"}:
        return A("levels", "WAIT", "lvl.conflict", "a break and a sweep in opposite directions within 3 candles", data)
    usable = []
    for d, kind, z, k in signals:
        if kind not in allowed and not (kind == "retest" and "break" in allowed and mkt == "BTC"):
            continue
        if mkt == "GOLD" and kind == "break" and not (z["_dayweek"] or z["_round"]):
            continue
        if mkt == "GOLD" and kind == "sweep" and not z["_dayweek"]:
            continue
        usable.append((d, kind, z, k))
    dirs = {u[0] for u in usable}
    if len(dirs) > 1:
        return A("levels", "WAIT", "lvl.conflict", "level signals point both ways", data)
    if not usable:
        return A("levels", "WAIT", "lvl.no_signal", "no break, retest or sweep of a strong zone (plain touches never "
                 "vote)", data)
    d, kind, z, k = sorted(usable, key=lambda u: u[3])[0]
    where = f" the zone at {pxm(z['mid'], mkt)} ({z['pts']:.1f} points)" if z else " a strong zone"
    up = d == "BUY"
    reason = {"break": f"1-hour close broke {'up' if up else 'down'} through{where}",
              "retest": f"price came back to test{where} after breaking it, and held",
              "sweep": f"price poked {'below' if up else 'above'}{where} and closed back {'above' if up else 'below'} it "
                       "(sweep and reclaim)",
              "eia_fade": "EIA spike faded back inside the pre-report range"}[kind]
    if z:
        data["zone_mid"] = float(z["mid"])
        data["zone_points"] = float(z["pts"])
    struct = None                  # the structure this setup depends on (for the stop)
    if kind in ("break", "retest") and z:
        struct = z["lo"] if d == "BUY" else z["hi"]
    elif kind == "sweep":
        struct = min(L[-1], z["lo"]) if d == "BUY" else max(H[-1], z["hi"])
    elif kind == "eia_fade":
        struct = float(L[-2:].min()) if d == "BUY" else float(H[-2:].max())
    if struct is not None:              # the signal must also pass the stop and room checks (research: break row)
        lp = plan_at(d, float(c), a, struct, struct, mkt)
        if not lp.get("valid"):
            return A("levels", "WAIT", "lvl.no_stop", f"{reason}, but the stop behind it would be too wide", data)
        room, zz = room_check(zones, d, lp["entry"], lp["dist"])
        if room is not None and room < 1.0:
            return A("levels", "WAIT", "lvl.no_room", f"{reason}, but a strong zone is only {room:.2f}R away", data)
    return A("levels", d, f"lvl.{kind}", reason, data, signal=kind, struct=struct)


# ----------------------------------------------------------------- special (derivatives / flows)
def agent_special(x):
    mkt = x["mkt"]
    if mkt == "BTC":
        return special_btc(x["deriv"], x)
    return special_flows(x)


def special_btc(dv, x):
    if not dv or dv.get("venues", 0) < 2:
        return A("special", "WAIT", "spc.no_data", "fewer than 2 venues answered - no data", {}, no_data=True)
    f24, rank, fnow = dv.get("funding24"), dv.get("rank"), dv.get("funding_now")   # 24h average vs this hour
    p1, p4, p24 = x["p_chg_1h"], x["p_chg_4h"], x["p_chg_24h"]
    o1, o4, o24 = dv.get("oi_1h"), dv.get("oi_4h"), dv.get("oi_24h")
    data = {k: v for k, v in {"funding_24h_avg_per_8h_pct": None if f24 is None else f24 * 100,
                              "funding_now_per_8h_pct": None if fnow is None else fnow * 100,
                              "funding_30d_rank_pct": None if rank is None else rank * 100,
                              "oi_change_1h_pct": o1, "oi_change_4h_pct": o4, "oi_change_24h_pct": o24,
                              "price_change_1h_pct": p1, "price_change_4h_pct": p4,
                              "price_change_24h_pct": p24, "venues": dv.get("venues")}.items() if v is not None}

    def ok(*v):
        return all(finite(i) for i in v)
    if ok(p1, o1) and abs(p1) >= 2 and o1 <= -3:
        return A("special", "WAIT", "spc.cascade", "liquidation cascade: price moved 2%+ in 1 hour while open interest "
                 "fell 3%+ - no new entries for 2 candles", data, blocks=["BUY", "SELL"], block_code="cascade",
                 block_bars=2)
    if ok(f24, rank, o24) and f24 >= 0.0003 and rank >= 0.9 and o24 >= 5:
        if ok(fnow) and fnow >= 0.0005 and ok(p1, o1) and p1 < 0 and o1 < 0:
            return A("special", "SELL", "spc.crowded_longs_sell", "crowded longs (funding high, top 10%, OI up 5%+) "
                     "and price and OI both fell in the last hour", data, blocks=["BUY"], block_code="crowded_longs")
        return A("special", "WAIT", "spc.crowded_longs", "crowded longs (funding 0.03%+, top 10% of 30 days, OI up "
                 "5%+ in 24h) - blocks BUY", data, blocks=["BUY"], block_code="crowded_longs")
    if ok(f24, rank, o24) and f24 <= -0.0001 and rank <= 0.1 and o24 >= 5:
        if ok(p1, o1) and p1 > 0 and o1 < 0:
            return A("special", "BUY", "spc.crowded_shorts_buy", "crowded shorts and price rising while OI falls",
                     data, blocks=["SELL"], block_code="crowded_shorts")
        return A("special", "WAIT", "spc.crowded_shorts", "crowded shorts (funding negative, bottom 10%, OI up 5%+) - "
                 "blocks SELL", data, blocks=["SELL"], block_code="crowded_shorts")
    if ok(p24, o24, fnow) and p24 <= -5 and o24 <= -7 and fnow <= 0.00005:
        return A("special", "WAIT", "spc.long_flush", "long flush done (price -5%, OI -7% in 24h) - do not chase a "
                 "SELL", data, blocks=["SELL"], block_code="long_flush")
    if ok(p24, o24, fnow) and p24 >= 5 and o24 <= -7 and fnow >= 0.0003:
        return A("special", "WAIT", "spc.squeeze_done", "short squeeze done (price +5%, OI -7% in 24h) - do not chase "
                 "a BUY", data, blocks=["BUY"], block_code="squeeze_done")
    if ok(p4, o4, fnow) and p4 >= 1 and o4 >= 1 and 0 <= fnow <= 0.0002:
        return A("special", "BUY", "spc.healthy_up", "healthy up-move: price and open interest both up 1%+ in 4 "
                 "hours, funding normal", data, support="BUY")
    if ok(p4, o4, fnow) and p4 <= -1 and o4 >= 1 and fnow <= 0.0001:
        return A("special", "SELL", "spc.healthy_down", "healthy down-move: price down, open interest up, funding "
                 "low", data, support="SELL")
    why = "neutral"
    if o24 is None:
        why = "neutral (open interest history still building)"
    return A("special", "WAIT", "spc.neutral", why, data)


def special_flows(x):
    mkt, fl = x["mkt"], x["flows"] or {}
    data, blocks, support, why = {}, [], None, []
    cot = fl.get("cot_" + mkt)
    if cot and finite(cot.get("rank")):
        data["cot_net_long_3y_rank_pct"] = cot["rank"] * 100
        if cot["rank"] >= 0.9:
            blocks.append("BUY")
            why.append("big funds' net long is in the top 10% of 3 years (crowded) - blocks BUY")
        elif cot["rank"] <= 0.1:
            blocks.append("SELL")
            why.append("big funds' net long is in the bottom 10% of 3 years - blocks SELL")
    if mkt == "GOLD":
        g = fl.get("gld")
        if g and finite(g.get("chg5")):
            data["gld_holdings_5d_change_pct"] = g["chg5"]
            if g["chg5"] >= 1:
                support = "BUY"
                why.append("GLD holdings up 1%+ in 5 days - supports BUY")
            elif g["chg5"] <= -1:
                blocks.append("BUY")
                why.append("GLD holdings down 1%+ in 5 days - blocks BUY")
    if mkt == "OIL":
        sp = fl.get("wti_spread")
        if sp and finite(sp.get("chg5")):
            data["front_minus_second_month"] = sp["spread"]
            data["spread_5d_change"] = sp["chg5"]
            if sp["chg5"] > 0.02:
                support = "BUY"
                why.append("front month getting dearer than the next month (tight supply) - supports BUY")
            elif sp["chg5"] < -0.02:
                blocks.append("BUY")
                why.append("front month getting cheaper than the next month (plenty of supply) - blocks BUY")
    if not data:
        return A("special", "WAIT", "spc.no_data", "no flow data (COT / GLD / spread) reachable", {}, no_data=True)
    vote = support if support and support not in blocks else "WAIT"
    rid = "spc.flows_support" if vote != "WAIT" else ("spc.flows_block" if blocks else "spc.flows_neutral")
    return A("special", vote, rid, "; ".join(why) or "flows neutral", data, blocks=sorted(set(blocks)),
             support=support if vote != "WAIT" else None, block_code="flows" if blocks else None)


# ----------------------------------------------------------------- macro
def agent_macro(x):
    mkt, mc = x["mkt"], x["macro"]
    if mkt == "BTC" and x["btc_weekend_macro"]:
        return A("macro", "WAIT", "mac.btc_weekend", "US markets closed for the weekend - macro sits out", {})
    if not mc or not mc.get("fresh"):
        return A("macro", "WAIT", "mac.stale", "all macro inputs are older than 90 minutes - WAIT", {}, no_data=True)
    z = mc["z"]
    data = {f"{k}_z": v for k, v in z.items() if v is not None}
    data.update({k: v for k, v in {"vix": mc.get("vix"), "risk_off": int(mc["risk_off"]),
                                   "dollar_trend_up": int(mc["dxy_up"]), "dollar_trend_down": int(mc["dxy_down"])}.items()
                 if v is not None})
    dz, yz, nz, vz, sz, cz = (z.get(k) for k in ("dollar", "yield10", "nasdaq", "vix", "sp500", "copper"))
    if mkt == "GOLD":
        if not finite(dz, yz):
            return A("macro", "WAIT", "mac.gold_missing", "dollar or 10-year yield input is old or missing (dropped) - "
                     "WAIT", data)
        ds = 1 if dz <= -0.5 else (-1 if dz >= 0.5 else 0)
        ys = 1 if yz <= -0.5 else (-1 if yz >= 0.5 else 0)
        vote, rid, why = "WAIT", "mac.gold_neutral", "dollar and yield moves not both clear"
        if ds == 1 and ys == 1 and not mc["dxy_up"]:
            vote, rid, why = "BUY", "mac.gold_buy", "dollar and 10-year yield both falling fast, dollar trend not up"
        elif ds == -1 and ys == -1 and not mc["dxy_down"]:
            vote, rid, why = "SELL", "mac.gold_sell", "dollar and 10-year yield both rising fast, dollar trend not down"
        if vote != "WAIT" and x["gold_asia"]:
            agree = (vote == "BUY" and mc["dxy_down"] and x["gold_overnight"] > 0) or \
                    (vote == "SELL" and mc["dxy_up"] and x["gold_overnight"] < 0)
            if not agree:
                vote, rid, why = "WAIT", "mac.gold_asia", "Asian session: daily background and overnight move do " \
                                                          "not agree"
        return A("macro", vote, rid, why, data)
    if mkt == "BTC":
        if finite(nz) and nz <= -1.0 and ((finite(vz) and vz >= 0.75) or mc["risk_off"]) and finite(dz) and dz >= 0:
            return A("macro", "SELL", "mac.btc_block", "Nasdaq falling hard, fear rising and dollar firm - blocks "
                     "BUYs (counts as a SELL vote)", data, blocks=["BUY"], block_code="macro_risk_off")
        if finite(nz, dz) and nz >= 0.75 and dz <= -0.5 and not mc["risk_off"]:
            return A("macro", "BUY", "mac.btc_buy", "Nasdaq rising, dollar falling, market not risk-off", data)
        return A("macro", "WAIT", "mac.btc_wait", "no strong macro push for Bitcoin", data)
    if finite(sz, cz):
        if sz >= 0.5 and cz >= 0.5 and not mc["risk_off"]:
            return A("macro", "BUY", "mac.oil_buy", "S&P futures and copper both rising, not risk-off", data)
        if sz <= -0.5 and cz <= -0.5:
            return A("macro", "SELL", "mac.oil_sell", "S&P futures and copper both falling", data)
        return A("macro", "WAIT", "mac.oil_wait", "S&P futures and copper not both clear", data)
    return A("macro", "WAIT", "mac.oil_missing", "S&P or copper input is old or missing (dropped) - WAIT", data)


# ----------------------------------------------------------------- mood
def agent_mood(x):
    mkt = x["mkt"]
    if mkt != "BTC":
        return A("mood", "WAIT", "mood.none", "no tested free mood index for this market - WAIT by default", {})
    fg = x["fng"]
    if not fg or not finite(fg.get("value")):
        return A("mood", "WAIT", "mood.no_data", "Fear & Greed index not reachable", {}, no_data=True)
    v, avg30, t = fg["value"], fg.get("avg30"), x["trend"]
    data = {"fear_greed": v, "fear_greed_30d_avg": avg30}
    data = {k: val for k, val in data.items() if val is not None}
    if v <= 15 and x["h4_close_above_swing"]:
        return A("mood", "BUY", "mood.extreme_fear_turn", "extreme fear (15 or less) and a 4H close above the last 4H "
                 "swing high (experimental)", data)
    if v >= 90 and x["h4_close_below_swing"]:
        return A("mood", "SELL", "mood.extreme_greed_turn", "extreme greed (90+) and a 4H close below the last 4H "
                 "swing low (experimental)", data)
    if 26 <= v <= 74:
        return A("mood", "WAIT", "mood.middle", "Fear & Greed in the middle (26-74)", data)
    if v >= 75:
        note = ["crowded"] if v >= 90 and finite(avg30) and avg30 >= 75 else []
        if t == "UP":
            return A("mood", "BUY", "mood.greed_with_trend", "greed (75+) and the trend is up - ride it, never sell "
                     "because of greed", data, notes=note)
        return A("mood", "WAIT", "mood.greed_no_trend", "greed but the trend is not up", data, notes=note)
    if t == "DOWN":
        return A("mood", "SELL", "mood.fear_with_trend", "fear (25 or less) and the trend is down - never buy just "
                 "because of fear", data)
    return A("mood", "WAIT", "mood.fear_no_trend", "fear but the trend is not down", data)


# ----------------------------------------------------------------- bull / bear (code rule = twin answer)
def fact_sheet(x, rule):
    """One code-made fact sheet for both debaters. Room is named by direction."""
    mkt = x["mkt"]
    stop_dist = x["plan_buy"]["dist"] if x["plan_buy"].get("valid") else 1.5 * x["atr"]
    stop_dist_s = x["plan_sell"]["dist"] if x["plan_sell"].get("valid") else 1.5 * x["atr"]
    room_up, _ = room_check(x["zones"], "BUY", x["c"], stop_dist)        # same measure as the room check
    room_dn, _ = room_check(x["zones"], "SELL", x["c"], stop_dist_s)
    facts = {"price": x["c"], "atr": x["atr"], "trend_switch": x["trend"], "trend_4h": x["trend4"],
             "daily_view": x["daily_view"][0], "rsi": x["rsi"], "macd_hist": x["hist"],
             "room_up_to_resistance_R": round(room_up, 2) if room_up is not None else 9.9,
             "room_down_to_support_R": round(room_dn, 2) if room_dn is not None else 9.9,
             "momentum_state": rule["momentum"].get("state", "neutral")}
    if mkt == "BTC" and x["deriv"]:
        if finite(x["deriv"].get("funding24")):
            facts["funding_24h_pct"] = round(x["deriv"]["funding24"] * 100, 4)
        if finite(x["deriv"].get("oi_24h")):
            facts["oi_change_24h_pct"] = round(x["deriv"]["oi_24h"], 2)
    if x["fng"] and mkt == "BTC" and finite(x["fng"].get("value")):
        facts["fear_greed"] = x["fng"]["value"]
    lv = rule["levels"]["data"]
    for k in ("nearest_resistance", "nearest_support"):
        if finite(lv.get(k)):
            facts[k] = lv[k]
    if x.get("news_score") is not None:
        facts["news_net_score"] = x["news_score"]
        items = x.get("news_items") or []
        facts["news_up_stories"] = sum(1 for s, l in items if s.get("valid") and l["direction"] == "UP")
        facts["news_down_stories"] = sum(1 for s, l in items if s.get("valid") and l["direction"] == "DOWN")
    fl = x.get("flows") or {}
    if mkt == "GOLD" and finite((fl.get("gld") or {}).get("chg5")):
        facts["gld_holdings_5d_change_pct"] = fl["gld"]["chg5"]
    if finite((fl.get("cot_" + mkt) or {}).get("rank")) and mkt != "BTC":
        facts["big_funds_net_long_3y_rank_pct"] = fl["cot_" + mkt]["rank"] * 100
    if mkt == "OIL" and finite((fl.get("wti_spread") or {}).get("chg5")):
        facts["front_minus_second_month_5d_change"] = fl["wti_spread"]["chg5"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in facts.items()}


def bull_bear_rule(side, facts):
    """Code version of the debater (used by the rules-only twin and when the AI is absent)."""
    if side == "bull":
        room = facts["room_up_to_resistance_R"]
        pts = [facts["trend_switch"] == "UP", facts["daily_view"] == "UP", finite(facts["rsi"]) and facts["rsi"] >= 55,
               finite(facts["macd_hist"]) and facts["macd_hist"] > 0, facts["trend_4h"] == "UP"]
        vote = "BUY" if room >= 2.0 and sum(pts) >= 2 else "WAIT"
    else:
        room = facts["room_down_to_support_R"]
        pts = [facts["trend_switch"] == "DOWN", facts["daily_view"] == "DOWN",
               finite(facts["rsi"]) and facts["rsi"] <= 45, finite(facts["macd_hist"]) and facts["macd_hist"] < 0,
               facts["trend_4h"] == "DOWN"]
        vote = "SELL" if room >= 2.0 and sum(pts) >= 2 else "WAIT"
    return vote, room, sum(pts)


def agent_debater(side, x, facts):
    vote, room, n = bull_bear_rule(side, facts)
    key = "room_up_to_resistance_R" if side == "bull" else "room_down_to_support_R"
    return A(side, vote, f"{side}.code", f"{n} supporting facts, {key} = {room}", {key: room, "supporting_facts": n})


# ----------------------------------------------------------------- time machine (rules only)
def time_machine(d, x):
    """Similar past setups on this market, judged with the exact live trade rules. Silent below 30 cases."""
    f, mkt = x["f"], x["mkt"]
    if d not in ("BUY", "SELL"):
        return A("time_machine", "WAIT", "tm.silent", "no direction to test", {})
    want = "UP" if d == "BUY" else "DOWN"
    n_all = len(f)
    i_now = n_all - 1
    O, H, L, C = (f[k].values for k in ("o", "h", "l", "c"))
    tr, t4 = f["trend"].values, f["trend4"].values
    feats = {k: f[k].values for k in ("atr_t", "rsi_g", "dist_g", "session", "wday")}
    use = ["atr_t", "rsi_g", "dist_g", "session"] + (["wday"] if not MARKETS[mkt]["is_247"] else [])
    now_f = {k: feats[k][i_now] for k in use}
    fb = MARKETS[mkt]["fill_bars"]                 # same fill rule as live trades (gold/oil: one candle later)
    last_ok = i_now - CFG["MAX_BARS"] - fb
    idx = np.arange(0, max(0, last_ok))
    if not len(idx):
        return A("time_machine", "WAIT", "tm.silent", "not enough history", {"cases": 0})
    mask = (tr[idx] == want) & (t4[idx] == t4[i_now])
    cand = idx[mask]
    if not len(cand):
        return A("time_machine", "WAIT", "tm.silent", "no similar past setup", {"cases": 0})
    score = np.zeros(len(cand), dtype=int)
    for k in use:
        score += (feats[k][cand] == now_f[k]).astype(int)
    order = np.lexsort((-cand, -score))          # best match first, then most recent
    atrv, slo, shi = f["atr"].values, f["stop_lo"].values, f["stop_hi"].values
    picked, rs, hits, costs, wts = [], [], [], [], []
    three_years = f.index[i_now] - pd.Timedelta(days=3 * 365)
    for j in cand[order]:
        if len(picked) >= 50:
            break
        if any(abs(j - p) < CFG["MAX_BARS"] for p in picked):
            continue
        p = plan_at(d, C[j], atrv[j], slo[j], shi[j], mkt)
        if not p["valid"] or p["cost_r"] > CFG["MAX_COST_R"]:
            continue
        out = sim_fixed(O, H, L, C, j + fb, d, p["stop"], p["t"][1], p["dist"], p["cost_r"])
        if not out or out.get("skip"):
            continue
        picked.append(j)
        rs.append(out["r"])
        hits.append(1.0 if out["hit"] else 0.0)
        costs.append(p["cost_r"])
        wts.append(0.5 if f.index[j] < three_years else 1.0)
    n = len(picked)
    if n < 30:
        return A("time_machine", "WAIT", "tm.silent", f"only {n} similar past cases (needs 30) - stays silent",
                 {"cases": n})
    w = np.array(wts)
    avg = float(np.dot(w, rs) / w.sum())
    hit = float(np.dot(w, hits) / w.sum())
    be = (1 + float(np.mean(costs))) / 3
    data = {"cases": n, "avg_R": round(avg, 3), "hit_rate_pct": round(hit * 100, 1), "break_even_pct": round(be * 100, 1)}
    if avg >= 0.15 and hit >= be + 0.05:
        return A("time_machine", d, "tm.confirm", f"{n} similar past cases: average {avg:+.2f}R, hit rate "
                 f"{hit * 100:.0f}% (break-even {be * 100:.0f}%) - CONFIRM", data)
    if avg <= -0.10:
        return A("time_machine", "WAIT", "tm.block", f"{n} similar past cases lost on average ({avg:+.2f}R) - BLOCK",
                 data, blocks=[d], block_code="time_machine")
    return A("time_machine", "WAIT", "tm.silent", f"{n} cases: average {avg:+.2f}R, hit rate {hit * 100:.0f}% "
             f"(CONFIRM needs +0.15R and {(be + 0.05) * 100:.0f}%) - stays silent", data)


def power_trend(h, h4, daily):
    """User's Power Trend (higher highs + higher lows on Daily, 4H and 1H) - test mode only."""
    def struct(df, k=3):
        if df is None or len(df) < 4 * k + 2:
            return "NONE"
        hi, lo = pivots(df["High"].values, df["Low"].values, k)
        if len(hi) < 2 or len(lo) < 2:
            return "NONE"
        if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]:
            return "UP"
        if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]:
            return "DOWN"
        return "NONE"
    s = {struct(h.iloc[-300:]), struct(h4.iloc[-200:]), struct(daily.iloc[-120:])}
    return s.pop() if len(s) == 1 else "NONE"


# =============================================================== outside data: derivatives, flows, macro, mood, events
def fetch_deriv(st, now):
    """BTC funding and open interest from Hyperliquid, OKX, Deribit, dYdX (funding + open interest) and Kraken
    Futures (open interest only). Middle value of the venues that answered; fewer than 2 = no data."""
    out = {"funding": {}, "oi": {}}
    j = http_post_json("https://api.hyperliquid.xyz/info", {"type": "metaAndAssetCtxs"})
    try:
        names = [u["name"] for u in j[0]["universe"]]
        ctx = j[1][names.index("BTC")]
        out["funding"]["hyperliquid"] = float(ctx["funding"]) * 8           # hourly rate -> per 8 hours
        out["oi"]["hyperliquid"] = float(ctx["openInterest"])               # in BTC
        note_source("hyperliquid", True)
    except Exception:
        note_source("hyperliquid", False)
    j = http_json("https://www.okx.com/api/v5/public/funding-rate", params={"instId": "BTC-USDT-SWAP"})
    try:
        out["funding"]["okx"] = float(j["data"][0]["fundingRate"])
        note_source("okx_funding", True)
    except Exception:
        note_source("okx_funding", False)
    j = http_json("https://www.okx.com/api/v5/public/open-interest", params={"instType": "SWAP",
                                                                             "instId": "BTC-USDT-SWAP"})
    try:
        out["oi"]["okx"] = float(j["data"][0]["oiCcy"])
        note_source("okx_oi", True)
    except Exception:
        note_source("okx_oi", False)
    j = http_json("https://www.deribit.com/api/v2/public/ticker", params={"instrument_name": "BTC-PERPETUAL"})
    try:
        r = j["result"]
        out["funding"]["deribit"] = float(r["funding_8h"])
        out["oi"]["deribit"] = float(r["open_interest"]) / float(r["mark_price"])     # USD -> BTC
        note_source("deribit", True)
    except Exception:
        note_source("deribit", False)
    j = http_json("https://indexer.dydx.trade/v4/perpetualMarkets", params={"ticker": "BTC-USD"})
    try:                       # dYdX v4: hourly funding rate, open interest in BTC
        mk = j["markets"]["BTC-USD"]
        out["funding"]["dydx"] = float(mk["nextFundingRate"]) * 8
        out["oi"]["dydx"] = float(mk["openInterest"])
        note_source("dydx", True)
    except Exception:
        note_source("dydx", False)
    j = http_json("https://futures.kraken.com/derivatives/api/v3/tickers")
    try:                       # Kraken Futures: open interest only (PF_XBTUSD: 1 contract = 1 BTC); its funding
        tk = next(t for t in j["tickers"] if str(t.get("symbol", "")).upper() == "PF_XBTUSD")   # units are unclear
        oi = float(tk["openInterest"])
        if 10 < oi < 2_000_000:
            out["oi"]["kraken"] = oi
        note_source("kraken_futures", True)
    except Exception:
        note_source("kraken_futures", False)
    for v, val in list(out["funding"].items()):          # sanity: funding per 8h beyond 1% is a unit error
        if not finite(val) or abs(val) > 0.01:
            out["funding"].pop(v)
    for v, val in list(out["oi"].items()):
        if not finite(val) or val <= 0:
            out["oi"].pop(v)
    # funding history for the 24h average and the 30-day rank (kept in the saved state)
    fh = st.deriv["fund_hist"]
    nowiso = pd.Timestamp(now).isoformat()
    for v, val in out["funding"].items():
        fh.setdefault(v, []).append([nowiso, val])
    if len(fh.get("hyperliquid", [])) < 200:
        try:
            start = int((pd.Timestamp(now) - pd.Timedelta(days=31)).timestamp() * 1000)
            rows = []
            for _ in range(2):
                jj = http_post_json("https://api.hyperliquid.xyz/info", {"type": "fundingHistory", "coin": "BTC",
                                                                         "startTime": start})
                if not jj:
                    break
                rows += [[pd.Timestamp(int(r["time"]), unit="ms").isoformat(), float(r["fundingRate"]) * 8] for r in jj]
                if len(jj) < 500:
                    break
                start = int(jj[-1]["time"]) + 1
            if rows:
                old = {r[0] for r in fh.get("hyperliquid", [])}
                fh["hyperliquid"] = sorted(fh.get("hyperliquid", []) + [r for r in rows if r[0] not in old])
        except Exception:
            pass
    if len(fh.get("okx", [])) < 60:
        jj = http_json("https://www.okx.com/api/v5/public/funding-rate-history", params={"instId": "BTC-USDT-SWAP",
                                                                                         "limit": 100})
        try:
            rows = [[pd.Timestamp(int(r["fundingTime"]), unit="ms").isoformat(), float(r["realizedRate"] or
                                                                                         r["fundingRate"])]
                    for r in jj["data"]]
            old = {r[0] for r in fh.get("okx", [])}
            fh["okx"] = sorted(fh.get("okx", []) + [r for r in rows if r[0] not in old])
        except Exception:
            pass
    snaps = st.deriv["snaps"]
    snaps.append({"ts": nowiso, "oi": out["oi"]})
    venues = len(set(out["funding"]) | set(out["oi"]))
    res = {"venues": min(len(out["funding"]), len(out["oi"])) if out["oi"] else len(out["funding"]),
           "venues_any": venues, "venues_funding": len(out["funding"]), "venues_oi": len(out["oi"])}
    res["funding_now"] = float(np.median(list(out["funding"].values()))) if len(out["funding"]) >= 2 else None
    f24s, ranks = [], []
    for v, rows in fh.items():
        if not rows or v not in out["funding"]:          # only venues that answered this hour
            continue
        try:
            s = pd.Series([float(r[1]) for r in rows],
                          index=pd.to_datetime([r[0] for r in rows], format="ISO8601")).sort_index()
        except Exception as e:
            log(f"funding history {v} unreadable ({type(e).__name__}) - reset")
            fh[v] = [[nowiso, out["funding"][v]]]
            continue
        s = s[~s.index.duplicated(keep="last")]
        s = s[s.index >= pd.Timestamp(now) - pd.Timedelta(days=31)]
        if len(s) < 2:
            continue
        roll = s.rolling("24h").mean()
        cur = float(roll.iloc[-1])
        f24s.append(cur)
        hist = roll[roll.index >= pd.Timestamp(now) - pd.Timedelta(days=30)]
        if len(hist) >= 20:
            ranks.append(float((hist < cur).mean()))
    res["funding24"] = float(np.median(f24s)) if len(f24s) >= 2 else (f24s[0] if f24s and len(out["funding"]) >= 2
                                                                      else None)
    res["rank"] = float(np.median(ranks)) if ranks else None
    for label, hours in (("oi_1h", 1), ("oi_4h", 4), ("oi_24h", 24)):
        target = pd.Timestamp(now) - pd.Timedelta(hours=hours)
        best = None
        for s in snaps[:-1]:
            ts = pd.Timestamp(s["ts"])
            if abs((ts - target).total_seconds()) <= 35 * 60:
                if best is None or abs((ts - target).total_seconds()) < abs((pd.Timestamp(best["ts"]) - target).total_seconds()):
                    best = s
        if best is None:
            res[label] = None
            continue
        chg = [(out["oi"][v] / best["oi"][v] - 1) * 100 for v in out["oi"] if v in best["oi"] and best["oi"][v]]
        res[label] = float(np.median(chg)) if len(chg) >= 2 else None
    return res


def fetch_flows(st, now):
    """GLD holdings and CFTC COT positions for gold and WTI (refreshed once a day)."""
    cache = st.cache.setdefault("flows", {})
    today = pd.Timestamp(now).strftime("%Y-%m-%d")
    wanted = ("gld", "cot_GOLD", "cot_OIL", "wti_spread")
    if cache.get("day") == today and (all(k in cache for k in wanted) or
                                      cache.get("tried", "") > (pd.Timestamp(now) - pd.Timedelta(hours=3)).isoformat()):
        return cache                        # refreshed once a day; missing parts are retried every 3 hours
    out = {"day": today, "tried": pd.Timestamp(now).isoformat()}
    if cache.get("day") == today:
        out.update({k: cache[k] for k in wanted if k in cache})
    status, text = http_get("https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv")
    try:
        lines = [l for l in text.splitlines() if l.strip()]
        head_i = next(i for i, l in enumerate(lines) if "tonnes" in l.lower())
        df = pd.read_csv(io.StringIO("\n".join(lines[head_i:])))
        col = [c for c in df.columns if "tonnes" in c.lower()][0]
        dcol = df.columns[0]
        df = df[[dcol, col]].dropna()
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna()
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True)
        df = df.dropna().sort_values(dcol)
        s = df[col].values
        if len(s) >= 6:
            out["gld"] = {"tonnes": float(s[-1]), "chg5": float((s[-1] / s[-6] - 1) * 100)}
        note_source("gld", True)
    except Exception:
        txt = (text or "").lstrip()[:200].lower()
        note_source("gld", False, f"{status}" if status != 200 else
                    ("the site sent a web page instead of the data file (blocked)" if txt.startswith("<") else
                     "the data file's layout changed"))
    for mkt, code in (("GOLD", "088691"), ("OIL", "067651")):
        j = http_json("https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
                      params={"cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd DESC",
                              "$limit": 160})
        try:
            net = [float(r["m_money_positions_long_all"]) - float(r["m_money_positions_short_all"]) for r in j]
            if len(net) >= 52:
                hist = np.array(net[:156])
                out["cot_" + mkt] = {"net": net[0], "rank": float((hist[1:] < hist[0]).mean())}
            note_source("cot_" + mkt.lower(), True)
        except Exception:
            note_source("cot_" + mkt.lower(), False)
    try:                                            # WTI front month minus second month (daily closes)
        (y1, m1), (y2, m2) = cl_front_months(pd.Timestamp(now).date())
        t1 = f"CL{CL_CODES[m1 - 1]}{y1 % 100:02d}.NYM"
        t2 = f"CL{CL_CODES[m2 - 1]}{y2 % 100:02d}.NYM"
        a, b = fetch_yahoo(t1, "1mo", "1d"), fetch_yahoo(t2, "1mo", "1d")
        if a is not None and b is not None:
            sp = (a["Close"].groupby(a.index.normalize()).last() - b["Close"].groupby(b.index.normalize()).last()).dropna()
            if len(sp) >= 6:
                out["wti_spread"] = {"spread": round(float(sp.iloc[-1]), 3), "chg5": round(float(sp.iloc[-1] - sp.iloc[-6]), 3),
                                     "contracts": f"{t1} - {t2}"}
        note_source("wti_spread", "wti_spread" in out)
    except Exception:
        note_source("wti_spread", False)
    st.cache["flows"] = out
    return out


MACRO_TICKERS = {"dollar": "DX-Y.NYB", "yield10": "^TNX", "nasdaq": "NQ=F", "sp500": "ES=F", "vix": "^VIX",
                 "copper": "HG=F", "tnote": "ZN=F"}   # 10-year note futures trade ~23h: yield backup (opposite sign)
MACRO_DELAY = {"dollar": 30, "yield10": 15, "nasdaq": 10, "sp500": 10, "vix": 15, "copper": 30, "tnote": 10}


def macro_from(series, now):
    """z = last 24h change / normal size of a daily change over the last 60 trading days."""
    out = {"z": {}, "fresh": False}
    fresh_any = False
    for k, h in series.items():
        if h is None or len(h) < 50:
            out["z"][k] = None
            continue
        age = (pd.Timestamp(now) - (h.index[-1] + pd.Timedelta(hours=1))).total_seconds() / 60
        daily = h["Close"].groupby(h.index.normalize()).last()
        sd = daily.diff().iloc[-61:].std()
        past = h["Close"][h.index <= h.index[-1] - pd.Timedelta(hours=24)]
        if age > CFG["MACRO_MAX_AGE_MIN"] or not len(past) or not finite(sd) or sd == 0:
            out["z"][k] = None
            continue
        fresh_any = True
        out["z"][k] = float((h["Close"].iloc[-1] - past.iloc[-1]) / sd)
        if k == "dollar":
            ema20 = ema(daily, 20)
            rise = (ema20.iloc[-1] / ema20.iloc[-6] - 1) * 100 if len(ema20.dropna()) > 6 else 0
            out["dxy_up"] = bool(daily.iloc[-1] > ema20.iloc[-1] and rise > 0.20)
            out["dxy_down"] = bool(daily.iloc[-1] < ema20.iloc[-1] and rise < -0.20)
    if out["z"].get("yield10") is None and out["z"].get("tnote") is not None:
        out["z"]["yield10"] = -out["z"]["tnote"]          # note prices move opposite to yields
        out["yield_source"] = "10-year note futures (ZN=F), sign flipped"
    out["z"].pop("tnote", None)
    vix = series.get("vix")
    if vix is not None and len(vix):
        vd = vix["Close"].groupby(vix.index.normalize()).last()
        v, avg = float(vd.iloc[-1]), float(vd.iloc[-20:].mean())
        out["vix"] = v
        out["risk_off"] = bool(v > 25 or v > 1.2 * avg)
        out["risk_on"] = bool(v < 18 and v < avg)
    out.setdefault("risk_off", False)
    out.setdefault("dxy_up", False)
    out.setdefault("dxy_down", False)
    out["fresh"] = fresh_any
    return out


def fetch_fng(st, now):
    c = st.cache.get("fng")
    if c and c.get("ts", "") >= (pd.Timestamp(now) - pd.Timedelta(hours=3)).isoformat():
        return c
    j = http_json("https://api.alternative.me/fng/", params={"limit": 31})
    try:
        vals = [float(d["value"]) for d in j["data"]]
        c = {"ts": pd.Timestamp(now).isoformat(), "value": vals[0], "avg30": float(np.mean(vals[:30]))}
        st.cache["fng"] = c
        note_source("fear_greed", True)
        return c
    except Exception:
        note_source("fear_greed", False)
        return c if c and c.get("ts", "") >= (pd.Timestamp(now) - pd.Timedelta(hours=30)).isoformat() else None


EVENT_RULES = [   # (regex, minutes before, minutes after, markets, label)
    (r"\bcpi\b", 90, 90, ("BTC", "GOLD", "OIL"), "US inflation (CPI)"),
    (r"^(?!.*\badp\b).*(non-?farm|\bunemployment rate\b)", 90, 90, ("GOLD", "OIL"), "US jobs report"),
    (r"\bpce\b", 60, 60, ("GOLD",), "PCE inflation"),
    (r"\bppi\b|retail sales|\bgdp\b|unemployment claims|jobless claims", 30, 60, ("GOLD",), "US data"),
    (r"\bism\b|jolts|consumer sentiment|\buom\b", 30, 60, ("GOLD",), "US data (10:00)"),
    (r"federal funds rate|fomc statement|fomc press conference|fomc economic projections", 60, 120,
     ("BTC", "GOLD", "OIL"), "Fed rate decision"),
    (r"fomc meeting minutes", 60, 60, ("BTC", "GOLD"), "Fed minutes"),
    (r"crude oil inventories", 30, 60, ("OIL",), "EIA oil stocks"),
    (r"\bapi weekly", 30, 60, ("OIL",), "API oil stocks"),
    (r"\bopec", None, None, ("OIL",), "OPEC+ meeting"),
]
# Backup dates if the calendar feed is down (official schedules: federalreserve.gov, bls.gov). Add new years here.
FOMC_DAYS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28",
             "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
             "2027-10-27", "2027-12-08"]
CPI_DAYS = ["2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12",
            "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10"]
JOBS_DAYS = ["2026-11-06", "2026-12-04", "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07",
             "2027-06-04", "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08", "2027-11-05", "2027-12-03"]
# US holidays / early closes for CME metals and energy (best effort - check each year): no new gold or WTI trades
# after 10:00 New York on these days; open trades are closed before any long break.
CME_SHORT_DAYS = {"2026-04-02", "2026-11-26", "2026-11-27", "2026-12-24", "2026-12-31", "2027-01-18", "2027-02-15", "2027-03-25",
                  "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-11-26", "2027-12-23",
                  "2027-12-24", "2027-12-30", "2027-12-31"}


def ny_to_utc(day, hh, mm):
    t = pd.Timestamp(dt.datetime.combine(pd.Timestamp(day).date(), dt.time(hh, mm)), tz=NY)
    return t.tz_convert("UTC").tz_localize(None)


def fetch_events(st, now):
    """High and medium impact US events from the ForexFactory weekly JSON, plus fixed backups."""
    cache = st.cache.setdefault("events", {})
    if not cache.get("fetched") or cache["fetched"] < (pd.Timestamp(now) - pd.Timedelta(hours=5)).isoformat():
        got, ok_any = [], False
        for url in ("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                    "https://nfs.faireconomy.media/ff_calendar_nextweek.json"):
            j = http_json(url)
            if isinstance(j, list):
                ok_any = True
                for e in j:
                    try:
                        title = str(e.get("title", ""))
                        country = str(e.get("country", ""))
                        imp = str(e.get("impact", ""))
                        if country not in ("USD", "ALL") and "opec" not in title.lower():
                            continue
                        if imp not in ("High", "Medium") and "opec" not in title.lower():
                            continue
                        t = utc_ts(pd.Timestamp(e["date"]))
                        got.append({"title": title, "time": t.isoformat(), "impact": imp})
                    except Exception:
                        continue
        if ok_any:
            cache["events"] = got
            cache["fetched"] = pd.Timestamp(now).isoformat()
            note_source("calendar", True)
        else:
            note_source("calendar", False)
    events = list(cache.get("events", []))
    # backups: EIA every Wednesday 10:30 NY, API every Tuesday 16:30 NY, Fed 14:00 NY, CPI and jobs 08:30 NY
    base = pd.Timestamp(now).normalize()
    for k in range(-2, 9):
        day = base + pd.Timedelta(days=k)
        if day.weekday() == 2:
            events.append({"title": "Crude Oil Inventories (backup)", "time": ny_to_utc(day, 10, 30).isoformat()})
        if day.weekday() == 1:
            events.append({"title": "API Weekly Statistical Bulletin (backup)", "time": ny_to_utc(day, 16, 30).isoformat()})
    for d in FOMC_DAYS:
        events.append({"title": "Federal Funds Rate (backup)", "time": ny_to_utc(d, 14, 0).isoformat()})
    for d in CPI_DAYS:
        events.append({"title": "CPI m/m (backup)", "time": ny_to_utc(d, 8, 30).isoformat()})
    for d in JOBS_DAYS:
        events.append({"title": "Non-Farm Employment Change (backup)", "time": ny_to_utc(d, 8, 30).isoformat()})
    return events


def event_status(events, mkt, now, horizon_h=24):
    """(active window or None, list of upcoming events in the next 24 hours)."""
    nowt = pd.Timestamp(now)
    active, upcoming, seen = None, [], set()
    for e in events:
        title = e["title"]
        t = pd.Timestamp(e["time"])
        for rx, before, after, mkts, label in EVENT_RULES:
            if mkt not in mkts or not re.search(rx, title.lower()):
                continue
            key = (label, t.floor("h"))
            if key in seen:
                break
            seen.add(key)
            if before is None:                         # OPEC+: the whole New York day
                ny_day = to_ny(t).date()
                start = ny_to_utc(ny_day, 0, 0)
                end = start + pd.Timedelta(days=1)
                if to_ny(t).weekday() >= 5:            # weekend meeting: no new trades from Friday 12:00 New York
                    fri = ny_day - dt.timedelta(days=to_ny(t).weekday() - 4)      # (22:30 IST in winter)
                    start = ny_to_utc(fri, 12, 0)
            else:
                start, end = t - pd.Timedelta(minutes=before), t + pd.Timedelta(minutes=after)
            if start <= nowt <= end and active is None:
                active = {"label": label, "time": t, "start": start, "end": end}
            if nowt <= t <= nowt + pd.Timedelta(hours=horizon_h):
                upcoming.append({"label": label, "time": t, "start": start, "end": end})
            break
    upcoming.sort(key=lambda e: e["time"])
    return active, upcoming


# ----------------------------------------------------------------- news (code filters, AI labels, code score)
_GDELT_LAST = 0.0
NEWS_FEEDS = {
    "BTC": ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss"],
    "GOLD": [],                       # Google News and GDELT cover gold (the Kitco feed looks discontinued)
    "OIL": ["https://oilprice.com/rss/main"],
}
NEWS_QUERY = {"BTC": "bitcoin OR BTC OR crypto", "GOLD": '"gold price" OR bullion OR "gold futures"',
              "OIL": '"crude oil" OR WTI OR OPEC OR "oil prices"'}
GDELT_QUERY = {"BTC": "(bitcoin OR BTC OR crypto)", "GOLD": '("gold price" OR bullion OR "gold futures")',
               "OIL": '("crude oil" OR WTI OR OPEC OR "oil prices")'}    # GDELT needs OR groups in brackets
NEWS_KEYWORDS = {"BTC": ["bitcoin", "btc", "crypto", "etf", "sec", "stablecoin", "binance", "coinbase", "fed"],
                 "GOLD": ["gold", "bullion", "fed", "inflation", "dollar", "treasury", "central bank", "yields"],
                 "OIL": ["oil", "crude", "wti", "brent", "opec", "eia", "inventor", "refiner", "sanction", "barrel"]}
NEWS_DROP = re.compile(r"\b(rises?|falls?|jumps?|slips?|gains?|drops?|climbs?|edges?) \d|price prediction|"
                       r"analyst says|here'?s why|could reach|forecast|technical analysis|opinion|will .* hit",
                       re.I)


def fetch_news_items(mkt, now):
    items = []
    urls = list(NEWS_FEEDS.get(mkt, []))
    urls.append("https://news.google.com/rss/search?q=" + quote_plus(NEWS_QUERY[mkt] + " when:1h") +
                "&hl=en-US&gl=US&ceid=US:en")
    for url in urls:
        status, text = http_get(url, timeout=15, tries=1)
        if status != 200:
            note_source("news:" + url.split("/")[2], False, str(status))
            continue
        try:
            root = ET.fromstring(text.encode("utf-8", "ignore"))
            for it in root.iter("item"):
                title = safe_str(it.findtext("title") or "").strip()
                pub = it.findtext("pubDate")
                src = it.find("source")
                domain = safe_str(src.text if src is not None and src.text else url.split("/")[2]).strip().lower()
                if not title or not pub:
                    continue
                t = utc_ts(parsedate_to_datetime(pub))
                items.append({"title": title, "time": t, "outlet": domain})
            note_source("news:" + url.split("/")[2], True)
        except Exception:
            note_source("news:" + url.split("/")[2], False, "parse")
    global _GDELT_LAST
    wait = 6.0 - (time.time() - _GDELT_LAST)            # GDELT asks for at most one request every 5 seconds
    if wait > 0:
        time.sleep(wait)
    _GDELT_LAST = time.time()
    status, text = http_get("https://api.gdeltproject.org/api/v2/doc/doc",
                            params={"query": GDELT_QUERY[mkt] + " sourcelang:english", "mode": "artlist",
                                    "format": "json", "timespan": "90min", "maxrecords": 50, "sort": "datedesc"},
                            timeout=25, tries=1)
    try:
        if status != 200:
            raise ValueError(f"{status}")
        if not text.strip():
            j = {}                                      # an empty answer = no stories right now
        else:
            j = json.loads(text, strict=False)          # GDELT titles can hold raw control characters
        for a in (j.get("articles") or []):
            try:
                t = pd.Timestamp(dt.datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ"))
                items.append({"title": safe_str(a["title"]).strip(), "time": t,
                              "outlet": safe_str(a.get("domain", "gdelt")).lower()})
            except Exception:
                continue
        note_source("news:gdelt", True)
    except Exception:
        note_source("news:gdelt", False, str(status) if status != 200 else
                    "not JSON: " + clip(re.sub(r"\s+", " ", text or ""), 40))
    return clean_news(items, mkt, now)


def _words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def clean_news(items, mkt, now):
    """Keep fresh, on-topic, factual stories; merge near-duplicates and count outlets. At most 12."""
    nowt = pd.Timestamp(now)
    keep = []
    for it in items:
        age = (nowt - pd.Timestamp(it["time"])).total_seconds() / 60
        if age < -5 or age > 90:
            continue
        low = it["title"].lower()
        if not any(k in low for k in NEWS_KEYWORDS[mkt]) or NEWS_DROP.search(it["title"]):
            continue
        it = dict(it, age=age)
        for g in keep:
            a, b = _words(g["title"]), _words(it["title"])
            if a and b and len(a & b) / max(1, min(len(a), len(b))) > 0.6:
                g["outlets"].add(it["outlet"])
                if pd.Timestamp(it["time"]) < pd.Timestamp(g["time"]):
                    g["time"], g["age"] = it["time"], age
                break
        else:
            keep.append(dict(it, outlets={it["outlet"]}))
    keep.sort(key=lambda g: g["age"])
    out = []
    for g in keep[:12]:
        out.append({"title": g["title"], "time": pd.Timestamp(g["time"]).isoformat(), "age": round(g["age"], 1),
                    "outlets": sorted(g["outlets"]), "key": hashlib.md5(g["title"].lower().encode()).hexdigest()[:12]})
    return out


def news_score(stories, labels, f, atr_v, now):
    """Code decides the news vote from the AI labels."""
    up = dn = 0.0
    best = None
    for s, lab in zip(stories, labels):
        if not lab:
            continue
        valid = (s["age"] <= 60 and lab["impact"] in ("A", "B") and lab["new"] and lab["direction"] in ("UP", "DOWN")
                 and lab["uncertainty"] == "low")
        s["valid"] = valid
        if not valid:
            continue
        val = lab["strength"] * (2 if lab["impact"] == "A" else 1) * min(len(s["outlets"]), 3) / 3
        if lab["direction"] == "UP":
            up += val
        else:
            dn += val
        if best is None or val > best[0]:
            best = (val, s, lab)
    net = up - dn                                   # the score is a sum of +/- values (research step 4)
    data = {"news_up_score": round(up, 2), "news_down_score": round(dn, 2), "news_net_score": round(net, 2)}
    if up >= 2 and dn >= 2:
        return "WAIT", "news.both", "strong stories on both sides", data, net
    if abs(net) < 3:
        return "WAIT", "news.weak", "no strong fresh news (net score under 3)", data, net
    d = "BUY" if net > 0 else "SELL"
    want = "UP" if d == "BUY" else "DOWN"
    best = max(((v, s, lab) for v, s, lab in [(lab["strength"] * (2 if lab["impact"] == "A" else 1) *
                                                min(len(s["outlets"]), 3) / 3, s, lab)
                                               for s, lab in zip(stories, labels) if lab and s.get("valid")]
                if lab["direction"] == want), key=lambda b: b[0])
    s = best[1]
    t = pd.Timestamp(s["time"])
    before = f["c"][f.index + pd.Timedelta(hours=1) <= t]
    if not len(before):
        return "WAIT", "news.no_price", "no price before the headline", data, net
    move = (f["c"].iloc[-1] - before.iloc[-1]) / atr_v * (1 if d == "BUY" else -1)
    data["move_since_headline_atr"] = round(float(move), 2)
    if move > 1.5:
        return "WAIT", "news.priced_in", "price already moved more than 1.5x ATR - priced in", data, up - dn
    if move <= -0.3:
        return "WAIT", "news.disagree", "price moved against the news - the market disagrees", data, up - dn
    if move < 0.3:
        if s["age"] > 30:
            return "WAIT", "news.no_follow", "story over 30 minutes old and price has not followed", data, up - dn
        return "WAIT", "news.too_early", "price has not followed yet (under 0.3x ATR)", data, up - dn
    return d, "news.vote", f"fresh {'good' if d == 'BUY' else 'bad'} news (score {max(up, dn):.1f}) and price " \
                           f"following ({move:.2f}x ATR)", data, up - dn


# =============================================================== AI brains
class Brains:
    def __init__(self, st, now):
        self.st, self.now = st, pd.Timestamp(now)
        self.key = {"gemini": os.getenv("GEMINI_API_KEY", ""), "groq": os.getenv("GROQ_API_KEY", ""),
                    "openrouter": os.getenv("OPENROUTER_API_KEY", "")}
        self.calls, self.down, self.last_t, self.log = 0, set(), {}, []
        self.available = {}
        self.t0 = time.time()
        self.last_err = {}          # provider -> (status, first words of the error) - shown in the test message
        self.list_status = {}       # provider -> status of the "which models exist" request

    def any(self):
        return any(v for v in self.key.values())

    def discover(self, force=False):
        """Which model names exist today (cached for a day per provider; a provider whose list failed - bad key,
        site down - is asked again next hour)."""
        d = self.st.models["discovered"]
        today = self.now.strftime("%Y-%m-%d")
        cached = dict(d.get("models", {})) if d.get("day") == today and not force else {}
        if all(p in cached for p in ("groq", "openrouter", "gemini") if self.key[p]):
            self.available = cached
            return
        found = dict(cached)

        def get(p, url, params=None, headers=None):
            status, text = http_get(url, params=params, headers=headers, timeout=20, tries=2)
            self.list_status[p] = (status, clip(re.sub(r"\s+", " ", text or ""), 160) if status != 200 else "")
            if status != 200:
                return None
            try:
                return json.loads(text)
            except Exception:
                self.list_status[p] = ("not JSON", clip(text, 80))
                return None
        def rows(j, key):                              # a list of dicts, whatever shape the answer has
            v = j.get(key) if isinstance(j, dict) else None
            return [m for m in v if isinstance(m, dict)] if isinstance(v, list) else []
        if self.key["groq"] and "groq" not in found:
            j = get("groq", "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.key['groq']}"})
            ids = [str(m["id"]) for m in rows(j, "data") if m.get("id")]
            if ids:
                found["groq"] = ids
        if self.key["openrouter"] and "openrouter" not in found:
            j = get("openrouter", "https://openrouter.ai/api/v1/models")
            ids = [str(m["id"]) for m in rows(j, "data") if m.get("id") and (
                str(m["id"]).endswith(":free") or
                str((m.get("pricing") or {}).get("prompt", "1") if isinstance(m.get("pricing"), dict) else "1")
                in ("0", "0.0"))]
            if ids:
                found["openrouter"] = ids
        if self.key["gemini"] and "gemini" not in found:
            j = get("gemini", "https://generativelanguage.googleapis.com/v1beta/models", params={"pageSize": 200},
                    headers={"x-goog-api-key": self.key["gemini"]})
            ids = [str(m["name"]).replace("models/", "") for m in rows(j, "models") if m.get("name") and
                   "generateContent" in (m.get("supportedGenerationMethods") or [])]
            if ids:
                found["gemini"] = ids
        self.available = found
        self.st.models["discovered"] = {"day": today, "models": found}

    def resolve(self, provider, tag):
        prefs = PREFS[(provider, tag)]
        have = self.available.get(provider)
        if not have:
            return prefs[0]
        for m in prefs:
            if m in have:
                return m
        keys = {"lite": ["flash-lite"], "flash": ["flash"], "big": ["120b", "70b", "ultra", "large"],
                "mid": ["32b", "27b", "20b"], "small": ["20b", "8b", "instant"], "alt": ["70b", "llama-4", "8b"],
                "tiny": ["8b", "instant", "20b"], "a": ["flash", "it"], "b": ["it"], "c": ["it", "instruct"]}.get(tag, [])
        for k in keys:
            for m in have:
                if k in m and ("embed" not in m and "tts" not in m and "image" not in m and "vision" not in m):
                    return m
        return have[0]

    def plan(self, role):
        out = []
        forced = os.getenv("MODEL_" + role.upper())
        if forced and ":" in forced:
            p, m = forced.split(":", 1)
            out.append((p, m))
        swap = int(self.st.models["swap"].get(role, 0))
        lst = ROLE_MODELS[role]
        lst = lst[swap % len(lst):] + lst[:swap % len(lst)]
        for p, tag in lst:
            out.append((p, self.resolve(p, tag)))
        return out

    def _err(self, p, status):
        k = f"{self.now.strftime('%Y-%m-%d')}|{p}|{status}"
        self.st.models["errors"][k] = self.st.models["errors"].get(k, 0) + 1

    def _send(self, p, m, prompt, temperature, max_tokens, json_mode, reasoning, simple=False):
        """One request. Returns (status, text, retry_after, error_text). status "length" = the model used its
        whole token budget (thinking) and gave no answer: retrying the same model does not help."""
        if p == "gemini":
            if simple:                                   # fallback after a 400: plainest possible request
                cfg = {"maxOutputTokens": max_tokens + 2048}
            elif "gemini-3" in m:                        # Gemini 3: keep the default temperature, think little
                cfg = {"maxOutputTokens": max_tokens + 2048, "thinkingConfig": {"thinkingLevel": "low"}}
            elif "gemini-2.5" in m and "pro" not in m:   # Gemini 2.5 Flash / Flash-Lite: thinking off
                cfg = {"temperature": temperature, "maxOutputTokens": max_tokens + 256,
                       "thinkingConfig": {"thinkingBudget": 0}}
            else:
                cfg = {"temperature": temperature, "maxOutputTokens": max_tokens + 1024}
            if json_mode and not simple:
                cfg["responseMimeType"] = "application/json"
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                              headers={"x-goog-api-key": self.key[p], "Content-Type": "application/json"},
                              json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                    "generationConfig": cfg}, timeout=CFG["AI_TIMEOUT"])
            if r.status_code != 200:
                return r.status_code, None, r.headers.get("retry-after"), r.text[:3000]
            j = self._json(r)
            if j is None:
                return "unreadable", None, None, clip(r.text, 120)
            cands = j.get("candidates")
            cand = cands[0] if isinstance(cands, list) and cands and isinstance(cands[0], dict) else {}
            parts = (cand.get("content") or {}).get("parts") if isinstance(cand.get("content"), dict) else []
            parts = [x for x in (parts or []) if isinstance(x, dict)]
            text = "".join(str(x.get("text", "")) for x in parts if not x.get("thought")).strip()
            if not text and str(cand.get("finishReason", "")).upper() == "MAX_TOKENS":
                return "length", None, None, "MAX_TOKENS"
            return (200, text, None, "") if text else ("empty", None, None, "an empty answer")
        url = {"groq": "https://api.groq.com/openai/v1/chat/completions",
               "openrouter": "https://openrouter.ai/api/v1/chat/completions"}[p]
        body = {"model": m, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if not simple:
            if p == "groq" and "gpt-oss" in m:
                body["reasoning_effort"] = reasoning or "low"       # reasoning tokens count against max_tokens
                body["max_tokens"] = max_tokens + 1024
            elif p == "groq" and "qwen" in m:
                body["reasoning_effort"] = "none"                    # Groq: switches Qwen's thinking off
                body["max_tokens"] = max_tokens + 512
            elif p == "openrouter":
                body["reasoning"] = {"effort": "low", "exclude": True}
                body["max_tokens"] = max_tokens + 1024
        r = requests.post(url, headers={"Authorization": f"Bearer {self.key[p]}", "Content-Type": "application/json"},
                          json=body, timeout=CFG["AI_TIMEOUT"])
        if r.status_code != 200:
            return r.status_code, None, r.headers.get("retry-after"), r.text[:3000]
        j = self._json(r)
        if j is None:
            return "unreadable", None, None, clip(r.text, 120)
        if j.get("error"):                                      # OpenRouter can return an error inside a 200
            code = (j["error"] or {}).get("code") if isinstance(j["error"], dict) else None
            return int(code) if str(code).isdigit() else 502, None, None, str(j["error"])[:3000]
        chs = j.get("choices")
        ch = chs[0] if isinstance(chs, list) and chs and isinstance(chs[0], dict) else {}
        msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        text = str(msg.get("content") or "").strip()
        if not text and str(ch.get("finish_reason", "")).lower() == "length":
            return "length", None, None, "length"
        return (200, text, None, "") if text else ("empty", None, None, "an empty answer")

    @staticmethod
    def _json(r):
        """The answer as a dict, or None if it is not readable JSON (a web page, a list, ...)."""
        try:
            j = r.json()
        except Exception:
            return None
        return j if isinstance(j, dict) else None

    def _skip(self, p, m):
        now = self.now.isoformat()
        for k in (f"{p}|{m}", f"{p}|*"):
            until = self.st.models["skip_until"].get(k)
            if until and until > now:
                return True
        return False

    def _daily_reset(self, p):
        """When a daily quota comes back: Gemini at midnight Pacific time, the others after 6 hours."""
        if p == "gemini":
            la = self.now.tz_localize("UTC").tz_convert(ZoneInfo("America/Los_Angeles"))
            nxt = la.normalize() + pd.DateOffset(days=1)
            return nxt.tz_convert("UTC").tz_localize(None) + pd.Timedelta(minutes=5)
        return self.now + pd.Timedelta(hours=6)

    def ask(self, role, prompt, temperature=0.2, max_tokens=350, json_mode=True, reasoning=None, exclude=()):
        """Main model, then backups. Busy/quota: retry the same model after 2 s and 6 s (+ up to 1 s), respecting
        'retry after'. A daily quota error skips that model (OpenRouter: the whole account) until the quota
        comes back. 3 failures in a row from one model = skip it for 60 minutes (saved between runs)."""
        if time.time() - self.t0 > CFG["RUN_DEADLINE_MIN"] * 60:
            return None, None, None                 # the run is getting long: no new AI calls (job limit 40 min)
        for p, m in self.plan(role):
            if not self.key.get(p) or p in self.down or p in exclude or self._skip(p, m):
                continue
            jm, simple = json_mode, False
            for attempt in range(3):
                if self.calls >= CFG["AI_MAX_CALLS"]:
                    return None, None, None
                gap = CFG["AI_GAP_S"] - (time.time() - self.last_t.get(p, 0.0))
                if gap > 0:
                    time.sleep(gap)
                self.calls += 1
                self.last_t[p] = time.time()
                try:
                    status, text, retry_after, err = self._send(p, m, prompt, temperature, max_tokens, jm, reasoning,
                                                                simple)
                except Exception as e:
                    status, text, retry_after, err = 0, None, None, type(e).__name__
                    log(f"AI {p}/{m}: {type(e).__name__}")
                if status == 200 and text:
                    self.st.models["fail_streak"][f"{p}|{m}"] = 0
                    self.log.append((role, p, m, "ok"))
                    return text, p, m
                self._err(p, status or "network")
                self.last_err[p] = (status or "network", m, clip(re.sub(r"\s+", " ", str(err or "")), 220))
                low = re.sub(r"[\s_-]", "", (err or "").lower())
                if status == 429 and any(s in low for s in ("perday", "daily", "tpd", "rpd")):
                    key = f"{p}|*" if p == "openrouter" else f"{p}|{m}"
                    self.st.models["skip_until"][key] = self._daily_reset(p).isoformat()
                    log(f"AI {p}/{m}: daily quota used up - skipped until {self.st.models['skip_until'][key]}")
                    break
                if status == "length":
                    break                               # thinking used the whole budget: try the next model
                if status in (400, 422) and not simple:
                    simple, jm = True, False            # unknown settings or no JSON mode: plainest request
                    continue
                if status in (401, 403):
                    self.down.add(p)
                    break
                if status in (0, 200, "empty", "unreadable", 408, 429, 500, 502, 503, 504) and attempt < 2:
                    wait = (2, 6)[attempt] + random.random()
                    try:
                        if retry_after:
                            wait = max(wait, min(float(retry_after), 15))
                    except ValueError:
                        pass
                    time.sleep(wait)
                    continue
                break
            k = f"{p}|{m}"
            self.st.models["fail_streak"][k] = self.st.models["fail_streak"].get(k, 0) + 1
            if self.st.models["fail_streak"][k] >= 3:
                self.st.models["skip_until"][k] = max(self.st.models["skip_until"].get(k, ""),
                                                      (self.now + pd.Timedelta(minutes=60)).isoformat())
                self.st.models["fail_streak"][k] = 0
            self.log.append((role, p, m, "failed"))
        return None, None, None


# ----------------------------------------------------------------- the code checker for AI answers
def number_ok(key, val, data):
    if key not in data:
        return False
    true = data[key]
    try:
        v, t = float(val), float(true)
    except (TypeError, ValueError, OverflowError):
        return str(val).strip().upper() == str(true).strip().upper()
    if not (math.isfinite(v) and math.isfinite(t)):
        return False
    tol = max(0.005 * abs(t), 1.0 if ("rsi" in key or "greed" in key) else 0.0)     # prices +/-0.5%, RSI +/-1
    if abs(v - t) <= tol + 1e-12:
        return True
    s = str(val).strip().lower()
    dec = len(s.split(".")[1]) if "." in s and "e" not in s else 0
    return dec >= 1 and abs(round(t, dec) - v) <= 0.5 * 10 ** -dec + 1e-12      # a correct rounding is fine


def forbidden_vote(agent, vote, rule):
    """Votes an agent's own table never allows, even as 'own judgement'."""
    if agent == "mood":
        fg = rule["data"].get("fear_greed")
        if finite(fg) and ((fg >= 75 and vote == "SELL") or (fg <= 25 and vote == "BUY")):
            return "the mood table never allows SELL on greed or BUY on fear"
    return ""


def check_voice(ans, agent, rule):
    """The fixed checks, in order: format, numbers, vote against the agent's own rule.
    Returns (ok, problem, vote, kind) with kind = "format", "numbers" or "contra"."""
    if not isinstance(ans, dict):
        return False, "the answer was not valid JSON", None, "format"
    vote = str(ans.get("vote", "")).upper().strip()
    if vote not in ("BUY", "SELL", "WAIT"):
        return False, f"vote must be BUY, SELL or WAIT (got {clip(vote, 20)})", None, "format"
    if str(ans.get("agent", "")).strip().lower() != agent:
        return False, f'"agent" must be "{agent}"', vote, "format"
    if str(ans.get("rule_id", "")).strip() != rule["rule_id"]:
        return False, f'"rule_id" must be "{rule["rule_id"]}"', vote, "format"
    if len(str(ans.get("reason", "")).split()) > 50:
        return False, "the reason must be 50 words or less", vote, "format"
    nums = ans.get("numbers") or {}
    if not isinstance(nums, dict) or not nums:
        return False, "quote 1 to 4 numbers from YOUR DATA in 'numbers'", vote, "format"
    shown = shown_data(rule["data"])
    for k, v in list(nums.items())[:6]:
        if not number_ok(str(k), v, shown):
            return False, f"the number {clip(k, 40)}={clip(v, 20)} does not match your data", vote, "numbers"
    rv = rule["vote"]
    if rv in ("BUY", "SELL") and vote != rv:
        return False, f"your rule says {rv} but you voted {vote}", vote, "contra"
    if rv == "WAIT" and vote != "WAIT":
        if agent not in OWN_JUDGEMENT_OK or rule["rule_id"] not in NEUTRAL_RULES:
            return False, f"your rule's WAIT ({rule['rule_id']}) is a deliberate block, not a neutral answer - " \
                          "you must vote WAIT", vote, "contra"
        bad = forbidden_vote(agent, vote, rule)
        if bad:
            return False, bad, vote, "contra"
    return True, "", vote, ""


VOICE_PROMPT = """You are the {name} agent on a {market} trading desk - a trader with 30 years of experience.
You see ONLY your own data below. Code has already worked out your rule.

YOUR DATA (numbers you may quote):
{data}

YOUR RULE ({rule_id}): {rule_vote} - {rule_reason}
{memory}
How to answer:
- If your rule says BUY or SELL, your vote must be the same as your rule.
- {wait_rule}
- Quote 1 to 4 numbers from YOUR DATA, with exactly the names and values shown.
- Reason: at most 50 words, plain English, no new numbers.
Reply with ONLY this JSON:
{{"agent": "{agent}", "vote": "BUY or SELL or WAIT", "rule_id": "{rule_id}", "numbers": {{"name": value}}, "reason": "..."}}"""


def memory_lines(st, mkt, agent, now, trend):
    """Statistics, not stories: this agent's own RULE record in the current trend state, last 60 days,
    only groups with 5+ graded calls, at most 5 lines."""
    cut = (pd.Timestamp(now) - pd.Timedelta(days=60)).isoformat()
    rows = [g for g in st.grading["graded"] if g["mkt"] == mkt and g["agent"] == agent and g["kind"] == "rule"
            and g["signal_ts"] >= cut and g.get("trend") == trend]
    groups = {}
    for g in rows:
        groups.setdefault((g.get("rule_id", "?"), g["d"]), []).append(g)
    lines = []
    for (rid, d), gs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(gs) < 5:
            continue
        hit = sum(1 for g in gs if g["hit"])
        avg = np.mean([g["r"] for g in gs])
        lines.append(f"{d} ({rid}): {len(gs)} graded, {hit} hit target ({hit / len(gs) * 100:.0f}%), average "
                     f"{avg:+.2f}R{'; small sample' if len(gs) < 30 else ''}")
        if len(lines) >= 5:
            break
    if not lines:
        return ""
    return f"YOUR RULE'S OWN RECORD in a {trend} trend, last 60 days (code statistics):\n" + "\n".join(lines) + "\n"


def ai_voice(brains, st, x, agent, rule, now):
    """The agent's AI brain explains (and, when its rule is neutral, may add its own vote). Retries (at most 2):
    bad format -> 1 retry on the same model, then 1 on a backup model; wrong numbers -> 1 retry; a vote against
    the rule -> 1 retry naming it. Still wrong: the rule's answer is used (contradiction = "corrected")."""
    final = dict(rule)
    final.update({"model": "code", "ai": "none", "flag": ""})
    if agent in NO_AI or not brains.any():
        final["ai"] = "code only"
        return final
    data = shown_data(rule["data"])
    if not data:
        final["ai"] = "code only (no data)"
        return final
    wait_rule = ("If your rule says WAIT, you may vote BUY or SELL only if your data clearly supports it; "
                 "otherwise WAIT." if (agent in OWN_JUDGEMENT_OK and rule["rule_id"] in NEUTRAL_RULES)
                 else "If your rule says WAIT, you must vote WAIT.")
    prompt = VOICE_PROMPT.format(name=PRETTY[agent], market=MARKETS[x["mkt"]]["name"], agent=agent,
                                 data=json.dumps(data), rule_id=rule["rule_id"], rule_vote=rule["vote"],
                                 rule_reason=rule["reason"], wait_rule=wait_rule,
                                 memory=memory_lines(st, x["mkt"], agent, now, x["trend"]))
    problem, kind, model, exclude, retries, used = "", "", None, (), 0, set()
    while True:
        p = prompt if not problem else prompt + f"\n\nYour last answer had a problem: {problem}. Answer again."
        text, prov, mdl = brains.ask(agent, p, exclude=exclude)
        if not text:
            if model is None:
                final["ai"] = "absent (AI not answering) - rule answer used"
                final["flag"] = "AI absent"
                return final
            break
        model = f"{BRAIN_NAME[prov]} {mdl}"
        ans = parse_json(text)
        ok, problem, vote, kind = check_voice(ans, agent, rule)
        _contra_stat(st, now, agent, model, kind == "contra")      # graded per agent + model
        if ok:
            final["model"] = model
            final["ai"] = "checked"
            final["reason"] = clip(ans.get("reason") or rule["reason"], 300)
            if rule["vote"] == "WAIT" and vote in ("BUY", "SELL"):
                final["vote"] = vote
                final["flag"] = "own judgement"
            return final
        if retries >= 2 or (kind in used and kind != "format"):
            break                                   # numbers / contradiction get one retry only
        exclude = (prov,) if (kind == "format" and "format" in used) else ()   # 2nd format retry: backup model
        used.add(kind)
        retries += 1
    final["model"] = model or "code"
    final["ai"] = f"rejected by the checker ({problem}) - rule answer used"
    final["flag"] = "corrected" if kind == "contra" else "AI rejected"
    return final


def shown_data(data):
    """The numbers exactly as the agent sees them (rounded); the checker compares against these."""
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}


def _contra_stat(st, now, agent, model, contra):
    k = f"{iso_week(now)}|{agent}|{model}"
    v = st.models["contra"].get(k, [0, 0])
    v[0] += 1
    v[1] += int(contra)
    st.models["contra"][k] = v


def weekly_model_swaps(st, now):
    """If an agent+model contradicts its rule more than 15% of the time in a week (10+ calls), swap its model."""
    wk = iso_week(pd.Timestamp(now) - pd.Timedelta(days=7))
    done = st.models.setdefault("swap_done", {})
    if done.get(wk):
        return []
    done[wk] = True
    msgs = []
    for k, (n, c) in st.models["contra"].items():
        week, agent, model = k.split("|", 2)
        if week == wk and n >= 10 and c / n > 0.15 and agent in ROLE_MODELS:
            st.models["swap"][agent] = int(st.models["swap"].get(agent, 0)) + 1
            msgs.append(f"{PRETTY.get(agent, agent)}: {model} went against its rule {c}/{n} times - model swapped")
    return msgs


# ----------------------------------------------------------------- news labels, debate, inspector (AI)
LABEL_PROMPT = """You label news for a {market} trading desk. For each story give:
impact: A (central bank, CPI, jobs, regulation, ETF, war, sanctions, supply shock, hack, OPEC, EIA surprise),
        B (industry news, big flows, official statements) or C (opinion, recap, promotion);
new: true if it reports a new fact, false if it repeats known news;
direction: UP, DOWN or UNKNOWN for the {market} price in the next 1-4 hours;
strength: 1, 2 or 3;  uncertainty: low or high;  quote: the headline's key words (at most 12 words).
Do not vote. Do not use outside knowledge about prices.
STORIES:
{stories}
Reply with ONLY this JSON: {{"labels": [{{"id": 1, "impact": "A", "new": true, "direction": "UP", "strength": 2,
"uncertainty": "low", "quote": "..."}}]}}"""


def label_news(brains, mkt, stories, seen):
    if not stories or not brains.any():
        return [None] * len(stories), None
    lines = "\n".join(f"{i + 1}. [{s['age']:.0f} min ago] {s['title']} ({len(s['outlets'])} outlets)"
                      for i, s in enumerate(stories))
    text, prov, mdl = brains.ask("news", LABEL_PROMPT.format(market=MARKETS[mkt]["name"], stories=lines),
                                 temperature=0.0, max_tokens=900)
    j = parse_json(text)
    labels = [None] * len(stories)
    if not j or not isinstance(j.get("labels"), list):
        return labels, None
    for lab in j["labels"]:
        try:
            i = int(lab["id"]) - 1
            if not 0 <= i < len(stories):
                continue
            impact = str(lab["impact"]).upper()
            direction = str(lab["direction"]).upper()
            strength = int(lab["strength"])
            unc = str(lab["uncertainty"]).lower()
            new = bool(lab["new"]) if not isinstance(lab["new"], str) else lab["new"].lower() == "true"
            if impact not in ("A", "B", "C") or direction not in ("UP", "DOWN", "UNKNOWN") or strength not in (1, 2, 3) \
                    or unc not in ("low", "high"):
                continue
            if f"{mkt}|{stories[i]['key']}" in seen:
                new = False                                   # already seen in an earlier hour
            labels[i] = {"impact": impact, "direction": direction, "strength": strength, "uncertainty": unc, "new": new,
                         "quote": clip(lab.get("quote", ""), 80)}
        except Exception:
            continue
    return labels, f"{BRAIN_NAME[prov]} {mdl}"


DEBATE_PROMPT = """You are the {SIDE} on a {market} trading desk - a trader with 30 years of experience.
Make the case to {ACTION} now, using ONLY the facts below.
FACTS (name = value):
{facts}
How to answer:
- Give 2 to 4 arguments. Each argument quotes ONE fact name and its exact value from FACTS.
- Your vote: {ACTION} only if {room_key} is 2.0 or more and you have at least 2 arguments; otherwise WAIT.
- At most 25 words per argument. Do not copy these instructions.
Reply with ONLY this JSON: {{"vote": "{ACTION} or WAIT", "arguments": [{{"fact": "name", "value": 1.0, "text": "..."}}]}}"""

REBUT_PROMPT = """You are the {SIDE} on a {market} trading desk. The other side said:
{other}
FACTS (name = value):
{facts}
Keep your view unless the other side quotes a fact that contradicts yours.
You may knock down an argument of the other side only by quoting a DIFFERENT fact from FACTS that contradicts it.
Reply with ONLY this JSON: {{"rebuttals": [{{"target": 1, "fact": "name", "value": 1.0, "text": "..."}}]}}"""


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9_.]+", str(text).lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def debate_open(brains, x, side, facts, exclude=()):
    """Opening statement. Code checks the quotes, the room rule and copying of the prompt."""
    action = "BUY" if side == "bull" else "SELL"
    room_key = "room_up_to_resistance_R" if side == "bull" else "room_down_to_support_R"
    fl = "\n".join(f"{k} = {v}" for k, v in facts.items())
    prompt = DEBATE_PROMPT.format(SIDE=side.upper(), market=MARKETS[x["mkt"]]["name"], ACTION=action, facts=fl,
                                  room_key=room_key)
    res = None
    for attempt in range(2):
        text, prov, mdl = brains.ask(side, prompt, exclude=exclude)
        if not text:
            return res                      # None = absent; otherwise the first (invalid) answer stands
        j = parse_json(text) or {}
        raw = j.get("arguments")
        args = []
        for a in (raw if isinstance(raw, list) else [])[:4]:
            if isinstance(a, dict) and len(str(a.get("value", ""))) <= 40 and \
                    number_ok(str(a.get("fact")), a.get("value"), facts) and \
                    str(a["fact"]) not in {b["fact"] for b in args}:       # one fact = one argument
                name = str(a["fact"])                  # keep the sheet's own value, not the AI's spelling of it
                args.append({"fact": name, "value": facts[name], "text": clip(a.get("text", ""), 160)})
        g_ans = ngrams(json.dumps(j))
        copied = len(g_ans & ngrams(prompt)) / len(g_ans) if g_ans else 0.0
        vote = str(j.get("vote", "")).upper()
        res = {"side": side, "model": f"{BRAIN_NAME[prov]} {mdl}", "provider": prov, "args": args,
               "raw_vote": vote, "copied": round(copied, 2)}
        if copied > 0.30 or not args:
            res["invalid"] = ("copied the prompt" if copied > 0.30 else
                              "no valid quoted facts" if j else "the answer was not valid JSON")
            exclude = tuple(exclude) + (prov,)          # ask once more on another provider
            continue
        room_ok = facts[room_key] >= 2.0
        res["vote"] = action if (vote == action and room_ok and len(args) >= 2) else "WAIT"
        if vote == action and not room_ok:
            res["note"] = f"{room_key} is under 2.0 - vote set to WAIT"
        return res
    res["vote"] = "WAIT"
    return res


def debate_rebut(brains, x, side, own, other, facts):
    """One rebuttal round (candidate trades only). A valid rebuttal quotes a different, true fact."""
    if not other or not other.get("args"):
        return own
    lines = "\n".join(f"{i + 1}. {a['fact']} = {a['value']}: {a['text']}" for i, a in enumerate(other["args"]))
    fl = "\n".join(f"{k} = {v}" for k, v in facts.items())
    text, prov, mdl = brains.ask(side, REBUT_PROMPT.format(SIDE=side.upper(), market=MARKETS[x["mkt"]]["name"],
                                                           other=lines, facts=fl))
    j = parse_json(text) or {}
    knocked = set()
    raw = j.get("rebuttals")
    for rb in (raw if isinstance(raw, list) else [])[:4]:
        try:
            t = int(rb["target"]) - 1
            name = str(rb["fact"])
            if 0 <= t < len(other["args"]) and number_ok(name, rb["value"], facts) and \
                    name != other["args"][t]["fact"] and fact_against(other["side"], name, facts):
                knocked.add(t)
        except Exception:
            continue
    other["knocked"] = sorted(knocked)
    return own


def fact_against(side, name, facts):
    """True if this fact from the sheet argues against the bull (BUY) or the bear (SELL) side."""
    v = facts.get(name)
    bull = side == "bull"
    num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and finite(v) else None
    if name in ("trend_switch", "trend_4h", "daily_view"):
        return v == ("DOWN" if bull else "UP") or (name == "trend_switch" and v == "NONE")
    if name == "momentum_state":
        return v in (("strong_down", "pullback_down", "overstretched") if bull else
                     ("strong_up", "pullback_up", "overstretched"))
    if num is None:
        return False
    rules = {   # (against the bull, against the bear)
        "rsi": (num <= 45, num >= 55),
        "macd_hist": (num < 0, num > 0),
        "room_up_to_resistance_R": (num < 2.0, False),
        "room_down_to_support_R": (False, num < 2.0),
        "news_net_score": (num <= -3, num >= 3),
        "funding_24h_pct": (num >= 0.03, num <= -0.01),
        "oi_change_24h_pct": (False, False),
        "fear_greed": (num <= 25, num >= 75),
        "gld_holdings_5d_change_pct": (num <= -1, num >= 1),
        "big_funds_net_long_3y_rank_pct": (num >= 90, num <= 10),
        "front_minus_second_month_5d_change": (num < -0.02, num > 0.02),
    }
    return bool(rules.get(name, (False, False))[0 if bull else 1])


def debate_result(res, side, facts):
    if not res or res.get("invalid"):
        return "WAIT", 0
    standing = len(res["args"]) - len(res.get("knocked", []))
    room_key = "room_up_to_resistance_R" if side == "bull" else "room_down_to_support_R"
    action = "BUY" if side == "bull" else "SELL"
    vote = action if (res.get("vote") == action and standing >= 2 and facts[room_key] >= 2.0) else "WAIT"
    return vote, standing


INSPECT_PROMPT = """You are the inspector of a {market} trading desk. A {d} trade has passed all the team's rules.
You may only BLOCK it or let it PASS. You cannot change any vote.
Block only with 1 or 2 reason codes from this list, and only if the table clearly shows it:
OPP_LEVEL_NEAR - a strong opposite level is within 0.5 ATR of the entry
NEWS_60 - high-impact news against the trade in the last 60 minutes
STALE_DATA - some input data is old
VOTE_CONTRADICTS - a row's vote goes against its own numbers (give the row number)
TREND_DISAGREES - the trend filter does not agree with the trade
EVENT_SOON - a big scheduled event within 60 minutes
TABLE (order shuffled):
{table}
TRADE: entry {entry}, stop {stop} ({stop_atr:.2f}x ATR), target {target}; room to the next strong level: {room}
Each reason at most 40 words.
Reply with ONLY this JSON: {{"decision": "PASS or BLOCK", "codes": [], "row": null, "reason": "..."}}"""


def inspect(brains, x, d, finals, plan, room, ctx_checks):
    rows = [a for a in VOTERS if a in finals]
    rnd = random.Random(int(pd.Timestamp(x["now"]).timestamp()))
    rnd.shuffle(rows)
    lines = []
    for i, a in enumerate(rows):
        fa = finals[a]
        nums = ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in list(fa["data"].items())[:5])
        rule_txt = clip(fa["rule_id"].split(".", 1)[-1].replace("_", " "), 30)     # no agent names
        lines.append(f"row {i + 1} | rule: {rule_txt} | numbers: {nums} | vote: {fa['vote']}")
    mkt = x["mkt"]
    prompt = INSPECT_PROMPT.format(market=MARKETS[mkt]["name"], d=d, table="\n".join(lines), entry=pxm(plan["entry"], mkt),
                                   stop=pxm(plan["stop"], mkt), stop_atr=plan["dist"] / plan["atr"],
                                   target=pxm(plan["t"][1], mkt), room="none in the way" if room is None else f"{room:.2f}R")
    text, prov, mdl = brains.ask("inspector", prompt, temperature=0.1, max_tokens=500, reasoning="medium")
    j = parse_json(text)
    if not j:
        return {"decision": "PASS", "why": "no AI answer, so only the code checks ran", "model": None, "valid_codes": []}
    dec = str(j.get("decision", "PASS")).upper().strip()
    raw = j.get("codes")
    raw = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    codes = [str(c).upper().strip() for c in raw][:2]
    row = j.get("row")
    valid = []
    for c in codes:
        if c == "VOTE_CONTRADICTS":
            try:
                r = int(row)
                if 1 <= r <= len(rows) and ctx_checks["contradicts"](rows[r - 1]):     # rows are numbered from 1
                    valid.append(c)
            except Exception:
                pass
        elif c in ctx_checks and ctx_checks[c]():
            valid.append(c)
    out = {"decision": dec, "codes": codes, "valid_codes": valid, "why": clip(j.get("reason", ""), 200),
           "model": f"{BRAIN_NAME[prov]} {mdl}"}
    if dec == "BLOCK" and not valid:
        out["decision"] = "PASS"
        out["why"] = "block ignored - its reasons did not check out in code (" + ", ".join(codes or ["none"]) + ")"
    return out


# =============================================================== one market, one hour
def market_snapshot(mkt, h, now, st, shared, f=None):
    """Everything the agents need, worked out in code. h = closed hourly candles."""
    if h is None or len(h) < 400:
        return None, "not enough hourly candles"
    age = (pd.Timestamp(now) - (h.index[-1] + pd.Timedelta(hours=1))).total_seconds() / 60
    if age - MARKETS[mkt]["data_delay_min"] > CFG["DATA_MAX_AGE_MIN"]:    # Yahoo's own delay does not count
        return None, f"market closed or data old ({age:.0f} min)"
    if f is None or len(f) != len(h):
        f = build_features(h, mkt)
    m = MARKETS[mkt]
    last = f.iloc[-1]
    if not finite(last["atr"], last["c"]) or last["atr"] <= 0:
        return None, "prices are not moving (no normal hourly move to measure) - no data"
    x = {"mkt": mkt, "now": now, "f": f, "h": h, "c": float(last["c"]), "atr": float(last["atr"]),
         "trend": last["trend"], "t_ups": int(last["t_ups"]), "t_downs": int(last["t_downs"]), "trend4": last["trend4"],
         "rsi": float(last["rsi"]), "rsi_prev": float(f["rsi"].iloc[-2]), "hist": float(last["hist"]),
         "hist_prev": float(f["hist"].iloc[-2]), "rsi_last8": [float(v) for v in f["rsi"].iloc[-8:]],
         "close_24": float(f["c"].iloc[-25]) if len(f) > 25 else np.nan, "shock": bool(last["shock"]),
         "bb_up": float(last["bb_up"]), "bb_lo": float(last["bb_lo"]), "bb_w": float(last["bb_w"]),
         "data_age_min": age,
         "fill_after": (pd.Timestamp(now) - pd.Timedelta(minutes=CFG["FILL_LAG_MIN"])).isoformat()}
    x["lag_closes"] = [float(f["c"].iloc[-1 - l]) if len(f) > l else np.nan for l in m["lags"]]
    atr_prev = float(last["atr_prev"])
    bar_up = last["c"] > last["o"]
    x["burst"] = bool(last["range"] >= CFG["BURST_ATR"] * atr_prev and
                      ((x["trend"] == "UP" and bar_up) or (x["trend"] == "DOWN" and not bar_up)))
    prev = f.iloc[-2]
    x["funding_shock_prev"] = bool(mkt == "BTC" and prev["shock"] and f.index[-2].hour in (0, 8, 16))
    a500 = f["atr"].iloc[-500:]
    x["atr_q95"] = float(a500.quantile(0.95)) if len(a500.dropna()) >= 100 else np.nan
    w500 = f["bb_w"].iloc[-500:]
    x["bbw_q15"] = float(w500.quantile(0.15)) if len(w500.dropna()) >= 100 else np.nan
    x["atr_5ago"] = float(f["atr"].iloc[-6])
    x["hi24_prev"] = float(f["h"].iloc[-25:-1].max())
    x["lo24_prev"] = float(f["l"].iloc[-25:-1].min())
    v20 = f["v"].iloc[-21:-1].mean()
    x["vol_ratio"] = float(last["v"] / v20) if v20 and v20 > 0 else None
    ist = to_ist(f.index[-1] + pd.Timedelta(hours=1))
    x["btc_quiet_time"] = bool(ist.weekday() >= 5 or dt.time(7, 30) < ist.time() <= dt.time(12, 30))
    # divergence: new 24-bar high but RSI below its value at the previous high
    hh = f["h"].iloc[-25:-1]
    x["bear_div"] = bool(len(hh) == 24 and last["h"] >= hh.max() and last["rsi"] < f["rsi"].loc[hh.idxmax()])
    # daily and weekly candles (finished only)
    daily = daily_from_hourly(h, mkt)
    cur_key = day_keys(pd.DatetimeIndex([pd.Timestamp(now)]), mkt)[0]
    dfin = daily[daily.index < cur_key]
    x["daily_bars"] = dfin
    wk = week_keys(dfin.index)
    weekly_c = dfin["Close"].groupby(wk).last()
    weekly_h = dfin["High"].groupby(wk).max()
    weekly_l = dfin["Low"].groupby(wk).min()
    cur_week = week_keys(pd.DatetimeIndex([cur_key]))[0]
    wfin = weekly_c[weekly_c.index < cur_week]
    slow_n, fast_n, wk_n = m["daily"]
    x["daily_view"] = daily_view_state(dfin["Close"], wfin, slow_n, fast_n, wk_n)
    x["yesterday"] = (float(dfin["High"].iloc[-1]), float(dfin["Low"].iloc[-1])) if len(dfin) else None
    wh = weekly_h[weekly_h.index < cur_week]
    wl = weekly_l[weekly_l.index < cur_week]
    x["last_week"] = (float(wh.iloc[-1]), float(wl.iloc[-1])) if len(wh) else None
    d20 = dfin.iloc[-20:]
    x["gold_top_quarter"] = bool(len(d20) >= 10 and x["c"] >= d20["Low"].min() + 0.75 * (d20["High"].max() -
                                                                                           d20["Low"].min()))
    x["h4"] = resample_4h(h)
    # 4H swing checks (mood experiments)
    h4 = x["h4"]
    if len(h4) > 15:
        hi, lo = pivots(h4["High"].values[:-1], h4["Low"].values[:-1], 3)
        x["h4_close_above_swing"] = bool(hi and h4["Close"].iloc[-1] > hi[-1][1])
        x["h4_close_below_swing"] = bool(lo and h4["Close"].iloc[-1] < lo[-1][1])
    else:
        x["h4_close_above_swing"] = x["h4_close_below_swing"] = False
    # WTI session bias and time window
    end_ny = to_ny(f.index[-1] + pd.Timedelta(hours=1))
    x["wti_window"] = bool(dt.time(10, 0) <= end_ny.time() <= dt.time(14, 30) and end_ny.weekday() < 5)
    x["wti_bias"] = "FLAT"
    if mkt == "OIL":
        ends = pd.DatetimeIndex([to_ny(t + pd.Timedelta(hours=1)) for t in f.index[-60:]])
        today_ny = end_ny.date()
        c_tail = f["c"].iloc[-60:].values
        at10 = [j for j, e in enumerate(ends) if e.date() == today_ny and e.time() == dt.time(10, 0)]
        prev1430 = [j for j, e in enumerate(ends) if e.date() < today_ny and e.time() <= dt.time(14, 30)
                    and e.weekday() < 5]
        if at10 and prev1430 and end_ny.time() >= dt.time(10, 0):
            diff = c_tail[at10[-1]] - c_tail[prev1430[-1]]
            x["wti_bias"] = "UP" if diff > 0 else ("DOWN" if diff < 0 else "FLAT")
    # EIA spike fade (WTI): pre-report candle 09-10 NY, spike candle 10-11, next candle 11-12 closes back inside
    x["eia_fade"] = None
    if mkt == "OIL" and end_ny.weekday() == 2 and end_ny.time() == dt.time(12, 0) and len(f) >= 3:
        pre, spike, nxt = f.iloc[-3], f.iloc[-2], f.iloc[-1]
        if spike["h"] > pre["h"] and pre["l"] <= nxt["c"] <= pre["h"]:
            x["eia_fade"] = "SELL"
        elif spike["l"] < pre["l"] and pre["l"] <= nxt["c"] <= pre["h"]:
            x["eia_fade"] = "BUY"
    # gold Asian session and overnight move
    ist_now = to_ist(pd.Timestamp(now))
    x["gold_asia"] = bool(mkt == "GOLD" and dt.time(5, 30) <= ist_now.time() <= dt.time(13, 30))
    x["gold_overnight"] = float(x["c"] - dfin["Close"].iloc[-1]) if len(dfin) else 0.0
    ny_now = to_ny(pd.Timestamp(now))
    wd, tm = ny_now.weekday(), ny_now.time()
    x["btc_weekend_macro"] = bool((wd == 4 and tm >= dt.time(17, 0)) or wd == 5 or (wd == 6 and tm < dt.time(18, 0)))
    # price changes for the special agent
    def pchg(hours):
        past = f["c"][f.index <= f.index[-1] - pd.Timedelta(hours=hours)]
        return float((x["c"] / past.iloc[-1] - 1) * 100) if len(past) else None
    x["p_chg_1h"], x["p_chg_4h"], x["p_chg_24h"] = pchg(1), pchg(4), pchg(24)
    x["plan_buy"] = plan_at("BUY", x["c"], x["atr"], float(last["stop_lo"]), float(last["stop_hi"]), mkt)
    x["plan_sell"] = plan_at("SELL", x["c"], x["atr"], float(last["stop_lo"]), float(last["stop_hi"]), mkt)
    x["zones"] = build_zones(x)
    x["deriv"] = shared.get("deriv") if mkt == "BTC" else None
    x["flows"] = shared.get("flows")
    x["macro"] = shared.get("macro")
    x["fng"] = shared.get("fng") if mkt == "BTC" else None
    a24 = f["atr24"][f.index > f.index[-1] - pd.Timedelta(days=30)]
    x["chop_pass"] = bool(f["atr24"].iloc[-1] >= a24.median())          # test mode: ATR(24) vs its 30-day median
    x["power_trend"] = power_trend(h, x["h4"], dfin)
    return x, ""


def rule_answers(x):
    """All 12 agents' rule answers (code only). This is also the rules-only twin."""
    out = {}
    if x["shock"]:
        for a in AGENTS:
            out[a] = A(a, "WAIT", "shock_bar", "shock candle (range 2.5x ATR or more) - no agent votes", {})
        out["volatility"] = agent_volatility(x)
        out["daily_view"] = agent_daily_view(x)
        out["daily_view"]["vote"] = "WAIT"
        out["special"] = agent_special(x)
        out["special"]["vote"] = "WAIT"
        out["levels"] = agent_levels(x)
        out["levels"]["vote"] = "WAIT"
        out["momentum"] = agent_momentum(x)
        out["momentum"]["vote"] = "WAIT"
        x["news_score"] = None
        facts = fact_sheet(x, out)
        x["facts"] = facts
        return out
    out["trend"] = agent_trend(x)
    out["daily_view"] = agent_daily_view(x)
    out["momentum"] = agent_momentum(x)
    out["volatility"] = agent_volatility(x)
    out["levels"] = agent_levels(x)
    out["special"] = agent_special(x)
    out["macro"] = agent_macro(x)
    out["news"] = A("news", "WAIT", "news.code", "news needs the AI labels - WAIT in the rules-only twin", {})
    out["mood"] = agent_mood(x)
    x["news_score"] = None
    facts = fact_sheet(x, out)
    x["facts"] = facts
    out["bull"] = agent_debater("bull", x, facts)
    out["bear"] = agent_debater("bear", x, facts)
    d = {"UP": "BUY", "DOWN": "SELL"}.get(x["trend"])
    out["time_machine"] = time_machine(d, x) if d else A("time_machine", "WAIT", "tm.silent", "no trend", {})
    return out


def count_votes(votes, weights, trend):
    """votes: agent -> final answer. Votes against the trend count 0 for their side but still count as 'against'."""
    d = {"UP": "BUY", "DOWN": "SELL"}.get(trend)
    side = opp = 0.0
    answered = 0
    for a in AGENTS:
        v = votes.get(a)
        if not v:
            continue
        if not v.get("no_data"):
            answered += 1
        if a not in VOTERS:
            continue
        if d and v["vote"] == d:
            side += weights.get(a, 1.0)          # weighted (all 1.0 until 30 graded calls)
        elif d and v["vote"] == opposite(d):
            opp += 1                             # "1 or fewer opposite" counts agents, not weights
    return d, side, opp, answered


def gate_blocks(x, votes, st, mkt, events_active, d):
    """Hard blockers for direction d (code)."""
    blocks = []
    if events_active:
        blocks.append(("event", f"{events_active['label']} at {ist_str(events_active['time'], '%H:%M')} - no new trades "
                                f"{ist_str(events_active['start'], '%H:%M')}-{ist_str(events_active['end'], '%H:%M')}"))
    if x["shock"]:
        blocks.append(("shock", "shock candle - no agent votes this hour"))
    if x["burst"]:
        blocks.append(("burst", "burst candle in the trend direction - wait one candle"))
    if x["funding_shock_prev"]:
        blocks.append(("funding_shock", "shock candle at a funding time last hour - wait one candle"))
    until = st.misc["blocks_until"].get(mkt)
    if until and until > pd.Timestamp(x["now"]).isoformat():
        blocks.append(("cascade", f"liquidation cascade - no new entries until {ist_str(until, '%H:%M')}"))
    gate = votes["daily_view"].get("gate", "NEUTRAL") if votes.get("daily_view") else "NEUTRAL"
    if d == "BUY" and gate == "DOWN" or d == "SELL" and gate == "UP":
        blocks.append(("daily_view", f"daily view says {gate} - no {d} against it"))
    for a in ("volatility", "special", "time_machine", "macro"):
        v = votes.get(a)
        if v and d in (v.get("blocks") or []):
            blocks.append((v.get("block_code") or a, f"{PRETTY[a]}: {v['reason']}"))
    if x.get("calendar_block"):
        blocks.append(("calendar", x["calendar_block"]))
    return blocks


def decide(x, votes, weights, st, mkt, events_active):
    d, side, opp, answered = count_votes(votes, weights, x["trend"])
    res = {"d": d, "side": round(side, 2), "opp": round(opp, 2), "answered": answered, "blocks": [], "trade": False}
    if not d:
        res["why"] = "no trend - no trade"
        return res
    res["blocks"] = gate_blocks(x, votes, st, mkt, events_active, d)
    plan = x["plan_buy"] if d == "BUY" else x["plan_sell"]
    res["plan"] = plan
    if not plan.get("valid"):
        res["blocks"].append(("stop", plan.get("why", "no valid stop")))
    else:
        if plan["cost_r"] > CFG["MAX_COST_R"]:
            res["blocks"].append(("costs", f"costs are {plan['cost_r']:.2f}R (more than 0.2R) - skip"))
        room, z = room_check(x["zones"], d, plan["entry"], plan["dist"])
        res["room"] = room
        if room is not None and room < 1.0:
            res["blocks"].append(("room", f"strong zone at {pxm(z['mid'], mkt)} is only {room:.2f}R away (under 1R)"))
    if side < CFG["MIN_SIDE"]:
        res["why"] = f"only {side:g} {d} votes (need {CFG['MIN_SIDE']:g})"
    elif opp > CFG["MAX_OPPOSITE"]:
        res["why"] = f"{opp:g} votes against (max 1)"
    elif answered < CFG["MIN_ANSWERED"]:
        res["why"] = f"only {answered} of 12 agents answered (need 10)"
    elif res["blocks"]:
        res["why"] = "blocked: " + "; ".join(b[1] for b in res["blocks"])
    else:
        res["trade"] = True
        res["why"] = f"{side:g} {d} votes, {opp:g} against, {answered} of 12 answered"
    return res


# =============================================================== books: trades, risk officer, management
def risk_check(book, mkt, d, now, book_name, half=False):
    """Risk officer (code). Returns (ok, reason, risk_pct). half=True when the daily view is NEUTRAL."""
    nowt = pd.Timestamp(now)
    day = to_ist(nowt).strftime("%Y-%m-%d")
    wk = iso_week(to_ist(nowt).tz_localize(None))
    month = to_ist(nowt).strftime("%Y-%m")
    if book["open"].get(mkt):
        return False, "a trade is already open in this market (one at a time)", 0
    closed = book["closed"]
    cap = CFG["CAPITAL"]

    def pnl(rows):
        return sum(r.get("pnl_rs", 0.0) for r in rows)
    today = [r for r in closed if r.get("exit_day") == day]
    if pnl(today) <= -0.02 * cap:
        return False, "daily loss limit (2%) reached - no new trades today", 0
    if pnl([r for r in closed if r.get("exit_week") == wk]) <= -0.05 * cap:
        return False, "weekly loss limit (5%) reached - paused for review", 0
    opened_today = [r for r in closed + [t for t in book["open"].values() if t] if r.get("open_day") == day]
    if len([r for r in opened_today if r["mkt"] == mkt]) >= 1:
        return False, "already 1 trade in this market today", 0
    if len(opened_today) >= 3:
        return False, "already 3 trades today", 0
    last_mkt = [r for r in closed if r["mkt"] == mkt]
    if last_mkt:
        lt = last_mkt[-1]
        if lt.get("why") in ("stop", "trail") and lt["r"] < 0 and lt["d"] == d and \
                nowt < pd.Timestamp(lt["exit_ts"]) + pd.Timedelta(hours=3):
            return False, "stopped out within the last 3 hours in this direction", 0
        if len(last_mkt) >= 2 and all(t.get("why") == "stop" for t in last_mkt[-2:]):
            pause = pd.Timedelta(hours=12) if mkt == "BTC" else None
            t_last = pd.Timestamp(last_mkt[-1]["exit_ts"])
            if (pause and nowt < t_last + pause) or (not pause and last_mkt[-1].get("exit_day") == day):
                return False, "two losses in a row in this market - paused", 0
    if len(closed) >= 3 and all(t["r"] < 0 for t in closed[-3:]) and closed[-1].get("exit_day") == day:
        return False, "three losses in a row - stop until tomorrow", 0
    open_risk = sum(t["risk_rs"] for t in book["open"].values() if t)
    graded = [r for r in closed if r.get("r") is not None]
    risk_pct = CFG["RISK_PCT_START"]
    if len(graded) >= 50 and np.mean([r["r"] for r in graded[-50:]]) > 0:
        risk_pct = CFG["RISK_PCT_MAX"]
    other = book["open"].get("GOLD" if mkt == "BTC" else ("BTC" if mkt == "GOLD" else None) or "")
    if other and other["d"] == d:                      # BTC and gold the same way = one bet: 0.5% each, and the
        risk_pct = min(risk_pct, 0.5, CFG["RISK_PCT_MAX"] - other.get("risk_pct", 0.5))   # pair never above 1 trade
        if risk_pct <= 0.05:
            return False, "BTC and gold the same way count as one bet - that bet is already at full size", 0
    if half:
        risk_pct = risk_pct / 2                        # daily view NEUTRAL: half size
    risk_pct = round(risk_pct, 3)
    if open_risk + risk_pct / 100 * cap > 0.02 * cap + 1e-9:
        return False, "open risk would be above 2% of capital", 0
    month_pnl = pnl([r for r in closed if r.get("exit_day", "")[:7] == month])
    if month_pnl <= -0.10 * cap and book_name == "system":
        return True, "monthly loss 10% - PAPER ONLY", risk_pct
    return True, "", risk_pct


def open_trade(book, mkt, d, plan, x, risk_pct, kind, now, extra=None):
    cap = CFG["CAPITAL"]
    t = {"mkt": mkt, "d": d, "signal_ts": pd.Timestamp(x["f"].index[-1]).isoformat(), "entry_plan": plan["entry"],
         "stop0": plan["stop"], "dist": plan["dist"], "t": plan["t"], "cost_r": plan["cost_r"], "kind": kind,
         "risk_pct": risk_pct, "risk_rs": round(risk_pct / 100 * cap, 2), "open_day": to_ist(now).strftime("%Y-%m-%d"),
         "status": "open", "events_sent": [], "trail_atr": MARKETS[mkt]["trail_atr"], "fill_after": x.get("fill_after")}
    t.update(extra or {})
    book["open"][mkt] = t
    return t


def manage(t, f, mkt):
    """Replay the trade from its fill: stop first; at 2R close half and move the stop to +1R; trail the rest
    (3x ATR22, gold 2.5x) below the highest high; close all at 4R; level/sweep trades close at 2R;
    24 candles time limit; gold and WTI close before Friday's close."""
    idx = f.index
    sig = pd.Timestamp(t["signal_ts"])
    pos = int(np.searchsorted(idx.values, np.datetime64(sig)))
    if pos >= len(f) or idx[pos] != sig:
        if len(f) and sig < idx[0]:
            return {"status": "skipped", "why": "its signal candle is no longer in the price history"}
        if pos < len(f):            # the signal candle vanished from the data: start from the next candle
            pos -= 1
        else:
            return {"status": "waiting"}
    start = fill_index(idx, pos, t.get("fill_after"))
    if start >= len(f):
        return {"status": "waiting"}
    O, H, L, C, A22 = (f[k].values for k in ("o", "h", "l", "c", "atr22"))
    d, dist = t["d"], t["dist"]
    sgn = 1 if d == "BUY" else -1
    fill = float(O[start])
    off = float(f["adj"].values[pos]) if "adj" in f.columns else 0.0     # WTI: later rolls shift old prices
    stop, t1, t2, t4 = t["stop0"] + off, t["t"][0] + off, t["t"][1] + off, t["t"][3] + off
    if (d == "BUY" and (fill <= stop or fill >= t2)) or (d == "SELL" and (fill >= stop or fill <= t2)):
        return {"status": "skipped", "why": "the next open jumped past the stop or Target 2", "fill": fill}
    half, realized, best = False, 0.0, fill
    events = []
    for j in range(start, min(len(f), start + CFG["MAX_BARS"])):
        hit_stop = (L[j] <= stop) if d == "BUY" else (H[j] >= stop)
        if hit_stop:
            part = 0.5 if half else 1.0
            gapped = (O[j] < stop) if d == "BUY" else (O[j] > stop)
            ex = float(O[j]) if gapped else stop          # a gap through the stop fills at the open
            realized += part * (ex - fill) * sgn / dist
            why = "trail" if half else "stop"
            return {"status": "closed", "fill": fill, "r": realized - t["cost_r"], "why": why, "exit": ex,
                    "exit_ts": idx[j].isoformat(), "events": events, "stop": stop}
        if not half and ((H[j] >= t2) if d == "BUY" else (L[j] <= t2)):
            if t["kind"] == "level":
                realized += (t2 - fill) * sgn / dist
                return {"status": "closed", "fill": fill, "r": realized - t["cost_r"], "why": "target", "exit": t2,
                        "exit_ts": idx[j].isoformat(), "events": events + [("T2", idx[j].isoformat(), t2)], "stop": stop}
            half = True
            realized += 0.5 * (t2 - fill) * sgn / dist
            stop = t1 if d == "BUY" else t1
            events.append(("T2", idx[j].isoformat(), t2))
        if half:
            if (H[j] >= t4) if d == "BUY" else (L[j] <= t4):
                realized += 0.5 * (t4 - fill) * sgn / dist
                return {"status": "closed", "fill": fill, "r": realized - t["cost_r"], "why": "target", "exit": t4,
                        "exit_ts": idx[j].isoformat(), "events": events + [("T4", idx[j].isoformat(), t4)], "stop": stop}
            best = max(best, H[j]) if d == "BUY" else min(best, L[j])
            if finite(A22[j]):
                trail = best - sgn * t["trail_atr"] * A22[j]
                stop = max(stop, trail) if d == "BUY" else min(stop, trail)
        end_ny = to_ny(idx[j] + pd.Timedelta(hours=1))
        if not MARKETS[mkt]["is_247"]:
            friday = end_ny.weekday() == 4 and end_ny.time() >= dt.time(16, 0)
            short_day = end_ny.strftime("%Y-%m-%d") in CME_SHORT_DAYS and end_ny.time() >= dt.time(12, 0)
            if friday or short_day:                     # scheduled closes only; a hole in the data is not a close
                part = 0.5 if half else 1.0
                realized += part * (C[j] - fill) * sgn / dist
                return {"status": "closed", "fill": fill, "r": realized - t["cost_r"],
                        "why": "friday" if friday else "holiday", "exit": float(C[j]),
                        "exit_ts": idx[j].isoformat(), "events": events, "stop": stop}
    if start + CFG["MAX_BARS"] <= len(f):
        j = start + CFG["MAX_BARS"] - 1
        part = 0.5 if half else 1.0
        realized += part * (C[j] - fill) * sgn / dist
        return {"status": "closed", "fill": fill, "r": realized - t["cost_r"], "why": "time", "exit": float(C[j]),
                "exit_ts": idx[j].isoformat(), "events": events, "stop": stop}
    return {"status": "open", "fill": fill, "stop": stop, "half": half, "events": events, "t1": t1,
            "open_r": realized + (0.5 if half else 1.0) * (C[-1] - fill) * sgn / dist}


def update_books(st, mkt, f, now, notify):
    for name, book in st.books.items():
        t = book["open"].get(mkt)
        if not t:
            continue
        res = manage(t, f, mkt)
        if res["status"] == "waiting":
            continue
        if res["status"] == "skipped":
            t.update({"status": "skipped", "why": res["why"], "fill": res.get("fill")})
            book["open"][mkt] = None
            if name == "system":
                notify(f"⏭️ {MARKETS[mkt]['name']} {t['d']}: skipped - {res['why']}.")
            continue
        t["fill"] = res.get("fill")
        t["stop_now"] = res.get("stop")
        for ev in res.get("events", []):
            if ev[0] not in t["events_sent"] and name == "system":
                t["events_sent"].append(ev[0])
                if ev[0] == "T2" and res["status"] == "open":
                    notify(f"🎯 {MARKETS[mkt]['name']} {t['d']}: Target 2 reached at {pxm(ev[2], mkt)}. Close half and "
                           f"move the stop to {pxm(res.get('t1', t['t'][0]), mkt)} (Target 1). Trail the rest.")
            elif ev[0] not in t["events_sent"]:
                t["events_sent"].append(ev[0])
        if res["status"] == "closed":
            t.update({"status": "closed", "r": float(res["r"]), "why": res["why"], "exit": res["exit"],
                      "exit_ts": res["exit_ts"]})
            ex = to_ist(res["exit_ts"])
            t["exit_day"] = ex.strftime("%Y-%m-%d")
            t["exit_week"] = iso_week(ex.tz_localize(None))
            t["pnl_rs"] = round(t["r"] * t["risk_rs"], 2)
            book["closed"].append(t)
            book["open"][mkt] = None
            if name == "system":
                icon = "✅" if t["r"] > 0.05 else ("❌" if t["r"] < -0.05 else "➖")
                why_word = {"stop": "stop loss", "trail": "trailing stop", "target": "target", "time": "24-hour limit",
                            "friday": "Friday close", "holiday": "holiday close"}.get(res["why"], res["why"])
                notify(f"{icon} {MARKETS[mkt]['name']} {t['d']} closed ({why_word}) at {pxm(res['exit'], mkt)}: "
                       f"{t['r']:+.2f}R (about Rs {t['pnl_rs']:+,.0f} at Rs {t['risk_rs']:,.0f} risk).")


# =============================================================== grading (report card)
def add_test(st, key, mkt, agent, kind, d, x, model="code", extra=None):
    """One graded test trade per agent at a time (no overlap)."""
    if d not in ("BUY", "SELL"):
        return
    if any(p["key"] == key for p in st.grading["pending"]):
        return
    plan = x["plan_buy"] if d == "BUY" else x["plan_sell"]
    if not plan.get("valid"):              # every vote is graded: no valid structure stop -> standard 1.5x ATR stop
        dist = CFG["STOP_MIN_ATR"] * x["atr"]
        if not finite(dist) or dist <= 0:
            return
        sgn = 1 if d == "BUY" else -1
        plan = {"stop": x["c"] - sgn * dist, "t": [x["c"] + sgn * k * dist for k in (1, 2, 3, 4)], "dist": dist,
                "cost_r": MARKETS[mkt]["cost_pct"] / 100.0 * x["c"] / dist}
        extra = dict(extra or {}, stop_kind="standard 1.5x ATR (no structure)")
    rec = {"key": key, "mkt": mkt, "agent": agent, "kind": kind, "d": d, "model": model,
           "signal_ts": pd.Timestamp(x["f"].index[-1]).isoformat(), "stop": plan["stop"], "target": plan["t"][1],
           "dist": plan["dist"], "cost_r": plan["cost_r"], "trend": x["trend"], "fill_after": x.get("fill_after")}
    rec.update(extra or {})
    st.grading["pending"].append(rec)


def grade_pending(st, mkt, f):
    O, H, L, C = (f[k].values for k in ("o", "h", "l", "c"))
    keep = []
    for p in st.grading["pending"]:
        if p["mkt"] != mkt:
            keep.append(p)
            continue
        sig = pd.Timestamp(p["signal_ts"])
        pos = int(np.searchsorted(f.index.values, np.datetime64(sig)))
        if pos >= len(f) or f.index[pos] != sig:
            if sig < f.index[0]:
                continue
            if pos >= len(f):
                keep.append(p)
                continue
            pos -= 1                            # the signal candle vanished from the data: fill after its time
        off = float(f["adj"].values[pos]) if "adj" in f.columns else 0.0
        start = fill_index(f.index, pos, p.get("fill_after"))
        out = sim_fixed(O, H, L, C, start, p["d"], p["stop"] + off, p["target"] + off, p["dist"], p["cost_r"])
        if out is None:
            keep.append(p)
            continue
        if out.get("skip"):
            continue
        g = dict(p, r=round(out["r"], 3), hit=bool(out["hit"]), why=out["why"])
        st.grading["graded"].append(g)
        if p["kind"] == "inspector":
            st.inspector["graded"].append(g)
    st.grading["pending"] = keep


def agent_stats(st, mkt, agent, kind="final", last=150):
    rows = [g for g in st.grading["graded"] if g["mkt"] == mkt and g["agent"] == agent and g["kind"] == kind][-last:]
    n = len(rows)
    hits = sum(1 for g in rows if g["hit"])
    lo, hi = wilson(hits, n)
    return {"n": n, "hits": hits, "rate": hits / n if n else None, "lo": lo, "hi": hi,
            "avg_r": float(np.mean([g["r"] for g in rows])) if n else None}


def current_weights(st, mkt, now):
    """1.0 until 30 graded calls; then blended rate / 33%, between 0.7 and 1.3, moving at most 0.1 a week.
    Agents that agree on 80%+ of BUY/SELL votes count as one family (combined weight 1.5 at most)."""
    wk = iso_week(now)
    store = st.misc["weights"].setdefault(mkt, {})
    w, active = {}, set()
    for a in VOTERS:
        s = agent_stats(st, mkt, a)
        if s["n"] >= 30:
            active.add(a)
        target = 1.0
        if s["n"] >= 30:
            target = float(np.clip(((s["hits"] + 10) / (s["n"] + 30)) / (1 / 3), 0.7, 1.3))
        old = store.get(a, {"w": 1.0, "week": wk})
        cur = old["w"]
        if old.get("week") != wk:
            cur = float(np.clip(target, cur - 0.1, cur + 0.1))
            store[a] = {"w": cur, "week": wk}
        elif a not in store:
            store[a] = {"w": cur, "week": wk}
        w[a] = cur
    log_rows = [r for r in st.grading["vote_log"] if r["mkt"] == mkt][-300:]
    parent = {a: a for a in VOTERS}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i, a in enumerate(VOTERS):
        for b in VOTERS[i + 1:]:
            if a not in active or b not in active:   # weights (and the family cap) start after 30 graded calls
                continue
            # look-alikes: each one copies the other's BUY/SELL votes 80%+ of the time. Checked BOTH ways, because
            # agents that only vote with the trend would otherwise all "agree" with the trend agent.
            va = [r["v"].get(b) == r["v"].get(a) for r in log_rows if r["v"].get(a) in ("BUY", "SELL")]
            vb = [r["v"].get(a) == r["v"].get(b) for r in log_rows if r["v"].get(b) in ("BUY", "SELL")]
            if len(va) >= 20 and len(vb) >= 20 and sum(va) / len(va) >= 0.8 and sum(vb) / len(vb) >= 0.8:
                parent[find(a)] = find(b)
    fam = {}
    for a in VOTERS:
        fam.setdefault(find(a), []).append(a)
    for members in fam.values():
        tot = sum(w[a] for a in members)
        if len(members) > 1 and tot > 1.5:
            for a in members:
                w[a] = w[a] * 1.5 / tot
    return w


# =============================================================== messages
VOTE_ICON = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪"}
MKT_ICON = {"BTC": "🟠", "GOLD": "🟡", "OIL": "🛢"}
SEC = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
LINE = "━━━━━━━━━━━━━━━━"
LAG_WORDS = ["1 day", "3 days", "1 week", "2 weeks"]          # the trend switch's 4 earlier closes
FACT_WORDS = {"price": "price", "atr": "normal hourly move", "trend_switch": "1-hour trend", "trend_4h": "4-hour trend",
              "daily_view": "daily view", "rsi": "RSI", "macd_hist": "MACD histogram",
              "room_up_to_resistance_R": "room up to resistance (R)", "room_down_to_support_R": "room down to support (R)",
              "momentum_state": "momentum", "funding_24h_pct": "funding 24h (%)", "oi_change_24h_pct": "open interest 24h (%)",
              "fear_greed": "Fear & Greed", "nearest_resistance": "nearest resistance", "nearest_support": "nearest support",
              "news_net_score": "news score", "news_up_stories": "good stories", "news_down_stories": "bad stories",
              "gld_holdings_5d_change_pct": "GLD holdings 5 days (%)",
              "big_funds_net_long_3y_rank_pct": "big funds rank (%)",
              "front_minus_second_month_5d_change": "oil month spread change"}


def esc(v):
    """Text for a Telegram HTML message."""
    return html.escape(str(v), quote=False)


def strip_html(text):
    return html.unescape(re.sub(r"</?(b|i|u|s|code|pre)>", "", text))


def cap1(s):
    s = str(s).strip()
    return s[:1].upper() + s[1:] if s else s


def dot(s):
    s = str(s).strip()
    return s if not s or s[-1] in ".!?)" else s + "."


def _num(v, d=1, sign=False):
    if not finite(v):
        return "-"
    return f"{float(v):+,.{d}f}" if sign else f"{float(v):,.{d}f}"


def explain_agent(a, fa, ra, x, mkt, debate=None):
    """Plain-English reason of one agent with its numbers, and 'wrong if' for a BUY/SELL.
    Returns (why, wrong_if). When the AI explained this hour (checked by code), its words are added."""
    m = MARKETS[mkt]
    c, atr_v = x["c"], x["atr"]
    data = fa.get("data") or {}
    rid = fa.get("rule_id", "")
    code_reason = dot(cap1(ra.get("reason", "")))
    vote = fa.get("vote", "WAIT")
    why, wrong = code_reason, None
    if rid == "shock_bar":
        return "Shock candle (range 2.5x the normal move or more) - no agent votes this hour.", None
    if a == "trend":
        lc = x.get("lag_closes") or []
        parts = ", ".join(f"{w} {pxm(v, mkt)}" for w, v in zip(LAG_WORDS, lc) if finite(v))
        why = (f"Price {pxm(c, mkt)} vs the closes {parts} ago: above {x['t_ups']}, below {x['t_downs']} "
               "(3 of 4 = a trend).")
        s = sorted(float(v) for v in lc if finite(v))
        if len(s) == 4:
            if x["trend"] == "UP":
                wrong = f"an hourly close below {pxm(s[2], mkt)} (the up-trend switches off)"
            elif x["trend"] == "DOWN":
                wrong = f"an hourly close above {pxm(s[1], mkt)} (the down-trend switches off)"
            else:
                why += f" Turns UP above {pxm(s[2], mkt)}, DOWN below {pxm(s[1], mkt)}."
        if rid == "trend.burst_wait":
            why += " The last candle was a burst (2x the normal move) in the trend direction - wait one candle."
    elif a == "daily_view":
        ans, dwhy, dd = x["daily_view"]
        slow, fast, wk = m["daily"]
        if not dd:
            why = "Not enough daily history yet - NEUTRAL (trades at half size)."
        else:
            phase = dwhy.split(" (")[0]
            fm = re.search(r"fast (up|down)", dwhy)
            dc, sv = dd["daily_close"], dd[f"close_{slow}_days_ago"]
            wc, wma = dd["weekly_close"], dd[f"weekly_avg_{wk}"]
            why = (f"{phase} phase: close {pxm(dc, mkt)} is {'above' if dc > sv else 'below'} {slow} days ago "
                   f"({pxm(sv, mkt)}); {fast}-day check {fm.group(1) if fm else '-'}; weekly close {pxm(wc, mkt)} "
                   f"{'above' if wc > wma else 'below'} its {wk}-week average {pxm(wma, mkt)}. "
                   + {"UP": "Only BUY trades allowed.", "DOWN": "Only SELL trades allowed.",
                      "NEUTRAL": "Both sides allowed, half size."}.get(ans, ""))
            if ans == "UP":
                wrong = f"a weekly close below {pxm(wma, mkt)}"
            elif ans == "DOWN":
                wrong = f"a weekly close above {pxm(wma, mkt)}"
    elif a == "momentum":
        r, rp, hst, hp = x["rsi"], x["rsi_prev"], x["hist"], x["hist_prev"]
        chg = (c / x["close_24"] - 1) * 100 if finite(x.get("close_24")) and x["close_24"] else None
        state = ra.get("reason", "").split("; RSI")[0]
        why = (f"RSI {_num(r)} (last hour {_num(rp)}), MACD histogram {fnum(hst, 2 if abs(hst) < 100 else 0)} "
               f"({'rising' if finite(hst, hp) and hst > hp else 'falling'}), price {_num(chg, 2, True)}% in 24 hours. "
               f"{dot(cap1(state))}")
        if ra.get("vote") == "BUY" and vote == "BUY":
            wrong = "RSI falls back below 50"
        elif ra.get("vote") == "SELL" and vote == "SELL":
            wrong = "RSI rises back above 50"
    elif a == "volatility":
        q95 = x.get("atr_q95")
        why = (f"Normal hourly move (ATR) {pxm(atr_v, mkt)} = {atr_v / c * 100:.2f}% of price"
               + (f" (danger level {pxm(q95, mkt)})" if finite(q95) else "") + f". {code_reason}")
        if rid == "vol.expanding_up":
            wrong = f"a close back below {pxm(x['hi24_prev'], mkt)} (the 24-hour high)"
        elif rid == "vol.expanding_down":
            wrong = f"a close back above {pxm(x['lo24_prev'], mkt)} (the 24-hour low)"
        elif rid == "vol.release":
            wrong = (f"a close back inside the bands (below {pxm(x['bb_up'], mkt)})" if vote == "BUY" else
                     f"a close back inside the bands (above {pxm(x['bb_lo'], mkt)})")
    elif a == "levels":
        strong = [z for z in x["zones"] if z["strong"]]
        sup = max((z for z in strong if z["hi"] < c), key=lambda z: z["hi"], default=None)
        res = min((z for z in strong if z["lo"] > c), key=lambda z: z["lo"], default=None)
        bits = []
        if sup:
            bits.append(f"support zone {pxm(sup['lo'], mkt)} - {pxm(sup['hi'], mkt)} ({sup['pts']:.1f} points, top "
                        f"{(c - sup['hi']) / atr_v:.1f}x ATR below)")
        if res:
            bits.append(f"resistance zone {pxm(res['lo'], mkt)} - {pxm(res['hi'], mkt)} ({res['pts']:.1f} points, "
                        f"bottom {(res['lo'] - c) / atr_v:.1f}x ATR above)")
        why = ("Strong levels (3+ points): " + "; ".join(bits) + ". " if bits else
               "No strong level (3+ points) near the price. ") + code_reason
        st_ = ra.get("struct")
        if ra.get("vote") in ("BUY", "SELL") and vote == ra.get("vote") and finite(st_):
            up = vote == "BUY"
            if ra.get("signal") == "sweep":
                wrong = f"price goes {'below' if up else 'above'} {pxm(st_, mkt)} (past the sweep's extreme)"
            elif ra.get("signal") == "eia_fade":
                wrong = f"price goes {'below' if up else 'above'} {pxm(st_, mkt)} (past the spike's extreme)"
            else:
                wrong = (f"an hourly close back {'below' if up else 'above'} {pxm(st_, mkt)} (back "
                         f"{'under' if up else 'over'} the whole zone - the {'breakout' if up else 'breakdown'} failed)")
    elif a == "special":
        sd = ra.get("data") or {}
        if mkt == "BTC":
            bits = []
            if finite(sd.get("funding_now_per_8h_pct")):
                extra = []
                if finite(sd.get("funding_24h_avg_per_8h_pct")):
                    extra.append(f"24h average {sd['funding_24h_avg_per_8h_pct']:.4f}%")
                if finite(sd.get("funding_30d_rank_pct")):
                    extra.append(f"higher than {sd['funding_30d_rank_pct']:.0f}% of the last 30 days")
                bits.append(f"funding {sd['funding_now_per_8h_pct']:.4f}% per 8 hours"
                            + (f" ({', '.join(extra)})" if extra else ""))
            oi = [f"{_num(sd.get(k), 1, True)}% in {w}" for k, w in (("oi_change_4h_pct", "4h"), ("oi_change_24h_pct", "24h"))
                  if finite(sd.get(k))]
            if oi:
                bits.append("open interest " + ", ".join(oi))
            if finite(sd.get("price_change_4h_pct")):
                bits.append(f"price {_num(sd['price_change_4h_pct'], 1, True)}% in 4h")
            why = (cap1("; ".join(bits)) + f" ({sd.get('venues', '?')} exchanges). " if bits else "") + code_reason
        else:
            bits = []
            if finite(sd.get("cot_net_long_3y_rank_pct")):
                bits.append(f"big funds' net buying is higher than {sd['cot_net_long_3y_rank_pct']:.0f}% of the last "
                            "3 years (COT)")
            if finite(sd.get("gld_holdings_5d_change_pct")):
                bits.append(f"GLD gold holdings {_num(sd['gld_holdings_5d_change_pct'], 2, True)}% in 5 days")
            if finite(sd.get("spread_5d_change")):
                bits.append(f"front month minus next month {_num(sd.get('front_minus_second_month'), 2, True)} "
                            f"({_num(sd['spread_5d_change'], 2, True)} in 5 days)")
            why = (cap1("; ".join(bits)) + ". " if bits else "") + code_reason
        if ra.get("blocks"):
            why += f" Blocks {' and '.join(ra['blocks'])} trades."
    elif a == "macro":
        mc = x.get("macro") or {}
        z = mc.get("z") or {}
        names = {"BTC": [("nasdaq", "Nasdaq"), ("dollar", "dollar")], "GOLD": [("dollar", "dollar"), ("yield10", "10-yr yield")],
                 "OIL": [("sp500", "S&P 500"), ("copper", "copper")]}[mkt]
        bits = [f"{w} {_num(z.get(k), 1, True)}" for k, w in names if finite(z.get(k))]
        if bits and rid not in ("mac.btc_weekend", "mac.stale"):
            why = ("24-hour moves (1.0 = a normal day's move): " + ", ".join(bits)
                   + (f"; VIX {mc['vix']:.1f}" if finite(mc.get("vix")) else "")
                   + ("; markets are nervous (risk-off)" if mc.get("risk_off") else "") + f". {code_reason}")
    elif a == "news":
        nd = fa.get("data") or {}
        if rid == "news.code" and fa.get("ai") == "no AI key":
            why = "No AI key added, so nobody reads the news - WAIT."
        elif rid == "news.code" and fa.get("ai") in ("no AI this hour", None):
            why = "Read by the AI only when a trade is possible (saves the free AI limit) - WAIT this hour."
        elif rid == "news.code":
            why = dot(cap1(fa.get("reason", "")))
        elif rid == "news.none":
            why = "No fresh stories in the last 90 minutes."
        elif fa.get("stories") is not None or "news_net_score" in nd:
            why = (f"Good-news score {_num(nd.get('news_up_score'))}, bad-news score {_num(nd.get('news_down_score'))}, "
                   f"net {_num(nd.get('news_net_score'), 1, True)} (a vote needs 3 or more). {dot(cap1(fa.get('reason', '')))}")
            for s in (fa.get("stories") or [])[:2]:
                lab = s.get("label") or {}
                arrow = {"UP": "⬆️", "DOWN": "⬇️"}.get(lab.get("direction"), "➖")
                why += f"\n   {arrow} {clip(s.get('title', ''), 95)}"
        else:
            why = dot(cap1(fa.get("reason", "")))
    elif a == "mood":
        md = ra.get("data") or {}
        if mkt != "BTC":
            why = f"There is no tested mood index for {m['name']} - always WAIT."
        elif finite(md.get("fear_greed")):
            why = (f"Fear & Greed {md['fear_greed']:.0f}/100" + (f" (30-day average {md['fear_greed_30d_avg']:.0f})"
                                                                  if finite(md.get("fear_greed_30d_avg")) else "")
                   + f". {code_reason}")
    elif a in ("bull", "bear"):
        side_word = "buy" if a == "bull" else "sell"
        dbt = (debate or {}).get(a)
        if rid.endswith(".ai") and dbt and dbt.get("args"):
            kn = set(dbt.get("knocked", []))
            args = [f"{FACT_WORDS.get(g['fact'], g['fact'])} = {fact_value(g['value'], g['fact'], mkt)}"
                    f"{' (knocked down)' if i in kn else ''}" for i, g in enumerate(dbt["args"])]
            why = (f"AI debater ({dbt.get('model', '')}) argued: " + "; ".join(args) + ". "
                   + ("Needs 2 standing arguments and 2R of room to vote." if vote == "WAIT" else ""))
        elif rid.endswith(".invalid"):
            why = dot(cap1(fa.get("reason", "")))
        else:
            rd = ra.get("data") or {}
            key = "room_up_to_resistance_R" if a == "bull" else "room_down_to_support_R"
            room = rd.get(key)
            room_txt = ("price is inside a strong level (no room yet)" if zone_around(x) else
                        "no strong level in the way" if finite(room) and room >= 9.9 else
                        f"{_num(room)}R room {'up' if a == 'bull' else 'down'} to the next strong level")
            why = (f"{rd.get('supporting_facts', 0)} of 5 {side_word} facts true (trends, daily view, RSI, MACD); "
                   f"{room_txt} (needs 2+ facts and 2R).")
    elif a == "time_machine":
        td = ra.get("data") or {}
        if td.get("cases", 0) >= 30 and finite(td.get("avg_R")):
            why = (f"{td['cases']} similar past setups: won {td['hit_rate_pct']:.0f}% (break-even "
                   f"{td['break_even_pct']:.0f}%), average {td['avg_R']:+.2f}R. Confirms at +0.15R, blocks below -0.10R.")
        if ra.get("blocks"):
            why += f" Blocks {' and '.join(ra['blocks'])}."
    model = fa.get("model", "code")
    if fa.get("ai") == "checked" and model != "code" and a not in ("news", "bull", "bear") and fa.get("reason"):
        why += f"\n   🤖 {clip(fa['reason'], 170)}"
    return why, wrong


def fact_value(v, name, mkt):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if name in ("price", "atr", "nearest_support", "nearest_resistance"):
        return pxm(v, mkt)
    return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def flag_words(fa, ra):
    f = fa.get("flag", "")
    if f == "own judgement":
        return f" (AI's own view - its rule said {ra.get('vote', 'WAIT')})"
    if f == "corrected":
        return " (AI said otherwise - corrected to the rule)"
    return ""


def plain_reasons(pre, dec, x, events_active, mkt):
    """Every reason the trade did not form, in plain words."""
    out = []
    gate = x["daily_view"][0]
    for p in pre:
        if p == "no trend":
            out.append("No trend: price is not above (or below) 3 of the 4 earlier closes.")
        elif p == "event window" and events_active:
            out.append(f"Big news event: {events_active['label']} at {ist_str(events_active['time'], '%a %H:%M')} - no new "
                       f"trades from {ist_str(events_active['start'], '%H:%M')} to {ist_str(events_active['end'], '%H:%M')}.")
        elif p == "extreme volatility":
            out.append("Price is moving too wildly (a shock candle or the top 5% of moves) - no new trades.")
        elif p.startswith("burst"):
            out.append("The last candle was unusually big - wait one candle.")
        elif p == "daily view against the trend":
            out.append(f"The daily view is {gate} but the hourly trend is {x['trend']} - no trades against the daily view.")
        elif p == "cascade cooldown":
            out.append("A liquidation cascade just happened - no new trades for 2 candles.")
        elif p.startswith("no valid stop"):
            out.append("No safe stop: " + dot(p.split(":", 1)[-1].strip()))
        elif p.startswith("costs "):
            out.append(f"Fees would be {p[6:]} of the risk (limit {CFG['MAX_COST_R']}R) - the stop is too close for "
                       "the fees.")
        elif p.startswith("strong zone only"):
            z_in = zone_around(x)
            if z_in:
                out.append(f"Price is sitting inside a strong level ({pxm(z_in['lo'], mkt)} - {pxm(z_in['hi'], mkt)}) - "
                           "no room for the target until it breaks out.")
            else:
                out.append(f"A strong level is only {p.split('only', 1)[1].strip().replace(' away', '')} away - not "
                           "enough room for the target.")
        else:
            out.append(dot(cap1(p)))
    if not pre and dec.get("d"):
        d = dec["d"]
        votes_ok = dec["side"] >= CFG["MIN_SIDE"] and dec["opp"] <= CFG["MAX_OPPOSITE"] and \
            dec["answered"] >= CFG["MIN_ANSWERED"]
        if not votes_ok:
            out.append(dot(cap1(dec.get("why", ""))))
        if dec.get("blocks"):
            if votes_ok:
                out.append(f"{d} passed the vote ({dec['side']:g} for, {dec['opp']:g} against), but:")
            for _, b in dec["blocks"]:
                out.append(dot(cap1(b)))
        if dec.get("why", "").startswith("risk officer"):
            out.append(dot(cap1(dec["why"])))
        if dec.get("why", "").startswith("inspector blocked"):
            out.append(dot(cap1(dec["why"])))
    return out or ["-"]


def zone_around(x):
    """The strong zone the price is sitting inside, or None."""
    for z in x["zones"]:
        if z["strong"] and z["lo"] <= x["c"] <= z["hi"]:
            return z
    return None


def room_lines(x, mkt):
    z_in = zone_around(x)
    if z_in:
        return [f"Price is inside a strong level ({pxm(z_in['lo'], mkt)} - {pxm(z_in['hi'], mkt)}, "
                f"{z_in['pts']:.1f} points) - no room either way until it breaks out."]
    out = []
    for d, word in (("BUY", "Up"), ("SELL", "Down")):
        p = x["plan_buy"] if d == "BUY" else x["plan_sell"]
        dist = p["dist"] if p.get("valid") else 1.5 * x["atr"]
        room, z = room_check(x["zones"], d, x["c"], dist)
        kind = "resistance" if d == "BUY" else "support"
        if room is None:
            out.append(f"{word}: no strong {kind} in the way")
        else:
            out.append(f"{word}: strong {kind} zone {pxm(z['lo'], mkt)} - {pxm(z['hi'], mkt)} starts {room:.1f}R away")
    return out


def plan_line(x, mkt):
    d = {"UP": "BUY", "DOWN": "SELL"}.get(x["trend"])
    if not d:
        return "No trade plan (no trend)."
    p = x["plan_buy"] if d == "BUY" else x["plan_sell"]
    if not p.get("valid"):
        return f"No {d} plan: {p.get('why', 'no valid stop')}."
    return (f"If a {d} formed now: entry {pxm(p['entry'], mkt)}, stop {pxm(p['stop'], mkt)} "
            f"({p['dist'] / p['atr']:.1f}x ATR), target {pxm(p['t'][1], mkt)} (2R), fees {p['cost_r']:.2f}R.")


def code_check_lines(finals, use_ai, pre):
    if not use_ai:
        if not pre:
            return ["No AI key added - all votes are the agents' code rules."]
        return ["AI not asked this hour (no trade was possible) - all votes are the agents' code rules, "
                "nothing to correct."]
    flags = {}
    for a in VOTERS:
        flags.setdefault(finals[a].get("flag", ""), []).append(PRETTY[a])
    asked = [a for a in VOTERS if a not in NO_AI]
    explained = [a for a in asked if finals[a].get("ai") in ("checked", "labels by AI, score by code")]
    if not explained:
        return ["The AI brains did not answer this hour - every vote is the agent's own code rule."]
    out = [f"AI answers checked by code: {len(explained)} of {len(asked)} agents (the others used their code rule)."]
    if flags.get("corrected"):
        out.append("Corrected (AI vote went against the agent's own rule): " + ", ".join(flags["corrected"]) + ".")
    if flags.get("AI rejected"):
        out.append("AI answer rejected (wrong numbers or format) - rule used: " + ", ".join(flags["AI rejected"]) + ".")
    if flags.get("own judgement"):
        out.append("AI's own view kept (rule was neutral): " + ", ".join(flags["own judgement"]) + ".")
    if flags.get("invalid"):
        out.append("Debate answer invalid - vote set to WAIT: " + ", ".join(flags["invalid"]) + ".")
    if len(out) == 1:
        out.append("Every AI vote matched its rule - nothing to correct.")
    return out


def inspector_lines(st, x, dec, insp_txt, events_up, mkt):
    out = []
    status = st.inspector["status"].get(mkt, "test")
    if insp_txt and insp_txt != "not needed":
        out.append(f"Decision: {insp_txt}")
    else:
        out.append("Not needed - it checks only a trade that passed every vote and rule.")
    out.append(f"Mode: {'LIVE on ' + MARKETS[mkt]['name'] + ' (can block)' if status == 'live' else 'TEST on ' + MARKETS[mkt]['name'] + ' (blocks are only recorded)'}.")
    age = x["data_age_min"] - MARKETS[mkt]["data_delay_min"]
    mc = x.get("macro") or {}
    macro_ok = bool(mc.get("fresh")) or (mkt == "BTC" and x.get("btc_weekend_macro"))
    soon = [e for e in events_up if 0 <= (e["time"] - pd.Timestamp(x["now"])).total_seconds() <= 3600]
    checks = [f"prices fresh {'✅' if age <= CFG['DATA_MAX_AGE_MIN'] else '❌'}",
              f"macro fresh {'✅' if macro_ok else '❌'}",
              f"no big event within 1 hour {'✅' if not soon else '❌'}"]
    out.append("Safety checks: " + " · ".join(checks))
    return out


def report_trade_lines(st, mkt, x, now):
    out = []
    t = st.books["system"]["open"].get(mkt)
    if t:
        fill = t.get("fill")
        sgn = 1 if t["d"] == "BUY" else -1
        r_now = (x["c"] - fill) * sgn / t["dist"] if finite(fill) and t.get("dist") else None
        out.append(f"Open {MARKETS[mkt]['name']} trade: {VOTE_ICON[t['d']]} {t['d']} from "
                   f"{pxm(fill if finite(fill) else t['entry_plan'], mkt)}, stop now "
                   f"{pxm(t.get('stop_now') or t['stop0'], mkt)}"
                   + (f", now {r_now:+.2f}R" if finite(r_now) else ", waiting for the fill")
                   + (" (half closed at T2)" if "T2" in (t.get("events_sent") or []) else ""))
    else:
        out.append(f"Open {MARKETS[mkt]['name']} trade: none")
    day = to_ist(now).strftime("%Y-%m-%d")
    closed = [r for r in st.books["system"]["closed"] if r.get("exit_day") == day]
    opened = [r for r in st.books["system"]["closed"] + [v for v in st.books["system"]["open"].values() if v]
              if r.get("open_day") == day]
    tot_r = sum(r.get("r", 0) for r in closed)
    tot_rs = sum(r.get("pnl_rs", 0) for r in closed)
    out.append(f"Today (India day), all markets: {len(opened)} opened, {len(closed)} closed, result {tot_r:+.2f}R "
               f"(Rs {tot_rs:+,.0f})")
    return out


def other_markets_line(st, mkt):
    last = st.misc.get("last_decision", {})
    bits = []
    for o in MARKETS:
        if o == mkt or o not in last:
            continue
        r = last[o]
        dec = str(r.get("decision", ""))
        word = {"WAIT": "no trade", "MARKET CLOSED": "market closed", "NO DATA": "no data"}.get(dec, dec)
        price = f" {pxm(r['price'], o)}" if finite(r.get("price")) else ""
        bits.append(f"{MKT_ICON[o]} {MARKETS[o]['name']}{price}: {word} ({ist_str(r['time'], '%H:%M')})")
    return " · ".join(bits)


def build_report(st, mkt, x, rules, finals, dec, pre, weights, events_active, events_up, insp_txt, final_trade,
                 use_ai, debate, now):
    """The full report of one market (Telegram HTML), in numbered sections."""
    m = MARKETS[mkt]
    end = pd.Timestamp(x["f"].index[-1]) + pd.Timedelta(hours=1)
    trend_d = {"UP": "BUY", "DOWN": "SELL"}.get(x["trend"])
    L = [f"{MKT_ICON[mkt]} <b>{esc(m['name'].upper())} - REPORT</b>",
         f"Price {esc(pxm(x['c'], mkt))} · candle closed {esc(ist_str(end, '%d %b %H:%M'))}"]
    other = other_markets_line(st, mkt)
    if other:
        L.append(f"🌐 Others: {esc(other)}")
    L.append(LINE)
    # 1 decision
    L.append(f"{SEC[0]} <b>DECISION</b>")
    if final_trade:
        L.append(f"✅ <b>TRADE: {dec['d']}</b> - the trade card was sent just before this report.")
    else:
        L.append("⛔ <b>NO TRADE</b>")
        for i, r in enumerate(plain_reasons(pre, dec, x, events_active, mkt), 1):
            L.append(f"{i}. {esc(r)}")
    # 2 market phase
    L += ["", f"{SEC[1]} <b>MARKET PHASE</b>"]
    ups_or = x["t_ups"] if x["trend"] != "DOWN" else x["t_downs"]
    dv = x["daily_view"]
    phase = dv[1].split(" (")[0] if dv[2] else "not enough history"
    vol_word = ("EXTREME" if rules["volatility"]["rule_id"] == "vol.extreme" else
                "squeezed (a move is coming)" if rules["volatility"]["rule_id"] == "vol.compressed" else "normal")
    L += [f"1. Trend (1 hour): <b>{x['trend']}</b>" + (f" ({ups_or} of 4 checks)" if x["trend"] != "NONE" else ""),
          f"2. Trend (4 hours): {x['trend4']}",
          f"3. Daily view: <b>{dv[0]}</b> ({esc(phase)})",
          f"4. Volatility: {vol_word} (normal hourly move {esc(pxm(x['atr'], mkt))} = {x['atr'] / x['c'] * 100:.2f}%)",
          f"5. Power Trend (your rule, test only): {x.get('power_trend', 'NONE')}"]
    # 3 room
    L += ["", f"{SEC[2]} <b>ROOM TO MOVE</b>"]
    L += [f"{i}. {esc(r)}" for i, r in enumerate(room_lines(x, mkt) + [plan_line(x, mkt)], 1)]
    L.append("<i>1R = the stop distance; a trade needs 1R+ of room (2R is best).</i>")
    # 4 votes
    cnt = {"BUY": 0, "SELL": 0, "WAIT": 0}
    for a in VOTERS:
        cnt[finals[a]["vote"]] = cnt.get(finals[a]["vote"], 0) + 1
    L += ["", f"{SEC[3]} <b>VOTES</b> (11 voting agents)"]
    L.append(f"🔴 SELL {cnt['SELL']} · 🟢 BUY {cnt['BUY']} · ⚪ WAIT {cnt['WAIT']}")
    if trend_d and any(abs(weights.get(a, 1.0) - 1.0) > 1e-9 for a in VOTERS):
        L.append(f"Weighted {trend_d} score: {dec['side']:g}")
    L.append(f"<i>Needs {CFG['MIN_SIDE']:g}+ on the trend side, max {CFG['MAX_OPPOSITE']} against, "
             f"{CFG['MIN_ANSWERED']} of 12 answering. Daily view = gate, not a vote.</i>")
    # 5 agents
    L += ["", f"{SEC[4]} <b>WHAT EACH AGENT SAID</b>"]
    for i, a in enumerate(AGENTS, 1):
        fa, ra = finals[a], rules[a]
        if a == "daily_view":
            head = f"<b>{i}. Daily view</b> (gate) - {fa.get('gate', x['daily_view'][0])}"
        else:
            w = weights.get(a, 1.0)
            head = (f"<b>{i}. {esc(PRETTY[a])}</b> - {VOTE_ICON.get(fa['vote'], '')} {fa['vote']}"
                    + esc(flag_words(fa, ra)) + (f" · weight {w:.2f}" if abs(w - 1.0) > 1e-9 else ""))
        why, wrong = explain_agent(a, fa, ra, x, mkt, debate)
        L.append(head)
        L.append(esc(why))
        if wrong and a == "daily_view":
            L.append(f"❗ Changes if: {esc(wrong)}")
        elif wrong and fa["vote"] in ("BUY", "SELL"):
            L.append(f"❗ Wrong if: {esc(wrong)}")
        L.append("")
    L.pop()
    # 6 code check
    L += ["", f"{SEC[5]} <b>CODE CHECK</b>"]
    L += [f"{i}. {esc(s)}" for i, s in enumerate(code_check_lines(finals, use_ai, pre), 1)]
    # 7 inspector
    L += ["", f"{SEC[6]} <b>AI INSPECTOR</b>"]
    L += [f"{i}. {esc(s)}" for i, s in enumerate(inspector_lines(st, x, dec, insp_txt, events_up, mkt), 1)]
    # 8 trades
    L += ["", f"{SEC[7]} <b>TRADES</b> (paper)"]
    L += [f"{i}. {esc(s)}" for i, s in enumerate(report_trade_lines(st, mkt, x, now), 1)]
    # 9 events
    L += ["", f"{SEC[8]} <b>NEXT 24 HOURS</b>"]
    if events_up:
        L += [f"{i}. 📅 {esc(e['label'])} - {esc(ist_str(e['time'], '%a %d %b %H:%M'))}" for i, e in
              enumerate(events_up[:4], 1)]
    else:
        L.append("No big scheduled event.")
    L += [LINE, "<i>Paper trades only. Not financial advice.</i>"]
    return "\n".join(L)


def trade_card(mkt, x, res, finals, plan, risk_pct, room, events_up, tm, insp, debate, paper, risk_note, rules=None):
    """The special card sent the moment a trade forms (Telegram HTML)."""
    m = MARKETS[mkt]
    d = res["d"]
    rules = rules or finals
    usdinr = x.get("usdinr", CFG["USDINR_DEFAULT"])
    risk_rs = risk_pct / 100 * CFG["CAPITAL"]
    tag = "PAPER" if paper else "SIGNAL"
    if m["paper_only"]:
        tag = "SIGNAL ONLY (paper) - account too small for MCX crude"
    nxt = pd.Timestamp(x["f"].index[-1]) + pd.Timedelta(hours=1)
    entry_how = ("use a LIMIT order at this price (Delta fee 0.02% instead of 0.05%); if it is not filled within "
                 "15 minutes, skip the trade" if mkt == "BTC" else "enter now; a limit order keeps costs low")
    risk_txt = f"Rs {risk_rs:,.0f} ({risk_pct:.2f}% of Rs {CFG['CAPITAL']:,.0f})"
    if mkt == "BTC":
        per = plan["dist"] * CFG["BTC_CONTRACT"] * usdinr
        n = int(risk_rs // per) if per > 0 else 0
        risk_txt += (f" = about {n} contract(s) of 0.001 BTC (Rs {per:,.0f} risk each)" if n >= 1 else
                     f" - too small for even 1 contract of 0.001 BTC (needs Rs {per:,.0f}); paper only")
    elif mkt == "GOLD":
        risk_txt += " - check your MCX lot size before trading"
    plan_txt = (f"Nothing at T1. At T2 close half and move the stop to T1. Trail the rest {m['trail_atr']}x ATR behind "
                "the best price. Close all at T4 or after 24 hours." if res.get("kind") != "level" else
                "Level trade - close everything at T2. Close after 24 hours if neither is hit.")
    L = ["🚨🚨 <b>NEW TRADE</b> 🚨🚨",
         f"{VOTE_ICON[d]} <b>{esc(m['name'].upper())} - {d}</b>  [{esc(tag)}]",
         f"Signal from the candle that closed at {esc(ist_str(nxt, '%d %b %H:%M'))}", LINE,
         f"<b>1. Entry:</b> about {esc(pxm(plan['entry'], mkt))} - {esc(entry_how)} (the paper result uses the open "
         f"of the {esc(ist_str(next_open_hour(mkt, x['fill_after']), '%H:%M'))} candle)",
         f"<b>2. Stop loss:</b> {esc(pxm(plan['stop'], mkt))} ({plan['dist'] / plan['atr']:.1f}x the normal hourly move)",
         "<b>3. Targets:</b> " + " · ".join(f"T{i + 1} {esc(pxm(v, mkt))}" for i, v in enumerate(plan["t"])),
         f"<b>4. Plan:</b> {esc(plan_txt)}",
         f"<b>5. Risk:</b> {esc(risk_txt)}",
         f"<b>6. Votes:</b> {res['side']:g} {d} · {res['opp']:g} against · {res['answered']} of 12 answered "
         f"(need {CFG['MIN_SIDE']:g}, at most {CFG['MAX_OPPOSITE']} against)",
         f"<b>7. Trend:</b> {x['trend']} ({x['t_ups'] if d == 'BUY' else x['t_downs']} of 4 checks) · Daily view: "
         f"{x['daily_view'][0]} · Time machine: {esc(tm)}",
         f"<b>8. Room</b> to the next strong level: {'none in the way' if room is None else f'{room:.1f}R'}"
         + (" (lower confidence: under 2R)" if room is not None and room < 2 else ""),
         f"<b>9. AI Inspector:</b> {esc(insp)}"]
    if events_up:
        L.append("<b>10. Events</b> in the next 24h: " + esc("; ".join(
            f"{e['label']} {ist_str(e['time'], '%a %H:%M')}" for e in events_up[:3])))
    if risk_note:
        L.append(f"⚠️ <b>Note:</b> {esc(risk_note)}")
    L += ["", "<b>WHY THE AGENTS SAY THIS</b>"]
    for i, a in enumerate(AGENTS, 1):
        fa = finals[a]
        why, _ = explain_agent(a, fa, rules.get(a, fa), x, mkt, debate)
        first = why.split("\n")[0]
        if fa.get("flag") == "own judgement" and fa.get("reason"):
            first = "AI: " + str(fa["reason"])          # its rule was neutral: the AI's own (checked) reason
        vote_txt = (f"gate {fa.get('gate', x['daily_view'][0])}" if a == "daily_view" else
                    f"{VOTE_ICON.get(fa['vote'], '')} {fa['vote']}")
        L.append(f"{i}. <b>{esc(PRETTY[a])}</b> {vote_txt}{esc(flag_words(fa, rules.get(a, fa)))} - "
                 f"{esc(clip(first, 230))}")
    L += [LINE, "<i>Paper trades only. Not financial advice.</i>"]
    return "\n".join(L)


def tlen(s):
    """Length as Telegram counts it (UTF-16 code units: an emoji counts 2)."""
    return len(s.encode("utf-16-le")) // 2


def split_message(text, limit=3900, measure=None):
    """Parts of at most "limit" units as Telegram counts them (it allows 4,096 UTF-16 units after the HTML tags are
    read; 'measure' gives that count). Cuts between paragraphs (blank lines) when it can, else between lines; a
    single very long line is cut by length."""
    measure = measure or tlen
    parts, cur = [], ""
    for block in text.split("\n\n"):
        piece = block if not cur else "\n\n" + block
        if measure(cur + piece) <= limit:
            cur += piece
            continue
        if cur:
            parts.append(cur)
            cur = ""
        if measure(block) <= limit:
            cur = block
            continue
        for line in block.split("\n"):             # one very big paragraph: cut between lines
            while tlen(line) > limit:
                if cur:
                    parts.append(cur)
                    cur = ""
                parts.append(line[:1700])           # 1,700 characters are at most 3,400 UTF-16 units
                line = line[1700:]
            piece = line if not cur else "\n" + line
            if measure(cur + piece) > limit:
                parts.append(cur)
                cur = line
            else:
                cur += piece
    if cur:
        parts.append(cur)
    parts = [p for p in parts if p.strip()]
    while len(parts) >= 2 and measure(parts[-1]) < 900 and "\n\n" in parts[-2]:
        head, tail = parts[-2].rsplit("\n\n", 1)      # no tiny last message: move whole paragraphs into it
        if not head.strip() or measure(tail + "\n\n" + parts[-1]) > limit:
            break
        parts[-2], parts[-1] = head, tail + "\n\n" + parts[-1]
    return parts


def tg_send(text, html_mode=False):
    """Send a message (cut into parts if long). html_mode: bold and italics with Telegram HTML; if Telegram cannot
    read the HTML, the part is sent again as plain text."""
    tok, chat = os.getenv("TG_TOKEN", ""), os.getenv("TG_CHAT", "")
    text = safe_str(text)
    if DRY_RUN or not (tok and chat):
        print("\n----- TELEGRAM -----\n" + (strip_html(text) if html_mode else text) + "\n--------------------\n")
        return not DRY_RUN and False
    parts = split_message(text, measure=(lambda t: tlen(strip_html(t))) if html_mode else tlen)
    ok = True
    for k, part in enumerate(parts):
        if k > 0:
            part = ("<i>(continued)</i>\n" if html_mode else "(continued)\n") + part
        body = {"chat_id": chat, "text": part, "disable_web_page_preview": "true"}
        if html_mode:
            body["parse_mode"] = "HTML"
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=body, timeout=30)
            if r.status_code == 400 and html_mode:          # HTML not accepted: send the same words as plain text
                log("Telegram HTML refused, sending plain text:", r.text[:200])
                body.pop("parse_mode", None)
                body["text"] = strip_html(part)
                r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=body, timeout=30)
            if r.status_code == 429:                        # too many messages: wait as Telegram asks, once
                try:
                    wait = float(r.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5.0
                time.sleep(min(wait, 30))
                r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=body, timeout=30)
            ok = ok and r.status_code == 200
            if r.status_code != 200:
                log("Telegram error", r.status_code, r.text[:200])
        except Exception as e:
            ok = False
            log("Telegram error", e)
        if len(parts) > 1:
            time.sleep(0.5)
    return ok


def tg_document(path, caption):
    tok, chat = os.getenv("TG_TOKEN", ""), os.getenv("TG_CHAT", "")
    if DRY_RUN or not (tok and chat):
        print(f"\n----- TELEGRAM FILE ----- {path} ({caption})\n")
        return False
    try:
        with open(path, "rb") as fh:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendDocument",
                              data={"chat_id": chat, "caption": caption[:1000]},
                              files={"document": (os.path.basename(path), fh, "application/pdf")}, timeout=60)
        return r.status_code == 200
    except Exception as e:
        log("Telegram file error", e)
        return False


# =============================================================== PDFs
def make_pdf(path, title, lines):
    from fpdf import FPDF
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, 12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for line in lines:
        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.multi_cell(0, 5.5, latin1(line[3:]), new_x="LMARGIN", new_y="NEXT")
            continue
        pdf.set_font("Courier", "", 7.4)
        pdf.multi_cell(0, 3.6, latin1(line) if line.strip() else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(path)
    return path


def trades_lines(st, day=None, since=None):
    rows = []
    for name in ("system",):
        for t in st.books[name]["closed"] + [v for v in st.books[name]["open"].values() if v]:
            od = t.get("open_day", "")
            if (day and (od == day or t.get("exit_day") == day)) or (since and od >= since):
                res = f"{t['r']:+.2f}R ({t.get('why')})" if t.get("r") is not None else "open"
                rows.append(f"  {ist_str(t['signal_ts'], '%d %b %H:%M')}  {t['mkt']:<5} {t['d']:<4} entry "
                            f"{pxm(t['entry_plan'], t['mkt'])} stop {pxm(t['stop0'], t['mkt'])}  risk Rs "
                            f"{t['risk_rs']:,.0f}  -> {res}")
    return rows or ["  none"]


def daily_pdf(st, day, now):
    recs = st.reports_for(day)
    lines = [f"All hourly reports for {day} (India time). Paper trades only. Not financial advice.",
             "## Trades opened or closed today (system)"] + trades_lines(st, day=day)
    for r in recs:
        mkt = r["mkt"]
        lines.append(f"## {r['time_ist']}  {MARKETS[mkt]['name']}  price {pxm(r.get('price'), mkt)}  ->  "
                     f"{r.get('decision', '-')}")
        if r.get("skip"):
            lines.append("  " + r["skip"])
            continue
        lines.append(f"  trend {r.get('trend')} | daily view {r.get('daily_view')} | votes {r.get('side')} for, "
                     f"{r.get('opp')} against, {r.get('answered')} answered | {r.get('why', '')}")
        for a in AGENTS:
            v = (r.get("agents") or {}).get(a)
            if v:
                flag = f" [{v['flag']}]" if v.get("flag") else ""
                lines.append(f"  {PRETTY[a]:<12} rule {v['rule']:<5} final {v['final']:<5}{flag} "
                             f"{clip(v.get('model', ''), 32):<32} {clip(v.get('reason', ''), 120)}")
        for b in r.get("blocks", []):
            lines.append(f"  blocker: {b}")
        if r.get("inspector") and r["inspector"] != "not needed":
            lines.append(f"  inspector: {r['inspector']}")
        if r.get("twin"):
            lines.append(f"  rules-only twin: {r['twin']}")
        if r.get("rolls"):
            lines.append("  WTI roll gaps removed: " + ", ".join(f"{x['time'][:10]} {x['jump']:+}" for x in r["rolls"]))
    path = os.path.join(PDF_DIR, f"daily_{day}.pdf")
    return make_pdf(path, f"AI Team {VERSION} - daily report {day}", lines)


def agent_status(s):
    """What you may change, by number of graded calls (from the research)."""
    if s["n"] < 30:
        return "learning"
    if s["n"] < 50:
        return "UNDER REVIEW" if (s["avg_r"] < -0.2 and s["rate"] < 0.30) else "ok"
    return "CHANGE / LOWER" if s["hi"] < 0.36 else "ok"


def avg_r(rows):
    return f"{np.mean(rows):+.2f}R" if len(rows) else "no data yet"


def pct(s):
    return f"{s['rate'] * 100:.0f}%" if s["n"] else "-"


def weekly_lines(st, now):
    L = []
    L.append("How to read: 'hit' = target (2x the stop) reached before the stop. Luck line = 33%. With costs, an agent "
             "must beat about 36-40%. 'range' = where the true hit rate probably is (95%). Change nothing before 30 "
             "graded calls; decide at about 100. Status: learning (<30 calls) / UNDER REVIEW (30-50 calls, average "
             "below -0.2R and hit rate below 30%) / CHANGE / LOWER (50+ calls and even the top of the range is below 36%).")
    since = (to_ist(now) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    cut7 = (pd.Timestamp(now) - pd.Timedelta(days=7)).isoformat()
    for mkt in MARKETS:
        L.append(f"## {MARKETS[mkt]['name']}")
        L.append(f"{'Agent':<13}{'calls':>6}{'hit':>5}{'rate':>6}{'range':>10}{'avg R':>7}{'rule avg R':>11}"
                 f"{'WAIT %':>8}{'missed R':>9}  status")
        vl = [r for r in st.grading["vote_log"] if r["mkt"] == mkt and r["ts"] >= cut7]
        for a in VOTERS:
            s = agent_stats(st, mkt, a)
            r = agent_stats(st, mkt, a, "rule")
            wm = [g["r"] for g in st.grading["graded"] if g["mkt"] == mkt and g["agent"] == a and g["kind"] == "wait"][-150:]
            rng = f"{s['lo'] * 100:.0f}-{s['hi'] * 100:.0f}%" if s["n"] else "-"
            wait_rate = f"{sum(1 for v in vl if v['v'].get(a) == 'WAIT') / len(vl) * 100:.0f}%" if vl else "-"
            L.append(f"{PRETTY[a]:<13}{s['n']:>6}{s['hits']:>5}{pct(s):>6}{rng:>10}"
                     f"{fnum(s['avg_r'], 2) if s['n'] else '-':>7}{fnum(r['avg_r'], 2) if r['n'] else '-':>11}"
                     f"{wait_rate:>8}{fnum(np.mean(wm), 2) if wm else '-':>9}  {agent_status(s)}")
        cf = agent_stats(st, mkt, "coin", "coin")
        L.append(f"{'Coin flip':<13}{cf['n']:>6}{cf['hits']:>5}{pct(cf):>6}{'':>10}"
                 f"{fnum(cf['avg_r'], 2) if cf['n'] else '-':>7}")
        combos = {}
        for g in st.grading["graded"]:
            if g["mkt"] == mkt and g["kind"] == "final" and g.get("model") not in (None, "", "code"):
                combos.setdefault((g["agent"], g["model"]), []).append(g)
        for (a, mdl), gs in sorted(combos.items()):
            if len(gs) >= 5:
                hits = sum(1 for g in gs if g["hit"])
                L.append(f"  agent + model: {PRETTY.get(a, a)} with {mdl}: {len(gs)} calls, {hits} hit, "
                         f"average {np.mean([g['r'] for g in gs]):+.2f}R")
        own = [g for g in st.grading["graded"] if g["mkt"] == mkt and g["kind"] == "own"]
        if own:
            L.append(f"AI 'own judgement' votes: {len(own)} graded, average {np.mean([g['r'] for g in own]):+.2f}R")
        opened = [t for t in st.books["system"]["closed"] + [v for v in st.books["system"]["open"].values() if v]
                  if t["mkt"] == mkt and t.get("open_day", "") >= since]
        L.append(f"Trades opened in the last 7 days: {len(opened)} (the research expects about 2-6 a week)")
        why = no_trade_reasons(st, mkt, now)
        if why:
            L.append("Why no trade, hours in the last 7 days: " + "; ".join(f"{k} {v}" for k, v in why))
        for name in ("system", "rules_twin", "trend_twin"):
            rows = [t for t in st.books[name]["closed"] if t["mkt"] == mkt and t.get("r") is not None]
            label = {"system": "AI team", "rules_twin": "rules-only twin", "trend_twin": "trend-only twin"}[name]
            if rows:
                rr = [t["r"] for t in rows]
                L.append(f"{label:<16}: {len(rows)} trades, {sum(1 for v in rr if v > 0)} in profit, "
                         f"average {np.mean(rr):+.2f}R, total {sum(rr):+.1f}R")
            else:
                L.append(f"{label:<16}: no closed trades yet")
        blocks = [g for g in st.inspector["graded"] if g["mkt"] == mkt]
        if blocks:
            val = sum(-g["r"] for g in blocks)
            won = sum(1 for g in blocks if g["hit"])
            L.append(f"Inspector ({st.inspector['status'].get(mkt)}): {len(blocks)} blocks graded, blocked trades won "
                     f"{won} ({won / len(blocks) * 100:.0f}%), block value {val:+.1f}R")
        else:
            L.append(f"Inspector ({st.inspector['status'].get(mkt)}): no graded blocks yet")
        for kind, label in (("special_block", "Special blocks"), ("tm_block", "Time machine blocks")):
            g = [r for r in st.grading["graded"] if r["mkt"] == mkt and r["kind"] == kind]
            if g:
                good = sum(1 for r in g if r["r"] < 0)
                L.append(f"{label}: {len(g)} graded, right {good} ({good / len(g) * 100:.0f}%; useful above 67%)")
        bl = {}
        for g in st.grading["graded"]:
            if g["mkt"] == mkt and g["kind"] == "blocked":
                bl.setdefault((g.get("codes") or ["?"])[0], []).append(g["r"])
        for code, rr in sorted(bl.items()):
            L.append(f"Signals blocked by '{code}': {len(rr)} graded, they would have made {np.mean(rr):+.2f}R on average "
                     f"({'the block helped' if np.mean(rr) < 0 else 'the block cost money'})")
        tr_rule = [g for g in st.grading["graded"] if g["mkt"] == mkt and g["agent"] == "trend" and g["kind"] == "rule"]
        pw = [g["r"] for g in st.grading["graded"] if g["mkt"] == mkt and g["kind"] == "power"]
        if tr_rule or pw:
            L.append(f"Test mode - Power Trend: {len(pw)} graded, average {avg_r(pw)} vs the 4-check trend switch "
                     f"{avg_r([g['r'] for g in tr_rule])} (keep the better one after 100+ signals)")
            on = [g["r"] for g in tr_rule if g.get("chop") is True]
            off = [g["r"] for g in tr_rule if g.get("chop") is False]
            L.append(f"Test mode - chop gate: trend signals it keeps {avg_r(on)} ({len(on)}), signals it would remove "
                     f"{avg_r(off)} ({len(off)}) - switch it on only if the removed ones are mostly losers")
        tw = [t for t in st.books["trend_twin"]["closed"] if t["mkt"] == mkt and t.get("r") is not None]
        with_d = [t["r"] for t in tw if t.get("daily") == ("UP" if t["d"] == "BUY" else "DOWN")]
        against = [t["r"] for t in tw if t.get("daily") == ("DOWN" if t["d"] == "BUY" else "UP")]
        if with_d or against:
            L.append(f"Daily view gate: trend-twin trades with it {avg_r(with_d)} ({len(with_d)}), against it "
                     f"{avg_r(against)} ({len(against)})")
    L.append("## AI errors in the last 7 days (busy = 503, quota = 429)")
    agg = {}
    cut = (pd.Timestamp(now) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for k, n in st.models["errors"].items():
        day, p, code = k.split("|")
        if day >= cut:
            agg[(p, code)] = agg.get((p, code), 0) + n
    if agg:
        for (p, code), n in sorted(agg.items()):
            L.append(f"{p:<11} {code:<8} {n}")
    else:
        L.append("none")
    L.append("## Trades in the last 7 days (system)")
    L += trades_lines(st, since=since)
    return L


REASON_GROUPS = [   # (words in the reason, short fixed name) - first match wins
    ("inspector", "inspector block"), ("no trend", "no trend"), ("event", "news event window"),
    ("US inflation", "news event window"), ("Fed ", "news event window"), ("extreme volatility", "extreme volatility"),
    ("burst", "burst candle"), ("shock", "shock candle"), ("daily view", "daily view against the trend"),
    ("cascade", "liquidation cascade"),
    ("Friday", "Friday close"), ("holiday", "holiday / early close"), ("no valid stop", "no valid stop"),
    ("stop would need", "no valid stop"), ("swing", "no valid stop"), ("costs", "costs too high"),
    ("strong zone", "strong level too close"), ("already open", "trade already open"),
    ("trade in this market today", "1 trade a day limit"), ("3 trades today", "3 trades a day limit"),
    ("loss", "loss limit / losing streak"), ("stopped out", "cooldown after a stop"), ("one bet", "BTC+gold one bet"),
    ("open risk", "open risk limit"), ("Time machine", "time machine block"), ("Special", "special block"),
    ("Macro", "macro block"), ("Volatility", "volatility block"), ("calendar", "holiday / early close"),
    ("votes against", "too many votes against"), ("answered", "too few answered"),
    ("votes", "not enough votes")]


def reason_group(text):
    for words, name in REASON_GROUPS:
        if words.lower() in str(text).lower():
            return name
    return clip(text, 30) or "other"


def no_trade_reasons(st, mkt, now):
    """Count the hours of the last 7 days by the FIRST reason there was no trade (a pre-gate comes before votes)."""
    cnt = {}
    for k in range(7):
        day = (to_ist(now) - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        for r in st.reports_for(day):
            if r.get("mkt") != mkt:
                continue
            if r.get("skip"):
                key = "market closed / no data"
            elif str(r.get("decision", "")).startswith("TRADE"):
                key = "trade"
            else:
                key = reason_group(r.get("first_reason") or r.get("why") or "")
            cnt[key] = cnt.get(key, 0) + 1
    return sorted(cnt.items(), key=lambda kv: -kv[1])[:7]


def weekly_pdf(st, now, label=None):
    wk = label or iso_week(to_ist(now).tz_localize(None))
    path = os.path.join(PDF_DIR, f"weekly_{wk}.pdf")
    return make_pdf(path, f"AI Team {VERSION} - weekly report card {wk}", weekly_lines(st, now))


# =============================================================== Google Sheet (optional)
def write_sheet(st):
    raw = os.getenv("GOOGLE_SA_JSON", "").strip()
    if not raw:
        return False, "no Google key added"
    if DRY_RUN:
        return True, "dry run"
    try:
        import gspread
        from gspread.exceptions import WorksheetNotFound
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = json.loads(base64.b64decode(raw).decode())
        gc = gspread.service_account_from_dict(info)
        sid = os.getenv("SHEET_ID", "").strip()
        sh = gc.open_by_key(sid) if sid else gc.open(os.getenv("SHEET_NAME", "").strip() or "AI Trading Team")
        rows = [["Signal time (IST)", "Market", "Side", "Entry plan", "Stop", "T1", "T2", "T3", "T4", "Bought at",
                 "Status", "Exit", "Result (x risk)", "Rs"]]
        for t in sorted(st.books["system"]["closed"] + [v for v in st.books["system"]["open"].values() if v],
                        key=lambda t: t["signal_ts"], reverse=True):
            rows.append([ist_str(t["signal_ts"], "%Y-%m-%d %H:%M"), t["mkt"], t["d"], round(t["entry_plan"], 4),
                         round(t["stop0"], 4)] + [round(v, 4) for v in t["t"]] +
                        [t.get("fill") or "", t.get("why") or t["status"], t.get("exit") or "",
                         round(t["r"], 2) if t.get("r") is not None else "", t.get("pnl_rs", "")])
        card = [[l] for l in weekly_lines(st, now_utc())]
        for name, data in (("AI Team v7 Trades", rows), ("AI Team v7 Report Card", card)):
            width = max(len(r) for r in data)
            data = [r + [""] * (width - len(r)) for r in data]
            try:
                ws = sh.worksheet(name)
            except WorksheetNotFound:
                ws = sh.add_worksheet(title=name, rows=len(data) + 50, cols=width + 2)
            ws.clear()
            if ws.row_count < len(data) + 5 or ws.col_count < width:
                ws.resize(rows=max(ws.row_count, len(data) + 50), cols=max(ws.col_count, width + 2))
            ws.update(values=data, range_name="A1")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {clip(str(e), 150)}"


def write_trades_csv(st):
    """Every trade the team took (open and closed) in state_v7/trades.csv - opens in Excel, and GitHub shows it as
    a table. Works without any Google key."""
    import csv
    head = ["Signal time (IST)", "Market", "Side", "Kind", "Entry plan", "Stop", "T1", "T2", "T3", "T4", "Filled at",
            "Status", "Exit", "Exit time (IST)", "Result (x risk)", "Risk Rs", "Result Rs"]
    rows = []
    for t in sorted(st.books["system"]["closed"] + [v for v in st.books["system"]["open"].values() if v],
                    key=lambda t: t["signal_ts"], reverse=True):
        dec = MARKETS[t["mkt"]]["decimals"]
        rows.append([ist_str(t["signal_ts"], "%Y-%m-%d %H:%M"), MARKETS[t["mkt"]]["name"], t["d"], t.get("kind", ""),
                     round(t["entry_plan"], dec + 1), round(t["stop0"], dec + 1)] +
                    [round(v, dec + 1) for v in t["t"]] +
                    [round(t["fill"], dec + 1) if finite(t.get("fill")) else "", t.get("why") or t["status"],
                     round(t["exit"], dec + 1) if finite(t.get("exit")) else "",
                     ist_str(t["exit_ts"], "%Y-%m-%d %H:%M") if t.get("exit_ts") else "",
                     round(t["r"], 2) if t.get("r") is not None else "", t.get("risk_rs", ""), t.get("pnl_rs", "")])
    try:
        with open(_p("trades.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)
        return True
    except Exception as e:
        log("trades.csv failed:", e)
        return False


# =============================================================== the hourly run
def get_hourly(mkt):
    h = fetch_yahoo(MARKETS[mkt]["ticker"], "729d", "1h")     # Yahoo keeps hourly bars for 730 days
    if (h is None or len(h) < 400) and mkt == "BTC":
        h = fetch_kraken_btc()
        note_source("btc_kraken_backup", h is not None)
    note_source("yahoo_" + mkt.lower(), h is not None and len(h) >= 400)
    return h


def get_shared(st, now, markets_live):
    shared = {}
    if "BTC" in markets_live:
        try:
            shared["deriv"] = fetch_deriv(st, now)
        except Exception as e:
            log("derivatives error", e)
            shared["deriv"] = None
        shared["fng"] = fetch_fng(st, now)
    if "GOLD" in markets_live or "OIL" in markets_live:
        try:
            shared["flows"] = fetch_flows(st, now)
        except Exception as e:
            log("flows error", e)
            shared["flows"] = None
    series = {}
    for k, tk in MACRO_TICKERS.items():
        h = fetch_yahoo(tk, "120d", "1h")
        series[k] = closed_any(h, now, MACRO_DELAY.get(k, 0)) if h is not None else None
        note_source("yahoo_" + k, series[k] is not None and len(series[k]) > 50)
    shared["macro"] = macro_from(series, now)
    fx = fetch_yahoo("INR=X", "5d", "1h")
    shared["usdinr"] = float(fx["Close"].iloc[-1]) if fx is not None and len(fx) else CFG["USDINR_DEFAULT"]
    shared["events"] = fetch_events(st, now)
    return shared


def remember_decision(st, rep):
    """The latest decision of each market (shown as 'other markets' in the reports)."""
    st.misc.setdefault("last_decision", {})[rep["mkt"]] = {
        "time": rep["time"], "decision": rep.get("decision", ""), "price": rep.get("price"),
        "why": clip(rep.get("first_reason") or rep.get("skip") or "", 80)}


def report_due(st, mkt, bar, now):
    """Full report every 2 hours (candles that close at an even UTC hour = 18:30, 20:30 ... IST), or every hour
    with TG_MODE=all. A report missed by a skipped run is sent at the next run."""
    if CFG["TG_MODE"] == "all":
        return True
    if CFG["TG_MODE"] != "2h":
        return False
    end = pd.Timestamp(bar) + pd.Timedelta(hours=1)
    last = (st.misc.get("last_report_sent") or {}).get(mkt)
    if not last:
        return True
    gap = end - pd.Timestamp(last)
    return (end.hour % 2 == 0 and gap >= pd.Timedelta(minutes=90)) or gap >= pd.Timedelta(minutes=150)


def run_market(mkt, h_raw, st, shared, brains, now, notify, report_rows):
    rep = {"time": pd.Timestamp(now).isoformat(), "time_ist": ist_str(now, "%H:%M"), "mkt": mkt}
    if h_raw is None or len(h_raw) == 0:
        rep.update({"skip": "no price data", "decision": "NO DATA"})
        st.add_report(rep)
        remember_decision(st, rep)
        return rep
    h_closed = closed_hourly(h_raw, now, MARKETS[mkt]["data_delay_min"])
    if mkt == "OIL" and h_closed is not None and len(h_closed):
        h_closed, rolls = wti_back_adjust(h_closed)          # contract rolls must not create fake levels
        rep["rolls"] = rolls[-3:]
    feats = None
    if h_closed is not None and len(h_closed) >= 30:
        feats = build_features(h_closed, mkt)
        grade_pending(st, mkt, feats)
        update_books(st, mkt, feats, now, notify)
    x, why = market_snapshot(mkt, h_closed, now, st, shared, feats)
    if x is None:
        rep.update({"skip": why, "decision": "MARKET CLOSED" if "closed" in why else "NO DATA"})
        st.add_report(rep)
        remember_decision(st, rep)
        return rep
    x["usdinr"] = shared.get("usdinr", CFG["USDINR_DEFAULT"])
    f = x["f"]
    bar = pd.Timestamp(f.index[-1]).isoformat()
    last_bar = st.misc.setdefault("last_bar", {})
    if last_bar.get(mkt) == bar:                 # a late or repeated run must not judge the same candle twice
        log(f"{mkt}: the candle of {bar} was already checked - nothing new")
        return {"mkt": mkt, "skip": "candle already checked", "decision": "ALREADY CHECKED"}
    x["calendar_block"] = calendar_block(mkt, x, now)
    rep.update({"price": x["c"], "trend": x["trend"], "daily_view": x["daily_view"][0], "atr": x["atr"]})
    rules = rule_answers(x)
    trend_plans = {"BUY": x["plan_buy"], "SELL": x["plan_sell"]}         # swing-point stops (trend-only twin)
    lv = rules["levels"]
    if lv.get("struct") is not None and lv["vote"] in ("BUY", "SELL"):     # level setup: stop behind its zone/wick
        key = "plan_buy" if lv["vote"] == "BUY" else "plan_sell"
        lvl_plan = plan_at(lv["vote"], x["c"], x["atr"], lv["struct"], lv["struct"], mkt)
        if lvl_plan.get("valid"):
            x[key] = dict(lvl_plan, structure="zone")
            if not x["shock"]:              # the bull/bear room (in R) must use the plan that would really trade
                facts = fact_sheet(x, rules)
                x["facts"] = facts
                rules["bull"] = agent_debater("bull", x, facts)
                rules["bear"] = agent_debater("bear", x, facts)
    events_active, events_up = event_status(shared.get("events", []), mkt, now)
    weights = current_weights(st, mkt, now)
    d = {"UP": "BUY", "DOWN": "SELL"}.get(x["trend"])
    half = x["daily_view"][0] == "NEUTRAL"
    # ---- an open trade running into a scheduled event: say so once
    t_open = st.books["system"]["open"].get(mkt)
    if t_open:
        for e in events_up:
            key = f"event {e['label']} {e['time'].isoformat()}"
            if e["start"] - pd.Timestamp(now) <= pd.Timedelta(hours=2) and key not in t_open["events_sent"]:
                t_open["events_sent"].append(key)
                notify(f"⚠️ {MARKETS[mkt]['name']} {t_open['d']} is still open and runs into {e['label']} at "
                       f"{ist_str(e['time'], '%a %H:%M')}. Consider closing it or moving the stop to the entry price "
                       f"before {ist_str(e['start'], '%H:%M')}.")
    # ---- rules-only twin and trend-only twin (paper, no AI)
    twin = decide(x, rules, {a: 1.0 for a in VOTERS}, st, mkt, events_active)
    rep["twin"] = ("TRADE " + twin["d"]) if twin["trade"] else twin.get("why", "")
    if twin["trade"]:
        ok, _, rp = risk_check(st.books["rules_twin"], mkt, twin["d"], now, "rules_twin", half)
        if ok:
            kind = "level" if rules["levels"].get("signal") in LEVEL_SIGNALS and \
                rules["levels"]["vote"] == twin["d"] else "trend"
            open_trade(st.books["rules_twin"], mkt, twin["d"], twin["plan"], x, rp, kind, now)
    if d:
        p = trend_plans[d]
        if p.get("valid") and p["cost_r"] <= CFG["MAX_COST_R"]:
            ok, _, rp = risk_check(st.books["trend_twin"], mkt, d, now, "trend_twin")
            if ok:
                open_trade(st.books["trend_twin"], mkt, d, p, x, rp, "trend", now, {"daily": x["daily_view"][0]})
    # ---- pre-gates: no AI calls if no trade is possible this hour
    pre = []
    if not d:
        pre.append("no trend")
    if events_active:
        pre.append("event window")
    if x["shock"] or rules["volatility"]["rule_id"] == "vol.extreme":
        pre.append("extreme volatility")
    if x["burst"] or x["funding_shock_prev"]:
        pre.append("burst / funding shock - wait one candle")
    if d and ((d == "BUY" and x["daily_view"][0] == "DOWN") or (d == "SELL" and x["daily_view"][0] == "UP")):
        pre.append("daily view against the trend")
    until = st.misc["blocks_until"].get(mkt)
    if until and until > pd.Timestamp(now).isoformat():
        pre.append("cascade cooldown")
    if x.get("calendar_block"):
        pre.append(x["calendar_block"])
    if d:                                   # no trade can pass without a valid stop and room: save the AI quota
        p = x["plan_buy"] if d == "BUY" else x["plan_sell"]
        if not p.get("valid"):
            pre.append("no valid stop: " + p.get("why", ""))
        elif p["cost_r"] > CFG["MAX_COST_R"]:
            pre.append(f"costs {p['cost_r']:.2f}R")
        else:
            room, _ = room_check(x["zones"], d, p["entry"], p["dist"])
            if room is not None and room < 1.0:
                pre.append(f"strong zone only {room:.2f}R away")
    ok_risk, risk_why, _ = risk_check(st.books["system"], mkt, d or "BUY", now, "system", half)
    if not ok_risk:
        pre.append(risk_why)
    use_ai = not pre and brains.any()
    finals, debate = {}, {}
    if use_ai:
        for a in AGENTS:
            if a in AI_NATIVE:
                continue
            finals[a] = ai_voice(brains, st, x, a, rules[a], now)
        # news: code collects and cleans, AI labels, code scores
        stories = fetch_news_items(mkt, now)
        seen = st.cache.setdefault("news_seen", {})
        labels, labels_model = label_news(brains, mkt, stories, seen)
        for s in stories:                            # remembered per market: BTC seeing a story first must not
            seen.setdefault(f"{mkt}|{s['key']}", pd.Timestamp(now).isoformat())     # make it "old" for oil
        if not stories:
            finals["news"] = A("news", "WAIT", "news.none", "no fresh stories in the last 90 minutes", {},
                               model="code", ai="no stories", flag="")
            x["news_score"] = 0.0
        elif labels_model:
            nv, nid, nwhy, ndata, nscore = news_score(stories, labels, f, x["atr"], now)
            finals["news"] = A("news", nv, nid, nwhy, ndata, model=labels_model, ai="labels by AI, score by code",
                               flag="", stories=[{"title": s["title"], "label": l} for s, l in zip(stories, labels) if l][:6])
            x["news_score"] = round(nscore, 2)
            x["news_items"] = [(s, l) for s, l in zip(stories, labels) if l]
        else:
            finals["news"] = dict(rules["news"], model="code", ai="AI not answering - code rule (WAIT) used",
                                  reason=f"{len(stories)} stories but no AI labels this hour - WAIT", flag="AI absent")
        facts = fact_sheet(x, rules)
        x["facts"] = facts
        bull = debate_open(brains, x, "bull", facts)
        bear = debate_open(brains, x, "bear", facts, exclude=((bull or {}).get("provider"),) if bull else ())
        debate = {"bull": bull, "bear": bear}
        for side, resd in (("bull", bull), ("bear", bear)):
            if resd and not resd.get("invalid"):
                v, stand = debate_result(resd, side, facts)
                finals[side] = A(side, v, f"{side}.ai", f"{stand} arguments standing; " + "; ".join(
                    f"{a['fact']}={a['value']}: {a['text']}" for a in resd["args"][:3]),
                    {"standing": stand}, model=resd["model"], ai="checked", flag=resd.get("note", "") and "room rule")
            elif resd:                       # copied the prompt or quoted nothing true, twice: vote becomes WAIT
                finals[side] = A(side, "WAIT", f"{side}.invalid", f"answer INVALID ({resd['invalid']}) - vote set to "
                                 "WAIT", {}, model=resd.get("model", ""), ai="invalid", flag="invalid")
            else:
                fb = dict(rules[side])
                fb.update({"model": "code", "ai": "AI not answering - code rule used", "flag": "AI absent"})
                finals[side] = fb
    else:
        for a in AGENTS:
            finals[a] = dict(rules[a], model="code", ai="no AI this hour" if pre else "no AI key", flag="")
    dec = decide(x, finals, weights, st, mkt, events_active)
    # rebuttal round only for a candidate trade
    if use_ai and dec["trade"] and debate.get("bull") and debate.get("bear"):
        facts = x["facts"]
        debate_rebut(brains, x, "bull", debate["bull"], debate["bear"], facts)
        debate_rebut(brains, x, "bear", debate["bear"], debate["bull"], facts)
        for side in ("bull", "bear"):
            resd = debate[side]
            if resd and not resd.get("invalid") and finals[side].get("rule_id") == f"{side}.ai":
                v, stand = debate_result(resd, side, facts)
                finals[side]["vote"] = v
                finals[side]["reason"] = f"{stand} arguments still standing after the rebuttal; " + \
                    finals[side]["reason"].split("; ", 1)[-1]
        dec = decide(x, finals, weights, st, mkt, events_active)
    # cascade blocker memory (2 candles)
    sp = rules["special"]
    if sp.get("block_code") == "cascade":
        st.misc["blocks_until"][mkt] = (pd.Timestamp(now) + pd.Timedelta(hours=sp.get("block_bars", 2))).isoformat()
    # ---- grading slots (each one waits until its previous test trade has finished)
    for a in VOTERS:
        fa, ra = finals[a], rules[a]
        if fa["vote"] in ("BUY", "SELL"):
            gk = "own" if fa.get("flag") == "own judgement" else "final"      # own-judgement graded separately
            add_test(st, f"{mkt}|{a}|{gk}", mkt, a, gk, fa["vote"], x, fa.get("model", "code"),
                     {"flag": fa.get("flag", ""), "rule_id": fa.get("rule_id", "")})
        elif d and fa["vote"] == "WAIT":
            add_test(st, f"{mkt}|{a}|wait", mkt, a, "wait", d, x)
        if ra["vote"] in ("BUY", "SELL"):
            add_test(st, f"{mkt}|{a}|rule", mkt, a, "rule", ra["vote"], x,
                     extra={"rule_id": ra["rule_id"], "chop": x["chop_pass"]})
    pt = {"UP": "BUY", "DOWN": "SELL"}.get(x["power_trend"])
    if pt:                                   # test mode: Power Trend graded next to the 4-check trend switch
        add_test(st, f"{mkt}|power_trend|test", mkt, "power_trend", "power", pt, x)
    if d:
        rnd = random.Random(int(pd.Timestamp(f.index[-1]).timestamp()))
        add_test(st, f"{mkt}|coin|coin", mkt, "coin", "coin", rnd.choice(["BUY", "SELL"]), x)
        for a in ("special", "time_machine"):
            if d in (rules[a].get("blocks") or []):
                add_test(st, f"{mkt}|{a}|block", mkt, a, "special_block" if a == "special" else "tm_block", d, x)
        votes_ok = dec["side"] >= CFG["MIN_SIDE"] and dec["opp"] <= CFG["MAX_OPPOSITE"] and \
            dec["answered"] >= CFG["MIN_ANSWERED"]
        if votes_ok and dec.get("blocks"):         # what would the blocked signal have done?
            code = dec["blocks"][0][0]
            add_test(st, f"{mkt}|blocked|{code}", mkt, "blocked", "blocked", d, x,
                     extra={"codes": [b[0] for b in dec["blocks"]]})
    st.grading["vote_log"].append({"mkt": mkt, "ts": pd.Timestamp(f.index[-1]).isoformat(), "ai": use_ai,
                                   "v": {a: finals[a]["vote"] for a in VOTERS}})
    # ---- risk officer, inspector, desk head
    insp_txt, risk_note = "not needed", ""
    final_trade = False
    if dec["trade"]:
        ok, rwhy, rp = risk_check(st.books["system"], mkt, dec["d"], now, "system", half)
        if not ok:
            dec["why"] = "risk officer: " + rwhy
        else:
            risk_note = "; ".join(s for s in (rwhy, "daily view NEUTRAL - half size" if half else "") if s)
            plan = dec["plan"]
            status = st.inspector["status"].get(mkt, "test")
            checks = inspector_checks(x, dec, finals, rules, events_up, plan)
            ins = inspect(brains, x, dec["d"], finals, plan, dec.get("room"), checks) if brains.any() else \
                {"decision": "PASS", "why": "no AI key", "valid_codes": []}
            insp_txt = f"{ins['decision']} ({status} mode)" + (f" - {ins['why']}" if ins.get("why") else "")
            if ins["decision"] == "BLOCK":
                add_test(st, f"{mkt}|inspector|{pd.Timestamp(f.index[-1]).isoformat()}", mkt, "inspector",
                         "inspector", dec["d"], x, ins.get("model") or "", {"status": status,
                                                                             "codes": ins["valid_codes"]})
            if ins["decision"] == "BLOCK" and status == "live":
                dec["why"] = f"inspector blocked ({', '.join(ins['valid_codes'])}): {ins['why']}"
            else:
                final_trade = True
                kind = "level" if finals["levels"].get("signal") in LEVEL_SIGNALS and \
                    finals["levels"]["vote"] == dec["d"] else "trend"
                dec["kind"] = kind
                paper = (mkt not in CFG["REAL_MONEY"]) or "PAPER" in risk_note
                tm = finals["time_machine"]
                tmd = tm.get("data") or {}
                tm_txt = {"tm.confirm": "CONFIRM", "tm.block": "BLOCK"}.get(tm["rule_id"], "silent") + \
                    (f" ({tmd['avg_R']:+.2f}R, {tmd['cases']} cases)" if finite(tmd.get("avg_R")) and tmd.get("cases")
                     else f" (only {tmd['cases']} similar cases)" if tmd.get("cases") else "")
                open_trade(st.books["system"], mkt, dec["d"], plan, x, rp, kind, now,
                           {"votes": {a: finals[a]["vote"] for a in VOTERS}, "daily": x["daily_view"][0]})
                st.save()                                # the trade is saved before the card goes out
                note = mood_note(x, dec["d"])
                risk_note = "; ".join(s for s in (risk_note, note) if s)
                card = trade_card(mkt, x, dec, finals, plan, rp, dec.get("room"), events_up, tm_txt, insp_txt, debate,
                                  paper, risk_note, rules)
                sent = notify(card, html_mode=True)
                st.books["system"]["open"][mkt]["msg_failed"] = not sent and not DRY_RUN
    first = pre[0] if pre else (dec["blocks"][0][1] if dec.get("blocks") and not dec["why"].startswith("only")
                                else dec.get("why", ""))
    rep.update({"first_reason": first,
                "side": dec["side"], "opp": dec["opp"], "answered": dec["answered"], "why": dec.get("why", ""),
                "decision": ("TRADE " + dec["d"]) if final_trade else "WAIT", "inspector": insp_txt,
                "blocks": [b[1] for b in dec.get("blocks", [])] + (["pre-gate: " + ", ".join(pre)] if pre else []),
                "agents": {a: {"rule": rules[a]["vote"], "final": finals[a]["vote"], "model": finals[a].get("model", ""),
                               "flag": finals[a].get("flag", ""), "reason": finals[a]["reason"]} for a in AGENTS},
                "test_mode": {"chop_gate_pass": x["chop_pass"], "power_trend": x["power_trend"]}})
    st.add_report(rep)
    remember_decision(st, rep)
    try:                                     # the full report (sent every 2 hours; the latest is kept for "report")
        text = build_report(st, mkt, x, rules, finals, dec, pre, weights, events_active, events_up, insp_txt,
                            final_trade, use_ai, debate, now)
        st.cache.setdefault("last_report_text", {})[mkt] = {"time": pd.Timestamp(now).isoformat(), "text": text}
        if report_due(st, mkt, f.index[-1], now):
            rep = dict(rep, report_text=text)
    except Exception as e:                   # a report problem must never stop the trading part
        traceback.print_exc()
        log(f"{mkt}: report failed ({type(e).__name__}: {e})")
    report_rows.append(rep)
    last_bar[mkt] = bar                      # only now is this candle done (a crash above means it is judged again)
    return rep


def next_open_hour(mkt, t):
    """Start of the first hourly candle at or after t that the market really trades (CME: no 17:00 New York
    candle, closed Friday 17:00 to Sunday 18:00)."""
    h = pd.Timestamp(t).ceil("h")
    if MARKETS[mkt]["is_247"]:
        return h
    for _ in range(80):
        ny = to_ny(h)
        closed = ny.hour == 17 or ny.weekday() == 5 or (ny.weekday() == 6 and ny.hour < 18) or \
            (ny.weekday() == 4 and ny.hour >= 17)
        if not closed:
            return h
        h += pd.Timedelta(hours=1)
    return h


def calendar_block(mkt, x, now):
    """Gold and WTI: no new trade whose paper fill would come too close to Friday's close or a holiday break."""
    if MARKETS[mkt]["is_247"]:
        return None
    fill_t = next_open_hour(mkt, x["fill_after"])
    fill_ny = to_ny(fill_t)
    now_ny = to_ny(pd.Timestamp(now))
    if fill_t - pd.Timestamp(x["fill_after"]) > pd.Timedelta(hours=3):
        return "the next candle opens only after a market break (weekend or holiday) - the signal would be too old"
    if fill_ny.weekday() == 4 and fill_ny.time() >= dt.time(14, 0):
        return "too close to Friday's close (gold and WTI are closed before the weekend)"
    if now_ny.strftime("%Y-%m-%d") in CME_SHORT_DAYS and now_ny.time() >= dt.time(10, 0):
        return "US holiday or early-close day - no new gold/WTI trades after 10:00 New York"
    return None


def mood_note(x, d):
    """Gold and WTI have no mood index: lower confidence on trades into crowded fund positions or volatility spikes."""
    if MARKETS[x["mkt"]]["is_247"]:
        return ""
    notes = []
    cot = (x.get("flows") or {}).get("cot_" + x["mkt"]) or {}
    if finite(cot.get("rank")) and ((d == "BUY" and cot["rank"] >= 0.8) or (d == "SELL" and cot["rank"] <= 0.2)):
        notes.append("big funds already crowded on this side")
    a500 = x["f"]["atr"].iloc[-500:]
    if len(a500.dropna()) >= 100 and x["atr"] >= a500.quantile(0.8):
        notes.append("volatility is high")
    return ("lower confidence: " + " and ".join(notes)) if notes else ""


def inspector_checks(x, dec, finals, rules, events_up, plan):
    d = dec["d"]

    def opp_level():
        strong = [z for z in x["zones"] if z["strong"]]
        if d == "BUY":
            return any(0 <= z["lo"] - plan["entry"] <= 0.5 * x["atr"] for z in strong)
        return any(0 <= plan["entry"] - z["hi"] <= 0.5 * x["atr"] for z in strong)

    def news60():
        for s, lab in x.get("news_items", []):
            if lab["impact"] == "A" and s["age"] <= 60 and lab["direction"] == ("DOWN" if d == "BUY" else "UP"):
                return True
        return False

    def stale():
        mc = x.get("macro") or {}
        macro_old = bool(mc) and not mc.get("fresh", True) and not (x["mkt"] == "BTC" and x.get("btc_weekend_macro"))
        age = x["data_age_min"] - MARKETS[x["mkt"]]["data_delay_min"]
        return age > CFG["DATA_MAX_AGE_MIN"] or macro_old or finals["special"].get("no_data", False)

    def contradicts(agent):            # only a counted vote that is opposite to the agent's own rule
        f = finals.get(agent, {})
        rv = rules.get(agent, {}).get("vote")
        return rv in ("BUY", "SELL") and f.get("vote") == opposite(rv)

    def trend_dis():
        return {"UP": "BUY", "DOWN": "SELL"}.get(x["trend"]) != d

    def event_soon():
        return any(0 <= (e["time"] - pd.Timestamp(x["now"])).total_seconds() <= 3600 for e in events_up)
    return {"OPP_LEVEL_NEAR": opp_level, "NEWS_60": news60, "STALE_DATA": stale, "contradicts": contradicts,
            "TREND_DISAGREES": trend_dis, "EVENT_SOON": event_soon}


def update_inspector_status(st):
    """Test mode -> live after 20+ graded blocks with blocked trades winning under 30%;
    live -> test if blocked trades win 45% or more after 10+ blocks."""
    msgs = []
    changed = st.inspector.setdefault("changed", {})
    for mkt in MARKETS:
        since = changed.get(mkt, "")
        g = [r for r in st.inspector["graded"] if r["mkt"] == mkt and r["signal_ts"] >= since]
        won = sum(1 for r in g if r["hit"])           # a "win" = the blocked trade would have hit its target
        cur = st.inspector["status"].get(mkt, "test")
        if cur == "test" and len(g) >= 20 and won / len(g) < 0.30:
            st.inspector["status"][mkt] = "live"
            changed[mkt] = max(r["signal_ts"] for r in g)
            msgs.append(f"Inspector {mkt}: switched ON (blocked trades won only {won}/{len(g)})")
        elif cur == "live" and len(g) >= 10 and won / len(g) >= 0.45:
            st.inspector["status"][mkt] = "test"
            changed[mkt] = max(r["signal_ts"] for r in g)
            msgs.append(f"Inspector {mkt}: back to TEST mode (blocked trades won {won}/{len(g)})")
    return msgs


def send_pdfs(st, now):
    """Nightly PDF of the whole India day, sent at the first run after midnight IST (so the late runs of the day
    are inside it); weekly card on Sunday after 21:00 IST (or on Monday if Sunday's run was missed)."""
    ist_now = to_ist(now)
    target = (ist_now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last = st.misc.get("last_daily_pdf") or ""
    if last < target and st.reports_for(target):
        try:
            if tg_document(daily_pdf(st, target, now), f"AI Team {VERSION} - all hourly reports {target}") or DRY_RUN:
                st.misc["last_daily_pdf"] = target
        except Exception as e:
            log("daily PDF failed:", type(e).__name__, e)
    naive = ist_now.tz_localize(None)
    wk = iso_week(naive)
    prev_wk = iso_week(naive - pd.Timedelta(days=7))
    want = wk if (ist_now.weekday() == 6 and ist_now.hour >= 21) else (prev_wk if ist_now.weekday() == 0 else None)
    recent = any(st.reports_for((ist_now - pd.Timedelta(days=k)).strftime("%Y-%m-%d")) for k in range(1, 8))
    if want and recent and (st.misc.get("last_weekly_pdf") or "") < want:
        try:
            if tg_document(weekly_pdf(st, now, want), f"AI Team {VERSION} - weekly report card {want}") or DRY_RUN:
                st.misc["last_weekly_pdf"] = want
        except Exception as e:
            log("weekly PDF failed:", type(e).__name__, e)


def run(mode="run", markets=None):
    now = now_utc()
    st = State()
    todo = [m for m in MARKETS if not markets or m in markets]
    if mode == "test":
        return setup_test(st, now)
    if mode == "daily-pdf":
        day = to_ist(now).strftime("%Y-%m-%d")
        tg_document(daily_pdf(st, day, now), f"AI Team {VERSION} - all reports {day}")
        st.save()
        return
    if mode == "weekly-pdf":
        tg_document(weekly_pdf(st, now), f"AI Team {VERSION} - weekly report card")
        st.save()
        return
    if mode == "report":                       # the latest full report of each market, again, now
        saved = st.cache.get("last_report_text", {})
        sent_any = False
        for mkt in todo:
            item = saved.get(mkt)
            if item:
                tg_send(f"<i>(Latest report, from the check at {esc(ist_str(item['time'], '%d %b %H:%M'))})</i>\n"
                        + item["text"], html_mode=True)
                sent_any = True
        if not sent_any:
            tg_send(f"AI Team {VERSION}: no report saved yet - the first one comes with the next hourly run.")
        return
    msgs = []
    outbox = st.misc.setdefault("outbox", [])
    if outbox and os.getenv("TG_TOKEN") and os.getenv("TG_CHAT") and not DRY_RUN:     # resend what failed before
        waiting, outbox[:] = list(outbox), []
        for item in waiting:
            hm = bool(item.get("html"))
            head = f"(Delayed message from {ist_str(item['time'], '%d %b %H:%M')}, Telegram was down)\n"
            if not tg_send((esc(head) if hm else head) + item["text"], html_mode=hm):
                outbox.append(item)

    def notify(text, html_mode=False):
        """Trade cards and trade updates (kept and sent again later if Telegram fails)."""
        if CFG["TG_MODE"] not in ("trades", "all", "2h"):
            return True
        ok = tg_send(text, html_mode=html_mode)
        if not ok and os.getenv("TG_TOKEN") and os.getenv("TG_CHAT") and not DRY_RUN:
            outbox.append({"time": pd.Timestamp(now).isoformat(), "text": text, "html": html_mode})
            del outbox[:-20]                         # keep at most 20 waiting messages
        return ok
    brains = Brains(st, now)
    if brains.any():
        try:
            brains.discover()
        except Exception as e:
            log("model discovery failed", e)
    data = {m: get_hourly(m) for m in todo}
    live = [m for m, h in data.items() if h is not None]
    shared = get_shared(st, now, live)
    rows = []
    for mkt in todo:
        try:
            run_market(mkt, data[mkt], st, shared, brains, now, notify, rows)
        except Exception as e:
            traceback.print_exc()
            msgs.append(f"{mkt}: error {type(e).__name__}: {clip(str(e), 120)}")
        st.save()                                    # nothing is lost if a later step is cut short
    msgs += update_inspector_status(st)
    msgs += weekly_model_swaps(st, now) if to_ist(now).weekday() == 0 else []
    for r in rows:                                   # the full reports (every 2 hours), one message per market
        if r.get("report_text"):
            sent = tg_send(r["report_text"], html_mode=True)
            if sent or DRY_RUN:
                end = pd.Timestamp(st.misc["last_bar"][r["mkt"]]) + pd.Timedelta(hours=1)
                st.misc.setdefault("last_report_sent", {})[r["mkt"]] = end.isoformat()
    if msgs:
        notify(f"AI Team {VERSION} notes:\n" + "\n".join(msgs))
    send_pdfs(st, now)
    st.prune(now)
    st.save()
    write_trades_csv(st)
    ok, msg = write_sheet(st)
    if not ok and os.getenv("GOOGLE_SA_JSON"):
        log("Google Sheet:", msg)
    log(f"done - AI calls used: {brains.calls}; sources: " + ", ".join(f"{k}={v}" for k, v in sorted(SOURCES.items())))


def setup_test(st, now):
    lines = [f"AI Team {VERSION} - setup test ({ist_str(now)})"]
    tok, chat = os.getenv("TG_TOKEN", ""), os.getenv("TG_CHAT", "")
    lines.append("Telegram: working (you are reading this)" if tok and chat else "Telegram: NO key added")
    for mkt in MARKETS:
        h = get_hourly(mkt)
        hc = closed_hourly(h, now) if h is not None else None
        lines.append(f"Prices {MARKETS[mkt]['name']}: " + (f"ok, {len(hc)} hourly candles, last "
                                                           f"{ist_str(hc.index[-1] + pd.Timedelta(hours=1))}"
                                                           if hc is not None and len(hc) else "NOT working"))
        if hc is not None and len(hc) > 200:
            a = (atr(hc, 14) / hc["Close"] * 100).iloc[-24 * 90:].median()
            cost_r = MARKETS[mkt]["cost_pct"] / (1.5 * a) if a > 0 else float("nan")
            lines.append(f"   typical hourly move (median ATR, 90 days): {a:.2f}% of price; fees of "
                         f"{MARKETS[mkt]['cost_pct']:.2f}% cost about {cost_r:.2f}R with a 1.5x ATR stop ("
                         + ("ok" if cost_r <= CFG["MAX_COST_R"] else
                            f"above the {CFG['MAX_COST_R']}R limit - trades with tight stops are skipped") + ")")
    shared = get_shared(st, now, list(MARKETS))
    dv = shared.get("deriv") or {}
    lines.append(f"BTC derivatives (funding, open interest): {dv.get('venues_any', 0)} of 5 exchanges answered "
                 "(2 are enough)")
    lines.append("Fear & Greed: " + ("ok" if shared.get("fng") else "NOT working"))
    lines.append("Macro (dollar, yields, Nasdaq, S&P, VIX, copper): " + ("ok" if shared["macro"].get("fresh") else
                                                                         "stale or NOT working (normal at weekends)"))
    lines.append(f"Event calendar: {len(shared.get('events', []))} events (backups included)")
    fl = shared.get("flows") or {}
    flow_names = {"gld": "GLD gold holdings", "cot_GOLD": "big funds in gold (COT)", "cot_OIL": "big funds in oil (COT)",
                  "wti_spread": "oil month spread"}
    lines.append("Gold/Oil flows: " + (", ".join(v for k, v in flow_names.items() if k in fl) or "none"))
    items = fetch_news_items("BTC", now)
    lines.append(f"News feeds: {len(items)} fresh BTC stories right now")
    # ---- AI brains: one direct check per provider, with the reason and the fix when it fails
    b = Brains(st, now)
    if b.any():
        try:
            b.discover(force=True)
        except Exception as e:                     # an odd answer must not stop the test message
            log("model discovery failed", type(e).__name__, e)
    lines += ["", "AI BRAINS"]
    probe = {"groq": "trend", "openrouter": "macro", "gemini": "momentum"}
    works = {}
    others = ("groq", "openrouter", "gemini")
    for p in others:
        need = {"groq": "main brain for most agents", "openrouter": "backup brain",
                "gemini": "backup brain"}[p]
        if not b.key[p]:
            lines.append(f"{BRAIN_NAME[p]}: no key added ({need}) - add the GitHub secret {p.upper()}_API_KEY")
            works[p] = False
            continue
        for k in [k for k in st.models["skip_until"] if k.startswith(p + "|")]:
            st.models["skip_until"].pop(k, None)             # a test always tries for real
        text, prov, mdl = b.ask(probe[p], 'Reply with only this JSON: {"ok": true}', max_tokens=20,
                                exclude=tuple(o for o in others if o != p))
        works[p] = bool(text)
        if text:
            n = len(b.available.get(p) or [])
            lines.append(f"{BRAIN_NAME[p]}: working ({mdl}{f', {n} models listed' if n else ''})")
        else:
            err = b.last_err.get(p) or b.list_status.get(p) or ("?", "", "")
            if len(err) == 2:
                err = (err[0], "", err[1])
            lines.append(f"{BRAIN_NAME[p]}: NOT working - " + ai_error_help(p, err[0], err[2]))
    covered = [a for a in list(ROLE_MODELS) if any(works.get(p) for p, _ in ROLE_MODELS[a])]
    missing = [a for a in ROLE_MODELS if a not in covered]
    lines.append(f"Agents with a working brain: {len(covered)} of {len(ROLE_MODELS)}" +
                 (f" (none for: {', '.join(PRETTY.get(a, a.title()) for a in missing)})" if missing else ""))
    for role in ("news", "inspector"):
        text, prov, mdl = b.ask(role, 'Reply with only this JSON: {"ok": true}', max_tokens=20)
        lines.append(f"Brain for {role}: " + (f"working ({BRAIN_NAME[prov]} {mdl})" if text else
                                              "NOT answering (its agent uses the code rule until a brain works)"))
    lines += ["", "DATA SOURCES"]
    bad = {k: v for k, v in SOURCES.items() if v != "ok"}
    lines.append(f"{len(SOURCES) - len(bad)} of {len(SOURCES)} working")
    for k, v in sorted(bad.items()):
        lines.append(f"Not working: {source_help(k)} ({v})")
    ok, msg = write_sheet(st)
    lines += ["", "Google Sheet: " + ("working" if ok else msg + (
        " - trades are still saved in the file state_v7/trades.csv in GitHub (opens in Excel)"
        if "no Google key" in msg else ""))]
    lines.append(f"Telegram reports: {telegram_mode_text()}")
    text = "\n".join(lines)
    print(text)
    tg_send(text)
    st.save()


def ai_error_help(p, status, text):
    """Plain words for an AI error, with the fix."""
    low = str(text).lower()
    secret = f"{p.upper()}_API_KEY"
    site = {"groq": "console.groq.com/keys", "openrouter": "openrouter.ai/keys", "gemini": "aistudio.google.com/apikey"}[p]
    if status in (401, 403) and p != "gemini" or "invalid api key" in low or "api key not valid" in low or \
            "unauthorized" in low or "user not found" in low:
        return (f"the key was NOT accepted (error {status}). Make a new key at {site}, then in GitHub open "
                f"Settings > Secrets and variables > Actions, click {secret} > Update, paste the key (nothing else, "
                "no spaces) and save. Then run the test again.")
    if p == "openrouter" and ("data policy" in low or "no endpoints found" in low or "privacy" in low):
        return ("OpenRouter blocks free models until you allow them: open openrouter.ai/settings/privacy, switch ON "
                "the option for free models, save, and run the test again.")
    if p == "gemini" and status == 403:
        return "Google refused this key (403) - make a new key at aistudio.google.com/apikey and update GEMINI_API_KEY."
    if status == 429:
        if any(s in re.sub(r"[\s_-]", "", low) for s in ("perday", "daily", "rpd", "tpd", "quota")):
            return "the free daily limit is used up. It comes back by itself (Gemini at about 13:30 India time)."
        return "busy right now (too many requests). It tries again by itself."
    if status == 402:
        return "OpenRouter asks for credit for this model (402). Free models need no credit - this usually fixes itself."
    if status == 404:
        return "this model name is not available (404) - the program moves to another model by itself."
    if status in (0, "network"):
        return "could not connect (network or time-out)."
    if status in ("empty", 200):
        return "it answered with an empty message (busy, or a problem with this model) - the program tries other " \
               "models by itself."
    if status in ("unreadable", "not JSON"):
        return f"it sent an answer the program cannot read (perhaps a web page from a firewall): {clip(text, 80)}"
    if status == "length":
        return "the model used all its words on thinking - the program moves to the next model."
    if status in (500, 502, 503, 504):
        return f"the service had a problem (error {status}) - it tries again by itself."
    return f"error {status}: {clip(text, 100)}"


SOURCE_HELP = {
    "gld": "GLD gold holdings (optional - gold's flow agent also uses COT)",
    "news:gdelt": "GDELT news search (optional - one of several news feeds)",
    "cot_gold": "COT big-fund positions for gold (optional)", "cot_oil": "COT big-fund positions for oil (optional)",
    "wti_spread": "oil month spread (optional)", "calendar": "event calendar (backup dates are used)",
    "fear_greed": "Fear & Greed index (Bitcoin mood agent)", "hyperliquid": "Hyperliquid (1 of 5 BTC exchanges)",
    "okx_funding": "OKX funding (1 of 5 BTC exchanges)", "okx_oi": "OKX open interest (1 of 5 BTC exchanges)",
    "deribit": "Deribit (1 of 5 BTC exchanges)", "dydx": "dYdX (1 of 5 BTC exchanges)",
    "kraken_futures": "Kraken Futures (1 of 5 BTC exchanges)",
}


def source_help(name):
    if name in SOURCE_HELP:
        return SOURCE_HELP[name]
    if name.startswith("news:"):
        return f"news feed {name[5:]} (optional - one of several news feeds)"
    if name.startswith("yahoo_"):
        return f"Yahoo prices: {name[6:]}"
    return name


def telegram_mode_text():
    return {"2h": "a full report of each market every 2 hours, plus a special card when a trade forms",
            "all": "a full report of each market every hour, plus a special card when a trade forms",
            "trades": "only trade cards and trade updates (no regular reports)"}.get(
        CFG["TG_MODE"], f"unknown setting TG_MODE={CFG['TG_MODE']} - nothing is sent")


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "run").lower()
    if mode not in ("run", "test", "report", "daily-pdf", "weekly-pdf"):
        print(__doc__)
        sys.exit(2)
    arg = (sys.argv[2] if len(sys.argv) > 2 else os.getenv("MARKETS", "")).upper().replace(" ", "")
    markets = [m for m in arg.split(",") if m in MARKETS] or None      # e.g. "BTC" or "GOLD,OIL"; empty = all
    try:
        run(mode, markets)
    except Exception as e:
        traceback.print_exc()
        tg_send(f"AI Team {VERSION}: the {mode} run failed ({type(e).__name__}: {clip(str(e), 150)}). "
                "Please check GitHub Actions.")
        raise


if __name__ == "__main__":
    main()
