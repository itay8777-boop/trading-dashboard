"""
paper.py — מסחר בנייר: מריץ אסטרטגיה מובנית על נרות חיים ורושם כניסות ויציאות
בלי לשלוח שום פקודה ל-IB.

העיקרון: בכל סגירת נר מושכים את הנרות האחרונים (מהקאש + רענון מ-IB דרך
data.load_ohlcv), מריצים את אותו מנוע בקטסט בדיוק על החלון האחרון, ומשקפים את
העסקאות שהמנוע פתח מאז תחילת הסשן. כך מסחר הנייר זהה לחלוטין ללוגיקת הבקטסט
(אותו מילוי בפתיחת הבר הבא, אותם סטופ/יעד) — בלי מימוש כפול של כל אסטרטגיה.

עסקה שהמנוע "סגר" על הבר האחרון בסגירה שלו (סגירה כפויה בסוף הדאטה) נחשבת
פתוחה עד שנר נוסף מכריע. הרווח הלא-ממומש מחושב מהנר החי (live.get_live_bar)
כשיש כזה, אחרת מהסגירה האחרונה.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

from . import lab, live
from .data import load_full_history

_SESSIONS_PATH = Path(__file__).resolve().parent.parent / "paper_sessions.json"
_BAR_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
# כמה נרות סגורים מזינים את המנוע בכל בדיקה (חימום לאינדיקטורים כמו EMA200 / נקודות מפנה)
_WARMUP_BARS = {"1m": 4000, "5m": 3000, "15m": 2500, "30m": 2000, "1h": 2000, "4h": 1500, "1d": 1500, "1w": 800}
_MULT_1M = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
_MAX_LOG = 150

_lock = threading.RLock()
_sessions: dict[int, dict] = {}
_threads: dict[int, threading.Thread] = {}
_next_id = 1


# ------------------------------------------------------------------ אחסון
def _save() -> None:
    try:
        rows = []
        for s in _sessions.values():
            rows.append({k: v for k, v in s.items() if k != "_stop"})
        _SESSIONS_PATH.write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        pass


def _load() -> None:
    global _next_id
    try:
        if not _SESSIONS_PATH.exists():
            return
        for s in json.loads(_SESSIONS_PATH.read_text(encoding="utf-8")):
            if s.get("status") == "running":
                s["status"] = "stopped"           # שרת הופעל מחדש — הסשן לא ממשיך לבד
                s["log"] = (s.get("log") or []) + [f"{_now()} הסשן נעצר כי הדשבורד הופעל מחדש"]
            _sessions[int(s["id"])] = s
        if _sessions:
            _next_id = max(_sessions) + 1
    except Exception:
        pass


def _now() -> str:
    return pd.Timestamp.now().strftime("%d/%m %H:%M:%S")


def _log(s: dict, text: str) -> None:
    s.setdefault("log", []).append(f"{_now()} {text}")
    if len(s["log"]) > _MAX_LOG:
        s["log"] = s["log"][-_MAX_LOG:]


# ------------------------------------------------------------------ API
def start_session(*, key: str, symbol: str, timeframe: str, params: dict, common: dict) -> dict:
    """פותח סשן חדש ומריץ אותו בשרשור רקע. מחזיר את רשומת הסשן."""
    global _next_id
    symbol = (symbol or "").strip().upper()
    with _lock:
        sid = _next_id; _next_id += 1
        s = {
            "id": sid, "key": key, "title": lab.ENGINES[key]["title"], "symbol": symbol, "timeframe": timeframe,
            "params": {k: v for k, v in params.items() if k in {n for n, *_ in lab.ENGINES[key]["params"]}},
            "common": {k: v for k, v in common.items()},
            "status": "running", "started_at": _now(), "start_bar": None,
            "last_check": None, "last_bar": None, "bars": 0, "ib": None, "error": "",
            "trades": [], "log": [], "_stop": threading.Event(),
        }
        _sessions[sid] = s
        _log(s, f"סשן נפתח: {s['title']} · {symbol} {timeframe}")
        t = threading.Thread(target=_loop, args=(sid,), name=f"paper-{sid}", daemon=True)
        _threads[sid] = t
        t.start()
        _save()
        return s


def stop_session(sid: int) -> None:
    with _lock:
        s = _sessions.get(sid)
        if not s:
            return
        ev = s.get("_stop")
        if ev is not None:
            ev.set()
        if s["status"] == "running":
            s["status"] = "stopped"
            _log(s, "הסשן נעצר על ידי המשתמש")
        _save()


def delete_session(sid: int) -> None:
    stop_session(sid)
    with _lock:
        _sessions.pop(sid, None)
        _threads.pop(sid, None)
        _save()


def list_sessions() -> list[dict]:
    with _lock:
        return [snapshot(s) for s in sorted(_sessions.values(), key=lambda x: -x["id"])]


def snapshot(s: dict) -> dict:
    """עותק בטוח לקריאה מה-callback (בלי אובייקטי threading)."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in s.items() if k != "_stop"}


# ------------------------------------------------------------------ הלולאה
def _bar_floor(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    sec = _BAR_SECONDS.get(timeframe, 60)
    if sec >= 86400:
        return ts.normalize()
    return pd.Timestamp((int(ts.timestamp()) // sec) * sec, unit="s")


def _closed_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """מסיר את הנר שעדיין נבנה (חותמת הזמן שלו = תחילת הנר הנוכחי בשעון ניו-יורק)."""
    if df is None or df.empty:
        return df
    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    cur = _bar_floor(now_et, timeframe)
    return df[df.index < cur]


def _loop(sid: int) -> None:
    s = _sessions[sid]
    stop: threading.Event = s["_stop"]
    tf = s["timeframe"]
    poll = max(10, min(60, _BAR_SECONDS.get(tf, 60) // 3))
    first = True
    while not stop.is_set():
        try:
            _tick(s, first)
            first = False
        except Exception as exc:  # noqa: BLE001
            with _lock:
                s["error"] = str(exc)[:300]
                _log(s, f"שגיאה: {str(exc)[:160]}")
                _save()
            traceback.print_exc()
        stop.wait(poll)
    with _lock:
        if s["status"] == "running":
            s["status"] = "stopped"
        _save()


def _tick(s: dict, first: bool) -> None:
    tf, symbol = s["timeframe"], s["symbol"]
    # נרות חיים מהמנוי של הגרף (reqHistoricalData keepUpToDate) - בלי שאיבה היסטורית.
    live.subscribe(symbol, tf)
    live_df = live.get_recent_bars(symbol, tf)
    ib_ok = live.is_connected()
    if live_df is None or live_df.empty:
        with _lock:
            s["ib"] = ib_ok; s["last_check"] = _now()
            s["error"] = ("IB Gateway לא מחובר" if not ib_ok else
                          f"ממתין למנוי חי ל-{symbol} {tf} (ודא שהגרף למעלה מציג את הסימבול והטיים-פריים האלה)")
            if first:
                _log(s, s["error"])
            _save()
        return
    hist = load_full_history(symbol, tf, extended_hours=True)
    warm = _WARMUP_BARS.get(tf, 3000)
    if hist is not None and not hist.empty:
        hist = hist[hist.index < live_df.index[0]].tail(warm)
        df = pd.concat([hist, live_df])
    else:
        df = live_df
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = _closed_bars(df, tf)
    if df is None or len(df) < 60:
        with _lock:
            s["last_check"] = _now(); s["ib"] = ib_ok; s["error"] = "אין מספיק נרות סגורים"
            _save()
        return
    last_bar = df.index[-1]
    with _lock:
        if s["start_bar"] is None:
            # העסקאות נספרות מהנר הסגור הראשון *אחרי* פתיחת הסשן
            s["start_bar"] = str(last_bar)
            _log(s, f"נר סגור אחרון בפתיחה: {last_bar:%d/%m %H:%M} (שעון ניו-יורק). עסקאות ייספרו מכאן והלאה")
        start_bar = pd.Timestamp(s["start_bar"])
        unchanged = s["last_bar"] == str(last_bar)
        s["last_check"] = _now(); s["ib"] = ib_ok; s["error"] = ""; s["bars"] = int(len(df))
    if unchanged and not first:
        _refresh_unrealized(s, df)
        return
    stats = lab.run_engine(s["key"], df, s["common"], s["params"])
    tr = stats["_trades"]
    tr = tr[tr["EntryTime"] > start_bar] if len(tr) else tr
    last_close = float(df["Close"].iloc[-1])
    with _lock:
        known = {t["entry_time"]: t for t in s["trades"]}
        for _, r in tr.iterrows():
            et = str(pd.Timestamp(r["EntryTime"]))
            forced = (pd.Timestamp(r["ExitTime"]) == last_bar) and abs(float(r["ExitPrice"]) - last_close) < 1e-9
            rec = known.get(et)
            if rec is None:
                rec = {"id": len(s["trades"]) + 1, "side": "long" if r["Size"] > 0 else "short",
                       "entry_time": et, "entry_price": float(r["EntryPrice"]), "size": float(abs(r["Size"])),
                       "exit_time": None, "exit_price": None, "pnl": None, "status": "open", "unrealized": 0.0}
                s["trades"].append(rec); known[et] = rec
                _log(s, f"כניסה {'לונג' if rec['side']=='long' else 'שורט'} ב-{rec['entry_price']:,.2f} ({pd.Timestamp(et):%d/%m %H:%M})")
            if rec["status"] == "open" and not forced:
                rec["exit_time"] = str(pd.Timestamp(r["ExitTime"])); rec["exit_price"] = float(r["ExitPrice"])
                rec["pnl"] = float(r["PnL"]); rec["status"] = "closed"; rec["unrealized"] = 0.0
                _log(s, f"יציאה ב-{rec['exit_price']:,.2f} → {'+' if rec['pnl']>=0 else ''}{rec['pnl']:,.0f}$ ({pd.Timestamp(rec['exit_time']):%d/%m %H:%M})")
        s["last_bar"] = str(last_bar)
        _save()
    _refresh_unrealized(s, df)


def _refresh_unrealized(s: dict, df: pd.DataFrame) -> None:
    bar = live.get_live_bar(s["symbol"], s["timeframe"])
    price = float(bar["close"]) if bar else float(df["Close"].iloc[-1])
    mult = float(s["common"].get("multiplier", 1.0))
    with _lock:
        for t in s["trades"]:
            if t["status"] == "open":
                sign = 1.0 if t["side"] == "long" else -1.0
                t["unrealized"] = (price - t["entry_price"]) * sign * t["size"] * mult
        s["last_price"] = price


def session_summary(s: dict) -> dict:
    closed = [t for t in s["trades"] if t["status"] == "closed"]
    open_ = [t for t in s["trades"] if t["status"] == "open"]
    net = sum(t["pnl"] for t in closed)
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return {"closed": len(closed), "open": len(open_), "net": net,
            "win_rate": (wins / len(closed) * 100) if closed else 0.0,
            "unrealized": sum(t["unrealized"] for t in open_)}


_load()
