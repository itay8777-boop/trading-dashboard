"""עדכון לייב: reqHistoricalData(keepUpToDate=True) + מעקב אחר הבר הפתוח.

חוזרים לגישה הזו (במקום reqMktData + צבירת טיקים ידנית) אחרי ש-reqMktData נכשל
בפועל עם Error 10197 ("No market data during competing live session") - עוד
session על אותו חשבון IB (למשל TWS/Gateway עצמו עם גרף/Watchlist פתוח על אותו
סימבול) כבר מחזיק את קו הדאטה-החי היחיד המותר לחוזה הזה. reqHistoricalData
(keepUpToDate=True) לא מתנגש עם זה: זו לא "הרשמה לצילום שוק חי" (top-of-book),
אלא סטרימינג של ברים היסטוריים עם עדכון מתמשך של הבר האחרון - מנגנון נפרד ב-IB
שלא כפוף לאותה מגבלת session-אחד-לקו.

בתמורה: IB עצמו מחשב open/high/low/close/volume של הבר הפתוח (לא אנחנו, כמו
בגישת reqMktData/_BarAggregator שהייתה כאן), ודוחף עדכון רק כשהוא מחליט - בפועל
לא יותר מפעם בשנייה בערך, לא "כל טיק" ממש. זה עדיין מספיק לתחושת live אמיתית
בשילוב עם ה-Interval הצפוף בצד הלקוח (200 מ"ש, ראו app.py: LIVE_INTERVAL_MS) -
כל עדכון שמגיע מ-IB מוצג כמעט מיד, גם אם התדירות עצמה מוכתבת על ידי IB ולא
עלינו. גם קריאת ה-seed הנפרדת (שהייתה כאן) מיותרת עכשיו: הבקשה עם
keepUpToDate=True כבר מחזירה בעצמה את הברים ההיסטוריים ומיד ממשיכה לעדכן את
האחרון שבהם - שני השלבים (seed + סטרימינג) התאחדו לקריאה אחת.

הארכיטקטורה נשארה זהה: תת-שרשור אחד עם event loop משלו מחזיק חיבור IB קבוע.
מנוי reqHistoricalData(keepUpToDate=True) פתוח על הסימבול הנוכחי, ו-IB דוחף
עדכון לבר האחרון (bars[-1]) בכל שינוי. ה-thread הראשי (Dash) לא נוגע ב-IB
בכלל — הוא רק קורא את הבר האחרון מהזיכרון כל טיק של ה-Interval, וכל בקשה לשנות
מנוי (סימבול/טיים-פריים חדשים) עוברת בתור בטוח-לשרשורים אל תת-השרשור.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ib_async import IB  # noqa: E402
from ib_async.ib import StartupFetch  # noqa: E402
from ib_data import _BAR_SECONDS, TIMEFRAMES, resolve_live_contract_async  # noqa: E402

from .data import _epoch  # noqa: E402  (המרת שעון בורסה -> שעון התצוגה)

# client ID קבוע (לא random) - כדי שכל הפעלה-מחדש של השרת תשתמש באותו slot ב-
# IB Gateway במקום לצבור client ID חדש בכל פעם (זו הייתה הסיבה לעשרות "clients"
# שנראו מחוברים ב-Gateway אחרי כל restart). IB מחליף אוטומטית חיבור קיים על
# אותו client id כשמתחברים איתו מחדש - כך שגם אם התהליך הקודם נסגר בלי לנתק
# (kill -9, קריסה) ההתחברות הבאה "תופסת" את אותו slot במקום לפתוח סלוט נוסף.
_CLIENT_ID = 15050

# משך הבקשה ל-reqHistoricalData(keepUpToDate=True) - כמה אחורה מבקשים
# מלכתחילה (לא רק את הבר הפתוח), כדי שהגרף לא יתחיל מ"ריק" ברגע ההרשמה.
_LIVE_DURATION = {
    "1m": "2 D", "5m": "2 D", "15m": "2 D", "30m": "2 D",
    "1h": "2 D", "4h": "5 D", "1d": "10 D", "1w": "60 D",
}

_lock = threading.Lock()
_bar: dict | None = None
_active: tuple[str, str] | None = None  # (symbol, timeframe) שמצוירים כרגע

# זמן (time.monotonic) של עדכון-הבר האחרון שהתקבל מ-IB (bars.updateEvent) -
# ראו is_market_live. זה האות היחיד שיש לנו ל"השוק פתוח וסוחר עכשיו": בלי
# תלות בשעון-לוח קבוע (שעות מסחר משתנות בחגים/ימי מסחר מקוצרים), ובלי תלות
# ב-_bar עצמו (שנשאר לא-None גם אחרי שהשוק נסגר - הוא רק מפסיק להתעדכן, לא
# מתאפס). כשעובר יותר מ-_STALE_THRESHOLD_SECONDS בלי עדכון - נחשב "שוק סגור".
_last_tick_at: float | None = None
_STALE_THRESHOLD_SECONDS = 30

_cmd_queue: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
_thread: threading.Thread | None = None
_connected = threading.Event()

# מוחזק ברמת המודול (לא רק מקומי בתוך _main) משתי סיבות: (1) _cleanup למטה
# צריך לגשת אליו מה-thread הראשי כדי לנתק בסגירת התהליך; (2) _main עצמו בודק
# אותו לפני חיבור חדש כדי לנתק חיבור קודם באותו client id אם משום מה עוד קיים.
_ib: IB | None = None


def _bar_from_ib(b) -> dict:
    """ממירה BarData בודד (bars[-1] - הבר הפתוח/האחרון שב-BarDataList) למבנה
    הפנימי שלנו. IB כבר מחשב את open/high/low/close/volume בעצמו - אין כאן
    שום צבירה/חישוב, רק המרת שעון (אותה מוסכמה בדיוק כמו ברים היסטוריים
    רגילים - ראו _epoch/data.py)."""
    t = pd.Timestamp(b.date)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    volume = float(b.volume) if b.volume == b.volume else 0.0
    return {
        "time": _epoch(t),
        "open": float(b.open), "high": float(b.high),
        "low": float(b.low), "close": float(b.close),
        "volume": volume,
    }


async def _main(host: str, port: int) -> None:
    # מוצהרים כולם כאן, פעם אחת בראש הפונקציה - כדי שלא יהיה משנה באיזה סדר
    # טקסטואלי הם נקראים/נכתבים בהמשך (הפייתון-קומפיילר דורש שהצהרת global
    # תקדים כל שימוש בשם באותו scope, גם אם יש עוד הצהרת global לאותו שם
    # מאוחר יותר בפונקציה - ראו reconnect logic למטה שקוראת ל-_active לפני
    # ההצהרה שהייתה קיימת רק בהמשך).
    global _ib, _bar, _active, _last_tick_at

    # לפני חיבור חדש: אם משום מה כבר קיים חיבור פתוח באותו client id (מריצה
    # קודמת של _main באותו תהליך - לא אמור לקרות היום כי _ensure_started מגן
    # מפני thread כפול, אבל זו בדיוק הבדיקה שהתבקשה, ומגנה גם אם ההגנה הזו
    # אי-פעם תוסר) - מנתקים אותו קודם, כדי שלא יישארו שני סוקטים על אותו slot.
    if _ib is not None and _ib.isConnected():
        print(f"[live] existing connection on clientId={_CLIENT_ID} found - disconnecting it first", flush=True)
        try:
            _ib.disconnect()
        except Exception:
            pass

    ib = IB()
    _ib = ib
    try:
        # readonly + fetchFields(0): בלי זה כל connect מבצע גם סנכרון פוזיציות/
        # הזמנות/executions (ברירת המחדל של ib_async) שהמנגנון הזה אף פעם לא
        # צריך - רק מזרים נרות. כש-IB Gateway עמוס/איטי הסנכרון הזה יכול לתקוע
        # את החיבור לדקה+ בלי שום תועלת, ולתת תחושה שהלייב "נתקע".
        await ib.connectAsync(host, port, clientId=_CLIENT_ID, timeout=10,
                              readonly=True, fetchFields=StartupFetch(0))
    except Exception as exc:
        print(f"[live] connectAsync({host}:{port}) failed: {exc!r}", flush=True)
        return  # אין TWS/Gateway זמין - הלייב פשוט לא יעבוד, שאר האפליקציה ממשיכה כרגיל
    print(f"[live] connected to {host}:{port} clientId={_CLIENT_ID}", flush=True)
    _connected.set()

    current_bars = None         # BarDataList מ-reqHistoricalData(keepUpToDate=True)
    current_contract = None
    # שעון-שמירה (watchdog): מתי נרשמנו למנוי הנוכחי, ואם כבר הוזהרנו על "אין
    # עדכונים בכלל" בשבילו - ראו התראה בסוף הלולאה למטה. מאפסים בכל מעבר-מנוי חדש.
    subscribed_at: float | None = None
    warned_no_ticks = False
    # וולום "ננעל" לערך שהיה לו ברגע שהבר הנוכחי (frozen_volume_bar_time, זמן ה-
    # bucket שלו) נראה לראשונה - לא מתעדכן שוב עד שמזוהה בר חדש (זמן אחר). מחיר
    # (open/high/low/close) ממשיך להתעדכן בזמן אמת כרגיל בכל קריאה - רק הוולום
    # נעול. IB עצמו כבר מדווח וולום-לפי-בר נכון (לא מצטבר בין ברים - נבדק בלוג),
    # אבל תוך כדי בר בודד הוא כן גדל עם כל עסקה חדשה; "נעילה" זו היא בקשה
    # מפורשת (לא תיקון של דליפה) - הוולום המוצג לבר החי קבוע לערכו-בפתיחה, לא
    # גדל תוך-כדי, ומתעדכן לערך האמיתי הבא רק כשבר חדש נפתח. מאופס בכל מעבר-מנוי.
    frozen_volume_bar_time: int | None = None
    frozen_volume_value = 0.0

    def make_bar(b) -> dict:
        nonlocal frozen_volume_bar_time, frozen_volume_value
        bar_dict = _bar_from_ib(b)
        if frozen_volume_bar_time != bar_dict["time"]:
            frozen_volume_bar_time = bar_dict["time"]
            frozen_volume_value = bar_dict["volume"]
        bar_dict["volume"] = frozen_volume_value
        return bar_dict

    def on_bar_update(bars, hasNewBar):  # noqa: N803 (שם הפרמטר כפי שה-event מגדיר אותו)
        if bars is not current_bars or not bars:
            return
        print(f"Bar update received: close={bars[-1].close} volume={bars[-1].volume}", flush=True)
        with _lock:
            global _bar, _last_tick_at
            _bar = make_bar(bars[-1])
            _last_tick_at = time.monotonic()

    while True:
        # התנתקות מ-IB Gateway (רשת/restart של Gateway/timeout) עוצרת את זרם
        # ה-keepUpToDate בשקט - IB לא "מודיע" בשום דרך שהנר הפסיק להתעדכן, הוא
        # פשוט מפסיק לירות bars.updateEvent (נצפה בפועל: Error 10182 "Failed to
        # request live updates (disconnected)" בלוג, ואז אפס עדכונים לצמיתות
        # עד restart ידני של השרת - זה בדיוק מה שנראה כמו "הנר קפא"). מתחברים
        # מחדש כאן, ואם הייתה הרשמה פעילה (_active) - מבקשים אותה מחדש (queue,
        # לא קריאה ישירה) כדי לעבור דרך אותה לוגיקת הרשמה רגילה בדיוק כמו הרשמה
        # ראשונה, בלי לשכפל אותה.
        if not ib.isConnected():
            print("[live] connection lost - reconnecting...", flush=True)
            _connected.clear()
            current_bars = None
            current_contract = None
            try:
                await ib.connectAsync(host, port, clientId=_CLIENT_ID, timeout=10,
                                      readonly=True, fetchFields=StartupFetch(0))
            except Exception as exc:
                print(f"[live] reconnect failed: {exc!r}", flush=True)
                await asyncio.sleep(2)  # לא להציף ניסיונות חיבור כשל ברצף
                continue
            print(f"[live] reconnected to {host}:{port} clientId={_CLIENT_ID}", flush=True)
            _connected.set()
            with _lock:
                active = _active
            if active is not None:
                _cmd_queue.put(active)

        try:
            cmd = _cmd_queue.get_nowait()
        except queue.Empty:
            cmd = "idle"

        if cmd != "idle":
            if current_bars is not None:
                try:
                    ib.cancelHistoricalData(current_bars)
                except Exception:
                    pass
                current_bars = None
                current_contract = None
            if cmd is not None:
                symbol, timeframe = cmd
                try:
                    contract = await resolve_live_contract_async(ib, symbol)
                except Exception as exc:
                    print(f"[live] resolve_live_contract_async({symbol}) failed: {exc!r}", flush=True)
                    contract = None
                if contract is None:
                    print(f"[live] no contract resolved for {symbol} — skipping subscribe", flush=True)
                else:
                    print(f"reqHistoricalData(keepUpToDate=True) called for {symbol}", flush=True)
                    try:
                        bars = await ib.reqHistoricalDataAsync(
                            contract, endDateTime="",
                            durationStr=_LIVE_DURATION.get(timeframe, "2 D"),
                            barSizeSetting=TIMEFRAMES[timeframe], whatToShow="TRADES",
                            useRTH=False, keepUpToDate=True, formatDate=1, timeout=15,
                        )
                    except Exception as exc:
                        print(f"[live] reqHistoricalData({symbol}) failed: {exc!r}", flush=True)
                        bars = None
                    if not bars:
                        print(f"[live] reqHistoricalData returned no bars for {symbol}/{timeframe}", flush=True)
                    else:
                        print(f"[live] subscribed {symbol}/{timeframe} via "
                              f"reqHistoricalData(keepUpToDate=True), {len(bars)} seed bars", flush=True)
                        bars.updateEvent += on_bar_update
                        current_bars = bars
                        current_contract = contract
                        subscribed_at = time.monotonic()
                        warned_no_ticks = False
                        # מאפסים את נעילת-הוולום למנוי החדש - לא לרשת את הערך
                        # הנעול מהסימבול/טיים-פריים הקודם.
                        frozen_volume_bar_time = None
                        with _lock:
                            _active = (symbol, timeframe)
                            _bar = make_bar(bars[-1])  # מיד מהברים ההיסטוריים, בלי לחכות לעדכון ראשון
                            # מאופס למנוי החדש - עדכון עדיין לא הגיע אליו, אז
                            # is_market_live לא אמור "לרשת" חיוּת מהסימבול הקודם.
                            _last_tick_at = None

        # שעון-שמירה: אם עברו 15 שניות ממש בלי אף עדכון-בר אחד על המנוי הנוכחי -
        # זה שונה לגמרי מ"שוק סגור" (שם פשוט מפסיקים לקבל עדכונים אחרי שכן
        # קיבלנו), וכנראה מסמן בעיה אחרת (חוזה שגוי, ניתוק). מזהירים פעם אחת
        # בלבד לכל מנוי, לא בכל סבב לולאה.
        if current_bars is not None and not warned_no_ticks and subscribed_at is not None:
            if time.monotonic() - subscribed_at > 15:
                with _lock:
                    got_update = _last_tick_at is not None
                if not got_update:
                    print(f"[live] WARNING: no bar updates received in 15s since subscribing "
                          f"to {current_contract.localSymbol if current_contract else '?'}", flush=True)
                warned_no_ticks = True

        await asyncio.sleep(0.25)


def _run_thread(host: str, port: int) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main(host, port))
    except Exception:
        pass


def _ensure_started(host: str = "127.0.0.1", port: int = 4002) -> None:
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_run_thread, args=(host, port), daemon=True)
    _thread.start()


def subscribe(symbol: str, timeframe: str) -> None:
    """מבקש (בלי לחסום) מעבר-מנוי לסימבול/טיים-פריים חדשים. תת-השרשור מטפל בזה."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return
    _ensure_started()
    with _lock:
        already = _active == (symbol, timeframe)
    if already:
        return
    _cmd_queue.put((symbol, timeframe))


def get_live_bar(symbol: str, timeframe: str) -> dict | None:
    """קריאה זולה מהזיכרון בלבד — לא נוגעת ב-IB. משמש את ה-Interval ב-app.py."""
    symbol = (symbol or "").strip().upper()
    with _lock:
        if _active != (symbol, timeframe) or _bar is None:
            return None
        return dict(_bar)


def is_market_live(symbol: str, timeframe: str) -> bool:
    """True אם התקבל עדכון-בר מ-IB ב-_STALE_THRESHOLD_SECONDS האחרונות למנוי
    הנוכחי - האות שממנו app.py מחליט אם להציג את טיימר-הספירה-לאחור (מוצג רק
    כשהשוק "חי" בפועל, לא לפי שעון-לוח קבוע - ראו _last_tick_at למעלה). False
    גם אם עדיין לא הגיע אף עדכון מאז ההרשמה (למשל רגע אחרי סימבול חדש) - זה
    נכון: אין עדיין ראיה שהשוק פעיל."""
    symbol = (symbol or "").strip().upper()
    with _lock:
        if _active != (symbol, timeframe) or _last_tick_at is None:
            return False
        return (time.monotonic() - _last_tick_at) < _STALE_THRESHOLD_SECONDS


def seconds_to_close(timeframe: str) -> int:
    """כמה שניות נותרו עד שהנר הנוכחי ייסגר — הטיימר שמוצג מתחת למחיר, כמו ב-TradingView.

    החישוב נעשה בשעון הבורסה (America/New_York), אותו שעון שבו נקבעות חותמות הנרות,
    ולא בשעון המקומי של הדפדפן — אחרת הספירה הייתה משקרת בכמה שעות.

    בתוך היום הנר נסגר בסוף ה"דלי" של הטיים-פריים. ביומי/שבועי אין משמעות לספירה עד
    חצות, ולכן סופרים עד סגירת המסחר הרגילה (16:00 ET) — זו הסגירה שמעניינת סוחר.
    """
    bar_seconds = _BAR_SECONDS.get(timeframe, 60)
    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)

    if bar_seconds < 86400:
        epoch = int(now_et.tz_localize("UTC").timestamp())
        return bar_seconds - (epoch % bar_seconds)

    close = now_et.normalize() + pd.Timedelta(hours=16)
    if now_et >= close:
        close = close + pd.Timedelta(days=1)
    if timeframe == "1w":
        # הנר השבועי נסגר בסגירת יום שישי, לא בסגירה של היום הנוכחי.
        close = close + pd.Timedelta(days=(4 - close.weekday()) % 7)
    return max(0, int((close - now_et).total_seconds()))


def is_connected() -> bool:
    return _connected.is_set()


def _cleanup() -> None:
    """מתנתק מ-IB בצורה מסודרת בסגירת התהליך. IB.disconnect() הוא מתודה
    סינכרונית רגילה (לא async) שרק סוגרת סוקט ומאפסת מצב פנימי - בטוח לקרוא לה
    מה-thread הראשי כאן, גם שהחיבור עצמו נוצר בתת-השרשור של _main.

    בלי זה, עצירת השרת (Ctrl+C/kill/restart) השאירה את הסוקט פתוח מצד IB
    Gateway עד שהוא עצמו מזהה timeout (יכול לקחת זמן, ולפעמים לא קורה בכלל
    לפני שהתהליך הבא כבר מתחבר) - זו הסיבה שהצטברו עשרות "clients" מחוברים
    ב-Gateway אחרי restart-ים חוזרים."""
    ib = _ib
    if ib is not None and ib.isConnected():
        print(f"[live] shutting down - disconnecting clientId={_CLIENT_ID}", flush=True)
        try:
            ib.disconnect()
        except Exception:
            pass


atexit.register(_cleanup)


def _handle_signal(signum, _frame):
    """SIGTERM (kill רגיל, כולל עצירת dev-server) לא מפעיל atexit מעצמו -
    ברירת המחדל של האות היא לסיים את התהליך מיד, בלי לתת להוקים של atexit
    לרוץ. מנקים כאן במפורש, ואז מפעילים מחדש את הטיפול המקורי באות ושולחים
    אותו לעצמנו - כך שהתהליך עדיין ייגמר בדיוק כמו שהיה בלי ה-handler הזה
    (לא "בולעים" את ה-kill, רק מוסיפים ניקוי לפניו)."""
    _cleanup()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


try:
    signal.signal(signal.SIGTERM, _handle_signal)
except (ValueError, OSError):
    # ValueError: לא ב-main thread של המתורגם (signal.signal מותר רק שם) -
    # atexit למעלה עדיין מכסה יציאה רגילה/Ctrl+C במקרה הזה.
    pass
