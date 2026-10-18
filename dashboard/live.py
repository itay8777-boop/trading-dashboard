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
from ib_data import (  # noqa: E402
    _BAR_SECONDS,
    CONTRACT_MISMATCH_THRESHOLD_PTS,
    TIMEFRAMES,
    _load_cache,
    _save_cache,
    resolve_live_contract_async,
)

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

# כמה זמן בלי אף עדכון-בר נחשב "הזרם מת בשקט" (ולא סתם "שוק שקט") - מפעיל
# resubscribe מלא (ראו הלולאה הראשית ב-_main). אובחן בפועל: bars.updateEvent
# מפסיק להיורות בלי ש-ib.isConnected() הופך ל-False בכלל - MNQ סוחר כמעט
# 23/5, אז שקט אמיתי של דקה שלמה ברצף על 1m הוא כבר חריג, לא מצב שוק תקין.
_STALE_RESUBSCRIBE_SECONDS = 60

# עד כמה זמן בלי עדכון-אמת עדיין "סביר לפתוח נר-ריק סינתטי" (ראו
# _synthetic_next_bar_if_boundary_passed) - כדי לא לגלגל נרות-ריקים לנצח
# לאורך סגירה אמיתית וארוכה (למשל סוף שבוע, שעות): שם ה-resubscribe החוזר
# כל _STALE_RESUBSCRIBE_SECONDS כבר "מתקן" בעצמו את _bar בחזרה לבר-האמת
# האחרון (מהסשן שלפני הסגירה) בכל פעם, ואין טעם להתערב באמצע עם עוד ועוד
# נרות-ריקים מזויפים. 10 דקות נותן מרווח נדיב מעל שקט תוך-יומי רגיל (MNQ
# סוחר כמעט 23/5), בלי לחכות לסגירה אמיתית שלמה.
_SYNTHETIC_BAR_MAX_STALENESS_SECONDS = 600

_cmd_queue: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
_thread: threading.Thread | None = None
_connected = threading.Event()

# מוחזק ברמת המודול (לא רק מקומי בתוך _main) משתי סיבות: (1) _cleanup למטה
# צריך לגשת אליו מה-thread הראשי כדי לנתק בסגירת התהליך; (2) _main עצמו בודק
# אותו לפני חיבור חדש כדי לנתק חיבור קודם באותו client id אם משום מה עוד קיים.
_ib: IB | None = None


def _request_realtime_market_data(ib: IB) -> None:
    """מבקש נתוני שוק חיים (marketDataType=1), לא מתעכבים (3) - נדרש כי בלי
    הבקשה המפורשת הזו IB Gateway עשוי לשרת נתונים מתעכבים גם כשה-Market Data
    Farm (usfuture/usfarm) מחובר ופעיל וההרשאה בפועל קיימת בחשבון: reqMarketDataType
    הוא הגדרת-session, לא נגזר אוטומטית מזמינות ה-farm בלבד. חובה לקרוא לזה
    מחדש אחרי כל connect/reconnect (ההגדרה לא שורדת חיבור-מחדש) - ראו שני
    מקומות הקריאה ב-_main. משפיע על reqHistoricalData(keepUpToDate=True)
    (הסטרימינג החי כאן) וגם על reqMktData (ib_data.py: בדיקות Open Interest).
    """
    try:
        ib.reqMarketDataType(1)
        print("[live] requested real-time market data (marketDataType=1)", flush=True)
    except Exception as exc:
        print(f"[live] reqMarketDataType(1) failed: {exc!r}", flush=True)


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


def _synthetic_next_bar_if_boundary_passed(current_bar: dict, bar_seconds: int,
                                            now_et: pd.Timestamp) -> dict | None:
    """פונקציה טהורה (בלי side-effects/global) - קלה לבדיקה בנפרד. משווה את
    ה-bucket שהשעון-בפועל (now_et, שעון הבורסה) נמצא בו כרגע מול ה-bucket
    שאליו current_bar (הבר החי הנוכחי, כפי שמוחזק ב-_bar) שייך. אם השעון כבר
    עבר ל-bucket מאוחר יותר - מחזירה בר-ריק חדש לתחילת אותו bucket
    (open=high=low=close=הסגירה האחרונה של current_bar, volume=0). אחרת None
    (עדיין לא הגיע הזמן להתקדם).

    משמש ב-_main כדי שהגרף יפתח נר חדש בדיוק בתחילת כל דקה, גם כשעדכון-בר
    אמיתי מ-IB (on_bar_update) מתעכב מעבר לרגע-האפס בפועל - נמדד ישירות
    שזה קורה (לפעמים כמה שניות, לפעמים הרבה יותר בשוק שקט)."""
    now_bucket = (int(_epoch(now_et)) // bar_seconds) * bar_seconds
    bar_bucket = (int(current_bar["time"]) // bar_seconds) * bar_seconds
    if now_bucket <= bar_bucket:
        return None
    close = current_bar["close"]
    return {"time": now_bucket, "open": close, "high": close,
            "low": close, "close": close, "volume": 0.0}


def _bars_to_cache_df(bars) -> pd.DataFrame:
    """ממירה רשימת BarData (או בר בודד, ברשימה בת-איבר-אחד) לפורמט בדיוק כמו
    fetch_ohlcv/_normalize שומרות לקאש (Open/High/Low/Close/Volume, אינדקס
    naive בשעון הבורסה) - כדי שתתמזג עם הקאש הקיים בלי שום המרה נוספת. בנייה
    וקטורית (רשימות + DataFrame אחד בסוף) ולא concat של DataFrame-ים בודדים
    בלולאה - חשוב כש-bars הוא seed שלם (עד כ-2 ימים = ~2800 ברי-דקה, ראו
    _persist_seed_bars).
    """
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for b in bars:
        t = pd.Timestamp(b.date)
        if t.tzinfo is not None:
            t = t.tz_convert("America/New_York").tz_localize(None)
        dates.append(t)
        opens.append(float(b.open))
        highs.append(float(b.high))
        lows.append(float(b.low))
        closes.append(float(b.close))
        volumes.append(float(b.volume) if b.volume == b.volume else 0.0)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _merge_and_save_cache(symbol: str, timeframe: str, new_df: pd.DataFrame, source: str) -> None:
    """מבצעת את המיזוג+שמירה בפועל, דרך אותן _load_cache/_save_cache
    ש-fetch_ohlcv (ib_data.py) כבר משתמש בהן - לא מנגנון-שמירה מקביל.
    משותפת לשני מקורות הנתונים למטה (בר-שנסגר בודד / seed שלם בהרשמה).

    בטיחות: אותה בדיקת-חפיפה-מול-מחיר-קיים בדיוק כמו fetch_ohlcv
    (CONTRACT_MISMATCH_THRESHOLD_PTS) - הגנה נוספת אם החוזה הפעיל אי-פעם
    יתבלבל שוב (ראו התקלה עם MNQZ6 שתוקנה ב-ib_data.py:_pick_most_liquid).
    כשל בשמירה כאן לא אמור להפיל את הסטרימינג החי - רק מדלג/מדפיס אזהרה.
    """
    if new_df is None or new_df.empty:
        return
    try:
        existing = _load_cache(symbol, timeframe, use_rth=False)
        if existing is not None and not existing.empty:
            overlap = new_df.index.intersection(existing.index)
            if len(overlap) > 0:
                diff = (new_df.loc[overlap, "Close"] - existing.loc[overlap, "Close"]).abs()
                bad = diff[diff > CONTRACT_MISMATCH_THRESHOLD_PTS]
                if len(bad) > 0:
                    print(f"[live] skipping {source} cache save for {symbol}/{timeframe} - "
                          f"{len(bad)} overlapping bar(s) disagree by up to {bad.max():.2f}pts "
                          f"with existing cache, looks like a contract mismatch, not a real "
                          f"update", flush=True)
                    return
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df
        _save_cache(symbol, timeframe, use_rth=False, df=merged)
    except Exception as exc:
        print(f"[live] warning: failed to persist {source} to cache "
              f"({symbol}/{timeframe}): {exc!r}", flush=True)


def _persist_closed_bar(symbol: str, timeframe: str, closed_bar) -> None:
    """שומרת בר-שנסגר (bars[-2] ברגע hasNewBar=True - עובדה סופית, לא ישתנה
    יותר) לקאש - הנתיב המהיר/העדין: בר בודד בכל פעם ש-IB בפועל דוחף עדכון.

    למה זה נחוץ כאן בכלל: הקאש המאוחד היה מתעדכן *רק* כש-fetch_ohlcv נקרא,
    וזה קורה *רק* מ-/api/lod (app.py) בתגובה לשינוי טווח-נראה בגרף בפועל
    (ראו assets/chart.js: onVisibleTimeRangeChanged) - אין שם שום טיימר
    תקופתי שמרענן מחדש. כל עוד המשתמש לא נוגע בגרף (רק צופה בקצה החי, שמוצג
    ישירות מהזיכרון של המודול הזה - get_live_bar - ולא מהקאש בכלל), הקאש
    הקבוע נשאר תקוע בדיוק בזמן הפעולה האחרונה על הגרף, גם כשעוברות שעות של
    מסחר אמיתי. שמירה כאן, ברגע שכל בר נסגר, סוגרת את הפער בלי לגעת ב-UI/
    ב-app.py בכלל - אותו thread שכבר מקבל את העדכון מ-IB, בלי תור/callback נוסף.

    הנתיב הזה תלוי ב-bars.updateEvent (on_bar_update) שנמדד בפועל כלא-אמין
    לגמרי לפרקים (על חשבון בלי הרשאת live-data - Error 354, delayed בלבד:
    נצפו ריצות שלמות של דקות בלי אף עדכון אחד) - ראו _persist_seed_bars
    למטה, שהיא רשת-הביטחון הבלתי-תלויה בזרם הזה בכלל.
    """
    try:
        row = _bars_to_cache_df([closed_bar])
    except Exception as exc:
        print(f"[live] warning: failed to convert closed bar for cache "
              f"({symbol}/{timeframe}): {exc!r}", flush=True)
        return
    _merge_and_save_cache(symbol, timeframe, row, source="closed-bar")


def _persist_seed_bars(symbol: str, timeframe: str, bars) -> None:
    """שומרת את כל ברי ה-seed (כל bars חוץ מהאחרון - הוא הבר הפתוח, עוד לא
    סגור) שמתקבלים מכל reqHistoricalData(keepUpToDate=True) מוצלח - גם
    הרשמה ראשונה וגם resubscribe שהופעל ע"י שעון-השמירה למטה (_STALE_
    RESUBSCRIBE_SECONDS).

    זו רשת-הביטחון האמיתית לצמיחת הקאש: reqHistoricalData עצמו (בקשת
    request/response רגילה) נמצא אמין לגמרי לאורך כל האבחון הזה, גם כש-
    bars.updateEvent (on_bar_update, הדחיפה החיה בפועל) משתתק לחלוטין
    לדקות ארוכות על החשבון הזה. שעון-השמירה כבר מכריח resubscribe לפחות
    כל _STALE_RESUBSCRIBE_SECONDS (60) גם כשהזרם החי מת לגמרי - כך שהקריאה
    הזו מבטיחה שהקאש יתקדם לפחות בקצב הזה, בלי שום תלות ב-hasNewBar בכלל.
    """
    if not bars or len(bars) < 2:
        return
    try:
        new_df = _bars_to_cache_df(bars[:-1])
        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()
    except Exception as exc:
        print(f"[live] warning: failed to convert seed bars for cache "
              f"({symbol}/{timeframe}): {exc!r}", flush=True)
        return
    _merge_and_save_cache(symbol, timeframe, new_df, source=f"seed ({len(new_df)} bars)")


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
    _request_realtime_market_data(ib)

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
        raw = bars[-1]
        with _lock:
            global _bar, _last_tick_at
            _bar = make_bar(raw)
            _last_tick_at = time.monotonic()
        # לוג מלא של הבר החי האחרון שהתקבל מ-IB, כולל raw_ib_date עם ה-tzinfo
        # הגולמי שלו (בפועל US/Central עבור MNQ/CME - לא US/Eastern) לצד הבר
        # אחרי ההמרה (_bar_from_ib כבר עושה tz_convert("America/New_York")
        # נכון) - כדי שיהיה אפשר לוודא בעין שהזמן/מחיר שמגיעים בפועל תואמים
        # למה שמוצג בגרף, ולא רק להסיק את זה מהתנהגות ה-UI.
        print(f"[live] bar update {_active}: raw_ib_date={raw.date!r} -> "
              f"O={_bar['open']} H={_bar['high']} L={_bar['low']} C={_bar['close']} "
              f"V={_bar['volume']} time_epoch={_bar['time']}", flush=True)
        # לוג פשוט וממוקד, בפורמט קבוע לחיפוש קל - כל עדכון-בר שבאמת מגיע מ-IBKR
        # (לא רק כשמזוהה בעיה, כמו הלוג המפורט למעלה).
        print(f"Live bar update: close={_bar['close']} time={_bar['time']}", flush=True)

        # hasNewBar=True: בר חדש נפתח, כלומר bars[-2] (לא bars[-1] - זה הבר
        # החדש/הפתוח) עכשיו סגור סופית ולא ישתנה יותר - זה הרגע הבטוח לשמור
        # אותו לקאש הקבוע (ראו _persist_closed_bar). שומרים כאן, לא בכל עדכון
        # (הבר הפתוח עצמו), כדי לא לכתוב את אותו קובץ פרקט שוב ושוב בתוך אותה
        # דקה - פעם אחת לכל בר, בדיוק כשהוא הופך לעובדה גמורה.
        if hasNewBar and _active is not None and len(bars) >= 2:
            symbol, timeframe = _active
            _persist_closed_bar(symbol, timeframe, bars[-2])

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
            _request_realtime_market_data(ib)
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
                    # localSymbol (למשל MNQU6/MNQZ6) - כדי לוודא בעין שהחוזה
                    # הפעיל הוא ה-front-month הנכון, בלי לנחש מהתנהגות ה-UI.
                    print(f"[live] resolved contract for {symbol}: "
                          f"localSymbol={getattr(contract, 'localSymbol', '?')} "
                          f"expiry={getattr(contract, 'lastTradeDateOrContractMonth', '?')} "
                          f"conId={getattr(contract, 'conId', '?')}", flush=True)
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
                        # רשת-הביטחון לצמיחת הקאש - לא תלויה ב-bars.updateEvent
                        # בכלל (ראו _persist_seed_bars). קורה בכל הרשמה/resubscribe,
                        # כולל resubscribe שהופעל ע"י שעון-השמירה למטה - לכן
                        # מבטיחה התקדמות גם כשהזרם החי משתתק לגמרי.
                        _persist_seed_bars(symbol, timeframe, bars)

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

        # תיקון "הנר קפא": התקלה שאובחנה בפועל (ראו הודעת המודול) - bars.updateEvent
        # מפסיק להיורות בשקט (בלי ib.isConnected() להפוך ל-False בכלל, אז לולאת
        # ה-reconnect למעלה לא רואה שום בעיה) ו-_bar נשאר קפוא לערך ישן, בעוד
        # push_live_update ב-app.py ממשיך לשלוח אותו שוב ושוב לדפדפן כאילו הכל
        # תקין. שעון-שמירה שני, נפרד מהאזהרה החד-פעמית למעלה: אם עברו
        # _STALE_RESUBSCRIBE_SECONDS בלי אף עדכון (בין אם מעולם לא הגיע ובין אם
        # הגיע ואז נעצר) - מניחים שהזרם מת ומכריחים resubscribe מלא, דרך אותה
        # תור-הפקודות ואותו קוד cancel+reqHistoricalData בדיוק כמו הרשמה רגילה
        # (לא כתיבה ישירה ל-current_bars/_bar כאן) - subscribed_at מתאפס מיד
        # כדי לתת ל-resubscribe החדש זמן לתפוס לפני שהשעון הזה יורה שוב.
        if current_bars is not None and subscribed_at is not None:
            with _lock:
                last_tick = _last_tick_at
            idle_reference = last_tick if last_tick is not None else subscribed_at
            idle_seconds = time.monotonic() - idle_reference
            if idle_seconds > _STALE_RESUBSCRIBE_SECONDS:
                with _lock:
                    active = _active
                print(f"[live] no bar updates in {idle_seconds:.0f}s for "
                      f"{current_contract.localSymbol if current_contract else '?'} "
                      f"({active}) - forcing resubscribe", flush=True)
                if active is not None:
                    _cmd_queue.put(active)
                subscribed_at = time.monotonic()
                warned_no_ticks = False

        # "הנר לא עובר לדקה הבאה": IB יכול להתעכב הרבה מעבר לרגע-האפס בפועל
        # לפני שהוא דוחף עדכון עם bars[-1] החדש (נמדד ישירות: לפעמים שניות
        # בודדות, לפעמים - במיוחד בשוק שקט - הרבה יותר) - עד אז _bar נשאר
        # קפוא על הבר הישן, בזמן שהטיימר בצד הלקוח (assets/chart.js, שעון
        # טהור בדפדפן) כבר מגיע לאפס ומתאפס, בלי שום קשר לזה שהגיע עדכון
        # אמת. במקום לחכות ל-IB (או לשעון-השמירה למעלה, שיכול לקחת עד 60
        # שניות שלמות), בודקים כאן בעצמנו בכל סבב לולאה (כל רבע שנייה) אם
        # השעון-בפועל כבר עבר ל-bucket הבא של הטיים-פריים הנוכחי - ואם כן,
        # "פותחים" נר ריק חדש (open=high=low=close=הסגירה האחרונה הידועה,
        # volume=0) בעצמנו, מיד. ברגע שעדכון אמיתי מ-IB כן מגיע (on_bar_update),
        # הוא פשוט דורס את הבר הסינתטי הזה בערכים האמיתיים כרגיל - אין כאן
        # שום "נעילה" על הערך הסינתטי, רק מילוי-מקום עד שהאמת מגיעה.
        if current_bars is not None:
            with _lock:
                bar_snapshot = dict(_bar) if _bar is not None else None
                active_now = _active
                last_tick = _last_tick_at
            # שער-בטיחות: לא מגלגלים נרות-ריקים סינתטיים לנצח לאורך סגירה
            # אמיתית וארוכה (סוף שבוע וכו') - ראו _SYNTHETIC_BAR_MAX_STALENESS_SECONDS.
            # last_tick=None (אף עדכון-אמת לא הגיע עדיין למנוי הנוכחי) נחשב
            # "יותר מדי זמן" גם הוא - עדיף לא לנחש בלי שום עוגן אמיתי.
            reference = last_tick if last_tick is not None else 0.0
            recently_live = (time.monotonic() - reference) < _SYNTHETIC_BAR_MAX_STALENESS_SECONDS
            if bar_snapshot is not None and active_now is not None and recently_live:
                _, tf_now = active_now
                bar_seconds = _BAR_SECONDS.get(tf_now, 60)
                now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
                synthetic = _synthetic_next_bar_if_boundary_passed(bar_snapshot, bar_seconds, now_et)
                if synthetic is not None:
                    with _lock:
                        # רק אם _bar לא כבר התקדם בעצמו בין המדידה למעלה לכאן
                        # (מירוץ עדין מול on_bar_update שרץ ב-thread הזה עצמו,
                        # אבל event loop יכול להחליף הקשר בין await-ים) - לא
                        # דורסים עדכון אמיתי שכבר הגיע.
                        if _bar is not None and _bar["time"] == bar_snapshot["time"]:
                            _bar = synthetic
                    print(f"[live] bar boundary passed with no IB update yet for "
                          f"{active_now} - opening synthetic empty bar at "
                          f"time_epoch={synthetic['time']} (prev close={synthetic['close']})", flush=True)

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
