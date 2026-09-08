"""שכבת הדאטה של הדשבורד: שאיבה מ-IB TWS והמרה לפורמט של Lightweight Charts.

השאיבה עצמה נשענת על ib_data שבשורש הפרויקט — שם כבר יושבים דפדוף אחורה בקריאות
מרובות (IB מגביל כל בקשה לפי גודל הבר: שבוע אחד בלבד לנרות דקה, חודש ב-5m..1h,
שנה ביומי) וקאש parquet מקומי. כאן רק מוסיפים את מה שהגרף צריך.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_data import TIMEFRAMES, _BAR_SECONDS, _load_cache, fetch_ohlcv  # noqa: E402

# הטיים-פריימים שמוצגים בסרגל העליון, לפי הסדר.
TIMEFRAME_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# 5 שנות היסטוריה בכל טיים-פריים. נבדק מול TWS ש-IB באמת מחזיק דאטה כמה שנים
# אחורה גם בנרות דקה. העלות היא מספר הקריאות: IB מגביל כל בקשה לשבוע אחד בנרות
# דקה (~261 קריאות ל-5 שנים, שאיבה ראשונה ארוכה — עד כ-30 דקות במדידה בפועל) מול
# חודש ב-5m..1h ושנה ומעלה ב-4h/1d/1w. הכל נשמר בקאש parquet, כך שהמחיר משולם פעם
# אחת לכל סימבול/טיים-פריים.
FIVE_YEARS = 1825
DEFAULT_DAYS = {tf: FIVE_YEARS for tf in TIMEFRAME_ORDER}

# הערכה גסה למספר הקריאות ל-IB לשאיבה מלאה — משמש לאזהרה בממשק לפני שאיבה ארוכה.
CHUNKS_PER_FETCH = {"1m": 261, "5m": 60, "15m": 60, "30m": 60, "1h": 60, "4h": 5, "1d": 5, "1w": 3}

# הטעינה הראשונה מביאה רק חלון קטן (מהיר: גם השאיבה מ-IB וגם ה-JSON לדפדפן).
# בטיים-פריימים גדולים 5 שנים זה ממילא מעט ברים (יומי/שבועי), אז אין צורך לחתוך.
# בעדינים (1m..1h) חלון גדול מדי הוא כבר בעיה בפני עצמה: 439 יום של 1m יצא
# 357K ברים ש-JSON חד-פעמי אליהם פשוט תקוע בדפדפן לדקות ארוכות — לא בעיית IB.
INITIAL_DAYS = {
    "1m": 10, "5m": 30, "15m": 90, "30m": 180,
    "1h": FIVE_YEARS, "4h": FIVE_YEARS, "1d": FIVE_YEARS, "1w": FIVE_YEARS,
}

# כשמזיזים את הגרף וקרבים לקצה השמאלי (הישן ביותר) של מה שכבר טעון, כל "העמסה"
# נוספת מביאה חלון בגודל הזה - לא הכל בבת אחת, כדי שכל בקשה תישאר קלה ומהירה.
HISTORY_CHUNK_DAYS = {
    "1m": 10, "5m": 30, "15m": 90, "30m": 180,
    "1h": 365, "4h": 365, "1d": 730, "1w": 1095,
}


class DataError(Exception):
    """שגיאה ידידותית להצגה בממשק (חיבור ל-TWS, סימבול לא תקין וכו')."""


def load_ohlcv(symbol: str, timeframe: str, days: int | None = None,
               extended_hours: bool = False, end_date=None) -> pd.DataFrame:
    """מחזיר DataFrame עם Open/High/Low/Close/Volume ואינדקס זמן עולה.

    `end_date` (naive, בשעון הבורסה) משמש לטעינת "עוד היסטוריה אחורה" כשמזיזים
    את הגרף — במקום מ"עכשיו", מבקשים `days` ימים שמסתיימים בדיוק לפני מה שכבר
    טעון בגרף.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise DataError("הזן סימבול")
    if timeframe not in TIMEFRAMES:
        raise DataError(f"טיים-פריים לא נתמך: {timeframe}")

    days = int(days or DEFAULT_DAYS.get(timeframe, 180))
    try:
        df = fetch_ohlcv(symbol, days, timeframe, use_rth=not extended_hours, end_date=end_date)
    except ConnectionError as exc:
        raise DataError(str(exc)) from exc
    except Exception as exc:
        raise DataError(f"שאיבת {symbol} נכשלה: {exc}") from exc

    if df is None or df.empty:
        raise DataError(f"לא התקבלו נתונים עבור {symbol}")
    return df


_RESAMPLE_RULE = {"5m": "5min", "15m": "15min", "30m": "30min",
                   "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """מדגם 1m לטיים-פריים גס יותר, בשביל מדרג-הפירוט (LOD)."""
    if timeframe == "1m" or df.empty:
        return df
    rule = _RESAMPLE_RULE.get(timeframe)
    if not rule:
        return df
    agg = df.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return agg.dropna(subset=["Open"])


def load_lod_window_from_1m_cache(symbol: str, timeframe: str, use_rth: bool,
                                   from_ts: pd.Timestamp, to_ts: pd.Timestamp) -> pd.DataFrame | None:
    """מנסה לבנות חלון LOD מדגימה מתוך קאש ה-1m הקיים בלבד, בלי שום קריאה ל-IB.

    לחוזים עתידיים (כמו MNQ) יש קאש 1m עמוק במיוחד (מ-Databento) בזמן שהקאש
    הטבעי של טיים-פריימים גסים יותר קצר בהרבה (שרשור-חוזים ידני של IB) - בקשת
    LOD גס במקום שבו הקאש הטבעי לא מכסה הייתה נופלת בחזרה לשאיבה חיה מ-IB
    (איטית, ולפעמים נתקעת ממש - בדיוק המנגנון שרצינו להימנע ממנו ב-LOD).
    מדגום מתוך ה-1m נמנע מזה לגמרי כל עוד יש שם קאש שמכסה את החלון המבוקש.
    מחזיר None אם אין קאש 1m בכלל, או שהוא לא מכסה את החלון המבוקש.
    """
    if timeframe == "1m":
        return None
    cached = _load_cache(symbol, "1m", use_rth)
    if cached is None or cached.empty:
        return None
    window = cached[(cached.index >= from_ts) & (cached.index <= to_ts)]
    if window.empty:
        return None
    return resample_ohlcv(window, timeframe)


def load_ohlcv_preferring_1m_cache(symbol: str, timeframe: str, days: int,
                                    extended_hours: bool = False) -> pd.DataFrame:
    """כמו load_ohlcv, אבל לטיים-פריים גס מ-1m מנסה קודם לדגום מתוך קאש ה-1m -
    בדיוק כמו load_lod_window_from_1m_cache, ומאותה סיבה: לחוזים כמו MNQ יש שם
    היסטוריה עמוקה מ-Databento, בזמן שהקאש הטבעי של 5m/15m/30m/1h/4h/1d/1w
    מוגבל לכמה שנים (שרשור-חוזים ידני של IB) - טעינה ראשונית של טיים-פריים גס
    בלי הניסיון הזה קודם הייתה נופלת ישר לשאיבה חיה איטית (ולפעמים כמעט תקועה,
    ראו _fetch_futures_history) בכל פעם שלוחצים על הכפתור, גם כשכל הדאטה כבר
    בקאש בשכבה עדינה יותר ורק צריך לדגום אותה.
    """
    if timeframe != "1m":
        now = pd.Timestamp.now()
        from_ts = now - pd.Timedelta(days=days)
        resampled = load_lod_window_from_1m_cache(
            symbol, timeframe, use_rth=not extended_hours, from_ts=from_ts, to_ts=now
        )
        if resampled is not None and not resampled.empty:
            return resampled
    return load_ohlcv(symbol, timeframe, days=days, extended_hours=extended_hours)


def load_full_history(symbol: str, timeframe: str, extended_hours: bool = False) -> pd.DataFrame | None:
    """כל ההיסטוריה הזמינה בקאש ה-1m (לא רק חלון LOD קטן) - לבקטסט, כדי
    שהאסטרטגיה תיבדק על כמה שיותר דאטה אמיתי (למשל כל 7 השנים של MNQ) ולא רק
    על החלון הראשוני הקטן שהוצג בגרף. מחזיר None אם אין בכלל קאש 1m לסימבול
    הזה (למשל סימבול שמעולם לא נצפה ב-1m) - במקרה כזה עדיף ליפול חזרה למה
    שכבר טעון (_FRAMES) מאשר לא להריץ בקטסט בכלל.
    """
    cached = _load_cache(symbol, "1m", use_rth=not extended_hours)
    if cached is None or cached.empty:
        return None
    # .copy(): cached הוא עכשיו אובייקט משותף מהקאש-בזיכרון של _load_cache (ראו
    # ib_data.py) - הבקטסט מקבל את זה ישירות (לא דרך resample שכבר מייצר עותק
    # חדש), אז חייבים עותק כדי שספריית backtesting.py לא תשנה בטעות את הדאטה
    # המשותף שכל שאר האפליקציה קוראת ממנו.
    return cached.copy() if timeframe == "1m" else resample_ohlcv(cached, timeframe)


def load_window_history(symbol: str, from_ts: pd.Timestamp, to_ts: pd.Timestamp,
                        extended_hours: bool = False, buffer_multiplier: float = 0.5) -> pd.DataFrame | None:
    """חותכת מתוך קאש ה-1m חלון סביב טווח נראה בגרף (with buffer) - לבקטסט מהיר
    על מה שבאמת מוצג עכשיו, לא על כל 7 שנות ההיסטוריה (load_full_history). זה
    מה ש-run-btn ("Add to chart") מריץ כברירת מחדל - backtesting.py מריץ בר-אחר-בר
    בפייתון, ועל 2.5 מיליון ברים זה יכול לקחת כמה דקות; על חלון של כמה שבועות/
    חודשים זה אמור לקחת שניות בודדות.

    buffer_multiplier=0.5 (לא ה-3.0 שמשמש ל-LOD): כאן המטרה היא הקשר קטן לחימום
    אינדיקטורים (SMA/EMA וכו') ולא buffer-לגלילה-חלקה, אז buffer גדול מדי היה
    רק מאט את הבקטסט בלי תועלת אמיתית - עדיין נותן חצי מרוחב-התצוגה נוסף מכל צד.
    מחזירה None אם אין קאש 1m בכלל או שהחלון (כולל buffer) יוצא ריק.
    """
    cached = _load_cache(symbol, "1m", use_rth=not extended_hours)
    if cached is None or cached.empty:
        return None
    span = to_ts - from_ts
    if span <= pd.Timedelta(0):
        span = pd.Timedelta(days=1)
    buffer = span * buffer_multiplier
    window = cached[(cached.index >= from_ts - buffer) & (cached.index <= to_ts + buffer)]
    # .copy(): אותה סיבה כמו ב-load_full_history - cached משותף מהקאש-בזיכרון,
    # לא בטוח למסור אותו (או פרוסה ממנו) ישירות ל-backtesting.py בלי עותק.
    return window.copy() if not window.empty else None


def warm_cache(symbol: str, extended_hours: bool = True) -> None:
    """טוען את קאש ה-1m לזיכרון ברקע, בלי לחסום את עליית השרת - כדי שהגלילה
    הראשונה של המשתמש (מייד אחרי שהדשבורד עולה) לא תשלם את עלות קריאת קובץ
    ה-parquet הרב-שנתי מהדיסק (יכולה לקחת כמה מאות מ"ש). ראו _load_cache
    ב-ib_data.py - הקריאה הראשונה בלבד קוראת מהדיסק, כל קריאה נוספת (מכל
    thread) מקבלת את אותו אובייקט מהזיכרון כל עוד הקובץ לא השתנה.
    """
    def _warm():
        try:
            _load_cache(symbol, "1m", use_rth=not extended_hours)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()


def load_latest_bars(symbol: str, timeframe: str, extended_hours: bool = False, n: int = 2) -> pd.DataFrame:
    """שאיבה חסכונית לעדכון לייב: רק N הנרות האחרונים, לא כל ההיסטוריה.

    משתמשת באותה fetch_ohlcv (וכך גם באותו קאש/הגנת-pacing), רק עם חלון ימים קטן
    שמובטח לכלול את הבר האחרון גם בטיים-פריימים גדולים (יומי/שבועי). ה-Interval
    ב-app.py קורא לפונקציה הזו כל כמה שניות במקום לבנות מחדש את כל המטען לגרף.
    """
    bar_days = -(-2 * _BAR_SECONDS.get(timeframe, 60) // 86400)  # ceil
    days = max(2, bar_days)
    df = load_ohlcv(symbol, timeframe, days=days, extended_hours=extended_hours)
    return df.tail(n)


# חותמות הזמן שמגיעות מ-IB הן שעון מקומי של הבורסה (ניו-יורק) בלי אזור זמן.
# הציר בגרף מוצג בשעון ישראל, ולכן ממירים ביניהם. ההמרה חייבת להיות לפי חותמת
# ולא בהיסט קבוע — המעבר לשעון קיץ בישראל ובארה"ב לא קורה באותם תאריכים, ולכן
# ההפרש בין השעונים אינו קבוע לאורך השנה.
SOURCE_TZ = "America/New_York"
DISPLAY_TZ = "Asia/Jerusalem"


def to_display_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """שעון בורסה (naive) -> שעון ישראל (naive). מוקטור, לכל האינדקס בבת אחת."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return (idx
            .tz_localize(SOURCE_TZ, nonexistent="shift_forward", ambiguous=True)
            .tz_convert(DISPLAY_TZ)
            .tz_localize(None))


def from_display_epoch(epoch: int) -> pd.Timestamp:
    """הופכת את _epoch(): epoch (שניות, "UTC מזויף" שמייצג שעון ישראל) -> חותמת
    naive בשעון הבורסה (ניו-יורק). משמשת כש"עוד היסטוריה" מתבקשת מהדפדפן — הוא
    שולח את הזמן הישן ביותר שכבר טעון בגרף (באותה מוסכמת "UTC מזויף"), וצריך
    להמיר אותו לשעון הבורסה כדי להעביר כ-end_date ל-fetch_ohlcv.
    """
    israel_naive = pd.Timestamp(epoch, unit="s")  # ה"ערכים הגולמיים" הם שעון ישראל
    return (israel_naive
            .tz_localize(DISPLAY_TZ, nonexistent="shift_forward", ambiguous=True)
            .tz_convert(SOURCE_TZ)
            .tz_localize(None))


def _epoch(ts) -> int:
    """זמן בודד ל-Lightweight Charts: שניות UNIX.

    הספרייה מציגה חותמות UNIX לפי UTC, ולכן הזמן מומר לשעון ישראל ואז "מתחזה"
    ל-UTC — כך שמה שנראה על הציר הוא בדיוק השעה הישראלית.
    """
    return int(to_display_naive(pd.DatetimeIndex([ts]))[0].tz_localize("UTC").timestamp())


def to_chart_payload(df: pd.DataFrame) -> dict:
    """ממיר את הנרות למבנה ש-assets/chart.js יודע לצייר."""
    # המרה מוקטורת: לולאה לכל נר הייתה כבדה מדי על עשרות אלפי נרות.
    # ההמרה עוברת דרך datetime64[s] במפורש. pandas 3 שומר אינדקסים ברזולוציית
    # מיקרו-שניות, ולכן astype("int64") מחזיר מיקרו-שניות ולא ננו — חלוקה ב-10^9
    # הייתה נותנת ערכים קטנים פי 1000 וכל הנרות היו נוחתים ב-1970.
    times = to_display_naive(df.index).astype("datetime64[s]").astype("int64").tolist()
    candles = [
        {"time": t, "open": float(o), "high": float(h), "low": float(low), "close": float(c)}
        for t, o, h, low, c in zip(times, df["Open"], df["High"], df["Low"], df["Close"])
    ]
    volume = [
        # צבע כקוד "u"/"d" בלבד (לא מחרוזת rgba מלאה) - על חלון היסטוריה גדול
        # (מאות אלפי ברים ב"Go to date" רחוק) זה לבד חוסך מגה-בייטים של טקסט
        # חוזר; assets/chart.js מרחיב בחזרה ל-rgba האמיתי לפני שמזין ל-LightweightCharts.
        {"time": t, "value": float(v), "color": "u" if c >= o else "d"}
        for t, v, o, c in zip(times, df["Volume"], df["Open"], df["Close"])
    ]
    return {"candles": candles, "volume": volume}
