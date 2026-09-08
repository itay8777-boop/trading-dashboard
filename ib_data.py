"""שאיבת נתוני מחיר מ-Interactive Brokers TWS דרך ib_async, בכל טיים-פריים נתמך.

IB מגביל כל קריאת reqHistoricalData בודדת למשך מקסימלי לפי גודל הבר (למשל שבוע
אחד בלבד לנרות של דקה). כדי לאפשר טווחים ארוכים יותר האפליקציה "מדפדפת" אחורה
בזמן עם כמה קריאות רצופות (pagination), עד לטווח המבוקש או עד שה-IB מפסיק
להחזיר דאטה (סוף ההיסטוריה השמורה אצלו).

כל שאיבה נשמרת גם בקאש מקומי (parquet, בתיקיית data_cache/) לפי סימבול/
טיים-פריים/RTH. בשאיבה הבאה לאותו סימבול נמשך מה-IB רק מה שבאמת חסר: ה"זנב"
החדש שנוסף מאז השאיבה הקודמת (ב-live, כלומר בלי end_date), וה"ראש" ההיסטורי
הישן יותר אם מבקשים טווח ארוך יותר ממה שכבר נשמר. חלון היסטורי קבוע (עם
end_date) שכבר מכוסה במלואו על ידי הקאש מוחזר ישירות בלי להתחבר ל-IB בכלל —
העבר לא משתנה. כך שאיבה חוזרת לאותו סימבול הופכת למהירה משמעותית, גם על פני
טווחים של כמה שנים.
"""
from __future__ import annotations

import os
import random
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_async import IB, Future, Stock, util
from ib_async.ib import StartupFetch

# בורסת ברירת המחדל לחוזים עתידיים נפוצים (מיקרו ורגילים). סימבול שלא ברשימה
# מטופל כמניה (Stock/SMART) כמו קודם. חוזה עתידי מזוהה אוטומטית מהסימבול בלבד —
# אין צורך לציין את זה בממשק, בדיוק כמו שמקלידים "MNQ" ומקבלים את המשכיות היומי.
FUTURES_EXCHANGES = {
    "MNQ": "CME", "NQ": "CME", "MES": "CME", "ES": "CME",
    "MYM": "CBOT", "YM": "CBOT", "M2K": "CME", "RTY": "CME",
    "MCL": "NYMEX", "CL": "NYMEX", "MGC": "COMEX", "GC": "COMEX", "SI": "COMEX",
    "ZB": "CBOT", "ZN": "CBOT", "ZF": "CBOT", "ZC": "CBOT", "ZS": "CBOT", "ZW": "CBOT",
    "6E": "CME", "6B": "CME", "6J": "CME",
}


def is_futures_symbol(symbol: str) -> bool:
    return symbol.strip().upper() in FUTURES_EXCHANGES


def _nearest_active(details, today: str):
    if not details:
        return None
    contracts = sorted((d.contract for d in details), key=lambda c: c.lastTradeDateOrContractMonth)
    for c in contracts:
        if c.lastTradeDateOrContractMonth >= today:
            return c
    return contracts[-1]


def resolve_live_contract(ib: IB, symbol: str):
    """חוזה למסחר/סטרימינג חי (סינכרוני — לשימוש מחוץ ל-event loop פעיל): לחוזה
    עתידי צריך את החוזה הקונקרטי הקרוב ביותר שעדיין לא פקע (ContFuture לא תמיד
    תומך ב-reqMktData) — נמצא דרך reqContractDetails וממוין לפי חודש הפקיעה.
    מניה כרגיל."""
    symbol = symbol.strip().upper()
    exchange = FUTURES_EXCHANGES.get(symbol)
    if not exchange:
        contract = Stock(symbol, "SMART", "USD")
        return contract if ib.qualifyContracts(contract) else None

    details = ib.reqContractDetails(Future(symbol, exchange=exchange, currency="USD"))
    return _nearest_active(details, datetime.now().strftime("%Y%m%d"))


async def resolve_live_contract_async(ib: IB, symbol: str):
    """כמו resolve_live_contract, אבל async — לשימוש מתוך event loop פעיל (כמו
    ב-dashboard/live.py, שרץ בתוך תת-שרשור עם ה-loop שלו)."""
    symbol = symbol.strip().upper()
    exchange = FUTURES_EXCHANGES.get(symbol)
    if not exchange:
        contract = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        return qualified[0] if qualified else None

    details = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange, currency="USD"))
    return _nearest_active(details, datetime.now().strftime("%Y%m%d"))


TIMEFRAMES = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
}

# משך מקסימלי לקריאת reqHistoricalData בודדת אחת, לפי גודל הבר (מגבלת IB עצמה).
MAX_CHUNK_DURATION = {
    "1m": "1 W",
    "5m": "1 M",
    "15m": "1 M",
    "30m": "1 M",
    "1h": "1 M",
    "4h": "1 Y",
    "1d": "1 Y",
    "1w": "2 Y",
}

# תקרת ביטחון על מספר הקריאות הרצופות בשאיבה אחת. הועלתה כדי לאפשר 3 שנות היסטוריה
# גם בטיים-פריימים עדינים: IB מגביל כל בקשה לשבוע אחד בנרות דקה, כלומר ~157 קריאות
# לשלוש שנים. נמדד שיש ל-IB דאטה של דקה גם 3 שנים אחורה, אז המגבלה היא רק מספר
# הקריאות ולא זמינות הנתונים.
MAX_CHUNKS = 200

# IB חוסם יותר מ-60 בקשות היסטוריה ב-10 דקות לאותו חוזה. בשאיבות ארוכות עוצרים
# לפני שמגיעים לסף, אחרת הבקשות מתחילות להידחות ומתקבל דאטה חלקי בלי אזהרה.
PACING_CHUNKS = 50
PACING_SLEEP_SECONDS = 60

# בקשה בודדת יכולה לחזור ריקה גם כשיש עוד היסטוריה — timeout חולף או האטה זמנית של
# IB. בלי ניסיון חוזר, תשובה ריקה אחת סיימה את הדפדוף וההיסטוריה נקטעה בשקט באמצע
# (נמדד: 30m החזיר 1.2 שנים במקום 3 בגלל timeout יחיד).
CHUNK_RETRIES = 2
RETRY_SLEEP_SECONDS = 3

# ib_async.reqHistoricalData ברירת המחדל שלו timeout=60 שניות. IB Gateway עצמו
# (הפורט המקומי) יכול להישאר "מחובר" ולקבל בקשות גם כש-IB בשרתים שלהם באמצע
# תחזוקה שבועית (קורה בפועל בשבתות) - במצב כזה כל בקשה תלויה 60 שניות מלאות עד
# שנכשלת, וכפול CHUNK_RETRIES+1 ניסיונות זה מצטבר לכמעט 3 דקות לכל שאיבה אחת -
# בזמן הזה הדשבורד כולו נראה "לא עולה", כי callback הטעינה הראשונית של הגרף
# תקוע מחכה. timeout קצר משמעותית: אם IB לא עונה תוך כמה שניות זה כמעט תמיד
# סימן שהוא לא זמין כרגע, לא שהבקשה "כמעט מוכנה" - עדיף להיכשל מהר ולהיפול
# חזרה על הקאש הקיים מאשר לתקוע את כל האפליקציה.
HISTORICAL_DATA_TIMEOUT = 8

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

# אורך בר בשניות — משמש לקביעה כמה זמן הקאש נחשב "טרי".
_BAR_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
}


def _cache_is_fresh(path: Path, timeframe: str) -> bool:
    """האם הקאש נכתב לאחרונה מספיק כדי לוותר על רענון הנרות האחרונים מול IB.

    בלי זה כל פעולה בממשק (זום, שינוי תקופה, הרצה מחדש) שילמה קריאה נוספת ל-IB —
    ~1.5 שניות לנרות דקה — גם כשהנתונים נשאבו רגע קודם. החלון נגזר מאורך הבר, כך
    שנר חדש עדיין נתפס בזמן: 60 שניות לנרות דקה, ולכל היותר 5 דקות ליומי.
    """
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age < min(_BAR_SECONDS.get(timeframe, 60), 300)


def _cache_path(symbol: str, timeframe: str, use_rth: bool) -> Path:
    suffix = "rth" if use_rth else "ext"
    return CACHE_DIR / f"{symbol}_{timeframe}_{suffix}.parquet"


# קאש בזיכרון של קבצי ה-parquet עצמם, לפי mtime - קורא MNQ_1m_ext.parquet (מיליוני
# שורות, קובץ של עשרות MB) מהדיסק בכל בקשת LOD בודדת היה עולה מאות מ"ש לכל גלילה,
# גם כשהתוכן לא השתנה כלל בין קריאה לקריאה. שני process (dashboard.app וה-thread של
# live.py) קוראים במקביל, לכן לוק - אבל בלי להחזיק אותו בזמן read_parquet עצמו
# (איטי, ולא נוגע במבנה הנתונים המשותף), רק סביב הבדיקה/עדכון של המפתח בקאש.
_mem_cache_lock = threading.Lock()
_mem_cache: dict[tuple[str, str, bool], tuple[float, pd.DataFrame]] = {}


def _load_cache(symbol: str, timeframe: str, use_rth: bool) -> pd.DataFrame | None:
    path = _cache_path(symbol, timeframe, use_rth)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    key = (symbol, timeframe, use_rth)
    with _mem_cache_lock:
        cached = _mem_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    # ניסיון חוזר על כשל קריאה: תהליך אחר (בקשת LOD אחרת שרצה במקביל) עלול
    # בדיוק ברגע הזה להיות באמצע כתיבה אטומית (tmp + os.replace) לאותו קובץ.
    # os.replace עצמו אטומי, אבל pyarrow פותח/קורא את הקובץ בכמה שלבים - אם
    # ה-replace קורה בדיוק בין שלב לשלב, הקריאה יכולה להיכשל זמנית. בלי ניסיון
    # חוזר, כישלון בודד כזה היה מתפרש כ"אין קאש בכלל" - זה בדיוק מה שגרם ל-
    # fetch_ohlcv לחשוב שאין קאש עמוק, לשלוף רק זנב קטן מ-IB, ולהחזיר את זה
    # לקורא (גם אם _save_cache חסם את השמירה של הזנב הקטן - התשובה לבקשה הזו
    # עצמה כבר יצאה שגויה/מכווצת).
    df = None
    for attempt in range(3):
        try:
            df = pd.read_parquet(path)
            break
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.05)
    if df is None or df.empty:
        return None

    with _mem_cache_lock:
        _mem_cache[key] = (mtime, df)
    return df


def _head_exhausted_path(symbol: str, timeframe: str, use_rth: bool) -> Path:
    return _cache_path(symbol, timeframe, use_rth).with_suffix(".headdone")


def _is_head_exhausted(symbol: str, timeframe: str, use_rth: bool) -> bool:
    return _head_exhausted_path(symbol, timeframe, use_rth).exists()


def _mark_head_exhausted(symbol: str, timeframe: str, use_rth: bool) -> None:
    # קובץ-סמן ריק: ברגע שקריאה אחת כבר ניסתה להאריך אחורה ולא קיבלה שום דבר
    # ישן יותר ממה שכבר בקאש, אין טעם לנסות שוב בכל בקשת "עוד היסטוריה" עתידית -
    # זו לא מגבלת pacing זמנית אלא תקרה אמיתית (IB לא מחזיק/לא היה מכשיר בכלל).
    try:
        _head_exhausted_path(symbol, timeframe, use_rth).touch()
    except Exception:
        pass


def _save_cache(symbol: str, timeframe: str, use_rth: bool, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, timeframe, use_rth)

    # מגן-כיווץ: קרה בפועל פעמיים היום ש-MNQ_1m_ext.parquet (7 שנות היסטוריה,
    # ~2.58M שורות מ-Databento) נדרס בתוצאה של כמה ימים בלבד - כנראה מירוץ שבו
    # _load_cache קרא None/מעט באותו רגע (או תהליך אחר כתב בו-זמנית), ואז fetch_ohlcv
    # התייחס לזה כ"אין קאש בכלל" ושמר רק את השאיבה הטרייה הקטנה מעליו. לא ניתן
    # לדעת כאן אם זה קורה כרגע - אבל לעולם לא הגיוני שקאש קיים "יתכווץ" באופן
    # שמאבד היסטוריה ישנה יותר משהיה בו; זה בטוח סימן לבאג/מירוץ, לא לעדכון לגיטימי.
    try:
        existing = pd.read_parquet(path) if path.exists() else None
    except Exception:
        existing = None
    if existing is not None and not existing.empty and not df.empty:
        lost_head = df.index.min() > existing.index.min() + pd.Timedelta(days=1)
        much_smaller = len(df) < len(existing) * 0.5
        if lost_head and much_smaller:
            print(f"[cache] מדלג על שמירה שהייתה מכווצת את הקאש של {symbol}/{timeframe}: "
                  f"קיים {len(existing):,} ברים ({existing.index.min()} ואילך), "
                  f"חדש רק {len(df):,} ברים ({df.index.min()} ואילך)")
            return
    # כתיבה אטומית: קודם לקובץ זמני, ואז החלפה (os.replace אטומי ב-POSIX) - לא
    # ישירות ליעד. בלי זה, הרג התהליך (kill -9) באמצע df.to_parquet היה משאיר
    # קובץ קאש חלקי/פגום; הקריאה הבאה (_load_cache) הייתה נכשלת בשקט וחוזרת
    # None, וה-fetch הבא היה "לא מוצא קאש בכלל" ודורס את כל הקאש הקיים (כולל
    # היסטוריה עמוקה שנבנתה בנפרד, כמו MNQ) בתוצאה קטנה בהרבה. זה בדיוק מה
    # שקרה בפועל היום אחרי כמה הפעלות-מחדש של השרת.
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        # קאש הוא אופטימיזציה בלבד — כשל בשמירה לא אמור להפיל את השאיבה


def _fetch_chunks(
    ib: IB,
    contract: Stock,
    timeframe: str,
    use_rth: bool,
    end_dt,
    cutoff: pd.Timestamp,
    max_chunks: int = MAX_CHUNKS,
    progress=None,
) -> list[pd.DataFrame]:
    """מדפדף אחורה מ-end_dt עד cutoff (או עד שה-IB מפסיק להחזיר דאטה), ומחזיר
    רשימת DataFrame-ים גולמיים (בפורמט שמחזיר ib_async, לפני נרמול עמודות)."""
    chunk_duration = MAX_CHUNK_DURATION[timeframe]
    chunks: list[pd.DataFrame] = []
    for _i in range(max_chunks):
        if _i and _i % PACING_CHUNKS == 0:
            # הפוגה יזומה כדי להישאר מתחת למגבלת ה-pacing של IB.
            if progress is not None:
                progress(len(chunks), paused=True)
            time.sleep(PACING_SLEEP_SECONDS)
        if progress is not None:
            progress(len(chunks) + 1)

        bars = None
        for attempt in range(CHUNK_RETRIES + 1):
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=chunk_duration,
                barSizeSetting=TIMEFRAMES[timeframe],
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                timeout=HISTORICAL_DATA_TIMEOUT,
            )
            if bars or attempt == CHUNK_RETRIES:
                break
            time.sleep(RETRY_SLEEP_SECONDS)

        # רק אחרי שכל הניסיונות חזרו ריקים מסיקים שנגמרה ההיסטוריה.
        if not bars:
            break

        chunk_df = util.df(bars)
        chunks.append(chunk_df)

        earliest = pd.Timestamp(chunk_df["date"].iloc[0])
        if earliest.tzinfo is not None:
            earliest = earliest.tz_localize(None)
        if earliest <= cutoff:
            break
        end_dt = (earliest - timedelta(seconds=1)).to_pydatetime()
    return chunks


def _futures_chain(ib: IB, symbol: str, exchange: str):
    """כל חוזי המשנה ההיסטוריים (כולל פקועים) של הסימבול, ממוינים מהישן לחדש.

    IB מציג חוזים חדשים חודשים-שנים מראש (למשל MNQ לספטמבר 2027), הרבה לפני
    שיש להם בכלל נתוני מסחר — reqContractDetails מחזיר אותם לצד הפעילים. מסננים
    כל מה שפוקע בעוד יותר מ-100 יום מהיום: זה עדיין תופס את החוזה הקרוב הנוכחי
    (שכבר נסחר גם אם עוד לא פקע), אבל חוסך המון קריאות ריקות לחוזי "רפאים"
    עתידיים-מדי בתחילת הדפדוף אחורה.
    """
    details = ib.reqContractDetails(Future(symbol, exchange=exchange, currency="USD", includeExpired=True))
    by_id = {d.contract.conId: d.contract for d in details}
    horizon = (datetime.now() + timedelta(days=100)).strftime("%Y%m%d")
    contracts = [c for c in by_id.values() if c.lastTradeDateOrContractMonth <= horizon]
    return sorted(contracts, key=lambda c: c.lastTradeDateOrContractMonth)


def _fetch_futures_history(
    ib: IB,
    symbol: str,
    exchange: str,
    timeframe: str,
    use_rth: bool,
    end_dt,
    cutoff: pd.Timestamp,
    max_chunks: int = MAX_CHUNKS,
    progress=None,
) -> list[pd.DataFrame]:
    """כמו _fetch_chunks, אבל לחוזה עתידי: IB אוסר endDateTime לא-ריק על ContFuture
    (שגיאה 10339 — "Setting end date/time for continuous future security type is
    not allowed"), כך שאי אפשר לדפדף אחורה דרכו מעבר לקריאה הראשונה — בפועל זה
    היה חותך את ההיסטוריה אחרי שבוע אחד בלבד בנרות דקה. הפתרון: מדפדפים ידנית על
    פני שרשרת חוזי המשנה ההיסטוריים (כולל פקועים) עצמם, מהקרוב ביותר אחורה. כשחוזה
    "נגמר" (reqHistoricalData מחזיר ריק) עוברים לחוזה הקודם באותה נקודת זמן בדיוק
    שבה עצרנו — כך שהרצף נשמר בלי לחשב תאריכי roll במפורש. הנתונים אינם
    roll-adjusted: כל קטע הוא המחיר הנקוב בפועל של אותו חוזה קונקרטי באותו זמן.
    """
    chain = _futures_chain(ib, symbol, exchange)
    if not chain:
        return []

    chunk_duration = MAX_CHUNK_DURATION[timeframe]
    chunks: list[pd.DataFrame] = []
    cursor = end_dt
    used = 0
    # "בר ריק" מ-IB פירושו גם "החוזה הזה נגמר, תעבור לקודם" (מצב תקין, מצפים
    # לזה בסוף ההיסטוריה) וגם "IB לא עונה כרגע בכלל" (תחזוקה שבועית, ניתוק) -
    # מ-reqHistoricalData בלבד אי אפשר להבחין ביניהם, שני המצבים חוזרים ריקים
    # באותה צורה בדיוק. בלי בלימה כאן, תקלת IB אמיתית הייתה גורמת לדפדוף להמשיך
    # "לנסות את החוזה הקודם" שוב ושוב על פני כל השרשרת (עשרות חוזים, שנים אחורה) -
    # נמדד בפועל: כמעט 15 דקות תלייה מוחלטת של הדשבורד בזמן תחזוקה שבועית של IB.
    # שני חוזים רצופים שחוזרים ריקים לגמרי (0 ברים, לא רק "פחות מהמצופה") הוא
    # כבר חשד סביר מספיק שמדובר בבעיית זמינות, לא בסוף היסטוריה אמיתי - עוצרים
    # לגמרי ומחזירים מה שכבר נאסף (fetch_ohlcv יפול חזרה על הקאש הקיים).
    consecutive_empty_contracts = 0

    for contract in reversed(chain):
        if used >= max_chunks or cursor is None or consecutive_empty_contracts >= 2:
            break
        got_any_bar_this_contract = False
        while used < max_chunks:
            if used and used % PACING_CHUNKS == 0:
                if progress is not None:
                    progress(len(chunks), paused=True)
                time.sleep(PACING_SLEEP_SECONDS)
            if progress is not None:
                progress(len(chunks) + 1)

            bars = None
            for attempt in range(CHUNK_RETRIES + 1):
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=cursor,
                    durationStr=chunk_duration,
                    barSizeSetting=TIMEFRAMES[timeframe],
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=1,
                    timeout=HISTORICAL_DATA_TIMEOUT,
                )
                if bars or attempt == CHUNK_RETRIES:
                    break
                time.sleep(RETRY_SLEEP_SECONDS)

            if not bars:
                break  # החוזה הזה מוצה (או שIB לא ענה) — עוברים לחוזה הקודם באותו cursor

            got_any_bar_this_contract = True
            chunk_df = util.df(bars)
            chunks.append(chunk_df)
            used += 1

            earliest = pd.Timestamp(chunk_df["date"].iloc[0])
            if earliest.tzinfo is not None:
                earliest = earliest.tz_localize(None)
            cursor = (earliest - timedelta(seconds=1)).to_pydatetime()
            if earliest <= cutoff:
                cursor = None
                break

        consecutive_empty_contracts = 0 if got_any_bar_this_contract else consecutive_empty_contracts + 1
    return chunks


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.drop_duplicates(subset="Date").set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def fetch_ohlcv(
    symbol: str,
    days: int,
    timeframe: str = "1m",
    host: str = "127.0.0.1",
    port: int = 4002,
    use_rth: bool = True,
    end_date: date | datetime | None = None,
    force_refresh: bool = False,
    progress=None,
) -> pd.DataFrame:
    """מתחבר ל-TWS ומחזיר DataFrame של נרות עבור `symbol` בטיים-פריים הנבחר.

    מנסה להגיע ל-`days` ימים אחורה (מ-`end_date`, או מעכשיו אם לא צוין) על ידי
    דפדוף אחורה בכמה קריאות, אך בפועל מוגבל למה שה-IB בעצמו שומר בהיסטוריה שלו
    לגודל הבר הזה. `end_date` מאפשר לצפות בחלון היסטורי ספציפי בעבר (לא רק
    "N ימים אחרונים עד היום") — למשל לבדוק אסטרטגיה על תקופה מלפני שנתיים.

    שולף מה-IB בפועל רק את מה שבאמת חסר מול הקאש המקומי (data_cache/) לאותו
    סימבול/טיים-פריים/RTH: ב"live" (בלי end_date) תמיד מרעננים את הנתונים
    האחרונים כדי לתפוס נרות חדשים, וב-both מצבים נשלף רק הטווח ההיסטורי הישן
    שעדיין חסר מעבר למה שכבר נשמר. אם החלון המבוקש כבר מכוסה במלואו — מוחזר
    ישירות מהקאש בלי להתחבר ל-IB בכלל.
    """
    days = max(1, int(days))
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("יש להזין סימבול")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"טיים-פריים לא נתמך: {timeframe}")

    live = end_date is None
    anchor = pd.Timestamp.now()
    if not live:
        anchor = pd.Timestamp(end_date)
        if anchor.tzinfo is not None:
            anchor = anchor.tz_localize(None)
        # אם ניתן רק תאריך (ללא שעה), מבקשים עד סוף אותו היום כדי לכלול את כל הנרות שלו
        if anchor.time() == pd.Timestamp(0).time():
            anchor = anchor + pd.Timedelta(hours=23, minutes=59, seconds=59)

    cutoff = anchor - pd.Timedelta(days=days)
    cached = _load_cache(symbol, timeframe, use_rth)
    cached_min = cached.index.min() if cached is not None else None
    cached_max = cached.index.max() if cached is not None else None

    need_head = cached is None or (
        cutoff < cached_min and not _is_head_exhausted(symbol, timeframe, use_rth)
    )
    if cached is None:
        need_tail = True
    elif live:
        # ברירת המחדל היא לרענן כדי לתפוס נרות חדשים, אבל אם הקאש נכתב הרגע אין מה
        # לחדש — וזה חוסך קריאה מיותרת ל-IB בכל פעולה בממשק. ⟳ מדלג על החיסכון הזה.
        need_tail = force_refresh or not _cache_is_fresh(
            _cache_path(symbol, timeframe, use_rth), timeframe
        )
    else:
        need_tail = anchor > cached_max  # בחלון היסטורי קבוע, רק אם באמת חסר משהו

    if cached is not None and not need_head and not need_tail:
        # החלון המבוקש מכוסה במלואו על ידי הקאש — התשובה חוזרת מיד בלי להתחבר ל-IB.
        return cached[(cached.index >= cutoff) & (cached.index <= anchor)]

    ib = IB()
    client_id = random.randint(1000, 9999)
    try:
        # readonly + fetchFields(0): מדלגים על סנכרון פוזיציות/הזמנות/executions
        # שברירת המחדל של ib_async מבצעת בכל connect - האפליקציה הזו אף פעם לא
        # שולחת הזמנות, רק שואבת דאטה היסטורי. בלי זה, ב-IB Gateway עמוס/איטי
        # ה-connect "מצליח" אבל נתקע דקה+ מחכה ל-5 בקשות sync חשבון שלא צריך
        # בכלל (positions/open orders/completed orders/account updates/executions)
        # - בדיוק מה שגרם לטעינה "לא לעלות בכלל".
        ib.connect(host, port, clientId=client_id, timeout=10,
                   readonly=True, fetchFields=StartupFetch(0))
    except Exception as exc:
        raise ConnectionError(
            f"לא ניתן להתחבר ל-TWS בכתובת {host}:{port}. "
            f"ודא ש-TWS פתוח, מחובר, ושה-API מופעל (Enable ActiveX and Socket Clients). "
            f"שגיאה: {exc}"
        ) from exc

    try:
        futures_exchange = FUTURES_EXCHANGES.get(symbol)
        if futures_exchange:
            # ולידציה מוקדמת שהסימבול קיים בכלל — השאיבה עצמה עוברת ב-_fetch_futures_history.
            if not ib.reqContractDetails(Future(symbol, exchange=futures_exchange, currency="USD")):
                raise ValueError(f"הסימבול '{symbol}' לא נמצא/לא תקין")

            def _fetch(end_dt_, cutoff_):
                return _fetch_futures_history(
                    ib, symbol, futures_exchange, timeframe, use_rth, end_dt_, cutoff_, progress=progress
                )
        else:
            contract = Stock(symbol, "SMART", "USD")
            if not ib.qualifyContracts(contract):
                raise ValueError(f"הסימבול '{symbol}' לא נמצא/לא תקין")

            def _fetch(end_dt_, cutoff_):
                return _fetch_chunks(ib, contract, timeframe, use_rth, end_dt_, cutoff_, progress=progress)

        new_chunks: list[pd.DataFrame] = []
        head_chunks: list[pd.DataFrame] = []
        if cached is None:
            end_dt = "" if live else anchor.to_pydatetime()
            new_chunks = _fetch(end_dt, cutoff)
        else:
            if need_tail:
                tail_end = "" if live else anchor.to_pydatetime()
                new_chunks += _fetch(tail_end, cached_max)
            if need_head:
                head_end = (cached_min - timedelta(seconds=1)).to_pydatetime()
                head_chunks = _fetch(head_end, cutoff)
                new_chunks += head_chunks

        if not new_chunks and cached is None:
            raise ValueError(f"לא התקבל דאטה מ-IB עבור {symbol}. ודא שיש הרשאת שוק חיה/מתעכבת לסימבול זה.")
    finally:
        ib.disconnect()

    parts = [cached] if cached is not None else []
    parts += [_normalize(c) for c in new_chunks]
    merged = pd.concat(parts) if parts else pd.DataFrame()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if merged.empty:
        raise ValueError(f"לא התקבל דאטה מ-IB עבור {symbol}. ודא שיש הרשאת שוק חיה/מתעכבת לסימבול זה.")

    # ניסינו להאריך אחורה (need_head) ולא קיבלנו שום דבר ישן יותר ממה שכבר היה -
    # לא pacing זמני, זו התקרה האמיתית (IB לא שומר/לא היה מכשיר). מסמנים כדי
    # שכל בקשת "עוד היסטוריה" הבאה תעצור מיד במקום לנסות שוב מול IB בכל פעם.
    # head_chunks (לא new_chunks) בכוונה: אם need_head רץ אבל בלימת-המעגל
    # ב-_fetch_futures_history עצרה מיד בגלל IB לא זמין (לא בגלל שהגענו באמת
    # לתחילת ההיסטוריה), head_chunks יוצא ריק - וזה בדיוק המקרה שבו *אסור*
    # לסמן "מוצה", אחרת תקלת IB חד-פעמית הייתה חוסמת לצמיתות כל ניסיון עתידי
    # להאריך אחורה גם אחרי ש-IB חוזר לפעול.
    if need_head and head_chunks and cached_min is not None and merged.index.min() >= cached_min:
        _mark_head_exhausted(symbol, timeframe, use_rth)

    _save_cache(symbol, timeframe, use_rth, merged)

    # לנרות יומיים חותמת הזמן היא תמיד חצות (00:00), בעוד ש-cutoff/anchor נושאים
    # את שעת ה"עכשיו" בפועל — השוואה ישירה הייתה חותכת כמעט את כל הימים הרלוונטיים.
    # לכן מיישרים את שניהם לגבולות היממה במקרה הזה בלבד.
    trim_start, trim_end = cutoff, anchor
    if timeframe in ("1d", "1w"):
        trim_start = cutoff.normalize()
        trim_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

    return merged[(merged.index >= trim_start) & (merged.index <= trim_end)]
