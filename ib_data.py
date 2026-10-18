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

import asyncio
import os
import random
import threading
import time
import uuid
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
    """גיבוי כשל-OI: החוזה הלא-פקוע הקרוב ביותר לפי תאריך פקיעה בלבד. פשוט
    ואמין, אבל יכול לפגר אחרי המעבר-בפועל בנזילות בימים שלפני/סביב פקיעה
    ממש (ה-front month האמיתי לפעמים כבר מתחלף כמה ימים לפני הפקיעה
    הקלנדרית) - ראו _pick_by_open_interest, שמנסה קודם."""
    if not details:
        return None
    contracts = sorted((d.contract for d in details), key=lambda c: c.lastTradeDateOrContractMonth)
    for c in contracts:
        if c.lastTradeDateOrContractMonth >= today:
            return c
    return contracts[-1]


# Generic tick 588 = Futures Open Interest (ib_async/TWS API) - אומת ישירות
# מול IB Gateway: ticker.futuresOpenInterest מתמלא רק עם ה-generic tick הזה
# מבוקש במפורש (reqMktData בלי genericTickList מחזיר NaN עבורו תמיד).
_OI_GENERIC_TICK = "588"
# זמן המתנה לתשובת ה-OI מ-IB אחרי reqMktData - זה snapshot לא-חוסם (הטיקר
# מתמלא באופן אסינכרוני ברקע), אז חייבים לתת לו רגע להגיע לפני שקוראים את
# הערך. נמדד מספיק בפועל; לא צריך יותר מזה לשני חוזים בלבד.
_OI_WAIT_SECONDS = 3.0


def _pick_by_open_interest(ib: IB, candidates: list) -> object | None:
    """בוחרת מבין candidates (רשימת Contract, כבר qualified) את זה עם ה-Open
    Interest הגבוה ביותר כרגע - ההגדרה התקנית ל"חודש-חזית"/החוזה הכי נזיל,
    אמינה יותר מ"הקרוב ביותר לפי תאריך פקיעה" בסביבות roll (נפח/OI יכולים
    לעבור לחוזה הבא כמה ימים לפני הפקיעה הקלנדרית עצמה). מחזירה None אם לא
    התקבל OI תקין מאף candidate (למשל אין הרשאת market-data, או candidates
    ריק) - הקורא אז חוזר ל-_nearest_active כגיבוי, לא נופל.
    """
    if not candidates:
        return None
    tickers = []
    try:
        tickers = [ib.reqMktData(c, genericTickList=_OI_GENERIC_TICK, snapshot=False) for c in candidates]
        ib.sleep(_OI_WAIT_SECONDS)
    finally:
        for c in candidates:
            try:
                ib.cancelMktData(c)
            except Exception:
                pass
    best, best_oi = None, -1.0
    for t in tickers:
        oi = t.futuresOpenInterest
        if oi is not None and oi == oi and oi > best_oi:  # oi == oi: מסנן NaN
            best_oi, best = oi, t.contract
    return best


async def _pick_by_open_interest_async(ib: IB, candidates: list) -> object | None:
    """כמו _pick_by_open_interest, אבל async (ib.sleep חוסם את ה-event loop -
    לא בטוח לקרוא לו מתוך תת-שרשור עם loop פעיל, ראו resolve_live_contract_async)."""
    if not candidates:
        return None
    tickers = []
    try:
        tickers = [ib.reqMktData(c, genericTickList=_OI_GENERIC_TICK, snapshot=False) for c in candidates]
        await asyncio.sleep(_OI_WAIT_SECONDS)
    finally:
        for c in candidates:
            try:
                ib.cancelMktData(c)
            except Exception:
                pass
    best, best_oi = None, -1.0
    for t in tickers:
        oi = t.futuresOpenInterest
        if oi is not None and oi == oi and oi > best_oi:
            best_oi, best = oi, t.contract
    return best


def _recent_volume(ib: IB, contract, minutes: int = 15) -> float:
    """נפח-מסחר (TRADES) בכמה הדקות האחרונות לחוזה נתון - גיבוי אמין ל-OI:
    reqMktData(genericTickList=OI) הוא subscription א-סינכרוני שמתמלא ברקע,
    ונמצא בפועל לא-אמין תחת עומס בו-זמני מכמה תהליכים שחולקים את אותו IB
    Gateway (נצפה ישירות: באחת הריצות ה-OI "נכשל בשקט" וה-fallback הישן חזר
    להתנהגות השגויה - בחר MNQZ6 שוב במקום MNQU6). reqHistoricalData לעומת
    זאת היא בקשת request/response רגילה, אותו דפוס בדיוק שכל שאר הקובץ הזה
    כבר סומך עליו באמינות - נפח-מסחר גבוה בדקות האחרונות הוא אינדיקציה
    ישירה מספיק ל"זה החוזה שבאמת נסחר עכשיו". מחזירה 0.0 בכל כשל (כולל
    חוזה שלא באמת נסחר עדיין).
    """
    try:
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=f"{minutes * 60} S",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
            formatDate=1, timeout=10,
        )
    except Exception:
        return 0.0
    if not bars:
        return 0.0
    return float(sum(b.volume for b in bars if b.volume == b.volume))  # b.volume == b.volume: מסנן NaN


async def _recent_volume_async(ib: IB, contract, minutes: int = 15) -> float:
    """כמו _recent_volume, אבל async."""
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr=f"{minutes * 60} S",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
            formatDate=1, timeout=10,
        )
    except Exception:
        return 0.0
    if not bars:
        return 0.0
    return float(sum(b.volume for b in bars if b.volume == b.volume))


def _pick_most_liquid(ib: IB, candidates: list) -> object | None:
    """בוחרת את ה-front month האמיתי מבין candidates (עד 2, הקרוב ביותר קודם -
    ראו הקוראים: תמיד ממוינים [קרוב, רחוק]). מנסה קודם Open Interest חי
    (_pick_by_open_interest) - ההגדרה התקנית ביותר, אבל מחזיר תמיד NaN
    בחשבון בלי הרשאת live-data מלאה (Error 354 - "not subscribed", רק
    delayed data זמין) - המצב בפועל כאן.

    כשאין OI: *לא* בוחרים לפי "מי מהשניים הכי נסחר עכשיו" (נמדד כלא-יציב
    בפועל - ראו התקלה למטה) - אלא מעדיפים תמיד את החוזה הקרוב-יותר
    בלוח-שנה, כל עוד יש לו בכלל מסחר (recent-volume > 0), ועוברים לחוזה
    הבא בשרשרת רק כשהקרוב-יותר כבר "נגמר" (נפח=0). זו המוסכמה התקנית:
    front month משמש ברציפות עד ממש הפקיעה (או חלון roll קצר מאוד ליד
    הפקיעה עצמה) - לא מוחלף מוקדם יותר סתם כי החוזה הבא הראה נפח גבוה
    רגעית בחלון-מדידה קצר ורועש (15 דקות, ראו _recent_volume).

    תקלה שנצפתה בפועל (2026-09-15, 3 ימים לפני פקיעת MNQU6): הבחירה
    הישנה ("מי מהשניים עם הנפח הגבוה יותר") קפצה ל-MNQZ6 (חוזה דצמבר,
    רחוק) כי מדידת-15-דקות נקודתית הראתה לו נפח גבוה יותר (למשל 2120 מול
    975) בזמן ש-MNQU6 עדיין נסחר לגמרי כרגיל - זה גרם למעבר-חוזה באמצע
    הסשן בסטרימינג החי, עם קפיצת-מחיר של כ-300 נקודה בין שני החוזים
    (MNQZ6 לא roll-adjusted מול MNQU6) - בדיוק ה"פער/קפיצה בגרף וגם
    הנר החי מפסיק להתעדכן" שדווח בפועל (MNQZ6 גם הראה זרם-לייב לא-יציב
    בחשבון הזה, לא רק מחיר שונה).
    """
    by_oi = _pick_by_open_interest(ib, candidates)
    if by_oi is not None:
        return by_oi
    for c in candidates:
        if _recent_volume(ib, c) > 0:
            return c
    return None


async def _pick_most_liquid_async(ib: IB, candidates: list) -> object | None:
    """כמו _pick_most_liquid, אבל async - ראו שם להסבר המלא."""
    by_oi = await _pick_by_open_interest_async(ib, candidates)
    if by_oi is not None:
        return by_oi
    for c in candidates:
        if await _recent_volume_async(ib, c) > 0:
            return c
    return None


def resolve_live_contract(ib: IB, symbol: str):
    """חוזה למסחר/סטרימינג חי (סינכרוני — לשימוש מחוץ ל-event loop פעיל): לחוזה
    עתידי בוחרים לפי הנזילות בפועל בין שני החוזים הקרובים ביותר (ראו
    _pick_most_liquid - OI קודם, נפח כגיבוי - זה ה"front month" האמיתי, לא
    בהכרח הקרוב ביותר בלוח-שנה), עם גיבוי סופי ל-_nearest_active אם שניהם
    נכשלו - כך שרולאובר (כמו MNQU6 -> MNQZ6) מזוהה ומטופל אוטומטית בכל
    קריאה, בלי תלות בתאריך פקיעה מקודד. מניה כרגיל (אין רולאובר)."""
    symbol = symbol.strip().upper()
    exchange = FUTURES_EXCHANGES.get(symbol)
    if not exchange:
        contract = Stock(symbol, "SMART", "USD")
        return contract if ib.qualifyContracts(contract) else None

    details = ib.reqContractDetails(Future(symbol, exchange=exchange, currency="USD"))
    if not details:
        return None
    contracts = sorted((d.contract for d in details), key=lambda c: c.lastTradeDateOrContractMonth)
    liquid = _pick_most_liquid(ib, contracts[:2])
    return liquid if liquid is not None else _nearest_active(details, datetime.now().strftime("%Y%m%d"))


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
    if not details:
        return None
    contracts = sorted((d.contract for d in details), key=lambda c: c.lastTradeDateOrContractMonth)
    liquid = await _pick_most_liquid_async(ib, contracts[:2])
    return liquid if liquid is not None else _nearest_active(details, datetime.now().strftime("%Y%m%d"))


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

# הפרש-מחיר מקסימלי (נקודות) בין שני ברים באותה חותמת-זמן בדיוק (אחד כבר
# בקאש, אחד משאיבה חדשה) לפני שנחשד בערבוב-חוזים (ראו fetch_ohlcv) - בר
# היסטורי סגור לא אמור "להשתנות" בין שאיבות כלל, אז כל הפרש משמעותי הוא כבר
# חשוד; 100 נק' (כפי שהתבקש במפורש) נותן מרווח-ביטחון סביר מעל רעש נורמלי.
CONTRACT_MISMATCH_THRESHOLD_PTS = 100.0

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

    # מגן-כיווץ: קרה בפועל כמה פעמים ש-MNQ_1m_ext.parquet (7 שנות היסטוריה,
    # ~2.5M שורות מ-Databento) נדרס בתוצאה של כמה ימים בלבד - מירוץ שבו
    # _load_cache (בקריאה של fetch_ohlcv, למעלה) קרא None באותו רגע, ואז
    # fetch_ohlcv התייחס לזה כ"אין קאש בכלל" ושמר רק את השאיבה הטרייה הקטנה
    # מעליו. אין שום קורא אחר ל-_save_cache עם דאטה קטן יותר בכוונה (הפונקציה
    # היחידה שקוראת לזה היא fetch_ohlcv עצמה, למטה) - אז אין תרחיש לגיטימי
    # שבו קאש קיים אמור "להתכווץ": אם len(df) < len(existing) בכלל, זה תמיד
    # סימן לבאג/מירוץ, לא עדכון תקין. לכן החסימה כאן פשוט: כל כיווץ, לא רק
    # "חצי מהגודל וגם אבד את ההתחלה" - התנאי הכפול הקודם פספס בדיוק את
    # התרחיש שקרה בפועל (ראו למטה, למה). "not df.empty" בכוונה לא חלק מהתנאי
    # (בניגוד לגרסה קודמת) - שמירה עם df ריק לגמרי חייבת להיחסם גם היא, אין
    # תרחיש לגיטימי שבו fetch_ohlcv מגיע לכאן עם df ריק כשכבר יש קאש עם דאטה.
    #
    # התיקון המרכזי כאן, אחרי חקירה: הקריאה המגוננת הזו עצמה (existing =
    # read_parquet(path)) לא ניסתה שוב בכשל, בניגוד ל-_load_cache שדווקא כן
    # מנסה שוב 3 פעמים בדיוק בגלל התקלה החולפת הזו (ראו התיעוד שם) - אז אם
    # אותה תקלת-קריאה חולפת שגרמה ל-_load_cache למעלה להחזיר None (ומכאן
    # לכל השרשרת שהובילה לשאיבה הקטנה) עדיין לא חלפה עד לרגע הזה, גם הקריאה
    # המגוננת כאן נכשלה בשקט ל-None - ואז "existing is not None" יצא False,
    # והמגן פשוט לא בדק כלום. באג במגן עצמו, לא רק תנאי-הכיווץ. עכשיו: אותו
    # ניסיון-חוזר בדיוק כמו _load_cache, ואם הקובץ קיים על הדיסק אבל אי-אפשר
    # לקרוא אותו בכלל אחרי 3 ניסיונות - זה fail-*safe*, לא fail-open: חוסמים
    # את השמירה (עדיף לפספס עדכון לגיטימי אחד מאשר למחוק בטעות שבע שנות דאטה).
    existing = None
    if path.exists():
        for attempt in range(3):
            try:
                existing = pd.read_parquet(path)
                break
            except Exception:
                if attempt == 2:
                    print(f"[cache] אזהרה: לא הצלחתי לקרוא את קאש {symbol}/{timeframe} הקיים "
                          f"כדי לוודא שהשמירה לא מכווצת אותו - מדלג על השמירה למקרה שהוא "
                          f"מכיל היסטוריה עמוקה יותר (בטוח יותר לפספס עדכון מאשר למחוק דאטה).", flush=True)
                    return
                time.sleep(0.05)
    if existing is not None and not existing.empty and len(df) < len(existing):
        print(f"[cache] מדלג על שמירה שהייתה מכווצת את הקאש של {symbol}/{timeframe}: "
              f"קיים {len(existing):,} ברים ({existing.index.min()} ואילך), "
              f"חדש רק {len(df):,} ברים "
              f"({df.index.min() if len(df) else 'ריק'} ואילך)", flush=True)
        return
    # כתיבה אטומית: קודם לקובץ זמני, ואז החלפה (os.replace אטומי ב-POSIX) - לא
    # ישירות ליעד. בלי זה, הרג התהליך (kill -9) באמצע df.to_parquet היה משאיר
    # קובץ קאש חלקי/פגום; הקריאה הבאה (_load_cache) הייתה נכשלת בשקט וחוזרת
    # None, וה-fetch הבא היה "לא מוצא קאש בכלל" ודורס את כל הקאש הקיים (כולל
    # היסטוריה עמוקה שנבנתה בנפרד, כמו MNQ) בתוצאה קטנה בהרבה.
    #
    # שם הקובץ הזמני חייב להיות ייחודי *לכל קריאה*, לא רק לכל תהליך: שרת ה-dev
    # של Flask/Dash רץ עם threaded=True כברירת מחדל (Flask.run: options.
    # setdefault("threaded", True)), כלומר כל בקשה (טעינת עמוד, טיק של
    # live-interval, מעבר סימבול/טיים-פריים) מטופלת ב-thread נפרד *באותו
    # תהליך*. os.getpid() לבדו זהה לכל ה-threads האלה - אז שני threads
    # שקוראים ל-_save_cache לאותו symbol/timeframe כמעט-בו-זמנית (למשל טעינת
    # עמוד וטיק live חופפים) היו מחשבים את *אותו* tmp_path בדיוק, וכותבים
    # אליו *בו-זמנית* משני threads - קובץ זמני קרוע/מעורבב, ואז os.replace
    # מעביר את הזבל הזה ליעד. זה בדיוק מה שגרם לשחיתות (לא רק לכיווץ) שראינו
    # אחרי שמגן-הכיווץ כבר טופל: שתי כתיבות "לגיטימיות" (אף אחת מהן לא קטנה
    # מהקיים) עדיין יכלו להתנגש זו בזו על אותו tmp_path. threading.get_ident()
    # + טוקן אקראי הופכים את tmp_path לייחודי per-call, כך שכל קריאה כותבת
    # לקובץ זמני נפרד משלה - ואז os.replace האטומי מבטיח ש"האחרון מנצח" בלי
    # שום אפשרות לתערובת בין שתי כתיבות.
    tmp_path = path.with_suffix(
        path.suffix + f".tmp{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
    )
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

        earliest_raw = pd.Timestamp(chunk_df["date"].iloc[0])
        if earliest_raw.tzinfo is not None:
            # tz_convert קודם, לא רק tz_localize(None): IB מחזיר את חותמות-הזמן
            # tz-aware לפי אזור-הזמן של הבורסה - עבור MNQ/CME זה בפועל
            # US/Central, לא US/Eastern (אומת ישירות מול IB: bar.date חוזר עם
            # tzinfo=ZoneInfo('US/Central')) - למרות ש-SOURCE_TZ בכל האפליקציה
            # (dashboard/data.py) מניח America/New_York.
            earliest_et = earliest_raw.tz_convert("America/New_York")
        else:
            earliest_et = earliest_raw.tz_localize("America/New_York")
        if earliest_et.tz_localize(None) <= cutoff:
            break
        # end_dt נשאר tz-aware (America/New_York) - לא naive: אומת ישירות
        # מול IB ש-endDateTime כ-naive datetime מתפרש לפי אזור-הזמן *המקומי
        # של המכונה שמריצה את הקוד* (על המכונה הזו: Asia/Jerusalem), לא
        # "כאילו ET". cursor naive שמייצג שעון ניו-יורק (למשל 18:00) היה
        # נשלח בחזרה ל-IB ומתפרש כ-18:00 שעון ישראל - בקשת-העמוד הבאה
        # הייתה מבקשת חלון בהפרש של שעות ממה שבאמת רצינו, ומייצרת בדיוק את
        # סוג הפערים/החפיפות שהובילו לערבוב-חוזים שנצפה בפועל. tz-aware
        # מסלק את האי-בהירות הזו לגמרי, בלי תלות באזור-הזמן של המכונה.
        end_dt = (earliest_et - timedelta(seconds=1)).to_pydatetime()
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

    # לבקשה "חיה" (end_dt=="" - זנב עד עכשיו, ראו fetch_ohlcv) חייבים להתחיל
    # מה-front month האמיתי (הכי הרבה Open Interest), לא סתם מהחוזה החדש
    # ביותר שנמצא בטווח 100 הימים של _futures_chain: חוזה מאוחר יותר (כמו
    # MNQZ6 לפני שהוא באמת הפך ל-front month) יכול כבר להחזיר דאטה (גם בנפח
    # דליל-יחסית) בלי להיות "ריק" - ואז הדפדוף מהחדש-ביותר-בשרשרת היה נתקע
    # עליו ולעולם לא מגיע ל-MNQU6 האמיתי, כי הוא אף פעם לא "נגמר". התוצאה:
    # הקאש ההיסטורי נבנה ממחיר של חוזה אחד (MNQZ6), בזמן שהסטרימינג החי
    # (resolve_live_contract_async, שכן משתמש ב-OI) נעול על חוזה אחר
    # (MNQU6) - קפיצת-מחיר בדיוק בגבול היסטוריה/לייב, אותה משפחת באגים
    # שכבר תוקנה כמה פעמים בשיחה הזו. בבקשת היסטוריה עם end_dt קבוע בעבר
    # (לא "" - טווח תאריכים סגור) אין טעם ב-OI בכלל: שם רוצים את החוזה
    # שהיה רלוונטי *אז*, וזה בדיוק מה שהדפדוף-הטבעי-אחורה כבר פותר נכון.
    if end_dt == "":
        try:
            candidates = sorted(chain, key=lambda c: c.lastTradeDateOrContractMonth)[-2:]
            front_month = _pick_most_liquid(ib, candidates)
        except Exception:
            front_month = None
        if front_month is not None:
            chain = [c for c in chain
                    if c.lastTradeDateOrContractMonth <= front_month.lastTradeDateOrContractMonth]

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
        contract_chunks: list[pd.DataFrame] = []
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
            contract_chunks.append(chunk_df)
            used += 1

            earliest_raw = pd.Timestamp(chunk_df["date"].iloc[0])
            if earliest_raw.tzinfo is not None:
                # ראו ההערה המקבילה ב-_fetch_chunks למעלה: IB מחזיר tz-aware
                # לפי US/Central (אומת ישירות מול IB), לא US/Eastern.
                earliest_et = earliest_raw.tz_convert("America/New_York")
            else:
                earliest_et = earliest_raw.tz_localize("America/New_York")
            # cursor נשאר tz-aware (America/New_York) - לא naive: ראו ההערה
            # המלאה ב-_fetch_chunks למעלה - naive datetime שנשלח כ-endDateTime
            # מתפרש ע"י IB לפי אזור-הזמן *המקומי של המכונה המריצה* (כאן:
            # Asia/Jerusalem), לא כ-ET - זה בדיוק מה שגרם לדפדוף להתבלבל בין
            # חוזים (MNQU6/MNQZ6) בסביבות roll.
            cursor = (earliest_et - timedelta(seconds=1)).to_pydatetime()
            if earliest_et.tz_localize(None) <= cutoff:
                cursor = None
                break

        consecutive_empty_contracts = 0 if got_any_bar_this_contract else consecutive_empty_contracts + 1
        if contract_chunks:
            _archive_contract_chunks(symbol, contract, timeframe, use_rth, contract_chunks)
    return chunks


def _archive_contract_chunks(symbol: str, contract, timeframe: str, use_rth: bool,
                             chunks: list[pd.DataFrame]) -> None:
    """שומרת את הברים שנשאבו עבור חוזה-משנה קונקרטי אחד (למשל MNQU6) לקובץ קאש
    נפרד משלו - MNQ_MNQU6_1m_ext.parquet - בנוסף לקאש המאוחד הרגיל (MNQ_1m_ext.
    parquet) שממשיך להיות מקור-האמת לתצוגה (build_full_history/load_full_history
    לא משתנים - עדיין קוראים רק את הקאש המאוחד). ארכיון נפרד לכל חוזה מבטיח
    שההיסטוריה הגולמית של כל חוזה שרדה בפני עצמה גם אם הקאש המאוחד אי-פעם
    יצטרך שחזור - לא תלוי בלוגיקת המיזוג/roll של הקאש הראשי.

    משתמשת ב-_load_cache/_save_cache/_cache_path הרגילים עם "symbol" מורכב
    (למשל "MNQ_MNQU6") - אותה הגנת-כיווץ, כתיבה אטומית וניסיון-חוזר בקריאה
    בחינם, בלי לשכפל אף לוגיקה. כשל כאן לא אמור להפיל את השאיבה הראשית - הקאש
    המאוחד עדיין נשמר כרגיל ע"י הקורא (fetch_ohlcv) בכל מקרה.
    """
    try:
        archive_key = f"{symbol}_{contract.localSymbol}"
        new_df = _normalize(pd.concat(chunks, ignore_index=True))
        existing = _load_cache(archive_key, timeframe, use_rth)
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df
        _save_cache(archive_key, timeframe, use_rth, merged)
    except Exception as exc:
        print(f"[archive] אזהרה: כשל בשמירת ארכיון per-contract ל-{symbol}/"
              f"{contract.localSymbol}: {exc!r}", flush=True)


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
        # תיקון קריטי: זה בדיוק מה שנשמר בפועל לקאש הפרקט - כל שאר האפליקציה
        # (dashboard/data.py: SOURCE_TZ="America/New_York") מניחה שהאינדקס
        # הנאיבי כבר בשעון ניו-יורק. IB מחזיר bar.date עם tzinfo=ZoneInfo(
        # 'US/Central') בפועל (אומת ישירות מול IB Gateway, לא הנחה) - Central
        # מפגר שעה קבועה אחרי Eastern. tz_localize(None) לבד רק מוריד את
        # התיוג בלי להמיר את הערך - נותן חותמת "05:56" שמתפרשת בהמשך כאילו
        # הייתה 05:56 ET, כשבפועל השעה האמיתית ב-ET היא 06:56. זו הסיבה
        # לפערים "לא-CME" ולמיקום השגוי של הבר האחרון/החי בגרף: ברים היסטוריים
        # שנשאבו מ-IB יושבים שעה שלמה מוקדם מדי ביחס לדאטה מ-Databento
        # (שכבר בשעון הנכון) ולברים החיים (dashboard/live.py: _bar_from_ib,
        # שכבר עשה tz_convert("America/New_York") נכון לפני tz_localize(None) -
        # אותה תבנית בדיוק, עכשיו גם כאן).
        df["Date"] = df["Date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.drop_duplicates(subset="Date").set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def rebuild_from_contract_archives(symbol: str, timeframe: str, use_rth: bool) -> pd.DataFrame | None:
    """בונה מחדש סדרה רציפה מתוך ארכיוני ה-per-contract שנשמרו בעבר (ראו
    _archive_contract_chunks: MNQ_MNQU6_1m_ext.parquet, MNQ_MNQZ6_1m_ext.parquet
    וכו') - בלי לפנות ל-IB בכלל. רשת-ביטחון נוספת: גם אם הקאש המאוחד
    (MNQ_1m_ext.parquet) יימחק/יתקלקל, ההיסטוריה לא אבודה - ניתן לשחזר אותה
    מהארכיונים הנפרדים בכל רגע. לא נקרא אוטומטית מתוך fetch_ohlcv הרגיל (שם
    ממשיך לפעול הקאש המאוחד כרגיל, כפי שכבר נבדק ועובד) - כלי-עזר לשחזור.

    מיזוג בין חוזים חופפים (נדיר בפועל - _fetch_futures_history כבר עובר
    לחוזה הבא רק כשהקודם "נגמר", כך שהחלונות כמעט תמיד נבדלים): בכפילות
    חותמת-זמן, מנצח הקובץ שנטען אחרון אחרי מיון-אלפביתי של שמות הקבצים -
    ולוגית localSymbol חוזים עתידיים (כמו MNQU6/MNQZ6) ממוינים אלפביתית פחות
    או יותר לפי סדר-פקיעה בפועל בטווח קצר, כך שה"מנצח" בדרך כלל הוא החוזה
    המאוחר יותר - התנהגות סבירה גם אם לא מובטחת ב-100% לכל טווח שנים.
    """
    suffix = "rth" if use_rth else "ext"
    pattern = f"{symbol}_*_{timeframe}_{suffix}.parquet"
    archive_files = sorted(CACHE_DIR.glob(pattern))
    frames = []
    for path in archive_files:
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[archive] אזהרה: לא ניתן לקרוא ארכיון {path.name}: {exc!r}", flush=True)
    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


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

    new_df = pd.concat([_normalize(c) for c in new_chunks]) if new_chunks else pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"])
    if not new_df.empty:
        new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()

    # בדיקת-רציפות לפני מיזוג עם הקאש הקיים: בר היסטורי סגור הוא עובדה קבועה -
    # אם לחותמת-זמן מסוימת יש כבר ערך בקאש *וגם* בשאיבה החדשה, שני הערכים
    # חייבים להסכים כמעט לגמרי; הם לעולם לא אמורים "להשתנות" בין שאיבה
    # לשאיבה. הפרש גדול הוא סימן חד-משמעי לערבוב-חוזים (למשל: כמה תהליכים
    # חולקים את אותו IB Gateway - dashboard.app ושני ה-Streamlit - וכל אחד
    # פותר "front month" בנפרד; תחת עומס בו-זמני reqMktData/reqHistoricalData
    # יכולים להחזיר תשובה לא-עקבית, ותהליך אחד נועל על MNQU6 בעוד תהליך אחר
    # באותו רגע נועל בטעות על MNQZ6) - לא תנודתיות שוק אמיתית. כשזה קורה,
    # השאיבה החדשה *כולה* חשודה (כל הבקשה הזו הגיעה מאותה בחירת-חוזה אחת) -
    # לא רק הברים החופפים - אז מדלגים על כל new_df ומשאירים את הקאש הקיים
    # כמו שהוא: עדיף לפספס עדכון אחד (ינסה שוב בפעם הבאה) מאשר לדרוס דאטה
    # תקין בדאטה מזוהם.
    if cached is not None and not cached.empty and not new_df.empty:
        overlap_idx = new_df.index.intersection(cached.index)
        if len(overlap_idx) > 0:
            price_diff = (new_df.loc[overlap_idx, "Close"] - cached.loc[overlap_idx, "Close"]).abs()
            bad = price_diff[price_diff > CONTRACT_MISMATCH_THRESHOLD_PTS]
            if len(bad) > 0:
                print(f"[cache] אזהרה: השאיבה החדשה של {symbol}/{timeframe} לא מסכימה עם "
                      f"הקאש הקיים ב-{len(bad):,} ברים חופפים (הפרש מקסימלי "
                      f"{bad.max():.2f} נק', למשל ב-{bad.index[0]}) - סימן לערבוב-חוזים "
                      f"(שאיבה בו-זמנית מתהליך אחר שנעל בטעות על חוזה אחר). מדלג על כל "
                      f"השאיבה החדשה הפעם - עדיף לפספס עדכון מאשר לדרוס דאטה תקין.", flush=True)
                new_df = pd.DataFrame(columns=new_df.columns)
                new_chunks = []
                head_chunks = []

    parts = [cached] if cached is not None else []
    if not new_df.empty:
        parts.append(new_df)
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
