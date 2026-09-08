"""ייבוא חד-פעמי: קובץ DBN של Databento (OHLCV-1m, MNQ) -> קאש ה-parquet של
ib_data.py, באותו פורמט בדיוק (Open/High/Low/Close/Volume, אינדקס naive בשעון
ניו-יורק) - כדי שהאפליקציה תזהה את זה כקאש 1m רגיל ולא תדע שהוא לא הגיע מ-IB.

למה בכלל צריך את זה: MNQ הוא חוזה עתידי - IB לא חושף מטא-דאטה על חוזים שפגו
לפני יותר משנה-שנה וחצי (reqContractDetails(includeExpired=True) מוגבל), אז
דפדוף אחורה רגיל דרך fetch_ohlcv נתקע בערך שנה אחורה. Databento מוכרים בדיוק
את ההיסטוריה המלאה כקובץ, וזה מייבא אותה פעם אחת לאותו קאש מקומי.

הרצה:  .venv/bin/python import_databento.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError as exc:
    raise SystemExit(
        "חסרה הספרייה databento. התקן עם: .venv/bin/pip install databento"
    ) from exc

from ib_data import _cache_path  # אותה פונקציה בדיוק ש-_load_cache/_save_cache משתמשות בה

SYMBOL = "MNQ"
TIMEFRAME = "1m"
USE_RTH = False  # "_ext" - שעות מורחבות, תואם למה שהאפליקציה טוענת כברירת מחדל

DBN_FILE = (
    Path(__file__).resolve().parent
    / "databento_data" / "GLBX-20260902-NV43FCYNXQ"
    / "glbx-mdp3-20100606-20260901.ohlcv-1m.dbn.zst"
)

SOURCE_TZ = "America/New_York"  # אותו אזור-זמן ש-ib_data.py מניח לכל הבארים ששמורים בקאש


def load_dbn(path: Path) -> pd.DataFrame:
    """קורא את קובץ ה-DBN וממיר ל-DataFrame גולמי (כל החוזים, שעון UTC)."""
    store = db.DBNStore.from_file(path)
    df = store.to_df()
    print(f"נקראו {len(df):,} שורות גולמיות מ-DBN (dataset={store.dataset}, schema={store.schema})")
    return df


def stitch_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """תופר את כל פקיעות החוזה לסדרה רציפה אחת: מסננים spreads (סימבולים עם '-'),
    ולכל יום בוחרים חוזה "פעיל" יחיד לפי נפח המסחר (roll וולומטרי).

    גרסה קודמת בחרה per-day argmax בלי שום אילוץ סדר - סביב תאריך הפקיעה, כשהנפח
    של החוזה הישן והחדש קרוב, זה יכול "להתנדנד" בין החוזים על פני כמה ימים
    (יום א' חוזה חדש מוביל בנפח, יום ב' חוזר להוביל הישן) - בדיוק הגורם לקפיצות
    מחיר חוזרות ("קופץ ומהבהב") שראינו בגרף, כי לכל חוזה יש רמת מחיר שונה במקצת
    (basis). התיקון: אוכפים שה"חוזה הפעיל" יכול להתקדם רק *קדימה* בציר הזמן של
    פקיעות (לפי סדר ההופעה הראשונה של כל חוזה בדאטה עצמו - לא parsing של קוד
    החודש בטיקר) ולעולם לא חוזר אחורה לחוזה מוקדם יותר, גם אם יום בודד מראה לו
    נפח גבוה יותר. כל חוזה נבחר לכל היותר פעם אחת, ברצף ימים אחד.
    """
    df = df[~df["symbol"].str.contains("-")].copy()
    print(f"אחרי סינון spreads: {len(df):,} שורות, {df['symbol'].nunique()} חוזים בודדים")

    # UTC -> שעון ניו-יורק naive, אותה מוסכמה בדיוק כמו כל שאר האפליקציה.
    df.index = df.index.tz_convert(SOURCE_TZ).tz_localize(None)
    df["day"] = df.index.date

    # סדר החוזים לפי מועד ההופעה הראשונה שלהם בדאטה - זה תמיד תואם את סדר
    # הפקיעות בפועל (חוזה מוקדם יותר תמיד מתחיל להיסחר לפני חוזה מאוחר ממנו).
    first_seen = df.groupby("symbol").apply(lambda g: g.index.min(), include_groups=False)
    contract_rank = {sym: i for i, sym in enumerate(first_seen.sort_values().index)}

    daily_volume = (
        df.groupby(["day", "symbol"])["volume"].sum()
        .reset_index()
        .sort_values("day")
    )
    daily_volume["rank"] = daily_volume["symbol"].map(contract_rank)

    front_by_day: dict = {}
    current_rank = -1
    for day, group in daily_volume.groupby("day", sort=True):
        eligible = group[group["rank"] >= current_rank]
        if eligible.empty:  # לא אמור לקרות (current_rank תמיד rank של יום קודם), רשת ביטחון
            eligible = group
        best = eligible.loc[eligible["volume"].idxmax()]
        current_rank = best["rank"]
        front_by_day[day] = best["symbol"]

    switches = len(set(front_by_day.values())) - 1
    print(f"נבחרו חוזים פעילים ל-{len(front_by_day):,} ימים ({switches} מעברי roll, כל אחד קדימה בלבד)")

    df["front_symbol"] = df["day"].map(front_by_day)
    df = df[df["symbol"] == df["front_symbol"]].copy()
    print(f"אחרי בחירת החוזה הפעיל ליום: {len(df):,} שורות")

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    df.index.name = "Date"
    df["Volume"] = df["Volume"].astype(float)

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    if len(df) != before:
        print(f"הוסרו {before - len(df):,} שורות כפולות")

    return df


def main() -> None:
    if not DBN_FILE.exists():
        raise SystemExit(f"קובץ ה-DBN לא נמצא: {DBN_FILE}")

    raw = load_dbn(DBN_FILE)
    clean = stitch_continuous(raw)

    if clean.empty:
        raise SystemExit("התוצאה אחרי העיבוד ריקה - כנראה בעיה בקובץ המקור")
    if not clean.index.is_monotonic_increasing:
        raise SystemExit("האינדקס אינו עולה חד-משמעית אחרי המיון - עצירה למניעת שמירת קאש שגוי")

    out_path = _cache_path(SYMBOL, TIMEFRAME, USE_RTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # כתיבה אטומית (קובץ זמני + rename) - אותה הגנה שיש ב-ib_data._save_cache,
    # כדי שהפרעה באמצע הכתיבה לא תשאיר קאש חלקי/פגום.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    clean.to_parquet(tmp_path)
    tmp_path.replace(out_path)

    years = (clean.index.max() - clean.index.min()).days / 365.25
    print()
    print(f"נשמר ל-{out_path}")
    print(f"סה\"כ {len(clean):,} ברים, {clean.index.min()} -> {clean.index.max()} ({years:.1f} שנים)")


if __name__ == "__main__":
    main()
