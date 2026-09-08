"""בונה רצף MNQ רציף (2019-2026) מקובץ ה-DBN של Databento ומאחד עם קאש IB הקיים.

לכל יום נבחר החוזה עם נפח המסחר הגבוה ביותר (roll וולומטרי) - כך נגיד "החוזה
הפעיל" מדי יום, בלי לגעת בימים שבהם שני חוזים נסחרים במקביל סביב תפוגה.
מתעלם מסימבולים של spread (מכילים '-', כמו MNQM9-MNQU9).

הפלט נשמר באותו פורמט בדיוק כמו data_cache/MNQ_1m_ext.parquet (Open/High/Low/
Close/Volume, אינדקס 'Date' naive בשעון ניו-יורק), כדי שהאפליקציה תמשיך לעבוד
בלי שינוי נוסף. חופף בין Databento ל-IB נפתר לטובת IB (המקור החי להמשך).
"""
from pathlib import Path

import pandas as pd
import databento as db

ROOT = Path(__file__).resolve().parent.parent
DBN_FILE = ROOT / "databento_data/GLBX-20260902-NV43FCYNXQ/glbx-mdp3-20100606-20260901.ohlcv-1m.dbn.zst"
IB_CACHE = ROOT / "data_cache/MNQ_1m_ext.parquet"
OUT_FILE = ROOT / "data_cache/MNQ_1m_ext.parquet"


def main():
    print("קורא DBN...")
    store = db.DBNStore.from_file(DBN_FILE)
    df = store.to_df()
    print(f"נטענו {len(df):,} שורות גולמיות (כל החוזים + spreads)")

    df = df[~df["symbol"].str.contains("-")].copy()
    print(f"אחרי סינון spreads: {len(df):,} שורות, {df['symbol'].nunique()} חוזים")

    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df["day"] = df.index.date

    daily_vol = df.groupby(["day", "symbol"])["volume"].sum()
    front = daily_vol.groupby("day").idxmax().apply(lambda x: x[1])
    front.name = "front_symbol"

    df = df.join(front, on="day")
    df = df[df["symbol"] == df["front_symbol"]].copy()
    print(f"אחרי בחירת החוזה הפעיל ביום: {len(df):,} שורות")

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    df.index.name = "Date"
    df["Volume"] = df["Volume"].astype(float)

    print(f"טווח Databento: {df.index[0]} -> {df.index[-1]}")

    # Databento הוא מקור נקי ורציף - עדיף על פני שרשור-החוזים הידני של IB בכל
    # מקום שהוא מכסה (נמצא פגם אמיתי בגבול הישן: קפיצת מחיר + וולום אפס ל-IB
    # סביב 2025-06-19 10:03, בעוד ל-Databento יש שם נתונים תקינים). IB משמש רק
    # לזנב שאחרי סוף קובץ ה-Databento (המשך חי/עדכני).
    if IB_CACHE.exists():
        ib_df = pd.read_parquet(IB_CACHE)
        print(f"קאש IB קיים: {len(ib_df):,} שורות, {ib_df.index[0]} -> {ib_df.index[-1]}")
        ib_tail = ib_df[ib_df.index > df.index[-1]]
        print(f"זנב IB אחרי סוף Databento: {len(ib_tail):,} שורות")
        merged = pd.concat([df, ib_tail])
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    else:
        merged = df

    print(f"סה\"כ אחרי מיזוג: {len(merged):,} שורות, {merged.index[0]} -> {merged.index[-1]}")
    merged.to_parquet(OUT_FILE)
    print(f"נשמר ל-{OUT_FILE}")


if __name__ == "__main__":
    main()
