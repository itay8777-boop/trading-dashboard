"""
adjust_rolls.py — התאמת גלגולים (back-adjustment) לקאש הדקה של חוזה עתידי.

הסדרה הרציפה שנבנתה ב-import_databento.py מחליפה חוזה לפי נפח אבל לא מתאימה
מחירים: בכל גלגול יש קפיצה בגודל הבסיס (חוזה חדש פחות ישן). כאן מזיזים את כל
ההיסטוריה שלפני כל גלגול בגודל הבסיס המצטבר, כך שהסדרה רציפה והמחיר העדכני
נשאר המחיר האמיתי של החוזה הנוכחי (שיטת "Panama canal", התאמה חיבורית).

שימוש:
    python tools/adjust_rolls.py MNQ            # מחיל (שומר גיבוי *_unadjusted.parquet)
    python tools/adjust_rolls.py MNQ --revert   # מחזיר את הגיבוי
הטבלה: data_cache/MNQ_rolls.json  {"rolls": [{"date","old","new","basis"}...], "adjusted": bool}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"


def main(symbol: str, revert: bool = False) -> None:
    symbol = symbol.upper()
    cache = CACHE / f"{symbol}_1m_ext.parquet"
    backup = CACHE / f"{symbol}_1m_ext_unadjusted.parquet"
    rolls_p = CACHE / f"{symbol}_rolls.json"
    rolls = json.loads(rolls_p.read_text(encoding="utf-8"))
    if revert:
        if not backup.exists():
            print("אין גיבוי להחזיר"); return
        backup.replace(cache); rolls["adjusted"] = False
        rolls_p.write_text(json.dumps(rolls, ensure_ascii=False, indent=1), encoding="utf-8")
        print("הוחזר הקאש הלא-מותאם"); return
    if rolls.get("adjusted"):
        print("כבר מותאם"); return
    df = pd.read_parquet(cache)
    usable = [r for r in rolls["rolls"] if r.get("basis") is not None]
    # התאמה מצטברת: לכל בר לפני גלגול d מוסיפים את סכום הבסיסים של כל הגלגולים אחריו
    offsets = pd.Series(0.0, index=df.index)
    cum = 0.0
    for r in sorted(usable, key=lambda x: x["date"], reverse=True):
        d = pd.Timestamp(r["date"])
        cum += float(r["basis"])
        offsets[df.index < d] = cum
    if not backup.exists():
        df.to_parquet(backup)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] + offsets.values
    df.to_parquet(cache)
    rolls["adjusted"] = True
    rolls_p.write_text(json.dumps(rolls, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"הותאמו {len(usable)} גלגולים; היסט מצטבר מקסימלי {offsets.max():+.2f} נק'. גיבוי: {backup.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MNQ", revert="--revert" in sys.argv)
