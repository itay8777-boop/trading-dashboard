"""Pattern Scanner: מוצאת כל נר "hammer" בהיסטוריה (פתיל תחתון ארוך, כמעט בלי
פתיל עליון, ירוק - הגדרה קבועה, לא ניתנת לקונפיגורציה מה-UI), ובודקת לכל hammer
אם ב-N הברים שמיד אחריו (N = "Bars to check", קלט מה-UI) ה-high הגיע אי-פעם
target_pts נקודות מעל ה-high של ה-hammer עצמו (הצלחה/כישלון).

מעבר לזה, כל hammer מסווג גם לפי התנודתיות שקדמה לו: סטיית-התקן (sample std,
ddof=1 - ברירת המחדל של pandas) של מחירי ה-Close על 20 הברים *שלפניו ממש* (לא
כולל את ה-hammer עצמו) - כדי לבדוק אם hammer שמופיע אחרי תקופה שקטה מצליח יותר
מ-hammer שמופיע אחרי תקופה תנודתית. התוצאה מקובצת ל-5 קבוצות סטיית-תקן קבועות
(0-5 / 5-10 / 10-20 / 20-50 / 50+ נקודות, חצי-פתוחות [lo,hi) כדי שלא יהיו
חפיפות/פערים), עם total/successful/hit-rate לכל קבוצה בנפרד ובסה"כ.

וקטורי לגמרי (numpy/pandas) - כל החישוב (זיהוי-hammer, חלון-N-הברים-הבאים לכל
hammer בבת אחת, וסטיית-התקן הגלגלת) הוא כמה פעולות numpy/pandas, בלי שום לולאת
פייתון על הברים עצמם (רק לולאה קטנה, קבועת-אורך, על 5 קבוצות הסטיית-תקן).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# חצי-פתוחות [lo, hi) - כל ערך non-negative נופל לקבוצה אחת בדיוק, בלי חפיפה.
STD_BINS = [
    (0.0, 5.0, "STD 0-5 points"),
    (5.0, 10.0, "STD 5-10 points"),
    (10.0, 20.0, "STD 10-20 points"),
    (20.0, 50.0, "STD 20-50 points"),
    (50.0, np.inf, "STD 50+ points"),
]


def _empty_result(bars_to_check: int, target_pts: float) -> dict:
    groups = [
        {"label": label, "n_hammers": 0, "n_hits": 0, "hit_rate_pct": 0.0}
        for _, _, label in STD_BINS
    ]
    return {
        "groups": groups,
        "total_hammers": 0,
        "total_hits": 0,
        "overall_hit_rate_pct": 0.0,
        "bars_to_check": bars_to_check,
        "target_pts": target_pts,
    }


def scan_hammers(df: pd.DataFrame, *, bars_to_check: int = 10, target_pts: float = 10.0) -> dict:
    """שלבי הסריקה (ראו גם תיאור המודול למעלה):

    1. זיהוי hammer על כל המערך בבת אחת:
       rng=high-low, body=|close-open|, lower_wick=min(open,close)-low,
       upper_wick=high-max(open,close),
       hammer = upper_wick<=0.1*rng AND lower_wick>=0.6*rng AND body>0
                AND close>=open.
    2. לכל hammer: הצלחה אם ב-bars_to_check הברים *שמיד אחריו* ה-high מגיע
       אי-פעם ל-target_pts נקודות (או יותר) מעל ה-high של ה-hammer עצמו -
       וקטורי (מטריצת חלונות (n_hammers, bars_to_check) + any לאורך השורה).
    3. סטיית-התקן של ה-Close על 20 הברים *שלפני* כל hammer (לא כולל אותו) -
       rolling(20).std().shift(1): rolling(20).std() בעמדה i מביט על
       close[i-19..i] (20 ברים שמסתיימים ב-i כולל); shift(1) מזיז את זה בר
       אחד קדימה כך שהערך בעמדה i הופך לסטיית-התקן על close[i-20..i-1] -
       בדיוק "20 הברים שלפני הבר i", בלי לכלול אותו.
    4. קיבוץ כל ה-hammers (בין אם הצליחו ובין אם לא) ל-5 קבוצות סטיית-תקן
       קבועות (STD_BINS), עם total/successful/hit-rate לכל קבוצה, ובנוסף
       total/successful/hit-rate כוללים על כל ה-hammers יחד.

    hammer קרוב מדי לתחילת ההיסטוריה (פחות מ-20 ברים לפניו, אז סטיית-התקן
    לא מוגדרת) או לסופה (פחות מ-bars_to_check ברים אחריו, אז אי-אפשר לבדוק
    הצלחה/כישלון) מוחרג אוטומטית מהסריקה כולה - לא רק מהצלחה/כישלון.
    """
    n = len(df)
    if n < 21 or bars_to_check < 1:
        return _empty_result(bars_to_check, target_pts)

    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    rng = high - low
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    hammer_mask = (
        (upper_wick <= 0.1 * rng) & (lower_wick >= 0.6 * rng)
        & (body > 0) & (close >= open_)
    )

    idx_all = np.arange(n)
    # צריך גם 20 ברים מלאים *לפני* (לסטיית-התקן) וגם bars_to_check ברים מלאים
    # *אחרי* (לבדיקת ההצלחה) - אחרת ה-hammer מוחרג מהסריקה כולה, לא רק מהתוצאה.
    hammer_mask &= (idx_all >= 20) & (idx_all <= n - 1 - bars_to_check)

    hammer_idx = np.flatnonzero(hammer_mask)
    n_hammers = int(len(hammer_idx))
    if n_hammers == 0:
        return _empty_result(bars_to_check, target_pts)

    hammer_high = high[hammer_idx]
    target = hammer_high + target_pts
    offsets = np.arange(1, bars_to_check + 1)
    windows = high[hammer_idx[:, None] + offsets[None, :]]
    success_mask = (windows >= target[:, None]).any(axis=1)

    std_before_20 = (
        pd.Series(close).rolling(window=20, min_periods=20).std().shift(1).to_numpy()
    )
    hammer_std = std_before_20[hammer_idx]  # hammer_idx>=20 מבטיח ערך מוגדר תמיד (לא NaN)

    groups = []
    for lo, hi, label in STD_BINS:
        sel = (hammer_std >= lo) & (hammer_std < hi)
        cnt = int(np.count_nonzero(sel))
        hits = int(np.count_nonzero(success_mask[sel]))
        groups.append({
            "label": label,
            "n_hammers": cnt,
            "n_hits": hits,
            "hit_rate_pct": (hits / cnt * 100) if cnt else 0.0,
        })

    total_hits = int(np.count_nonzero(success_mask))

    return {
        "groups": groups,
        "total_hammers": n_hammers,
        "total_hits": total_hits,
        "overall_hit_rate_pct": (total_hits / n_hammers * 100) if n_hammers else 0.0,
        "bars_to_check": bars_to_check,
        "target_pts": target_pts,
    }
