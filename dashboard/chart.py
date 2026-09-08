"""לוגיקת הגרף בצד Python: אפשרויות התצוגה והרכבת המטען ש-assets/chart.js מצייר.

הציור עצמו נעשה ב-TradingView Lightweight Charts בדפדפן. הקובץ הזה לא יודע דבר על
Dash — הוא רק בונה מילון JSON-סריאליזבילי, כדי שיהיה קל לבדוק אותו ולהחליף ממשק.
"""
from __future__ import annotations

import pandas as pd

from .data import to_chart_payload

# רקע שחור מלא גם בשטח הנרות עצמו (לא רק "מאחורה"), ורשת הקווים כבויה לגמרי
# (בקשה מפורשת: "לא יהיה את המשבצות האלה"). --bg ב-style.css תואם לאותו שחור.
BG = "#000000"
PANEL = "#252B3A"
BORDER = "#363C4A"
TEXT = "#E8EAF0"
MUTED = "#9296A3"
GRID = "rgba(240,243,250,0.09)"
# ירוק/אדום "זוהרים" - רוויים ובהירים במיוחד (לא זוג TradingView הרגיל), כדי
# שיבלטו בניגוד חד על רקע שחור מלא. גם על הפתילים, לא אפור אחיד.
UP = "#00E676"
DOWN = "#FF1744"
GREEN = "#00E676"
RED = "#FF1744"

# אפשרויות שמועברות כמות שהן ל-LightweightCharts.createChart בדפדפן.
CHART_OPTIONS = {
    # attributionLogo=False: מסתיר את לוגו "TradingView Lightweight Charts" הקטן
    # שהספרייה שמה בפינה השמאלית-תחתונה של הגרף כברירת מחדל (אופציה נתמכת רשמית
    # מ-4.1+ בדיוק לצורך זה).
    "layout": {"background": {"type": "solid", "color": BG}, "textColor": TEXT, "fontSize": 12,
               "attributionLogo": False},
    "grid": {"vertLines": {"visible": False}, "horzLines": {"visible": False}},
    "crosshair": {"mode": 0},
    "rightPriceScale": {"borderColor": BORDER, "scaleMargins": {"top": 0.08, "bottom": 0.28}},
    # shiftVisibleRangeOnNewBar=True (ברירת המחדל של LWC): כל עוד המשתמש בקצה
    # החי, טיקים חדשים ממשיכים "לדחוף" את הטווח הנראה - בדיוק כמו TradingView.
    # זה כבה בעבר כי זה יצר שיטפון בקשות LOD אינסופי שדרס גלילה אחורה אמיתית -
    # אבל שני תיקונים אחר כך טיפלו בשורש הבעיה עצמה (לא בסימפטום): assets/chart.js
    # מבטל בקשות שהמרכז שלהן לא זז משמעותית (lastRequestedCenter), ו-applyLodWindow
    # ממזג דאטה חדש לתוך מה שכבר טעון במקום להחליף אותו. עכשיו אפשר להחזיר את
    # ההתנהגות הטבעית בבטחה - וגם כפתור "Now" (reset-live-btn) סומך על זה
    # שהמעקב האוטומטי דלוק כשחוזרים לקצה החי.
    "timeScale": {"borderColor": BORDER, "timeVisible": True, "secondsVisible": False,
                  "rightOffset": 6, "barSpacing": 8, "shiftVisibleRangeOnNewBar": True},
    "localization": {"locale": "en-US"},
}

CANDLE_OPTIONS = {
    "upColor": UP, "downColor": DOWN,
    "borderUpColor": UP, "borderDownColor": DOWN,
    "wickUpColor": UP, "wickDownColor": DOWN,
}

# הווליום יושב על סקאלה נפרדת שנדחקת לתחתית, כך שהוא לא מועך את הנרות.
VOLUME_OPTIONS = {
    "priceFormat": {"type": "volume"},
    "priceScaleId": "volume",
}
VOLUME_SCALE_MARGINS = {"top": 0.78, "bottom": 0.0}


def build_payload(df: pd.DataFrame, symbol: str, timeframe: str,
                  markers: list[dict] | None = None) -> dict:
    """מרכיב את כל מה שהגרף צריך: נרות, ווליום, סימוני עסקאות וכותרת."""
    payload = to_chart_payload(df)
    payload.update({
        "symbol": symbol,
        "timeframe": timeframe,
        "markers": markers or [],
        "options": CHART_OPTIONS,
        "candleOptions": CANDLE_OPTIONS,
        "volumeOptions": VOLUME_OPTIONS,
        "volumeScaleMargins": VOLUME_SCALE_MARGINS,
    })
    return payload


def build_live_bar_payload(bar: dict | None, symbol: str, timeframe: str) -> dict | None:
    """מטען קל לעדכון לייב: הנר הפתוח הנוכחי בלבד, כפי שמצטבר ב-dashboard/live.py
    מתוך reqMktData בזמן אמת (לא reqHistoricalData, שמחזיר רק נרות סגורים).

    נשלח ל-window.dash_clientside.chart.updateLive ב-assets/chart.js, שמפעיל
    series.update() במקום series.setData() — עדכון זול שלא בונה מחדש את כל הגרף
    ולא נוגע בזום/פאן הנוכחיים של המשתמש.
    """
    if not bar:
        return None
    candle = {"time": bar["time"], "open": bar["open"], "high": bar["high"],
              "low": bar["low"], "close": bar["close"]}
    # קוד קצר, לא rgba מלא - ראו to_chart_payload ב-data.py (אותה מוסכמת).
    color = "u" if bar["close"] >= bar["open"] else "d"
    volume = {"time": bar["time"], "value": bar["volume"], "color": color}
    return {"candles": [candle], "volume": [volume], "symbol": symbol, "timeframe": timeframe}


def build_history_chunk_payload(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """מטען לחלון היסטוריה נוסף (טעינה הדרגתית כשגוללים אחורה): אותה צורה בדיוק
    כמו build_payload (candles/volume), אבל בלי אופציות/מרקרים — הדפדפן ממזג את
    זה עם מה שכבר טעון (assets/chart.js: prependHistory) במקום setData() מלא."""
    payload = to_chart_payload(df)
    payload["symbol"] = symbol
    payload["timeframe"] = timeframe
    return payload


def ohlc_summary(df: pd.DataFrame) -> str:
    """שורת ה-OHLC שמופיעה בפינת הגרף, בסגנון TradingView."""
    last = df.iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else last["Open"]
    chg = last["Close"] - prev
    pct = (chg / prev * 100) if prev else 0.0
    return (f"O{last['Open']:.2f}  H{last['High']:.2f}  "
            f"L{last['Low']:.2f}  C{last['Close']:.2f}  "
            f"{chg:+.2f} ({pct:+.2f}%)")
