"""
audit.py — ביקורת איכות לקאש הנרות (מה הבקטסט באמת רץ עליו).

בודק את קאש הדקה של סימבול: טווח, כפילויות, נרות בלי נפח, פערים בתוך הסשן
הרגיל (מסווגים: חצאי ימים, תחזוקה יומית, אחר), קפיצות פתיחה חריגות, ימי גלגול
חוזים (מתוך data_cache/<SYMBOL>_rolls.json אם קיים) והאם הסדרה מותאמת לגלגולים.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import html

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def rolls_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_rolls.json"


def load_rolls(symbol: str) -> dict | None:
    p = rolls_path(symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def audit_cache(symbol: str) -> dict:
    path = CACHE_DIR / f"{symbol.upper()}_1m_ext.parquet"
    if not path.exists():
        return {"ok": False, "error": f"אין קאש דקה ל-{symbol}"}
    df = pd.read_parquet(path)
    idx = df.index
    out = {"ok": True, "symbol": symbol.upper(), "rows": int(len(df)), "from": str(idx.min())[:16], "to": str(idx.max())[:16],
           "dup": int(idx.duplicated().sum()), "monotonic": bool(idx.is_monotonic_increasing),
           "nan": int(df.isna().sum().sum()), "zero_vol": int((df["Volume"] <= 0).sum()) if "Volume" in df else 0,
           "zero_range": int((df["High"] == df["Low"]).sum()),
           "bad_ohlc": int(((df["High"] < df[["Open", "Close"]].max(axis=1)) | (df["Low"] > df[["Open", "Close"]].min(axis=1))).sum())}
    # פערים בתוך הסשן הרגיל (10:00–16:00 ניו-יורק, ימי חול)
    d = idx.to_series().diff().dt.total_seconds()
    mask = (d > 300) & (idx.dayofweek < 5) & (idx.hour >= 10) & ((idx.hour < 16) | ((idx.hour == 16) & (idx.minute == 0)))
    gaps = idx[mask]
    prev = idx[np.searchsorted(idx, gaps) - 1] if len(gaps) else idx[:0]
    g = pd.DataFrame({"from": prev, "to": gaps, "minutes": (gaps - prev).total_seconds() / 60}) if len(gaps) else pd.DataFrame(columns=["from", "to", "minutes"])
    if len(g):
        g["half_day"] = g["from"].dt.hour.eq(13) & g["from"].dt.minute.le(1)
        g["maint"] = g["from"].dt.hour.isin([16, 17])
        real = g[~g.half_day & ~g.maint]
    else:
        real = g
    out["gaps_total"] = int(len(g)); out["gaps_half_day"] = int(g["half_day"].sum()) if len(g) else 0
    out["gaps_other"] = int(len(real))
    out["gaps_by_year"] = {int(k): int(v) for k, v in real.groupby(real["from"].dt.year).size().items()} if len(real) else {}
    out["gaps_top"] = [(str(r["from"])[:16], str(r["to"])[:16], float(r["minutes"])) for _, r in real.sort_values("minutes", ascending=False).head(8).iterrows()] if len(real) else []
    # קפיצות פתיחה חריגות (פתיחה מול הסגירה הקודמת)
    jump = (df["Open"] / df["Close"].shift(1) - 1)
    big = jump[jump.abs() > 0.01]
    out["jumps_1pct"] = int(len(big))
    out["jumps_top"] = [(str(t)[:16], float(round(v * 100, 2))) for t, v in big.abs().sort_values(ascending=False).head(8).items()]
    per_day = df.groupby(idx.normalize()).size()
    out["days"] = int(len(per_day)); out["days_short"] = int((per_day < 200).sum()); out["bars_per_day_median"] = float(per_day.median())
    rolls = load_rolls(symbol)
    out["rolls"] = rolls
    return out


def _row(k, v, cls=""):
    return html.Div(className="audit-row", children=[html.Span(k), html.Span(v, className=cls)])


def audit_view(a: dict) -> html.Div:
    if not a.get("ok"):
        return html.Div(a.get("error", "שגיאה"), className="tester-error")
    ok = lambda c: "pos" if c else "neg"
    rolls = a.get("rolls")
    cards = [
        html.Div(className="audit-card", children=[
            html.Div("הקאש", className="split-title"),
            _row("סימבול", a["symbol"]), _row("נרות דקה", f"{a['rows']:,}"), _row("טווח", f"{a['from']} עד {a['to']}"),
            _row("ימי מסחר", f"{a['days']:,} (חציון {a['bars_per_day_median']:.0f} נרות ליום)"),
            _row("ימים קצרים (<200 נרות)", str(a["days_short"]), "neu"),
        ]),
        html.Div(className="audit-card", children=[
            html.Div("תקינות", className="split-title"),
            _row("חותמות כפולות", str(a["dup"]), ok(a["dup"] == 0)), _row("סדר כרונולוגי", "כן" if a["monotonic"] else "לא", ok(a["monotonic"])),
            _row("ערכים חסרים", str(a["nan"]), ok(a["nan"] == 0)), _row("OHLC לא עקבי", str(a["bad_ohlc"]), ok(a["bad_ohlc"] == 0)),
            _row("נרות בלי נפח", f"{a['zero_vol']:,}", "neu"), _row("נרות בלי טווח (H=L)", f"{a['zero_range']:,}", "neu"),
        ]),
        html.Div(className="audit-card", children=[
            html.Div("פערים בתוך הסשן הרגיל (>5 דק')", className="split-title"),
            _row("סה\"כ", str(a["gaps_total"])), _row("חצאי ימים (סגירה 13:00)", str(a["gaps_half_day"])),
            _row("פערים אחרים", str(a["gaps_other"]), ok(a["gaps_other"] == 0)),
            _row("לפי שנה", ", ".join(f"{y}: {n}" for y, n in sorted(a["gaps_by_year"].items())) or "—"),
        ] + [html.Div(f"{f} → {t} ({m:.0f} דק')", className="audit-line") for f, t, m in a["gaps_top"]]),
        html.Div(className="audit-card", children=[
            html.Div("קפיצות פתיחה מעל 1%", className="split-title"),
            _row("כמות", str(a["jumps_1pct"]), "neu"),
        ] + [html.Div(f"{t}: {v:+.2f}%", className="audit-line") for t, v in a["jumps_top"]]),
    ]
    if rolls:
        adjusted = rolls.get("adjusted", False)
        lines = [html.Div(f"{r['date']}: {r['old']} → {r['new']} · בסיס {r['basis']:+.2f} נק'" if r.get("basis") is not None else f"{r['date']}: {r['old']} → {r['new']} · בסיס לא חושב",
                          className="audit-line") for r in rolls.get("rolls", [])]
        cards.append(html.Div(className="audit-card", children=[
            html.Div("גלגולי חוזים", className="split-title"),
            _row("מספר גלגולים", str(len(rolls.get("rolls", [])))),
            _row("הסדרה מותאמת לגלגולים (back-adjusted)", "כן" if adjusted else "לא", ok(adjusted)),
            html.Div("ללא התאמה, כל גלגול מכניס קפיצת מחיר מלאכותית בגודל הבסיס בין החוזים, שמייצרת אותות ותנועות שלא היו בשוק.", className="pane-sub"),
        ] + lines))
    else:
        cards.append(html.Div(className="audit-card", children=[
            html.Div("גלגולי חוזים", className="split-title"),
            html.Div("אין טבלת גלגולים (data_cache/<סימבול>_rolls.json). הסדרה כנראה לא מותאמת.", className="pane-sub")]))
    return html.Div(className="audit-grid", children=cards)
