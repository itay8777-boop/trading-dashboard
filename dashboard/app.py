"""דשבורד Dash: גרף TradingView Lightweight Charts + פאנל אסטרטגיה מתחתיו.

הפעלה:  .venv/bin/python -m dashboard.app     ואז http://127.0.0.1:8050
דורש TWS/IB Gateway פתוח על 127.0.0.1:7497 עם API מופעל.

הפריסה: הגרף תופס את מלוא הרוחב, ומתחתיו פאנל נפתח/נסגר עם שתי לשוניות —
Strategy Tester (ביצועים, תשואה, עסקאות) ו-Pine Editor (כתיבת קוד האסטרטגיה).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import flask
import pandas as pd
from dash import ALL, Dash, Input, Output, State, callback, callback_context, dcc, html, no_update

# ייבוא מוחלט — מגיע ל-chart.py שבשורש הפרויקט (גרף הביצועים של Plotly),
# ולא ל-dashboard/chart.py שנטען כ-`.chart`.
from chart import _evenly_sampled_indices, build_performance_chart

from . import live, panels
from .chart import build_history_chunk_payload, build_live_bar_payload, build_payload
from .data import (
    INITIAL_DAYS,
    TIMEFRAME_ORDER,
    DataError,
    from_display_epoch,
    load_full_history,
    load_lod_window_from_1m_cache,
    load_ohlcv,
    load_ohlcv_preferring_1m_cache,
    warm_cache,
)
from .strategy import (
    DEFAULT_CODE,
    HAMMER_RSI_CODE,
    RECOGNIZED_STRATEGY_INPUTS,
    StrategyCompileError,
    detect_builtin_strategy,
    parse_strategy_inputs,
    run_fast_sma_crossover_backtest,
    run_hammer_rsi_backtest,
    to_markers,
)

DEFAULT_SYMBOL = "MNQ"
DEFAULT_TF = "1m"

# מחמם את קאש ה-1m בזיכרון ברקע ברגע שהמודול נטען (לפני שהמשתמש בכלל פתח את
# הדפדפן) - כדי שהגלילה הראשונה שלו כבר לא תשלם את עלות הקריאה החד-פעמית
# מהדיסק. ראו warm_cache/_load_cache.
warm_cache(DEFAULT_SYMBOL)

# הנר הפתוח מתעדכן בזיכרון מ-reqHistoricalData(keepUpToDate=True) (dashboard/
# live.py) — ה-Interval רק קורא אותו, לא נוגע ב-IB, אז אפשר אינטרוול קצר בלי
# לחשוש מ-pacing. 200 מ"ש (5/שנייה) במקום 1000 - כדי שכל עדכון-בר שמגיע מ-IB
# (ראו live.py) יוצג כמעט מיד בדפדפן, לא יחכה עד לטיק ה-Interval הבא.
LIVE_INTERVAL_MS = 200

# הנרות נשמרים בצד השרת ולא ב-dcc.Store: כמה אלפי נרות הם מאות KB של JSON שהיו
# נוסעים לדפדפן ובחזרה בכל הרצה. ההנחה היא תהליך יחיד — מספיק לשימוש מקומי.
_FRAMES: dict[tuple[str, str], object] = {}
_RESULTS: dict[str, object] = {}

# סימוני-עסקאות (משולשים) נשלחים לגרף לפי דרישה, רק על טווח הזמן הנראה כרגע -
# לא כל 150K+ העסקאות בבת אחת. הם מצוירים ידנית על ה-canvas בכל פריים (ראו
# dashboard/assets/chart.js: createTradeMarkers), ולולאה על מאות אלפי נקודות
# בכל תזוזת עכבר/גלילה היא בדיוק מה שהקפיא את כל הטאב אחרי "Full backtest" -
# לא רק את הגרף, את כל הדף (אותו thread ראשי מריץ גם קליק/גלילה/רינדור).
# assets/chart.js קורא ל-/api/markers (למטה) בכל שינוי טווח נראה, בדיוק כמו
# מדרג-הפירוט של הנרות עצמם (/api/lod) - כך שגלילה ל-2019 מציגה את עסקאות 2019,
# וזום ל-2023 מציג את עסקאות 2023, בלי לשלוח יותר מ-MAX_VISIBLE_MARKER_TRADES
# עסקאות (= עד פי 2 נקודות-סימון) בכל תשובה.
MAX_VISIBLE_MARKER_TRADES = 250

# "List of Trades" מוצג בעמודים של כמה שורות - 150K+ שורות DOM בטבלה אחת (גם
# אחרי אופטימיזציית בניית ה-HTML עצמה) עדיין מקפיאות את הדפדפן ברגע שהוא מנסה
# לפרסר/לצייר אותן. ראו ui_components.trades_table_html.
TRADES_PAGE_SIZE = 100

# פילטר תקופה מעל גרף ה-Performance (Overview) - בדיוק כמו ב-Strategy Tester
# של TradingView. "range" (ברירת מחדל) ו-"all" שקולים באפליקציה הזו: הבקטסט
# תמיד רץ על כל הקאש (ראו run_or_switch), אז "טווח הגרף הזמין" הוא ממילא כל
# ההיסטוריה - נשארים שני ערכים נפרדים רק כי זה בדיוק המינוח של TradingView.
# משפיע רק על גרף ה-Overview ועל Key stats - לא נוגע בגרף הנרות הראשי, לא
# בבקטסט עצמו, ולא בלשוניות Performance Summary/List of Trades (ראו _result_body).
PERIOD_LABELS = {
    "range": "Available chart range",
    "7d": "Last 7 Days",
    "30d": "Last 30 Days",
    "90d": "Last 90 Days",
    "365d": "Last 365 Days",
    "all": "Entire History",
    "custom": "Custom Range",
}
_PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90, "365d": 365}

# ספריית אסטרטגיות שמורות של Pine Editor - קובץ JSON מקומי ולא זיכרון-תהליך
# (כמו _FRAMES/_RESULTS), כדי שהאסטרטגיות ישרדו גם הפעלה מחדש של השרת, בדיוק
# כמו ה-"My Scripts" של TradingView. assets/strategy_library.js קורא/כותב
# דרך /api/strategies* למטה; אין כאן dcc.Store/callback של Dash בכלל - הלשונית
# "Pine Editor" (code-input) והגדרות (Settings) מתעדכנות ישירות מה-JS דרך
# dash_clientside.set_props, בלי סבב-חזור לשרת.
STRATEGIES_FILE = Path(__file__).resolve().parent.parent / "strategies.json"
_STRATEGIES_LOCK = threading.Lock()

# אסטרטגיות מובנות (built-in) - תמיד מופיעות בספריית האסטרטגיות (ליד לשונית
# Pine Editor), בלי צורך לשמור אותן ל-strategies.json בכלל: ראו _load_strategies
# למטה, שממזג אותן עם השמורות-משתמש בזמן קריאה, לא כותב אותן לדיסק. כך אי-אפשר
# "למחוק" אותן בטעות (מחיקה דרך /api/strategies/delete פשוט לא תמצא אותן בקובץ),
# והן ממשיכות להופיע גם על strategies.json ריק/חדש. שם מזהה זהה לשם שנשמר -
# שמירה בשם הזה על ידי המשתמש פשוט "תדרוס חזותית" (לא לצמיתות) את המובנית.
BUILTIN_STRATEGIES = [
    {"name": "Hammer + RSI (built-in)", "code": HAMMER_RSI_CODE, "settings": {}},
]
_BUILTIN_STRATEGY_NAMES = {s["name"] for s in BUILTIN_STRATEGIES}


def _load_strategies() -> list[dict]:
    saved: list[dict] = []
    if STRATEGIES_FILE.exists():
        try:
            with STRATEGIES_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            saved = data.get("strategies", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError):
            saved = []
    user_only = [s for s in saved if s.get("name") not in _BUILTIN_STRATEGY_NAMES]
    return BUILTIN_STRATEGIES + user_only


def _save_strategies(strategies: list[dict]) -> None:
    with STRATEGIES_FILE.open("w", encoding="utf-8") as f:
        json.dump({"strategies": strategies}, f, ensure_ascii=False, indent=2)


def _html(markup: str, **kw):
    """מזריק HTML גולמי שנבנה ב-Python (טבלאות TradingView) לתוך עץ ה-Dash."""
    return dcc.Markdown(markup, dangerously_allow_html=True, **kw)


def _ic_chart_svg():
    """אייקון "עקומת מדדים" בכפתור Indicators - כמו ב-TradingView (לא טקסט/אימוג'י)."""
    return html.Img(src=(
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
        "fill='none' stroke='%239296A3' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M3 17l4-9 5 6 4-8 5 11'/></svg>"
    ), className="indicators-btn-icon")


def _indicator_row(key: str, label: str, params: list[tuple[str, float]]):
    """שורה אחת בתפריט Indicators - ראו assets/indicators.js: הפרמטרים נקראים
    ישירות מה-DOM (id="ind-{key}-{param}") בלחיצה על Add, לא דרך State של Dash -
    כל האינדיקטור (חישוב + ציור כ-series על הגרף) הוא לוגיקת-לקוח טהורה."""
    param_fields = []
    for pname, pdefault in params:
        param_fields.append(html.Div(className="indicator-param", children=[
            html.Span(pname, className="indicator-param-label"),
            dcc.Input(id=f"ind-{key}-{pname}", type="number", value=pdefault,
                      className="indicator-param-input"),
        ]))
    return html.Div(className="indicator-row", **{"data-key": key}, children=[
        html.Span(label, className="indicator-row-name"),
        html.Div(className="indicator-row-params", children=param_fields),
        html.Button("Add", id=f"ind-{key}-add", className="indicator-add-btn"),
        html.Button("✕", id=f"ind-{key}-remove", className="indicator-remove-btn",
                    style={"display": "none"}, title=f"Remove {label}"),
    ])


def _strategy_inputs_section(code: str):
    """בונה את תוכן "Strategy Inputs" בלשונית Settings, לפי משתני "# input" שזוהו
    בקוד ה-Pine Editor הנוכחי (ראו strategy.parse_strategy_inputs). כל שדה הוא
    רכיב Dash אמיתי (לא HTML גולמי כמו שאר פאנל הביצועים) - כדי שהערך שהמשתמש
    עורך יהיה קריא בחזרה דרך State({"type":"strategy-input","name":ALL}, ...)
    ב-run_or_switch. ה-id הוא dict תלוי-שם (pattern-matching) כי מספר וזהות
    השדות משתנים לפי הקוד - לא ניתן לדעת אותם מראש בזמן בניית ה-layout.
    """
    inputs = parse_strategy_inputs(code)
    if not inputs:
        return html.Div("לא זוהו משתני \"# input\" בקוד הנוכחי.", className="settings-hint")

    fields = []
    for item in inputs:
        name, value, type_name = item["name"], item["value"], item["type"]
        label = name if item["recognized"] else f"{name} (לא נתמך במנוע המהיר)"
        field_id = {"type": "strategy-input", "name": name}
        if type_name == "bool":
            control = dcc.Checklist(
                id=field_id, options=[{"label": "", "value": "on"}],
                value=["on"] if value else [], className="strategy-input-checkbox",
            )
        else:
            control = dcc.Input(
                id=field_id, type="number", value=value,
                step=1 if type_name == "int" else 0.01,
                className="num-input tiny", debounce=True,
            )
        fields.append(html.Div(className="settings-field", children=[
            html.Span(label, className="ctl-label"), control,
        ]))
    return html.Div(className="settings-grid", children=fields)


def build_layout() -> html.Div:
    return html.Div([
        dcc.Store(id="tf-store", data=DEFAULT_TF),
        dcc.Store(id="chart-store"),
        dcc.Store(id="chart-sink"),
        dcc.Store(id="live-store"),
        dcc.Store(id="live-sink"),
        # מתמלא ב-JS (clientside) מיד בלחיצה על "Add to chart" - רק כדי לשמש
        # כ-Input שמפעיל את run_or_switch למטה; הבקטסט עצמו תמיד רץ על כל
        # ההיסטוריה, לא על טווח התצוגה הנוכחי בגרף.
        dcc.Store(id="run-window-store"),
        dcc.Store(id="panel-open", data=True),
        dcc.Interval(id="live-interval", interval=LIVE_INTERVAL_MS, n_intervals=0),

        html.Div(className="topbar", children=[
            # debounce=True משדר את הערך רק ביציאה מהשדה, כך שלא רצה שאיבה על כל תו.
            # ב-Dash הזה Enter לבדו אינו מבצע commit, ולכן assets/chart.js הופך Enter
            # ל-blur — וכך הקלדת סימבול + Enter טוענת אותו, כמצופה.
            dcc.Input(id="symbol-input", value=DEFAULT_SYMBOL, debounce=True,
                      className="symbol-input", placeholder="Symbol"),
            html.Div(className="tf-group", children=[
                html.Button(tf, id={"type": "tf-btn", "tf": tf},
                            className="tf-btn active" if tf == DEFAULT_TF else "tf-btn")
                for tf in TIMEFRAME_ORDER
            ]),
            html.Div(className="sep"),
            # תפריט "Indicators" - לגמרי בצד לקוח (assets/indicators.js): חישוב
            # האינדיקטורים וציורם כ-series על הגרף לא צריכים סבב לשרת בכלל (בניגוד
            # לבקטסט עצמו) - בדיוק כמו שהגרף הראשי כבר מנוהל ב-JS טהור. השדות
            # (אורך/std) נקראים ישירות מה-DOM בלחיצה על Add, לא דרך State של Dash.
            html.Div(className="indicators-wrap", children=[
                html.Button(id="indicators-btn", className="indicators-btn", children=[
                    _ic_chart_svg(),
                    html.Span("Indicators", className="indicators-btn-label"),
                    html.Span("▾", className="indicators-btn-caret"),
                ]),
                html.Div(id="indicators-menu", className="indicators-menu",
                         style={"display": "none"}, children=[
                             _indicator_row("sma", "SMA", [("length", 20)]),
                             _indicator_row("ema", "EMA", [("length", 20)]),
                             _indicator_row("vwap", "VWAP", []),
                             _indicator_row("rsi", "RSI", [("length", 14)]),
                             _indicator_row("bb", "Bollinger Bands", [("length", 20), ("std", 2)]),
                         ]),
            ]),
            html.Div(className="sep"),
            dcc.Loading(html.Div(id="status", className="status"), type="dot", color="#2962FF"),
        ]),

        html.Div(id="workspace", className="workspace", children=[
            html.Div(className="chart-pane", children=[
                html.Div(id="chart-container"),
                # הספירה-לאחור בפועל היא countdownEl שנוצר דינמית ב-chart.js
                # (ensureChart), לא div סטטי כאן.

                # מוצג ב-JS (assets/chart.js: requestMarkersWindow) כשיש בטווח
                # הנראה יותר עסקאות מ-MAX_VISIBLE_MARKER_TRADES - ראו /api/markers.
                html.Div(id="marker-note", className="marker-note", style={"display": "none"}),

                html.Div(id="goto-modal-overlay", className="goto-modal-overlay", children=[
                    html.Div(id="goto-modal", className="goto-modal", children=[
                        html.Div(className="goto-modal-header", children=[
                            html.Span("Go to", className="goto-modal-title"),
                            html.Button("✕", id="goto-close", className="goto-modal-close"),
                        ]),
                        html.Div(className="goto-tabs", children=[
                            html.Button("Date", id="goto-tab-date", className="goto-tab goto-tab-active"),
                            html.Button("Custom range", id="goto-tab-range", className="goto-tab"),
                        ]),
                        html.Div(id="goto-pane-date", className="goto-pane", children=[
                            html.Div(className="goto-row", children=[
                                dcc.Input(id="goto-date", type="date", className="goto-date-input"),
                                dcc.Input(id="goto-time", type="time", value="00:00", className="goto-time-input"),
                            ]),
                        ]),
                        html.Div(id="goto-pane-range", className="goto-pane", style={"display": "none"}, children=[
                            html.Div(className="goto-row", children=[
                                dcc.Input(id="goto-range-from-date", type="date", className="goto-date-input"),
                                dcc.Input(id="goto-range-from-time", type="time", value="00:00", className="goto-time-input"),
                            ]),
                            html.Div(className="goto-row", children=[
                                dcc.Input(id="goto-range-to-date", type="date", className="goto-date-input"),
                                dcc.Input(id="goto-range-to-time", type="time", value="00:00", className="goto-time-input"),
                            ]),
                        ]),
                        html.Div(className="goto-modal-footer", children=[
                            html.Button("Cancel", id="goto-cancel", className="goto-cancel-btn"),
                            html.Button("Go to", id="goto-submit", className="goto-submit-btn"),
                        ]),
                    ]),
                ]),
            ]),

            # סרגל תחתון מתחת לציר הזמן של הגרף - כמו ב-TradingView: שורה קבועה
            # (לא absolute מעל הגרף כמו קודם) עם כפתורי "Go to"/"חזרה לעכשיו"
            # בזרימה רגילה. שניהם עדיין מזוהים לפי אותם id-ים (goto-btn/
            # reset-live-btn) - assets/chart.js מוצא אותם לפי id ולא לפי מיקום
            # ב-DOM, אז שום לוגיקת JS לא צריכה להשתנות בעקבות ההעברה.
            html.Div(className="chart-bottom-toolbar", children=[
                html.Button(
                    html.Img(src="/assets/goto.svg", className="goto-btn-icon"),
                    id="goto-btn", className="goto-btn", title="Go to date",
                ),
                # מוצג רק כשגללו הרחק מהקצה החי (assets/chart.js:
                # updateResetLiveVisibility, לפי timeScale().scrollPosition());
                # בלחיצה - scrollToRealTime().
                html.Button("⏭", id="reset-live-btn", className="reset-live-btn",
                            title="Back to live", style={"display": "none"}),
                # שעון בזמן אמת בשעון ישראל (Asia/Jerusalem) - עם קיץ/חורף אוטומטי
                # דרך Intl בדפדפן (assets/clock.js), לא offset קבוע (UTC+2 בחורף,
                # UTC+3 בקיץ - קבוע היה מראה שעה שגויה חצי מהשנה).
                html.Div(id="market-clock", className="market-clock"),
            ]),

            html.Div(className="bottom-panel", children=[
                html.Div(className="panel-bar", children=[
                    dcc.Tabs(id="panel-tabs", value="tester", className="panel-tabs",
                             parent_className="panel-tabs-parent", children=[
                                 dcc.Tab(label="Strategy Tester", value="tester",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="Pine Editor", value="editor",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="Settings", value="settings",
                                         className="ptab", selected_className="ptab-sel"),
                             ]),
                    # חץ ספריית האסטרטגיות, ליד לשונית "Pine Editor" - כמו ב-TradingView.
                    # dcc.Tab לא מאפשר להטמיע רכיב בתוך כותרת הלשונית עצמה (label הוא
                    # טקסט בלבד), אז זה אלמנט נפרד הממוקם ב-JS מעל הלשונית (ראו
                    # assets/strategy_library.js: positionArrow) - חייב .panel-bar
                    # עם position:relative בשביל זה (ראו style.css).
                    html.Div("▾", id="strategy-lib-arrow", className="strategy-lib-arrow",
                             title="Strategy library"),
                    html.Div(id="strategy-lib-menu", className="strategy-lib-menu",
                             style={"display": "none"}),
                    html.Button("⌄", id="panel-toggle", n_clicks=0, className="panel-toggle",
                                title="פתח/סגור את הפאנל"),
                ]),
                html.Div(id="panel-body", className="panel-body", children=[
                    html.Div(id="tester-pane", children=[
                        # dcc.Loading עוטף את שני ה-Output-ים של run_or_switch (למטה) -
                        # Dash מזהה לבד קריאת callback פעילה שמשפיעה על ילד עטוף ומציג
                        # ספינר אוטומטית. בלי זה, לחיצה על "Add to chart" לא נתנה שום
                        # משוב חזותי במשך כל משך הריצה (בקטסט על 7 שנות MNQ/1m יכול
                        # לקחת כ-3 דקות, backtesting.py מריץ בר-אחר-בר בפייתון, לא
                        # מוקטר) - בדיוק מה שנראה כאילו "הכפתור לא עושה כלום".
                        dcc.Loading(html.Div(id="key-stats"), type="dot", color="#2962FF"),
                        dcc.Tabs(id="result-tabs", value="overview", className="result-tabs",
                                 children=[
                                     dcc.Tab(label="Overview", value="overview",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="Performance Summary", value="summary",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="List of Trades", value="trades",
                                             className="rtab", selected_className="rtab-sel"),
                                 ]),
                        # פילטר תקופה מעל גרף ה-Performance - מוצג רק בלשונית overview
                        # (ראו run_or_switch: Output("perf-period-bar","style")). Custom Range
                        # פותח שני שדות תאריך (perf-custom-range) - טוגל טהור ב-JS, בלי
                        # סבב לשרת (ראו clientside_callback למטה).
                        dcc.Store(id="perf-period", data="range"),
                        html.Div(id="perf-period-bar", className="perf-period-bar", children=[
                            dcc.Dropdown(
                                id="perf-period-select",
                                options=[{"label": label, "value": key} for key, label in PERIOD_LABELS.items()],
                                value="range", clearable=False, searchable=False,
                                className="perf-period-dropdown",
                            ),
                            html.Div(id="perf-custom-range", className="perf-custom-range",
                                     style={"display": "none"}, children=[
                                         dcc.Input(id="perf-custom-from", type="date",
                                                   className="perf-custom-date"),
                                         html.Span("\u2192", className="perf-custom-arrow"),
                                         dcc.Input(id="perf-custom-to", type="date",
                                                   className="perf-custom-date"),
                                     ]),
                        ]),
                        dcc.Loading(html.Div(id="result-body", className="result-body",
                                 children=_html(panels.empty(
                                     "פתח את Pine Editor, כתוב אסטרטגיה ולחץ "
                                     "\u25b6 Add to chart כדי לראות ביצועים."))),
                                    type="dot", color="#2962FF"),
                        # ניווט עמודים ל"List of Trades" - מוצג רק בלשונית trades וכשיש
                        # יותר מ-TRADES_PAGE_SIZE עסקאות (ראו run_or_switch/_trades_pagination_state).
                        dcc.Store(id="trades-page", data=0),
                        html.Div(id="trades-pagination", className="trades-pagination",
                                 style={"display": "none"}, children=[
                                     html.Button("\u2039 Prev", id="trades-prev-btn", n_clicks=0,
                                                 className="trades-page-btn"),
                                     html.Span(id="trades-page-info", className="trades-page-info"),
                                     html.Button("Next \u203a", id="trades-next-btn", n_clicks=0,
                                                 className="trades-page-btn"),
                                 ]),
                    ]),
                    html.Div(id="editor-pane", style={"display": "none"}, children=[
                        html.Div(className="editor-row", children=[
                            html.Div("כתוב def init(self) ו-def next(self). "
                                     "זמינים: self.vwap (מתאפס כל יום), self.position_size, "
                                     "sma / ema / rsi / crossover.", className="editor-hint"),
                            html.Div(className="editor-controls", children=[
                                html.Button("▶  Add to chart", id="run-btn", n_clicks=0,
                                            className="run-btn",
                                            title="בקטסט על כל ההיסטוריה"),
                            ]),
                        ]),
                        dcc.Textarea(id="code-input", value=DEFAULT_CODE, className="code-area",
                                     spellCheck=False),
                        html.Div(id="run-error", className="run-error"),
                    ]),
                    html.Div(id="settings-pane", style={"display": "none"}, children=[
                        html.Div(className="settings-hint",
                                 children="הגדרות חשבון וביצוע — משפיעות על חישוב הבקטסט "
                                          "(PnL, סליפג' ועמלות) בלחיצה הבאה על Add to chart."),
                        html.Div(className="settings-grid", children=[
                            html.Div(className="settings-field", children=[
                                html.Span("Account Value", className="ctl-label"),
                                dcc.Input(id="account-value-input", type="number", value=100000,
                                          min=1000, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("Contracts per trade", className="ctl-label"),
                                dcc.Input(id="contracts-input", type="number", value=1, min=1,
                                          step=1, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("Contract multiplier", className="ctl-label"),
                                dcc.Input(id="multiplier-input", type="number", value=2, min=0,
                                          step=0.01, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("Slippage (ticks)", className="ctl-label"),
                                dcc.Input(id="slippage-input", type="number", value=2, min=0,
                                          step=1, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("Commission / contract", className="ctl-label"),
                                dcc.Input(id="settings-commission-input", type="number", value=0,
                                          min=0, step=0.01, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("Order type", className="ctl-label"),
                                dcc.Dropdown(id="order-type-input", className="settings-dropdown",
                                             options=[{"label": "Market", "value": "Market"},
                                                      {"label": "Limit", "value": "Limit"}],
                                             value="Market", clearable=False, searchable=False),
                            ]),
                        ]),
                        # "Strategy Inputs" - שדות שזוהו אוטומטית מ-"# input" בקוד ה-Pine
                        # Editor (ראו strategy.parse_strategy_inputs/_strategy_inputs_section).
                        # מתעדכן בכל שינוי בקוד (Input("code-input","value") למטה); הערכים
                        # שהמשתמש עורך כאן נקראים ב-run_or_switch דרך pattern-matching
                        # State ומשפיעים בפועל רק על שמות מוכרים (RECOGNIZED_STRATEGY_INPUTS) -
                        # שאר השדות מוצגים לשקיפות בלבד, בלי אפקט על המנוע המהיר עדיין.
                        html.Div(className="settings-section-title", children="Strategy Inputs"),
                        html.Div(id="strategy-inputs-container",
                                 children=_strategy_inputs_section(DEFAULT_CODE)),
                    ]),
                ]),
            ]),
        ]),

        # מודל "שם אסטרטגיה" - נפתח מ-"Save current strategy"/"Save as new strategy"
        # בתפריט ספריית האסטרטגיות (ראו strategy-lib-menu למעלה). מבנה זהה בכוונה
        # ל-goto-modal-overlay (overlay קבוע + תיבה, נפתח/נסגר ע"י class "open"),
        # רק עם שדה טקסט יחיד במקום תאריך/טווח.
        html.Div(id="strategy-save-modal-overlay", className="strategy-save-modal-overlay", children=[
            html.Div(className="strategy-save-modal", children=[
                html.Div(className="strategy-save-modal-header", children=[
                    html.Span("Save strategy", className="strategy-save-modal-title"),
                    html.Button("✕", id="strategy-save-close", className="strategy-save-modal-close"),
                ]),
                dcc.Input(id="strategy-save-name-input", type="text", placeholder="Strategy name",
                          maxLength=80, className="strategy-save-name-input"),
                html.Div(className="strategy-save-modal-footer", children=[
                    html.Button("Cancel", id="strategy-save-cancel", className="goto-cancel-btn"),
                    html.Button("Save", id="strategy-save-confirm", className="goto-submit-btn"),
                ]),
            ]),
        ]),
    ])


app = Dash(
    __name__,
    title="Strategy Dashboard",
    external_scripts=[
        # גרסה מוצמדת בכוונה: גרסה צפה עלולה לקפוץ ל-major חדש ולשבור את ה-API.
        "https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"
    ],
)
app.layout = build_layout()


# ---------------------------------------------------------------- סרגל עליון
@callback(
    Output("tf-store", "data"),
    Output({"type": "tf-btn", "tf": ALL}, "className"),
    Input({"type": "tf-btn", "tf": ALL}, "n_clicks"),
    State("tf-store", "data"),
    prevent_initial_call=True,
)
def pick_timeframe(_clicks, current):
    trigger = callback_context.triggered_id
    tf = trigger["tf"] if trigger else current
    return tf, ["tf-btn active" if t == tf else "tf-btn" for t in TIMEFRAME_ORDER]


@callback(
    Output("chart-store", "data"),
    Output("status", "children"),
    Output("status", "className"),
    Input("symbol-input", "value"),
    Input("tf-store", "data"),
)
def load_chart(symbol, timeframe):
    """שאיבה מ-IB בכל שינוי סימבול או טיים-פריים, וציור מחדש."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return no_update, "הזן סימבול", "status error"
    try:
        df = load_ohlcv_preferring_1m_cache(symbol, timeframe, days=INITIAL_DAYS[timeframe], extended_hours=True)
    except DataError as exc:
        return no_update, str(exc), "status error"

    _FRAMES[(symbol, timeframe)] = df
    _RESULTS.pop("last", None)
    live.subscribe(symbol, timeframe)  # מעביר את מנוי ה-streaming לסימבול/טיים-פריים החדשים
    years = (df.index.max() - df.index.min()).days / 365.25
    status = (f"{len(df):,} bars · {df.index.min():%b %d, %Y} → {df.index.max():%b %d, %Y} "
              f"({years:.1f}y)")
    return build_payload(df, symbol, timeframe), status, "status"


@callback(
    Output("live-store", "data"),
    Input("live-interval", "n_intervals"),
    State("symbol-input", "value"),
    State("tf-store", "data"),
    prevent_initial_call=True,
)
def push_live_update(_n, symbol, timeframe):
    """טיק תקופתי (כל 200 מ"ש, ראו LIVE_INTERVAL_MS): קורא את הנר הפתוח מהזיכרון
    של dashboard/live.py — לא נוגע ב-IB בכלל, reqHistoricalData(keepUpToDate=True)
    כבר רץ ברקע ומעדכן את _bar בכל פעם ש-IB דוחף עדכון לבר האחרון (ראו live.py:
    on_bar_update). לכן אפשר לקרוא לזה בלי שום עלות."""
    symbol = (symbol or "").strip().upper()
    if not symbol or (symbol, timeframe) not in _FRAMES:
        return no_update
    bar = live.get_live_bar(symbol, timeframe)
    payload = build_live_bar_payload(bar, symbol, timeframe) if bar else {
        "symbol": symbol, "timeframe": timeframe, "candles": [], "volume": []
    }
    # הטיימר מוצג רק כשהשוק "חי" בפועל (טיק אמיתי הגיע לאחרונה - ראו
    # live.is_market_live) - לא לפי חישוב שעון-לוח קבוע כמו קודם, שהראה ספירה
    # גם כששוק סגור/TWS מנותק ואין שום טיק בפועל. marketOpen=False אומר ל-JS
    # (assets/chart.js: updateLive) להסתיר את הטיימר לגמרי; חוזר אוטומטית
    # ברגע שטיק חדש מגיע (marketOpen שוב True בקריאה הבאה).
    market_live = live.is_market_live(symbol, timeframe)
    payload["marketOpen"] = market_live
    if market_live:
        payload["secondsToClose"] = live.seconds_to_close(timeframe)
    if bar:
        print(f"Sending live bar to chart: close={bar['close']}", flush=True)
    return payload


def _lod_window_payload(symbol: str, tier: str, center_epoch: int, span: float, seq) -> dict:
    """מדרג-פירוט (LOD): assets/chart.js קורא לזה (דרך /api/lod, לא Dash callback -
    ראו למטה) בכל שינוי זום/גלילה עם הרזולוציה (tier) שמתאימה לרוחב הטווח הנראה
    כרגע ומרכז (center) - לא כל ההיסטוריה בבת אחת, רק חלון סביב מה שבאמת נראה
    על המסך (עם רווח נוסף לגלילה קלה). ככה שום פעולה לא דוחפת מיליוני נרות
    לדפדפן: מזוגם רחוק => tier גס (למשל 1d) עם מעט נקודות; מתקרבים => tier
    עדין (עד 1m) אבל רק לחלון הזמן הצר שבתצוגה.
    """
    center = from_display_epoch(center_epoch)
    # buffer גדול מהטווח הנראה עצמו (לא רק 1.5x) - כדי שגלילה רציפה באותו כיוון
    # תמשיך לקבל דאטה שכבר טעון בלי לבקש שוב על כל תזוזה קטנה. זו הסיבה
    # שגלילה "הרגישה תקועה" - כל תזוזה קטנה מעבר לחיץ הצר הקודם דרשה סבב חדש.
    buffer = pd.Timedelta(seconds=span * 3)
    from_ts = center - buffer
    end_dt = center + buffer
    print(f"[lod] request symbol={symbol} tier={tier} seq={seq} "
          f"center_epoch={center_epoch} center={center} span={span}s "
          f"from_ts={from_ts} end_dt={end_dt}", flush=True)
    # בלי ההגבלה הזו, כל בקשה סביב "עכשיו" (הנפוצה ביותר) הייתה מבקשת עד לתוך
    # העתיד - ו-fetch_ohlcv (need_tail: anchor > cached_max) היה מפרש את זה
    # כ"תמיד חסר נתונים חדשים", מתחבר בפועל ל-IB בכל בקשת LOD במקום לענות
    # ישירות מהקאש. אין דאטה מעבר לרגע האמיתי בכל מקרה.
    now = pd.Timestamp.now()
    if end_dt > now:
        end_dt = now

    empty_payload = {"symbol": symbol, "tier": tier, "seq": seq, "candles": [], "volume": []}

    # קודם מנסים לדגום מתוך קאש ה-1m הקיים (מהיר, בלי לגעת ב-IB בכלל) - חשוב
    # במיוחד לחוזים עתידיים שבהם רק קאש ה-1m עמוק (ראו load_lod_window_from_1m_cache).
    # רק אם אין שם כלום לחלון הזה נופלים חזרה לשאיבה הטבעית של ה-tier עצמו.
    df = load_lod_window_from_1m_cache(symbol, tier, use_rth=False, from_ts=from_ts, to_ts=end_dt)
    print(f"[lod]   1m-cache-resample -> {'MISS' if df is None or df.empty else f'{len(df)} rows'}", flush=True)
    if df is None or df.empty:
        days = max(1, int((buffer * 2).total_seconds() // 86400) + 1)
        try:
            df = load_ohlcv(symbol, tier, days=days, end_date=end_dt, extended_hours=True)
            print(f"[lod]   native load_ohlcv(days={days}) -> {len(df)} rows", flush=True)
        except DataError as exc:
            print(f"[lod]   native load_ohlcv failed: {exc}", flush=True)
            return empty_payload
    if df is None or df.empty:
        print("[lod]   -> EMPTY payload returned", flush=True)
        return empty_payload
    print(f"[lod]   -> returning {len(df)} rows, {df.index.min()} -> {df.index.max()}", flush=True)

    # build_history_chunk_payload שם את הטיים-פריים במפתח "timeframe" - assets/chart.js
    # קורא כאן "tier" (זו לא בהכרח currentTimeframe הרשמי, ראו applyLodWindow), אז
    # מוסיפים אותו במפורש בנוסף.
    payload = build_history_chunk_payload(df, symbol, tier)
    payload["tier"] = tier
    payload["seq"] = seq
    return payload


@app.server.route("/api/lod")
def api_lod():
    """נתיב Flask ישיר (לא Dash callback) לבקשות LOD - עוקף את מנגנון ה-callback
    של Dash (Store, כפתור מוסתר, סבב serialization של כל עץ ה-state) בשביל הנתיב
    החם ביותר באפליקציה: כל תזוזת עכבר/גלגלת. assets/chart.js קורא לזה ישירות
    עם fetch() - סבב הלוך-חזור קצר משמעותית מלחיצה על כפתור מוסתר + Store.
    """
    args = flask.request.args
    symbol = (args.get("symbol") or "").strip().upper()
    tier = args.get("tier")
    # int(...): args.get מחזיר תמיד מחרוזת (זה query string) - assets/chart.js
    # משווה payload.seq !== lodRequestSeq בהשוואה מחמירה (===), ומספר לעולם לא
    # שווה למחרוזת שלו ב-JS. בלי ההמרה כאן כל תשובת LOD הייתה נפסלת בשקט כ"מיושנת"
    # (seq לא תואם), גם כשהיא בדיוק העדכנית ביותר - זה בדיוק מה שגרם לגלילה אחורה
    # להיראות "שבורה לגמרי" אחרי המעבר מ-Dash Store (ששמר את הטיפוס המספרי) ל-fetch().
    try:
        seq = int(args.get("seq"))
    except (TypeError, ValueError):
        seq = args.get("seq")
    empty = {"symbol": symbol, "tier": tier, "seq": seq, "candles": [], "volume": []}
    try:
        center_epoch = int(args.get("center"))
        span = float(args.get("span"))
    except (TypeError, ValueError):
        return flask.jsonify(empty)
    if not symbol or not tier:
        return flask.jsonify(empty)
    return flask.jsonify(_lod_window_payload(symbol, tier, center_epoch, span, seq))


@app.server.route("/api/markers")
def api_markers():
    """נתיב Flask ישיר (כמו /api/lod) לסימוני-עסקאות: assets/chart.js קורא לזה
    בכל שינוי טווח נראה בגרף (requestMarkersWindow), ומקבל רק את העסקאות שחופפות
    לחלון הזה - לא את כל 156K+ העסקאות של "Full backtest" בבת אחת. כל 156K+
    העסקאות עצמן נשארות בזיכרון השרת (_RESULTS["last"]) לכל אורך החיים של
    התוצאה; כאן רק מסננים תת-קבוצה קטנה בכל בקשה.
    """
    args = flask.request.args
    symbol = (args.get("symbol") or "").strip().upper()
    timeframe = args.get("timeframe") or ""
    try:
        seq = int(args.get("seq"))
    except (TypeError, ValueError):
        seq = args.get("seq")
    empty = {"symbol": symbol, "timeframe": timeframe, "seq": seq, "markers": [], "note": ""}

    saved = _RESULTS.get("last")
    if not saved:
        return flask.jsonify(empty)
    stats, _cash, saved_symbol, saved_timeframe = saved
    # תוצאה של סימבול/טיים-פריים אחר (המשתמש החליף בינתיים) - לא רלוונטית לגרף הנוכחי.
    if symbol != saved_symbol or timeframe != saved_timeframe:
        return flask.jsonify(empty)

    try:
        from_epoch = int(args.get("from"))
        to_epoch = int(args.get("to"))
    except (TypeError, ValueError):
        return flask.jsonify(empty)

    trades = stats["_trades"]
    if trades.empty:
        return flask.jsonify(empty)

    from_ts = from_display_epoch(from_epoch)
    to_ts = from_display_epoch(to_epoch)
    # חפיפה עם החלון הנראה, לא רק כניסה/יציאה שלמות בתוכו - עסקה שנפתחה לפני
    # הגלילה הנוכחית ועוד לא נסגרה (או להפך) עדיין צריכה להופיע.
    in_view = trades[(trades["EntryTime"] <= to_ts) & (trades["ExitTime"] >= from_ts)]

    note = ""
    if len(in_view) > MAX_VISIBLE_MARKER_TRADES:
        total_in_view = len(in_view)
        idx = _evenly_sampled_indices(total_in_view, MAX_VISIBLE_MARKER_TRADES)
        in_view = in_view.iloc[idx]
        note = (f"Showing {len(in_view):,} of {total_in_view:,} trades in this view "
                "- zoom in for more detail")

    markers = to_markers(in_view, _epoch) if len(in_view) else []
    return flask.jsonify({"symbol": symbol, "timeframe": timeframe, "seq": seq,
                          "markers": markers, "note": note})


# ------------------------------------------------------ ספריית אסטרטגיות (Pine Editor)
@app.server.route("/api/strategies")
def api_strategies_list():
    """כל האסטרטגיות השמורות (שם/קוד/הגדרות) - assets/strategy_library.js קורא
    לזה בכל פתיחה של התפריט ליד לשונית Pine Editor."""
    with _STRATEGIES_LOCK:
        strategies = _load_strategies()
    return flask.jsonify({"strategies": strategies})


@app.server.route("/api/strategies/save", methods=["POST"])
def api_strategies_save():
    """שומר/מחליף אסטרטגיה לפי שם (upsert) - גם "Save current strategy" וגם
    "Save as new strategy" בתפריט קוראים לנתיב הזה; ה-JS שולח את השם שהמשתמש
    הזין בתיבת הטקסט (ראו strategy-save-name-input)."""
    body = flask.request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return flask.jsonify({"ok": False, "error": "שם ריק"}), 400
    entry = {
        "name": name,
        "code": body.get("code") or "",
        "settings": body.get("settings") or {},
    }
    with _STRATEGIES_LOCK:
        strategies = [s for s in _load_strategies() if s.get("name") != name]
        strategies.append(entry)
        strategies.sort(key=lambda s: (s.get("name") or "").lower())
        _save_strategies(strategies)
    return flask.jsonify({"ok": True, "strategies": strategies})


@app.server.route("/api/strategies/delete", methods=["POST"])
def api_strategies_delete():
    """מוחק אסטרטגיה שמורה לפי שם - "Delete strategy" בתפריט."""
    body = flask.request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    with _STRATEGIES_LOCK:
        strategies = [s for s in _load_strategies() if s.get("name") != name]
        _save_strategies(strategies)
    return flask.jsonify({"ok": True, "strategies": strategies})


# ---------------------------------------------------------------- פאנל תחתון
@callback(
    Output("panel-open", "data"),
    Output("workspace", "className"),
    Output("panel-toggle", "children"),
    Input("panel-toggle", "n_clicks"),
    State("panel-open", "data"),
    prevent_initial_call=True,
)
def toggle_panel(_n, is_open):
    is_open = not is_open
    return is_open, ("workspace" if is_open else "workspace collapsed"), ("⌄" if is_open else "⌃")


@callback(
    Output("tester-pane", "style"),
    Output("editor-pane", "style"),
    Output("settings-pane", "style"),
    Input("panel-tabs", "value"),
)
def switch_panel_tab(tab):
    show, hide = {"display": "block"}, {"display": "none"}
    return (
        show if tab == "tester" else hide,
        show if tab == "editor" else hide,
        show if tab == "settings" else hide,
    )


@callback(
    Output("strategy-inputs-container", "children"),
    Input("code-input", "value"),
)
def update_strategy_inputs(code):
    """מרענן את "Strategy Inputs" (לשונית Settings) בכל שינוי בקוד ה-Pine Editor -
    ראו _strategy_inputs_section. שים לב: זה בונה את השדות מחדש מהערכים
    שבטקסט הקוד עצמו, כך שעריכת ערך שדה ואז שינוי חלק *אחר* של הקוד תאפס אותו
    שדה בחזרה לברירת המחדל שבטקסט - מגבלה מוכרת, לא נפתרה כאן."""
    return _strategy_inputs_section(code or "")


def _trades_pagination_state(tab: str, trade_count: int, page: int):
    """סטייל/טקסט ניווט העמודים ל"List of Trades" - מוצג רק בלשונית הזו, ורק
    כשבאמת יש יותר מעמוד אחד."""
    if tab != "trades" or trade_count <= TRADES_PAGE_SIZE:
        return {"display": "none"}, ""
    n_pages = max(1, -(-trade_count // TRADES_PAGE_SIZE))  # ceil division
    return ({"display": "flex"},
            f"Page {page + 1} / {n_pages} ({trade_count:,} trades)")


def _perf_period_bar_style(tab: str):
    """פילטר התקופה רלוונטי רק לגרף ה-Overview - מוסתר בשאר הלשוניות."""
    return {"display": "flex"} if tab == "overview" else {"display": "none"}


def _filter_trades_by_period(trades: pd.DataFrame, period: str,
                             custom_from: str | None, custom_to: str | None) -> pd.DataFrame:
    """מסנן עסקאות לתקופה שנבחרה בפילטר מעל גרף ה-Performance (Overview) -
    "range"/"all" שקולים (ראו PERIOD_LABELS למעלה - הבקטסט תמיד רץ על כל הקאש).
    מסננים לפי ExitTime (מתי העסקה נסגרה בפועל, לא מתי נפתחה) - זו הנקודה שבה
    ה-PnL שלה נכנס בפועל ל-equity, אותה מוסכמה בדיוק כמו build_performance_chart
    (ordered = trades.sort_values("ExitTime")). "Last N days" נמדד מהעסקה
    האחרונה בדאטה (לא מ-"היום" בפועל) - הדאטה עצמו כבר מתעדכן live עד עכשיו,
    אז זה בפועל אותו דבר, אבל נשאר נכון גם אם ההיסטוריה לא מגיעה עד היום."""
    if trades.empty or not period or period in ("range", "all"):
        return trades
    if period == "custom":
        if not custom_from or not custom_to:
            return trades
        try:
            start = pd.Timestamp(custom_from)
            end = pd.Timestamp(custom_to) + pd.Timedelta(days=1)  # כולל את כל יום ה-"עד"
        except (ValueError, TypeError):
            return trades
        return trades[(trades["ExitTime"] >= start) & (trades["ExitTime"] < end)]
    days = _PERIOD_DAYS.get(period)
    if days is None:
        return trades
    anchor = trades["ExitTime"].max()
    start = anchor - pd.Timedelta(days=days)
    return trades[trades["ExitTime"] >= start]


def _key_stats_body(stats, cash: float, period: str, custom_from, custom_to):
    """Key stats תמיד לפי הפילטר הנוכחי - גם כשלא נמצאים בלשונית Overview
    (הכותרת יושבת מעל הלשוניות, לא בתוכן, ראו requirement המפורש)."""
    trades = stats["_trades"]
    if trades.empty or not period or period in ("range", "all"):
        return _html(panels.key_stats(stats, cash))
    filtered = _filter_trades_by_period(trades, period, custom_from, custom_to)
    label = PERIOD_LABELS.get(period, period)
    return _html(panels.key_stats_for_period(filtered, cash, len(trades), label))


# ---------------------------------------------------------------- הרצת אסטרטגיה
@callback(
    Output("chart-store", "data", allow_duplicate=True),
    Output("key-stats", "children"),
    Output("result-body", "children"),
    Output("run-error", "children"),
    Output("panel-tabs", "value"),
    Output("trades-page", "data"),
    Output("trades-pagination", "style"),
    Output("trades-page-info", "children"),
    Output("perf-period-bar", "style"),
    Input("run-window-store", "data"),
    Input("result-tabs", "value"),
    Input("trades-prev-btn", "n_clicks"),
    Input("trades-next-btn", "n_clicks"),
    Input("perf-period-select", "value"),
    Input("perf-custom-from", "value"),
    Input("perf-custom-to", "value"),
    State("symbol-input", "value"), State("tf-store", "data"),
    State("code-input", "value"),
    # מקור-האמת היחיד להגדרות הבקטסט הוא לשונית Settings - Pine Editor כבר לא
    # מכיל שדות Qty/Capital/Comm% משלו (הוסרו; ראו לשונית ההיסטוריה).
    State("account-value-input", "value"), State("contracts-input", "value"),
    State("multiplier-input", "value"), State("slippage-input", "value"),
    State("settings-commission-input", "value"), State("order-type-input", "value"),
    # "Strategy Inputs" (Settings) - pattern-matching כי מספר/זהות השדות תלויים
    # בקוד ה-Pine Editor הנוכחי, לא ידועים מראש (ראו _strategy_inputs_section).
    State({"type": "strategy-input", "name": ALL}, "value"),
    State({"type": "strategy-input", "name": ALL}, "id"),
    State("trades-page", "data"),
    prevent_initial_call=True,
)
def run_or_switch(_window_data, result_tab, _prev_n, _next_n, period, custom_from, custom_to,
                   symbol, timeframe, _code,
                   account_value, contracts, multiplier, slippage_ticks,
                   commission_per_contract, order_type,
                   strategy_input_values, strategy_input_ids, current_page):
    """מריץ בקטסט על כל ההיסטוריה (run-window-store, מ-"Add to chart" - כן, למרות
    השם: לפי בקשה מפורשת "Add to chart" רץ תמיד על כל הקאש), או רק מציג תוצאות
    שכבר קיימות ב-_RESULTS["last"] אחרי החלפת לשונית-תוצאות רגילה/דפדוף בטבלת
    העסקאות/שינוי פילטר התקופה מעל גרף ה-Overview. סינכרוני כאן (לא thread נפרד)
    בטוח: המנוע הווקטורי (run_fast_sma_crossover_backtest) מסיים תוך כ-2-3
    שניות על 2.5 מיליון ברים, לא כמה דקות כמו backtesting.py הכללי - חסימה
    קצרה כזו של ה-callback סבירה, לא צריך תשתית job/polling נפרדת.

    שים לב: period/custom_from/custom_to משפיעים *רק* על key-stats ועל גרף
    ה-Overview (דרך _key_stats_body/_result_body) - הבקטסט עצמו (df_backtest,
    למטה) תמיד רץ על כל ההיסטוריה בלי קשר לפילטר, ו-Performance Summary/List
    of Trades (בתוך _result_body) גם הם תמיד מציגים את כל ההיסטוריה."""
    triggered = callback_context.triggered_id
    symbol = (symbol or "").strip().upper()
    df = _FRAMES.get((symbol, timeframe))
    cash = float(account_value or 100_000)
    size = float(contracts or 1)
    perf_bar_style = _perf_period_bar_style(result_tab)

    # "Strategy Inputs": מיישמים רק שמות מוכרים (RECOGNIZED_STRATEGY_INPUTS) על
    # המנוע המהיר שנבחר - ראו _strategy_inputs_section להסבר המלא. dcc.Checklist
    # (בוליאני) מחזיר רשימה (["on"]/[]), לא ערך יחיד.
    strategy_overrides = {
        id_dict.get("name"): (bool(val) if isinstance(val, list) else val)
        for val, id_dict in zip(strategy_input_values, strategy_input_ids)
        if isinstance(id_dict, dict) and id_dict.get("name")
    }

    def _override(engine_param: str, default, caster=float):
        """קלט מוכר בודד: RECOGNIZED_STRATEGY_INPUTS ממפה כמה שמות-משתנה
        אפשריים לאותו engine_param (למשל SMA_LENGTH/SMA_PERIOD -> sma_period) -
        לוקחים את הראשון שבאמת מופיע בקוד הנוכחי."""
        for name, param in RECOGNIZED_STRATEGY_INPUTS.items():
            if param != engine_param:
                continue
            raw = strategy_overrides.get(name)
            if raw is None:
                continue
            try:
                return caster(raw)
            except (TypeError, ValueError):
                continue
        return default

    # בוחרים איזה מנוע וקטורי להריץ לפי שורת-הסימון הראשונה בקוד (# strategy:
    # <key>) - לא מריצים את הקוד עצמו. בלי סימון (DEFAULT_CODE, או אסטרטגיה
    # ישנה שנשמרה לפני התכונה הזו) - ברירת המחדל היא חציית-SMA, כמו קודם.
    builtin_key = detect_builtin_strategy(_code)

    if triggered == "run-window-store":
        # תמיד כל ההיסטוריה - לא רק החלון הנראה בגרף (ראו הערת הפונקציה למעלה).
        df_backtest = load_full_history(symbol, timeframe, extended_hours=True)
        source = "load_full_history"
        # לא "or df": load_full_history יכולה להחזיר DataFrame אמיתי (לא
        # None), ו-pandas אוסר bool() מרומז על DataFrame ("truth value is
        # ambiguous") - "X or Y" חייב להעריך את bool(X) כדי להחליט, אז זה קורס
        # בכל פעם שיש קאש עמוק אמיתי - בדיוק למה הכפתור "לא עשה כלום" בעבר.
        if df_backtest is None:
            df_backtest = df
            source = "_FRAMES fallback (load_full_history returned None - no 1m cache for this symbol)"
        if df_backtest is None:
            return (no_update, no_update, no_update, "טען נתונים קודם", no_update,
                    no_update, no_update, no_update, no_update)
        print(f"[backtest] source={source} (button='Add to chart') "
              f"symbol={symbol} timeframe={timeframe} "
              f"Backtest running on {len(df_backtest):,} bars from "
              f"{df_backtest.index.min():%Y-%m-%d} to {df_backtest.index.max():%Y-%m-%d}", flush=True)
        common_kwargs = dict(
            cash=cash, size=size,
            multiplier=float(multiplier or 1),
            slippage_ticks=float(slippage_ticks or 0),
            commission_per_contract=float(commission_per_contract or 0),
            market_order=(order_type != "Limit"),
        )
        try:
            if builtin_key == "hammer_rsi":
                stats = run_hammer_rsi_backtest(
                    df_backtest, **common_kwargs,
                    rsi_period=int(_override("rsi_period", 14)),
                    rsi_oversold=_override("rsi_oversold", 30.0),
                    rsi_overbought=_override("rsi_overbought", 70.0),
                    hammer_ratio=_override("hammer_ratio", 2.0),
                    stop_loss_pts=_override("stop_loss_pts", 50.0),
                    take_profit_pts=_override("take_profit_pts", 100.0),
                )
            else:
                stats = run_fast_sma_crossover_backtest(
                    df_backtest, **common_kwargs,
                    sma_period=int(_override("sma_period", 20)),
                )
        except StrategyCompileError as exc:
            return (no_update, no_update, no_update, str(exc), no_update,
                    no_update, no_update, no_update, no_update)
        _RESULTS["last"] = (stats, cash, symbol, timeframe)
        trade_count = len(stats["_trades"])
        # לא שולחים סימונים כאן בכלל - assets/chart.js מבקש אותם לפי דרישה
        # דרך /api/markers ברגע שהוא רואה backtestReady, על הטווח הנראה כרגע
        # בגרף, ואז שוב בכל שינוי טווח (בדיוק כמו LOD לנרות עצמם).
        payload = {"symbol": symbol, "timeframe": timeframe, "candles": [], "volume": [],
                   "markers": [], "backtestReady": True}
        page = 0
        pag_style, page_info = _trades_pagination_state(result_tab, trade_count, page)
        # אחרי הרצה עוברים אוטומטית ללשונית התוצאות — זה מה שרוצים לראות עכשיו.
        return (payload, _key_stats_body(stats, cash, period, custom_from, custom_to),
                _result_body(result_tab, stats, cash, page=page,
                             period=period, custom_from=custom_from, custom_to=custom_to),
                "", "tester", page, pag_style, page_info, perf_bar_style)

    saved = _RESULTS.get("last")
    if not saved:
        return (no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update)
    stats, cash, _saved_symbol, _saved_timeframe = saved
    trade_count = len(stats["_trades"])
    page = current_page or 0
    n_pages = max(1, -(-trade_count // TRADES_PAGE_SIZE))
    if triggered == "trades-next-btn":
        page = min(page + 1, n_pages - 1)
    elif triggered == "trades-prev-btn":
        page = max(page - 1, 0)
    # triggered == "result-tabs"/"perf-period-select"/"perf-custom-from"/"perf-custom-to":
    # מחליפים לשונית/תקופה בלבד, נשארים על אותו עמוד.

    pag_style, page_info = _trades_pagination_state(result_tab, trade_count, page)
    body = _result_body(result_tab, stats, cash, page=page,
                        period=period, custom_from=custom_from, custom_to=custom_to)
    key_stats_out = _key_stats_body(stats, cash, period, custom_from, custom_to)
    # לשונית-תוצאות/עמוד/תקופה - רק מרנדרים את מה שכבר קיים, בלי לגעת בגרף/בלשונית העליונה.
    return (no_update, key_stats_out, body, no_update, no_update,
            page, pag_style, page_info, perf_bar_style)


def _result_body(tab: str, stats, cash: float, page: int = 0,
                 period: str = "range", custom_from: str | None = None, custom_to: str | None = None):
    if tab == "summary":
        return _html(panels.performance_summary(stats, cash))
    if tab == "trades":
        return _html(panels.trades_list(stats, page=page, page_size=TRADES_PAGE_SIZE))

    # tab == "overview": כאן בלבד מוחל פילטר התקופה - Performance Summary/List
    # of Trades תמיד מציגים את כל ההיסטוריה (ראו requirement המפורש).
    trades = stats["_trades"]
    if trades.empty:
        return _html(panels.empty("לא בוצעו עסקאות."))
    filtered = _filter_trades_by_period(trades, period, custom_from, custom_to)
    if filtered.empty:
        return _html(panels.empty("אין עסקאות בטווח התאריכים שנבחר."))
    fig = build_performance_chart(filtered, start_time=filtered["EntryTime"].min())
    fig.update_layout(height=260, margin=dict(l=10, r=80, t=28, b=10))
    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                     style={"height": "260px"})


from .data import _epoch  # noqa: E402  (מיובא כאן כדי להימנע מייבוא מעגלי בזמן טעינה)


# הציור עצמו רץ בדפדפן — שולחים את המטען ל-assets/chart.js בלי סבב נוסף לשרת.
app.clientside_callback(
    "function(payload) { return window.dash_clientside.chart.render(payload); }",
    Output("chart-sink", "data"),
    Input("chart-store", "data"),
)

# טוגל טהור ב-JS: שדות "Custom Range" מוצגים רק כשזו הבחירה בפילטר התקופה -
# לא צריך סבב לשרת בשביל הצגה/הסתרה של שני שדות תאריך.
app.clientside_callback(
    """
    function(period) {
        return period === "custom" ? {display: "flex"} : {display: "none"};
    }
    """,
    Output("perf-custom-range", "style"),
    Input("perf-period-select", "value"),
)

# עדכון לייב: series.update() על הנר האחרון בלבד, לא setData() מלא — זול, ולא נוגע
# בזום/פאן של המשתמש (בניגוד ל-render למעלה, שמיועד לטעינה מלאה/החלפת סימבול).
app.clientside_callback(
    "function(payload) { return window.dash_clientside.chart.updateLive(payload); }",
    Output("live-sink", "data"),
    Input("live-store", "data"),
)

# מדרג-פירוט (LOD): assets/chart.js קורא ל-/api/lod ישירות עם fetch() בכל שינוי
# זום/גלילה (ראו requestLodWindow שם) ומיישם את התשובה בעצמו דרך
# window.dash_clientside.chart.applyLodWindow - אין כאן Store/callback של Dash
# בכלל, בכוונה: זה הנתיב הכי חם באפליקציה, וסבב-הלוך-חזור דרך Dash (Store +
# כפתור מוסתר + עוד Store) הוסיף שכבת serialization/dispatch מיותרת לגמרי.

# run-window-store מתמלא ב-JS מיד בלחיצה על "Add to chart", מהטווח הנראה כרגע
# בגרף (window.__currentVisibleRange, מתעדכן בכל שינוי זום/גלילה - ראו
# onVisibleTimeRangeChanged ב-chart.js). run_or_switch למעלה מגיב לשינוי ב-Store
# הזה, לא ישירות ל-n_clicks - כך run_or_switch יודע בדיוק על איזה טווח להריץ.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var vr = window.__currentVisibleRange || {};
        return {from: vr.from, to: vr.to, n: n_clicks};
    }
    """,
    Output("run-window-store", "data"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)

# משוב מיידי בלחיצה על "Add to chart": run_or_switch (למעלה) לוקח כ-2-3 שניות
# על כל ההיסטוריה, וה-dcc.Loading על key-stats/result-body לא נראה בכלל כל עוד
# לשונית "Pine Editor" עדיין פתוחה (tester-pane מוסתר עד ש-run_or_switch עצמו
# מחליף לשונית, בסוף הריצה). בלי המשוב המיידי הזה המשתמש לא רואה שום דבר במשך
# כל משך הריצה - בדיוק הבאג המקורי. עוברים ללשונית Tester ומשביתים את הכפתור
# מיד עם הלחיצה (JS טהור, בלי סבב לשרת) כדי שלא יהיה אפשר להתחיל ריצה שנייה
# במקביל לראשונה.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        return ["tester", true, "Running..."];
    }
    """,
    Output("panel-tabs", "value", allow_duplicate=True),
    Output("run-btn", "disabled"),
    Output("run-btn", "children"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)

# מאפסים את הכפתור ברגע ש-run_or_switch בצד השרת סיים (result-body תמיד מתעדכן
# ממנו, גם בסוף ריצה אמיתית וגם במעבר לשונית-תוצאות רגיל - איפוס במקרה השני
# תמים לגמרי, הכפתור כבר היה במצב רגיל).
app.clientside_callback(
    """
    function(_children) {
        return [false, "\\u25b6  Add to chart"];
    }
    """,
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("run-btn", "children", allow_duplicate=True),
    Input("result-body", "children"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
