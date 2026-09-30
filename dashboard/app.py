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
import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, callback, callback_context, dcc, html, no_update

# ייבוא מוחלט — מגיע ל-chart.py שבשורש הפרויקט (גרף הביצועים של Plotly),
# ולא ל-dashboard/chart.py שנטען כ-`.chart`.
from chart import _evenly_sampled_indices, build_performance_chart

from . import lab, live, panels, paper, results
from .chart import build_history_chunk_payload, build_live_bar_payload, build_payload
from .data import (
    from_display_epoch,
    to_display_naive,
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
    FIB_GOLDEN_CODE,
    RECOGNIZED_STRATEGY_INPUTS,
    StrategyCompileError,
    detect_builtin_strategy,
    parse_strategy_inputs,
    run_fast_sma_crossover_backtest,
    run_hammer_rsi_backtest,
    run_fib_golden_backtest,
    to_markers,
)

DEFAULT_SYMBOL = "MNQ"
DEFAULT_TF = "5m"

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
TRADES_PAGE_SIZE = results.TRADES_PAGE

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
    {"name": f"{eng['title']} (built-in)", "code": eng["code"], "settings": {}}
    for key, eng in lab.ENGINES.items() if not eng.get("default_marker")
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
        en, he = lab.param_en(name), lab.param_he(name)
        label = [html.Span(en, className="ctl-en"), html.Span((he or "") + ("" if item["recognized"] else " · לא פעיל"), className="ctl-he")]
        field_id = {"type": "strategy-input", "name": name}
        if type_name == "bool":
            control = dcc.Checklist(
                id=field_id, options=[{"label": "", "value": "on"}],
                value=["on"] if value else [], className="strategy-input-checkbox",
            )
        else:
            control = dcc.Input(
                id=field_id, type="number", value=value,
                step=1 if type_name == "int" else "any",
                className="num-input tiny", debounce=True,
            )
        fields.append(html.Div(className="settings-field param-field", title=name, children=[
            html.Div(label, className="ctl-label"), control,
        ]))
    return html.Div(className="settings-grid", children=fields)


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def _catalog_children():
    """כרטיס לכל אסטרטגיה מובנית: שם, הסבר, פרמטרים, תוצאה אחרונה (אם רצה
    בסשן הזה) וכפתורי טעינה/הרצה."""
    by_engine = _RESULTS.get("by_engine", {})
    cards = []
    for key, eng in lab.ENGINES.items():
        last = by_engine.get(key)
        if last:
            st = last["stats"]
            cls = "cat-result pos" if st["net"] > 0 else "cat-result neg"
            result = html.Div(className=cls, children=[
                html.Div(f"ריצה אחרונה · {last['symbol']} {last['timeframe']} · {last['at']}", className="cat-result-when"),
                html.Div(className="cat-stats", children=[
                    html.Div([html.Span(_fmt_money(st["net"]), className="v"), html.Span("רווח נקי", className="k")]),
                    html.Div([html.Span(str(st["trades"]), className="v"), html.Span("עסקאות", className="k")]),
                    html.Div([html.Span(f"{st['win_rate']:.0f}%", className="v"), html.Span("זכייה", className="k")]),
                    html.Div([html.Span(f"{st['pf']:.2f}", className="v"), html.Span("Profit Factor", className="k")]),
                    html.Div([html.Span(f"{st['max_dd_pct']:.1f}%", className="v"), html.Span("Drawdown", className="k")]),
                ]),
            ])
        else:
            result = html.Div("עדיין לא הורץ בסשן הזה", className="cat-result none")
        chips = [html.Span([html.Span(lab.param_en(n), className="chip-k"), html.Span(lab.fmt_value(d), className="chip-v")],
                          className="chip", title=f"{lab.param_he(n)} ({n})") for n, _p, _c, d in eng["params"]]
        cards.append(html.Div(className="cat-card", children=[
            html.Div(eng["title"], className="cat-title"),
            html.Div(eng["desc"], className="cat-desc"),
            html.Div(chips, className="cat-chips"),
            result,
            html.Div(className="cat-actions", children=[
                html.Button("טען לעורך", id={"type": "catalog-load", "key": key},
                            n_clicks=0, className="cat-btn"),
                html.Button("טען והרץ", id={"type": "catalog-run", "key": key},
                            n_clicks=0, className="cat-btn primary"),
            ]),
        ]))
    return [
        html.Div(className="pane-head", children=[
            html.Div("ספריית האסטרטגיות", className="pane-title"),
            html.Div("בחר אסטרטגיה. \"טען והרץ\" מריץ אותה מיד על הסימבול והטיים-פריים שנבחרו למעלה; "
                     "\"טען לעורך\" פותח אותה לעריכה.", className="pane-sub"),
        ]),
        html.Div(className="cat-grid", children=cards),
    ]


def _fld(label, control):
    return html.Div(className="settings-field", children=[html.Span(label, className="ctl-label"), control])


def _optimizer_children():
    return [
        html.Div(className="pane-head", children=[
            html.Div("אופטימיזציית פרמטרים", className="pane-title"),
            html.Div("בוחרים פרמטר אחד או שניים, טווח וצעד. הדירוג לפי 70% הראשונים של ההיסטוריה; "
                     "העמודות \"מחוץ למדגם\" מראות את אותם ערכים על 30% האחרונים, כדי לא להתאהב בתוצאה שלא מחזיקה.",
                     className="pane-sub"),
        ]),
        html.Div(id="opt-engine-label", className="opt-engine"),
        html.Div(className="opt-card", children=[
            html.Div(className="opt-row", children=[
                _fld("פרמטר 1", dcc.Dropdown(id="opt-p1", className="settings-dropdown opt-dd", clearable=False, searchable=False)),
                _fld("מ-", dcc.Input(id="opt-p1-min", type="number", step="any", className="num-input tiny")),
                _fld("עד", dcc.Input(id="opt-p1-max", type="number", step="any", className="num-input tiny")),
                _fld("צעד", dcc.Input(id="opt-p1-step", type="number", step="any", className="num-input tiny")),
            ]),
            html.Div(className="opt-row", children=[
                _fld("פרמטר 2 (אופציונלי)", dcc.Dropdown(id="opt-p2", className="settings-dropdown opt-dd", clearable=True,
                                                     searchable=False, placeholder="ללא")),
                _fld("מ-", dcc.Input(id="opt-p2-min", type="number", step="any", className="num-input tiny")),
                _fld("עד", dcc.Input(id="opt-p2-max", type="number", step="any", className="num-input tiny")),
                _fld("צעד", dcc.Input(id="opt-p2-step", type="number", step="any", className="num-input tiny")),
            ]),
            html.Div(className="opt-row opt-run-row", children=[
                html.Button("▶  הרץ אופטימיזציה", id="opt-run-btn", n_clicks=0, className="run-btn"),
                html.Span(id="opt-status", className="opt-status"),
            ]),
        ]),
        dcc.Loading(html.Div(id="opt-results", className="opt-results"), type="dot", color="#2962FF"),
        dcc.Store(id="opt-store"),
        html.Div(className="pane-head wf-head", children=[
            html.Div("Walk-forward: אופטימיזציה מתגלגלת", className="pane-title"),
            html.Div("ההיסטוריה נחתכת לחלונות רצופים. בכל חלון בוחרים את הפרמטרים הטובים ביותר על 70% הראשונים "
                     "ובודקים אותם על 30% שאחריהם. סך ה\"מחוץ למדגם\" של כל החלונות הוא הביצוע הכן ביותר שאפשר לקבל "
                     "לפרמטרים שנבחרו באופטימיזציה, ו\"יציבות\" אומרת אם אותם ערכים נבחרים שוב ושוב.",
                     className="pane-sub"),
        ]),
        html.Div(className="opt-card", children=[
            html.Div(className="opt-row", children=[
                _fld("חלונות", dcc.Input(id="wf-windows", type="number", value=5, min=2, max=12, step=1, className="num-input tiny")),
                _fld("דירוג לפי", dcc.Dropdown(id="wf-rank", className="settings-dropdown opt-dd", clearable=False, searchable=False,
                                              options=[{"label": "רווח נקי", "value": "net"}, {"label": "Profit Factor", "value": "pf"}], value="net")),
                html.Div(className="settings-field", children=[html.Span("\u00a0", className="ctl-label"),
                    html.Button("▶  הרץ Walk-forward", id="wf-run-btn", n_clicks=0, className="run-btn",
                                title="משתמש באותם פרמטרים וטווחים שהוגדרו למעלה")]),
            ]),
        ]),
        dcc.Loading(html.Div(id="wf-results", className="opt-results"), type="dot", color="#2962FF"),
    ]



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
            html.Div(className="brand", children=[html.Span("Strategy", className="brand-a"), html.Span("Lab", className="brand-b")]),
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
            html.Div(className="bottom-panel", children=[
            html.Div(className="lab-head", children=[
                dcc.Dropdown(id="strategy-select", className="settings-dropdown strategy-dd", clearable=False, searchable=False,
                             options=[{"label": eng["title"], "value": key} for key, eng in lab.ENGINES.items()],
                             value="sma_cross"),
                html.Span(id="lab-head-result", className="lab-head-result"),
            ]),
                html.Div(className="panel-bar", children=[
                    dcc.Tabs(id="panel-tabs", value="tester", className="panel-tabs", persistence=True,
                             parent_className="panel-tabs-parent", children=[
                                 dcc.Tab(label="בדיקה", value="tester",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="עמידות", value="optimize",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="נייר", value="paper",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="היסטוריה", value="history",
                                         className="ptab", selected_className="ptab-sel"),
                                 dcc.Tab(label="עוד", value="more",
                                         className="ptab ptab-more", selected_className="ptab-sel"),
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
                        # סרגל עבודה (שלב 2): פרמטרים + טווח + הרצה ליד התוצאות, בלי לעבור לשוניות.
                        html.Div(className="tester-bar", children=[
                            html.Div(className="tester-bar-row", children=[
                                html.Div(className="wb-head", children=[
                                    html.Div(id="wb-title", className="wb-title"),
                                    html.Div(id="run-label", className="run-label"),
                                ]),
                                html.Div(className="wb-actions", children=[
                                    html.Span("טווח הבדיקה", className="ctl-label"),
                                    dcc.Dropdown(id="bt-range", className="settings-dropdown bt-range-dd", persistence=True,
                                                 options=[{"label": l, "value": k} for k, l in lab.RANGE_OPTIONS],
                                                 value="all", clearable=False, searchable=False),
                                    html.Button("▶  הרץ בדיקה", id="tester-run-btn", n_clicks=0, className="run-btn",
                                                title="הרצת האסטרטגיה שבעורך עם הפרמטרים שלמטה"),
                                    html.Label(className="auto-run", title="מריץ מחדש בכל שינוי פרמטר (לא על נרות דקה)", children=[
                                        dcc.Checklist(id="auto-run", options=[{"label": "", "value": "on"}], value=[], persistence=True,
                                                      className="strategy-input-checkbox"),
                                        html.Span("הרצה אוטומטית"),
                                    ]),
                                ]),
                            ]),
                            html.Div(id="strategy-inputs-container",
                                     children=_strategy_inputs_section(lab.ENGINES["sma_cross"]["code"])),
                        ]),
                        # dcc.Loading עוטף את שני ה-Output-ים של run_or_switch (למטה) -
                        # Dash מזהה לבד קריאת callback פעילה שמשפיעה על ילד עטוף ומציג
                        # ספינר אוטומטית. בלי זה, לחיצה על "Add to chart" לא נתנה שום
                        # משוב חזותי במשך כל משך הריצה (בקטסט על 7 שנות MNQ/1m יכול
                        # לקחת כ-3 דקות, backtesting.py מריץ בר-אחר-בר בפייתון, לא
                        # מוקטר) - בדיוק מה שנראה כאילו "הכפתור לא עושה כלום".
                        html.Div(id="tester-error", className="tester-error", hidden=True),
                        dcc.Loading(html.Div(id="key-stats"), type="dot", color="#2962FF"),
                        dcc.Tabs(id="result-tabs", value="overview", className="result-tabs",
                                 children=[
                                     dcc.Tab(label="סקירה", value="overview",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="ביצועים", value="summary",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="עסקאות", value="trades",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="חודשי", value="monthly",
                                             className="rtab", selected_className="rtab-sel"),
                                     dcc.Tab(label="מונטה קרלו", value="montecarlo",
                                             className="rtab", selected_className="rtab-sel"),
                                 ]),
                        # פילטר תקופה מעל גרף ה-Performance - מוצג רק בלשונית overview
                        # (ראו run_or_switch: Output("perf-period-bar","style")). Custom Range
                        # פותח שני שדות תאריך (perf-custom-range) - טוגל טהור ב-JS, בלי
                        # סבב לשרת (ראו clientside_callback למטה).
                        dcc.Store(id="perf-period", data="range"),
                        html.Div(id="perf-period-bar", className="perf-period-bar", style={"display": "none"}, children=[
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
                                     "בחר אסטרטגיה בבורר שלמעלה, כוונן פרמטרים ולחץ Enter או \"הרץ בדיקה\"."))),
                                    type="dot", color="#2962FF"),
                        # ניווט עמודים ל"List of Trades" - מוצג רק בלשונית trades וכשיש
                        # יותר מ-TRADES_PAGE_SIZE עסקאות (ראו run_or_switch/_trades_pagination_state).
                        dcc.Store(id="trades-page", data=0),
                        html.Div(id="trades-pagination", className="trades-pagination",
                                 style={"display": "none"}, children=[
                                     html.Button("‹ הקודם", id="trades-prev-btn", n_clicks=0,
                                                 className="trades-page-btn"),
                                     html.Span(id="trades-page-info", className="trades-page-info"),
                                     html.Button("הבא ›", id="trades-next-btn", n_clicks=0,
                                                 className="trades-page-btn"),
                                 ]),
                    ]),
                    html.Div(id="editor-pane", style={"display": "none"}, className="rtl-pane", children=[
                        html.Div(className="editor-row", children=[
                            html.Div("הקוד מגדיר את האסטרטגיה והפרמטרים שלה (שורות עם # input הופכות לשדות בבודק). "
                                     "שורת # strategy: בראש הקוד קובעת איזה מנוע מובנה רץ.", className="editor-hint"),
                            html.Div(className="editor-controls", children=[
                                html.Button("▶  הרץ", id="run-btn", n_clicks=0,
                                            className="run-btn",
                                            title="בקטסט על כל ההיסטוריה"),
                            ]),
                        ]),
                        html.Div("שים לב: בגרסה הנוכחית רצים רק המנועים המובנים (לפי שורת # strategy). "
                                 "קוד חופשי בפונקציות init/next עדיין אינו מורץ; שינוי הפרמטרים כן משפיע.",
                                 className="editor-note"),
                        dcc.Textarea(id="code-input", value=lab.ENGINES["sma_cross"]["code"], className="code-area", persistence=True, persistence_type="local",
                                     spellCheck=False),
                        html.Div(id="run-error", className="run-error"),
                    ]),
                    html.Div(id="settings-pane", style={"display": "none"}, className="rtl-pane", children=[
                        html.Div(className="settings-hint",
                                 children="הגדרות חשבון וביצוע. משפיעות על חישוב הרווח, הסליפג' והעמלות בריצה הבאה."),
                        html.Div(className="settings-grid", children=[
                            html.Div(className="settings-field", children=[
                                html.Span("הון התחלתי ($)", className="ctl-label"),
                                dcc.Input(id="account-value-input", type="number", value=100000,
                                          min=1000, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("חוזים לעסקה", className="ctl-label"),
                                dcc.Input(id="contracts-input", type="number", value=1, min=1,
                                          step=1, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("מכפיל חוזה", className="ctl-label"),
                                dcc.Input(id="multiplier-input", type="number", value=2, min=0,
                                          step=0.01, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("סליפג' (טיקים)", className="ctl-label"),
                                dcc.Input(id="slippage-input", type="number", value=2, min=0,
                                          step=1, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("עמלה לחוזה ($)", className="ctl-label"),
                                dcc.Input(id="settings-commission-input", type="number", value=0,
                                          min=0, step=0.01, className="num-input tiny", debounce=True),
                            ]),
                            html.Div(className="settings-field", children=[
                                html.Span("סוג פקודה", className="ctl-label"),
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
                        html.Div(className="settings-hint",
                                 children="פרמטרי האסטרטגיה עצמה נמצאים בראש לשונית \"בודק אסטרטגיות\"."),
                    ]),
                    html.Div(id="more-pane", style={"display": "none"}, className="rtl-pane", children=[
                        dcc.Tabs(id="more-tabs", value="catalog", className="result-tabs", children=[
                            dcc.Tab(label="ספריית אסטרטגיות", value="catalog", className="rtab", selected_className="rtab-sel"),
                            dcc.Tab(label="קוד", value="editor", className="rtab", selected_className="rtab-sel"),
                            dcc.Tab(label="הגדרות חשבון", value="settings", className="rtab", selected_className="rtab-sel"),
                        ]),
                    ]),
                    html.Div(id="catalog-pane", style={"display": "none"}, className="rtl-pane",
                             children=_catalog_children()),
                    html.Div(id="optimize-pane", style={"display": "none"}, className="rtl-pane",
                             children=_optimizer_children()),
                    html.Div(id="history-pane", style={"display": "none"}, className="rtl-pane",
                             children=[html.Div(id="history-list"), html.Div(id="history-compare")]),
                    html.Div(id="paper-pane", style={"display": "none"}, className="rtl-pane", children=[
                        html.Div(className="pane-head", children=[
                            html.Div("מסחר בנייר", className="pane-title"),
                            html.Div("האסטרטגיה שבעורך רצה על נרות חיים מ-IB ורושמת כניסות ויציאות, בלי לשלוח פקודות. "
                                     "הלוגיקה זהה לבקטסט: מילוי בפתיחת הנר הבא, אותם סטופ ויעד. עסקאות נספרות מהנר הסגור הראשון אחרי הפתיחה.",
                                     className="pane-sub"),
                        ]),
                        html.Div(className="opt-card", children=[
                            html.Div(className="opt-row", children=[
                                html.Div(id="paper-strategy-label", className="opt-engine"),
                                html.Button("▶  התחל סשן בנייר", id="paper-start-btn", n_clicks=0, className="run-btn",
                                            title="משתמש בסימבול ובטיים-פריים שנבחרו למעלה ובפרמטרים שבבודק"),
                                html.Span(id="paper-msg", className="opt-status"),
                            ]),
                        ]),
                        dcc.Interval(id="paper-interval", interval=5000, n_intervals=0),
                        html.Div(id="paper-body"),
                    ]),
                ]),
            ]),
            html.Div(id="col-resizer", className="col-resizer", title="גרור לשינוי רוחב · לחיצה כפולה לכיווץ הפאנל"),
            html.Div(className="chart-pane", children=[
                html.Div(id="chart-container"),
                html.Div(id="chart-toolbar", className="chart-toolbar", children=[
                    html.Button("‹", id="ct-prev", className="ct-btn", title="עסקה קודמת (←)"),
                    html.Span("—", id="ct-counter", className="ct-counter", title="עסקה נוכחית / סה\"כ"),
                    html.Button("›", id="ct-next", className="ct-btn", title="עסקה הבאה (→)"),
                    html.Span(className="ct-sep"),
                    html.Button("⟲", id="ct-fit", className="ct-btn", title="איפוס זום (R)"),
                    html.Button("⏭", id="ct-now", className="ct-btn", title="קפיצה לעכשיו (N)"),
                    html.Button("⛶", id="ct-full", className="ct-btn", title="גרף במסך מלא (G)"),
                    html.Span(className="ct-sep"),
                    html.Button("▤", id="ct-eq", className="ct-btn on", title="רצועת עקומת הון (E)"),
                    html.Button("◐", id="ct-sess", className="ct-btn on", title="הצללת סשן רגיל של ארה\"ב (S)"),
                    html.Button("▦", id="ct-ctx", className="ct-btn on", title="טווח הבדיקה וחלונות Walk-forward"),
                    html.Span(className="ct-sep"),
                    html.Button("─", id="ct-hline", className="ct-btn", title="קו אופקי (H): לחיצה על הגרף"),
                    html.Button("╱", id="ct-tline", className="ct-btn", title="קו מגמה (T): שתי לחיצות על הגרף"),
                    html.Button("✕", id="ct-clear", className="ct-btn", title="מחיקת כל הקווים שציירת"),
                ]),
                html.Div(id="chart-datawin", className="chart-datawin"),
                html.Div(id="chart-tooltip", className="chart-tooltip", style={"display": "none"}),
                html.Div(id="equity-strip", className="equity-strip"),
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
    title="Strategy Lab",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap",
    ],
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
    from_cache_note = ""
    try:
        df = load_ohlcv_preferring_1m_cache(symbol, timeframe, days=INITIAL_DAYS[timeframe], extended_hours=True)
    except DataError as exc:
        # IB לא זמין (Gateway סגור / API כבוי): מציגים את הקאש המקומי במקום גרף שחור.
        cached = load_full_history(symbol, timeframe, extended_hours=True)
        if cached is None or cached.empty:
            return no_update, str(exc), "status error"
        cutoff = cached.index[-1] - pd.Timedelta(days=INITIAL_DAYS[timeframe])
        df = cached[cached.index >= cutoff]
        from_cache_note = " · מהקאש (IB מנותק)"

    _FRAMES[(symbol, timeframe)] = df
    _RESULTS.pop("last", None)
    live.subscribe(symbol, timeframe)  # מעביר את מנוי ה-streaming לסימבול/טיים-פריים החדשים
    years = (df.index.max() - df.index.min()).days / 365.25
    status = (f"{len(df):,} bars · {df.index.min():%b %d, %Y} → {df.index.max():%b %d, %Y} "
              f"({years:.1f}y){from_cache_note}")
    return build_payload(df, symbol, timeframe), status, ("status warn" if from_cache_note else "status")


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
    Output("catalog-pane", "style"),
    Output("optimize-pane", "style"),
    Output("history-pane", "style"),
    Output("paper-pane", "style"),
    Output("more-pane", "style"),
    Input("panel-tabs", "value"), Input("more-tabs", "value"),
)
def switch_panel_tab(tab, more):
    show, hide = {"display": "block"}, {"display": "none"}
    in_more = tab == "more"
    return (
        show if tab == "tester" else hide,
        show if (in_more and more == "editor") else hide,
        show if (in_more and more == "settings") else hide,
        show if (in_more and more == "catalog") else hide,
        show if tab == "optimize" else hide,
        show if tab == "history" else hide,
        show if tab == "paper" else hide,
        show if in_more else hide,
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
            f"עמוד {page + 1} מתוך {n_pages} · {trade_count:,} עסקאות")


def _perf_period_bar_style(tab: str):
    """פילטר התקופה רלוונטי רק לגרף ה-Overview - מוסתר בשאר הלשוניות."""
    return {"display": "none"}


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
    """כרטיסי ה-KPI בראש התוצאות (results.kpi_tiles). פילטר התקופה הישן בוטל -
    הטווח נבחר לפני הריצה בסרגל העבודה."""
    return results.kpi_tiles(stats, cash)


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
    State("bt-range", "value"),
    prevent_initial_call=True,
)
def run_or_switch(_window_data, result_tab, _prev_n, _next_n, period, custom_from, custom_to,
                   symbol, timeframe, _code,
                   account_value, contracts, multiplier, slippage_ticks,
                   commission_per_contract, order_type,
                   strategy_input_values, strategy_input_ids, current_page, bt_range="all"):
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

    # בוחרים איזה מנוע וקטורי להריץ לפי שורת-הסימון הראשונה בקוד (# strategy:
    # <key>) - לא מריצים את הקוד עצמו. בלי סימון (DEFAULT_CODE, או אסטרטגיה
    # ישנה שנשמרה לפני התכונה הזו) - ברירת המחדל היא חציית-SMA, כמו קודם.
    builtin_key = lab.engine_key_for_code(_code)

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
        df_backtest, range_label = lab.slice_range(df_backtest, bt_range, _window_data, from_display_epoch)
        if df_backtest is None or len(df_backtest) < 50:
            return (no_update, no_update, no_update, "אין מספיק נרות בטווח שנבחר", no_update,
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
            stats = lab.run_engine(builtin_key, df_backtest, common_kwargs, strategy_overrides)
        except StrategyCompileError as exc:
            return (no_update, no_update, no_update, str(exc), no_update,
                    no_update, no_update, no_update, no_update)
        _RESULTS["last"] = (stats, cash, symbol, timeframe)
        _RESULTS["last_range"] = (df_backtest.index[0], df_backtest.index[-1])
        _RESULTS.pop("wf", None)
        rec = lab.record_run(key=builtin_key, symbol=symbol, timeframe=timeframe, range_label=range_label,
                             params=strategy_overrides, stats=stats, cash=cash)
        _RESULTS["last_label"] = (f"ריצה #{rec['id']} · {rec['title']} · {symbol} {timeframe} · "
                                  f"{range_label} · {len(df_backtest):,} נרות · {rec['at']}")
        _RESULTS.setdefault("by_engine", {})[builtin_key] = {
            "symbol": symbol, "timeframe": timeframe, "at": pd.Timestamp.now().strftime("%d/%m %H:%M"),
            "stats": lab.quick_stats(stats, cash), "params": dict(strategy_overrides),
        }
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
        return results.performance_table(stats, cash)
    if tab == "trades":
        return results.trades_table(stats, page=page, page_size=TRADES_PAGE_SIZE)
    if tab == "monthly":
        return results.monthly(stats, cash)
    if tab == "montecarlo":
        return results.monte_carlo_view(stats, cash)
    return results.overview(stats, cash)



# ================================================================ מעבדת אסטרטגיות (lab.py)
def _opt_param_options(code: str):
    key = lab.engine_key_for_code(code)
    present = {it["name"] for it in parse_strategy_inputs(code or "")}
    opts = [{"label": lab.param_label(name), "value": name}
            for name, _p, caster, _d in lab.engine_params(key)
            if caster is not None and name in present]
    return key, opts


@callback(
    Output("opt-p1", "options"), Output("opt-p2", "options"),
    Output("opt-p1", "value"), Output("opt-engine-label", "children"),
    Output("wb-title", "children"), Output("paper-strategy-label", "children"),
    Output("strategy-select", "value"),
    Input("code-input", "value"),
)
def refresh_opt_params(code):
    key, opts = _opt_param_options(code)
    first = opts[0]["value"] if opts else None
    title = lab.ENGINES[key]["title"]
    return opts, opts, first, f"אסטרטגיה בעורך: {title}", title, f"אסטרטגיה: {title}", key


def _suggest_range(name, code):
    """טווח התחלתי סביר סביב הערך שבקוד: חצי עד כפול, בכ-8 צעדים."""
    if not name:
        return None, None, None
    cur = lab.code_overrides(code).get(name)
    if not isinstance(cur, (int, float)) or isinstance(cur, bool):
        return None, None, None
    cur = float(cur)
    if cur == 0:
        return 0, 10, 1
    lo, hi = cur * 0.5, cur * 2.0
    step = (hi - lo) / 8
    if float(cur).is_integer() and cur >= 4:
        lo, hi, step = int(round(lo)), int(round(hi)), max(1, int(round(step)))
    else:
        step = round(step, 3)
    return lo, hi, step


@callback(
    Output("opt-p1-min", "value"), Output("opt-p1-max", "value"), Output("opt-p1-step", "value"),
    Input("opt-p1", "value"), State("code-input", "value"),
)
def suggest_p1_range(name, code):
    return _suggest_range(name, code)


@callback(
    Output("opt-p2-min", "value"), Output("opt-p2-max", "value"), Output("opt-p2-step", "value"),
    Input("opt-p2", "value"), State("code-input", "value"),
)
def suggest_p2_range(name, code):
    return _suggest_range(name, code)


@callback(
    Output("opt-results", "children"), Output("opt-store", "data"), Output("opt-status", "children"),
    Input("opt-run-btn", "n_clicks"),
    State("code-input", "value"), State("symbol-input", "value"), State("tf-store", "data"),
    State("account-value-input", "value"), State("contracts-input", "value"),
    State("multiplier-input", "value"), State("slippage-input", "value"),
    State("settings-commission-input", "value"), State("order-type-input", "value"),
    State({"type": "strategy-input", "name": ALL}, "value"),
    State({"type": "strategy-input", "name": ALL}, "id"),
    State("opt-p1", "value"), State("opt-p1-min", "value"), State("opt-p1-max", "value"), State("opt-p1-step", "value"),
    State("opt-p2", "value"), State("opt-p2-min", "value"), State("opt-p2-max", "value"), State("opt-p2-step", "value"),
    State("bt-range", "value"),
    prevent_initial_call=True,
)
def run_optimizer(n_clicks, code, symbol, timeframe, account_value, contracts, multiplier, slippage_ticks,
                  commission, order_type, in_vals, in_ids, p1, p1_lo, p1_hi, p1_st, p2, p2_lo, p2_hi, p2_st,
                  bt_range="all"):
    if not n_clicks:
        return no_update, no_update, no_update
    symbol = (symbol or "").strip().upper()
    key = lab.engine_key_for_code(code)
    grid = {}
    if p1:
        vals = lab.value_grid(p1_lo, p1_hi, p1_st)
        if not vals:
            return html.Div("טווח לא תקין לפרמטר 1 (מ- / עד / צעד).", className="run-error"), no_update, ""
        grid[p1] = vals
    if p2 and p2 != p1:
        vals = lab.value_grid(p2_lo, p2_hi, p2_st)
        if not vals:
            return html.Div("טווח לא תקין לפרמטר 2.", className="run-error"), no_update, ""
        grid[p2] = vals
    if not grid:
        return html.Div("בחר לפחות פרמטר אחד.", className="run-error"), no_update, ""
    df = load_full_history(symbol, timeframe, extended_hours=True)
    if df is None:
        df = _FRAMES.get((symbol, timeframe))
    if df is None:
        return html.Div("טען נתונים קודם (בחר סימבול וטיים-פריים למעלה).", className="run-error"), no_update, ""
    df, range_label = lab.slice_range(df, "all" if bt_range == "chart" else bt_range, None)
    overrides = lab.code_overrides(code)
    overrides.update({
        id_dict.get("name"): (bool(val) if isinstance(val, list) else val)
        for val, id_dict in zip(in_vals, in_ids) if isinstance(id_dict, dict) and id_dict.get("name")
    })
    common = dict(cash=float(account_value or 100_000), size=float(contracts or 1),
                  multiplier=float(multiplier or 1), slippage_ticks=float(slippage_ticks or 0),
                  commission_per_contract=float(commission or 0), market_order=(order_type != "Limit"))
    res = lab.optimize(key, df, common, overrides, grid)
    rows = res["rows"]
    if not rows:
        return html.Div("לא התקבלו תוצאות.", className="run-error"), no_update, ""
    header = [html.Th(lab.param_en(n), title=f"{lab.param_he(n)} ({n})") for n in res["names"]] + [
        html.Th("עסקאות"), html.Th("זכייה %"), html.Th("PF"), html.Th("רווח נקי"), html.Th("DD %"),
        html.Th("מחוץ למדגם: רווח"), html.Th("מחוץ למדגם: PF"), html.Th("")]
    body = []
    for i, r in enumerate(rows[:60]):
        a, b = r["in"], r["out"]
        body.append(html.Tr(className=("opt-best" if i == 0 else ""), children=(
            [html.Td(f"{v:g}") for v in r["params"].values()] + [
                html.Td(a["trades"]), html.Td(f"{a['win_rate']:.0f}"), html.Td(f"{a['pf']:.2f}"),
                html.Td(_fmt_money(a["net"]), className=("pos" if a["net"] > 0 else "neg")),
                html.Td(f"{a['max_dd_pct']:.1f}"),
                html.Td(_fmt_money(b["net"]), className=("pos" if b["net"] > 0 else "neg")),
                html.Td(f"{b['pf']:.2f}"),
                html.Td(html.Button("החל", id={"type": "opt-apply", "index": i}, n_clicks=0, className="cat-btn")),
            ])))
    note = (f"{res['n_combos']} שילובים ב-{res['seconds']:.1f} שניות · {symbol} {timeframe} · {range_label} · "
            f"במדגם {res['in_from']} עד {res['in_to']} · מחוץ למדגם {res['out_from']} עד {res['out_to']}"
            + (" · נחתך ל-400 שילובים, צמצם את הטווח" if res["truncated"] else ""))
    table = html.Table(className="opt-table", children=[html.Thead(html.Tr(header)), html.Tbody(body)])
    store = [r["params"] for r in rows[:60]]
    return [html.Div(note, className="opt-note"), html.Div(table, className="opt-table-wrap")], store, ""


@callback(
    Output("code-input", "value", allow_duplicate=True),
    Output("panel-tabs", "value", allow_duplicate=True),
    Output("opt-status", "children", allow_duplicate=True),
    Input({"type": "opt-apply", "index": ALL}, "n_clicks"),
    State("opt-store", "data"), State("code-input", "value"),
    prevent_initial_call=True,
)
def apply_optimized(n_clicks_list, store, code):
    trig = callback_context.triggered_id
    if not isinstance(trig, dict) or not any(n for n in n_clicks_list if n):
        return no_update, no_update, no_update
    idx = trig.get("index")
    if store is None or idx is None or idx >= len(store):
        return no_update, no_update, no_update
    return lab.apply_inputs_to_code(code, store[idx]), "editor", "הערכים הוחלו בעורך וב-Settings. לחץ Add to chart להרצה מלאה."


@callback(
    Output("code-input", "value", allow_duplicate=True),
    Output("panel-tabs", "value", allow_duplicate=True),
    Output("run-btn", "n_clicks", allow_duplicate=True),
    Input({"type": "catalog-load", "key": ALL}, "n_clicks"),
    Input({"type": "catalog-run", "key": ALL}, "n_clicks"),
    State("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def catalog_action(load_clicks, run_clicks, run_n):
    trig = callback_context.triggered_id
    if not isinstance(trig, dict) or not (any(n for n in load_clicks if n) or any(n for n in run_clicks if n)):
        return no_update, no_update, no_update
    key = trig.get("key")
    if key not in lab.ENGINES:
        return no_update, no_update, no_update
    code = lab.ENGINES[key]["code"]
    if trig.get("type") == "catalog-run":
        # העלאת n_clicks של "Add to chart" מפעילה את אותה שרשרת בדיוק כמו לחיצה ידנית
        return code, "tester", (run_n or 0) + 1
    return code, "editor", no_update


@callback(
    Output("catalog-pane", "children"),
    Input("panel-tabs", "value"),
    prevent_initial_call=True,
)
def refresh_catalog(tab):
    if tab != "more":
        return no_update
    return _catalog_children()



# ---------------------------------------------------------------- שלב 2: סרגל עבודה + היסטוריה
@callback(
    Output("run-btn", "n_clicks", allow_duplicate=True),
    Input("tester-run-btn", "n_clicks"), State("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def tester_run(n, run_n):
    if not n:
        return no_update
    return (run_n or 0) + 1


@callback(Output("run-label", "children"), Input("key-stats", "children"), prevent_initial_call=True)
def show_run_label(_):
    return _RESULTS.get("last_label", "")


def _history_list_children():
    if not lab.RUN_HISTORY:
        return html.Div(className="empty-state", children=[
            html.Div("עדיין אין ריצות", className="empty-title"),
            html.Div("כל לחיצה על \"הרץ בדיקה\" או \"טען והרץ\" נרשמת כאן, ואפשר להשוות בין ריצות.", className="empty-sub")])
    header = html.Tr([html.Th("השווה"), html.Th("#"), html.Th("שעה"), html.Th("אסטרטגיה"), html.Th("סימבול"),
                      html.Th("טווח"), html.Th("עסקאות"), html.Th("זכייה %"), html.Th("PF"), html.Th("רווח נקי"),
                      html.Th("DD %"), html.Th("פרמטרים"), html.Th("")])
    rows = []
    for r in lab.RUN_HISTORY:
        q = r["q"]
        params = " · ".join(f"{lab.param_en(k)}: {lab.fmt_value(v)}" for k, v in r["params"].items())
        rows.append(html.Tr([
            html.Td(dcc.Checklist(id={"type": "hist-sel", "index": r["id"]}, options=[{"label": "", "value": "on"}],
                                  value=[], className="hist-sel")),
            html.Td(f"#{r['id']}"), html.Td(r["at"]), html.Td(r["title"]),
            html.Td(f"{r['symbol']} {r['timeframe']}"), html.Td(r["range"]),
            html.Td(q["trades"]), html.Td(f"{q['win_rate']:.0f}"), html.Td(f"{q['pf']:.2f}"),
            html.Td(_fmt_money(q["net"]), className=("pos" if q["net"] > 0 else "neg")),
            html.Td(f"{q['max_dd_pct']:.1f}"),
            html.Td(params, className="params", title=params),
            html.Td([html.Button("טען", id={"type": "hist-load", "index": r["id"]}, n_clicks=0, className="cat-btn"),
                     " ",
                     html.Button("✕", id={"type": "hist-del", "index": r["id"]}, n_clicks=0, className="cat-btn",
                                 title="מחיקה מההיסטוריה")]),
        ]))
    return [
        html.Div(className="pane-head", children=[
            html.Div("היסטוריית ריצות", className="pane-title"),
            html.Div("סמן עד 3 ריצות בעמודה \"השווה\" כדי לראות אותן זו לצד זו. \"טען\" מחזיר את הקוד והפרמטרים לעורך.",
                     className="pane-sub"),
        ]),
        html.Div(html.Table(className="hist-table", children=[html.Thead(header), html.Tbody(rows)]),
                 className="opt-table-wrap"),
    ]


@callback(
    Output("history-list", "children"),
    Input("panel-tabs", "value"), Input("key-stats", "children"),
    Input({"type": "hist-del", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def refresh_history(tab, _stats, del_clicks):
    trig = callback_context.triggered_id
    if isinstance(trig, dict) and trig.get("type") == "hist-del":
        if any(n for n in del_clicks if n):
            lab.delete_run(trig.get("index"))
        return _history_list_children()
    if tab != "history" and trig == "panel-tabs":
        return no_update
    return _history_list_children()


@callback(
    Output("history-compare", "children"),
    Input({"type": "hist-sel", "index": ALL}, "value"),
    State({"type": "hist-sel", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def compare_runs(values, ids):
    chosen = [i["index"] for v, i in zip(values, ids) if v]
    recs = [lab.get_run(rid) for rid in chosen[:3]]
    recs = [r for r in recs if r]
    if not recs:
        return ""
    metrics = [("רווח נקי", lambda r: _fmt_money(r["q"]["net"]), lambda r: r["q"]["net"]),
               ("תשואה %", lambda r: f"{r['q']['ret_pct']:.1f}%", lambda r: r["q"]["ret_pct"]),
               ("קנה והחזק %", lambda r: f"{r['bh']:.1f}%", None),
               ("עסקאות", lambda r: str(r["q"]["trades"]), None),
               ("זכייה %", lambda r: f"{r['q']['win_rate']:.0f}%", None),
               ("Profit Factor", lambda r: f"{r['q']['pf']:.2f}", None),
               ("Drawdown מקסימלי", lambda r: f"{r['q']['max_dd_pct']:.1f}%", None),
               ("רווח ממוצע לעסקה", lambda r: _fmt_money(r["q"]["avg_trade"]), lambda r: r["q"]["avg_trade"])]
    cols = []
    for r in recs:
        rows = [html.Div(className="cmp-row", children=[
            html.Span(label), html.Span(fmt(r), className=("" if sign is None else ("pos" if sign(r) > 0 else "neg")))])
            for label, fmt, sign in metrics]
        cols.append(html.Div(className="cmp-col", children=[
            html.Div(f"#{r['id']} {r['title']}", className="cmp-title"),
            html.Div(f"{r['symbol']} {r['timeframe']} · {r['range']} · {r['at']}", className="cat-result-when"),
        ] + rows))
    fig = lab.compare_figure(recs)
    has_curves = any(r.get("trades") is not None for r in recs)
    return [html.Div(className="cmp-grid", children=cols),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "280px"}) if has_curves
            else html.Div("עקומת ההון נשמרת רק ל-12 הריצות האחרונות.", className="settings-hint")]


@callback(
    Output("code-input", "value", allow_duplicate=True),
    Output("panel-tabs", "value", allow_duplicate=True),
    Input({"type": "hist-load", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_from_history(n_clicks_list):
    trig = callback_context.triggered_id
    if not isinstance(trig, dict) or not any(n for n in n_clicks_list if n):
        return no_update, no_update
    rec = lab.get_run(trig.get("index"))
    if not rec:
        return no_update, no_update
    return lab.run_code(rec), "tester"


@callback(Output("tester-error", "children"), Output("tester-error", "hidden"),
          Input("run-error", "children"), prevent_initial_call=True)
def mirror_run_error(text):
    text = (text or "").strip()
    return (f"⚠ {text}", False) if text else ("", True)


@callback(
    Output("wf-results", "children"),
    Input("wf-run-btn", "n_clicks"),
    State("code-input", "value"), State("symbol-input", "value"), State("tf-store", "data"),
    State("account-value-input", "value"), State("contracts-input", "value"),
    State("multiplier-input", "value"), State("slippage-input", "value"),
    State("settings-commission-input", "value"), State("order-type-input", "value"),
    State({"type": "strategy-input", "name": ALL}, "value"),
    State({"type": "strategy-input", "name": ALL}, "id"),
    State("opt-p1", "value"), State("opt-p1-min", "value"), State("opt-p1-max", "value"), State("opt-p1-step", "value"),
    State("opt-p2", "value"), State("opt-p2-min", "value"), State("opt-p2-max", "value"), State("opt-p2-step", "value"),
    State("bt-range", "value"), State("wf-windows", "value"), State("wf-rank", "value"),
    prevent_initial_call=True,
)
def run_walk_forward(n_clicks, code, symbol, timeframe, account_value, contracts, multiplier, slippage_ticks,
                     commission, order_type, in_vals, in_ids, p1, p1_lo, p1_hi, p1_st, p2, p2_lo, p2_hi, p2_st,
                     bt_range, n_windows, rank_by):
    if not n_clicks:
        return no_update
    symbol = (symbol or "").strip().upper()
    key = lab.engine_key_for_code(code)
    grid = {}
    if p1:
        vals = lab.value_grid(p1_lo, p1_hi, p1_st)
        if vals:
            grid[p1] = vals
    if p2 and p2 != p1:
        vals = lab.value_grid(p2_lo, p2_hi, p2_st)
        if vals:
            grid[p2] = vals
    if not grid:
        return html.Div("הגדר לפחות פרמטר אחד עם טווח וצעד תקינים (למעלה, באופטימיזציה).", className="run-error")
    df = load_full_history(symbol, timeframe, extended_hours=True)
    if df is None:
        df = _FRAMES.get((symbol, timeframe))
    if df is None:
        return html.Div("טען נתונים קודם.", className="run-error")
    df, range_label = lab.slice_range(df, "all" if bt_range == "chart" else bt_range, None)
    overrides = lab.code_overrides(code)
    overrides.update({id_dict.get("name"): (bool(val) if isinstance(val, list) else val)
                      for val, id_dict in zip(in_vals, in_ids) if isinstance(id_dict, dict) and id_dict.get("name")})
    common = dict(cash=float(account_value or 100_000), size=float(contracts or 1),
                  multiplier=float(multiplier or 1), slippage_ticks=float(slippage_ticks or 0),
                  commission_per_contract=float(commission or 0), market_order=(order_type != "Limit"))
    wf = lab.walk_forward(key, df, common, overrides, grid, n_windows=int(n_windows or 5), rank_by=rank_by or "net")
    _RESULTS["wf"] = [{"train_from": w["train_from"], "train_to": w["train_to"], "test_from": w["test_from"], "test_to": w["test_to"]}
                      for w in wf["windows"]]
    return results.walk_forward_view(wf, symbol, timeframe, range_label)


# ---------------------------------------------------------------- מסחר בנייר
@callback(
    Output("paper-msg", "children"),
    Input("paper-start-btn", "n_clicks"),
    State("code-input", "value"), State("symbol-input", "value"), State("tf-store", "data"),
    State("account-value-input", "value"), State("contracts-input", "value"),
    State("multiplier-input", "value"), State("slippage-input", "value"),
    State("settings-commission-input", "value"), State("order-type-input", "value"),
    State({"type": "strategy-input", "name": ALL}, "value"),
    State({"type": "strategy-input", "name": ALL}, "id"),
    prevent_initial_call=True,
)
def paper_start(n, code, symbol, timeframe, account_value, contracts, multiplier, slippage_ticks, commission,
                order_type, in_vals, in_ids):
    if not n:
        return no_update
    key = lab.engine_key_for_code(code)
    params = lab.code_overrides(code)
    params.update({id_dict.get("name"): (bool(val) if isinstance(val, list) else val)
                   for val, id_dict in zip(in_vals, in_ids) if isinstance(id_dict, dict) and id_dict.get("name")})
    common = dict(cash=float(account_value or 100_000), size=float(contracts or 1),
                  multiplier=float(multiplier or 1), slippage_ticks=float(slippage_ticks or 0),
                  commission_per_contract=float(commission or 0), market_order=(order_type != "Limit"))
    s = paper.start_session(key=key, symbol=symbol, timeframe=timeframe, params=params, common=common)
    return f"סשן #{s['id']} נפתח ({s['title']} · {s['symbol']} {s['timeframe']}). הבדיקה הראשונה תוך כמה שניות."


@callback(
    Output("paper-body", "children"),
    Input("paper-interval", "n_intervals"), Input("panel-tabs", "value"), Input("paper-msg", "children"),
    Input({"type": "paper-stop", "index": ALL}, "n_clicks"), Input({"type": "paper-del", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def paper_refresh(_n, tab, _msg, stop_clicks, del_clicks):
    trig = callback_context.triggered_id
    if isinstance(trig, dict):
        if trig.get("type") == "paper-stop" and any(c for c in stop_clicks if c):
            paper.stop_session(int(trig["index"]))
        elif trig.get("type") == "paper-del" and any(c for c in del_clicks if c):
            paper.delete_session(int(trig["index"]))
    elif tab != "paper":
        return no_update
    return _paper_body()


def _paper_body():
    sessions = paper.list_sessions()
    if not sessions:
        return html.Div(className="empty-state", children=[
            html.Div("אין סשנים עדיין", className="empty-title"),
            html.Div("בחר אסטרטגיה, ודא ש-IB Gateway מחובר, ולחץ \"התחל סשן בנייר\".", className="empty-sub")])
    cards = []
    for s in sessions:
        q = paper.session_summary(s)
        running = s["status"] == "running"
        ib = s.get("ib")
        badge = html.Span("פעיל" if running else "נעצר", className=("badge on" if running else "badge off"))
        ibtxt = "IB מחובר" if ib else ("IB לא זמין" if ib is False else "ממתין לבדיקה ראשונה")
        meta = (f"נפתח {s['started_at']} · בדיקה אחרונה {s.get('last_check') or '—'} · {ibtxt} · "
                f"{s.get('bars', 0):,} נרות · נר סגור אחרון {str(s.get('last_bar') or '—')[:16]}"
                + (f" · מחיר {s.get('last_price'):,.2f}" if s.get("last_price") else ""))
        kpis = html.Div(className="paper-kpis", children=[
            html.Div([html.Span(results.money(q["net"]), className=f"v {results._cls(q['net'])}"), html.Span("רווח ממומש", className="k")]),
            html.Div([html.Span(results.money(q["unrealized"]), className=f"v {results._cls(q['unrealized'])}"), html.Span("לא ממומש", className="k")]),
            html.Div([html.Span(str(q["closed"]), className="v"), html.Span("עסקאות סגורות", className="k")]),
            html.Div([html.Span(str(q["open"]), className="v"), html.Span("פוזיציות פתוחות", className="k")]),
            html.Div([html.Span(f"{q['win_rate']:.0f}%", className="v"), html.Span("אחוז זכייה", className="k")]),
        ])
        rows = []
        for t in sorted(s["trades"], key=lambda x: -x["id"])[:25]:
            is_open = t["status"] == "open"
            pnl = t["unrealized"] if is_open else t["pnl"]
            rows.append(html.Tr([
                html.Td(f"#{t['id']}", className="muted"),
                html.Td("לונג" if t["side"] == "long" else "שורט", className=("pos" if t["side"] == "long" else "neg")),
                html.Td(pd.Timestamp(t["entry_time"]).strftime("%d/%m %H:%M")), html.Td(f"{t['entry_price']:,.2f}"),
                html.Td(pd.Timestamp(t["exit_time"]).strftime("%d/%m %H:%M") if t["exit_time"] else "—"),
                html.Td(f"{t['exit_price']:,.2f}" if t["exit_price"] else "—"),
                html.Td(("פתוחה" if is_open else "סגורה"), className=("badge on" if is_open else "muted")),
                html.Td(results.money(pnl), className=results._cls(pnl)),
            ]))
        table = html.Div(html.Table(className="trades-table", children=[
            html.Thead(html.Tr([html.Th("#"), html.Th("כיוון"), html.Th("כניסה"), html.Th("מחיר"), html.Th("יציאה"), html.Th("מחיר"), html.Th("מצב"), html.Th("רווח/הפסד")])),
            html.Tbody(rows)]), className="table-wrap") if rows else html.Div("עדיין אין עסקאות בסשן הזה.", className="settings-hint")
        log = html.Div(className="paper-log", children=[html.Div(line) for line in (s.get("log") or [])[-10:][::-1]])
        err = html.Div(f"⚠ {s['error']}", className="tester-error") if s.get("error") else None
        params = " · ".join(f"{lab.param_en(k)}: {lab.fmt_value(v)}" for k, v in s["params"].items())
        cards.append(html.Div(className="paper-card", children=[
            html.Div(className="paper-head", children=[
                html.Div([html.Span(f"#{s['id']} {s['title']}", className="cat-title"), html.Span(f" · {s['symbol']} {s['timeframe']}", className="muted"), badge]),
                html.Div(className="cat-actions", children=[
                    html.Button("עצור", id={"type": "paper-stop", "index": s["id"]}, n_clicks=0, className="cat-btn", disabled=not running),
                    html.Button("מחק", id={"type": "paper-del", "index": s["id"]}, n_clicks=0, className="cat-btn"),
                ]),
            ]),
            html.Div(meta, className="cat-result-when"),
            html.Div(params, className="cat-result-when", title=params),
            err, kpis, table,
            html.Div("יומן", className="paper-log-title"), log,
        ]))
    return html.Div(cards)


@callback(
    Output("code-input", "value", allow_duplicate=True),
    Output("panel-tabs", "value", allow_duplicate=True),
    Input("strategy-select", "value"), State("code-input", "value"),
    prevent_initial_call=True,
)
def strategy_selected(key, code):
    """בחירה בבורר בראש הפאנל טוענת את האסטרטגיה מיד. אם הקוד כבר של אותה
    אסטרטגיה (הבורר סונכרן מהקוד) - לא נוגעים, כדי לא לדרוס פרמטרים שנערכו."""
    if not key or key not in lab.ENGINES or lab.engine_key_for_code(code) == key:
        return no_update, no_update
    return lab.ENGINES[key]["code"], "tester"


@callback(Output("lab-head-result", "children"), Output("strategy-select", "options"),
          Input("key-stats", "children"), Input("code-input", "value"), prevent_initial_call=True)
def refresh_lab_head(_stats, code):
    by = _RESULTS.get("by_engine", {})
    opts = []
    for key, eng in lab.ENGINES.items():
        last = by.get(key)
        tag = f"  ·  {_fmt_money(last['stats']['net'])}" if last else ""
        opts.append({"label": "\u200f" + eng["title"] + tag, "value": key})
    key = lab.engine_key_for_code(code)
    last = by.get(key)
    if not last:
        return "עדיין לא הורץ", opts
    st = last["stats"]
    return html.Span([html.Span(_fmt_money(st["net"]), className=("pos" if st["net"] > 0 else "neg")),
                      html.Span(f" · {st['trades']} עסקאות · PF {st['pf']:.2f}", className="muted")]), opts


# ---------------------------------------------------------------- כלי הגרף (assets/chart_tools.js)
def _ep(ts) -> int:
    from .data import _epoch as _e
    return int(_e(pd.Timestamp(ts)))


def _last_for(symbol: str, timeframe: str):
    saved = _RESULTS.get("last")
    if not saved:
        return None
    stats, cash, saved_symbol, saved_timeframe = saved
    if symbol != saved_symbol or timeframe != saved_timeframe:
        return None
    return stats, cash


@app.server.route("/api/trades")
def api_trades():
    """העסקאות שחופפות לטווח הנראה, עם סטופ/יעד וגיאומטריה - לציור על הגרף."""
    args = flask.request.args
    symbol = (args.get("symbol") or "").strip().upper(); timeframe = args.get("timeframe") or ""
    seq = args.get("seq")
    out = {"seq": seq, "trades": [], "total": 0}
    last = _last_for(symbol, timeframe)
    if not last:
        return flask.jsonify(out)
    stats, _cash = last
    tr = stats["_trades"]
    if tr.empty:
        return flask.jsonify(out)
    try:
        from_ts = from_display_epoch(int(args.get("from"))); to_ts = from_display_epoch(int(args.get("to")))
    except (TypeError, ValueError):
        return flask.jsonify(out)
    tr = tr.sort_values("EntryTime").reset_index(drop=True)
    tr["_i"] = np.arange(len(tr))
    view = tr[(tr["EntryTime"] <= to_ts) & (tr["ExitTime"] >= from_ts)]
    out["total"] = int(len(view))
    if len(view) > 400:
        view = view.iloc[_evenly_sampled_indices(len(view), 400)]
    rows = []
    has_sl = "SL" in view.columns; has_tp = "TP" in view.columns; has_ex = "Extra" in view.columns
    for _, r in view.iterrows():
        d = {"i": int(r["_i"]), "et": _ep(r["EntryTime"]), "xt": _ep(r["ExitTime"]),
             "ep": float(r["EntryPrice"]), "xp": float(r["ExitPrice"]), "side": "long" if r["Size"] > 0 else "short",
             "pnl": float(r["PnL"]), "ret": float(r["ReturnPct"]) * 100,
             "dur": float((pd.Timestamp(r["ExitTime"]) - pd.Timestamp(r["EntryTime"])).total_seconds() / 60)}
        if has_sl and np.isfinite(r["SL"]): d["sl"] = float(r["SL"])
        if has_tp and np.isfinite(r["TP"]): d["tp"] = float(r["TP"])
        if has_ex and isinstance(r["Extra"], list):
            geo = []
            for g in r["Extra"]:
                g2 = dict(g)
                for k in ("x1", "x2"):
                    if k in g2: g2[k] = _ep(g2[k])
                geo.append(g2)
            d["extra"] = geo
        rows.append(d)
    out["trades"] = rows
    return flask.jsonify(out)


@app.server.route("/api/trade_index")
def api_trade_index():
    """כל העסקאות בתמצית (זמן כניסה, רווח) לניווט קודם/הבא."""
    args = flask.request.args
    symbol = (args.get("symbol") or "").strip().upper(); timeframe = args.get("timeframe") or ""
    last = _last_for(symbol, timeframe)
    if not last:
        return flask.jsonify({"index": []})
    tr = last[0]["_trades"].sort_values("EntryTime")
    if tr.empty:
        return flask.jsonify({"index": []})
    et = to_display_naive(pd.DatetimeIndex(tr["EntryTime"])).astype("datetime64[s]").astype("int64").to_numpy()
    w = (tr["PnL"].to_numpy() > 0).astype(int)
    return flask.jsonify({"index": [[int(a), int(b)] for a, b in zip(et, w)]})


@app.server.route("/api/context")
def api_context():
    """הקשר לציור: טווח הבדיקה, חלונות Walk-forward, עקומת הון דגומה, ועסקאות נייר חיות."""
    args = flask.request.args
    symbol = (args.get("symbol") or "").strip().upper(); timeframe = args.get("timeframe") or ""
    out = {"tested": None, "wf": [], "equity": [], "paper": {"trades": [], "open": []}}
    last = _last_for(symbol, timeframe)
    if last:
        stats, _cash = last
        rng = _RESULTS.get("last_range")
        if rng:
            out["tested"] = {"from": _ep(rng[0]), "to": _ep(rng[1])}
        out["wf"] = [{k: _ep(pd.Timestamp(v)) for k, v in w.items()} for w in _RESULTS.get("wf", [])]
        tr = stats["_trades"]
        if not tr.empty:
            o = tr.sort_values("ExitTime")
            xt = to_display_naive(pd.DatetimeIndex(o["ExitTime"])).astype("datetime64[s]").astype("int64").to_numpy()
            cum = np.cumsum(o["PnL"].to_numpy(dtype=float))
            idx = _evenly_sampled_indices(len(cum), 3000)
            out["equity"] = [[int(xt[i]), float(round(cum[i], 2))] for i in idx]
    for sess in paper.list_sessions():
        if sess["symbol"] != symbol or sess["timeframe"] != timeframe or sess["status"] != "running":
            continue
        for t in sess["trades"]:
            d = {"side": t["side"], "et": _ep(t["entry_time"]), "ep": t["entry_price"], "id": t["id"], "session": sess["id"]}
            if t["status"] == "closed":
                d.update({"xt": _ep(t["exit_time"]), "xp": t["exit_price"], "pnl": t["pnl"]})
                out["paper"]["trades"].append(d)
            else:
                d.update({"unrealized": t.get("unrealized", 0.0)})
                out["paper"]["open"].append(d)
    return flask.jsonify(out)


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
        return ["tester", true, "מריץ..."];
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
        return [false, "\\u25b6  הרץ"];
    }
    """,
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("run-btn", "children", allow_duplicate=True),
    Input("result-body", "children"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
