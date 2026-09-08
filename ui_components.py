"""עיצוב בסגנון TradingView: CSS גלובלי, כותרת סימבול, פאנל Strategy Tester וטבלאות.

הצבעים והמידות נלקחו מקונפיג הגרף האמיתי של המשתמש ב-TradingView:
נרות ירוק #4CAF50 / אדום #F23645 עם פתילים אפורים #636363, ווליום #26A69A/#EF5350,
טקסט צירים #B2B5BE, מפריד פאנלים #2A2E39, גריד עדין rgba(240,243,250,0.06),
ורקע גרף שחור. שאר ה"קישוט" (סרגלים, טאבים, לוחות) בפלטת TradingView Dark הסטנדרטית.
"""
from __future__ import annotations

import html

import pandas as pd

# ---------------------------------------------------------------- design tokens
TV_BG = "#131722"          # רקע האפליקציה (TradingView Dark)
TV_CHART_BG = "#000000"    # רקע לוח הגרף — כפי שמוגדר אצל המשתמש
TV_PANEL = "#1E222D"
TV_PANEL_ALT = "#181B25"
TV_BORDER = "#2A2E39"
TV_TEXT = "#D1D4DC"
TV_TEXT_2 = "#B2B5BE"      # scalesProperties.textColor
TV_MUTED = "#787B86"
TV_BLUE = "#2962FF"
TV_UP = "#089981"          # ירוק Strategy Tester
TV_DOWN = "#F23645"        # אדום Strategy Tester

TV_CSS = """
<style>
:root {
    --tv-bg: #131722;
    --tv-chart-bg: #000000;
    --tv-panel: #1E222D;
    --tv-panel-alt: #181B25;
    --tv-panel-hover: #2A2E39;
    --tv-border: #2A2E39;
    --tv-text: #D1D4DC;
    --tv-text-2: #B2B5BE;
    --tv-muted: #787B86;
    --tv-blue: #2962FF;
    --tv-up: #089981;
    --tv-down: #F23645;
    --tv-font: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
    --tv-mono: 'SFMono-Regular', Consolas, Menlo, 'Fira Code', monospace;
    --tv-pine-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M12 3l4 5h-3v5h3l-4 5-4-5h3V8H8z' fill='black'/></svg>");
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"] {
    background-color: var(--tv-bg) !important;
    font-family: var(--tv-font) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppViewContainer"] * { color: var(--tv-text); }
[data-testid="stSidebar"] {
    background-color: var(--tv-panel) !important;
    border-right: 1px solid var(--tv-border);
}
[data-testid="stSidebar"] * { color: var(--tv-text) !important; }
section.main > div.block-container { padding-top: .9rem; max-width: 1600px; }
h1, h2, h3, h4 { color: var(--tv-text) !important; font-weight: 600 !important; }
hr { border-color: var(--tv-border) !important; }

/* ---------- inputs ---------- */
.stTextInput input, .stNumberInput input {
    background-color: var(--tv-panel-alt) !important;
    border: 1px solid var(--tv-border) !important;
    color: var(--tv-text) !important;
    border-radius: 4px !important;
    font-size: 13px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--tv-blue) !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--tv-panel-alt) !important;
    border-color: var(--tv-border) !important;
    border-radius: 4px !important;
    font-size: 13px !important;
}

/* Pine-editor-like code area */
.stTextArea textarea {
    background-color: #0D1117 !important;
    color: #D1D4DC !important;
    border: 1px solid var(--tv-border) !important;
    font-family: var(--tv-mono) !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    direction: ltr !important;
    text-align: left !important;
    border-radius: 0 0 4px 4px !important;
}
.stTextArea textarea:focus { border-color: var(--tv-blue) !important; box-shadow: none !important; }

/* ---------- buttons ---------- */
.stButton > button {
    background-color: transparent;
    border: 1px solid var(--tv-border);
    color: var(--tv-text);
    border-radius: 4px;
    font-weight: 500;
    font-size: 13px;
    transition: background-color .12s, border-color .12s, color .12s;
}
.stButton > button:hover { background-color: var(--tv-panel-hover); border-color: var(--tv-panel-hover); color: var(--tv-text); }
.stButton > button[kind="primary"] {
    background-color: var(--tv-blue) !important;
    border: none !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); }

/* ---------- tabs (TradingView underline style) ---------- */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--tv-border) !important;
    gap: 22px;
    background: transparent !important;
    padding-left: 2px;
}
[data-baseweb="tab"] {
    color: var(--tv-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 0 !important;
    background: transparent !important;
}
[data-baseweb="tab"]:hover { color: var(--tv-text) !important; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--tv-text) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--tv-blue) !important; height: 2px !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ---------- expanders ---------- */
[data-testid="stExpander"] {
    background-color: var(--tv-panel);
    border: 1px solid var(--tv-border) !important;
    border-radius: 6px;
    margin-top: 8px;
}
[data-testid="stExpander"] summary { font-weight: 600; font-size: 13px; padding: 9px 14px !important; }
[data-testid="stExpander"] summary:hover { color: var(--tv-blue) !important; }
[data-testid="stExpander"] summary svg { fill: var(--tv-muted); }
[data-testid="stExpanderDetails"] { padding: 4px 14px 14px !important; }

/* ---------- top toolbar ---------- */
div[data-testid="stHorizontalBlock"]:has(.tv-toolbar-marker) {
    background: var(--tv-panel);
    border: 1px solid var(--tv-border);
    border-radius: 6px;
    padding: 5px 8px;
    margin-bottom: 10px;
    align-items: center;
    gap: 6px !important;
}
div[data-testid="stHorizontalBlock"]:has(.tv-toolbar-marker) [data-testid="stTextInput"] input {
    font-weight: 700;
    font-size: 15px;
    letter-spacing: .02em;
    background: transparent !important;
    border-color: transparent !important;
}
div[data-testid="stHorizontalBlock"]:has(.tv-toolbar-marker) div[data-baseweb="select"] > div {
    background: transparent !important;
    border-color: transparent !important;
    font-weight: 600;
}
div[data-testid="stHorizontalBlock"]:has(.tv-toolbar-marker) .stButton > button { padding: .3rem .55rem; }

/* subtle "load more history" button above the chart */

/* ---------- symbol header bar ---------- */
.tv-symbol-bar {
    display: flex; align-items: center; flex-wrap: wrap; gap: 0 18px;
    background: var(--tv-panel); border: 1px solid var(--tv-border);
    border-radius: 6px; padding: 8px 14px; margin-bottom: 8px; font-size: 13px;
}
.tv-symbol-name { font-size: 15px; font-weight: 700; color: var(--tv-text); letter-spacing: .02em; }
.tv-symbol-meta { color: var(--tv-muted); font-size: 12px; }
.tv-symbol-ohlc { color: var(--tv-muted); font-size: 12.5px; letter-spacing: .01em; }
.tv-symbol-ohlc b { color: var(--tv-text); font-weight: 600; }
.tv-chip-label { color: var(--tv-muted); margin-right: 5px; }
.tv-chip-value { color: var(--tv-text); font-weight: 600; }
.tv-sep-dot { color: var(--tv-border); }

/* ---------- Strategy Tester panel (שחזור של הפאנל האמיתי ב-TradingView) ---------- */
.tv-st-panel {
    background: var(--tv-bg); border: 1px solid var(--tv-border);
    border-radius: 6px; margin-top: 4px; overflow: hidden;
}
/* הפאנל מפוצל לשני חלקים כדי שבורר "תקופת הבדיקה" (ווידג'ט Streamlit אמיתי) ישב
   בתוכו. הקצוות מיושרים כך שזה נראה כמו מסגרת אחת רציפה. */
.tv-st-panel-top { border-radius: 6px 6px 0 0; border-bottom: none; margin-bottom: 0; }
.tv-st-panel-bottom { border-radius: 0 0 6px 6px; border-top: none; margin-top: 0; }

/* שורת בורר תקופת הבדיקה — ממשיכה את מסגרת הפאנל משני הצדדים */
.st-key-tv_btrange {
    background: var(--tv-bg);
    border-left: 1px solid var(--tv-border);
    border-right: 1px solid var(--tv-border);
    border-bottom: 1px solid var(--tv-border);
    padding: 8px 14px 10px;
    gap: 0 !important;
}
.tv-btrange-label {
    color: var(--tv-muted); font-size: 11.5px; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
}
.tv-btrange-label b { color: var(--tv-text); font-weight: 600; }
.st-key-tv_btrange [role="radiogroup"] label,
.st-key-tv_btrange button {
    font-size: 12px !important;
}

/* רצועת הטאב העליונה עם שם האסטרטגיה */
.tv-st-tabbar {
    display: flex; align-items: stretch; height: 38px;
    background: var(--tv-panel-alt); border-bottom: 1px solid var(--tv-border);
}
.tv-st-tab {
    display: flex; align-items: center; gap: 8px; padding: 0 12px; max-width: 360px;
    background: var(--tv-bg); border-right: 1px solid var(--tv-border);
    font-size: 13px; font-weight: 500; color: var(--tv-text);
}
.tv-st-tab-icon { display: flex; color: var(--tv-up); flex: none; }
.tv-st-tab-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tv-st-tab-caret { color: var(--tv-muted); flex: none; }
.tv-st-tabbar-spacer { flex: 1; }
.tv-st-winbtns { display: flex; align-items: center; gap: 2px; padding-right: 8px; }
.tv-st-winbtn {
    display: flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 4px; color: var(--tv-muted);
}
.tv-st-winbtn:hover { background: var(--tv-panel-hover); color: var(--tv-text); }

/* סרגל הכלים: טווח תאריכים · הון · דטליזציה · הרצת סקריפט */
.tv-st-toolbar {
    display: flex; align-items: center; flex-wrap: wrap; gap: 1px;
    padding: 7px 8px; border-bottom: 1px solid var(--tv-border);
}
.tv-st-tool {
    display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    padding: 5px 8px; border-radius: 4px; font-size: 13px; color: var(--tv-text);
}
.tv-st-tool:hover { background: var(--tv-panel-hover); }
.tv-st-tool svg { color: var(--tv-muted); flex: none; }
.tv-st-tool-unit { color: var(--tv-muted); font-size: 11px; }
.tv-st-tool-badge {
    background: var(--tv-blue); color: #fff; border-radius: 9px;
    font-size: 10.5px; font-weight: 600; padding: 1px 6px; line-height: 1.5;
}
.tv-st-caret { color: var(--tv-muted); flex: none; }
.tv-st-toolsep { width: 1px; height: 18px; background: var(--tv-border); margin: 0 7px; }
.tv-st-toggles {
    display: inline-flex; border: 1px solid var(--tv-border);
    border-radius: 4px; overflow: hidden; margin-right: 5px;
}
.tv-st-toggle {
    display: flex; align-items: center; padding: 5px 9px; color: var(--tv-muted);
}
.tv-st-toggle.is-active { background: var(--tv-panel-hover); color: var(--tv-text); }

/* Key stats — ערך ואחוז על אותה שורה, "USD" כסיומת קטנה, בלי מפרידים אנכיים */
.tv-ks-wrap { padding: 15px 16px 18px; }
.tv-ks-title { font-size: 19px; font-weight: 600; color: var(--tv-text); margin-bottom: 15px; }
.tv-ks-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px 24px; }
@media (max-width: 900px) { .tv-ks-grid { grid-template-columns: repeat(2, 1fr); } }
.tv-ks-label { color: var(--tv-muted); font-size: 13px; margin-bottom: 8px; white-space: nowrap; }
.tv-ks-value {
    display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap;
    font-size: 17px; font-weight: 500; letter-spacing: -.01em; line-height: 1.2;
}
.tv-ks-unit { font-size: 11px; color: var(--tv-muted); font-weight: 400; }
.tv-ks-frac { color: var(--tv-text); }

/* כותרת ומקרא של אזור ה-Performance */
.tv-perf-head { display: flex; align-items: center; gap: 7px; margin: 2px 0 10px; }
.tv-perf-title { font-size: 15px; font-weight: 600; color: var(--tv-text); }
.tv-perf-help {
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%;
    border: 1px solid var(--tv-muted); color: var(--tv-muted);
    font-size: 10px; font-weight: 600;
}

.tv-pos { color: var(--tv-up) !important; }
.tv-neg { color: var(--tv-down) !important; }
.tv-neutral { color: var(--tv-text) !important; }

/* ---------- performance summary table ---------- */
.tv-table-wrap { overflow-x: auto; border: 1px solid var(--tv-border); border-radius: 6px; }
table.tv-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    background: var(--tv-panel); font-variant-numeric: tabular-nums;
}
table.tv-table th {
    text-align: right; color: var(--tv-muted); font-weight: 500; font-size: 11px;
    text-transform: uppercase; letter-spacing: .04em;
    padding: 10px 14px; border-bottom: 1px solid var(--tv-border);
    position: sticky; top: 0; background: var(--tv-panel-alt); white-space: nowrap;
}
table.tv-table th:first-child { text-align: left; }
table.tv-table td {
    padding: 8px 14px; border-bottom: 1px solid rgba(42,46,57,.55);
    white-space: nowrap; text-align: right; color: var(--tv-text);
}
table.tv-table td:first-child { text-align: left; color: var(--tv-text-2); }
table.tv-table tr:hover td { background: rgba(255,255,255,.028); }
table.tv-table tr:last-child td { border-bottom: none; }
.tv-row-group td:first-child { color: var(--tv-text); font-weight: 600; }

/* ---------- list of trades ---------- */
table.tv-trades tr.tv-entry-row td { color: var(--tv-muted); border-bottom: 1px solid var(--tv-border); }
.tv-long { color: var(--tv-up); font-weight: 600; }
.tv-short { color: var(--tv-down); font-weight: 600; }
.tv-num { color: var(--tv-muted); font-weight: 600; }
.tv-sig { color: var(--tv-muted); font-size: 12px; }

/* ---------- misc ---------- */
.tv-section-title { font-size: 13px; font-weight: 600; color: var(--tv-text); margin: 10px 0 8px; }
.tv-empty {
    padding: 26px 16px; text-align: center; color: var(--tv-muted); font-size: 13px;
    background: var(--tv-panel); border: 1px solid var(--tv-border); border-radius: 6px;
}

/* ================================================================
   פריסת העמוד המלאה של TradingView: סרגל כלים עליון, סרגל ציור אנכי משמאל,
   סרגל ווידג'טים אנכי מימין, סרגל ציר-זמן מתחת לגרף, ופאנלים תחתונים.
   המבנה, שמות הכפתורים והתפריטים נלקחו מקונפיג העמוד האמיתי של המשתמש.
   ================================================================ */

/* אין sidebar ואין סרגל של Streamlit — כמו ב-TradingView, כל ההגדרות יושבות בתוך
   העמוד עצמו (הגדרות ההרצה נמצאות בטאב Properties של ה-Strategy Tester, בדיוק כמו שם). */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu, footer { display: none !important; }

/* הכותרת הקבועה של Streamlit שקופה אך תופסת את כל רוחב ראש העמוד ובולעת קליקים —
   כך שכל סרגל הכלים העליון (סימבול, טיים-פריים, שאיבה, התפריטים) היה בלתי לחיץ.
   היא ריקה אצלנו ממילא, ולכן מוסרת לגמרי. */
[data-testid="stHeader"], header.stAppHeader { display: none !important; }

/* כפתור "טען היסטוריה נוספת" — נלחץ אוטומטית מה-JS של הגרף בגרירה אחורה, ולכן הוא
   קיים בעמוד אך מוסתר לגמרי (ב-TradingView אין כפתור כזה). לחיצה תכנותית מה-JS
   עובדת גם על אלמנט עם display:none, ולכן הגלילה האינסופית ממשיכה לעבוד. */
[data-testid="stElementContainer"]:has(.tv-load-more-marker) { display: none !important; }

/* אותו דבר לכפתורי החלפת הרזולוציה האוטומטית — ה-JS של הגרף לוחץ עליהם בזום. */
/* מיקוד לפי מפתח הווידג'ט: Streamlit מוסיף לכל ווידג'ט עם key מחלקה st-key-<key>.
   ניסיון קודם הסתיר את מיכל האב עם :has(> div .marker) — הסלקטור הזה מתלכד גם על
   מיכלים גבוהים יותר בעץ (לכל אחד מהם יש div-בן שמכיל את הסמן), וכך הוסתר כל העמוד. */
[class*="st-key-autores_"], .st-key-tv_autores, .st-key-load_more_btn { display: none !important; }

/* מתג ה-Auto בסרגל ציר-הזמן */
.st-key-tv_taxis [data-testid="stCheckbox"] label {
    font-size: 12px !important; color: var(--tv-text-2) !important;
}
.st-key-tv_taxis [data-testid="stCheckbox"] { padding: 0 4px; }

/* מרווח לסרגלים האנכיים הקבועים משני הצדדים */
section.main > div.block-container,
[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 100% !important;
    padding: 8px 62px 8px 62px !important;
}

/* ---------- סרגלים אנכיים קבועים (ציור משמאל, ווידג'טים מימין) ---------- */
.tv-vbar {
    position: fixed; top: 0; bottom: 0; width: 52px; z-index: 50;
    background: var(--tv-panel);
    display: flex; flex-direction: column; align-items: center;
    padding-top: 48px; overflow: hidden;
}
.tv-vbar-left { left: 0; border-right: 1px solid var(--tv-border); }
.tv-vbar-right { right: 0; border-left: 1px solid var(--tv-border); }
.tv-vbtn {
    position: relative; width: 38px; height: 34px; margin: 1px 0;
    display: flex; align-items: center; justify-content: center;
    border-radius: 4px; color: var(--tv-text-2); cursor: pointer;
}
.tv-vbtn:hover { background: var(--tv-panel-hover); color: #fff; }
.tv-vbtn.is-active { color: var(--tv-blue); }
.tv-vcaret {
    position: absolute; right: 3px; bottom: 3px; width: 0; height: 0;
    border-left: 3.5px solid transparent; border-bottom: 3.5px solid var(--tv-muted);
}
.tv-vbadge {
    position: absolute; top: 2px; right: 1px; background: var(--tv-down); color: #fff;
    font-size: 8.5px; font-weight: 700; line-height: 1; padding: 2px 3px; border-radius: 6px;
}
.tv-vsep { width: 30px; height: 1px; background: var(--tv-border); margin: 5px 0; flex: none; }
.tv-vspacer { flex: 1; }

/* ---------- סרגל הכלים העליון ---------- */
.st-key-tv_header {
    background: var(--tv-panel); border: 1px solid var(--tv-border);
    border-radius: 6px 6px 0 0; border-bottom: none;
    padding: 3px 6px; margin-bottom: 0; align-items: center; gap: 2px !important;
    position: relative; z-index: 120;
}
.st-key-tv_header [data-testid="stTextInput"] input {
    font-weight: 700; font-size: 15px; letter-spacing: .02em; height: 30px;
    background: transparent !important; border-color: transparent !important;
}
.st-key-tv_header [data-testid="stTextInput"] input:hover,
.st-key-tv_header div[data-baseweb="select"] > div:hover {
    background: var(--tv-panel-hover) !important;
}
.st-key-tv_header div[data-baseweb="select"] > div {
    background: transparent !important; border-color: transparent !important;
    font-weight: 600; min-height: 30px;
}
.st-key-tv_header .stButton > button {
    padding: 0 8px; height: 30px; border-color: transparent; background: transparent;
}
.st-key-tv_header [data-testid="stElementContainer"] { margin: 0 !important; }
/* המיכל עוטף את שורת העמודות, ולכן היישור והמרווח מוגדרים על הבלוק האופקי שבתוכו. */
.st-key-tv_header [data-testid="stHorizontalBlock"] { align-items: center; gap: 2px !important; }

.tv-hchrome { display: flex; align-items: center; gap: 1px; }
.tv-hbtn {
    display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 8px;
    border-radius: 4px; font-size: 13px; color: var(--tv-text); white-space: nowrap; cursor: pointer;
}
.tv-hbtn:hover { background: var(--tv-panel-hover); }
.tv-hbtn svg { color: var(--tv-text-2); flex: none; }
.tv-hbtn.is-accent, .tv-hbtn.is-accent svg { color: var(--tv-blue); }
.tv-hsep { width: 1px; height: 20px; background: var(--tv-border); margin: 0 5px; flex: none; }
.tv-hspacer { flex: 1; }
.tv-hnote { color: var(--tv-muted); font-size: 12px; padding: 0 6px; white-space: nowrap; }

/* תפריטים נפתחים ב-hover — מציגים את אותן אפשרויות שיש בכל כפתור ב-TradingView */
.tv-menu-wrap { position: relative; z-index: 200; }
.tv-menu {
    display: none; position: absolute; top: 32px; left: 0; min-width: 220px; z-index: 300;
    background: var(--tv-panel); border: 1px solid var(--tv-border); border-radius: 6px;
    box-shadow: 0 10px 28px rgba(0,0,0,.55); padding: 5px 0; cursor: default;
}
.tv-menu-wrap:hover > .tv-menu { display: block; }
.tv-menu-right { left: auto; right: 0; }
.tv-menu-up { top: auto; bottom: 32px; }
.tv-menu-head {
    padding: 7px 12px 4px; color: var(--tv-muted); font-size: 10.5px;
    text-transform: uppercase; letter-spacing: .06em;
}
.tv-menu-item {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 6px 12px; font-size: 13px; color: var(--tv-text); white-space: nowrap;
}
.tv-menu-item:hover { background: var(--tv-panel-hover); }
.tv-menu-item.is-checked { color: var(--tv-blue); }
.tv-menu-sub { color: var(--tv-muted); font-size: 11.5px; }
.tv-menu-sep { height: 1px; background: var(--tv-border); margin: 5px 0; }

/* ---------- סרגל ציר-הזמן שמתחת לגרף ---------- */
.st-key-tv_taxis {
    background: var(--tv-panel);
    border: 1px solid var(--tv-border); border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 2px 8px; gap: 0 !important;
}
.st-key-tv_taxis [data-testid="stElementContainer"] { margin: 0 !important; }
.st-key-tv_taxis [data-testid="stHorizontalBlock"] { align-items: center; gap: 0 !important; }
.st-key-tv_taxis button {
    font-size: 12px !important; padding: 2px 9px !important; min-height: 26px !important;
    border-color: transparent !important; background: transparent !important;
}
.st-key-tv_taxis button:hover { background: var(--tv-panel-hover) !important; }
.st-key-tv_taxis button[aria-checked="true"],
.st-key-tv_taxis button[kind="segmented_controlActive"] {
    background: var(--tv-panel-hover) !important; color: var(--tv-blue) !important;
}
.tv-taxis-right { display: flex; align-items: center; justify-content: flex-end; gap: 1px; }

/* הגרף עצמו — ממשיך את מסגרת הסרגל העליון בלי רווח ביניהם */
.st-key-tv_chart {
    border-left: 1px solid var(--tv-border); border-right: 1px solid var(--tv-border);
    background: var(--tv-chart-bg); gap: 0 !important;
}
.st-key-tv_chart iframe { display: block; }

/* ---------- Pine Editor (עורך הקוד כפאנל תחתון בסגנון TradingView) ---------- */
[data-testid="stExpander"]:has(.tv-pine-marker) {
    background: var(--tv-panel-alt); border-radius: 6px; margin-top: 6px;
}
[data-testid="stExpander"]:has(.tv-pine-marker) summary {
    font-size: 13px !important; font-weight: 500 !important; color: var(--tv-text) !important;
    background: var(--tv-panel-alt); border-radius: 6px 6px 0 0; padding: 8px 12px !important;
}
[data-testid="stExpander"]:has(.tv-pine-marker) summary::before {
    content: ""; display: inline-block; width: 15px; height: 15px; margin-left: 0; margin-right: 7px;
    vertical-align: -2px; background-color: var(--tv-blue);
    -webkit-mask: var(--tv-pine-icon) center/contain no-repeat;
    mask: var(--tv-pine-icon) center/contain no-repeat;
}

/* ---------- Strategy Tester: טאבים תחתונים ---------- */
.tv-props-grid { padding: 4px 2px 2px; }
.tv-props-row {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    padding: 9px 4px; border-bottom: 1px solid rgba(42,46,57,.55); font-size: 13px;
}
.tv-props-row:last-child { border-bottom: none; }
.tv-props-label { color: var(--tv-text-2); }
.tv-props-value { color: var(--tv-text); font-weight: 500; }
.tv-props-hint { color: var(--tv-muted); font-size: 11.5px; margin: 2px 0 10px; }
.tv-props-head {
    font-size: 12px; font-weight: 600; color: var(--tv-text); text-transform: uppercase;
    letter-spacing: .05em; margin: 14px 0 2px;
}
</style>
"""


# ---------------------------------------------------------------- helpers
def _no_blank_lines(fragment: str) -> str:
    """מסיר שורות ריקות כדי שה-Markdown parser של Streamlit לא ייצא ממצב HTML גולמי
    (שורה ריקה בתוך בלוק HTML מסיימת אותו ומפרשת את ההמשך כקוד)."""
    return "\n".join(line for line in fragment.splitlines() if line.strip())


def _cls(v: float) -> str:
    return "tv-pos" if v > 0 else ("tv-neg" if v < 0 else "tv-neutral")


def _money(v: float, signed: bool = True) -> str:
    sign = "−" if v < 0 else ("+" if signed and v > 0 else "")
    return f"{sign}{abs(v):,.2f}"


def _pct(v: float, signed: bool = True) -> str:
    sign = "−" if v < 0 else ("+" if signed and v > 0 else "")
    return f"{sign}{abs(v):.2f}%"


def _num(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (pd.isna(v) or v in (float("inf"), float("-inf")))):
        return "—"
    return f"{float(v):,.{digits}f}"


# ---------------------------------------------------------------- symbol header
def symbol_header_html(symbol: str, timeframe: str, data: pd.DataFrame, cash: float) -> str:
    """שורת הכותרת מעל הגרף: סימבול · טיים-פריים · OHLC של הנר האחרון · טווח · מס' נרות."""
    last = data.iloc[-1]
    prev_close = data["Close"].iloc[-2] if len(data) > 1 else last["Open"]
    change = last["Close"] - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    cls = _cls(change)

    start_s = html.escape(pd.Timestamp(data.index.min()).strftime("%b %d, %Y"))
    end_s = html.escape(pd.Timestamp(data.index.max()).strftime("%b %d, %Y"))

    out = f"""
    <div class="tv-symbol-bar">
        <div>
            <span class="tv-symbol-name">{html.escape(symbol)}</span>
            <span class="tv-symbol-meta"> · {html.escape(timeframe)}</span>
        </div>
        <div class="tv-symbol-ohlc">
            O<b>{last['Open']:.2f}</b>
            H<b>{last['High']:.2f}</b>
            L<b>{last['Low']:.2f}</b>
            C<b>{last['Close']:.2f}</b>
            <span class="{cls}">{_money(change)} ({_pct(change_pct)})</span>
        </div>
        <div><span class="tv-chip-label">Range</span><span class="tv-chip-value">{start_s} — {end_s}</span></div>
        <div><span class="tv-chip-label">Bars</span><span class="tv-chip-value">{len(data):,}</span></div>
        <div><span class="tv-chip-label">Capital</span><span class="tv-chip-value">${cash:,.0f}</span></div>
    </div>
    """
    return _no_blank_lines(out)


# ---------------------------------------------------------------- strategy tester
def _svg(paths: str, size: int = 15) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 18 18" fill="none">{paths}</svg>'
    )


_ICON_CARET = '<span class="tv-st-caret">▾</span>'
_ICON_CAL = _svg(
    '<rect x="2.5" y="3.5" width="13" height="12" rx="1.5" stroke="currentColor"/>'
    '<path d="M2.5 7h13M6 2v3M12 2v3" stroke="currentColor"/>'
)
_ICON_CASH = _svg(
    '<circle cx="9" cy="9" r="6.5" stroke="currentColor"/>'
    '<path d="M9 5.5v7M7 7.5h3.2M7 10.5h3.2" stroke="currentColor"/>'
)
_ICON_DETAIL = _svg(
    '<circle cx="8" cy="8" r="4.5" stroke="currentColor"/>'
    '<path d="M11.5 11.5l3.5 3.5M6 8h4" stroke="currentColor"/>'
)
_ICON_SCRIPT = _svg(
    '<path d="M3 12l3-4 2.5 2L12 5" stroke="currentColor"/>'
    '<path d="M13 3l.7 1.8L15.5 5.5l-1.8.7L13 8l-.7-1.8L10.5 5.5l1.8-.7z" fill="currentColor"/>'
)
_ICON_TARGET = _svg(
    '<circle cx="9" cy="9" r="6" stroke="currentColor"/><circle cx="9" cy="9" r="2" fill="currentColor"/>'
)
_ICON_CLOCK = _svg(
    '<circle cx="9" cy="10" r="5.5" stroke="currentColor"/>'
    '<path d="M9 7.5V10l1.8 1.2M4 4l2-1.5M14 4l-2-1.5" stroke="currentColor"/>'
)
_ICON_CURVE = _svg('<path d="M3 12c3 0 4-6 7-6s4 4 5 4" stroke="currentColor"/>', 16)
_ICON_TABLE = _svg(
    '<rect x="3" y="3.5" width="12" height="11" rx="1" stroke="currentColor"/>'
    '<path d="M3 7h12M9 7v7.5" stroke="currentColor"/>',
    16,
)
_ICON_CHART_TAB = _svg('<path d="M3 13l4-5 3 2.5L15 4" stroke="currentColor" stroke-width="1.6"/>', 15)


def strategy_tester_top_html(strategy_name: str, date_range_text: str, capital: float) -> str:
    """החלק העליון של הפאנל: רצועת הטאב עם שם האסטרטגיה וסרגל הכלים.

    טווח התאריכים שמוצג כאן הוא **תקופת הבדיקה של האסטרטגיה** — לא הטווח שנשאב לגרף.
    אלה שני דברים נפרדים: הטיים-פריים של הגרף הוא רזולוציית תצוגה לניתוח, ואילו
    תקופת הבדיקה קובעת על איזה חלון היסטוריה נבדקת האסטרטגיה.
    """
    cap_txt = f"{capital / 1000:g} K" if capital >= 1000 else f"{capital:g}"
    out = f"""
    <div class="tv-st-panel tv-st-panel-top">
        <div class="tv-st-tabbar">
            <div class="tv-st-tab">
                <span class="tv-st-tab-icon">{_ICON_CHART_TAB}</span>
                <span class="tv-st-tab-name">{html.escape(strategy_name)}</span>
                <span class="tv-st-tab-caret">▾</span>
            </div>
            <div class="tv-st-tabbar-spacer"></div>
            <div class="tv-st-winbtns">
                <span class="tv-st-winbtn">–</span>
                <span class="tv-st-winbtn">⤢</span>
            </div>
        </div>
        <div class="tv-st-toolbar">
            <span class="tv-st-toggles">
                <span class="tv-st-toggle is-active">{_ICON_CURVE}</span>
                <span class="tv-st-toggle">{_ICON_TABLE}</span>
            </span>
            <span class="tv-st-toolsep"></span>
            <span class="tv-st-tool">{_ICON_CAL}{html.escape(date_range_text)}</span>
            <span class="tv-st-toolsep"></span>
            <span class="tv-st-tool">{_ICON_CASH}{cap_txt}<span class="tv-st-tool-unit">USD</span>{_ICON_CARET}</span>
            <span class="tv-st-toolsep"></span>
            <span class="tv-st-tool">{_ICON_DETAIL}Default detalization{_ICON_CARET}</span>
            <span class="tv-st-toolsep"></span>
            <span class="tv-st-tool">{_ICON_SCRIPT}Script execution<span class="tv-st-tool-badge">1</span>{_ICON_CARET}</span>
            <span class="tv-st-toolsep"></span>
            <span class="tv-st-tool">{_ICON_TARGET}</span>
            <span class="tv-st-tool">{_ICON_CLOCK}</span>
        </div>
    </div>
    """
    return _no_blank_lines(out)


def key_stats_html(
    net_profit: float,
    net_profit_pct: float,
    max_dd: float,
    max_dd_pct: float,
    total_trades: int,
    profit_factor,
    percent_profitable: float,
    winning_trades: int,
    period_note: str = "",
) -> str:
    """ארבעת מדדי ה-Key stats, בהיררכיה של TradingView: ערך ואחוז על אותה שורה,
    'USD' כסיומת קטנה, בלי מפרידים אנכיים, ו-Max drawdown כגודל חיובי בצבע רגיל.

    period_note: כשמסננים לפי תקופה מעל גרף ה-Performance (Overview) - שורה
    קטנה מתחת לכותרת שמראה איזו תקופה נבחרה וכמה עסקאות מתוך הסך-הכול נופלות
    בה (ראו dashboard/panels.key_stats_for_period), כדי שהמשתמש לא יחשוב בטעות
    שהמספרים משקפים את כל ההיסטוריה. ריק (ברירת המחדל) - בלי שינוי מהתצוגה הרגילה."""
    pnl_cls = _cls(net_profit)
    note_html = f'<div class="tv-ks-period">{html.escape(period_note)}</div>' if period_note else ""
    out = f"""
    <div class="tv-st-panel tv-st-panel-bottom">
        <div class="tv-ks-wrap">
            <div class="tv-ks-title">Key stats</div>
            {note_html}
            <div class="tv-ks-grid">
                <div>
                    <div class="tv-ks-label">Total PnL</div>
                    <div class="tv-ks-value {pnl_cls}">
                        <span>{_money(net_profit)}<span class="tv-ks-unit">USD</span></span>
                        <span>{_pct(net_profit_pct)}</span>
                    </div>
                </div>
                <div>
                    <div class="tv-ks-label">Max drawdown</div>
                    <div class="tv-ks-value">
                        <span>{_money(abs(max_dd), signed=False)}<span class="tv-ks-unit">USD</span></span>
                        <span>{abs(max_dd_pct):,.2f}%</span>
                    </div>
                </div>
                <div>
                    <div class="tv-ks-label">Profitable trades</div>
                    <div class="tv-ks-value">
                        <span>{percent_profitable:.2f}%</span>
                        <span class="tv-ks-frac">{winning_trades}/{total_trades}</span>
                    </div>
                </div>
                <div>
                    <div class="tv-ks-label">Profit factor</div>
                    <div class="tv-ks-value"><span>{_num(profit_factor, 3)}</span></div>
                </div>
            </div>
        </div>
    </div>
    """
    return _no_blank_lines(out)


def performance_heading_html() -> str:
    return _no_blank_lines(
        '<div class="tv-perf-head">'
        '<span class="tv-perf-title">Performance</span>'
        '<span class="tv-perf-help">?</span>'
        "</div>"
    )


def performance_summary_html(stats, trades: pd.DataFrame, cash: float) -> str:
    """טבלת 'Performance Summary' בסגנון TradingView, עם פירוק ל-All / Long / Short."""

    def _slice(df: pd.DataFrame, side: str | None) -> pd.DataFrame:
        if side == "long":
            return df[df["Size"] > 0]
        if side == "short":
            return df[df["Size"] < 0]
        return df

    def _block(df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                "net": 0.0, "gross_p": 0.0, "gross_l": 0.0, "pf": None, "n": 0,
                "win": 0, "loss": 0, "wr": 0.0, "avg": 0.0, "best": 0.0, "worst": 0.0,
            }
        pnl = df["PnL"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_p, gross_l = float(wins.sum()), float(abs(losses.sum()))
        return {
            "net": float(pnl.sum()),
            "gross_p": gross_p,
            "gross_l": gross_l,
            "pf": (gross_p / gross_l) if gross_l > 0 else None,
            "n": len(df),
            "win": int((pnl > 0).sum()),
            "loss": int((pnl <= 0).sum()),
            "wr": float((pnl > 0).mean() * 100),
            "avg": float(pnl.mean()),
            "best": float(pnl.max()),
            "worst": float(pnl.min()),
        }

    cols = [_block(_slice(trades, s)) for s in (None, "long", "short")]

    def _row(label: str, fmt, key: str, colored: bool = True, group: bool = False) -> str:
        tds = []
        for c in cols:
            v = c[key]
            if v is None:
                tds.append("<td>—</td>")
                continue
            cls = _cls(v) if colored and isinstance(v, (int, float)) else ""
            tds.append(f'<td class="{cls}">{fmt(v)}</td>')
        cls_tr = ' class="tv-row-group"' if group else ""
        return f"<tr{cls_tr}><td>{label}</td>{''.join(tds)}</tr>"

    money = lambda v: _money(v)                       # noqa: E731
    money_ns = lambda v: _money(v, signed=False)      # noqa: E731
    plain = lambda v: f"{v:,}"                        # noqa: E731
    pctf = lambda v: f"{v:.2f}%"                      # noqa: E731
    ratio = lambda v: _num(v)                         # noqa: E731

    # מדדים שהם ברמת התיק כולו (לא ניתנים לפירוק לונג/שורט) נלקחים מ-stats.
    sharpe = _num(stats.get("Sharpe Ratio"))
    sortino = _num(stats.get("Sortino Ratio"))
    bh = float(stats.get("Buy & Hold Return [%]", 0.0) or 0.0)
    max_dd_pct = abs(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0))
    max_dd_usd = max_dd_pct / 100 * cash

    rows = [
        _row("Net profit", money, "net", group=True),
        _row("Gross profit", money_ns, "gross_p", colored=False),
        _row("Gross loss", money_ns, "gross_l", colored=False),
        _row("Profit factor", ratio, "pf", colored=False),
        _row("Total closed trades", plain, "n", colored=False, group=True),
        _row("Winning trades", plain, "win", colored=False),
        _row("Losing trades", plain, "loss", colored=False),
        _row("Percent profitable", pctf, "wr", colored=False),
        _row("Avg P&amp;L per trade", money, "avg", group=True),
        _row("Largest winning trade", money, "best"),
        _row("Largest losing trade", money, "worst"),
    ]

    portfolio_rows = f"""
        <tr class="tv-row-group"><td>Max equity drawdown</td>
            <td class="tv-neg">{_money(-max_dd_usd, signed=False)}</td><td>—</td><td>—</td></tr>
        <tr><td>Max drawdown %</td>
            <td class="tv-neg">{max_dd_pct:.2f}%</td><td>—</td><td>—</td></tr>
        <tr><td>Buy &amp; hold return</td>
            <td class="{_cls(bh)}">{_pct(bh)}</td><td>—</td><td>—</td></tr>
        <tr><td>Sharpe ratio</td><td>{sharpe}</td><td>—</td><td>—</td></tr>
        <tr><td>Sortino ratio</td><td>{sortino}</td><td>—</td><td>—</td></tr>
    """

    out = f"""
    <div class="tv-table-wrap">
    <table class="tv-table">
        <thead><tr><th>Metric</th><th>All</th><th>Long</th><th>Short</th></tr></thead>
        <tbody>
            {''.join(rows)}
            {portfolio_rows}
        </tbody>
    </table>
    </div>
    """
    return _no_blank_lines(out)


# ---------------------------------------------------------------- list of trades
def trades_table_html(trades: pd.DataFrame, page: int = 0, page_size: int | None = None) -> str:
    """'List of Trades' בסגנון TradingView: שתי שורות לעסקה (Exit מעל Entry),
    מספור יורד (העסקה האחרונה למעלה) ו-PnL מצטבר.

    TradeNum/CumPnL מחושבים תמיד על כל הטבלה (סדר כרונולוגי, וקטורי וזול גם
    ב-150K+ עסקאות) *לפני* דיפוג ל-page_size שורות - אחרת מספור העסקאות
    והמצטבר של עמוד 2+ היו שגויים (מתחילים מ-1/0 מחדש בכל עמוד). רק בניית ה-HTML
    בפועל (הלולאה למטה) מוגבלת לעמוד המבוקש - זה החלק שהיה תופס עשרות שניות
    ומקפיא את הדפדפן (עשרות/מאות אלפי שורות DOM) על "Full backtest".
    """
    ordered = trades.sort_values("EntryTime").reset_index(drop=True)
    ordered["TradeNum"] = range(1, len(ordered) + 1)
    ordered["CumPnL"] = ordered["PnL"].cumsum()
    ordered = ordered.sort_values("TradeNum", ascending=False).reset_index(drop=True)

    if page_size:
        start = max(0, page) * page_size
        ordered = ordered.iloc[start:start + page_size]

    # עיצוב התאריכים בבת אחת (וקטורי - Series.dt.strftime), לא
    # pd.Timestamp(...).strftime(...) לכל שורה בלולאה - על "Full backtest"
    # (יכול לצאת 150K+ עסקאות על כל ההיסטוריה) ההבדל נמדד כ-25 שניות מול
    # פחות משנייה.
    exit_str = pd.to_datetime(ordered["ExitTime"]).dt.strftime("%b %d, %Y %H:%M").to_numpy()
    entry_str = pd.to_datetime(ordered["EntryTime"]).dt.strftime("%b %d, %Y %H:%M").to_numpy()

    # מערכי numpy במקום DataFrame.iterrows(): .iterrows() בונה אובייקט Series
    # חדש לכל שורה בנפרד (עלות עצומה על 150K+ שורות - נמדד לבדו כ-20 שניות
    # מתוך כ-25 הכוללות, בלי שום קשר לתוכן הלולאה עצמו).
    trade_num = ordered["TradeNum"].to_numpy()
    size_arr = ordered["Size"].to_numpy()
    exit_price = ordered["ExitPrice"].to_numpy()
    entry_price = ordered["EntryPrice"].to_numpy()
    pnl_arr = ordered["PnL"].to_numpy()
    ret_arr = ordered["ReturnPct"].to_numpy()
    cum_arr = ordered["CumPnL"].to_numpy()

    rows = []
    for i in range(len(ordered)):
        is_long = size_arr[i] > 0
        direction = "Long" if is_long else "Short"
        dir_cls = "tv-long" if is_long else "tv-short"
        pnl, ret, cum = pnl_arr[i], ret_arr[i], cum_arr[i]
        size = abs(size_arr[i])

        rows.append(
            f"""
            <tr>
                <td rowspan="2" class="tv-num">{int(trade_num[i])}</td>
                <td rowspan="2" class="{dir_cls}">{direction}</td>
                <td class="tv-sig">Exit</td>
                <td>{exit_str[i]}</td>
                <td>{exit_price[i]:.4f}</td>
                <td>{size:g}</td>
                <td class="{_cls(pnl)}">{_money(pnl)}</td>
                <td class="{_cls(ret)}">{_pct(ret * 100)}</td>
                <td rowspan="2" class="{_cls(cum)}">{_money(cum)}</td>
            </tr>
            <tr class="tv-entry-row">
                <td class="tv-sig">Entry</td>
                <td>{entry_str[i]}</td>
                <td>{entry_price[i]:.4f}</td>
                <td>{size:g}</td>
                <td>—</td>
                <td>—</td>
            </tr>
            """
        )

    out = f"""
    <div class="tv-table-wrap">
    <table class="tv-table tv-trades">
        <thead>
            <tr>
                <th>Trade #</th><th>Type</th><th>Signal</th><th>Date / Time</th>
                <th>Price</th><th>Qty</th><th>Net P&amp;L</th><th>Return</th><th>Cumulative P&amp;L</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    """
    return _no_blank_lines(out)


def empty_state_html(text: str) -> str:
    return _no_blank_lines(f'<div class="tv-empty">{html.escape(text)}</div>')


# ================================================================ full TV chrome
# הסרגלים, שמות הכפתורים והתפריטים למטה משחזרים אחד-לאחד את מה שקיים בעמוד
# TradingView של המשתמש (aria-label / data-name של הפקדים האמיתיים שם).


def _ic(paths: str, size: int = 20, w: float = 1.3) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="{w}" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )


# --- סרגל הציור השמאלי -------------------------------------------------------
_LEFT_TOOLS = [
    ("Cursors", '<path d="M5 3l14 8-6 1.5L10 19z"/>', True, True),
    ("Trend line tools", '<path d="M4 19L20 5"/><circle cx="4" cy="19" r="2"/><circle cx="20" cy="5" r="2"/>', True, False),
    ("Gann and Fibonacci tools", '<path d="M3 6h18M3 10h18M3 14h18M3 18h18"/><path d="M3 18L21 6"/>', True, False),
    ("Patterns", '<path d="M3 17l4-9 5 6 4-8 5 11"/>', True, False),
    ("Forecasting and measurement tools", '<rect x="3" y="4" width="18" height="7" rx="1"/><rect x="3" y="13" width="18" height="7" rx="1"/>', True, False),
    ("Geometric shapes", '<path d="M4 18c3-10 8 6 12-6"/><path d="M16 12l4-3"/>', True, False),
    ("Annotation tools", '<path d="M5 6V4h14v2"/><path d="M12 4v16"/><path d="M9 20h6"/>', True, False),
    ("Icons", '<path d="M12 3l2.5 5.5L20 9.5l-4 4 1 6-5-2.8L7 19.5l1-6-4-4 5.5-1z"/>', True, False),
    ("Measure", '<path d="M3 9h18v6H3z"/><path d="M7 9v3M11 9v4M15 9v3M19 9v4"/>', False, False),
    ("Zoom in", '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21M8 10.5h5M10.5 8v5"/>', False, False),
    ("Magnets", '<path d="M6 4v7a6 6 0 0012 0V4"/><path d="M6 9h4M14 9h4"/>', True, False),
]

_LEFT_MODES = [
    ("Keep drawing", '<path d="M4 20l4-1 9-9-3-3-9 9z"/><path d="M14 6l3 3"/><path d="M18 18h4"/>'),
    ("Lock drawings and indicators", '<rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/>'),
    ("Hide all drawings", '<path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="2.5"/><path d="M4 20L20 4"/>'),
    ("Sync drawings options", '<path d="M4 10a8 8 0 0113-4M20 14a8 8 0 01-13 4"/><path d="M17 3v3.5h-3.5M7 21v-3.5h3.5"/>'),
    ("Remove objects", '<path d="M4 7h16"/><path d="M9 7V5h6v2"/><path d="M6 7l1 13h10l1-13"/>'),
]

# --- סרגל הווידג'טים הימני ---------------------------------------------------
_RIGHT_TOOLS = [
    ("Watchlist, details, and news", '<path d="M4 6h16M4 10h16M4 14h10M4 18h10"/><path d="M17 15l2 2 3-4"/>', None),
    ("Alerts", '<path d="M18 9a6 6 0 10-12 0c0 6-2 7-2 7h16s-2-1-2-7z"/><path d="M10.5 20a2 2 0 003 0"/>', "143"),
    ("Object tree and data window", '<path d="M4 5h6v6H4zM14 5h6v6h-6zM4 15h6v4H4zM14 15h6v4h-6z"/>', None),
    ("Chats", '<path d="M4 5h16v11H9l-5 4z"/>', None),
    ("Screeners", '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>', None),
    ("Pine", '<path d="M12 3l4 5h-3v5h3l-4 5-4-5h3V8H8z"/>', None),
    ("Calendars", '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>', None),
    ("Community", '<circle cx="9" cy="9" r="3"/><circle cx="17" cy="11" r="2.5"/><path d="M3 19c0-3 3-5 6-5s6 2 6 5"/>', None),
]

_RIGHT_FOOTER = [
    ("Notifications", '<path d="M18 9a6 6 0 10-12 0c0 6-2 7-2 7h16s-2-1-2-7z"/><path d="M10.5 20a2 2 0 003 0"/>'),
    ("Help Center", '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 113 2.5v1.5"/><path d="M12 17h.01"/>'),
]


def left_toolbar_html() -> str:
    """סרגל כלי הציור האנכי השמאלי — אותן קבוצות כלים ואותם שמות כמו ב-TradingView
    (linetool-group-*, magnet, lock-all, hide-all, removeAllDrawingTools וכו')."""
    parts = ['<div class="tv-vbar tv-vbar-left">']
    for name, path, has_caret, active in _LEFT_TOOLS:
        caret = '<span class="tv-vcaret"></span>' if has_caret else ""
        cls = "tv-vbtn is-active" if active else "tv-vbtn"
        parts.append(f'<span class="{cls}" title="{name}">{_ic(path)}{caret}</span>')
    parts.append('<span class="tv-vsep"></span>')
    for name, path in _LEFT_MODES:
        parts.append(f'<span class="tv-vbtn" title="{name}">{_ic(path)}<span class="tv-vcaret"></span></span>')
    parts.append('<span class="tv-vspacer"></span>')
    parts.append(
        '<span class="tv-vbtn" title="Show Favorite Drawing Tools Toolbar">'
        + _ic('<path d="M12 4l2.4 5 5.6.8-4 4 1 5.6-5-2.7-5 2.7 1-5.6-4-4 5.6-.8z"/>')
        + "</span>"
    )
    parts.append("</div>")
    return _no_blank_lines("".join(parts))


def right_toolbar_html() -> str:
    """סרגל הווידג'טים האנכי הימני (widgetbar / right-toolbar) — Watchlist, Alerts,
    Object tree, Chats, Screeners, Pine, Calendars, Community, Notifications, Help."""
    parts = ['<div class="tv-vbar tv-vbar-right">']
    for name, path, badge in _RIGHT_TOOLS:
        badge_html = f'<span class="tv-vbadge">{badge}</span>' if badge else ""
        parts.append(f'<span class="tv-vbtn" title="{name}">{_ic(path)}{badge_html}</span>')
    parts.append('<span class="tv-vspacer"></span>')
    parts.append('<span class="tv-vsep"></span>')
    for name, path in _RIGHT_FOOTER:
        parts.append(f'<span class="tv-vbtn" title="{name}">{_ic(path)}</span>')
    parts.append("</div>")
    return _no_blank_lines("".join(parts))


# --- סרגל הכלים העליון -------------------------------------------------------
def _menu(items, head: str | None = None, right: bool = False, up: bool = False) -> str:
    cls = "tv-menu" + (" tv-menu-right" if right else "") + (" tv-menu-up" if up else "")
    rows = []
    if head:
        rows.append(f'<div class="tv-menu-head">{head}</div>')
    for it in items:
        if it is None:
            rows.append('<div class="tv-menu-sep"></div>')
            continue
        label, sub, checked = (list(it) + [None, False])[:3] if isinstance(it, (list, tuple)) else (it, None, False)
        sub_html = f'<span class="tv-menu-sub">{html.escape(str(sub))}</span>' if sub else ""
        item_cls = "tv-menu-item is-checked" if checked else "tv-menu-item"
        rows.append(f'<div class="{item_cls}"><span>{html.escape(str(label))}</span>{sub_html}</div>')
    return f'<div class="{cls}">{"".join(rows)}</div>'


_H_ICONS = {
    "compare": '<path d="M12 5v14M5 12h14"/>',
    "candles": '<path d="M7 4v16M7 8h0M17 4v16"/><rect x="5" y="8" width="4" height="8" rx="1"/><rect x="15" y="6" width="4" height="10" rx="1"/>',
    "indicators": '<path d="M4 17c3 0 3-10 6-10s3 6 5 6 3-4 5-4"/>',
    "templates": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M9 9v11"/>',
    "alert": '<path d="M18 9a6 6 0 10-12 0c0 6-2 7-2 7h16s-2-1-2-7z"/><path d="M10.5 20a2 2 0 003 0"/>',
    "replay": '<path d="M4 5v6h6"/><path d="M4.5 11a8 8 0 113 8"/><path d="M11 9l5 3-5 3z"/>',
    "undo": '<path d="M4 8h9a5 5 0 010 10h-6"/><path d="M8 4L4 8l4 4"/>',
    "redo": '<path d="M20 8h-9a5 5 0 000 10h6"/><path d="M16 4l4 4-4 4"/>',
    "layout": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/>',
    "search": '<circle cx="11" cy="11" r="6"/><path d="M16 16l5 5"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
    "fullscreen": '<path d="M4 9V4h5M20 15v5h-5M20 9V4h-5M4 15v5h5"/>',
    "camera": '<rect x="3" y="7" width="18" height="13" rx="2"/><circle cx="12" cy="13.5" r="3.5"/><path d="M8 7l1.5-3h5L16 7"/>',
    "share": '<circle cx="6" cy="12" r="2.5"/><circle cx="17" cy="6" r="2.5"/><circle cx="17" cy="18" r="2.5"/><path d="M8.2 10.8l6.6-3.6M8.2 13.2l6.6 3.6"/>',
}

_CHART_TYPES = [
    ("Bars", "Alt+1"), ("Candles", "Alt+2", True), ("Hollow candles", "Alt+3"),
    ("Line", "Alt+4"), ("Line with markers", None), ("Step line", None), None,
    ("Area", "Alt+5"), ("HLC area", None), ("Baseline", None), ("Columns", None), None,
    ("High-low", None), ("Heikin Ashi", None), ("Renko", None), ("Line break", None),
    ("Kagi", None), ("Point & figure", None), ("Range", None),
]

_INDICATOR_MENU = [
    ("Indicators, metrics, and strategies", None), ("Favorite indicators", None),
    ("Indicator templates", None), None,
    ("Volume", "on", True), ("SMA", "on", True), ("VWAP", "on", True),
    ("Volume MA", None), ("Bollinger Bands", None), ("RSI", None),
]

_SETTINGS_MENU = [
    ("Symbol", None), ("Scales", None), ("Canvas", None), ("Trading", None),
    ("Events", None), None,
    ("Chart Snapshot", None), ("Save chart image", None), ("Reset chart", None),
]

_LAYOUT_MENU = [
    ("Layout setup", None), ("Manage layouts", None), ("Save chart layout", None),
    ("Make a copy", None), ("Rename chart layout", None), None,
    ("Symbol/interval chart syncing", None), ("Compare symbols", None),
]


def header_chrome_html(chart_type: str = "Candles", saved_text: str = "All changes saved") -> str:
    """כל שאר סרגל הכלים העליון של TradingView, מימין לבורר הסימבול/הטיים-פריים.

    כל כפתור נושא את שם הפקד האמיתי מהעמוד שלך (aria-label), ולכפתורים שיש להם
    תפריט ב-TradingView נפתח כאן אותו תפריט עם אותן אפשרויות.
    """
    def btn(icon_key: str, label: str = "", title: str = "", caret: bool = False, accent: bool = False) -> str:
        cls = "tv-hbtn is-accent" if accent else "tv-hbtn"
        text = f"<span>{html.escape(label)}</span>" if label else ""
        car = '<span class="tv-st-caret">▾</span>' if caret else ""
        return f'<span class="{cls}" title="{html.escape(title or label)}">{_ic(_H_ICONS[icon_key], 18)}{text}{car}</span>'

    out = f"""
    <div class="tv-hchrome">
        {btn("compare", "", "Compare symbols")}
        <span class="tv-hsep"></span>
        <span class="tv-menu-wrap">{btn("candles", chart_type, "Chart type", caret=True)}
            {_menu(_CHART_TYPES, head="Chart type")}</span>
        <span class="tv-hsep"></span>
        <span class="tv-menu-wrap">{btn("indicators", "Indicators", "Indicators, metrics, and strategies")}
            {_menu(_INDICATOR_MENU, head="Indicators")}</span>
        {btn("templates", "", "Indicator templates")}
        <span class="tv-hsep"></span>
        {btn("alert", "Alert", "Create alert")}
        {btn("replay", "Replay", "Bar replay")}
        <span class="tv-hsep"></span>
        {btn("undo", "", "Undo")}
        {btn("redo", "", "Redo")}
        <span class="tv-hspacer"></span>
        <span class="tv-hnote">{html.escape(saved_text)}</span>
        <span class="tv-menu-wrap">{btn("layout", "", "Layout setup", caret=True)}
            {_menu(_LAYOUT_MENU, head="Layouts", right=True)}</span>
        {btn("search", "", "Quick search")}
        <span class="tv-menu-wrap">{btn("settings", "", "Settings", caret=True)}
            {_menu(_SETTINGS_MENU, head="Chart settings", right=True)}</span>
        {btn("fullscreen", "", "Fullscreen mode")}
        {btn("camera", "", "Take a snapshot")}
        {btn("share", "", "Share your idea with the trade community")}
    </div>
    """
    return _no_blank_lines(out)


# --- סרגל ציר-הזמן שמתחת לגרף ------------------------------------------------
_TZ_MENU = [
    ("Exchange time zone", None), ("UTC", None), ("New York", "UTC-4", True),
    ("Chicago", "UTC-5"), ("London", "UTC+1"), ("Tel Aviv", "UTC+3"), ("Tokyo", "UTC+9"),
]
_SESSION_MENU = [("Regular trading hours", None, True), ("Extended trading hours", None)]


def time_axis_right_html(timezone_label: str = "New York", session_label: str = "Regular",
                         adjusted: bool = False) -> str:
    """הצד הימני של סרגל ציר-הזמן: מעבר לתאריך, אזור זמן, סשן, התאמה לדיבידנדים
    וכפתורי הגלילה — בדיוק כמו ב-TradingView (go-to-date / time-zone-menu / session-menu / adj)."""
    def b(title: str, inner: str, caret: bool = False) -> str:
        car = '<span class="tv-st-caret">▾</span>' if caret else ""
        return f'<span class="tv-hbtn" title="{html.escape(title)}">{inner}{car}</span>'

    cal = _ic('<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>', 16)
    clock = _ic('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>', 16)
    adj_txt = "adj" if adjusted else "adj"
    out = f"""
    <div class="tv-taxis-right">
        {b("Go to", cal + '<span>Go to</span>')}
        <span class="tv-hsep"></span>
        <span class="tv-menu-wrap">{b("Timezone", clock + f'<span>{html.escape(timezone_label)}</span>', caret=True)}
            {_menu(_TZ_MENU, head="Time zone", right=True, up=True)}</span>
        <span class="tv-menu-wrap">{b("Session", f'<span>{html.escape(session_label)}</span>', caret=True)}
            {_menu(_SESSION_MENU, head="Session", right=True, up=True)}</span>
        <span class="tv-hbtn" title="Adjust data for dividends"><span>{adj_txt}</span></span>
        <span class="tv-hsep"></span>
        <span class="tv-hbtn" title="Scroll to the left">{_ic('<path d="M15 5l-7 7 7 7"/>', 16)}</span>
        <span class="tv-hbtn" title="Scroll to the right">{_ic('<path d="M9 5l7 7-7 7"/>', 16)}</span>
        <span class="tv-hbtn" title="Scroll to the most recent bar">{_ic('<path d="M7 5l7 7-7 7"/><path d="M17 5v14"/>', 16)}</span>
        <span class="tv-hbtn" title="Maximize chart">{_ic(_H_ICONS["fullscreen"], 16)}</span>
    </div>
    """
    return _no_blank_lines(out)


def pine_editor_tag_html() -> str:
    """סמן בלבד — מזהה את האקורדיון של עורך הקוד כך שה-CSS יעצב אותו כפאנל
    Pine Editor של TradingView (אייקון הברק בכותרת ורקע הפאנל)."""
    return '<span class="tv-pine-marker" style="display:none"></span>'


def strategy_properties_html(pyramiding: int = 1, slippage: int = 0, margin_long: int = 100,
                             margin_short: int = 100, recalculate: str = "After order is filled") -> str:
    """טאב Properties של ה-Strategy Tester — אותם שדות שיש ב-TradingView.
    הערכים שניתן לשנות באמת (הון, גודל פוזיציה, עמלה) הם ווידג'טים אמיתיים מעל,
    והשורות כאן הן שאר ההגדרות של מנוע ההרצה."""
    rows = [
        ("Base currency", "USD"),
        ("Pyramiding", f"{pyramiding} order"),
        ("Slippage", f"{slippage} ticks"),
        ("Verify price for limit orders", "0 ticks"),
        ("Margin for long positions", f"{margin_long}%"),
        ("Margin for short positions", f"{margin_short}%"),
        ("Recalculate", recalculate),
        ("Fill orders", "On bar close"),
    ]
    body = "".join(
        f'<div class="tv-props-row"><span class="tv-props-label">{html.escape(k)}</span>'
        f'<span class="tv-props-value">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    return _no_blank_lines(f'<div class="tv-props-grid">{body}</div>')
