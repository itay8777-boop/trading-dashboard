"""TradingView-style Strategy Tester מקומי, מבוסס Streamlit + IB TWS + backtesting.py.

פריסת העמוד משחזרת את ממשק TradingView: סרגל כלים עליון, סרגל כלי ציור אנכי משמאל,
סרגל ווידג'טים אנכי מימין, גרף במרכז עם סרגל ציר-זמן וטווחי תאריכים מתחתיו, ומתחת
לכל אלה פאנל Pine Editor ופאנל Strategy Tester — לפי הפקדים והתפריטים שקיימים בפועל
בעמוד TradingView של המשתמש.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backtest_runner import DEFAULT_STRATEGY_CODE, StrategyCompileError, compute_vwap, run_backtest
from chart import (
    AUTO_RES_PREFIX,
    LOAD_MORE_BUTTON_LABEL,
    PLOTLY_CONFIG,
    RESOLUTION_STEPS,
    build_performance_chart,
    build_price_chart,
    build_price_chart_html,
)
from ib_data import MAX_CHUNK_DURATION, TIMEFRAMES, fetch_ohlcv
from ui_components import (
    TV_CSS,
    empty_state_html,
    header_chrome_html,
    key_stats_html,
    left_toolbar_html,
    performance_heading_html,
    performance_summary_html,
    pine_editor_tag_html,
    right_toolbar_html,
    strategy_properties_html,
    strategy_tester_top_html,
    time_axis_right_html,
    trades_table_html,
)

st.set_page_config(page_title="Strategy Tester", layout="wide", initial_sidebar_state="collapsed")
st.markdown(TV_CSS, unsafe_allow_html=True)
st.markdown(left_toolbar_html(), unsafe_allow_html=True)
st.markdown(right_toolbar_html(), unsafe_allow_html=True)

# תקרת הטווח שניתן לבקש. זו תקרת בטיחות בלבד — נקודת העצירה האמיתית היא
# history_exhausted, שנקבע לפי מה ש-IB באמת מחזיר. בעבר התקרה הייתה 1825 (5 שנים),
# מה שגרם לכך שבחירת "5 שנים" בפאנל הביצועים נעלה לגמרי את טעינת ההיסטוריה בגרף.
MAX_HISTORY_DAYS = 7300  # ~20 שנה

# הטיים-פריימים שהאפליקציה הזו תומכת בהם. ib_data מכיר גם 4h/1w (עבור דשבורד ה-Dash),
# אבל כאן יש טבלאות פנימיות (RESOLUTION_STEPS, _RESAMPLE_RULE) שבנויות על הסט הזה בלבד.
STREAMLIT_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d"]

for key, default in {
    "data": None,
    "stats": None,
    "stats_cash": None,
    "data_error": None,
    "run_error": None,
    "symbol_loaded": None,
    "timeframe_loaded": None,
    "days_requested": None,
    "extended_hours_loaded": None,
    "end_date_loaded": None,
    # True רק כאשר בקשת "טען עוד" חזרה בלי נתונים ישנים יותר — כלומר ל-IB אין יותר
    # היסטוריה בגודל הבר הזה. מתאפס בכל שינוי סימבול/טיים-פריים/שעות מסחר.
    "history_exhausted": False,
    "load_more_pending": False,
    # ההקשר האחרון שנוסתה עבורו שאיבה (סימבול/טיים-פריים/שעות/טווח). נרשם גם בהצלחה
    # וגם בכישלון, וזה מה שמונע לולאת ניסיונות אינסופית מול TWS סגור: בלי זה, כישלון
    # משאיר symbol_loaded ריק ולכן כל rerun היה מנסה לשאוב שוב מאפס.
    "last_attempt": None,
    # כמה היסטוריה לשאוב. לא ווידג'ט — נקבע מטווחי ציר-הזמן, מ"טען עוד" ומתקופת
    # הבדיקה של האסטרטגיה, בדיוק כמו ש-TradingView טוען לפי הטווח הנבחר.
    "days_input": 5,
    "tf_input": "1m",
    # טווח ציר-הזמן של הגרף (1D/5D/1M/...) — תצוגה + כמות הדאטה הנטענת
    "chart_range": "5D",
    "chart_range_applied": "5D",
    # תקופת הבדיקה של האסטרטגיה (נפרד לגמרי מהטיים-פריים ומטווח הגרף)
    "bt_range": "All",
    "bt_range_applied": None,
    "bt_window": None,
    # הגדרות ההרצה — יושבות בטאב Properties של ה-Strategy Tester, כמו ב-TradingView
    "prop_capital": 100_000.0,
    "prop_size": 100,
    "prop_commission": 0.0,
    "ext_hours": True,
    # התאמת רזולוציה אוטומטית בזום — הגרף עובר לנרות גסים/עדינים יותר לפי הטווח הנראה
    "auto_resolution": True,
    # דאטת התצוגה של הגרף, נפרדת לחלוטין מדאטת הבדיקה. כשהזום בוחר רזולוציה אחרת
    # משלך, היא נשאבת לכאן — ו-data (שעליו רצה האסטרטגיה) לא משתנה בכלל.
    "chart_tf": None,
    "chart_days": None,
    "chart_data": None,
    "chart_data_key": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# מאפשר לרכיבים שנמצאים נמוך יותר בעמוד (טווחי ציר-הזמן, תקופת הבדיקה, "טען עוד")
# לבקש ערך חדש לווידג'ט שכבר נוצר למעלה — Streamlit אוסר לשנות אותו ישירות אחרי היצירה.
if "_pending_days_override" in st.session_state:
    st.session_state["days_input"] = st.session_state.pop("_pending_days_override")
if "_pending_tf_override" in st.session_state:
    st.session_state["tf_input"] = st.session_state.pop("_pending_tf_override")

# הגדרות ההרצה נקראות כאן מה-session_state (הווידג'טים עצמם נוצרים בטאב Properties
# בהמשך העמוד) כדי שהשאיבה וההרצה האוטומטית שלמעלה יוכלו להשתמש בהן כבר עכשיו.
initial_cash = float(st.session_state["prop_capital"])
position_size = int(st.session_state["prop_size"])
commission = float(st.session_state["prop_commission"]) / 100.0
extended_hours = bool(st.session_state["ext_hours"])
auto_resolution = bool(st.session_state["auto_resolution"])
days = int(st.session_state["days_input"])

# כמה היסטוריה לטעון לכל רזולוציה כשההחלפה נעשית אוטומטית בזום. הערכים נותנים בערך
# 2,000–3,000 נרות לכל רזולוציה — מספיק הקשר מסביב לטווח שמסתכלים עליו, וקל לשליחה.
AUTO_TF_DAYS = {"1m": 5, "5m": 20, "15m": 45, "30m": 90, "1h": 180, "1d": 1825}

# תקרת נרות לציור. מעליה הגרף מצויר ברזולוציה גסה יותר (aggregation מקומי, בלי פנייה
# ל-IB) — 200 אלף נרות של דקה הם עשרות MB ל-HTML אחד ומקפיאים את הדפדפן. הבדיקה של
# האסטרטגיה עצמה תמיד רצה על הדאטה המלאה, כך שהתוצאות לא מושפעות מזה.
MAX_RENDER_BARS = 6000
_RESAMPLE_RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1d": "1D"}


def _display_frame(data, timeframe: str):
    """מחזיר (נרות לציור, הרזולוציה בפועל) — מקטין רזולוציה רק אם יש יותר מדי נרות."""
    if data is None or len(data) <= MAX_RENDER_BARS:
        return data, timeframe

    steps = [tf for tf, _ in RESOLUTION_STEPS]
    start = steps.index(timeframe) if timeframe in steps else 0
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out, out_tf = data, timeframe
    for tf in steps[start + 1:]:
        out = data.resample(_RESAMPLE_RULE[tf]).agg(agg).dropna()
        out_tf = tf
        if len(out) <= MAX_RENDER_BARS:
            break
    return out, out_tf


def _chart_dataset(symbol: str, ext_hours: bool):
    """מחזיר (נרות לגרף, הטיים-פריים שלהם) — דאטת תצוגה בלבד.

    כשהזום בחר רזולוציה שונה מזו שנשאבה לבדיקה, הנרות לגרף נשאבים לסט נפרד
    (chart_data) שנשמר באותו קאש מקומי. st.session_state.data — הדאטה שעליה רצה
    האסטרטגיה — לא נוגעים בו כאן בכלל, כך שזום בגרף לא יכול לשנות תוצאות בדיקה.
    """
    base, base_tf = st.session_state.data, st.session_state.timeframe_loaded
    ctf = st.session_state.get("chart_tf")
    if base is None or not ctf or ctf == base_tf:
        return base, base_tf

    cdays = int(st.session_state.get("chart_days") or AUTO_TF_DAYS.get(ctf, 30))
    key = (symbol, ctf, cdays, ext_hours)
    if st.session_state.get("chart_data_key") != key or st.session_state.get("chart_data") is None:
        try:
            st.session_state.chart_data = fetch_ohlcv(
                symbol, cdays, ctf, use_rth=not ext_hours
            )
            st.session_state.chart_data_key = key
        except Exception:
            # תצוגה בלבד — כישלון כאן לא אמור להפיל את העמוד או את הבדיקה.
            st.session_state.chart_tf = None
            return base, base_tf
    return st.session_state.chart_data, ctf


def _reset_chart_view() -> None:
    """מבטל את רזולוציית התצוגה שנבחרה בזום — הגרף חוזר לעקוב אחרי הטיים-פריים שבחרת."""
    st.session_state.chart_tf = None
    st.session_state.chart_days = None
    st.session_state.chart_data = None
    st.session_state.chart_data_key = None


def _extend_days_for_load_more() -> None:
    """מגדיל בהדרגה את כמות ההיסטוריה הנטענת — נקרא מכפתור 'טען היסטוריה נוספת' שמתחת
    לגרף, שגם נלחץ אוטומטית מה-JS של הגרף כשגוררים/מזמזמים אחורה לקצה הדאטה הטעונה
    (ראה chart.py). מסמן load_more_pending כדי שנוכל לזהות אחרי השאיבה אם באמת התקבלה
    היסטוריה ישנה יותר, או שהגענו לסוף מה שיש ב-IB."""
    if st.session_state.get("chart_tf"):
        # הגרף מציג רזולוציה משלו — מרחיבים את דאטת התצוגה בלבד.
        base = int(st.session_state.get("chart_days") or 30)
        st.session_state["chart_days"] = min(MAX_HISTORY_DAYS, base + max(30, int(base * 0.5)))
        st.session_state["chart_data_key"] = None
        return
    base_days = st.session_state.get("days_requested") or st.session_state.get("days_input") or 5
    extra = max(30, int(base_days * 0.5))
    st.session_state["_pending_days_override"] = min(MAX_HISTORY_DAYS, int(base_days) + extra)
    st.session_state["load_more_pending"] = True


def _ytd_days() -> int:
    today = dt.date.today()
    return max(1, (today - dt.date(today.year, 1, 1)).days)


# טווחי ציר-הזמן של הגרף — אותם טאבים שיש ב-TradingView (date-range-tab-*), וכל טווח
# גורר גם את רזולוציית הנרות המתאימה לו, בדיוק כמו שם ("1 month in 30 minutes intervals").
# מותאם לטיים-פריימים שיש ל-IB אצלנו (אין 2h/1w/1M — נופלים ליומי/שעתי).
CHART_RANGES: dict[str, tuple[int | None, str]] = {
    "1D": (1, "1m"),
    "5D": (5, "5m"),
    "1M": (30, "30m"),
    "3M": (90, "1h"),
    "6M": (180, "1h"),
    "YTD": (None, "1d"),
    "12M": (365, "1d"),
    "60M": (1825, "1d"),
    "ALL": (MAX_HISTORY_DAYS, "1d"),
}
CHART_RANGE_HELP = (
    "טווח התצוגה של הגרף — קובע כמה היסטוריה נטענת ובאיזו רזולוציית נרות, "
    "בדיוק כמו טאבי הטווח ב-TradingView. זה **לא** תקופת הבדיקה של האסטרטגיה."
)

# תקופות הבדיקה של האסטרטגיה. חשוב: זה **לא** הטיים-פריים של הגרף ולא טווח התצוגה —
# אלה דברים נפרדים. הטיים-פריים קובע את רזולוציית הנרות שבה מנתחים ומריצים, ותקופת
# הבדיקה קובעת על איזה חלון היסטוריה נמדדת האסטרטגיה. אם התקופה ארוכה מהדאטה הטעונה,
# האפליקציה תשאב את מה שחסר ותריץ עליו.
BT_RANGES = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "All": None}


def _backtest_slice(data):
    """מחזיר את חלון ההיסטוריה שעליו תיבדק האסטרטגיה, לפי התקופה שנבחרה."""
    days_ = BT_RANGES.get(st.session_state.get("bt_range", "All"))
    if data is None or data.empty or days_ is None:
        return data
    cutoff = data.index.max() - pd.Timedelta(days=days_)
    sliced = data[data.index >= cutoff]
    # אם החיתוך משאיר פחות מדי נרות להרצה, נשארים עם כל מה שיש במקום להיכשל.
    return sliced if len(sliced) >= 2 else data


def _run_backtest_now(code: str, cash: float, size: int) -> None:
    """מריץ את האסטרטגיה על חלון הבדיקה הנבחר בלבד, ושומר גם את גבולות החלון בפועל."""
    bt_data = _backtest_slice(st.session_state.data)
    st.session_state.bt_range_applied = st.session_state.get("bt_range", "All")

    if bt_data is None or len(bt_data) < 2:
        st.session_state.stats = None
        return
    try:
        st.session_state.stats = run_backtest(
            bt_data, code, cash=cash, position_size=int(size), commission=commission
        )
        st.session_state.stats_cash = cash
        st.session_state.bt_window = (bt_data.index.min(), bt_data.index.max())
        st.session_state.run_error = None
    except StrategyCompileError as exc:
        st.session_state.run_error = str(exc)
        st.session_state.stats = None


# ---------------------------------------------------------------- סרגל כלים עליון
tv_header = st.container(key="tv_header")
with tv_header:
    hdr_symbol, hdr_tf, hdr_fetch, hdr_chrome = st.columns([1.25, 0.8, 0.35, 9.6])
with hdr_symbol:
    symbol = st.text_input(
        "סימבול", value="AAPL", label_visibility="collapsed", placeholder="Symbol", help="Change symbol"
    ).strip().upper()
with hdr_tf:
    timeframe = st.selectbox(
        "Chart interval", options=STREAMLIT_TIMEFRAMES, key="tf_input",
        label_visibility="collapsed", help="Chart interval",
    )
with hdr_fetch:
    fetch_clicked = st.button(
        "⟳", use_container_width=True,
        help="רענן מ-TWS. השאיבה קורית לבד בכל שינוי סימבול/טיים-פריים — הכפתור נועד "
        "לרענון ידני ולניסיון חוזר אחרי תקלת חיבור.",
    )
with hdr_chrome:
    st.markdown(header_chrome_html(), unsafe_allow_html=True)

# מקומות בעמוד שימולאו אחרי שהאסטרטגיה תרוץ (הגרף צריך את תוצאות הבדיקה, אבל צריך
# להופיע מעל עורך הקוד — כמו ב-TradingView).
chart_area = st.container(key="tv_chart")
taxis_area = st.container(key="tv_taxis")
msg_area = st.container()

# ---------------------------------------------------------------- טווח מותאם אישית
with msg_area:
    with st.expander("🗓 Go to — טווח תאריכים מותאם אישית", expanded=False):
        use_custom_range = st.checkbox(
            "השתמש בטווח תאריכים מותאם אישית (במקום טאבי הטווח שמתחת לגרף)",
            value=False,
            key="use_custom_range",
        )
        # גבולות לוח השנה נגזרים מאותה תקרה כמו טווח הטעינה, וברירת המחדל נצמדת אליהם.
        _default_end = dt.date.today()
        _min_date = _default_end - dt.timedelta(days=MAX_HISTORY_DAYS)
        _default_start = max(_min_date, _default_end - dt.timedelta(days=int(days)))
        date_range = st.date_input(
            "בחר תאריך התחלה וסיום",
            value=(_default_start, _default_end),
            max_value=_default_end,
            min_value=_min_date,
            label_visibility="collapsed",
            disabled=not use_custom_range,
            help="בחר תאריך התחלה ותאריך סיום כדי לצפות בכל חלון היסטורי — למשל תקופה "
            "מלפני שנתיים — במקום רק ב-N הימים האחרונים.",
        )
        custom_range_active = use_custom_range and isinstance(date_range, tuple) and len(date_range) == 2
        if custom_range_active:
            range_start, range_end = date_range
            if range_start > range_end:
                st.warning("תאריך ההתחלה מאוחר מתאריך הסיום — בחר טווח תקין.")
                custom_range_active = False

effective_days = days
effective_end_date = None
if custom_range_active:
    effective_days = max(1, (range_end - range_start).days) + 1
    effective_end_date = range_end

# ---------------------------------------------------------------- שאיבת נתונים
# כל שינוי בהקשר שואב אוטומטית ומריץ מחדש את האסטרטגיה על הנתונים החדשים: הזנת
# סימבול ולחיצת Enter, בחירת טיים-פריים אחר, שינוי שעות המסחר או טווח התאריכים —
# וגם הטעינה הראשונה של העמוד. כפתור ⟳ נשאר לרענון/ניסיון חוזר ידני.
fetch_context = (symbol, timeframe, extended_hours, effective_days, effective_end_date)
auto_refetch = bool(symbol) and fetch_context != st.session_state.get("last_attempt")

if fetch_clicked or auto_refetch:
    # נרשם לפני הניסיון (ולא רק אחריו) כדי שגם שאיבה שנכשלה לא תנוסה שוב בכל rerun.
    st.session_state.last_attempt = fetch_context
    # שינוי הקשר (סימבול/טיים-פריים/שעות מסחר) מתחיל היסטוריה מאפס — ולכן מאפס גם
    # את סימון "אין יותר היסטוריה", אחרת נעילה מטיים-פריים אחד הייתה נגררת לאחר.
    context_changed = (
        st.session_state.symbol_loaded != symbol
        or st.session_state.timeframe_loaded != timeframe
        or st.session_state.get("extended_hours_loaded") != extended_hours
    )
    # סימבול או טיים-פריים חדשים שנבחרו במפורש — הגרף חוזר לעקוב אחריהם במקום
    # להישאר ברזולוציית התצוגה שנבחרה קודם בזום.
    if context_changed:
        _reset_chart_view()

    prev_earliest = None
    if st.session_state.data is not None and not st.session_state.data.empty:
        prev_earliest = st.session_state.data.index.min()
    was_load_more = bool(st.session_state.get("load_more_pending"))

    _range_desc = (
        f"{range_start:%b %d, %Y} — {range_end:%b %d, %Y}"
        if custom_range_active
        else f"עד {effective_days} ימים אחורה"
    )
    # IB מגביל כל בקשה לפי גודל הבר — שבוע אחד בלבד לנרות דקה, מול חודש ב-5m/15m/30m/1h
    # ושנה ביומי. לכן טווח ארוך בנרות דקה מתורגם להרבה קריאות רצופות, וזה הזמן שנצפה
    # בממשק. מציגים ספירת קריאות כדי שיהיה ברור שהתהליך מתקדם ולא תקוע.
    _progress_box = st.empty()

    def _report(chunk_no: int, paused: bool = False) -> None:
        if paused:
            _progress_box.caption("⏸ הפוגה קצרה כדי לא לחרוג ממגבלת הקצב של IB...")
            return
        _progress_box.caption(
            f"⏳ שואב {symbol} · {timeframe} — קריאה {chunk_no} ל-IB "
            f"(כל קריאה מביאה {MAX_CHUNK_DURATION.get(timeframe, '?')} של נרות)"
        )

    with st.spinner(f"שואב נתוני {timeframe} עבור {symbol} מ-TWS ({_range_desc})..."):
        try:
            st.session_state.data = fetch_ohlcv(
                symbol, effective_days, timeframe, use_rth=not extended_hours,
                end_date=effective_end_date,
                force_refresh=bool(fetch_clicked),
                progress=_report,
            )
            st.session_state.symbol_loaded = symbol
            st.session_state.timeframe_loaded = timeframe
            st.session_state.days_requested = effective_days
            st.session_state.extended_hours_loaded = extended_hours
            st.session_state.end_date_loaded = effective_end_date
            st.session_state.data_error = None

            # "מיצוי היסטוריה" נקבע רק מהאמת של השרת: בקשת "טען עוד" שחזרה בלי אף נר
            # ישן יותר. חשוב שזה ייקבע כאן ולא בצד הדפדפן — קודם זה נשמר ב-sessionStorage
            # ונשאר תקוע לכל החיים, כך שגם תקלת תקשורת חולפת מול TWS נעלה את הגרף לצמיתות.
            if context_changed:
                st.session_state.history_exhausted = False
            elif was_load_more and prev_earliest is not None and not st.session_state.data.empty:
                st.session_state.history_exhausted = bool(st.session_state.data.index.min() >= prev_earliest)

            # מריצים מיד את האסטרטגיה על הנתונים שנשאבו, כך שכל החישובים (עסקאות,
            # רווח/הפסד, drawdown, Profit factor) מתעדכנים לבד. ה-fallback לקוד ברירת
            # המחדל נחוץ לריצה הראשונה של העמוד: ווידג'ט העורך נוצר נמוך יותר בסקריפט,
            # ולכן st.session_state["code"] עוד לא קיים בשלב הזה.
            _run_backtest_now(
                st.session_state.get("code") or DEFAULT_STRATEGY_CODE, initial_cash, position_size
            )
        except Exception as exc:
            st.session_state.data_error = str(exc)
            st.session_state.data = None
            st.session_state.stats = None
            # רושמים את הטווח שנוסה גם בכישלון: אחרת בחירת טווח הייתה רואה שהערך לא
            # השתנה, מבקשת שוב, ונכנסת ללולאת rerun אינסופית מול TWS תקול.
            st.session_state.days_requested = effective_days
            st.session_state.end_date_loaded = effective_end_date
        finally:
            st.session_state.load_more_pending = False
            _progress_box.empty()

# עצירת "טען עוד היסטוריה" רק משתי סיבות אמיתיות: הגענו לתקרת הבטיחות של האפליקציה,
# או ש-IB באמת הפסיק להחזיר נתונים ישנים יותר (history_exhausted, נקבע אחרי שאיבה).
at_history_limit = bool(
    (st.session_state.get("days_requested") or 0) >= MAX_HISTORY_DAYS
    or st.session_state.get("history_exhausted")
)

# ---------------------------------------------------------------- Pine Editor
with st.expander("Pine Editor", expanded=False):
    st.markdown(pine_editor_tag_html(), unsafe_allow_html=True)
    st.caption("כתוב רק את def init(self) ו-def next(self) — האפליקציה עוטפת אוטומטית ל-Strategy עם VWAP מובנה.")
    code = st.text_area(
        "strategy_code",
        value=st.session_state.get("code", DEFAULT_STRATEGY_CODE),
        height=320,
        key="code",
        label_visibility="collapsed",
    )
    run_clicked = st.button("▶ Add to chart", type="primary", use_container_width=True)

if run_clicked:
    if st.session_state.data is None:
        st.warning("קודם שאב נתונים מ-TWS (כפתור ⟳ בסרגל העליון).")
    else:
        with st.spinner("מריץ בדיקה..."):
            _run_backtest_now(code, initial_cash, position_size)

# ---------------------------------------------------------------- הגרף
with chart_area:
    if st.session_state.data is not None:
        stats = st.session_state.stats
        trades = stats["_trades"] if stats is not None else None
        vwap = stats["_strategy"].vwap if stats is not None else compute_vwap(st.session_state.data)

        enable_load_more = not custom_range_active and not at_history_limit
        if enable_load_more:
            # כפתור סמוי (מוסתר ב-CSS) שה-JS של הגרף "לוחץ" עליו אוטומטית כשגוררים אחורה
            # לקצה הדאטה הטעונה — ראה LOAD_MORE_BUTTON_LABEL ב-chart.py.
            st.markdown('<span class="tv-load-more-marker"></span>', unsafe_allow_html=True)
            if st.button(LOAD_MORE_BUTTON_LABEL, key="load_more_btn"):
                _extend_days_for_load_more()
                st.rerun()

        # אותו רעיון עבור החלפת רזולוציה: ה-JS של הגרף מודד את הטווח הנראה בזום ולוחץ
        # על הכפתור הסמוי של הרזולוציה המתאימה, שטוען מחדש בטיים-פריים הזה.
        if auto_resolution and not custom_range_active:
            with st.container(key="tv_autores"):
                for _tf, _ in RESOLUTION_STEPS:
                    if _tf in TIMEFRAMES and st.button(f"{AUTO_RES_PREFIX}{_tf}", key=f"autores_{_tf}"):
                        # חשוב: לא נוגעים ב-tf_input/days_input. זום הוא עניין של תצוגה
                        # בלבד, ואסור לו להחליף את הנתונים שהאסטרטגיה נבדקת עליהם.
                        st.session_state["chart_tf"] = _tf
                        st.session_state["chart_days"] = AUTO_TF_DAYS[_tf]
                        st.rerun()

        chart_src, chart_src_tf = _chart_dataset(symbol, extended_hours)
        plot_data, plot_tf = _display_frame(chart_src, chart_src_tf)
        # אם הוקטנה הרזולוציה לצורך הציור, ה-VWAP שחושב על הדאטה המלאה כבר לא באורך
        # הנכון — הוא מחושב מחדש על הנרות המצוירים (לתצוגה בלבד; הבדיקה לא מושפעת).
        plot_vwap = vwap if plot_data is st.session_state.data else compute_vwap(plot_data)

        fig = build_price_chart(
            plot_data,
            plot_vwap,
            trades,
            symbol=st.session_state.symbol_loaded or "",
            timeframe=plot_tf,
            extended_hours=bool(st.session_state.extended_hours_loaded),
        )
        components.html(
            build_price_chart_html(
                fig,
                symbol=st.session_state.symbol_loaded or "",
                timeframe=plot_tf,
                enable_load_more=enable_load_more,
                auto_resolution=auto_resolution and not custom_range_active,
                extended_hours=bool(st.session_state.extended_hours_loaded),
                # שמירת החלון הנצפה תקפה כל עוד מדובר באותה תצוגה. סימבול אחר או בחירה
                # מפורשת בטאבי הטווח מתחילים תצוגה חדשה ולכן מאפסים אותה; החלפת רזולוציה
                # אוטומטית בזום לא נכללת כאן במכוון — שם דווקא רוצים לשמר את החלון.
                view_token=f"{st.session_state.symbol_loaded}|{st.session_state.chart_range}",
            ),
            height=636,
            scrolling=False,
        )
        if plot_tf != st.session_state.timeframe_loaded:
            st.caption(
                f"🔍 הגרף מוצג כרגע ב-{plot_tf} (התאמת רזולוציה לזום). "
                f"בדיקת האסטרטגיה רצה בנפרד על נרות {st.session_state.timeframe_loaded} — "
                f"שינוי הזום לא משפיע על התוצאות."
            )
    else:
        st.info(
            "אין נתונים להצגה. הזן סימבול בסרגל העליון ולחץ Enter — הנתונים יישאבו "
            "והאסטרטגיה תרוץ עליהם אוטומטית. אם השאיבה נכשלה, ודא ש-TWS פתוח ולחץ ⟳."
        )

# ---------------------------------------------------------------- סרגל ציר-הזמן
with taxis_area:
    c_ranges, c_auto, c_right = st.columns([2.4, 1.1, 2.5])
    with c_ranges:
        st.segmented_control(
            "Date range",
            options=list(CHART_RANGES),
            key="chart_range",
            label_visibility="collapsed",
            help=CHART_RANGE_HELP,
        )
    with c_auto:
        st.session_state["auto_resolution"] = st.checkbox(
            "Auto",
            value=bool(st.session_state["auto_resolution"]),
            key="auto_res_w",
            help="התאמת רזולוציה אוטומטית: בזום החוצה הגרף עובר לנרות גסים יותר "
            "ובזום פנימה לעדינים יותר, כמו ב-TradingView. כבה כדי לנעול את "
            "הטיים-פריים שבחרת בסרגל העליון.",
        )
    with c_right:
        st.markdown(
            time_axis_right_html(session_label="Extended" if extended_hours else "Regular"),
            unsafe_allow_html=True,
        )

# בחירת טווח בציר-הזמן קובעת גם כמה היסטוריה לטעון וגם את רזולוציית הנרות — בדיוק
# כמו ב-TradingView, שבו לכל טאב טווח יש interval משלו.
if st.session_state.chart_range != st.session_state.chart_range_applied:
    _days, _tf = CHART_RANGES[st.session_state.chart_range]
    st.session_state.chart_range_applied = st.session_state.chart_range
    st.session_state["_pending_days_override"] = _days if _days is not None else _ytd_days()
    st.session_state["_pending_tf_override"] = _tf
    _reset_chart_view()
    st.rerun()

# ---------------------------------------------------------------- הודעות מצב
with msg_area:
    if st.session_state.data_error:
        st.error(st.session_state.data_error)
    if st.session_state.run_error:
        st.error(st.session_state.run_error)

    if st.session_state.data is not None and st.session_state.get("days_requested"):
        actual_days = (st.session_state.data.index.max() - st.session_state.data.index.min()).days
        if st.session_state.get("history_exhausted"):
            st.caption(
                f"⛔ הגעת לתחילת ההיסטוריה שיש ל-IB עבור {st.session_state.symbol_loaded} "
                f"בטיים-פריים {st.session_state.timeframe_loaded} (~{actual_days} ימים). "
                f"טיים-פריים גדול יותר (למשל 1d) בדרך כלל מגיע הרבה יותר אחורה."
            )
        elif actual_days < st.session_state.days_requested * 0.9:
            st.caption(
                f"ℹ️ ביקשת {st.session_state.days_requested} ימים אחורה, אך IB החזיר כיסוי של "
                f"~{actual_days} ימים בלבד עבור {timeframe} — כנראה שזה כל מה שקיים בהיסטוריה "
                f"שלו בגודל בר הזה (בטיים-פריים עדין כמו 1m זה נפוץ)."
            )

# ---------------------------------------------------------------- Strategy Tester
if st.session_state.stats is not None and st.session_state.data is not None:
    stats = st.session_state.stats
    trades: pd.DataFrame = stats["_trades"]

    # ההון שבו הבדיקה באמת רצה — לא הערך הנוכחי בטאב Properties. אחרת שינוי ההון
    # אחרי הרצה היה מחשב PnL/Drawdown מול בסיס שונה מזה שהתוצאות נוצרו איתו.
    run_cash = st.session_state.get("stats_cash") or initial_cash

    # ספירת הזכיות ו-Profit Factor מחושבים כאן מתוך טבלת העסקאות עצמה, באותה נוסחה
    # בדיוק שבה משתמש טאב "Performance Summary" (רווח ברוטו / הפסד ברוטו — ההגדרה של
    # TradingView), כדי שהכותרת והטבלה לא יראו מספרים שונים על אותן עסקאות.
    n_trades = len(trades)
    total_pnl = float(stats["Equity Final [$]"]) - run_cash
    max_dd_pct = abs(float(stats["Max. Drawdown [%]"]))
    max_dd_usd = max_dd_pct / 100 * run_cash

    if n_trades:
        _pnl = trades["PnL"]
        wins = int((_pnl > 0).sum())
        win_rate = wins / n_trades * 100
        _gross_profit = float(_pnl[_pnl > 0].sum())
        _gross_loss = float(abs(_pnl[_pnl <= 0].sum()))
        profit_factor = (_gross_profit / _gross_loss) if _gross_loss > 0 else None
    else:
        wins, win_rate, profit_factor = 0, 0.0, None

    # הטווח שמוצג בסרגל הוא חלון הבדיקה בפועל שעליו רצה האסטרטגיה — לא כל הדאטה שנשאבה.
    _w = st.session_state.get("bt_window") or (
        st.session_state.data.index.min(),
        st.session_state.data.index.max(),
    )
    _range_txt = f"{pd.Timestamp(_w[0]):%b %-d, %Y} — {pd.Timestamp(_w[1]):%b %-d, %Y}"

    st.markdown(
        strategy_tester_top_html(
            strategy_name=f"{st.session_state.symbol_loaded} · {st.session_state.timeframe_loaded}",
            date_range_text=_range_txt,
            capital=run_cash,
        ),
        unsafe_allow_html=True,
    )

    # בורר תקופת הבדיקה — יושב בתוך מסגרת הפאנל, בין סרגל הכלים ל-Key stats.
    with st.container(key="tv_btrange"):
        _loaded_days = (st.session_state.data.index.max() - st.session_state.data.index.min()).days
        st.markdown(
            '<div class="tv-btrange-label">תקופת הבדיקה של האסטרטגיה — בחירת תקופה '
            'ארוכה יותר תשאב את הנתונים החסרים ותריץ עליהם '
            f'<b>·</b> כרגע {_loaded_days} ימים של נרות {st.session_state.timeframe_loaded}</div>',
            unsafe_allow_html=True,
        )
        st.segmented_control(
            "תקופת הבדיקה",
            options=list(BT_RANGES),
            key="bt_range",
            label_visibility="collapsed",
        )

    # שינוי תקופת הבדיקה:
    #   1. אם התקופה המבוקשת ארוכה ממה שכבר נטען — מבקשים שאיבה שתכסה אותה. השאיבה
    #      עצמה קורית בראש הסקריפט בריצה הבאה, ומיד אחריה רצה הבדיקה על החלון החדש.
    #   2. אם כבר יש מספיק דאטה — רק חותכים ומריצים מחדש, בלי לגעת ב-IB.
    # bt_range_applied מתעדכן רק אחרי הרצה בפועל, וזה מה שסוגר את הלולאה: גם אם IB
    # מחזיר פחות ימים מהמבוקש, days_requested נרשם בכל מקרה ולכן לא נשאב שוב ושוב.
    if st.session_state.get("bt_range") != st.session_state.get("bt_range_applied"):
        _needed = BT_RANGES.get(st.session_state["bt_range"])
        _have = st.session_state.get("days_requested") or 0
        if _needed is not None and _needed > _have:
            st.session_state["_pending_days_override"] = min(MAX_HISTORY_DAYS, _needed)
            st.rerun()
        elif st.session_state.get("code"):
            _run_backtest_now(st.session_state["code"], initial_cash, position_size)
            st.rerun()

    st.markdown(
        key_stats_html(
            net_profit=total_pnl,
            net_profit_pct=float(stats["Return [%]"]),
            max_dd=max_dd_usd,
            max_dd_pct=max_dd_pct,
            total_trades=n_trades,
            profit_factor=profit_factor,
            percent_profitable=win_rate,
            winning_trades=wins,
        ),
        unsafe_allow_html=True,
    )

    tab_overview, tab_summary, tab_trades, tab_props = st.tabs(
        ["Overview", "Performance Summary", "List of Trades", "Properties"]
    )

    with tab_overview:
        st.markdown(performance_heading_html(), unsafe_allow_html=True)
        if not trades.empty:
            st.plotly_chart(
                build_performance_chart(trades, start_time=_w[0]),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.markdown(
                empty_state_html("לא בוצעו עסקאות בתקופת הבדיקה שנבחרה."),
                unsafe_allow_html=True,
            )

    with tab_summary:
        if trades.empty:
            st.markdown(empty_state_html("אין עסקאות לסיכום."), unsafe_allow_html=True)
        else:
            st.markdown(performance_summary_html(stats, trades, run_cash), unsafe_allow_html=True)

    with tab_trades:
        if trades.empty:
            st.markdown(empty_state_html("לא בוצעו עסקאות."), unsafe_allow_html=True)
        else:
            st.markdown(trades_table_html(trades), unsafe_allow_html=True)

    with tab_props:
        st.markdown('<div class="tv-props-head">Strategy properties</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tv-props-hint">שינוי כאן דורש הרצה מחדש (▶ Add to chart) '
            "כדי שהתוצאות למעלה יחושבו מחדש.</div>",
            unsafe_allow_html=True,
        )
        # הווידג'טים כאן משתמשים במפתח נפרד (סיומת _w) והערך מועתק חזרה למפתח ה"רגיל".
        # אי אפשר לתת לווידג'ט להיות מקור האמת: הוא נוצר בתחתית העמוד, בעוד שהשאיבה
        # וההרצה שלמעלה צריכות את הערך כבר בתחילת אותה ריצה.
        p1, p2, p3 = st.columns(3)
        with p1:
            st.session_state["prop_capital"] = st.number_input(
                "Initial capital", min_value=1000.0, step=1000.0,
                value=float(st.session_state["prop_capital"]), key="prop_capital_w",
            )
        with p2:
            st.session_state["prop_size"] = st.number_input(
                "Order size (shares)", min_value=1, step=1,
                value=int(st.session_state["prop_size"]), key="prop_size_w",
            )
        with p3:
            st.session_state["prop_commission"] = st.number_input(
                "Commission (%)", min_value=0.0, max_value=5.0, step=0.01, format="%.3f",
                value=float(st.session_state["prop_commission"]), key="prop_commission_w",
                help="עמלה כאחוז מכל עסקה, מוחלת על כניסה ועל יציאה — כמו Commission ב-TradingView.",
            )
        st.session_state["ext_hours"] = st.checkbox(
            "Extended trading hours (Pre/Post-market)",
            value=bool(st.session_state["ext_hours"]), key="ext_hours_w",
            help="כשמסומן: שואב מ-IB גם נתוני טרום-שוק (~4:00) ואחרי-שוק (~20:00 ET), לא רק 9:30–16:00.",
        )
        st.markdown(strategy_properties_html(), unsafe_allow_html=True)

elif st.session_state.data is not None:
    st.markdown(
        empty_state_html("פתח את Pine Editor ולחץ '▶ Add to chart' כדי לראות את תוצאות האסטרטגיה."),
        unsafe_allow_html=True,
    )
