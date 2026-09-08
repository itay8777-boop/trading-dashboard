"""בניית גרפים בסגנון TradingView (Plotly): נרות + VWAP + סימוני עסקאות, וגרף PnL מצטבר."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# צבעים מקונפיג הגרף האמיתי של המשתמש ב-TradingView (chartproperties):
# רקע שחור, גריד עדין (unitedGridProperties), טקסט צירים #B2B5BE,
# נרות ירוק/אדום עם פתילים אפורים, וווליום בצבעי volumePalette.
TV_BG = "#000000"          # backgroundGradientStart/End
TV_GRID = "rgba(240,243,250,0.06)"
TV_TEXT = "#B2B5BE"        # scalesProperties.textColor
TV_SEP = "#2A2E39"         # paneProperties.separatorColor

TV_CANDLE_UP = "#4CAF50"   # candleStyle.upColor  rgba(76,175,80,1)
TV_CANDLE_DOWN = "#F23645" # candleStyle.downColor
TV_WICK = "#636363"        # candleStyle.wickUp/DownColor  rgba(99,99,99,1)

TV_VOL_UP = "#26A69A"      # volumePalette growing
TV_VOL_DOWN = "#EF5350"    # volumePalette falling

TV_GREEN = "#089981"       # ירוק/אדום של Strategy Tester — לסימוני עסקאות
TV_RED = "#F23645"
TV_BLUE = "#2962FF"
TV_VWAP = "#FFB300"        # קו ה-VWAP


def _tv_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_BG,
        font=dict(color=TV_TEXT, size=12),
        margin=dict(l=10, r=54, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=TV_GRID, zeroline=False, showline=False),
        yaxis=dict(gridcolor=TV_GRID, zeroline=False, showline=False, side="right", automargin=True),
        hoverlabel=dict(bgcolor="#1e222d", font_color=TV_TEXT, bordercolor=TV_GRID),
        dragmode="pan",
    )
    return fig


# config מומלץ ל-st.plotly_chart: גרירה = pan (כמו TradingView), גלגלת עכבר = זום, בלי לוגו פלוטלי.
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

# הטקסט המדויק של כפתור "טען היסטוריה נוספת" ב-app.py — ה-JS בגרף מאתר אותו לפי הטקסט
# הזה בדף ההורה (components.html רץ ב-iframe מבודד ואסור עליו לנווט את העמוד הראשי,
# לכן במקום זה "לוחצים" תכנותית על כפתור Streamlit אמיתי כדי להפעיל rerun רגיל).
LOAD_MORE_BUTTON_LABEL = "⬅ טען היסטוריה נוספת"

# תחילית הכפתורים הסמויים שדרכם ה-JS של הגרף מבקש מעבר לרזולוציה אחרת (ראה app.py).
AUTO_RES_PREFIX = "⟲res:"

# סדר הרזולוציות מהעדינה לגסה, עם אורך הבר בדקות. משמש את בחירת הרזולוציה האוטומטית
# בצד הדפדפן: מתוך טווח הזמן הנראה מחושב כמה נרות ייצאו בכל רזולוציה, ונבחרת העדינה
# ביותר שעדיין לא צפופה מדי — בדיוק ההתנהגות של TradingView בזום פנימה/החוצה.
RESOLUTION_STEPS = [("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("1d", 1440)]

# יעד הנרות הנראים בו-זמנית, והגבולות שמעבר להם מוחלפת הרזולוציה. הפער הרחב בין
# הגבולות (היסטרזיס) מונע "ריצוד" של החלפות הלוך-ושוב סביב סף אחד.
TARGET_VISIBLE_BARS = 600
MAX_VISIBLE_BARS = 1400
MIN_VISIBLE_BARS = 100

# plotly.min.js מוגש מ-static/ (ראה .streamlit/config.toml) כדי שהדפדפן יאחסן אותו
# ב-cache. קודם הוא הוטמע בתוך ה-HTML של הגרף, כלומר 4.3MB נשלחו מחדש בכל שינוי
# זום/טווח/סימבול — זו הייתה הסיבה העיקרית לאיטיות בטעינת הגרף.
PLOTLY_JS_URL = "/app/static/plotly.min.js"
_PLOTLY_JS_FILE = Path(__file__).resolve().parent / "static" / "plotly.min.js"


def build_price_chart(
    data: pd.DataFrame,
    vwap,
    trades: pd.DataFrame | None = None,
    symbol: str = "",
    timeframe: str = "",
    extended_hours: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color=TV_CANDLE_UP,
            decreasing_line_color=TV_CANDLE_DOWN,
            increasing_fillcolor=TV_CANDLE_UP,
            decreasing_fillcolor=TV_CANDLE_DOWN,
            # פתילים אפורים בשני הכיוונים, כמו בהגדרות שלך
            increasing=dict(line=dict(color=TV_CANDLE_UP), fillcolor=TV_CANDLE_UP),
            decreasing=dict(line=dict(color=TV_CANDLE_DOWN), fillcolor=TV_CANDLE_DOWN),
            line=dict(width=1),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=vwap,
            mode="lines",
            name="VWAP",
            line=dict(color=TV_VWAP, width=1.3),
        ),
        row=1, col=1,
    )

    if "Volume" in data.columns:
        vol_colors = [TV_VOL_UP if c >= o else TV_VOL_DOWN for o, c in zip(data["Open"], data["Close"])]
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=vol_colors, marker_line_width=0),
            row=2, col=1,
        )

    if trades is not None and not trades.empty:
        # כל עסקה מסומנת בצבע אחיד (ירוק=רווח, אדום=הפסד) עם משולש-כניסה ומשולש-יציאה
        # מחוברים בקו מקווקו דק — כך רואים בבירור אם העסקה הרוויחה בלי לתפוס שטח על הגרף
        # (בלי מלבן מוצלל גדול), ופרטים מדויקים (זמן/מחיר/PnL) מופיעים ב-hover.
        winners = trades[trades["PnL"] >= 0]
        losers = trades[trades["PnL"] < 0]

        for subset, color, name in ((winners, TV_GREEN, "Winning trades"), (losers, TV_RED, "Losing trades")):
            if subset.empty:
                continue
            marker_x, marker_y, marker_symbol, hover_text = [], [], [], []
            line_x, line_y = [], []
            for _, t in subset.iterrows():
                direction = "Long" if t["Size"] > 0 else "Short"
                pnl_sign = "+" if t["PnL"] >= 0 else ""
                ret_sign = "+" if t["ReturnPct"] >= 0 else ""
                entry_hover = (
                    f"<b>{direction} entry</b><br>{pd.Timestamp(t['EntryTime']):%b %d, %H:%M}<br>"
                    f"${t['EntryPrice']:.2f}"
                )
                exit_hover = (
                    f"<b>{direction} exit</b><br>{pd.Timestamp(t['ExitTime']):%b %d, %H:%M}<br>"
                    f"${t['ExitPrice']:.2f}<br>"
                    f"PnL: {pnl_sign}${t['PnL']:,.2f} ({ret_sign}{t['ReturnPct'] * 100:.2f}%)"
                )
                marker_x += [t["EntryTime"], t["ExitTime"]]
                marker_y += [t["EntryPrice"], t["ExitPrice"]]
                marker_symbol += ["triangle-up", "triangle-down"]
                hover_text += [entry_hover, exit_hover]
                line_x += [t["EntryTime"], t["ExitTime"], None]
                line_y += [t["EntryPrice"], t["ExitPrice"], None]

            fig.add_trace(
                go.Scatter(
                    x=line_x, y=line_y, mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=marker_x, y=marker_y, mode="markers", name=name,
                    marker=dict(symbol=marker_symbol, size=9, color=color, line=dict(width=1, color="#131722")),
                    text=hover_text, hovertemplate="%{text}<extra></extra>",
                ),
                row=1, col=1,
            )

    fig = _tv_layout(fig, height=620)
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    fig.update_xaxes(gridcolor=TV_GRID, zeroline=False, showline=False, row=2, col=1)
    fig.update_yaxes(gridcolor=TV_GRID, zeroline=False, showline=False, side="right", automargin=True, row=2, col=1)

    # מסתיר את הפערים בציר הזמן (סופ"ש, ולילה מחוץ לשעות המסחר) כדי שהנרות ייראו רציפים
    # כמו ב-TradingView, במקום להשאיר "חורים" ריקים בגרף.
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if timeframe != "1d":
        hour_bounds = [20, 4] if extended_hours else [16, 9.5]
        rangebreaks.append(dict(bounds=hour_bounds, pattern="hour"))
    fig.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
    fig.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)

    # uirevision יציב כל עוד הסימבול/טיים-פריים/כמות הדאטה זהים — כך זום/פאן שהמשתמש עשה
    # לא מתאפסים בכל rerun של Streamlit (למשל שינוי הון התחלתי), רק כשבאמת נטען דאטה חדש.
    fig.update_layout(uirevision=f"{symbol}|{timeframe}|{len(data)}|{extended_hours}")

    if not data.empty:
        last = data.iloc[-1]
        prev_close = data["Close"].iloc[-2] if len(data) > 1 else last["Open"]
        change = last["Close"] - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        move_color = TV_GREEN if change >= 0 else TV_RED
        sign = "+" if change >= 0 else "-"

        header = f"<b>{symbol}</b>"
        if timeframe:
            header += f" <span style='color:{TV_TEXT}'>· {timeframe}</span>"
        ohlc_line = (
            f"O<span style='color:#d1d4dc'>{last['Open']:.2f}</span> "
            f"H<span style='color:#d1d4dc'>{last['High']:.2f}</span> "
            f"L<span style='color:#d1d4dc'>{last['Low']:.2f}</span> "
            f"C<span style='color:#d1d4dc'>{last['Close']:.2f}</span> "
            f"<span style='color:{move_color}'>{sign}{abs(change):.2f} ({sign}{abs(change_pct):.2f}%)</span>"
        )

        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.006, y=0.99,
            xanchor="left", yanchor="top",
            showarrow=False, align="left",
            font=dict(size=13, color=TV_TEXT),
            text=f"{header}<br>{ohlc_line}",
            row=1, col=1,
        )

        fig.add_hline(
            y=last["Close"],
            line_dash="dot", line_width=1, line_color=move_color,
            annotation_text=f"{last['Close']:.2f}",
            annotation_position="right",
            annotation_font_color="#ffffff",
            annotation_font_size=11,
            annotation_bgcolor=move_color,
            annotation_borderpad=3,
            row=1, col=1,
        )

    return fig


def _plotly_js_tag() -> str:
    """מחזיר את תגית הטעינה של plotly.js.

    מעדיף קובץ סטטי שמוגש מהשרת (נשמר ב-cache של הדפדפן ולכן נטען פעם אחת בלבד),
    ונופל חזרה להטמעה בתוך ה-HTML אם הקובץ חסר — כך הגרף עובד גם בלי static serving.
    """
    if _PLOTLY_JS_FILE.exists():
        return f'<script src="{PLOTLY_JS_URL}"></script>'
    import plotly.offline as _po

    return f"<script>{_po.get_plotlyjs()}</script>"


def build_price_chart_html(
    fig: go.Figure,
    div_id: str = "tv-price-chart",
    symbol: str = "",
    timeframe: str = "",
    enable_load_more: bool = True,
    auto_resolution: bool = True,
    extended_hours: bool = True,
    view_token: str = "",
) -> str:
    """עוטף את גרף המחיר ב-HTML עם JS שמבצע שני דברים:

    1. מחשב מחדש את טווח ציר ה-Y לפי הנרות הנראים בלבד בכל זום/פאן על ציר הזמן —
       בדיוק כמו מצב ה-'Auto' של ציר המחיר ב-TradingView (Plotly לא עושה את זה מובנה).
    2. "גלילה אינסופית" של היסטוריה: כשגוררים את הגרף אחורה וקרבים לקצה השמאלי של
       הדאטה הטעונה, מבקש אוטומטית מהאפליקציה לטעון עוד היסטוריה אחורה (במקום לשאוב
       הכל מראש בבת אחת) — על ידי לחיצה תכנותית על כפתור Streamlit אמיתי (LOAD_MORE_BUTTON_LABEL,
       מוצג בעדינות מעל הגרף) שמפעיל rerun רגיל ומגדיל את "ימים אחורה" (ב-app.py, דרך
       fetch_ohlcv עם הקאש). לאחר מכן הגרף מציג את מלוא הטווח שנטען מחדש.

    שני אלה מוטמעים כאן ישירות ב-HTML/JS במקום דרך st.plotly_chart הרגיל, שלא חושף
    אירועי zoom/pan ואין לו ערוץ תקשורת חוזר ל-Python.
    """
    fragment = fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id, config=PLOTLY_CONFIG)
    fragment = _plotly_js_tag() + fragment
    load_more_enabled = "true" if enable_load_more else "false"
    load_more_label = json.dumps(LOAD_MORE_BUTTON_LABEL)
    auto_res_enabled = "true" if auto_resolution else "false"
    res_steps_json = json.dumps(RESOLUTION_STEPS)
    res_prefix = json.dumps(AUTO_RES_PREFIX)
    minutes_per_day = 960 if extended_hours else 390
    current_tf = json.dumps(timeframe)
    view_token_js = json.dumps(view_token)

    auto_fit_js = f"""
    <script>
    (function() {{
        var gd = document.getElementById("{div_id}");
        if (!gd) return;

        // אירועי relayout שנוצרים על ידי הקוד עצמו (התאמת ציר Y) מסומנים כאן כדי
        // שלא ייחשבו כפעולת זום/גרירה של המשתמש.
        var programmatic = false;
        // חלון חסד קצר אחרי הרינדור: plotly יורה relayout גם בזמן הפריסה הראשונית.
        var ready = false;
        setTimeout(function() {{ ready = true; }}, 900);

        var candleIdx = -1;
        for (var i = 0; i < gd.data.length; i++) {{
            if (gd.data[i].type === "candlestick") {{ candleIdx = i; break; }}
        }}

        // חותמות הזמן מחושבות פעם אחת בלבד. קודם הן נבנו מחדש (new Date לכל נר) בכל
        // אירוע זום/פאן — עם עשרות אלפי נרות זה יצר עיכוב מורגש שגרם לגרף להיראות תקוע.
        var candleTimes = [];
        if (candleIdx >= 0) {{
            var _xs = gd.data[candleIdx].x;
            for (var _i = 0; _i < _xs.length; _i++) candleTimes.push(new Date(_xs[_i]).getTime());
        }}

        function fitYAxis(x0, x1) {{
            if (candleIdx < 0) return;
            var trace = gd.data[candleIdx];
            var highs = trace.high, lows = trace.low;
            var t0 = x0 != null ? new Date(x0).getTime() : null;
            var t1 = x1 != null ? new Date(x1).getTime() : null;
            var minY = Infinity, maxY = -Infinity;
            for (var i = 0; i < candleTimes.length; i++) {{
                if (t0 != null && (candleTimes[i] < t0 || candleTimes[i] > t1)) continue;
                if (lows[i] < minY) minY = lows[i];
                if (highs[i] > maxY) maxY = highs[i];
            }}
            if (!isFinite(minY) || !isFinite(maxY)) return;
            var pad = (maxY - minY) * 0.08 || (Math.abs(maxY) * 0.01) || 1;
            // relayout תכנותי מפעיל בעצמו plotly_relayout. בלי הדגל הזה כל התאמת ציר Y
            // הייתה נראית כמו זום של המשתמש ומפעילה החלפת רזולוציה/טעינת היסטוריה.
            programmatic = true;
            Plotly.relayout(gd, {{"yaxis.range": [minY - pad, maxY + pad], "yaxis.autorange": false}})
                .then(function() {{ programmatic = false; }}, function() {{ programmatic = false; }});
        }}

        function extractXRange(evt) {{
            if (evt["xaxis.autorange"] || evt["xaxis2.autorange"]) return null;
            var r00 = evt["xaxis.range[0]"], r01 = evt["xaxis.range[1]"];
            var r10 = evt["xaxis2.range[0]"], r11 = evt["xaxis2.range[1]"];
            if (r00 !== undefined && r01 !== undefined) return [r00, r01];
            if (r10 !== undefined && r11 !== undefined) return [r10, r11];
            var r = evt["xaxis.range"] || evt["xaxis2.range"];
            if (r) return r;
            return undefined;
        }}

        // --- גלילה אינסופית: טעינת עוד היסטוריה כשמתקרבים לקצה השמאלי הטעון ---
        // הקביעה אם נגמרה ההיסטוריה נעשית בצד השרת (history_exhausted ב-app.py) ומגיעה
        // לכאן דרך LOAD_MORE_ENABLED. בעבר זה נשמר ב-sessionStorage בצד הדפדפן ונשאר
        // תקוע לצמיתות — כך שגם תקלת תקשורת חולפת מול TWS נעלה את הגרף לכל הסשן.
        var LOAD_MORE_ENABLED = {load_more_enabled};
        var loadMoreTriggered = false;

        function maybeLoadMoreHistory(x0, x1) {{
            if (!LOAD_MORE_ENABLED || loadMoreTriggered || candleIdx < 0) return;
            if (!candleTimes.length) return;
            var earliest = candleTimes[0];
            var t0 = new Date(x0).getTime(), t1 = new Date(x1).getTime();
            var span = t1 - t0;
            if (!(span > 0)) return;
            // נטען עוד כשהקצה השמאלי של התצוגה מגיע לרבע הראשון של הדאטה הקיימת —
            // תופס גם גרירה אחורה וגם זום-אאוט (שבו t0 יוצא אל מעבר לנר הראשון).
            if (t0 - earliest <= span * 0.25) {{
                loadMoreTriggered = true;
                saveRestoreRange(x0, x1);
                clickLoadMoreButton();
            }}
        }}

        // --- התאמת רזולוציה אוטומטית בזום פנימה/החוצה (כמו TradingView) ---
        // מתוך טווח הזמן הנראה מחושב כמה נרות ייצאו בכל רזולוציה נתמכת, ונבחרת העדינה
        // ביותר שעדיין לא צפופה מדי. ההחלפה מתבצעת רק כשהמצב הנוכחי באמת חורג מהגבולות
        // (היסטרזיס), כדי לא לקפוץ הלוך-ושוב סביב סף אחד.
        var AUTO_RES = {auto_res_enabled};
        var RES_STEPS = {res_steps_json};
        var RES_PREFIX = {res_prefix};
        var CURRENT_TF = {current_tf};
        var MINUTES_PER_DAY = {minutes_per_day};
        var TARGET_BARS = {TARGET_VISIBLE_BARS};
        var MAX_BARS = {MAX_VISIBLE_BARS};
        var MIN_BARS = {MIN_VISIBLE_BARS};
        var resolutionSwitched = false;

        // כמה נרות של רזולוציה מסוימת נכנסים בטווח הנראה. ימי מסחר בלבד (≈5/7),
        // ובתוך היום רק שעות המסחר — אחרת זום על סוף שבוע/לילה היה נספר כנרות.
        function barsAt(tfMinutes, spanMs) {{
            var tradingDays = (spanMs / 86400000) * (5 / 7);
            if (tfMinutes >= 1440) return tradingDays;
            return tradingDays * MINUTES_PER_DAY / tfMinutes;
        }}

        function currentTfMinutes() {{
            for (var i = 0; i < RES_STEPS.length; i++) {{
                if (RES_STEPS[i][0] === CURRENT_TF) return RES_STEPS[i][1];
            }}
            return 1;
        }}

        function pickResolution(spanMs) {{
            for (var i = 0; i < RES_STEPS.length; i++) {{
                if (barsAt(RES_STEPS[i][1], spanMs) <= TARGET_BARS) return RES_STEPS[i][0];
            }}
            return RES_STEPS[RES_STEPS.length - 1][0];
        }}

        function maybeSwitchResolution(x0, x1) {{
            if (!AUTO_RES || resolutionSwitched) return false;
            var t0 = new Date(x0).getTime(), t1 = new Date(x1).getTime();
            var span = t1 - t0;
            if (!(span > 0)) return false;

            var nowBars = barsAt(currentTfMinutes(), span);
            // בתוך הגבולות — הרזולוציה הנוכחית עדיין קריאה, לא נוגעים בה.
            if (nowBars <= MAX_BARS && nowBars >= MIN_BARS) return false;

            var want = pickResolution(span);
            if (want === CURRENT_TF) return false;
            resolutionSwitched = true;
            // שומרים את החלון שהמשתמש מסתכל עליו כדי לשחזר אותו אחרי הטעינה מחדש —
            // אחרת החלפת רזולוציה הייתה "קופצת" לכל טווח הדאטה החדש ומאבדת את המיקום.
            saveRestoreRange(x0, x1);
            return clickButtonByText(RES_PREFIX + want);
        }}

        // components.html רץ ב-iframe מוגבל (sandboxed) שאסור עליו לנווט את העמוד הראשי
        // (window.top.location) — הדפדפן חוסם זאת. לכן במקום ניווט, "לוחצים" תכנותית על
        // כפתור Streamlit אמיתי (מוצג בעדינות מעל הגרף) שמפעיל rerun רגיל בצד Python.
        // החלון שהמשתמש מסתכל עליו נשמר ב-sessionStorage של העמוד ההורה (ה-iframe נבנה
        // מחדש בכל rerun ואין לו זיכרון משלו), ומשוחזר בכל רינדור. הוא *לא* נמחק אחרי
        // שחזור: rerun אחד של Streamlit יכול לגרור רינדור נוסף (למשל הרצה מחדש של
        // הבדיקה), וניקוי מוקדם היה מחזיר את הגרף לטווח המלא ומאבד את הזום.
        // VIEW_TOKEN מבטל את השמירה כשבאמת מתחילים תצוגה חדשה — סימבול אחר או בחירת
        // טווח מפורשת בטאבי ציר-הזמן. החלפת רזולוציה בזום שומרת עליו במכוון.
        var VIEW_TOKEN = {view_token_js};

        function saveRestoreRange(x0, x1) {{
            try {{
                window.parent.sessionStorage.setItem(
                    "tvRestoreRange", JSON.stringify({{token: VIEW_TOKEN, range: [x0, x1]}})
                );
            }} catch (e) {{}}
        }}

        function applyRestoreRange() {{
            var saved = null;
            try {{
                saved = JSON.parse(window.parent.sessionStorage.getItem("tvRestoreRange"));
            }} catch (e) {{ return false; }}
            if (!saved || saved.token !== VIEW_TOKEN || !saved.range || saved.range.length !== 2) {{
                return false;
            }}
            var r = saved.range;
            // החלון השמור נחתך לגבולות הדאטה שנטענה. בלי זה חלון רחב שנשמר ברזולוציה
            // גסה (למשל חצי שנה ביומי) היה משוחזר כמות שהוא אחרי מעבר לנרות דקה, ודוחס
            // את כל הנרות לקצה הימני. חיתוך גם מטפל בחוסר חפיפה מוחלט — אין מה לשחזר.
            if (candleTimes.length) {{
                var dataStart = candleTimes[0], dataEnd = candleTimes[candleTimes.length - 1];
                var t0 = Math.max(new Date(r[0]).getTime(), dataStart);
                var t1 = Math.min(new Date(r[1]).getTime(), dataEnd);
                if (!(t1 > t0)) return false;
                r = [new Date(t0).toISOString(), new Date(t1).toISOString()];
            }}
            programmatic = true;
            // ציר X של פאנל המחיר כפוף (matches) לציר של פאנל הווליום, ולכן קביעת
            // "xaxis.range" בלבד נבלעת ומתאפסת. הציר המוביל הוא xaxis2 — שניהם נקבעים
            // כאן, כולל ביטול autorange, אחרת הגרף חוזר להציג את כל הדאטה שנטענה.
            Plotly.relayout(gd, {{
                "xaxis2.range": r, "xaxis2.autorange": false,
                "xaxis.range": r, "xaxis.autorange": false
            }}).then(function() {{
                programmatic = false;
                fitYAxis(r[0], r[1]);
            }}, function() {{ programmatic = false; }});
            return true;
        }}

        function clickButtonByText(label) {{
            try {{
                var buttons = window.parent.document.querySelectorAll("button");
                for (var i = 0; i < buttons.length; i++) {{
                    if (buttons[i].textContent.trim() === label) {{
                        buttons[i].click();
                        return true;
                    }}
                }}
            }} catch (e) {{}}
            return false;
        }}

        function clickLoadMoreButton() {{
            clickButtonByText({load_more_label});
        }}

        gd.on("plotly_relayout", function(evt) {{
            if (programmatic) return;
            var range = extractXRange(evt);
            if (range === null) {{ fitYAxis(null, null); return; }}
            if (!range) return;
            fitYAxis(range[0], range[1]);
            if (!ready) return;
            saveRestoreRange(range[0], range[1]);
            // החלפת רזולוציה גוררת ממילא טעינה מחדש עם טווח מתאים, ולכן אם היא נדרשת
            // אין טעם לבקש באותו רגע גם "עוד היסטוריה" באותה רזולוציה.
            if (maybeSwitchResolution(range[0], range[1])) return;
            maybeLoadMoreHistory(range[0], range[1]);
        }});

        // אחרי טעינת היסטוריה נוספת מתאימים תצוגה: מציגים את כל הטווח הטעון (במקום
        // לנסות לשחזר בדיוק פיקסל את המיקום הקודם — פחות שברירי, ותמיד נכון ויזואלית).
        if (!applyRestoreRange()) {{
            var xr0 = gd.layout.xaxis && gd.layout.xaxis.range;
            fitYAxis(xr0 ? xr0[0] : null, xr0 ? xr0[1] : null);
        }}
    }})();
    </script>
    """
    return fragment + auto_fit_js


# מספר הנקודות המרבי שנשלח בפועל לדפדפן עבור גרף ה-Performance. גם אחרי תיקון
# עלות הבנייה הפייתונית (ראו למטה), 150K+ בארים/נקודות זה עדיין מאות אלפי
# אלמנטים ש-plotly.js מצייר בדפדפן (SVG לבארים, DOM לכל hover) - זה מה שהקפיא
# בפועל את הטאב אחרי "Full backtest", לא זמן הריצה בפייתון. הדגימה כאן היא על
# העקומה המצטברת המלאה (cumsum על כל העסקאות) ולכן הערכים בכל נקודה נדגמת
# עדיין נכונים - רק מספר הנקודות המוצגות קטן.
MAX_PERFORMANCE_CHART_POINTS = 1000


def _evenly_sampled_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))


def build_performance_chart(trades: pd.DataFrame, start_time=None) -> go.Figure:
    """אזור ה-Performance של TradingView: עמודת PnL לכל עסקה (ירוקה מעל האפס, אדומה
    מתחתיו) ומעליהן קו ה-Cumulative PnL עם נקודות, שטח צבוע עדין מתחתיו, ותווית ערך
    אחרונה בקצה ציר ה-Y הימני."""
    # מערכי numpy, לא רשימות פייתון רגילות (list(...) / list-comprehension) -
    # על "Full backtest" (כל ההיסטוריה, יכול לצאת 150K+ עסקאות) plotly מאמת כל
    # איבר בנפרד ברשימת פייתון רגילה, ומדדנו שזה לוקח מעל 16 שניות; עם מערכי
    # numpy plotly מאמת בבת אחת - עשיריות שנייה. זה בדיוק מה שהיה הופך "בקטסט
    # מהיר" ל"תקוע על הצגת התוצאה" בפועל, כי הלשונית הזו (Overview) היא ברירת
    # המחדל שמוצגת מיד אחרי כל ריצה.
    ordered = trades.sort_values("ExitTime")
    bar_x_full = ordered["ExitTime"].to_numpy()
    bar_y_full = ordered["PnL"].to_numpy(dtype=float)

    # ה-cumsum רץ על המערך המלא (כל העסקאות) לפני הדגימה - כדי שכל נקודה נדגמת
    # תציג את סך ה-PnL המצטבר האמיתי עד אותה עסקה, לא סכום חלקי מוטעה על תת-קבוצה.
    cum_y_full = np.cumsum(bar_y_full)
    final = float(cum_y_full[-1]) if len(cum_y_full) else 0.0

    idx = _evenly_sampled_indices(len(bar_x_full), MAX_PERFORMANCE_CHART_POINTS)
    bar_x = bar_x_full[idx]
    bar_y = bar_y_full[idx]
    cum_x = bar_x_full[idx]
    cum_y = cum_y_full[idx]

    if start_time is not None:
        cum_x = np.concatenate([[np.datetime64(start_time)], cum_x])
        cum_y = np.concatenate([[0.0], cum_y])
    line_color = TV_GREEN if final >= 0 else TV_RED
    fill_color = "rgba(8,153,129,0.13)" if final >= 0 else "rgba(242,54,69,0.13)"

    fig = go.Figure()

    # עמודות PnL לכל עסקה: שני traces בצבע אחיד (לא marker_color עם מערך צבע
    # פר-נקודה) - plotly מאמת כל איבר במערך-צבעים בנפרד (color parsing), לא
    # במסלול-מהיר של numpy כמו x/y מספריים. על 150K+ עסקאות ("Full backtest")
    # זו הייתה כמעט כל עלות בניית הגרף (מדדנו כ-10 שניות, ירד לעשיריות שנייה
    # עם שני traces צבע-אחיד וערכי NaN בצד השני של כל trace).
    pos_y = np.where(bar_y >= 0, bar_y, np.nan)
    neg_y = np.where(bar_y >= 0, np.nan, bar_y)
    fig.add_trace(
        go.Bar(
            x=bar_x, y=pos_y, name="Trade PnL",
            marker_color="rgba(38,166,154,0.85)", marker_line_width=0,
            hovertemplate="%{x}<br>Trade PnL: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=bar_x, y=neg_y, name="Trade PnL", showlegend=False,
            marker_color="rgba(242,54,69,0.85)", marker_line_width=0,
            hovertemplate="%{x}<br>Trade PnL: %{y:,.2f}<extra></extra>",
        )
    )

    # שטח + קו מצטבר עם נקודות
    fig.add_trace(
        go.Scatter(
            x=cum_x, y=cum_y, mode="lines", fill="tozeroy", fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_x, y=cum_y, mode="lines+markers", name="Cumulative PnL",
            line=dict(color=line_color, width=1.8),
            marker=dict(color=line_color, size=5, line=dict(width=0)),
            hovertemplate="%{x}<br>Cumulative: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_color="rgba(240,243,250,0.18)", line_width=1)

    fig = _tv_layout(fig, height=300)
    fig.update_layout(
        barmode="relative",
        bargap=0.55,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=1.14, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=TV_TEXT),
        ),
        margin=dict(l=10, r=88, t=34, b=10),
    )

    # תווית הערך האחרון על ציר ה-Y, כמו התגית האדומה ב-TradingView
    if len(cum_y):
        fig.add_annotation(
            xref="paper", x=1.0, y=final, xanchor="left", yanchor="middle",
            text=f"{final:,.2f}", showarrow=False,
            font=dict(size=11, color="#ffffff"),
            bgcolor=line_color, borderpad=3,
        )
    return fig
