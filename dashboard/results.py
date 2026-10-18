"""
results.py — פאנל התוצאות של בודק האסטרטגיות (עברית, Dash components).

מקבל את מבנה ה-stats שכל המנועים מחזירים (מפתחות TradingView-style + _trades
DataFrame עם EntryTime/ExitTime/EntryPrice/ExitPrice/Size/PnL/ReturnPct) ובונה:
כרטיסי KPI, סקירה (עקומת הון + drawdown), טבלת ביצועים (הכל/לונג/שורט),
טבלת עסקאות עם דפדוף, ומפת חום חודשית.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

UP, DOWN, MUTED, LINE = "#1FBF8F", "#F0475A", "#8B90A0", "#262B36"
TRADES_PAGE = 50


# ------------------------------------------------------------------ עזרי פורמט
def money(v: float, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    sign = "-" if (signed and v < 0) else ""
    return f"{sign}${abs(v):,.0f}" if abs(v) >= 100 else f"{sign}${abs(v):,.2f}"


def pct(v: float, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}%"


def num(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{digits}f}" if isinstance(v, float) else f"{v:,}"


def _cls(v: float) -> str:
    if v is None or not np.isfinite(v) or v == 0:
        return "neu"
    return "pos" if v > 0 else "neg"


# ------------------------------------------------------------------ מדדים
def metrics(trades: pd.DataFrame, cash: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "net": 0.0, "ret": 0.0, "gp": 0.0, "gl": 0.0, "pf": 0.0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "avg": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "best": 0.0, "worst": 0.0,
                "max_dd": 0.0, "max_dd_pct": 0.0, "expectancy": 0.0, "avg_dur": None, "streak_w": 0, "streak_l": 0}
    pnl = trades["PnL"].to_numpy(dtype=float)
    wins_mask = pnl > 0
    gp = float(pnl[wins_mask].sum()); gl = float(abs(pnl[~wins_mask].sum()))
    ordered = trades.sort_values("ExitTime")
    cum = np.cumsum(ordered["PnL"].to_numpy(dtype=float))
    equity = cash + cum
    peak = np.maximum.accumulate(np.concatenate([[cash], equity]))[1:]
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float((dd / peak).min() * 100) if len(dd) else 0.0
    dur = (ordered["ExitTime"] - ordered["EntryTime"]).dt.total_seconds() / 60.0
    # רצפים
    sw = sl = cw = cl = 0
    for p in ordered["PnL"].to_numpy():
        if p > 0:
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        sw, sl = max(sw, cw), max(sl, cl)
    wins = int(wins_mask.sum()); losses = n - wins
    win_rate = wins / n * 100
    avg_win = gp / wins if wins else 0.0
    avg_loss = gl / losses if losses else 0.0
    return {
        "n": n, "net": float(pnl.sum()), "ret": float(pnl.sum()) / cash * 100 if cash else 0.0,
        "gp": gp, "gl": gl, "pf": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg": float(pnl.mean()), "avg_win": avg_win, "avg_loss": avg_loss,
        "best": float(pnl.max()), "worst": float(pnl.min()),
        "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "expectancy": (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss,
        "avg_dur": float(dur.mean()) if len(dur) else None,
        "streak_w": sw, "streak_l": sl,
    }


def _dur_text(minutes: float | None) -> str:
    if minutes is None or not np.isfinite(minutes):
        return "—"
    if minutes < 90:
        return f"{minutes:.0f} דק'"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} שע'"
    return f"{minutes / 1440:.1f} ימים"


# ------------------------------------------------------------------ KPI
def kpi_tiles(stats: dict, cash: float, bh_pct: float | None = None) -> html.Div:
    tr = stats["_trades"]
    m = metrics(tr, cash)
    bh = bh_pct if bh_pct is not None else float(stats.get("Buy & Hold Return [%]", 0.0))
    tiles = [
        ("רווח נקי", money(m["net"]), _cls(m["net"]), pct(m["ret"])),
        ("עסקאות", num(m["n"], 0), "neu", f"{m['wins']} רווח · {m['losses']} הפסד"),
        ("אחוז זכייה", pct(m["win_rate"], 0), _cls(m["win_rate"] - 50), f"רצף: {m['streak_w']} ▲ / {m['streak_l']} ▼"),
        ("Profit Factor", num(m["pf"]) if np.isfinite(m["pf"]) else "∞", _cls(m["pf"] - 1), "רווח גולמי ÷ הפסד גולמי"),
        ("Drawdown מקסימלי", pct(abs(m["max_dd_pct"])), "neg" if m["max_dd"] < 0 else "neu", money(m["max_dd"])),
        ("ממוצע לעסקה", money(m["avg"]), _cls(m["avg"]), f"תוחלת {money(m['expectancy'])}"),
        ("משך עסקה ממוצע", _dur_text(m["avg_dur"]), "neu", f"הטובה {money(m['best'])} · הגרועה {money(m['worst'])}"),
        ("קנה והחזק", pct(bh), _cls(bh), "לאותה תקופה"),
    ]
    grid = html.Div(className="kpi-grid", children=[
        html.Div(className=f"kpi {cls}", children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub"),
        ]) for label, value, cls, sub in tiles
    ])
    # בדיקת "ההון נגמר": ההון המינימלי לאורך הדרך מול מרווח הביטחון (MNQ ≈ 2,500$ לחוזה)
    warn = None
    if m["n"]:
        o = tr.sort_values("ExitTime"); eq_min = float(cash + np.cumsum(o["PnL"].to_numpy(dtype=float)).min())
        contracts = float(abs(tr["Size"]).max()) if len(tr) else 1.0
        maint = 2500.0 * contracts
        if eq_min <= 0:
            warn = f"ההון ירד מתחת לאפס במהלך הבדיקה (מינימום {money(eq_min)}). התוצאות אחרי הנקודה הזו לא היו קורות בחשבון אמיתי."
        elif eq_min < maint:
            warn = f"ההון ירד ל-{money(eq_min)}, מתחת למרווח הביטחון המשוער ({money(maint)} ל-{contracts:g} חוזים). בחשבון אמיתי היו נסגרות פוזיציות."
    return html.Div([grid, html.Div(f"⚠ {warn}", className="tester-error")] if warn else [grid])


# ------------------------------------------------------------------ סקירה: עקומת הון + drawdown
def _sample(x, y, max_points=2000):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def overview(stats: dict, cash: float) -> html.Div:
    tr = stats["_trades"]
    if tr.empty:
        return _empty("לא בוצעו עסקאות בטווח הזה.")
    o = tr.sort_values("ExitTime")
    x = o["ExitTime"].to_numpy()
    pnl = o["PnL"].to_numpy(dtype=float)
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = cum - peak
    xs, ys = _sample(x, cum); _, ds = _sample(x, dd)
    final = float(cum[-1])
    color = UP if final >= 0 else DOWN
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="רווח מצטבר", line=dict(color=color, width=1.8),
                             fill="tozeroy", fillcolor=("rgba(31,191,143,.10)" if final >= 0 else "rgba(240,71,90,.10)"),
                             hovertemplate="%{x|%d/%m/%y %H:%M}<br>%{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=ds, mode="lines", name="Drawdown", line=dict(color=DOWN, width=1),
                             fill="tozeroy", fillcolor="rgba(240,71,90,.18)", yaxis="y2",
                             hovertemplate="%{x|%d/%m/%y %H:%M}<br>%{y:$,.0f}<extra></extra>"))
    fig.update_layout(
        height=330, margin=dict(l=8, r=64, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2B5BE", size=11, family="Heebo, sans-serif"), showlegend=False, hovermode="x unified",
        yaxis=dict(domain=[0.34, 1], gridcolor=LINE, zeroline=True, zerolinecolor="#3a3f4a", tickformat="$,.0f", side="right"),
        yaxis2=dict(domain=[0, 0.28], gridcolor=LINE, zeroline=False, tickformat="$,.0f", side="right"),
        xaxis=dict(gridcolor=LINE, showgrid=False),
    )
    m_all = metrics(tr, cash)
    m_long = metrics(tr[tr["Size"] > 0], cash)
    m_short = metrics(tr[tr["Size"] < 0], cash)
    split = html.Div(className="split-row", children=[
        _split_card("לונג", m_long), _split_card("שורט", m_short), _split_card("הכל", m_all),
    ])
    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div(className="chart-card-head", children=[html.Span("עקומת הון ו-Drawdown"), html.Span(f"{len(tr):,} עסקאות", className="muted")]),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "330px"}),
        ]),
        split,
    ])


def _split_card(title: str, m: dict) -> html.Div:
    rows = [("עסקאות", num(m["n"], 0), "neu"), ("רווח נקי", money(m["net"]), _cls(m["net"])),
            ("אחוז זכייה", pct(m["win_rate"], 0), "neu"),
            ("Profit Factor", num(m["pf"]) if np.isfinite(m["pf"]) else "∞", _cls(m["pf"] - 1)),
            ("ממוצע לעסקה", money(m["avg"]), _cls(m["avg"]))]
    return html.Div(className="split-card", children=[html.Div(title, className="split-title")] + [
        html.Div(className="split-line", children=[html.Span(k), html.Span(v, className=c)]) for k, v, c in rows])


# ------------------------------------------------------------------ ביצועים
def performance_table(stats: dict, cash: float) -> html.Div:
    tr = stats["_trades"]
    if tr.empty:
        return _empty("לא בוצעו עסקאות.")
    cols = [("הכל", tr), ("לונג", tr[tr["Size"] > 0]), ("שורט", tr[tr["Size"] < 0])]
    ms = [metrics(t, cash) for _, t in cols]
    rows = [
        ("רווח נקי", lambda m: (money(m["net"]), _cls(m["net"]))),
        ("תשואה", lambda m: (pct(m["ret"], 2), _cls(m["ret"]))),
        ("רווח גולמי", lambda m: (money(m["gp"]), "pos" if m["gp"] > 0 else "neu")),
        ("הפסד גולמי", lambda m: (money(-m["gl"]), "neg" if m["gl"] > 0 else "neu")),
        ("Profit Factor", lambda m: (num(m["pf"]) if np.isfinite(m["pf"]) else "∞", _cls(m["pf"] - 1))),
        ("עסקאות", lambda m: (num(m["n"], 0), "neu")),
        ("עסקאות רווחיות", lambda m: (f"{m['wins']} ({pct(m['win_rate'], 0)})", "neu")),
        ("עסקאות מפסידות", lambda m: (num(m["losses"], 0), "neu")),
        ("ממוצע לעסקה", lambda m: (money(m["avg"]), _cls(m["avg"]))),
        ("ממוצע עסקה רווחית", lambda m: (money(m["avg_win"]), "pos" if m["avg_win"] else "neu")),
        ("ממוצע עסקה מפסידה", lambda m: (money(-m["avg_loss"]), "neg" if m["avg_loss"] else "neu")),
        ("יחס רווח/הפסד ממוצע", lambda m: (num(m["avg_win"] / m["avg_loss"]) if m["avg_loss"] else "—", "neu")),
        ("תוחלת לעסקה", lambda m: (money(m["expectancy"]), _cls(m["expectancy"]))),
        ("העסקה הטובה ביותר", lambda m: (money(m["best"]), "pos")),
        ("העסקה הגרועה ביותר", lambda m: (money(m["worst"]), "neg")),
        ("רצף זכיות מקסימלי", lambda m: (num(m["streak_w"], 0), "neu")),
        ("רצף הפסדים מקסימלי", lambda m: (num(m["streak_l"], 0), "neu")),
        ("Drawdown מקסימלי", lambda m: (f"{money(m['max_dd'])} ({pct(abs(m['max_dd_pct']))})", "neg" if m["max_dd"] < 0 else "neu")),
        ("משך עסקה ממוצע", lambda m: (_dur_text(m["avg_dur"]), "neu")),
    ]
    head = html.Tr([html.Th("מדד")] + [html.Th(name) for name, _ in cols])
    body = []
    for label, fn in rows:
        cells = [html.Td(label, className="metric")]
        for m in ms:
            v, c = fn(m) if m["n"] else ("—", "neu")
            cells.append(html.Td(v, className=c))
        body.append(html.Tr(cells))
    return html.Div(html.Table(className="perf-table", children=[html.Thead(head), html.Tbody(body)]), className="table-wrap")


# ------------------------------------------------------------------ עסקאות
def apply_trade_filter(tr: pd.DataFrame, flt: dict | None) -> pd.DataFrame:
    """מסנן וממיין את טבלת העסקאות לפי בחירת המשתמש (צד, תוצאה, גודל מינימלי, מיון)."""
    o = tr.sort_values("ExitTime").reset_index(drop=True)
    o["Cum"] = o["PnL"].cumsum()
    o["No"] = np.arange(1, len(o) + 1)
    if not flt:
        return o.iloc[::-1]
    side = flt.get("side") or "all"; result = flt.get("result") or "all"; min_abs = float(flt.get("min_abs") or 0)
    if side == "long": o = o[o["Size"] > 0]
    elif side == "short": o = o[o["Size"] < 0]
    if result == "win": o = o[o["PnL"] > 0]
    elif result == "loss": o = o[o["PnL"] <= 0]
    if min_abs > 0: o = o[o["PnL"].abs() >= min_abs]
    sort = flt.get("sort") or "time_desc"
    if sort == "time_asc": o = o.sort_values("ExitTime")
    elif sort == "pnl_desc": o = o.sort_values("PnL", ascending=False)
    elif sort == "pnl_asc": o = o.sort_values("PnL", ascending=True)
    elif sort == "dur_desc": o = o.assign(_d=(o["ExitTime"] - o["EntryTime"])).sort_values("_d", ascending=False)
    else: o = o.sort_values("ExitTime", ascending=False)
    return o


def filtered_count(stats: dict, flt: dict | None) -> int:
    tr = stats["_trades"]
    return 0 if tr.empty else int(len(apply_trade_filter(tr, flt)))


def trades_table(stats: dict, page: int = 0, page_size: int = TRADES_PAGE, flt: dict | None = None) -> html.Div:
    tr = stats["_trades"]
    if tr.empty:
        return _empty("לא בוצעו עסקאות.")
    o = apply_trade_filter(tr, flt)
    if o.empty:
        return _empty("אין עסקאות שמתאימות לסינון.")
    start = page * page_size
    chunk = o.iloc[start:start + page_size]
    head = html.Tr([html.Th("#"), html.Th("כיוון"), html.Th("כניסה"), html.Th("שעה NY"), html.Th("יציאה"), html.Th("מחיר כניסה"),
                    html.Th("מחיר יציאה"), html.Th("כמות"), html.Th("רווח/הפסד"), html.Th("תשואה"), html.Th("מצטבר")])
    from .data import _epoch
    body = []
    for _, r in chunk.iterrows():
        long = r["Size"] > 0
        body.append(html.Tr(className="trade-row", title="לחיצה מקפיצה את הגרף לעסקה", **{"data-entry": int(_epoch(pd.Timestamp(r["EntryTime"])))}, children=[
            html.Td(int(r["No"]), className="muted"),
            html.Td("לונג" if long else "שורט", className=("pos" if long else "neg")),
            html.Td(pd.Timestamp(r["EntryTime"]).strftime("%d/%m/%y %H:%M")),
            html.Td(pd.Timestamp(r["EntryTime"]).strftime("%H:%M"), className="muted"),
            html.Td(pd.Timestamp(r["ExitTime"]).strftime("%d/%m/%y %H:%M")),
            html.Td(f"{r['EntryPrice']:,.2f}"), html.Td(f"{r['ExitPrice']:,.2f}"),
            html.Td(f"{abs(r['Size']):g}"),
            html.Td(money(r["PnL"]), className=_cls(r["PnL"])),
            html.Td(pct(r["ReturnPct"] * 100, 2), className=_cls(r["ReturnPct"])),
            html.Td(money(r["Cum"]), className=_cls(r["Cum"])),
        ]))
    return html.Div(html.Table(className="trades-table", children=[html.Thead(head), html.Tbody(body)]), className="table-wrap")


# ------------------------------------------------------------------ חודשי
_MONTHS = ["ינו", "פבר", "מרץ", "אפר", "מאי", "יונ", "יול", "אוג", "ספט", "אוק", "נוב", "דצמ"]


def monthly(stats: dict, cash: float) -> html.Div:
    tr = stats["_trades"]
    if tr.empty:
        return _empty("לא בוצעו עסקאות.")
    o = tr.copy()
    o["y"] = o["ExitTime"].dt.year; o["m"] = o["ExitTime"].dt.month
    piv = o.pivot_table(index="y", columns="m", values="PnL", aggfunc="sum").reindex(columns=range(1, 13))
    years = [str(y) for y in piv.index]
    z = piv.to_numpy(dtype=float)
    text = [[("" if not np.isfinite(v) else money(v)) for v in row] for row in z]
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=_MONTHS, y=years, text=text, texttemplate="%{text}", textfont=dict(size=11, family="Heebo, sans-serif"),
        colorscale=[[0, "#F0475A"], [0.5, "#181C25"], [1, "#1FBF8F"]], zmin=-lim, zmax=lim, showscale=False,
        hovertemplate="%{y} %{x}<br>%{text}<extra></extra>", xgap=3, ygap=3,
    ))
    fig.update_layout(height=max(160, 46 * len(years) + 60), margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#B2B5BE", size=11, family="Heebo, sans-serif"),
                      xaxis=dict(side="top", showgrid=False, zeroline=False, autorange="reversed", fixedrange=True),
                      yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, fixedrange=True))
    yearly = piv.sum(axis=1)
    cards = [html.Div(className="year-card", children=[html.Div(str(y), className="year-k"),
                                                       html.Div(money(v), className=f"year-v {_cls(v)}")])
             for y, v in yearly.items()]
    # רווח לפי שעת הכניסה (שעון ניו-יורק): חושף אם האסטרטגיה עובדת רק בחלק מהיום
    hrs = tr.assign(h=tr["EntryTime"].dt.hour).groupby("h")["PnL"].agg(["sum", "count"]).reindex(range(24), fill_value=0)
    colors = [UP if v >= 0 else DOWN for v in hrs["sum"]]
    hfig = go.Figure(go.Bar(x=[f"{h:02d}" for h in hrs.index], y=hrs["sum"], marker_color=colors,
                            text=[f"{int(c)}" if c else "" for c in hrs["count"]], textposition="outside", textfont=dict(size=10),
                            hovertemplate="%{x}:00 ניו-יורק<br>%{y:$,.0f}<br>%{text} עסקאות<extra></extra>"))
    hfig.update_layout(height=220, margin=dict(l=8, r=8, t=18, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#B2B5BE", size=11, family="Heebo, sans-serif"), showlegend=False,
                       xaxis=dict(showgrid=False, fixedrange=True), yaxis=dict(gridcolor=LINE, zeroline=True, zerolinecolor="#3a3f4a", tickformat="$,.0f", fixedrange=True))
    hfig.add_vrect(x0=8.5, x1=15.5, fillcolor="rgba(61,123,255,.08)", line_width=0, annotation_text="סשן רגיל", annotation_position="top left", annotation_font_size=10)
    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div(className="chart-card-head", children=[html.Span("רווח/הפסד לפי חודש"), html.Span("לפי מועד סגירת העסקה", className="muted")]),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]),
        html.Div(className="year-row", children=cards),
        html.Div(className="chart-card", style={"marginTop": "12px"}, children=[
            html.Div(className="chart-card-head", children=[html.Span("רווח/הפסד לפי שעת הכניסה"), html.Span("שעון ניו-יורק · המספר מעל העמודה = כמות עסקאות", className="muted")]),
            dcc.Graph(figure=hfig, config={"displayModeBar": False}, style={"height": "220px"}),
        ]),
    ])


def _empty(text: str) -> html.Div:
    return html.Div(className="empty-state", children=[html.Div(text, className="empty-title")])


def pagination(n_trades: int, page: int, page_size: int = TRADES_PAGE) -> tuple[dict, str]:
    if n_trades <= page_size:
        return {"display": "none"}, ""
    n_pages = max(1, -(-n_trades // page_size))
    return {"display": "flex"}, f"עמוד {page + 1} מתוך {n_pages} · {n_trades:,} עסקאות"


# ------------------------------------------------------------------ מונטה קרלו
def monte_carlo_view(stats: dict, cash: float) -> html.Div:
    from . import lab
    tr = stats["_trades"]
    mc = lab.monte_carlo(tr, cash, n_sims=1000)
    if not mc.get("ok"):
        return _empty("צריך לפחות 5 עסקאות לסימולציה.")
    f, d = mc["final"], mc["dd"]
    tiles = [
        ("הסתברות לרווח", pct(mc["p_profit"], 0), _cls(mc["p_profit"] - 50), f"מתוך {mc['n_sims']:,} סימולציות"),
        ("רווח סופי חציוני", money(f["p50"]), _cls(f["p50"]), f"בפועל {money(f['actual'])}"),
        ("טווח רווח 5%–95%", f"{money(f['p5'])} עד {money(f['p95'])}", "neu", "90% מהתרחישים בתוך הטווח"),
        ("Drawdown חציוני", money(d["p50"]), "neg", f"בפועל {money(d['actual'])}"),
        ("Drawdown גרוע (5%)", money(d["p5"]) if "p5" in d else "—", "neg", f"הגרוע ביותר {money(d['worst'])}"),
        ("סיכוי ל-DD גרוע מהמקור", pct(mc["p_dd_worse_than_actual"], 0), "neu", "מה שקרה בפועל היה מזל?"),
        ("סיכוי ל-DD של 20%+", pct(mc["risk_of_ruin_20"], 0), "neg" if mc["risk_of_ruin_20"] > 5 else "neu", "מההון ההתחלתי"),
        ("עסקאות בכל סימולציה", num(mc["n"], 0), "neu", "דגימה מחדש עם החזרה"),
    ]
    tiles_div = html.Div(className="kpi-grid", children=[
        html.Div(className=f"kpi {c}", children=[html.Div(l, className="kpi-label"), html.Div(v, className="kpi-value mc-val"),
                                                 html.Div(sub, className="kpi-sub")]) for l, v, c, sub in tiles])
    fig = go.Figure()
    x = mc["path_x"]
    for p in mc["paths"]:
        fig.add_trace(go.Scatter(x=x[:len(p)], y=p, mode="lines", line=dict(color="rgba(139,144,160,.18)", width=1),
                                 hoverinfo="skip", showlegend=False))
    ap = mc["actual_path"]
    fig.add_trace(go.Scatter(x=x[:len(ap)], y=ap, mode="lines", name="בפועל", line=dict(color="#3D7BFF", width=2.2),
                             hovertemplate="עסקה %{x}<br>%{y:$,.0f}<extra>בפועל</extra>"))
    fig.update_layout(height=320, margin=dict(l=8, r=64, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#B2B5BE", size=11, family="Heebo, sans-serif"), showlegend=False,
                      yaxis=dict(gridcolor=LINE, zeroline=True, zerolinecolor="#3a3f4a", tickformat="$,.0f", side="right"),
                      xaxis=dict(gridcolor=LINE, showgrid=False, title="מספר עסקה"))
    return html.Div([
        tiles_div,
        html.Div(className="chart-card", children=[
            html.Div(className="chart-card-head", children=[html.Span("מסלולי הון אפשריים"),
                                                             html.Span("אפור: 60 סימולציות מדגמיות · כחול: מה שקרה בפועל", className="muted")]),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "320px"}),
        ]),
        html.Div("איך לקרוא: אם ההסתברות לרווח נמוכה מ-60% או שה-Drawdown החציוני גדול בהרבה ממה שראית בבדיקה, "
                 "התוצאה בבדיקה תלויה בסדר המקרי של העסקאות ולא ביתרון אמיתי.", className="pane-sub"),
    ])


# ------------------------------------------------------------------ Walk-forward
def walk_forward_view(wf: dict, symbol: str, timeframe: str, range_label: str) -> html.Div:
    from . import lab
    ws = wf["windows"]
    if not ws:
        return _empty("לא היו מספיק נתונים לחלונות. הקטן את מספר החלונות או הרחב את הטווח.")
    pf = wf["oos_pf"]
    tiles = [
        ("רווח מחוץ למדגם (סה\"כ)", money(wf["oos_net"]), _cls(wf["oos_net"]), f"{wf['oos_trades']} עסקאות בכל חלונות הבדיקה"),
        ("Profit Factor מחוץ למדגם", num(pf) if np.isfinite(pf) else "∞", _cls(pf - 1), "על עסקאות הבדיקה בלבד"),
        ("חלונות רווחיים", f"{wf['profitable_windows']} / {len(ws)}", _cls(wf["profitable_windows"] - len(ws) / 2), "בחלק הבדיקה של כל חלון"),
    ]
    for nm, stt in wf["stability"].items():
        tiles.append((f"יציבות: {lab.param_en(nm)}", f"{stt['share']:.0f}%", _cls(stt["share"] - 60),
                      f"הערך הנפוץ: {lab.fmt_value(stt['mode'])}"))
    tiles_div = html.Div(className="kpi-grid", children=[
        html.Div(className=f"kpi {c}", children=[html.Div(l, className="kpi-label"), html.Div(v, className="kpi-value mc-val"),
                                                 html.Div(sub, className="kpi-sub")]) for l, v, c, sub in tiles])
    head = html.Tr([html.Th("חלון"), html.Th("אימון"), html.Th("בדיקה")] + [html.Th(lab.param_en(n), title=f"{lab.param_he(n)} ({n})") for n in wf["names"]] +
                   [html.Th("רווח באימון"), html.Th("PF אימון"), html.Th("עסקאות בבדיקה"), html.Th("רווח בבדיקה"), html.Th("PF בדיקה")])
    body = []
    for w in ws:
        tr_, te = w["train"], w["test"]
        body.append(html.Tr([
            html.Td(f"#{w['window']}"), html.Td(f"{w['train_from']} עד {w['train_to']}", className="muted"),
            html.Td(f"{w['test_from']} עד {w['test_to']}")] +
            [html.Td(lab.fmt_value(w["params"][n])) for n in wf["names"]] +
            [html.Td(money(tr_["net"]), className=_cls(tr_["net"])), html.Td(num(tr_["pf"]) if np.isfinite(tr_["pf"]) else "∞"),
             html.Td(te["trades"]), html.Td(money(te["net"]), className=_cls(te["net"])),
             html.Td(num(te["pf"]) if np.isfinite(te["pf"]) else "∞", className=_cls(te["pf"] - 1))]))
    table = html.Div(html.Table(className="opt-table", children=[html.Thead(head), html.Tbody(body)]), className="opt-table-wrap")
    curve = wf.get("oos_curve")
    parts = [html.Div(f"{symbol} {timeframe} · {range_label} · {len(ws)} חלונות · {wf['seconds']:.1f} שניות", className="opt-note"),
             tiles_div, table]
    if curve is not None and len(curve):
        x = curve["ExitTime"].to_numpy(); y = np.cumsum(curve["PnL"].to_numpy(dtype=float))
        xs, ys = _sample(x, y)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=(UP if y[-1] >= 0 else DOWN), width=1.8),
                                   fill="tozeroy", fillcolor=("rgba(31,191,143,.10)" if y[-1] >= 0 else "rgba(240,71,90,.10)"),
                                   hovertemplate="%{x|%d/%m/%y}<br>%{y:$,.0f}<extra></extra>"))
        fig.update_layout(height=240, margin=dict(l=8, r=64, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#B2B5BE", size=11, family="Heebo, sans-serif"), showlegend=False,
                          yaxis=dict(gridcolor=LINE, zeroline=True, zerolinecolor="#3a3f4a", tickformat="$,.0f", side="right"),
                          xaxis=dict(showgrid=False))
        parts.append(html.Div(className="chart-card", children=[
            html.Div(className="chart-card-head", children=[html.Span("עקומת ההון מחוץ למדגם"), html.Span("רק עסקאות מחלקי הבדיקה, ברצף", className="muted")]),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "240px"})]))
    return html.Div(parts)


# ------------------------------------------------------------------ רגישות לעלויות
def cost_sensitivity_view(res: dict, symbol: str, timeframe: str, range_label: str) -> html.Div:
    rows = res["rows"]
    if not rows:
        return _empty("לא התקבלו תוצאות.")
    by = {(r["slip"], r["comm"]): r for r in rows}
    head = html.Tr([html.Th("סליפג' כניסה (טיקים) \\ עמלה לצד")] + [html.Th(f"${c:.2f}") for c in res["commissions"]])
    body = []
    for sl in res["slippages"]:
        cells = [html.Td(f"{sl} טיקים ({sl * 0.25:g} נק')", className="metric")]
        for c in res["commissions"]:
            r = by.get((sl, c))
            if not r:
                cells.append(html.Td("—")); continue
            cells.append(html.Td([html.Div(money(r["net"]), className=f"{_cls(r['net'])} sens-net"),
                                  html.Div(f"PF {num(r['pf']) if np.isfinite(r['pf']) else '∞'} · {r['trades']} עסקאות", className="muted sens-sub")]))
        body.append(html.Tr(cells))
    base = res.get("base_net")
    real = by.get((2, 0.75)); harsh = by.get((4, 1.5))
    verdict = ""
    if base is not None and real and harsh:
        if harsh["net"] > 0 and real["net"] > 0:
            verdict = "שורד עלויות: רווחי גם בתרחיש הקשה (4 טיקים, 1.50$)."; vc = "pos"
        elif real["net"] > 0:
            verdict = "גבולי: רווחי בעלויות ריאליות אבל נשבר בתרחיש הקשה. היתרון דק."; vc = "neu"
        else:
            verdict = "לא שורד עלויות: מפסיד כבר בעלויות ריאליות (2 טיקים, 0.75$). הרווח בבדיקה בלי עלויות אינו אמיתי."; vc = "neg"
    else:
        vc = "neu"
    return html.Div([
        html.Div(f"{symbol} {timeframe} · {range_label} · {len(rows)} ריצות", className="opt-note"),
        html.Div(verdict, className=f"sens-verdict {vc}") if verdict else None,
        html.Div(html.Table(className="perf-table sens-table", children=[html.Thead(head), html.Tbody(body)]), className="table-wrap"),
        html.Div("ללא עלויות בכלל (0 טיקים, 0$): " + (money(base) if base is not None else "—") + ". ההפרש בין התא הזה לתא של 2 טיקים ו-0.75$ הוא מחיר הביצוע האמיתי.", className="pane-sub"),
    ])
