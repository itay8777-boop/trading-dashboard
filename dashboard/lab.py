"""
lab.py — "מעבדת אסטרטגיות": קטלוג האסטרטגיות המובנות, הרצה אחידה של המנועים
המהירים, ואופטימיזציית פרמטרים.

מרכז במקום אחד את מה שהיה מפוזר בין app.py (if/elif לפי המנוע) ל-strategy.py
(תבניות הקוד ורשימת ה-inputs המוכרים): לכל מנוע רשומה אחת עם שם בעברית, הסבר,
תבנית הקוד, מפת הפרמטרים (שם ה-"# input" -> פרמטר המנוע) והפונקציה שמריצה.

הוספת אסטרטגיה מובנית חדשה = פונקציית מנוע ב-strategy.py + רשומה אחת ב-ENGINES.
"""
from __future__ import annotations

import itertools
import re
import time

import numpy as np
import pandas as pd

from . import strategy as S

# ------------------------------------------------------------------ רישום המנועים
# params: (INPUT_NAME, engine_kwarg, caster, default)
_I, _F, _B = int, float, None   # _B = בוליאני (ראו _cast)


def _cast(caster, raw):
    if caster is None:
        if isinstance(raw, list):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "on", "yes")
        return bool(raw)
    return caster(raw)


ENGINES: dict[str, dict] = {
    "sma_cross": {
        "title": "חציית ממוצע נע (SMA)",
        "desc": "לונג כשהסגירה חוצה מעל ממוצע SMA, יציאה כשחוצה מתחתיו. "
                "האסטרטגיה הבסיסית ביותר — טובה כקו ייחוס להשוואה.",
        "code": S.SMA_CODE,
        "params": [("SMA_PERIOD", "sma_period", _I, 20)],
        "runner": S.run_fast_sma_crossover_backtest,
        "default_marker": True,
    },
    "hammer_rsi": {
        "title": "נר פטיש + RSI",
        "desc": "היפוך: נר פטיש (או פטיש הפוך) אחרי RSI קיצוני. סטופ ויעד קבועים "
                "בנקודות. לונג ושורט.",
        "code": S.HAMMER_RSI_CODE,
        "params": [("RSI_PERIOD", "rsi_period", _I, 14), ("RSI_OVERSOLD", "rsi_oversold", _F, 30.0),
                   ("RSI_OVERBOUGHT", "rsi_overbought", _F, 70.0), ("HAMMER_RATIO", "hammer_ratio", _F, 2.0),
                   ("STOP_LOSS_PTS", "stop_loss_pts", _F, 50.0), ("TAKE_PROFIT_PTS", "take_profit_pts", _F, 100.0)],
        "runner": S.run_hammer_rsi_backtest,
    },
    "fib_golden": {
        "title": "פיבונאצ'י + יחס הזהב",
        "desc": "נקודות מפנה מגדירות רגל; כניסה בתיקון 0.618 עם אישור RSI/נר, "
                "סטופ מעבר לנקודת המפנה, יעד בהארכת 1.618. לונג ושורט.",
        "code": S.FIB_GOLDEN_CODE,
        "params": [("SWING_BARS", "swing_bars", _I, 20), ("MIN_LEG_PTS", "min_leg_pts", _F, 40.0),
                   ("ENTRY_FIB", "entry_fib", _F, 0.618), ("STOP_FIB", "stop_fib", _F, 1.0),
                   ("STOP_BUFFER_PTS", "stop_buffer_pts", _F, 5.0), ("TARGET_EXT", "target_ext", _F, 1.618),
                   ("RSI_CONFIRM", "rsi_confirm", _B, True), ("RSI_LEN", "rsi_len", _I, 14),
                   ("RSI_LONG_MAX", "rsi_long_max", _F, 45.0), ("RSI_SHORT_MIN", "rsi_short_min", _F, 55.0),
                   ("CANDLE_CONFIRM", "candle_confirm", _B, True), ("EMA_TREND", "ema_trend", _I, 200),
                   ("ALLOW_SHORT", "allow_short", _B, True)],
        "runner": S.run_fib_golden_backtest,
    },
    "orb": {
        "title": "פריצת טווח פתיחה (ORB)",
        "desc": "הגבוה/הנמוך של הדקות הראשונות בסשן האמריקאי מגדירים טווח; סגירה "
                "מחוצה לו = כניסה. סטופ בצד השני, יעד לפי יחס סיכון/סיכוי, יציאה בסוף היום.",
        "code": S.ORB_CODE,
        "params": [("RANGE_MINUTES", "range_minutes", _I, 15), ("BUFFER_PTS", "buffer_pts", _F, 2.0),
                   ("STOP_PTS", "stop_pts", _F, 0.0), ("TARGET_RR", "target_rr", _F, 2.0),
                   ("MIN_RANGE_PTS", "min_range_pts", _F, 20.0), ("MAX_RANGE_PTS", "max_range_pts", _F, 200.0),
                   ("LAST_ENTRY_MIN", "last_entry_min", _I, 180), ("ALLOW_SHORT", "allow_short", _B, True)],
        "runner": S.run_orb_backtest,
    },
    "wyckoff": {
        "title": "וויקוף: Spring / Upthrust",
        "desc": "טווח מסחר (צבירה/הפצה) בין תמיכה להתנגדות; שבירה כוזבת מתחת לתמיכה "
                "שנסגרת חזרה בטווח, על נפח נמוך, עם אישור בנר הבא. יעד ההתנגדות, סטופ מתחת ל-Spring.",
        "code": S.WYCKOFF_CODE,
        "params": [("RANGE_BARS", "range_bars", _I, 40), ("RANGE_ATR_MULT", "range_atr_mult", _F, 6.0),
                   ("SPRING_PEN_PTS", "spring_pen_pts", _F, 5.0), ("VOL_MAX_RATIO", "vol_max_ratio", _F, 1.2),
                   ("CONFIRM_NEXT", "confirm_next", _B, True), ("TARGET_FRAC", "target_frac", _F, 1.0),
                   ("STOP_BUFFER_PTS", "stop_buffer_pts", _F, 8.0), ("MAX_BARS", "max_bars", _I, 120),
                   ("EMA_TREND", "ema_trend", _I, 0), ("ALLOW_SHORT", "allow_short", _B, True)],
        "runner": S.run_wyckoff_backtest,
    },
    "vwap_revert": {
        "title": "חזרה ל-VWAP",
        "desc": "כשהמחיר מתרחק מה-VWAP היומי ו-RSI קיצוני ומתהפך — נכנסים נגד "
                "התנועה עם יעד ב-VWAP וסטופ קבוע. סשן רגיל בלבד כברירת מחדל.",
        "code": S.VWAP_REVERT_CODE,
        "params": [("DEVIATION_PTS", "deviation_pts", _F, 40.0), ("RSI_LEN", "rsi_len", _I, 14),
                   ("RSI_LONG_MAX", "rsi_long_max", _F, 30.0), ("RSI_SHORT_MIN", "rsi_short_min", _F, 70.0),
                   ("STOP_PTS", "stop_pts", _F, 30.0), ("RTH_ONLY", "rth_only", _B, True),
                   ("ALLOW_SHORT", "allow_short", _B, True)],
        "runner": S.run_vwap_revert_backtest,
    },
}

# שמות ה-"# input" המוכרים לכל מנוע (משלים את RECOGNIZED_STRATEGY_INPUTS ב-strategy.py
# כך שגם המנועים החדשים מסומנים כ"נתמכים" בלשונית Settings).
for _key, _eng in ENGINES.items():
    for _name, _param, _c, _d in _eng["params"]:
        S.RECOGNIZED_STRATEGY_INPUTS.setdefault(_name, _param)


def engine_key_for_code(code: str) -> str:
    """המנוע שהקוד הנוכחי מפעיל: לפי '# strategy: <key>' בשורה הראשונה, אחרת
    ברירת המחדל (חציית SMA) — בדיוק אותה החלטה כמו run_or_switch המקורי."""
    key = S.detect_builtin_strategy(code or "")
    return key if key in ENGINES else "sma_cross"


def engine_params(key: str) -> list[tuple]:
    return ENGINES[key]["params"]


def build_kwargs(key: str, overrides: dict) -> dict:
    """ערכי ה-"# input" (כפי שנערכו בלשונית Settings או כפי שנכתבו בקוד) ->
    ארגומנטים למנוע. ערך חסר/לא תקין -> ברירת המחדל של הרשומה."""
    kwargs = {}
    for name, param, caster, default in ENGINES[key]["params"]:
        raw = overrides.get(name)
        if raw is None or raw == "":
            kwargs[param] = default
            continue
        try:
            kwargs[param] = _cast(caster, raw)
        except (TypeError, ValueError):
            kwargs[param] = default
    return kwargs


def run_engine(key: str, df: pd.DataFrame, common: dict, overrides: dict) -> dict:
    import inspect
    runner = ENGINES[key]["runner"]
    params = inspect.signature(runner).parameters
    common = {k: v for k, v in common.items() if k in params}
    return runner(df, **common, **build_kwargs(key, overrides))


def code_overrides(code: str) -> dict:
    """ערכי ה-"# input" כפי שכתובים בקוד (ברירת המחדל כשלא נערכו בממשק)."""
    return {it["name"]: it["value"] for it in S.parse_strategy_inputs(code or "")}


# ------------------------------------------------------------------ כתיבת ערכים לקוד
_INPUT_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.+?)(\s*#\s*input\s*)$")


def apply_inputs_to_code(code: str, values: dict) -> str:
    """מחליף את הערך בשורות '<NAME> = <value>  # input' לפי המילון, בלי לגעת
    בשאר הקוד. כך "החל" מהאופטימיזציה מעדכן גם את העורך וגם את שדות Settings
    (שנבנים מחדש מהקוד)."""
    out = []
    for line in (code or "").splitlines():
        m = _INPUT_LINE.match(line)
        if m and m.group(2) in values:
            v = values[m.group(2)]
            if isinstance(v, bool):
                txt = "True" if v else "False"
            elif isinstance(v, float) and float(v).is_integer() and abs(v) < 1e9 and not _looks_float(m.group(4)):
                txt = str(int(v))
            else:
                txt = f"{v:g}" if isinstance(v, float) else str(v)
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}{txt}{m.group(5)}"
        out.append(line)
    return "\n".join(out) + ("\n" if (code or "").endswith("\n") else "")


def _looks_float(raw: str) -> bool:
    return "." in raw.strip()


# ------------------------------------------------------------------ סטטיסטיקות קצרות
def quick_stats(stats: dict, cash: float) -> dict:
    tr: pd.DataFrame = stats["_trades"]
    n = len(tr)
    pnl = tr["PnL"] if n else pd.Series(dtype=float)
    wins = int((pnl > 0).sum()) if n else 0
    gp = float(pnl[pnl > 0].sum()) if n else 0.0
    gl = float(abs(pnl[pnl <= 0].sum())) if n else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    net = float(stats["Equity Final [$]"]) - cash
    return {
        "trades": n, "win_rate": (wins / n * 100) if n else 0.0, "pf": pf,
        "net": net, "ret_pct": float(stats["Return [%]"]),
        "max_dd_pct": abs(float(stats["Max. Drawdown [%]"])),
        "avg_trade": (net / n) if n else 0.0,
    }


# ------------------------------------------------------------------ אופטימיזציה
MAX_COMBOS = 400


def value_grid(lo, hi, step) -> list:
    try:
        lo, hi, step = float(lo), float(hi), float(step)
    except (TypeError, ValueError):
        return []
    if step <= 0 or hi < lo:
        return []
    vals = np.arange(lo, hi + step * 0.5, step)
    return [float(round(v, 6)) for v in vals]


def optimize(key: str, df: pd.DataFrame, common: dict, base_overrides: dict,
             grid: dict[str, list], split_frac: float = 0.7,
             max_combos: int = MAX_COMBOS) -> dict:
    """מריץ את המנוע על כל שילובי הפרמטרים ב-grid. הדירוג לפי החלק הראשון של
    ההיסטוריה (in-sample, split_frac), והחלק השני (out-of-sample) מוצג ליד כל
    שורה — כדי לראות אם התוצאה מחזיקה על נתונים שלא שימשו לבחירה."""
    names = list(grid.keys())
    combos = list(itertools.product(*[grid[n] for n in names]))
    truncated = len(combos) > max_combos
    combos = combos[:max_combos]
    cut = int(len(df) * float(split_frac))
    df_in, df_out = df.iloc[:cut], df.iloc[cut:]
    cash = float(common.get("cash", 100_000.0))
    rows = []
    t0 = time.time()
    for combo in combos:
        ov = dict(base_overrides)
        ov.update(dict(zip(names, combo)))
        try:
            s_in = quick_stats(run_engine(key, df_in, common, ov), cash)
            s_out = quick_stats(run_engine(key, df_out, common, ov), cash)
        except S.StrategyCompileError:
            continue
        rows.append({"params": dict(zip(names, combo)), "in": s_in, "out": s_out})
    rows.sort(key=lambda r: r["in"]["net"], reverse=True)
    return {"rows": rows, "names": names, "truncated": truncated, "n_combos": len(combos),
            "seconds": time.time() - t0, "in_from": str(df_in.index[0])[:10] if len(df_in) else "",
            "in_to": str(df_in.index[-1])[:10] if len(df_in) else "",
            "out_from": str(df_out.index[0])[:10] if len(df_out) else "",
            "out_to": str(df_out.index[-1])[:10] if len(df_out) else ""}


# ================================================================ שלב 2: טווח, היסטוריה, השוואה
from collections import deque
import itertools as _it

import plotly.graph_objects as go

RANGE_OPTIONS = [
    ("all", "כל ההיסטוריה"), ("1y", "שנה אחרונה"), ("6m", "6 חודשים"),
    ("3m", "3 חודשים"), ("1m", "חודש אחרון"), ("chart", "הטווח שבגרף"),
]
_RANGE_DAYS = {"1y": 365, "6m": 182, "3m": 91, "1m": 30}


def slice_range(df: pd.DataFrame, bt_range: str | None, window: dict | None,
                epoch_to_source=None) -> tuple[pd.DataFrame, str]:
    """חותך את ההיסטוריה לטווח שנבחר בסרגל הבקטסט. 'chart' = הטווח הנראה כרגע
    בגרף (window: {from,to} epochs במוסכמת התצוגה, מומרים עם epoch_to_source).
    מחזיר (df, תווית קריאה)."""
    if df is None or not len(df):
        return df, ""
    key = bt_range or "all"
    if key == "chart" and window and window.get("from") and window.get("to") and epoch_to_source:
        try:
            lo = epoch_to_source(int(window["from"]))
            hi = epoch_to_source(int(window["to"]))
            out = df[(df.index >= lo) & (df.index <= hi)]
            if len(out) > 10:
                return out, f"הטווח שבגרף ({lo:%d/%m/%y}–{hi:%d/%m/%y})"
        except Exception:
            pass
        key = "all"
    if key in _RANGE_DAYS:
        cutoff = df.index[-1] - pd.Timedelta(days=_RANGE_DAYS[key])
        return df[df.index >= cutoff], dict(RANGE_OPTIONS)[key]
    return df, "כל ההיסטוריה"


RUN_HISTORY: deque = deque(maxlen=40)
_run_ids = _it.count(1)
_KEEP_TRADES = 12   # רק הריצות האחרונות שומרות את טבלת העסקאות (לעקומת ההשוואה)
_HISTORY_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "run_history.json"


def _save_history() -> None:
    """שומר את ההיסטוריה (בלי טבלאות העסקאות) כדי שתשרוד הפעלה מחדש."""
    try:
        import json
        rows = [{k: v for k, v in r.items() if k != "trades"} for r in RUN_HISTORY]
        _HISTORY_PATH.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_history() -> None:
    try:
        import json
        if not _HISTORY_PATH.exists():
            return
        rows = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        for r in reversed(rows):
            r["trades"] = None
            RUN_HISTORY.appendleft(r)
        if rows:
            global _run_ids
            _run_ids = _it.count(max(r["id"] for r in rows) + 1)
    except Exception:
        pass


_load_history()


def record_run(*, key: str, symbol: str, timeframe: str, range_label: str,
               params: dict, stats: dict, cash: float) -> dict:
    rec = {
        "id": next(_run_ids), "at": pd.Timestamp.now().strftime("%d/%m %H:%M:%S"),
        "key": key, "title": ENGINES[key]["title"], "symbol": symbol, "timeframe": timeframe,
        "range": range_label, "params": {k: v for k, v in params.items() if k in {n for n, *_ in ENGINES[key]["params"]}},
        "cash": cash, "q": quick_stats(stats, cash),
        "bh": float(stats.get("Buy & Hold Return [%]", 0.0)),
        "trades": stats["_trades"][["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size", "PnL"]].copy(),
    }
    RUN_HISTORY.appendleft(rec)
    for old in list(RUN_HISTORY)[_KEEP_TRADES:]:
        old["trades"] = None
    _save_history()
    return rec


def get_run(rid: int):
    return next((r for r in RUN_HISTORY if r["id"] == rid), None)


def delete_run(rid: int):
    for r in list(RUN_HISTORY):
        if r["id"] == rid:
            RUN_HISTORY.remove(r)
    _save_history()


def run_code(rec: dict) -> str:
    """קוד האסטרטגיה של ריצה מההיסטוריה, עם הפרמטרים שלה כתובים בשורות ה-# input."""
    return apply_inputs_to_code(ENGINES[rec["key"]]["code"], rec["params"])


def compare_figure(recs: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = ["#2962FF", "#26A69A", "#FF9800", "#EF5350"]
    for i, r in enumerate(recs):
        tr = r.get("trades")
        if tr is None or tr.empty:
            continue
        o = tr.sort_values("ExitTime")
        x = o["ExitTime"].to_numpy(); y = np.cumsum(o["PnL"].to_numpy(dtype=float))
        if len(x) > 1500:
            idx = np.linspace(0, len(x) - 1, 1500).astype(int); x, y = x[idx], y[idx]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="\u202b" + f"{r['title']} · {r['timeframe']} · #{r['id']}" + "\u202c",
                                 line=dict(color=colors[i % len(colors)], width=1.6)))
    fig.update_layout(height=280, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#B2B5BE", size=11),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    fig.update_xaxes(gridcolor="#2A2E37", zeroline=False)
    fig.update_yaxes(gridcolor="#2A2E37", zeroline=True, zerolinecolor="#3a3f4a", tickprefix="$")
    return fig


# ================================================================ תוויות בעברית לפרמטרים
# שם ה-"# input" בקוד נשאר באנגלית (הוא מזהה), אבל בממשק מציגים תווית קריאה.
LABELS: dict[str, tuple[str, str]] = {
    # שם ה-input -> (שם באנגלית כמקובל בפלטפורמות מסחר, הסבר בעברית)
    "SMA_PERIOD": ("SMA Length", "אורך הממוצע הנע"), "SMA_LENGTH": ("SMA Length", "אורך הממוצע הנע"),
    "RSI_PERIOD": ("RSI Length", "אורך RSI"), "RSI_LEN": ("RSI Length", "אורך RSI"),
    "RSI_OVERSOLD": ("RSI Oversold", "רמת מכירת יתר"), "RSI_OVERBOUGHT": ("RSI Overbought", "רמת קניית יתר"),
    "HAMMER_RATIO": ("Wick / Body Ratio", "יחס פתיל לגוף"),
    "STOP_LOSS_PTS": ("Stop Loss (pts)", "סטופ בנקודות"), "TAKE_PROFIT_PTS": ("Take Profit (pts)", "יעד בנקודות"),
    "SWING_BARS": ("Swing Lookback", "נרות לזיהוי נקודת מפנה"), "MIN_LEG_PTS": ("Min Leg Size (pts)", "רגל מינימלית בנקודות"),
    "ENTRY_FIB": ("Entry Retracement", "רמת פיבונאצ'י לכניסה"), "STOP_FIB": ("Stop Retracement", "רמת פיבונאצ'י לסטופ"),
    "STOP_BUFFER_PTS": ("Stop Buffer (pts)", "מרווח סטופ בנקודות"), "TARGET_EXT": ("Target Extension", "הארכת פיבונאצ'י ליעד"),
    "RSI_CONFIRM": ("RSI Confirmation", "אישור RSI"), "RSI_LONG_MAX": ("Max RSI for Long", "RSI מקסימלי ללונג"),
    "RSI_SHORT_MIN": ("Min RSI for Short", "RSI מינימלי לשורט"), "CANDLE_CONFIRM": ("Candle Confirmation", "אישור נר"),
    "EMA_TREND": ("Trend EMA Length", "EMA למגמה, 0 = כבוי"), "ALLOW_SHORT": ("Allow Shorts", "לאפשר שורט"),
    "RANGE_MINUTES": ("Opening Range (min)", "דקות טווח הפתיחה"), "BUFFER_PTS": ("Breakout Buffer (pts)", "מרווח פריצה"),
    "STOP_PTS": ("Stop Loss (pts)", "סטופ בנקודות"), "TARGET_RR": ("Reward / Risk", "יחס יעד לסיכון"),
    "MIN_RANGE_PTS": ("Min Range (pts)", "טווח מינימלי"), "MAX_RANGE_PTS": ("Max Range (pts)", "טווח מקסימלי"),
    "LAST_ENTRY_MIN": ("Last Entry (min)", "כניסה אחרונה, דקות מהפתיחה"),
    "DEVIATION_PTS": ("VWAP Deviation (pts)", "מרחק מ-VWAP"), "RTH_ONLY": ("RTH Only", "סשן רגיל בלבד"),
    "RANGE_BARS": ("Range Lookback", "נרות לזיהוי הטווח"), "RANGE_ATR_MULT": ("Max Range Width (×ATR)", "רוחב טווח מקסימלי"),
    "SPRING_PEN_PTS": ("Spring Penetration (pts)", "חדירה מינימלית מתחת לתמיכה"), "VOL_MAX_RATIO": ("Max Volume (×avg)", "נפח מקסימלי ביחס לממוצע"),
    "CONFIRM_NEXT": ("Next-Bar Confirmation", "אישור בנר הבא"), "TARGET_FRAC": ("Target (× range)", "יעד כחלק מרוחב הטווח"),
    "MAX_BARS": ("Max Bars in Trade", "מקסימום נרות בעסקה"),
}


def param_en(name: str) -> str:
    return LABELS.get(name, (name.replace("_", " ").title(), ""))[0]


def param_he(name: str) -> str:
    return LABELS.get(name, ("", ""))[1]


def param_label(name: str) -> str:
    """'English (עברית)' — כמו בפלטפורמות מסחר, עם ההסבר בעברית בסוגריים."""
    en, he = LABELS.get(name, (name.replace("_", " ").title(), ""))
    return f"{en} ({he})" if he else en


def fmt_value(v) -> str:
    if isinstance(v, bool):
        return "כן" if v else "לא"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ================================================================ מונטה קרלו
def monte_carlo(trades: pd.DataFrame, cash: float, n_sims: int = 1000, seed: int = 7,
                bootstrap: bool = True, max_paths: int = 60) -> dict:
    """מערבב את סדר העסקאות (או דוגם מחדש עם החזרה) n_sims פעמים ומחזיר את
    התפלגות הרווח הסופי וה-Drawdown המקסימלי. התוצאה: אחוזונים, הסתברות
    לרווח, ומדגם של מסלולי הון לציור."""
    pnl = trades["PnL"].to_numpy(dtype=float)
    n = len(pnl)
    if n < 5:
        return {"n": n, "ok": False}
    rng = np.random.default_rng(seed)
    finals = np.empty(n_sims); dds = np.empty(n_sims); dd_pcts = np.empty(n_sims)
    paths = []
    step = max(1, n // 400)
    for i in range(n_sims):
        sample = rng.choice(pnl, size=n, replace=True) if bootstrap else rng.permutation(pnl)
        cum = np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        dd = cum - peak
        finals[i] = cum[-1]; dds[i] = dd.min(); dd_pcts[i] = (dd / (cash + peak)).min() * 100
        if len(paths) < max_paths:
            paths.append(cum[::step])
    actual_cum = np.cumsum(pnl)
    actual_peak = np.maximum.accumulate(np.concatenate([[0.0], actual_cum]))[1:]
    actual_dd = float((actual_cum - actual_peak).min())
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "n": n, "ok": True, "n_sims": n_sims, "bootstrap": bootstrap,
        "final": {"p5": q(finals, 5), "p25": q(finals, 25), "p50": q(finals, 50), "p75": q(finals, 75), "p95": q(finals, 95),
                  "mean": float(finals.mean()), "actual": float(actual_cum[-1])},
        "dd": {"p5": q(dds, 5), "p50": q(dds, 50), "p95": q(dds, 95), "worst": float(dds.min()), "actual": actual_dd,
               "p95_pct": q(dd_pcts, 5)},
        "p_profit": float((finals > 0).mean() * 100),
        "p_dd_worse_than_actual": float((dds < actual_dd).mean() * 100),
        "risk_of_ruin_20": float((dd_pcts <= -20).mean() * 100),
        "paths": np.array(paths), "path_x": np.arange(0, n, step)[:len(paths[0])] if paths else np.array([]),
        "actual_path": actual_cum[::step],
    }


# ================================================================ Walk-forward
def walk_forward(key: str, df: pd.DataFrame, common: dict, base_overrides: dict,
                 grid: dict[str, list], n_windows: int = 5, train_frac: float = 0.7,
                 rank_by: str = "net", max_combos: int = MAX_COMBOS) -> dict:
    """אופטימיזציה מתגלגלת: ההיסטוריה נחתכת ל-n_windows חלונות רצופים; בכל חלון
    בוחרים את הפרמטרים הטובים ביותר על החלק הראשון (train) ובודקים אותם על
    החלק שאחריו (test). סך ה-test הוא ביצוע "כפי שהיה קורה בפועל" למי שמכוונן
    מחדש כל תקופה. rank_by: 'net' או 'pf'."""
    names = list(grid.keys())
    combos = list(itertools.product(*[grid[n] for n in names]))[:max_combos]
    n = len(df)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    cash = float(common.get("cash", 100_000.0))
    windows = []
    oos_trades = []
    t0 = time.time()
    for w in range(n_windows):
        seg = df.iloc[edges[w]:edges[w + 1]]
        cut = int(len(seg) * train_frac)
        train, test = seg.iloc[:cut], seg.iloc[cut:]
        if len(train) < 200 or len(test) < 50:
            continue
        best = None
        for combo in combos:
            ov = dict(base_overrides); ov.update(dict(zip(names, combo)))
            try:
                st = quick_stats(run_engine(key, train, common, ov), cash)
            except S.StrategyCompileError:
                continue
            score = st["net"] if rank_by == "net" else (st["pf"] if np.isfinite(st["pf"]) else -1)
            if st["trades"] < 3:
                score = -1e18
            if best is None or score > best[0]:
                best = (score, dict(zip(names, combo)), st)
        if best is None:
            continue
        ov = dict(base_overrides); ov.update(best[1])
        try:
            res_test = run_engine(key, test, common, ov)
        except S.StrategyCompileError:
            continue
        st_test = quick_stats(res_test, cash)
        oos_trades.append(res_test["_trades"])
        windows.append({
            "window": w + 1, "train_from": str(train.index[0])[:10], "train_to": str(train.index[-1])[:10],
            "test_from": str(test.index[0])[:10], "test_to": str(test.index[-1])[:10],
            "params": best[1], "train": best[2], "test": st_test,
        })
    all_oos = pd.concat(oos_trades) if oos_trades else pd.DataFrame(columns=["ExitTime", "PnL"])
    oos_net = float(all_oos["PnL"].sum()) if len(all_oos) else 0.0
    # יציבות פרמטרים: כמה חלונות בחרו את אותו ערך לכל פרמטר
    stability = {}
    for nm in names:
        vals = [w["params"][nm] for w in windows]
        if vals:
            top = max(set(vals), key=vals.count)
            stability[nm] = {"mode": top, "share": vals.count(top) / len(vals) * 100}
    return {
        "windows": windows, "names": names, "seconds": time.time() - t0,
        "oos_net": oos_net, "oos_trades": int(len(all_oos)),
        "oos_pf": (lambda p: (p[p > 0].sum() / abs(p[p <= 0].sum())) if (p <= 0).any() and abs(p[p <= 0].sum()) > 0 else float("inf"))(all_oos["PnL"].to_numpy(dtype=float)) if len(all_oos) else 0.0,
        "profitable_windows": sum(1 for w in windows if w["test"]["net"] > 0),
        "stability": stability, "oos_curve": all_oos.sort_values("ExitTime") if len(all_oos) else None,
    }
