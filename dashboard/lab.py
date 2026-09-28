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
        "code": S.DEFAULT_CODE,
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
    return ENGINES[key]["runner"](df, **common, **build_kwargs(key, overrides))


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
