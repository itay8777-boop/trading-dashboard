"""מנוע האסטרטגיה: קומפילציה של קוד שהמשתמש כותב והרצתו כבקטסט.

המשתמש כותב שתי פונקציות — init/next — בדיוק כמו ב-Strategy Tester של TradingView,
והן נעטפות למחלקת Strategy של backtesting.py. הבחירה ב-backtesting.py היא מכוונת:
היא מפיקה בדיוק את מערך הסטטיסטיקות ש-ui_components יודע להציג (Sharpe, Sortino,
Buy & Hold, Max Drawdown), כך שפאנל הביצועים כאן זהה לזה שבאפליקציה הקיימת.
"""
from __future__ import annotations

import ast
import re
import warnings

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from .data import to_display_naive

# שורות בקוד ה-Pine Editor שמסתיימות ב-"# input" נחשפות כשדות עריכה בלשונית
# Settings ("Strategy Inputs" - ראו app.py: strategy-inputs-container).
# ast.literal_eval (לא eval) על צד ימין ה-"=" - בטוח, לא מריץ קוד שרירותי, רק
# מפרש literal פשוט (מספר/בוליאני). שורות עם ביטוי מורכב מדי לפירוש-literal
# (משתנה אחר, קריאת פונקציה) פשוט מדולגות בשקט - לא כל "# input" חייב להיות נתמך.
_INPUT_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*#\s*input\s*$")

# מנועי הבקטסט המהירים (run_fast_sma_crossover_backtest, run_hammer_rsi_backtest
# למטה) הם וקטוריים-קבועים-מראש - לא מריצים את קוד ה-Pine Editor בפועל (ראו
# app.py: run_or_switch, שבוחר איזה מהם להריץ לפי detect_builtin_strategy).
# לכן רק שמות-משתנה שמצביעים במפורש על פרמטר אמיתי של אחד המנועים באמת
# משפיעים על התוצאה; כל שאר ה-"# input" שמזוהים עדיין מוצגים בממשק (שקיפות:
# המשתמש רואה מה קיים בקוד), אבל בלי אפקט עד שמנוע מהיר יתמוך בהם. אליאסים
# נפוצים בלבד - לא ניחוש חופשי לפי שם, כדי לא "לזייף" תמיכה שלא קיימת.
RECOGNIZED_STRATEGY_INPUTS = {
    "SMA_LENGTH": "sma_period",
    "SMA_PERIOD": "sma_period",
    "RSI_PERIOD": "rsi_period",
    "RSI_OVERSOLD": "rsi_oversold",
    "RSI_OVERBOUGHT": "rsi_overbought",
    "HAMMER_RATIO": "hammer_ratio",
    "USE_RSI": "use_rsi",
    "STOP_LOSS_PTS": "stop_loss_pts",
    "MAX_BODY_PTS": "max_body_pts",
    "MIN_STD_BEFORE": "min_std_before",
    "EXCLUDE_OPEN_HOURS": "exclude_open_hours",
    "BREAK_EVEN_PTS": "break_even_pts",
    "USE_BREAK_EVEN": "use_break_even",
    "PEAK_DISTANCE_PTS": "peak_distance_pts",
    "EXIT_DISTANCE_PTS": "exit_distance_pts",
    "MIN_DISTANCE_PTS": "min_distance_pts",
    "STREAK_LENGTH": "streak_length",
    "HOLD_BARS": "hold_bars",
    "USE_PRIOR_LOW_EXIT": "use_prior_low_exit",
    "USE_STOP_LOSS": "use_stop_loss",
    "USE_VOLATILITY_FILTER": "use_volatility_filter",
    "VOLATILITY_THRESHOLD": "volatility_threshold",
    "POC_POSITION_MAX": "poc_position_max",
    "TAKE_PROFIT_PTS": "take_profit_pts",
    # fib_golden (run_fib_golden_backtest)
    "SWING_BARS": "swing_bars",
    "MIN_LEG_PTS": "min_leg_pts",
    "ENTRY_FIB": "entry_fib",
    "STOP_FIB": "stop_fib",
    "STOP_BUFFER_PTS": "stop_buffer_pts",
    "TARGET_EXT": "target_ext",
    "RSI_CONFIRM": "rsi_confirm",
    "RSI_LEN": "rsi_len",
    "RSI_LONG_MAX": "rsi_long_max",
    "RSI_SHORT_MIN": "rsi_short_min",
    "CANDLE_CONFIRM": "candle_confirm",
    "EMA_TREND": "ema_trend",
    "ALLOW_SHORT": "allow_short",
}


def parse_strategy_inputs(code: str) -> list[dict]:
    """סורקת את קוד ה-Pine Editor ומחזירה את כל משתני ה-"# input" שזוהו, לפי
    סדר הופעתם בקוד. כל פריט: {"name", "value", "type" ("int"/"float"/"bool"),
    "recognized" (True אם יש לו אפקט אמיתי על run_fast_sma_crossover_backtest -
    ראו RECOGNIZED_STRATEGY_INPUTS)}. שמות כפולים - האחרון בקוד מנצח (תואם
    לאיך שפייתון עצמו היה מתנהג: הקצאה חוזרת דורסת את הקודמת)."""
    seen: dict[str, dict] = {}
    for line in (code or "").splitlines():
        m = _INPUT_LINE_RE.match(line)
        if not m:
            continue
        name, raw_value = m.group(1), m.group(2)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            continue  # ביטוי לא-literal (משתנה/קריאת פונקציה) - לא נתמך, מדולג
        # bool הוא תת-מחלקה של int בפייתון - הבדיקה חייבת לבוא קודם, אחרת
        # True/False היו מסווגים בטעות כ-int.
        if isinstance(value, bool):
            type_name = "bool"
        elif isinstance(value, int):
            type_name = "int"
        elif isinstance(value, float):
            type_name = "float"
        else:
            continue  # מחרוזת/רשימה/וכו' - אין עדיין ווידג'ט מתאים לזה
        seen[name] = {"name": name, "value": value, "type": type_name,
                      "recognized": name in RECOGNIZED_STRATEGY_INPUTS}
    return list(seen.values())


class StrategyCompileError(Exception):
    """שגיאה בקוד האסטרטגיה — מוצגת למשתמש כמו שהיא."""


# ------------------------------------------------------------------ אינדיקטורים
# כולם מקבלים מערך ומחזירים מערך, כדי שיעבדו ישירות עם self.I(...) בקוד המשתמש.
def sma(values, n):
    return pd.Series(values).rolling(int(n)).mean().to_numpy()


def ema(values, n):
    return pd.Series(values).ewm(span=int(n), adjust=False).mean().to_numpy()


def rsi(values, n=14):
    s = pd.Series(values)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / int(n), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / int(n), adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def session_vwap(df: pd.DataFrame) -> np.ndarray:
    """VWAP מצטבר שמתאפס בכל יום מסחר — ההגדרה שבה משתמש TradingView.

    VWAP שרץ ברצף על פני שבועות מפסיק להיות "המחיר הממוצע של הסשן" ונגרר אחרי
    ההיסטוריה, כך שהסיגנלים שנגזרים ממנו סוטים משמעותית. לכן האיפוס היומי כאן.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index, index=df.index).dt.normalize()
    cum_pv = (typical * df["Volume"]).groupby(day).cumsum()
    cum_v = df["Volume"].groupby(day).cumsum()
    return (cum_pv / cum_v.replace(0, np.nan)).bfill().to_numpy()


DEFAULT_CODE = """\
def init(self):
    # זמינים אוטומטית: self.vwap (מתאפס כל יום), self.position_size
    # אינדיקטורים: sma / ema / rsi  —  למשל  self.I(sma, self.data.Close, 20)
    self.ma = self.I(sma, self.data.Close, 20)

def next(self):
    # כניסה כשהמחיר חוצה מעל ה-VWAP, יציאה כשהוא חוצה מתחתיו
    if not self.position and self.data.Close[-1] > self.vwap[-1]:
        self.buy(size=self.position_size)
    elif self.position and self.data.Close[-1] < self.vwap[-1]:
        self.position.close()
"""


def _compile(code: str):
    ns: dict = {"pd": pd, "np": np, "crossover": crossover,
                "sma": sma, "ema": ema, "rsi": rsi}
    try:
        exec(code, ns)
    except Exception as exc:
        raise StrategyCompileError(f"שגיאה בקוד: {exc}") from exc

    user_init, user_next = ns.get("init"), ns.get("next")
    if not callable(user_init) or not callable(user_next):
        raise StrategyCompileError(
            "הקוד חייב להגדיר בדיוק שתי פונקציות: def init(self) ו-def next(self)"
        )
    return user_init, user_next


def build_strategy_class(code: str, position_size: int):
    user_init, user_next = _compile(code)

    class UserStrategy(Strategy):
        _size = position_size

        def init(self):
            vwap = session_vwap(self.data.df)
            self.vwap = self.I(lambda: vwap, name="VWAP", overlay=True)
            self.position_size = self._size
            try:
                user_init(self)
            except Exception as exc:
                raise StrategyCompileError(f"שגיאה ב-init: {exc}") from exc

        def next(self):
            try:
                user_next(self)
            except Exception as exc:
                raise StrategyCompileError(f"שגיאה ב-next: {exc}") from exc

    return UserStrategy


def run_backtest(df: pd.DataFrame, code: str, *, cash: float = 100_000.0,
                 size: int = 100, commission_pct: float = 0.0):
    """מריץ את האסטרטגיה ומחזיר את אובייקט הסטטיסטיקות של backtesting.py."""
    if df is None or len(df) < 2:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)
    bt = Backtest(df, build_strategy_class(code, int(size)),
                  cash=float(cash), commission=float(commission_pct) / 100.0,
                  exclusive_orders=True)
    try:
        # backtesting.py מזהיר (warnings.warn) על כל הזמנה שנדחתה בגלל מרווח בטחון
        # לא מספיק - על היסטוריה של מיליוני ברים (למשל 7 שנים של MNQ ב-1m) ו-size
        # שגדול מדי ביחס להון, זה קרה כמעט על כל בר בנפרד: מיליוני אזהרות שהציפו
        # את הלוג ו(warnings module overhead) האטו את הריצה עצמה משמעותית. האזהרה
        # לא מעשית ברמת-בר בודד - הסטטיסטיקות הסופיות מ-bt.run() כבר משקפות נכון
        # כמה הזמנות נדחו.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return bt.run()
    except StrategyCompileError:
        raise
    except Exception as exc:
        raise StrategyCompileError(f"שגיאה בהרצה: {exc}") from exc


# גודל טיק סטנדרטי של MNQ (Micro E-mini Nasdaq-100): 0.25 נקודה. משמש להמרת
# "סליפג' בטיקים" (מה שהמשתמש מזין) למחיר בפועל. לא מוצג/ניתן לעריכה בממשק -
# המשתמש ביקש רק שדה סליפג' ב"טיקים", לא גודל-טיק לכל סימבול; אם ירצה לסחור
# בחוזה אחר עם גודל טיק שונה זה יצטרך הרחבה נפרדת.
MNQ_TICK_SIZE = 0.25


def run_fast_sma_crossover_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                    size: float = 1.0, commission_pct: float = 0.0,
                                    sma_period: int = 20,
                                    multiplier: float = 1.0,
                                    slippage_ticks: float = 0.0,
                                    commission_per_contract: float = 0.0,
                                    market_order: bool = True,
                                    tick_size: float = MNQ_TICK_SIZE,
                                    exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    """מנוע וקטורי (numpy/pandas) מיוחד לאסטרטגיית חציית SMA20 - לא לולאת בר-אחר-בר
    כמו run_backtest/backtesting.py. משמש את "Full backtest" (2.5 מיליון ברים):
    profiling מדד ש-backtesting.py, בגלל חשבונאות פר-בר (הזמנות/מרווח/equity),
    לוקח כ-12,500 ברים/שנייה בלבד - כ-3.4 דקות על 7 שנות MNQ ב-1m, בלי קשר
    לכמות הליבה של האסטרטגיה עצמה. וקטוריזציה מלאה אפשרית רק לאסטרטגיה קבועה
    מראש (לא קוד שרירותי של המשתמש) - זו הסיבה שזה מנוע נפרד, לא תחליף כללי.

    שלבים:
    1. SMA על כל המערך בבת אחת (pandas.rolling) - לא בר-אחר-בר.
    2. חציות (כניסה/יציאה) כמסכות בוליאניות וקטוריות על כל המערך.
    3. בניית העסקאות עצמן היא כן לולאה, אבל רק על אירועי-חצייה (בדרך כלל אלפים
       בודדים, לא מיליונים) - זניח בזמן ריצה לעומת השלב הראשון.

    עלויות ביצוע (לשונית Settings בממשק):
    - multiplier: ערך-הנקודה של החוזה — תנועה של נקודה אחת שווה size*multiplier
      דולר (למשל MNQ: 1 נקודה = 2$ לחוזה). בלי זה PnL היה בנקודות מחיר גולמיות,
      לא בדולרים אמיתיים.
    - slippage_ticks: מוחסר ממחיר הכניסה ומתווסף-בהיפוך למחיר היציאה (בכיוון
      הגרוע ביותר עבור long) — לא מוחל אם market_order=False (הזמנת Limit
      ממלאת במחיר המבוקש בדיוק, בהגדרה, ולכן בלי סליפג').
    - commission_per_contract: סכום קבוע לחוזה, נגבה פעמיים לכל עסקה (כניסה
      ויציאה) — לא אחוז מערך העסקה כמו commission_pct הישן.

    מחזירה dict בפורמט תואם בדיוק למה ש-panels.py/ui_components.py/to_markers/
    build_performance_chart כבר יודעים לצייר (Equity Final [$], Return [%],
    Max. Drawdown [%], Buy & Hold Return [%], _trades עם EntryTime/ExitTime/
    EntryPrice/ExitPrice/Size/PnL/ReturnPct) - בלי צורך לשנות אותם בכלל.
    """
    if df is None or len(df) < sma_period + 2:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    sma = df["Close"].rolling(sma_period).mean().to_numpy(dtype=float)

    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    prev_sma = np.empty(n)
    prev_sma[0] = np.nan
    prev_sma[1:] = sma[:-1]

    valid = ~np.isnan(sma) & ~np.isnan(prev_sma)
    entry_signal = valid & (prev_close < prev_sma) & (close > sma)
    exit_signal = valid & (prev_close > prev_sma) & (close < sma)

    entry_idx = np.flatnonzero(entry_signal)
    exit_idx = np.flatnonzero(exit_signal)

    # ממזגים את שני מערכי-האירועים לפי סדר זמן (עדיין וקטורי - concatenate+argsort
    # על כמה אלפי אינדקסים, לא על מיליוני ברים).
    all_idx = np.concatenate([entry_idx, exit_idx])
    is_entry = np.concatenate([
        np.ones(len(entry_idx), dtype=bool), np.zeros(len(exit_idx), dtype=bool),
    ])
    order = np.argsort(all_idx, kind="stable")
    all_idx = all_idx[order]
    is_entry = is_entry[order]

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    size = float(size)

    # מחיר/זמן המילוי: לא הבר שבו הזוהה הסימן, אלא הבר *הבא* אחריו (open שלו) -
    # תואם את trade_on_close=False (ברירת המחדל ב-run_backtest/backtesting.py):
    # הזמנה שמוגשת תוך next() על סמך נתוני הבר הנוכחי מתמלאת בפתיחת הבר הבא,
    # לא בסגירת הבר שעליו התקבלה ההחלטה. בלי זה קיבלנו את מספר העסקאות הנכון
    # (156,113 - החצייה עצמה כן וקטורית ונכונה) אבל PnL שגוי לגמרי (3,706$
    # במקום 15,811$ שמדד run_backtest על אותה אסטרטגיה בדיוק).
    def _fill(i: int) -> int | None:
        return i + 1 if i + 1 < n else None

    # הלולאה היחידה כאן: רק על אירועי-חצייה (כניסה+יציאה יחד, בדרך כלל אלפים
    # בודדים גם על 7 שנים) - לא על כל בר. "long-only, עמדה אחת בכל רגע" בדיוק
    # כמו exclusive_orders=True שהיה ב-run_backtest.
    entry_times, exit_times, entry_prices, exit_prices = [], [], [], []
    entry_fill_idxs, exit_fill_idxs = [], []
    in_position = False
    pending_fill_i = 0
    for i, entry_flag in zip(all_idx.tolist(), is_entry.tolist()):
        fill_i = _fill(i)
        if fill_i is None:
            continue  # סימן על הבר האחרון - אין בר הבא שבו למלא את ההזמנה
        if entry_flag and not in_position:
            pending_fill_i = fill_i
            in_position = True
        elif not entry_flag and in_position:
            entry_times.append(times[pending_fill_i])
            exit_times.append(times[fill_i])
            entry_prices.append(open_[pending_fill_i])
            exit_prices.append(open_[fill_i])
            entry_fill_idxs.append(pending_fill_i)
            exit_fill_idxs.append(fill_i)
            in_position = False

    n_trades = len(entry_times)
    raw_entry_prices_arr = np.asarray(entry_prices, dtype=float)
    raw_exit_prices_arr = np.asarray(exit_prices, dtype=float)

    # סליפג': מוחל רק על הזמנת Market (הזמנת Limit ממלאת במחיר המבוקש בהגדרה,
    # אין לה סליפג'). מחיר-כניסה גרוע יותר (גבוה יותר ל-long), מחיר-יציאה גרוע
    # יותר (נמוך יותר) - בדיוק כיוון ההפסד הצפוי מהחלקה בביצוע אמיתי.
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    entry_prices_arr = raw_entry_prices_arr + slippage_amount
    exit_prices_arr = raw_exit_prices_arr - slippage_amount

    # PnL בדולרים אמיתיים: תנועת מחיר (אחרי סליפג') כפול size כפול ערך-הנקודה
    # של החוזה (multiplier) - למשל על MNQ, נקודה אחת = 2$ לחוזה.
    pnl = (exit_prices_arr - entry_prices_arr) * size * multiplier

    # עמלה קבועה לחוזה, גבייה בשני הצדדים (כניסה+יציאה) - "2 commissions" לעסקה.
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    if commission_pct:
        # תאימות לאחור: עמלה כאחוז מערך העסקה (על המחירים הגולמיים, לפני
        # סליפג') - עקבי עם commission ב-Backtest(...) של backtesting.py.
        # ברירת המחדל היא 0, ולא מוצג יותר בממשק (הוחלף ב-Settings), אבל
        # נשאר נתמך אם קורא אחר לפונקציה מעביר אותו.
        pnl = pnl - (raw_entry_prices_arr + raw_exit_prices_arr) * size * (commission_pct / 100.0)

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    )

    # Equity מלא לכל בר (mark-to-market על עמדה פתוחה, לא רק ברגע הסגירה) - כדי
    # ש-Max Drawdown ישקף גם נפילה זמנית בזמן שעמדה עדיין פתוחה, לא רק את
    # התוצאה הסופית של כל עסקה. עדיין וקטורי לגמרי (בלי לולאת בר-אחר-בר):
    # position_flag בנוי מ-cumsum של +1/-1 בנקודות המילוי, ומחיר-הכניסה "נגרר
    # קדימה" (ffill) מנקודת הכניסה - position_flag==0 מאפס את התרומה בזמן שטוח,
    # כך שערך ה-ffill הישן/לא-רלוונטי בין עסקאות לא משפיע על התוצאה.
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_fill_idxs, dtype=np.int64)
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    # ffill של מחיר-הכניסה *האפקטיבי* (אחרי סליפג') - זה המחיר שבו העמדה באמת
    # נפתחה, ולכן ה-mark-to-market בזמן שהיא פתוחה חייב להיגזר ממנו, לא מהמחיר
    # הגולמי לפני סליפג'.
    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag > 0, (close - entry_price_ffilled) * size * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": np.full(n_trades, size),  # תמיד long - כניסה רק על חצייה כלפי מעלה
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ============================================================ sma_cross
# תבנית ברירת המחדל של חציית ממוצע נע - אותו מנוע מהיר (run_fast_sma_crossover_backtest),
# עם הפרמטר חשוף כ-"# input" כדי שיופיע בבודק כמו בשאר האסטרטגיות.
SMA_CODE = """\
# strategy: sma_cross
# Simple moving-average crossover (long only).
# Entry: close crosses above the SMA. Exit: close crosses back below it.
SMA_PERIOD = 20        # input
"""


# ============================================================ hammer_rsi
# מנוע מובנה שני, נבחר ב-app.py:run_or_switch לפי שורת-הסימון הראשונה בקוד
# (# strategy: hammer_rsi - ראו detect_builtin_strategy למטה) - לא הרצת קוד
# שרירותי, בדיוק כמו run_fast_sma_crossover_backtest. קוד בלי סימון (למשל
# DEFAULT_CODE, או אסטרטגיה שנשמרה לפני התכונה הזו) ממשיך להריץ את מנוע
# חציית-ה-SMA כרגיל - תאימות לאחור מלאה.
_BUILTIN_STRATEGY_RE = re.compile(r"^#\s*strategy:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


def detect_builtin_strategy(code: str) -> str | None:
    """קוראת רק את השורה הראשונה הלא-ריקה בקוד (לא מריצה כלום) ומחפשת סימון
    "# strategy: <key>". None אם אין סימון כזה - ברירת המחדל (חציית SMA)."""
    for line in (code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _BUILTIN_STRATEGY_RE.match(stripped)
        return m.group(1) if m else None
    return None


HAMMER_RSI_CODE = """\
# strategy: hammer_rsi
# Hammer / Inverted-Hammer reversal, optionally filtered by RSI extremes.
# Long:  current bar is a hammer          (AND RSI(prev bar) < RSI_OVERSOLD   if USE_RSI)
# Short: current bar is an inverted hammer (AND RSI(prev bar) > RSI_OVERBOUGHT if USE_RSI)
# Exit:  stop-loss or take-profit, whichever the bar's high/low touches first.
USE_RSI = True         # input
RSI_PERIOD = 14        # input
RSI_OVERSOLD = 30      # input
RSI_OVERBOUGHT = 70    # input
HAMMER_RATIO = 2.0     # input
STOP_LOSS_PTS = 50     # input
TAKE_PROFIT_PTS = 100  # input
"""


def run_hammer_rsi_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                            size: float = 1.0,
                            multiplier: float = 1.0,
                            slippage_ticks: float = 0.0,
                            commission_per_contract: float = 0.0,
                            market_order: bool = True,
                            tick_size: float = MNQ_TICK_SIZE,
                            rsi_period: int = 14,
                            rsi_oversold: float = 30.0,
                            rsi_overbought: float = 70.0,
                            hammer_ratio: float = 2.0,
                            stop_loss_pts: float = 50.0,
                            take_profit_pts: float = 100.0,
                            use_rsi: bool = True,
                            exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    """היפוך-מגמה וקטורי (long *ו-*short, בניגוד ל-run_fast_sma_crossover_backtest
    שהוא long-only): נר פטיש/פטיש-הפוך מסונן ע"י קיצון RSI. יציאה בסטופ-לוס/
    טייק-פרופיט - הראשון שנוגע בו high/low בפועל של הבר, לא רק close.

    שלבים (כולם וקטוריים על כל המערך בבת אחת, בלי שום לולאת-בר):
    1. RSI (Wilder, EWM) על כל הסגירות.
    2. זיהוי פטיש/פטיש-הפוך: גוף (body), פתיל עליון/תחתון ומיקום הגוף בתוך
       טווח הבר (high-low) - כולם השוואות איברים-מול-איברים על מערכי numpy.
    3. אירועי-כניסה (long/short) ממוזגים למערך אחד ממוין-זמן (concatenate+
       argsort), בדיוק כמו run_fast_sma_crossover_backtest.

    הלולאה היחידה כאן היא על אירועי-כניסה בלבד (לא על כל בר) - לכל אירוע,
    חיפוש הבר-הראשון-שנוגע-בסף מתבצע וקטורית (מסכה בוליאנית + argmax על פרוסת
    numpy, לא לולאת-פייתון פר-בר). מכיוון שעסקאות בלעדיות (לא חופפות), סך כל
    העבודה על פני *כל* העסקאות מצטברת ל-O(n) בסך הכול - כל בר "נסרק" לכל
    היותר פעם אחת, כחלק מחיפוש-היציאה של עסקה פתוחה אחת. זה מה שמאפשר לסיים
    2.5 מיליון ברים תוך שניות בודדות, לא תוך לולאת בר-אחר-בר.
    """
    if df is None or len(df) < rsi_period + 2:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    # --- RSI (Wilder) על הבר הקודם - אותה נוסחה בדיוק כמו rsi() למעלה, רק בלי
    # לעטוף שוב ב-DataFrame נפרד. prev_rsi[i] = rsi של close[i-1].
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_arr = (100 - 100 / (1 + rs)).fillna(50).to_numpy()
    prev_rsi = np.empty(n)
    prev_rsi[0] = np.nan
    prev_rsi[1:] = rsi_arr[:-1]

    # --- זיהוי נר פטיש / פטיש-הפוך (וקטורי, כל הנרות בבת אחת) ---
    body = np.abs(close - open_)
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    upper_wick = high - body_top
    lower_wick = body_bottom - low
    rng = high - low

    is_hammer = (
        (rng > 0)
        & (lower_wick >= hammer_ratio * body)
        & (upper_wick < 0.5 * body)
        & (body_bottom >= low + (2.0 / 3.0) * rng)
    )
    is_inv_hammer = (
        (rng > 0)
        & (upper_wick >= hammer_ratio * body)
        & (lower_wick < 0.5 * body)
        & (body_top <= low + (1.0 / 3.0) * rng)
    )

    # USE_RSI כבוי = פטיש טהור, בלי פילטר מומנטום (להשוואה ישירה מול מנועי הפטיש של איתי)
    if use_rsi:
        long_signal = is_hammer & (prev_rsi < rsi_oversold)
        short_signal = is_inv_hammer & (prev_rsi > rsi_overbought)
    else:
        long_signal = is_hammer.copy()
        short_signal = is_inv_hammer.copy()

    long_idx = np.flatnonzero(long_signal)
    short_idx = np.flatnonzero(short_signal)
    all_idx = np.concatenate([long_idx, short_idx])
    is_long_arr = np.concatenate([
        np.ones(len(long_idx), dtype=bool), np.zeros(len(short_idx), dtype=bool),
    ])
    order = np.argsort(all_idx, kind="stable")
    all_idx = all_idx[order]
    is_long_arr = is_long_arr[order]

    # סליפג' רק על הכניסה (הזמנת Market) - בדיוק כמו run_fast_sma_crossover_backtest;
    # היציאות עצמן (סטופ/טייק) כבר מוגדרות כמחיר-סף קבוע, לא הזמנת Market.
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    mode = fill_mode if fill_mode in FILL_MODES else "realistic"
    x_slip = slippage_amount if exit_slippage_ticks is None else float(exit_slippage_ticks) * float(tick_size)
    if mode == "worst":
        x_slip *= 2.0
    tp_pad = float(tick_size) if mode == "worst" else 0.0

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []
    sl_out, tp_out = [], []

    n_events = len(all_idx)
    i_ptr = 0
    while i_ptr < n_events:
        sig_i = int(all_idx[i_ptr])
        is_long = bool(is_long_arr[i_ptr])
        fill_i = sig_i + 1
        if fill_i >= n:
            break  # סימן על הבר האחרון - אין בר הבא למלא בו את הכניסה

        raw_entry = open_[fill_i]
        entry_price = raw_entry + slippage_amount if is_long else raw_entry - slippage_amount
        if is_long:
            tp_level = entry_price + take_profit_pts
            sl_level = entry_price - stop_loss_pts
        else:
            tp_level = entry_price - take_profit_pts
            sl_level = entry_price + stop_loss_pts

        # חיפוש וקטורי (לא לולאת-בר) של הבר הראשון *אחרי* הכניסה שבו high/low
        # נוגע ב-SL או ב-TP.
        search_high = high[fill_i + 1:]
        search_low = low[fill_i + 1:]
        if is_long:
            hit_tp = search_high >= tp_level + tp_pad
            hit_sl = search_low <= sl_level
        else:
            hit_tp = search_low <= tp_level - tp_pad
            hit_sl = search_high >= sl_level
        hit_any = hit_tp | hit_sl

        if not hit_any.any():
            # לא נסגר עד סוף הדאטה - סוגרים על הסגירה האחרונה הידועה (כמו
            # backtesting.py: פוזיציה פתוחה בסוף הריצה נסגרת, לא נעלמת).
            exit_i = n - 1
            exit_price = _exit_fill(is_long=is_long, kind="forced", level=float(close[exit_i]), bar_open=float(open_[exit_i]),
                                    bar_close=float(close[exit_i]), slip=x_slip, mode=mode)
        else:
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            # שני הרמות נחצו באותו בר - מניחים תרחיש פסימי (סטופ) כברירת מחדל
            # שמרנית, כי לא ידוע איזה מהם נגע קודם בפועל בתוך הבר.
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level
            kind = "sl" if sl_hit_here else "tp"
            exit_price = _exit_fill(is_long=is_long, kind=kind, level=(sl_level if kind == "sl" else tp_level),
                                    bar_open=float(open_[exit_i]), bar_close=float(close[exit_i]), slip=x_slip, mode=mode)

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)
        sl_out.append(sl_level); tp_out.append(tp_level)

        # מדלגים לאירוע-הכניסה הבא *אחרי* שהעסקה הזו נסגרה - בלעדיות (לא
        # חופפות), בדיוק כמו exclusive_orders=True.
        i_ptr += 1
        while i_ptr < n_events and all_idx[i_ptr] <= exit_i:
            i_ptr += 1

    n_trades = len(entry_prices)
    entry_prices_arr = np.asarray(entry_prices, dtype=float)
    exit_prices_arr = np.asarray(exit_prices, dtype=float)
    sizes_arr = np.asarray(sizes, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    # PnL עם size חתום (חיובי=long, שלילי=short) - נוסחה אחת לשני הכיוונים:
    # ל-short, ירידת מחיר (exit<entry) כפול size שלילי נותנת רווח חיובי, נכון.
    pnl = (exit_prices_arr - entry_prices_arr) * sizes_arr * multiplier
    commission_total = float(commission_per_contract) * np.abs(sizes_arr) * 2.0
    pnl = pnl - commission_total

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    ) * np.sign(sizes_arr)

    # Equity מלא לכל בר (mark-to-market, לא רק ברגע הסגירה) - position_flag
    # חתום (חיובי=long פתוח, שלילי=short פתוח, 0=שטוח) מ-cumsum של ה-size
    # החתום בנקודות המילוי/סגירה, אותה טכניקה בדיוק כמו run_fast_sma_crossover_backtest
    # אבל עם size חתום כדי שהיא תעבוד גם ל-short.
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, sizes_arr)
    np.add.at(position_delta, exit_idx_arr, -sizes_arr)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag != 0,
                          (close - entry_price_ffilled) * position_flag * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": sizes_arr,
        "PnL": pnl,
        "ReturnPct": return_frac,
    })
    if len(sl_out) == n_trades:
        trades_df["SL"] = np.asarray(sl_out, dtype=float); trades_df["TP"] = np.asarray(tp_out, dtype=float)

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ============================================================ fib_golden
# אסטרטגיית "פיבונאצ'י + יחס הזהב": פיבונאצ'י מגדיר את המפה (עד לאן צפוי
# התיקון להגיע ועד לאן התנועה יכולה להמשיך), ואינדיקטור שני (RSI ו/או נר
# היפוך) נותן את הטריגר בתוך האזור. מנוע וקטורי-קבוע-מראש, אותו דפוס בדיוק כמו
# run_hammer_rsi_backtest: מסומן ב-"# strategy: fib_golden" בשורה הראשונה.
#
# הלוגיקה:
# 1. נקודות מפנה (swing high / swing low): בר שהוא הגבוה/הנמוך ביותר בחלון של
#    SWING_BARS ברים לכל צד. נקודת מפנה "ידועה" רק SWING_BARS ברים אחרי שקרתה
#    (אין הצצה לעתיד).
# 2. הרגל (leg) = מנקודת המפנה הישנה לחדשה. מפנה-נמוך ואז מפנה-גבוה = רגל
#    עולה (מחפשים לונג בתיקון); הפוך = רגל יורדת (שורט).
# 3. רמות פיבונאצ'י על הרגל: כניסה בתיקון של ENTRY_FIB (0.618 = יחס הזהב),
#    סטופ מעבר לתיקון של STOP_FIB (1.0 = נקודת המפנה) פחות/ועוד STOP_BUFFER_PTS,
#    יעד בהארכה של TARGET_EXT מהרגל (1.0 = חזרה לקצה הרגל, 1.618 = הארכת זהב).
# 4. טריגר: הבר נוגע ברמת הכניסה (low<=רמה בלונג) אבל נסגר מעל רמת הסטופ, ובנוסף
#    RSI_CONFIRM: RSI מתחת ל-RSI_LONG_MAX ומתחיל לעלות (בשורט - מעל RSI_SHORT_MIN
#    ומתחיל לרדת); CANDLE_CONFIRM: נר בכיוון העסקה (close>open בלונג).
#    EMA_TREND>0: לונג רק מעל ה-EMA, שורט רק מתחתיו (0 = כבוי).
# 5. מילוי בפתיחת הבר הבא (+סליפג'), יציאה בסטופ/יעד - הראשון שנוגע בו
#    high/low. עסקה אחת לכל רגל (לא נכנסים שוב לאותו תיקון), עסקאות בלעדיות.

FIB_GOLDEN_CODE = """\
# strategy: fib_golden
# Fibonacci retracement + golden ratio, confirmed by a second indicator.
# Map:     swing low -> swing high (or reverse) defines the leg.
# Entry:   pullback touches ENTRY_FIB of the leg (0.618 = golden ratio).
# Trigger: RSI turning back in the trade direction and/or a candle closing that way.
# Stop:    beyond STOP_FIB of the leg (1.0 = the swing point) + buffer.
# Target:  TARGET_EXT of the leg (1.0 = retest the swing, 1.618 = golden extension).
SWING_BARS = 20        # input
MIN_LEG_PTS = 40       # input
ENTRY_FIB = 0.618      # input
STOP_FIB = 1.0         # input
STOP_BUFFER_PTS = 5    # input
TARGET_EXT = 1.618     # input
RSI_CONFIRM = True     # input
RSI_LEN = 14           # input
RSI_LONG_MAX = 45      # input
RSI_SHORT_MIN = 55     # input
CANDLE_CONFIRM = True  # input
EMA_TREND = 200        # input
ALLOW_SHORT = True     # input
"""


def _known_pivots(values: np.ndarray, n_side: int, is_high: bool):
    """מחזירה (ערך, אינדקס) של נקודת המפנה האחרונה ש*כבר ידועה* בכל בר.
    מפנה בבר j מאושר רק בבר j+n_side (החלון סביבו סגור) - משם והלאה הוא
    "הידוע האחרון" עד שמאושר מפנה חדש (ffill). NaN עד המפנה הראשון."""
    n = len(values)
    win = 2 * n_side + 1
    s = pd.Series(values)
    ext = s.rolling(win, center=True).max() if is_high else s.rolling(win, center=True).min()
    is_piv = (s == ext).to_numpy() & ~np.isnan(ext.to_numpy())
    piv_idx = np.flatnonzero(is_piv)
    confirm_at = piv_idx + n_side
    keep = confirm_at < n
    piv_idx, confirm_at = piv_idx[keep], confirm_at[keep]
    val = np.full(n, np.nan)
    idx = np.full(n, np.nan)
    val[confirm_at] = values[piv_idx]
    idx[confirm_at] = piv_idx
    val = pd.Series(val).ffill().to_numpy()
    idx = pd.Series(idx).ffill().to_numpy()
    return val, idx


def run_fib_golden_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                            size: float = 1.0,
                            multiplier: float = 1.0,
                            slippage_ticks: float = 0.0,
                            commission_per_contract: float = 0.0,
                            market_order: bool = True,
                            tick_size: float = MNQ_TICK_SIZE,
                            swing_bars: int = 20,
                            min_leg_pts: float = 40.0,
                            entry_fib: float = 0.618,
                            stop_fib: float = 1.0,
                            stop_buffer_pts: float = 5.0,
                            target_ext: float = 1.618,
                            rsi_confirm: bool = True,
                            rsi_len: int = 14,
                            rsi_long_max: float = 45.0,
                            rsi_short_min: float = 55.0,
                            candle_confirm: bool = True,
                            ema_trend: int = 200,
                            allow_short: bool = True,
                            exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    """ראו ההסבר בראש הקטע (fib_golden). מחזיר את אותו מבנה תוצאה בדיוק כמו
    run_hammer_rsi_backtest, כך שכל הממשק (Key Stats / Performance / Trades /
    markers) עובד בלי שינוי."""
    swing_bars = max(2, int(swing_bars))
    if df is None or len(df) < 2 * swing_bars + 3:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    # --- 1+2. נקודות מפנה ידועות והרגל הנוכחית ---
    ph_val, ph_idx = _known_pivots(high, swing_bars, True)
    pl_val, pl_idx = _known_pivots(low, swing_bars, False)
    have_both = ~np.isnan(ph_idx) & ~np.isnan(pl_idx)
    up_leg = have_both & (ph_idx > pl_idx)     # נמוך ואז גבוה: תיקון כלפי מטה, לונג
    down_leg = have_both & (pl_idx > ph_idx)   # גבוה ואז נמוך: תיקון כלפי מעלה, שורט
    leg = ph_val - pl_val
    big_enough = leg >= float(min_leg_pts)

    # --- 3. רמות פיבונאצ'י (וקטורי, לכל בר לפי הרגל הידועה לו) ---
    buf = float(stop_buffer_pts)
    long_entry_lvl = ph_val - float(entry_fib) * leg
    long_stop_lvl = ph_val - float(stop_fib) * leg - buf
    long_target_lvl = pl_val + float(target_ext) * leg
    short_entry_lvl = pl_val + float(entry_fib) * leg
    short_stop_lvl = pl_val + float(stop_fib) * leg + buf
    short_target_lvl = ph_val - float(target_ext) * leg

    # --- 4. טריגרים ---
    rsi_arr = rsi(close, int(rsi_len))
    prev_rsi = np.empty(n); prev_rsi[0] = np.nan; prev_rsi[1:] = rsi_arr[:-1]
    if rsi_confirm:
        rsi_ok_long = (rsi_arr <= float(rsi_long_max)) & (rsi_arr > prev_rsi)
        rsi_ok_short = (rsi_arr >= float(rsi_short_min)) & (rsi_arr < prev_rsi)
    else:
        rsi_ok_long = rsi_ok_short = np.ones(n, dtype=bool)
    if candle_confirm:
        candle_ok_long = close > open_
        candle_ok_short = close < open_
    else:
        candle_ok_long = candle_ok_short = np.ones(n, dtype=bool)
    if int(ema_trend) > 0:
        ema_arr = ema(close, int(ema_trend))
        trend_ok_long = close > ema_arr
        trend_ok_short = close < ema_arr
    else:
        trend_ok_long = trend_ok_short = np.ones(n, dtype=bool)

    long_signal = (up_leg & big_enough
                   & (low <= long_entry_lvl) & (close > long_stop_lvl)
                   & rsi_ok_long & candle_ok_long & trend_ok_long)
    short_signal = (down_leg & big_enough
                    & (high >= short_entry_lvl) & (close < short_stop_lvl)
                    & rsi_ok_short & candle_ok_short & trend_ok_short)
    if not allow_short:
        short_signal = np.zeros(n, dtype=bool)

    long_idx = np.flatnonzero(long_signal)
    short_idx = np.flatnonzero(short_signal)
    all_idx = np.concatenate([long_idx, short_idx])
    is_long_arr = np.concatenate([
        np.ones(len(long_idx), dtype=bool), np.zeros(len(short_idx), dtype=bool),
    ])
    order = np.argsort(all_idx, kind="stable")
    all_idx = all_idx[order]
    is_long_arr = is_long_arr[order]

    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    mode = fill_mode if fill_mode in FILL_MODES else "realistic"
    x_slip = slippage_amount if exit_slippage_ticks is None else float(exit_slippage_ticks) * float(tick_size)
    if mode == "worst":
        x_slip *= 2.0
    tp_pad = float(tick_size) if mode == "worst" else 0.0

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []
    sl_out, tp_out, extra_out = [], [], []
    used_legs: set = set()   # (pivot_low_idx, pivot_high_idx) - עסקה אחת לכל רגל

    n_events = len(all_idx)
    i_ptr = 0
    while i_ptr < n_events:
        sig_i = int(all_idx[i_ptr])
        is_long = bool(is_long_arr[i_ptr])
        leg_key = (int(pl_idx[sig_i]), int(ph_idx[sig_i]), is_long)
        fill_i = sig_i + 1
        if fill_i >= n:
            break
        if leg_key in used_legs:
            i_ptr += 1
            continue

        raw_entry = open_[fill_i]
        entry_price = raw_entry + slippage_amount if is_long else raw_entry - slippage_amount
        if is_long:
            sl_level = float(long_stop_lvl[sig_i])
            tp_level = float(long_target_lvl[sig_i])
            if not (sl_level < entry_price < tp_level):
                i_ptr += 1
                continue  # הפתיחה כבר מעבר לסטופ/ליעד - אין עסקה הגיונית
            hit_tp = high[fill_i + 1:] >= tp_level + tp_pad
            hit_sl = low[fill_i + 1:] <= sl_level
        else:
            sl_level = float(short_stop_lvl[sig_i])
            tp_level = float(short_target_lvl[sig_i])
            if not (tp_level < entry_price < sl_level):
                i_ptr += 1
                continue
            hit_tp = low[fill_i + 1:] <= tp_level - tp_pad
            hit_sl = high[fill_i + 1:] >= sl_level
        hit_any = hit_tp | hit_sl

        if not hit_any.any():
            exit_i = n - 1
            exit_price = _exit_fill(is_long=is_long, kind="forced", level=float(close[exit_i]), bar_open=float(open_[exit_i]),
                                    bar_close=float(close[exit_i]), slip=x_slip, mode=mode)
        else:
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level   # שני הצדדים באותו בר: פסימי
            kind = "sl" if sl_hit_here else "tp"
            exit_price = _exit_fill(is_long=is_long, kind=kind, level=(sl_level if kind == "sl" else tp_level),
                                    bar_open=float(open_[exit_i]), bar_close=float(close[exit_i]), slip=x_slip, mode=mode)

        used_legs.add(leg_key)
        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)
        sl_out.append(sl_level); tp_out.append(tp_level)
        # גיאומטריה לגרף: הרגל (מפנה→מפנה) ורמת הכניסה בפיבונאצ'י
        a_i, b_i = int(pl_idx[sig_i]), int(ph_idx[sig_i])
        lo_t, hi_t = times[a_i], times[b_i]
        extra_out.append([
            {"t": "seg", "x1": str(pd.Timestamp(lo_t)), "y1": float(pl_val[sig_i]), "x2": str(pd.Timestamp(hi_t)), "y2": float(ph_val[sig_i]), "label": "leg"},
            {"t": "hline", "x1": str(pd.Timestamp(times[min(a_i, b_i)])), "x2": str(pd.Timestamp(times[exit_i])),
             "y": float(long_entry_lvl[sig_i] if is_long else short_entry_lvl[sig_i]), "label": f"fib {entry_fib:g}"},
        ])

        i_ptr += 1
        while i_ptr < n_events and all_idx[i_ptr] <= exit_i:
            i_ptr += 1

    return _finalize_signed_trades(
        times=times, close=close, n=n, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        entry_times=entry_times, exit_times=exit_times,
        entry_prices=entry_prices, exit_prices=exit_prices, sizes=sizes,
        entry_fill_idxs=entry_fill_idxs, exit_idxs=exit_idxs,
        sl=sl_out, tp=tp_out, extra=extra_out,
    )


def _finalize_signed_trades(*, times, close, n, cash, multiplier, commission_per_contract,
                            entry_times, exit_times, entry_prices, exit_prices, sizes,
                            entry_fill_idxs, exit_idxs, sl=None, tp=None, extra=None) -> dict:
    """סטטיסטיקות + equity מרשימת עסקאות עם size חתום (לונג חיובי, שורט שלילי).
    אותו חישוב בדיוק כמו הזנב של run_hammer_rsi_backtest, מופרד כדי שמנועים
    נוספים (fib_golden) לא ישכפלו אותו."""
    n_trades = len(entry_prices)
    entry_prices_arr = np.asarray(entry_prices, dtype=float)
    exit_prices_arr = np.asarray(exit_prices, dtype=float)
    sizes_arr = np.asarray(sizes, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    pnl = (exit_prices_arr - entry_prices_arr) * sizes_arr * multiplier
    pnl = pnl - float(commission_per_contract) * np.abs(sizes_arr) * 2.0
    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    ) * np.sign(sizes_arr)

    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, sizes_arr)
    np.add.at(position_delta, exit_idx_arr, -sizes_arr)
    position_flag = np.cumsum(position_delta)
    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    if n_trades:
        entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag != 0,
                          (close - entry_price_ffilled) * position_flag * multiplier, 0.0)
    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    equity_curve = cash + np.cumsum(realized_pnl_step) + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0
    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": sizes_arr,
        "PnL": pnl,
        "ReturnPct": return_frac,
    })
    # רמות הסטופ/היעד וגיאומטריה נוספת (רגל פיבונאצ'י, טווח וויקוף...) - לציור על הגרף
    if sl is not None and len(sl) == n_trades:
        trades_df["SL"] = np.asarray(sl, dtype=float)
    if tp is not None and len(tp) == n_trades:
        trades_df["TP"] = np.asarray(tp, dtype=float)
    if extra is not None and len(extra) == n_trades:
        trades_df["Extra"] = list(extra)
    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ============================================================ כלי עזר משותפים
def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """דקה-ביום (0..1439) לכל בר. הקאש נשמר בשעון ניו-יורק (naive, ראו
    data.SOURCE_TZ), לכן 09:30 = פתיחת הסשן הרגיל של ארה"ב."""
    idx = pd.DatetimeIndex(index)
    return (idx.hour * 60 + idx.minute).to_numpy()


def _hhmm(text, default: int) -> int:
    """'09:30' -> 570. מספר (למשל 930) מתקבל גם הוא. לא תקין -> ברירת מחדל."""
    try:
        if isinstance(text, (int, float)):
            v = int(text)
            return (v // 100) * 60 + (v % 100)
        h, m = str(text).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default


# מודל מילוי ליציאות: מצבים
#   optimistic  - המודל הישן: סטופ/יעד מתמלאים בדיוק ברמה, בלי סליפג' ביציאה.
#   realistic   - (ברירת מחדל) סטופ = פקודת שוק: סליפג' ביציאה + פער דרך הסטופ מתמלא בפתיחת הנר;
#                 יעד = פקודת לימיט: מתמלא ברמה (או בפתיחה אם הנר נפתח מעבר לה, לטובתנו); יציאה
#                 כפויה (זמן/סוף דאטה) = שוק: סגירה פחות סליפג'.
#   worst       - כמו realistic אבל סליפג' כפול, והיעד דורש חדירה של טיק מעבר לרמה (לא רק נגיעה).
FILL_MODES = ("optimistic", "realistic", "worst")


def _exit_fill(*, is_long: bool, kind: str, level: float, bar_open: float, bar_close: float,
               slip: float, mode: str) -> float:
    """מחיר יציאה בפועל. kind: 'sl' | 'tp' | 'forced'."""
    if mode == "optimistic":
        return float(level if kind != "forced" else bar_close)
    if kind == "tp":
        # לימיט: אם הנר נפתח כבר מעבר ליעד, מתמלא בפתיחה (טוב יותר); אחרת ברמה
        if is_long:
            return float(max(level, bar_open)) if bar_open > level else float(level)
        return float(min(level, bar_open)) if bar_open < level else float(level)
    if kind == "sl":
        # שוק: פער דרך הסטופ -> פתיחה; ואז סליפג' נגדנו
        if is_long:
            base = min(level, bar_open) if bar_open < level else level
            return float(base - slip)
        base = max(level, bar_open) if bar_open > level else level
        return float(base + slip)
    # forced (זמן / סוף דאטה): שוק בסגירה
    return float(bar_close - slip) if is_long else float(bar_close + slip)


def _simulate_signed_signals(*, times, open_, high, low, close, n,
                             sig_idx: np.ndarray, sig_is_long: np.ndarray,
                             sl_levels: np.ndarray, tp_levels: np.ndarray,
                             max_exit_idx: np.ndarray | None,
                             dedupe_key: np.ndarray | None,
                             size: float, slippage_amount: float,
                             cash: float, multiplier: float,
                             commission_per_contract: float,
                             extras: list | None = None,
                             exit_slippage_amount: float | None = None, fill_mode: str = "realistic",
                             tick_size: float = MNQ_TICK_SIZE) -> dict:
    """הלב המשותף של המנועים המהירים: לכל אות (ממוין בזמן) מילוי בפתיחת הבר
    הבא, ואז חיפוש וקטורי של הבר הראשון שנוגע בסטופ/יעד (או יציאה כפויה ב-
    max_exit_idx - למשל סוף הסשן). עסקאות בלעדיות; dedupe_key (למשל תאריך)
    מאפשר "עסקה אחת לכל מפתח". סטופ ויעד באותו בר -> פסימי (סטופ)."""
    order = np.argsort(sig_idx, kind="stable")
    sig_idx, sig_is_long = sig_idx[order], sig_is_long[order]
    sl_levels, tp_levels = sl_levels[order], tp_levels[order]
    if max_exit_idx is not None:
        max_exit_idx = max_exit_idx[order]
    if dedupe_key is not None:
        dedupe_key = dedupe_key[order]
    if extras is not None:
        extras = [extras[k] for k in order]
    mode = fill_mode if fill_mode in FILL_MODES else "realistic"
    x_slip = slippage_amount if exit_slippage_amount is None else float(exit_slippage_amount)
    if mode == "worst":
        x_slip *= 2.0
    tp_pad = float(tick_size) if mode == "worst" else 0.0   # היעד חייב להיחצות, לא רק להיגע

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []
    sl_out, tp_out, extra_out = [], [], []
    used: set = set()
    n_events = len(sig_idx)
    i_ptr = 0
    while i_ptr < n_events:
        sig_i = int(sig_idx[i_ptr])
        is_long = bool(sig_is_long[i_ptr])
        fill_i = sig_i + 1
        if fill_i >= n:
            break
        key = None if dedupe_key is None else (dedupe_key[i_ptr], is_long)
        if key is not None and key in used:
            i_ptr += 1
            continue
        raw_entry = open_[fill_i]
        entry_price = raw_entry + slippage_amount if is_long else raw_entry - slippage_amount
        sl_level, tp_level = float(sl_levels[i_ptr]), float(tp_levels[i_ptr])
        ok = (sl_level < entry_price < tp_level) if is_long else (tp_level < entry_price < sl_level)
        if not ok or not np.isfinite(sl_level) or not np.isfinite(tp_level):
            i_ptr += 1
            continue
        last_i = n - 1 if max_exit_idx is None else int(min(n - 1, max_exit_idx[i_ptr]))
        seg_hi = high[fill_i + 1:last_i + 1]
        seg_lo = low[fill_i + 1:last_i + 1]
        if is_long:
            hit_tp, hit_sl = seg_hi >= tp_level + tp_pad, seg_lo <= sl_level
        else:
            hit_tp, hit_sl = seg_lo <= tp_level - tp_pad, seg_hi >= sl_level
        hit_any = hit_tp | hit_sl
        if hit_any.size and hit_any.any():
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level
            kind = "sl" if sl_hit_here else "tp"
            exit_price = _exit_fill(is_long=is_long, kind=kind, level=(sl_level if kind == "sl" else tp_level),
                                    bar_open=float(open_[exit_i]), bar_close=float(close[exit_i]), slip=x_slip, mode=mode)
        else:
            exit_i = max(fill_i, last_i)     # יציאה כפויה (סוף סשן / סוף דאטה)
            exit_price = _exit_fill(is_long=is_long, kind="forced", level=float(close[exit_i]),
                                    bar_open=float(open_[exit_i]), bar_close=float(close[exit_i]), slip=x_slip, mode=mode)
        if key is not None:
            used.add(key)
        entry_times.append(times[fill_i]); exit_times.append(times[exit_i])
        entry_prices.append(entry_price); exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i); exit_idxs.append(exit_i)
        sl_out.append(sl_level); tp_out.append(tp_level)
        extra_out.append(extras[i_ptr] if extras is not None else None)
        i_ptr += 1
        while i_ptr < n_events and sig_idx[i_ptr] <= exit_i:
            i_ptr += 1

    return _finalize_signed_trades(
        times=times, close=close, n=n, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        entry_times=entry_times, exit_times=exit_times,
        entry_prices=entry_prices, exit_prices=exit_prices, sizes=sizes,
        entry_fill_idxs=entry_fill_idxs, exit_idxs=exit_idxs,
        sl=sl_out, tp=tp_out, extra=extra_out if extras is not None else None,
    )


# ============================================================ orb
# פריצת טווח הפתיחה (Opening Range Breakout): הגבוה/הנמוך של RANGE_MINUTES
# הדקות הראשונות של הסשן הרגיל (09:30 ניו-יורק) מגדירים טווח; סגירה מעליו =
# לונג, מתחתיו = שורט. סטופ בצד השני של הטווח (או STOP_PTS קבוע אם > 0), יעד
# = TARGET_RR כפול הסיכון. עסקה אחת ליום לכל כיוון, יציאה כפויה ב-EXIT_TIME.
ORB_CODE = """\
# strategy: orb
# Opening Range Breakout (US regular session, New York time).
# Range:  high/low of the first RANGE_MINUTES after SESSION_OPEN.
# Entry:  a bar closes above the range high (+BUFFER_PTS) -> long; below the low -> short.
# Stop:   other side of the range (or STOP_PTS if > 0). Target: TARGET_RR x risk.
# Exit:   forced flat at EXIT_TIME. One trade per day per direction.
RANGE_MINUTES = 15     # input
BUFFER_PTS = 2         # input
STOP_PTS = 0           # input
TARGET_RR = 2.0        # input
MIN_RANGE_PTS = 20     # input
MAX_RANGE_PTS = 200    # input
LAST_ENTRY_MIN = 180   # input
ALLOW_SHORT = True     # input
"""


def run_orb_backtest(df: pd.DataFrame, *, cash: float = 100_000.0, size: float = 1.0,
                     multiplier: float = 1.0, slippage_ticks: float = 0.0,
                     commission_per_contract: float = 0.0, market_order: bool = True,
                     tick_size: float = MNQ_TICK_SIZE,
                     range_minutes: int = 15, buffer_pts: float = 2.0, stop_pts: float = 0.0,
                     target_rr: float = 2.0, min_range_pts: float = 20.0,
                     max_range_pts: float = 200.0, last_entry_min: int = 180,
                     allow_short: bool = True,
                     session_open: str = "09:30", exit_time: str = "15:55",
                         exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    if df is None or len(df) < 50:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)
    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float); high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float); close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    mod = _minute_of_day(df.index)
    open_min = _hhmm(session_open, 570)
    exit_min = _hhmm(exit_time, 955)
    rng_end = open_min + int(range_minutes)
    day = pd.DatetimeIndex(df.index).normalize()
    day_codes, _ = pd.factorize(day)

    in_range = (mod >= open_min) & (mod < rng_end)
    s_hi = pd.Series(np.where(in_range, high, np.nan))
    s_lo = pd.Series(np.where(in_range, low, np.nan))
    or_high = s_hi.groupby(day_codes).transform("max").to_numpy()
    or_low = s_lo.groupby(day_codes).transform("min").to_numpy()
    or_size = or_high - or_low
    # אינדקס הבר האחרון של כל יום שעדיין <= exit_time (יציאה כפויה)
    pos = np.arange(n)
    ok_exit = mod <= exit_min
    last_ok = pd.Series(np.where(ok_exit, pos, -1)).groupby(day_codes).transform("max").to_numpy()
    day_exit_idx = np.where(last_ok >= 0, last_ok, pos)

    window = (mod >= rng_end) & (mod < min(exit_min, rng_end + int(last_entry_min))) \
        & np.isfinite(or_high) & (or_size >= float(min_range_pts)) & (or_size <= float(max_range_pts))
    long_sig = window & (close > or_high + float(buffer_pts))
    short_sig = window & (close < or_low - float(buffer_pts)) if allow_short else np.zeros(n, dtype=bool)

    li, si = np.flatnonzero(long_sig), np.flatnonzero(short_sig)
    sig_idx = np.concatenate([li, si])
    is_long = np.concatenate([np.ones(len(li), bool), np.zeros(len(si), bool)])
    fill = np.minimum(sig_idx + 1, n - 1)
    entry_est = open_[fill]
    risk_long = (entry_est - or_low[sig_idx]) if stop_pts <= 0 else np.full(len(sig_idx), float(stop_pts))
    risk_short = (or_high[sig_idx] - entry_est) if stop_pts <= 0 else np.full(len(sig_idx), float(stop_pts))
    risk = np.where(is_long, risk_long, risk_short)
    risk = np.where(risk > 0, risk, np.nan)
    sl = np.where(is_long, entry_est - risk, entry_est + risk)
    tp = np.where(is_long, entry_est + float(target_rr) * risk, entry_est - float(target_rr) * risk)
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    day_start = pd.Series(np.where(in_range, pos, np.iinfo(np.int64).max)).groupby(day_codes).transform("min").to_numpy()
    extras = []
    for si in sig_idx:
        ds = int(day_start[si]) if day_start[si] < n else int(si)
        extras.append([{"t": "box", "x1": str(pd.Timestamp(times[ds])), "x2": str(pd.Timestamp(times[int(day_exit_idx[si])])),
                        "y1": float(or_low[si]), "y2": float(or_high[si]), "label": "opening range"}])
    return _simulate_signed_signals(
        times=times, open_=open_, high=high, low=low, close=close, n=n,
        sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp,
        max_exit_idx=day_exit_idx[sig_idx], dedupe_key=day_codes[sig_idx],
        size=size, slippage_amount=slippage_amount, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract, extras=extras,
        exit_slippage_amount=(None if exit_slippage_ticks is None else float(exit_slippage_ticks) * float(tick_size)),
        fill_mode=fill_mode, tick_size=tick_size,
    )


# ============================================================ vwap_revert
# חזרה ל-VWAP: כשהמחיר מתרחק DEVIATION_PTS מה-VWAP היומי ו-RSI קיצוני ומתחיל
# להתהפך - נכנסים נגד התנועה עם יעד ב-VWAP עצמו וסטופ קבוע. סשן רגיל בלבד
# כברירת מחדל, יציאה כפויה בסוף הסשן.
VWAP_REVERT_CODE = """\
# strategy: vwap_revert
# Mean reversion to the session VWAP (resets daily).
# Entry:  price is DEVIATION_PTS below VWAP, RSI <= RSI_LONG_MAX and turning up -> long
#         (mirror above VWAP with RSI >= RSI_SHORT_MIN -> short).
# Target: the VWAP level at signal time. Stop: STOP_PTS. Flat at EXIT_TIME.
DEVIATION_PTS = 40     # input
RSI_LEN = 14           # input
RSI_LONG_MAX = 30      # input
RSI_SHORT_MIN = 70     # input
STOP_PTS = 30          # input
RTH_ONLY = True        # input
ALLOW_SHORT = True     # input
"""


def run_vwap_revert_backtest(df: pd.DataFrame, *, cash: float = 100_000.0, size: float = 1.0,
                             multiplier: float = 1.0, slippage_ticks: float = 0.0,
                             commission_per_contract: float = 0.0, market_order: bool = True,
                             tick_size: float = MNQ_TICK_SIZE,
                             deviation_pts: float = 40.0, rsi_len: int = 14,
                             rsi_long_max: float = 30.0, rsi_short_min: float = 70.0,
                             stop_pts: float = 30.0, rth_only: bool = True,
                             allow_short: bool = True,
                             session_open: str = "09:30", exit_time: str = "15:55",
                         exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    if df is None or len(df) < 50:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)
    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float); high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float); close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    vwap = session_vwap(df)
    rsi_arr = rsi(close, int(rsi_len))
    prev_rsi = np.empty(n); prev_rsi[0] = np.nan; prev_rsi[1:] = rsi_arr[:-1]
    mod = _minute_of_day(df.index)
    open_min, exit_min = _hhmm(session_open, 570), _hhmm(exit_time, 955)
    day_codes, _ = pd.factorize(pd.DatetimeIndex(df.index).normalize())
    pos = np.arange(n)
    if rth_only:
        window = (mod >= open_min) & (mod < exit_min)
        last_ok = pd.Series(np.where(mod <= exit_min, pos, -1)).groupby(day_codes).transform("max").to_numpy()
        day_exit_idx = np.where(last_ok >= 0, last_ok, pos)
        max_exit = day_exit_idx
    else:
        window = np.ones(n, dtype=bool)
        max_exit = None
    dev = float(deviation_pts)
    long_sig = window & (close <= vwap - dev) & (rsi_arr <= float(rsi_long_max)) & (rsi_arr > prev_rsi)
    short_sig = (window & (close >= vwap + dev) & (rsi_arr >= float(rsi_short_min)) & (rsi_arr < prev_rsi)) \
        if allow_short else np.zeros(n, dtype=bool)
    li, si = np.flatnonzero(long_sig), np.flatnonzero(short_sig)
    sig_idx = np.concatenate([li, si])
    is_long = np.concatenate([np.ones(len(li), bool), np.zeros(len(si), bool)])
    fill = np.minimum(sig_idx + 1, n - 1)
    entry_est = open_[fill]
    sl = np.where(is_long, entry_est - float(stop_pts), entry_est + float(stop_pts))
    tp = vwap[sig_idx]
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    return _simulate_signed_signals(
        times=times, open_=open_, high=high, low=low, close=close, n=n,
        sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp,
        max_exit_idx=None if max_exit is None else max_exit[sig_idx], dedupe_key=None,
        size=size, slippage_amount=slippage_amount, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        exit_slippage_amount=(None if exit_slippage_ticks is None else float(exit_slippage_ticks) * float(tick_size)),
        fill_mode=fill_mode, tick_size=tick_size,
    )


# ============================================================ wyckoff
# תורת וויקוף, בגרסה מעשית לנרות תוך-יומיים: טווח מסחר (צבירה/הפצה) מזוהה
# כשהמחיר "נעול" בין תמיכה להתנגדות במשך RANGE_BARS נרות ורוחב הטווח קטן ביחס
# ל-ATR. האירוע המסחרי הוא ה-Spring (שבירה כוזבת מתחת לתמיכה שנסגרת חזרה בתוך
# הטווח - היצע נבדק ולא נמצא) וה-Upthrust (המראה בכיוון ההפוך מעל ההתנגדות).
# אישור אופציונלי: הנר הבא נסגר מעל הגבוה של ה-Spring (Sign of Strength מוקטן).
# פילטר נפח: נפח ה-Spring עד VOL_MAX_RATIO מהממוצע (מבחן על נפח נמוך = חוסר היצע).
# יעד: TARGET_FRAC מרוחב הטווח מעל התמיכה (1.0 = ההתנגדות, 1.5 = מהלך נמדד).
# סטופ: מתחת לנמוך של ה-Spring פחות STOP_BUFFER_PTS. יציאה כפויה אחרי MAX_BARS.
WYCKOFF_CODE = """\
# strategy: wyckoff
# Wyckoff trading-range spring / upthrust.
# Range:   support = lowest low, resistance = highest high of the last RANGE_BARS bars,
#          accepted only when (resistance - support) <= RANGE_ATR_MULT x ATR (a real range).
# Spring:  low pierces support by >= SPRING_PEN_PTS but the bar closes back inside -> long.
# Upthrust: mirror above resistance -> short.
# Filter:  spring-bar volume <= VOL_MAX_RATIO x 20-bar average (no supply); optional EMA trend.
# Confirm: CONFIRM_NEXT requires the next bar to close beyond the spring bar's extreme.
# Target:  support + TARGET_FRAC x range width. Stop: spring low - STOP_BUFFER_PTS. Time stop MAX_BARS.
RANGE_BARS = 40        # input
RANGE_ATR_MULT = 6     # input
SPRING_PEN_PTS = 5     # input
VOL_MAX_RATIO = 1.2    # input
CONFIRM_NEXT = True    # input
TARGET_FRAC = 1.0      # input
STOP_BUFFER_PTS = 8    # input
MAX_BARS = 120         # input
EMA_TREND = 0          # input
ALLOW_SHORT = True     # input
"""


def _atr(high, low, close, n=14) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def run_wyckoff_backtest(df: pd.DataFrame, *, cash: float = 100_000.0, size: float = 1.0,
                         multiplier: float = 1.0, slippage_ticks: float = 0.0,
                         commission_per_contract: float = 0.0, market_order: bool = True,
                         tick_size: float = MNQ_TICK_SIZE,
                         range_bars: int = 40, range_atr_mult: float = 6.0, spring_pen_pts: float = 5.0,
                         vol_max_ratio: float = 1.2, confirm_next: bool = True, target_frac: float = 1.0,
                         stop_buffer_pts: float = 8.0, max_bars: int = 120, ema_trend: int = 0,
                         allow_short: bool = True,
                         exit_slippage_ticks: float | None = None, fill_mode: str = "realistic") -> dict:
    range_bars = max(5, int(range_bars))
    if df is None or len(df) < range_bars + 30:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)
    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float); high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float); close = df["Close"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.ones(len(df))
    n = len(close)

    # טווח המסחר שנבנה *לפני* הבר הנוכחי (shift 1 - בלי הצצה קדימה)
    hs, ls = pd.Series(high), pd.Series(low)
    resistance = hs.shift(1).rolling(range_bars).max().to_numpy()
    support = ls.shift(1).rolling(range_bars).min().to_numpy()
    width = resistance - support
    atr = _atr(high, low, close, 14)
    is_range = np.isfinite(width) & (width > 0) & (width <= float(range_atr_mult) * atr)
    avg_vol = pd.Series(vol).shift(1).rolling(20).mean().to_numpy()
    vol_ok = ~np.isfinite(avg_vol) | (vol <= float(vol_max_ratio) * avg_vol)
    if int(ema_trend) > 0:
        ema_arr = ema(close, int(ema_trend)); trend_long = close > ema_arr; trend_short = close < ema_arr
    else:
        trend_long = trend_short = np.ones(n, dtype=bool)

    pen = float(spring_pen_pts)
    spring = is_range & (low <= support - pen) & (close > support) & vol_ok & trend_long
    upthrust = is_range & (high >= resistance + pen) & (close < resistance) & vol_ok & trend_short
    if not allow_short:
        upthrust = np.zeros(n, dtype=bool)

    # אישור: הנר הבא נסגר מעבר לקיצון של נר ה-Spring; האות עובר לנר המאשר
    if confirm_next:
        nxt_close = np.concatenate([close[1:], [np.nan]])
        spring_conf = spring & (nxt_close > high)
        upthrust_conf = upthrust & (nxt_close < low)
        long_sig_bar = np.flatnonzero(spring_conf); short_sig_bar = np.flatnonzero(upthrust_conf)
        long_sig = long_sig_bar + 1; short_sig = short_sig_bar + 1
    else:
        long_sig_bar = np.flatnonzero(spring); short_sig_bar = np.flatnonzero(upthrust)
        long_sig, short_sig = long_sig_bar, short_sig_bar
    keep_l = long_sig < n - 1; keep_s = short_sig < n - 1
    long_sig, long_sig_bar = long_sig[keep_l], long_sig_bar[keep_l]
    short_sig, short_sig_bar = short_sig[keep_s], short_sig_bar[keep_s]

    sig_idx = np.concatenate([long_sig, short_sig])
    sig_bar = np.concatenate([long_sig_bar, short_sig_bar])
    is_long = np.concatenate([np.ones(len(long_sig), bool), np.zeros(len(short_sig), bool)])
    buf = float(stop_buffer_pts); frac = float(target_frac)
    sl = np.where(is_long, low[sig_bar] - buf, high[sig_bar] + buf)
    tp = np.where(is_long, support[sig_bar] + frac * width[sig_bar], resistance[sig_bar] - frac * width[sig_bar])
    max_exit = sig_idx + int(max_bars) if int(max_bars) > 0 else None
    # עסקה אחת לכל טווח: מפתח = רמת התמיכה/ההתנגדות (ברזולוציית טיק)
    dedupe = np.where(is_long, np.round(support[sig_bar] / tick_size), np.round(resistance[sig_bar] / tick_size))
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    extras = [[{"t": "box", "x1": str(pd.Timestamp(times[max(0, int(b) - range_bars)])), "x2": str(pd.Timestamp(times[min(n - 1, int(b) + 1)])),
                "y1": float(support[b]), "y2": float(resistance[b]), "label": "range"}] for b in sig_bar]
    return _simulate_signed_signals(
        times=times, open_=open_, high=high, low=low, close=close, n=n,
        sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp,
        max_exit_idx=max_exit, dedupe_key=dedupe,
        size=size, slippage_amount=slippage_amount, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract, extras=extras,
        exit_slippage_amount=(None if exit_slippage_ticks is None else float(exit_slippage_ticks) * float(tick_size)),
        fill_mode=fill_mode, tick_size=tick_size,
    )


# סגול אחיד לכניסה וליציאה - בדיוק כמו חצי הכניסה/יציאה של Strategy Tester ב-TradingView
# (שם צובעים את שני האירועים באותו צבע, ולא ירוק/אדום לפי סוג האירוע).
ENTRY_COLOR = "#C77DFF"
EXIT_COLOR = "#C77DFF"
LABEL_COLOR = "#FFFFFF"


# ======================================================================
# המנועים של איתי (מוזגו מ-main, 2026-10-18): hammer_breakout, hammer_swing_exit,
# sma20_mean_reversion, down_streak_reversion, overnight_val_reversion + חישוב
# Value Area לילי. עצמאיים לחלוטין (לא משתמשים ב-_simulate_signed_signals), ולכן
# מודל המילוי (fill_mode / exit_slippage) לא חל עליהם - התוצאות שלהם "אופטימיות".
# ======================================================================
# ======================================================= hammer_breakout
# מנוע מובנה, נבחר ב-app.py:run_or_switch לפי סימון "# strategy: hammer_breakout"
# (ראו detect_builtin_strategy למעלה). long-only: hammer מגדיר "רמת-פריצה" (high + טיק אחד), וכניסה
# מתבצעת רק אם הבר *הבא* ממש (לא כל בר עתידי) נוגע ברמה הזו - הזמנת-stop
# תקפה לבר אחד בלבד, לא מחכה ללא הגבלת זמן.
HAMMER_BREAKOUT_CODE = """\
# strategy: hammer_breakout
# Signal candle sets a breakout trigger = signal bar's high + 1 tick.
# Signal shape - EITHER of two bullish-reversal patterns qualifies:
#   Type 1 (classic hammer): upper_wick <= 10% of range AND
#     lower_wick >= 60% of range AND close >= open AND body > 0.
#   Type 2 (bullish strong candle): close >= open AND body >= 40% of range
#     AND lower_wick >= 20% of range AND upper_wick <= 15% of range.
# Extra filters - must hold on top of the shape match:
#   abs(close-open) <= MAX_BODY_PTS on the signal candle itself (applies to
#   both Type 1 and Type 2), AND the standard deviation of the 20 closes
#   strictly before the signal candle must be >= MIN_STD_BEFORE (entry
#   quality filter only - skip signals that form after an unusually quiet
#   stretch; this has no effect on the exit), AND - if EXCLUDE_OPEN_HOURS is
#   on - the signal candle must NOT fall in the 09:00-11:00 ET window (skip
#   the open-volatility whipsaw hours).
# Long entry: only if the very next bar's high crosses above the trigger
#             AND that same bar's close is also above the trigger (a wick
#             through the level that closes back below it doesn't count) -
#             single-bar stop order, not an open-ended wait. Fill price is
#             the trigger itself, not the next bar's open.
# Exit - two conditions, whichever comes first:
#   1. Stop Loss: price drops STOP_LOSS_PTS below entry price.
#   2. SMA20 Distance Exit: on every bar after entry, distance = close -
#      SMA20 (20-bar rolling mean of Close) is tracked. Once distance has
#      reached at least PEAK_DISTANCE_PTS above SMA20 at any point since
#      entry (the "peak" is armed - and stays armed for the rest of the
#      trade, even if distance later drops), exit at the close of the first
#      subsequent bar where distance drops back below EXIT_DISTANCE_PTS.
STOP_LOSS_PTS = 5             # input
MAX_BODY_PTS = 7              # input
MIN_STD_BEFORE = 20           # input
EXCLUDE_OPEN_HOURS = False    # input
PEAK_DISTANCE_PTS = 30        # input
EXIT_DISTANCE_PTS = 5         # input
"""


def _find_first_stop_or_exit_signal(low: np.ndarray, exit_signal_trigger: np.ndarray, start: int,
                                    n: int, sl_level: float) -> int | None:
    """מוצאת את האינדקס הראשון (>= start) שבו low<=sl_level (סטופ) *או*
    exit_signal_trigger[i] אמת (סיגנל-יציאה מבוסס-צורה, למשל Inverted Hammer -
    ראו הקורא), בלי לבנות פרוסה של *כל* שארית המערך בכל קריאה. חלון-חיפוש
    גדל בהדרגה (4096, 16384, ...) - קריטי לביצועים: numpy מחשב את ההשוואה
    האלמנט-מול-אלמנט על *כל* הפרוסה לפני ש-argmax בכלל מריץ (אין
    קיצור-דרך/short-circuit ב-numpy), אז פרוסה של "כל השארית" עולה O(n)
    *לכל עסקה בנפרד* גם כשההתאמה עצמה קרובה - קריטי לעמוד בדרישת "מתחת
    ל-10 שניות" על 2.5 מיליון ברים.

    exit_signal_trigger מחושב *מראש* פעם אחת על כל המערך - בלתי-תלוי בעסקה
    הספציפית, ולכן זול לחשב פעם אחת בחוץ ולהעביר כפרמטר, לא לחשב מחדש לכל
    עסקה.
    """
    chunk = 4096
    pos = start
    while pos < n:
        end = min(n, pos + chunk)
        seg_low = low[pos:end]
        seg_exit_signal = exit_signal_trigger[pos:end]
        hit = (seg_low <= sl_level) | seg_exit_signal
        if hit.any():
            return pos + int(np.argmax(hit))
        pos = end
        chunk *= 4  # לא נמצא בחלון קטן - כנראה נדרש חלון גדול משמעותית, מכפילים מהר
    return None


def _find_first_sl_exit_signal_or_arm(low: np.ndarray, high: np.ndarray,
                                      exit_signal_trigger: np.ndarray, start: int, n: int,
                                      sl_level: float, arm_level: float) -> int | None:
    """שלב 1 של Break Even: מוצאת את האינדקס הראשון (>= start) שבו מתקיים אחד
    משלושה - low<=sl_level (הסטופ המקורי), exit_signal_trigger[i] אמת
    (Inverted Hammer Exit), או high>=arm_level (המחיר עלה BREAK_EVEN_PTS מעל
    הכניסה - "מזין" את Break Even) - אותו חלון-חיפוש-גדל בדיוק כמו
    _find_first_stop_or_exit_signal. לא קובעת *איזה* מהשלושה קרה בפועל (זה
    מטופל בקוד הקורא, שצריך את סדר-העדיפות המדויק בין השלושה במקרה של
    בר-תיקו) - רק את מיקום הבר הראשון.
    """
    chunk = 4096
    pos = start
    while pos < n:
        end = min(n, pos + chunk)
        seg_low = low[pos:end]
        seg_high = high[pos:end]
        seg_exit_signal = exit_signal_trigger[pos:end]
        hit = (seg_low <= sl_level) | seg_exit_signal | (seg_high >= arm_level)
        if hit.any():
            return pos + int(np.argmax(hit))
        pos = end
        chunk *= 4
    return None


def _find_first_sl_or_distance_exit(low: np.ndarray, distance: np.ndarray, start: int, n: int,
                                    sl_level: float, peak_distance_pts: float,
                                    exit_distance_pts: float, peak_so_far: float,
                                    ) -> tuple[int | None, float]:
    """SMA20 Distance Exit (hammer_breakout, במקום Break Even): מוצאת את
    האינדקס הראשון (>= start) שבו low<=sl_level (Stop Loss) *או* ה-Distance-
    Exit מתקיים - ה-"peak" (המרחק המקסימלי של distance=close-SMA20 מאז
    הכניסה) הגיע אי-פעם ל->=peak_distance_pts *וגם* ה-distance בבר הזה עצמו
    ירד מתחת ל-exit_distance_pts. בניגוד ל-exit_signal_trigger הסטטי
    (Inverted Hammer Exit וכו') ב-שתי הפונקציות למעלה, "האם כבר הושג שיא"
    הוא מצב תלוי-מסלול (תלוי בבר-הכניסה הספציפי של העסקה) - אי אפשר לחשב
    boolean array גלובלי מראש. כל צ'אנק מחשב cummax מקומי
    (np.maximum.accumulate) ומשרשר עם peak_so_far שנכנס מבחוץ (0.0 בקריאה
    הראשונה לכל עסקה) - בדיוק running-max רגיל, רק שמפוצל לצ'אנקים גדלים
    (4096, 16384, ...) מאותה סיבת-ביצועים כמו שתי הפונקציות למעלה. מחזירה
    גם את peak_so_far המעודכן - כדי שקריאה הבאה (אם לא נמצא כאן, עם חלון
    גדול יותר) תמשיך מהשיא הנכון, לא תתחיל מ-0.0 מחדש. distance צפוי כבר
    "מטופל" מ-NaN (למשל np.nan_to_num(..., nan=-np.inf)) לפני הקריאה
    הראשונה - אין טיפול ב-NaN כאן.
    """
    chunk = 4096
    pos = start
    peak = peak_so_far
    while pos < n:
        end = min(n, pos + chunk)
        seg_low = low[pos:end]
        seg_distance = distance[pos:end]
        running_peak = np.maximum(peak, np.maximum.accumulate(seg_distance))
        distance_exit = (running_peak >= peak_distance_pts) & (seg_distance < exit_distance_pts)
        hit = (seg_low <= sl_level) | distance_exit
        if hit.any():
            i = int(np.argmax(hit))
            return pos + i, float(running_peak[i])
        peak = float(running_peak[-1]) if len(running_peak) else peak
        pos = end
        chunk *= 4
    return None, peak


def run_hammer_breakout_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                 size: float = 1.0,
                                 multiplier: float = 1.0,
                                 slippage_ticks: float = 0.0,
                                 commission_per_contract: float = 0.0,
                                 market_order: bool = True,
                                 tick_size: float = MNQ_TICK_SIZE,
                                 stop_loss_pts: float = 20.0,
                                 max_body_pts: float = 7.0,
                                 min_std_before: float = 20.0,
                                 exclude_open_hours: bool = True,
                                 peak_distance_pts: float = 30.0,
                                 exit_distance_pts: float = 15.0,
                                 debug_log_hammers: bool = False) -> dict:
    """long-only, וקטורי לגמרי: פריצת-stop אחרי נר-סיגנל bullish-reversal.

    שלבים:
    1. זיהוי נר-סיגנל על כל המערך בבת אחת (numpy) - שני סוגי-צורה אפשריים,
       "או" ביניהם (מספיק שאחד מתקיים), שניהם יחסית לטווח הבר כולו (לא לגוף):
       rng=high-low, body=|close-open|, lower_wick=min(open,close)-low,
       upper_wick=high-max(open,close), green=close>=open.
         Type 1 (hammer קלאסי):   upper_wick<=0.1*rng AND lower_wick>=0.6*rng
                                   AND body>0 AND green.
         Type 2 (bullish strong): green AND body>=0.4*rng
                                   AND lower_wick>=0.2*rng AND upper_wick<=0.15*rng.
       שני סוגי-הצורה כוללים body>0/green במפורש כתנאי-סף - מוציא דוג'י
       מוחלט ו-rng==0 (בר שטוח) אוטומטית, אחרת type2 עם rng==0 היה "עובר"
       באופן שגוי (0>=0.4*0 אמת).
       עוד תנאים חייבים להתקיים *בנוסף* לצורה (AND, לא חלק מ"או" הצורה, וחלים
       זהה על שני הסוגים):
         - body (|close-open|) של נר-הסיגנל עצמו <= max_body_pts - "נר-ענק"
           (body גדול מדי, גם אם הצורה היחסית-לטווח תקינה) פסול.
         - סטיית-התקן (sample std, ddof=1) של ה-Close על 20 הברים *שלפני*
           נר-הסיגנל (לא כולל אותו) חייבת להיות >= min_std_before - אותה
           טכניקה בדיוק כמו dashboard/pattern_scanner.py:scan_hammers
           (rolling(20).std().shift(1) - "20 הברים שלפני", לא כולל את הבר
           עצמו) - מדלגים על סיגנל שנוצר אחרי מקטע שקט מדי. **סינון-כניסה
           בלבד** - אין לזה שום השפעה על היציאה (ראו שלב 3 למטה, שמבוסס
           כולו על Inverted Hammer Exit, לא על סטיית-תקן).
         - אם exclude_open_hours=True: נר-הסיגנל לא יכול ליפול בחלון
           09:00-11:00 שעון ET (חצי-פתוח [9,11) - שעה 11 עצמה כן מותרת) -
           נמצא בניתוח נפרד שהחלון הזה אחראי לכ-46% מכל עסקאות ה-stop-loss.
           df.index כבר בשעון ET tz-naive (ראו ib_data.py) - משתמשים
           ב-df.index.hour ישירות, בלי המרת אזור-זמן.
    2. רמת-פריצה לכל נר-סיגנל: trigger = high(signal) + tick_size. כניסה רק אם
       גם ה-high וגם ה-close של הבר *הבא ממש* (לא כל בר עתידי - הזמנת-stop
       לבר אחד בלבד) מגיעים ל-trigger או יותר - וקטורי: signal_idx+1 מול
       trigger, השוואת איברים-מול-איברים על שניהם. תנאי-הסגירה הוא תיקון
       מפורש: פתיל שחוצה את ה-trigger אבל נסגר מתחתיו שוב *לא* נחשב פריצה -
       מדלגים על נר-הסיגנל הזה לגמרי (לא מחכים לבר הבא). מחיר המילוי הוא
       ה-trigger עצמו (כפי שהתבקש במפורש - *לא* ה-open של הבר הבא, ולכן,
       בשונה משני המנועים האחרים כאן, slippage *לא* מוחל על הכניסה/יציאה -
       כל המחירים כאן מוגדרים כמחירי-סף/סגירה מדויקים בדרישה, לא
       הזמנות-Market; market_order/slippage_ticks עדיין מתקבלים כפרמטרים
       (עקביות-ממשק עם common_kwargs ב-app.py) אבל לא משפיעים על שום מחיר כאן.
    3. יציאה - שני תנאים במקביל, מי שקורה קודם:
         - Stop Loss: sl_level = entry_price - stop_loss_pts, נבדק מול ה-low
           של כל בר.
         - SMA20 Distance Exit: distance = close - SMA20 (ממוצע-נע פשוט של
           20 ה-Close האחרונים, rolling(20).mean() - מחושב פעם אחת מראש על
           כל המערך, לא תלוי בעסקה הספציפית). "peak" - המרחק המקסימלי של
           distance שנצפה מאז הכניסה (כולל הבר הנוכחי) - הוא מצב תלוי-מסלול
           (בשונה מ-Inverted Hammer Exit הסטטי), ולכן מחושב per-trade עם
           running-max מצטבר (_find_first_sl_or_distance_exit, חלון-חיפוש
           גדל בצ'אנקים כמו שתי הפונקציות האחרות כאן, אבל משרשר cummax בין
           צ'אנקים במקום boolean array סטטי). היציאה מופעלת רק אחרי
           שה-peak הגיע אי-פעם ל->=peak_distance_pts (מאז אותה נקודה,
           "נדלק" לתמיד עבור העסקה הזו, גם אם ה-distance יורד מתחת לזה
           אחר-כך), *וגם* ה-distance בבר הנוכחי ירד מתחת ל-exit_distance_pts
           - יציאה ב-Close של אותו בר.
       סדר-עדיפות בתיקו על אותו בר: אם ה-low נגע ב-sl_level - זו היציאה,
       יציאה ב-sl_level (הסטופ הוא הזמנה תוך-ברית, "מקדים" את ה-Distance
       Exit). אחרת - SMA20 Distance Exit, יציאה ב-Close.
    4. הלולאה היחידה היא על אירועי-פריצה (בדרך כלל אלפים-עשרות-אלפים, לא
       מיליונים), ובלעדיות (מדלגים לאירוע הבא רק אחרי סגירת העסקה הנוכחית) -
       ולכן זמן ריצה מהיר (שניות בודדות על 2.5 מיליון ברים).
    """
    if df is None or len(df) < 3:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    rng = high - low
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    green = close >= open_

    body_ok = body <= max_body_pts  # נר-ענק פסול, בלי קשר לצורה היחסית-לטווח - חל על שני הסוגים

    is_hammer_shape = (
        (upper_wick <= 0.1 * rng) & (lower_wick >= 0.6 * rng)
        & (body > 0) & green & body_ok
    )
    is_strong_bullish_shape = (
        green & (body >= 0.4 * rng)
        & (lower_wick >= 0.2 * rng) & (upper_wick <= 0.15 * rng)
        & (body > 0) & body_ok
    )
    is_signal_shape = is_hammer_shape | is_strong_bullish_shape

    # סטיית-התקן (sample std, ddof=1 - ברירת המחדל של pandas) של ה-Close על 20
    # הברים *שלפני* נר-הסיגנל (לא כולל אותו) - סינון-כניסה בלבד, אין לזה שום
    # קשר ליציאה (Inverted Hammer Exit, ראו למטה): rolling(20).std() בעמדה i
    # מביט על close[i-19..i] (20 ברים שמסתיימים ב-i, כולל); shift(1) מזיז את
    # זה בר אחד קדימה כך שהערך בעמדה i הופך לסטיית-התקן על close[i-20..i-1] -
    # "20 הברים שלפני", בלי לכלול את i עצמו. NaN בברים הראשונים (פחות מ-20
    # ברים לפניהם) -> ההשוואה מולה נותנת False אוטומטית (לא צריך guard נפרד).
    idx_all = np.arange(n)
    std_before_20 = (
        pd.Series(close).rolling(window=20, min_periods=20).std().shift(1).to_numpy()
    )
    std_ok = std_before_20 >= min_std_before

    is_hammer = is_signal_shape & std_ok

    if exclude_open_hours:
        # 09:00-11:00 ET (חצי-פתוח [9,11) - שעה 11 עצמה מותרת) - נמצא בניתוח
        # נפרד שהחלון הזה אחראי לכ-46% מכל עסקאות ה-stop-loss. df.index כבר
        # tz-naive בשעון ET (ראו ib_data.py), אז df.index.hour הוא ישירות
        # השעה בניו יורק - אין צורך בהמרת אזור-זמן נוספת.
        hour_et = df.index.hour.to_numpy()
        in_open_hours = (hour_et >= 9) & (hour_et < 11)
        is_hammer &= ~in_open_hours

    is_hammer &= idx_all <= n - 2  # צריך "בר הבא" קיים כדי לבדוק פריצה

    hammer_idx = np.flatnonzero(is_hammer)

    if debug_log_hammers:
        # דיבוג: OHLC המדויק של *כל* נר שסומן כסיגנל (לא רק כאלה שהובילו
        # לכניסה בפועל) - כדי לוודא שה-mask עצמו עובד נכון, בנפרד לגמרי
        # מהשאלה אם/מתי נפתחת עסקה על סמך הפריצה שאחריו.
        for i in hammer_idx:
            kind = "hammer" if is_hammer_shape[i] else "strong_bullish"
            print(
                f"[hammer-debug] idx={i} time={times[i]} type={kind} "
                f"O={open_[i]} H={high[i]} L={low[i]} C={close[i]} "
                f"rng={rng[i]:.4f} upper_wick={upper_wick[i]:.4f} "
                f"lower_wick={lower_wick[i]:.4f} body={body[i]:.4f} "
                f"close>=open:{bool(green[i])} "
                f"std_before={std_before_20[i]:.4f}",
                flush=True,
            )

    trigger_at_hammer = high[hammer_idx] + tick_size
    # פריצה אמיתית: גם ה-high וגם ה-close של בר-המילוי חייבים להגיע ל-trigger -
    # פתיל שחוצה את הרמה ונסגר מתחתיה שוב לא נחשב (מונע כניסה על "wick" שנכשל
    # מיד). לא ממתינים לבר נוסף - נר-הסיגנל נפסל לגמרי אם זה קורה.
    breakout_hit = (
        (high[hammer_idx + 1] >= trigger_at_hammer)
        & (close[hammer_idx + 1] >= trigger_at_hammer)
    )
    entry_signal_idx = hammer_idx[breakout_hit]  # אינדקס ה-hammer עצמו (לא בר-המילוי)
    entry_trigger = trigger_at_hammer[breakout_hit]

    # SMA20 Distance Exit: distance = close - SMA20 (ממוצע-נע פשוט על 20 ה-Close
    # האחרונים, כולל הבר הנוכחי - rolling(20).mean(), לא shift(1): בשונה
    # מ-std_before_20 למעלה שמשמש לסינון-כניסה על הבר *שלפני* הסיגנל, כאן
    # מדובר ביציאה - הבר הנוכחי עצמו רלוונטי). מחושב פעם אחת מראש על כל
    # המערך, לא תלוי בעסקה הספציפית. NaN בברים הראשונים (פחות מ-20 ברים
    # לפניהם, כולל) מוחלף ב--inf כדי ש-np.maximum.accumulate לא "יזהם" קדימה
    # (np.maximum(x, NaN) הוא NaN) - שקול ל"עדיין לא הגענו לשיא" (אף פעם לא
    # יעבור peak_distance_pts), בלי טיפול-NaN מפורש בתוך _find_first_sl_or_distance_exit.
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().to_numpy()
    distance = np.nan_to_num(close - sma20, nan=-np.inf)

    n_events = len(entry_signal_idx)

    entry_times, exit_times = [], []
    entry_prices, exit_prices = [], []
    entry_fill_idxs, exit_idxs = [], []

    ptr = 0
    while ptr < n_events:
        hammer_i = int(entry_signal_idx[ptr])
        fill_i = hammer_i + 1  # הבר שבו הפריצה בפועל נוגעת ב-trigger
        entry_price = float(entry_trigger[ptr])
        sl_level = entry_price - stop_loss_pts

        # Stop Loss או SMA20 Distance Exit - מי שקורה קודם (ראו
        # _find_first_sl_or_distance_exit וסדר-העדיפות בתיקו בדוקסטרינג).
        # peak_so_far=0.0: תחילת המעקב אחר peak היא fill_i+1 (הבר הראשון
        # שנבדק), לא הבר הקודם - אותה מוסכמה בדיוק כמו stage1 בשני המנועים
        # האחרים (שגם הם מתחילים לבדוק מ-fill_i+1, לא מ-fill_i עצמו).
        exit_i_opt, _peak = _find_first_sl_or_distance_exit(
            low, distance, fill_i + 1, n, sl_level, peak_distance_pts, exit_distance_pts, 0.0
        )

        if exit_i_opt is None:
            exit_i = n - 1
            exit_price = float(close[exit_i])
        elif low[exit_i_opt] <= sl_level:
            exit_i = exit_i_opt
            exit_price = sl_level
        else:
            exit_i = exit_i_opt
            exit_price = float(close[exit_i])

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

        # מדלגים לאירוע-הכניסה הבא *אחרי* שהעסקה הזו נסגרה - בלעדיות, אותה
        # מוסכמה בדיוק כמו שני המנועים האחרים.
        ptr += 1
        while ptr < n_events and entry_signal_idx[ptr] <= exit_i:
            ptr += 1

    n_trades = len(entry_prices)
    entry_prices_arr = np.asarray(entry_prices, dtype=float)
    exit_prices_arr = np.asarray(exit_prices, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    pnl = (exit_prices_arr - entry_prices_arr) * size * multiplier
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    )

    # Equity מלא לכל בר (mark-to-market על עמדה פתוחה) - אותה טכניקה בדיוק
    # כמו שני המנועים האחרים.
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag > 0, (close - entry_price_ffilled) * size * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": np.full(n_trades, size),  # תמיד long
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ================================================ hammer_swing_exit
# מנוע עצמאי לגמרי (כפילות מכוונת מול שאר המנועים בקובץ - כל אסטרטגיה
# מובנית כאן היא פונקציה נפרדת, לא שיתוף קוד עם דגלים). כניסה: hammer/
# strong-bullish -> פריצה ב-close של הבר הבא, ואותו Stop Loss כמו שאר
# המנועים - אבל היציאה השנייה היא סטופ-נגרר מבוסס מבנה-מחיר: "swing low
# break" (הבר הנוכחי שובר מתחת ל-low של הבר *שלפניו*, לא מרחק קבוע
# מהכניסה) - וה-Break Even אופציונלי (USE_BREAK_EVEN), כדי לאפשר להשוות
# תוצאות עם/בלי Break Even.
HAMMER_SWING_EXIT_CODE = """\
# strategy: hammer_swing_exit
# Signal candle sets a breakout trigger = signal bar's high + 1 tick.
# Signal shape - EITHER of two bullish-reversal patterns qualifies:
#   Hammer: upper_wick <= 10% of range AND lower_wick >= 60% of range AND
#     close >= open AND body > 0.
#   Strong bullish: close >= open AND body >= 40% of range AND
#     lower_wick >= 20% of range AND upper_wick <= 15% of range.
# Extra filter: abs(close-open) <= MAX_BODY_PTS on the signal candle itself
#   (applies to both shapes). No volatility or time-of-day entry filter.
# Long entry: only if the very next bar's close crosses above the trigger -
#             single-bar stop order, not an open-ended wait. Fill price is
#             the trigger itself, not the next bar's open.
# Exit - whichever comes first:
#   1. Stop Loss: price drops STOP_LOSS_PTS below entry price. This is the
#      INITIAL stop only - see Break Even below, which can replace it.
#   2. Swing Low Exit: on any bar after entry, if that bar's LOW breaks
#      below the PREVIOUS bar's LOW, exit at that bar's CLOSE. A trailing
#      stop based on price structure, not a fixed distance from entry.
#   3. Break Even (optional - see USE_BREAK_EVEN): once price rises
#      BREAK_EVEN_PTS above entry, the stop moves from STOP_LOSS_PTS below
#      entry up to breakeven (entry price itself) - the trade can then only
#      exit at breakeven or better. When USE_BREAK_EVEN is False, this step
#      is skipped entirely: only Stop Loss and Swing Low Exit apply.
STOP_LOSS_PTS = 5       # input
MAX_BODY_PTS = 7        # input
BREAK_EVEN_PTS = 10     # input
USE_BREAK_EVEN = False  # input
"""


def run_hammer_swing_exit_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                   size: float = 1.0,
                                   multiplier: float = 1.0,
                                   slippage_ticks: float = 0.0,
                                   commission_per_contract: float = 0.0,
                                   market_order: bool = True,
                                   tick_size: float = MNQ_TICK_SIZE,
                                   stop_loss_pts: float = 5.0,
                                   max_body_pts: float = 7.0,
                                   break_even_pts: float = 10.0,
                                   use_break_even: bool = True,
                                   debug_log_hammers: bool = False) -> dict:
    """long-only, וקטורי לגמרי - עצמאי לחלוטין (כפילות מכוונת, לא שיתוף קוד
    עם דגלים - כל מנוע מובנה בקובץ הזה הוא פונקציה עצמאית).

    שלבים:
    1-2. זיהוי-סיגנל וכניסה: hammer/strong-bullish -> trigger=high+tick ->
       כניסה כש-close של הבר הבא >= trigger.
    3. יציאה - שלושה תנאים, מי שקורה קודם:
       - Stop Loss: sl_level = entry_price - stop_loss_pts, מול ה-low.
       - Swing Low Exit ("סטופ נגרר מבוסס מבנה"): swing_low_break[i] =
         low[i] < low[i-1] (i>=1; אין "בר קודם" ל-i=0, False שם) - מחושב
         מראש על כל המערך, לא תלוי בעסקה הספציפית.
         יציאה ב-Close של הבר שבו זה קרה.
       - Break Even: רק אם use_break_even=True - חיפוש דו-שלבי
         (_find_first_sl_exit_signal_or_arm ואז שלב 2 אחרי armed), עם
         swing_low_break כתנאי-היציאה השני. כש-use_break_even=False השלב
         הזה לא קיים בכלל: מחפשים ישירות עם _find_first_stop_or_exit_signal
         (Stop Loss + Swing Low Exit בלבד, בלי arm_level/high בכלל) - שלב
         יחיד, לא שניים.
    4. הלולאה היחידה היא על אירועי-פריצה, ובלעדיות - אותו סדר-גודל של זמן
       ריצה (שניות בודדות על 2.5 מיליון ברים) כמו שאר המנועים.
    """
    if df is None or len(df) < 3:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    rng = high - low
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    green = close >= open_

    body_ok = body <= max_body_pts  # נר-ענק פסול, בלי קשר לצורה היחסית-לטווח - חל על שני הסוגים

    is_hammer_shape = (
        (upper_wick <= 0.1 * rng) & (lower_wick >= 0.6 * rng)
        & (body > 0) & green & body_ok
    )
    is_strong_bullish_shape = (
        green & (body >= 0.4 * rng)
        & (lower_wick >= 0.2 * rng) & (upper_wick <= 0.15 * rng)
        & (body > 0) & body_ok
    )
    is_signal_shape = is_hammer_shape | is_strong_bullish_shape

    idx_all = np.arange(n)
    is_hammer = is_signal_shape & (idx_all <= n - 2)  # צריך "בר הבא" קיים כדי לבדוק פריצה

    hammer_idx = np.flatnonzero(is_hammer)

    if debug_log_hammers:
        for i in hammer_idx:
            kind = "hammer" if is_hammer_shape[i] else "strong_bullish"
            print(
                f"[hammer-swing-exit-debug] idx={i} time={times[i]} type={kind} "
                f"O={open_[i]} H={high[i]} L={low[i]} C={close[i]} "
                f"rng={rng[i]:.4f} upper_wick={upper_wick[i]:.4f} "
                f"lower_wick={lower_wick[i]:.4f} body={body[i]:.4f} "
                f"close>=open:{bool(green[i])}",
                flush=True,
            )

    trigger_at_hammer = high[hammer_idx] + tick_size
    # פריצה: רק close של בר-המילוי חייב להגיע ל-trigger - גורר high>=trigger
    # אוטומטית (close<=high תמיד), אז אין תנאי נפרד על ה-high.
    breakout_hit = close[hammer_idx + 1] >= trigger_at_hammer
    entry_signal_idx = hammer_idx[breakout_hit]  # אינדקס ה-hammer עצמו (לא בר-המילוי)
    entry_trigger = trigger_at_hammer[breakout_hit]

    # Swing Low Exit: הבר הנוכחי שובר מתחת ל-low של הבר שלפניו - מחושב מראש
    # על כל המערך, לא תלוי בעסקה הספציפית (אותה מוסכמה בדיוק כמו
    # inv_hammer_exit_trigger בשני המנועים האחרים). אין "בר קודם" ל-i=0.
    swing_low_break = np.zeros(n, dtype=bool)
    swing_low_break[1:] = low[1:] < low[:-1]

    n_events = len(entry_signal_idx)

    entry_times, exit_times = [], []
    entry_prices, exit_prices = [], []
    entry_fill_idxs, exit_idxs = [], []

    ptr = 0
    while ptr < n_events:
        hammer_i = int(entry_signal_idx[ptr])
        fill_i = hammer_i + 1  # הבר שבו הפריצה בפועל נוגעת ב-trigger
        entry_price = float(entry_trigger[ptr])
        sl_level = entry_price - stop_loss_pts

        if use_break_even:
            arm_level = entry_price + break_even_pts
            stage1_i = _find_first_sl_exit_signal_or_arm(low, high, swing_low_break,
                                                         fill_i + 1, n, sl_level, arm_level)

            if stage1_i is None:
                exit_i = n - 1
                exit_price = float(close[exit_i])
            elif low[stage1_i] <= sl_level:
                exit_i = stage1_i
                exit_price = sl_level
            elif swing_low_break[stage1_i]:
                exit_i = stage1_i
                exit_price = float(close[exit_i])
            elif low[stage1_i] <= entry_price:
                exit_i = stage1_i
                exit_price = entry_price
            else:
                stage2_i = _find_first_stop_or_exit_signal(low, swing_low_break,
                                                           stage1_i + 1, n, entry_price)
                if stage2_i is None:
                    exit_i = n - 1
                    exit_price = float(close[exit_i])
                else:
                    exit_i = stage2_i
                    be_hit_here = low[exit_i] <= entry_price
                    exit_price = entry_price if be_hit_here else float(close[exit_i])
        else:
            # Break Even מבוטל לגמרי (USE_BREAK_EVEN=False) - Stop Loss +
            # Swing Low Exit בלבד, שלב יחיד, בלי arm_level/high בכלל.
            found_i = _find_first_stop_or_exit_signal(low, swing_low_break,
                                                      fill_i + 1, n, sl_level)
            if found_i is None:
                exit_i = n - 1
                exit_price = float(close[exit_i])
            elif low[found_i] <= sl_level:
                exit_i = found_i
                exit_price = sl_level
            else:
                exit_i = found_i
                exit_price = float(close[exit_i])

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

        # מדלגים לאירוע-הכניסה הבא *אחרי* שהעסקה הזו נסגרה - בלעדיות.
        ptr += 1
        while ptr < n_events and entry_signal_idx[ptr] <= exit_i:
            ptr += 1

    n_trades = len(entry_prices)
    entry_prices_arr = np.asarray(entry_prices, dtype=float)
    exit_prices_arr = np.asarray(exit_prices, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    pnl = (exit_prices_arr - entry_prices_arr) * size * multiplier
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    )

    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag > 0, (close - entry_price_ffilled) * size * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": np.full(n_trades, size),  # תמיד long
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ======================================================= sma20_mean_reversion
# מנוע מובנה חדש ועצמאי לגמרי - הראשון בקובץ שפותח גם short (לא רק long). Size
# חתום (חיובי=long, שלילי=short) - כבר נתמך במלואו בתשתית הקיימת
# (to_markers/ui_components.performance_summary_html/trades_table_html, ראו
# שם "Size > 0"/"Size < 0"), אף שאף מנוע קודם לא הפיק בפועל Size שלילי. נבחר
# ב-app.py:run_or_switch לפי סימון "# strategy: sma20_mean_reversion" (ראו
# detect_builtin_strategy למעלה).
SMA20_MEAN_REVERSION_CODE = """\
# strategy: sma20_mean_reversion
# Signal - distance = close - SMA20 (20-bar simple moving average), computed
#   on every bar.
# Long entry: distance <= -MIN_DISTANCE_PTS (price MIN_DISTANCE_PTS or more
#             BELOW SMA20) - enter long at the very next bar's OPEN.
# Short entry: distance >= MIN_DISTANCE_PTS (price MIN_DISTANCE_PTS or more
#              ABOVE SMA20) - enter short at the very next bar's OPEN.
# Only one trade open at a time (long and short signals share one position).
# Exit - two conditions, whichever comes first:
#   1. Stop Loss: Long exits if price drops STOP_LOSS_PTS below entry
#      (checked on low). Short exits if price rises STOP_LOSS_PTS above entry
#      (checked on high).
#   2. Mean Reversion Exit: Long exits when close >= SMA20 (price returned to
#      or above SMA20). Short exits when close <= SMA20 (price returned to or
#      below SMA20). Exit at that bar's close.
MIN_DISTANCE_PTS = 20  # input
STOP_LOSS_PTS = 15     # input
"""


def _find_first_short_stop_or_exit_signal(high: np.ndarray, exit_signal_trigger: np.ndarray, start: int,
                                          n: int, sl_level: float) -> int | None:
    """גרסת-מראה של _find_first_stop_or_exit_signal, לפוזיציית SHORT: מוצאת
    את האינדקס הראשון (>= start) שבו high>=sl_level (סטופ, *מעל* הכניסה,
    כיוון-ההשוואה הפוך ל-long) *או* exit_signal_trigger[i] אמת - אותו
    חלון-חיפוש-גדל בדיוק (4096, 16384, ...), רק high במקום low.
    """
    chunk = 4096
    pos = start
    while pos < n:
        end = min(n, pos + chunk)
        seg_high = high[pos:end]
        seg_exit_signal = exit_signal_trigger[pos:end]
        hit = (seg_high >= sl_level) | seg_exit_signal
        if hit.any():
            return pos + int(np.argmax(hit))
        pos = end
        chunk *= 4
    return None


def run_sma20_mean_reversion_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                      size: float = 1.0,
                                      multiplier: float = 1.0,
                                      slippage_ticks: float = 0.0,
                                      commission_per_contract: float = 0.0,
                                      market_order: bool = True,
                                      tick_size: float = MNQ_TICK_SIZE,
                                      min_distance_pts: float = 20.0,
                                      stop_loss_pts: float = 15.0,
                                      debug_log_signals: bool = False) -> dict:
    """long+short, וקטורי לגמרי: חזרה-לממוצע מ-SMA20. המנוע העצמאי הראשון כאן
    שפותח גם short - Size חתום (חיובי=long/שלילי=short), נתמך כבר בתשתית
    הקיימת (to_markers/ui_components), לא נדרש שום שינוי שם.

    שלבים:
    1. distance = close - SMA20 (rolling(20).mean() על ה-Close) - מחושב פעם
       אחת על כל המערך.
    2. סיגנל-כניסה: long אם distance[i]<=-min_distance_pts, short אם
       distance[i]>=min_distance_pts (זרים הדדית - min_distance_pts חיובי) -
       שני סוגי-הסיגנל ממוזגים לרצף-אירועים אחד ממוין-בזמן, כי רק פוזיציה
       אחת פתוחה בכל רגע (long/short חולקים את אותה "בלעדיות"). כניסה
       בלתי-מותנית ב-open של הבר הבא - לא הזמנת-stop.
    3. יציאה - שני תנאים במקביל, מי שקורה קודם:
       - Stop Loss: long נבדק מול ה-low (sl_level=entry-stop_loss_pts), short
         מול ה-high (sl_level=entry+stop_loss_pts) -
         _find_first_short_stop_or_exit_signal הוא גרסת-המראה של
         _find_first_stop_or_exit_signal (הקיים, המשותף) לכיוון ההפוך.
       - Mean Reversion Exit: long כש-close[i]>=SMA20[i], short כש-
         close[i]<=SMA20[i] (שני boolean arrays סטטיים, מחושבים מראש פעם אחת
         על כל המערך - לא תלויים בעסקה הספציפית). יציאה ב-Close של אותו בר.
    4. הלולאה היחידה היא על אירועי-סיגנל הממוזגים (long+short יחד), ובלעדיות
       (מדלגים לאירוע הבא רק אחרי סגירת העסקה הנוכחית, בלי קשר לכיוון) -
       ולכן זמן ריצה מהיר (שניות בודדות על 2.5 מיליון ברים).
    """
    if df is None or len(df) < 21:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().to_numpy()
    distance = close - sma20

    idx_all = np.arange(n)
    can_enter = idx_all <= n - 2  # צריך "בר הבא" קיים כדי להיכנס ב-open שלו

    long_signal = (distance <= -min_distance_pts) & can_enter
    short_signal = (distance >= min_distance_pts) & can_enter

    signal_idx = np.flatnonzero(long_signal | short_signal)
    is_long_signal = long_signal[signal_idx]

    if debug_log_signals:
        for k, i in enumerate(signal_idx):
            kind = "long" if is_long_signal[k] else "short"
            print(
                f"[sma20-mean-reversion-debug] idx={i} time={times[i]} type={kind} "
                f"close={close[i]:.2f} sma20={sma20[i]:.2f} distance={distance[i]:.2f}",
                flush=True,
            )

    # Mean Reversion Exit triggers - סטטיים, מחושבים מראש פעם אחת על כל
    # המערך, לא תלויים בעסקה הספציפית (אותה מוסכמה בדיוק כמו
    # inv_hammer_exit_trigger בשאר המנועים).
    cross_up = close >= sma20     # long exit trigger
    cross_down = close <= sma20   # short exit trigger

    n_events = len(signal_idx)

    entry_times, exit_times = [], []
    entry_prices, exit_prices = [], []
    entry_fill_idxs, exit_idxs = [], []
    signed_sizes = []

    ptr = 0
    while ptr < n_events:
        signal_i = int(signal_idx[ptr])
        is_long = bool(is_long_signal[ptr])
        fill_i = signal_i + 1  # הבר שבו נכנסים - ב-open שלו, ללא תנאי
        entry_price = float(open_[fill_i])

        if is_long:
            sl_level = entry_price - stop_loss_pts
            exit_i_opt = _find_first_stop_or_exit_signal(low, cross_up, fill_i + 1, n, sl_level)
            if exit_i_opt is None:
                exit_i = n - 1
                exit_price = float(close[exit_i])
            elif low[exit_i_opt] <= sl_level:
                exit_i = exit_i_opt
                exit_price = sl_level
            else:
                exit_i = exit_i_opt
                exit_price = float(close[exit_i])
        else:
            sl_level = entry_price + stop_loss_pts
            exit_i_opt = _find_first_short_stop_or_exit_signal(high, cross_down, fill_i + 1, n, sl_level)
            if exit_i_opt is None:
                exit_i = n - 1
                exit_price = float(close[exit_i])
            elif high[exit_i_opt] >= sl_level:
                exit_i = exit_i_opt
                exit_price = sl_level
            else:
                exit_i = exit_i_opt
                exit_price = float(close[exit_i])

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)
        signed_sizes.append(size if is_long else -size)

        # מדלגים לאירוע-הכניסה הבא *אחרי* שהעסקה הזו נסגרה - בלעדיות, בלי
        # קשר לכיוון (long/short חולקים את אותה בלעדיות).
        ptr += 1
        while ptr < n_events and signal_idx[ptr] <= exit_i:
            ptr += 1

    n_trades = len(entry_prices)
    entry_prices_arr = np.asarray(entry_prices, dtype=float)
    exit_prices_arr = np.asarray(exit_prices, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)
    signed_size_arr = np.asarray(signed_sizes, dtype=float)
    is_long_arr = signed_size_arr > 0

    price_diff = exit_prices_arr - entry_prices_arr
    signed_diff = np.where(is_long_arr, price_diff, -price_diff)
    pnl = signed_diff * size * multiplier
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    return_frac = np.where(
        is_long_arr,
        np.divide(price_diff, entry_prices_arr, out=np.zeros(n_trades), where=entry_prices_arr != 0),
        np.divide(-price_diff, entry_prices_arr, out=np.zeros(n_trades), where=entry_prices_arr != 0),
    )

    # Equity מלא לכל בר (mark-to-market על עמדה פתוחה) - אותה טכניקה כמו שאר
    # המנועים, עם סימן-כיוון (direction_ffilled: 1.0=long/-1.0=short) על
    # ה-unrealized, כי כאן זה יכול להיות גם short.
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()

    direction_series = pd.Series(np.nan, index=np.arange(n))
    direction_series.iloc[entry_idx_arr] = np.where(is_long_arr, 1.0, -1.0)
    direction_ffilled = direction_series.ffill().fillna(1.0).to_numpy()

    unrealized = np.where(
        position_flag > 0,
        (close - entry_price_ffilled) * direction_ffilled * size * multiplier,
        0.0,
    )

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": signed_size_arr,  # חיובי=long, שלילי=short
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ================================================ down_streak_reversion
# מנוע עצמאי לגמרי (כפילות מכוונת, לא שיתוף קוד עם דגלים - כל מנוע מובנה
# בקובץ הזה הוא פונקציה נפרדת). המנוע *הפשוט* ביותר מבין המובנים כאן: יציאה
# טהורה מבוססת-זמן (בלי Stop Loss/יעד/trailing) - אין צורך בשום חיפוש
# חלון-גדל (_find_first_*), כי אינדקס-היציאה של כל עסקה ידוע מיד ברגע
# הכניסה (entry_idx+hold_bars), לא תלוי בשום דבר שקורה אחר-כך במחיר. נבחר
# ב-app.py:run_or_switch לפי סימון "# strategy: down_streak_reversion".
DOWN_STREAK_REVERSION_CODE = """\
# strategy: down_streak_reversion
# Down bar at i: close[i] < low[i-1] (i=0 has no prior bar -> not a down bar).
# Down streak of length STREAK_LENGTH: exactly STREAK_LENGTH consecutive down
#   bars, bounded by non-down bars on both sides (the bar right before the
#   streak is NOT a down bar, and the streak is broken by the very next bar -
#   caught the moment it ends, not mid-run).
# Long entry: at the OPEN of the bar immediately after the streak ends -
#             unconditional (not a stop/trigger order like the hammer
#             strategies - always fills).
# Exit - depends on USE_PRIOR_LOW_EXIT:
#   OFF (default): purely time-based - close the position at the CLOSE of the
#     bar HOLD_BARS bars after the entry bar (HOLD_BARS=1 exits one bar after
#     entry). This is the only exit condition unless USE_STOP_LOSS is also on.
#   ON: REPLACES the time exit. Exit as soon as a bar's LOW trades below the
#     LOW of the bar right before it (intrabar touch) - exit at that broken
#     level (the previous bar's low), not at the close. The level trails:
#     each new bar is compared to the one just before it. If it never
#     triggers, the position is held to the close of the last available bar.
# Optional stop-loss (USE_STOP_LOSS, works alongside whichever exit above is
#   active): if price trades STOP_LOSS_PTS points below the entry price
#   (intrabar touch), exit at that stop level. If the stop and the other
#   exit would both trigger on the same bar, the stop takes priority
#   (worst-case fill assumption).
# Only one position open at a time: a streak that completes while a position
# is already open is skipped entirely (not queued).
# Optional entry filter (USE_VOLATILITY_FILTER): only affects WHICH streak
#   signals are taken - never the entry price, the exit logic, or anything
#   else. Volatility = standard deviation of the last 20 one-minute returns
#   (return[i] = close[i]/close[i-1]-1), using the 20 bars up to and
#   including the streak-end bar (the bar right before entry) - no look-ahead
#   into the entry bar itself. When ON, a streak signal is skipped entirely
#   (not queued, not delayed) if that volatility is below
#   VOLATILITY_THRESHOLD. When OFF, every streak signal is taken, exactly as
#   without this filter.
STREAK_LENGTH = 3              # input
HOLD_BARS = 1                  # input
USE_PRIOR_LOW_EXIT = True      # input
USE_STOP_LOSS = False          # input
STOP_LOSS_PTS = 5              # input
USE_VOLATILITY_FILTER = True   # input
VOLATILITY_THRESHOLD = 0.0004  # input
"""


def run_down_streak_reversion_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                       size: float = 1.0,
                                       multiplier: float = 1.0,
                                       slippage_ticks: float = 0.0,
                                       commission_per_contract: float = 0.0,
                                       market_order: bool = True,
                                       tick_size: float = MNQ_TICK_SIZE,
                                       streak_length: int = 3,
                                       hold_bars: int = 1,
                                       use_prior_low_exit: bool = False,
                                       use_stop_loss: bool = False,
                                       stop_loss_pts: float = 5.0,
                                       use_volatility_filter: bool = True,
                                       volatility_threshold: float = 0.0004,
                                       debug_log_streaks: bool = False) -> dict:
    """long-only, וקטורי לגמרי - עצמאי לחלוטין. ראו הערת המודול למעלה: זה
    היה המנוע הפשוט ביותר מבין המובנים כאן (יציאה תנאי-זמן טהור בלבד) -
    עדיין נכון כשברירות-המחדל של שני הדגלים החדשים (use_prior_low_exit,
    use_stop_loss) כבויות: הענף הזה (ראו למטה) זהה שורה-בשורה למה שהיה כאן
    לפני ההוספה, כדי לא לשנות שום התנהגות קיימת.

    שלבים:
    1. down bar: close[i] < low[i-1] (i=0 - False, אין בר קודם) - מחושב
       וקטורית פעם אחת על כל המערך.
    2. down streak: אורך-רצף-רצוף וקטורי (runlen[i] = i - last_non_down_idx,
       טריק "האינדקס-הלא-אדום האחרון" + np.maximum.accumulate - אותה טכניקה
       שכבר שימשה בניתוחי-המחקר הנפרדים על האסטרטגיה הזו). streak-end בדיוק
       באורך streak_length: runlen[e]==streak_length (מבטיח אוטומטית
       שהבר-לפני-הרצף אינו down - זו הבניית runlen עצמה, לא בדיקה נפרדת)
       *וגם* הבר הבא (e+1) אינו down (runlen[e+1]==0) - "נתפס ברגע שהוא
       נגמר", לא נר-אמצע של רצף ארוך יותר.
    3. כניסה: ב-open של הבר מיד אחרי streak-end (e+1), ללא תנאי (לא
       הזמנת-stop כמו המנועים האחרים - כניסה מיידית, בדיוק כמו
       run_fast_sma_crossover_backtest).
    4. יציאה - תלוי בדגלים (ברירת המחדל של שניהם - False):
       - use_prior_low_exit=False, use_stop_loss=False (ברירת המחדל, ההתנהגות
         המקורית): exit_idx = entry_idx + hold_bars בדיוק, יציאה ב-Close של
         אותו בר - בלי שום חיפוש-חלון, קוד זהה למה שהיה כאן לפני התכונה הזו.
       - use_stop_loss=True בלבד: אותו יציאת-זמן קבועה כמו למעלה, אבל נבדק
         first-hit של low<=sl_level בחלון *חסום* [fill_i+1, fill_i+hold_i] -
         אם נמצא, יציאה מוקדמת ב-sl_level; אחרת ממשיכים כרגיל ליציאת-הזמן.
         חלון קטן וידוע-מראש (hold_bars) - לא צריך את טכניקת-הצ'אנקים-הגדלים.
       - use_prior_low_exit=True (עם/בלי use_stop_loss): מחליף לגמרי את
         יציאת-הזמן. prior_low_break[i] = low[i] < low[i-1] (i>=1, אחרת
         False) - מערך סטטי, לא תלוי-עסקה, מחושב פעם אחת על כל המערך (אותה
         מוסכמה בדיוק כמו swing_low_break ב-run_hammer_swing_exit_backtest).
         מחפשים first-hit ב-_find_first_stop_or_exit_signal (כבר קיימת,
         משמשת גם את מנועי-ה-hammer) בחלון *לא-חסום* [fill_i+1, n) -
         sl_level=entry_price-stop_loss_pts אם use_stop_loss, אחרת -inf (כדי
         שתנאי ה-Stop Loss בתוך הפונקציה המשותפת לעולם לא יתקיים). בתיקו על
         אותו בר - ה-Stop Loss מנצח (low[i]<=sl_level נבדק קודם), בדיוק כפי
         שהתבקש. יציאה: אם ה-Stop Loss ניצח - ב-sl_level; אחרת -
         ב-low[i-1] (הרמה שנשברה בפועל, לא ה-close של הבר). לא נמצא כלל עד
         סוף הנתונים - מחזיקים עד ה-Close של הבר האחרון (אותה מוסכמה בדיוק
         כמו מנועי ה-hammer כש-_find_first_stop_or_exit_signal מחזירה None).
    5. בלעדיות: אותה מוסכמה בדיוק כמו שאר המנועים - מדלגים על אירוע-כניסה
       הבא כל עוד entry_idx שלו <= exit_idx של העסקה הפתוחה.

    סינון-כניסה אופציונלי (use_volatility_filter, ברירת המחדל True): מיושם
    *לפני* לולאת-העסקאות, כמסכה על entry_signal_idx בלבד - לא נוגע במחיר-
    כניסה, ביציאה, או בכל דבר אחר בהמשך. volatility = סטיית-התקן (ddof=1,
    ברירת המחדל של pandas) של 20 התשואות-של-דקה האחרונות (תשואה[i] =
    close[i]/close[i-1]-1), על החלון המסתיים *בבר-סוף-הרצף* (fill_i-1, הבר
    ממש לפני הכניסה) - בדיוק "20 הברים עד ובכולל בר-סוף-הרצף", בלי הצצה
    לתוך בר-הכניסה עצמו. מחושב וקטורית פעם אחת על כל המערך
    (pandas rolling(20).std(), אותה טכניקה בדיוק כמו std_before_20 ב-
    run_hammer_breakout_backtest) ואז נבדק per-candidate על ידי אינדוקס
    ב-entry_signal_idx-1. NaN (פחות מ-20 ברים לפניו - רק ממש בתחילת
    ההיסטוריה) מוחלף ב--inf לפני ההשוואה - סיגנל כזה נכשל בסינון (לא ניתן
    לאשר שהוא עומד בסף, אז נפסל) כשהדגל דלוק. כש-use_volatility_filter=False,
    entry_signal_idx לא נוגע בכלל - זהה לחלוטין למה שהיה כאן לפני התכונה הזו.

    עלויות: בשונה מהמנועים מבוססי-hammer (שם מחיר-הכניסה/היציאה מוגדר
    כרמת-סף/סגירה מדויקת לפי הדרישה, ולכן slippage לא מוחל בהם בכלל) -
    כאן הכניסה היא מילוי-שוק בלתי-מותנה ב-open, מבנית זהה לגמרי לכניסה
    ב-run_fast_sma_crossover_backtest - אז slippage_ticks מוחל כאן באותה
    צורה בדיוק (רק על הזמנת Market; Limit ממלאת במחיר המבוקש בהגדרה, בלי
    סליפג'). הערה: slippage מוחל *אחרי* קביעת exit_price (לרבות sl_level/
    prior-low level) - אותה מוסכמה בדיוק כמו הענף המקורי, לא שונה כאן.
    """
    if df is None or len(df) < int(streak_length) + 2:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    times = df.index.to_numpy()
    open_ = df["Open"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(close)

    down = np.zeros(n, dtype=bool)
    down[1:] = close[1:] < low[:-1]

    idx_all = np.arange(n)
    last_non_down = np.maximum.accumulate(np.where(~down, idx_all, -1))
    runlen = idx_all - last_non_down

    streak_len_i = int(streak_length)
    hold_i = int(hold_bars)

    is_streak_end = (
        (idx_all + 1 < n) & (runlen == streak_len_i)
        & (runlen[np.clip(idx_all + 1, 0, n - 1)] == 0)
    )
    streak_end_idx = np.flatnonzero(is_streak_end)

    if debug_log_streaks:
        for e in streak_end_idx:
            nxt = times[e + 1] if e + 1 < n else None
            print(f"[down-streak-debug] streak_end_idx={e} time={times[e]} "
                  f"runlen={runlen[e]} entry_bar_time={nxt}", flush=True)

    entry_signal_idx = streak_end_idx + 1                     # הבר שבו נכנסים (open שלו) - לא streak_end עצמו
    entry_signal_idx = entry_signal_idx[entry_signal_idx + hold_i < n]  # בר-היציאה חייב להתקיים

    if use_volatility_filter:
        # סינון-כניסה אופציונלי - ראו דוקסטרינג. מסכה על entry_signal_idx
        # בלבד, לפני לולאת-העסקאות - לא משנה מחיר-כניסה/יציאה של שום עסקה
        # שכן נכנסת, רק מסיר מועמדים. וקטורי פעם אחת על כל המערך.
        ret = pd.Series(close).pct_change()
        vol20 = ret.rolling(window=20, min_periods=20).std().to_numpy()
        vol20_filled = np.nan_to_num(vol20, nan=-np.inf)  # אין-מספיק-היסטוריה -> נכשל בסינון
        # ה-20 ברים "עד ובכולל בר-סוף-הרצף" = fill_i-1 (הבר ממש לפני הכניסה) -
        # entry_signal_idx-1 בדיוק, בלי הצצה לתוך בר-הכניסה עצמו.
        vol_at_streak_end = vol20_filled[entry_signal_idx - 1]
        entry_signal_idx = entry_signal_idx[vol_at_streak_end >= float(volatility_threshold)]

    n_events = len(entry_signal_idx)

    # prior_low_break: מערך סטטי, לא תלוי-עסקה, מחושב פעם אחת - אותה מוסכמה
    # בדיוק כמו swing_low_break ב-run_hammer_swing_exit_backtest. רלוונטי רק
    # כש-use_prior_low_exit=True (ראו דוקסטרינג) - מחושב בכל מקרה (זול,
    # וקטורי) כדי לא להסתעף בקוד לפי הדגל בשלב הזה.
    prior_low_break = np.zeros(n, dtype=bool)
    prior_low_break[1:] = low[1:] < low[:-1]

    use_prior_low_exit_b = bool(use_prior_low_exit)
    use_stop_loss_b = bool(use_stop_loss)
    stop_loss_pts_f = float(stop_loss_pts)

    entry_times, exit_times = [], []
    entry_prices, exit_prices = [], []
    entry_fill_idxs, exit_idxs = [], []

    ptr = 0
    while ptr < n_events:
        fill_i = int(entry_signal_idx[ptr])   # כניסה ב-open של הבר הזה, בלי תנאי
        entry_price = float(open_[fill_i])
        sl_level = entry_price - stop_loss_pts_f

        if use_prior_low_exit_b:
            # מחליף לגמרי את יציאת-הזמן - חלון *לא-חסום* עד סוף הנתונים.
            # sl_level אמיתי רק אם use_stop_loss; אחרת -inf כדי שהתנאי
            # low<=sl_level בתוך הפונקציה המשותפת לעולם לא יתקיים.
            search_sl_level = sl_level if use_stop_loss_b else -np.inf
            exit_i_opt = _find_first_stop_or_exit_signal(
                low, prior_low_break, fill_i + 1, n, search_sl_level
            )
            if exit_i_opt is None:
                exit_i = n - 1
                exit_price = float(close[exit_i])
            elif use_stop_loss_b and low[exit_i_opt] <= sl_level:
                # תיקו על אותו בר - ה-Stop Loss מנצח (worst-case fill), בדיוק כפי שהתבקש.
                exit_i = exit_i_opt
                exit_price = sl_level
            else:
                exit_i = exit_i_opt
                exit_price = float(low[exit_i - 1])  # הרמה שנשברה בפועל, לא ה-close של הבר
        elif use_stop_loss_b:
            # יציאת-הזמן הרגילה נשארת, אבל נבדק first-hit של הסטופ בחלון
            # *חסום* [fill_i+1, fill_i+hold_i] - חלון קטן וידוע-מראש, בלי
            # צורך בטכניקת-הצ'אנקים-הגדלים.
            exit_i = fill_i + hold_i
            window_low = low[fill_i + 1: exit_i + 1]
            hit = window_low <= sl_level
            if hit.any():
                exit_i = fill_i + 1 + int(np.argmax(hit))
                exit_price = sl_level
            else:
                exit_price = float(close[exit_i])
        else:
            # ברירת המחדל המקורית - זהה שורה-בשורה למה שהיה כאן לפני התכונה הזו.
            exit_i = fill_i + hold_i
            exit_price = float(close[exit_i])     # תנאי-יציאה יחיד: מבוסס-זמן בלבד

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

        # מדלגים לאירוע-הכניסה הבא *אחרי* שהעסקה הזו נסגרה - בלעדיות, אותה
        # מוסכמה בדיוק כמו שאר המנועים.
        ptr += 1
        while ptr < n_events and entry_signal_idx[ptr] <= exit_i:
            ptr += 1

    n_trades = len(entry_prices)
    raw_entry_prices_arr = np.asarray(entry_prices, dtype=float)
    raw_exit_prices_arr = np.asarray(exit_prices, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    # סליפג': מוחל רק על הזמנת Market (ראו דוקסטרינג) - מחיר-כניסה גרוע יותר
    # (גבוה יותר), מחיר-יציאה גרוע יותר (נמוך יותר) - אותה מוסכמה בדיוק כמו
    # run_fast_sma_crossover_backtest.
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    entry_prices_arr = raw_entry_prices_arr + slippage_amount
    exit_prices_arr = raw_exit_prices_arr - slippage_amount

    pnl = (exit_prices_arr - entry_prices_arr) * size * multiplier
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    )

    # Equity מלא לכל בר (mark-to-market על עמדה פתוחה) - אותה טכניקה בדיוק
    # כמו שאר המנועים.
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag > 0, (close - entry_price_ffilled) * size * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": np.full(n_trades, size),  # תמיד long
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ================================================ overnight_val_reversion
# מנוע עצמאי לגמרי (כפילות מכוונת, לא שיתוף קוד עם דגלים - כל מנוע מובנה
# בקובץ הזה הוא פונקציה נפרדת). בשונה משאר המנועים כאן - הסיגנל הוא ברמת-
# יום (פרופיל-נפח), לא ברמת-בר בודד, ולכן יש כאן לולאת-ימים (כ-1,900
# איטרציות על כל ההיסטוריה, לא לולאת-ברים) - זה המנוע היחיד בקובץ עם
# המבנה הזה. נבחר ב-app.py:run_or_switch לפי סימון "# strategy:
# overnight_val_reversion".
OVERNIGHT_VAL_REVERSION_CODE = """\
# strategy: overnight_val_reversion
# Setup (per day, no look-ahead): volume profile over the OVERNIGHT window -
#   from 16:00 ET (prior day's RTH close) to 09:30 ET (today's RTH open).
#   5-point bins, volume assigned by each bar's Close. POC = highest-volume
#   bin. Value Area = contiguous range around the POC holding 70% of the
#   overnight volume (single-bin-step expansion from the POC, comparing the
#   next bin above vs below at each step, ties toward the upper bin). VAH =
#   top of the Value Area, VAL = bottom.
# VAL = bottom of that Value Area. Already look-ahead-free as-is - the
#   overnight window ends strictly before today's RTH open, so no further
#   day-shift is needed (unlike a "prior day's RTH POC" signal).
# Entry: every day (no POC-position filter), long on EITHER of two VAL-
#   reclaim patterns during today's RTH session, whichever bar qualifies
#   first:
#   1. Same-bar wick reclaim: a bar's LOW dips below VAL but its CLOSE is
#      back above VAL (pierced VAL intrabar, reclaimed by the close).
#   2. Next-bar reclaim: the PREVIOUS bar closed below VAL, and THIS bar
#      closes above VAL (broke below VAL, reclaimed it the bar after).
#   Entry fill is the CLOSE of the triggering (reclaim) bar itself - not
#   the VAL level. Only the first qualifying bar per day; one position at a
#   time (automatic here - every trade opens and fully resolves within the
#   same RTH session).
# Exit - whichever of the two hits first, checked only within the REST of
#   today's RTH session:
#   1. Stop Loss: price drops STOP_LOSS_PTS below entry price.
#   2. Take Profit: price rises TAKE_PROFIT_PTS above entry price.
#   Same-bar tie -> Stop Loss wins (worst-case fill assumption, same
#   convention used by every other engine in this file). If neither hits by
#   the last RTH bar of the day, forced exit at that day's 16:00 RTH close.
STOP_LOSS_PTS = 50       # input
TAKE_PROFIT_PTS = 90     # input
"""


def run_overnight_val_reversion_backtest(df: pd.DataFrame, *, cash: float = 100_000.0,
                                         size: float = 1.0,
                                         multiplier: float = 1.0,
                                         slippage_ticks: float = 0.0,
                                         commission_per_contract: float = 0.0,
                                         market_order: bool = True,
                                         tick_size: float = MNQ_TICK_SIZE,
                                         poc_position_max: float = 0.25,  # לא בשימוש - ראו דוקסטרינג
                                         stop_loss_pts: float = 50.0,
                                         take_profit_pts: float = 90.0,
                                         debug_log_touches: bool = False) -> dict:
    """long-only. בשונה משאר המנועים בקובץ - הסיגנל כאן הוא תכונה של היום
    כולו (פרופיל-נפח על חלון-הלילה), לא של בר בודד, ולכן הזיהוי עצמו לא
    ניתן לווקטוריזציה מלאה ברמת-בר: יש כאן לולאת-ימים (כ-1,900 איטרציות על
    כל ההיסטוריה - זול, לא לולאת-ברים על 2.5 מיליון ברים). חלק הבנייה היקר
    באמת (סכימת-נפח לפי (יום,בין) על פני כל הברים) כן וקטורי לגמרי -
    pandas groupby יחיד, לא לולאת-פייתון פר-בר - זו בדיוק הסיבה שהגרסה כאן
    שונה מסקריפט-המחקר שקדם לה (ששם השתמש בלולאת-פייתון פר-בר, מקובל
    למחקר תיאורי אבל לא מספיק מהיר ל"Full backtest" כאן).

    שלבים:
    1. RTH mask (09:30-16:00 ET) ו-day_label - בדיוק כמו בסקריפטי-המחקר.
       df.index כבר tz-naive בשעון ET (ראו ib_data.py) - בלי המרת אזור-זמן.
    2. לכל יום עם סשן RTH, אינדקס-הברים שלו ב-RTH (rth_idx_by_day) - בנוי
       פעם אחת דרך groupby, לא לולאה פר-בר.
    3. חלון-הלילה של כל יום = הברים שבין הבר האחרון ב-RTH של היום *הקודם*
       לבין הבר הראשון ב-RTH של היום הזה - טווח-אינדקסים רציף (לא טווח-
       תאריכים), ולכן "חוצה" סופי-שבוע/חגים אוטומטית בלי טיפול מיוחד.
       מתויג על מערך overnight_day_code (אורך n, ברירת מחדל -1 = "לא בחלון-
       לילה של אף יום") דרך השמת-פרוסה פר-יום (לולאת-ימים, לא לולאת-ברים).
    4. סכימת-נפח לפי (day_code, bin_idx) - pandas groupby יחיד על כל הברים
       שסומנו בשלב 3, וקטורי לגמרי (לא תלוי במספר הברים הכולל, רק בגודל
       הפלט - כ-1,900 ימים × כמה עשרות bins לכל היותר).
    5. POC + Value Area (70%) לכל יום - לולאת-פייתון על קבוצות ה-groupby
       (כ-1,900 איטרציות, לא יותר), אותו אלגוריתם-הרחבה-בצעד-בין-יחיד
       שאומת בסקריפטי-המחקר שקדמו למנוע הזה (העדפה לבין העליון בתיקו).
       ה-POC עצמו משמש רק כנקודת-ההתחלה של הרחבת ה-Value Area (כנדרש
       באלגוריתם) - לא נשמר/נבדק אחרי זה; רק VAL (תחתית ה-Value Area) נשמר
       לכל יום (val_price_by_code), כי זו הרמה היחידה שהכניסה תלויה בה כעת.
    6. לכל יום (ללא סינון לפי מיקום-POC): מחפשים את הבר הראשון ב-RTH של היום
       הזה שמקיים אחד משני תבניות-ה-reclaim סביב VAL (ראו דוקסטרינג המודול
       למעלה) - pattern1 (low[i]<VAL<close[i], reclaim באותו בר) *או*
       pattern2 (close[i-1]<VAL<close[i], reclaim בבר הבא) - לולאה קטנה (עד
       ~390 ברים) לכל יום. close[i-1] הוא הבר הקודם בסדרה הרציפה (כולל
       פרה-מרקט אם i הוא הבר הראשון ב-RTH של היום - אותה מוסכמה בדיוק כמו
       גרסת "מגיע מלמעלה" הקודמת של האסטרטגיה הזו). כניסה ב-Close של בר-
       ה-reclaim עצמו (לא ב-VAL_level) - מילוי-שוק, לא רמת-סף מדויקת.
    7. יציאה: סורקים קדימה רק על יתרת ברי-ה-RTH של אותו יום בלבד (לא חוצים
       ליום אחר) - Stop Loss/Take Profit, מי שקורה קודם; תיקו על אותו בר -
       Stop Loss מנצח. לא נמצא עד הבר האחרון - יציאה ב-Close של הבר הזה.

    עלויות: בשונה מ-run_hammer_breakout_backtest (שם slippage לא מוחל בכלל
    כי כל המחירים שם מוגדרים כרמות-סף מדויקות) - כאן slippage_ticks *כן*
    מוחל, על הכניסה והיציאה כאחד (גם Stop/Take Profit וגם ה-Close-fallback)
    - נדרש במפורש: הנחת-עלות השמרנית יותר, לא ברירת-המחדל של מנועי-hammer.

    poc_position_max: נשאר בחתימה לצורך תאימות-לאחור (קריאות קיימות
    שעדיין מעבירות אותו לא ייפלו), אבל לא נקרא/לא משפיע על שום דבר יותר -
    הסינון לפי מיקום-POC הוסר לגמרי מכניסת האסטרטגיה (בקשה מפורשת).
    """
    if df is None or len(df) < 2:
        raise StrategyCompileError("אין מספיק נרות להרצה")
    print(f"Backtest starting on {len(df)} bars from {df.index[0]} to {df.index[-1]}", flush=True)

    idx = df.index
    times = idx.to_numpy()
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n = len(close)

    BIN_WIDTH = 5.0
    VALUE_AREA_PCT = 0.70
    RTH_START_MIN = 9 * 60 + 30
    RTH_END_MIN = 16 * 60

    minutes_of_day = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    rth_mask = (minutes_of_day >= RTH_START_MIN) & (minutes_of_day < RTH_END_MIN)
    day_label = idx.normalize()

    rth_positions = np.flatnonzero(rth_mask)
    if len(rth_positions) == 0:
        raise StrategyCompileError("אין ברים בחלון ה-RTH (09:30-16:00 ET) בנתונים")
    rth_series = pd.Series(rth_positions, index=pd.Index(day_label[rth_mask], name="day"))
    rth_idx_by_day: dict = {d: g.to_numpy() for d, g in rth_series.groupby(level=0, observed=True)}
    sorted_days = sorted(rth_idx_by_day.keys())

    # שלב 3: תיוג חלון-הלילה של כל יום - לולאת-ימים עם השמת-פרוסה (זול,
    # לא לולאת-ברים).
    day_code = {d: i for i, d in enumerate(sorted_days)}
    overnight_day_code = np.full(n, -1, dtype=np.int64)
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prior_day = sorted_days[i - 1]
        start_pos = int(rth_idx_by_day[prior_day][-1])
        end_pos = int(rth_idx_by_day[day][0])
        if end_pos > start_pos + 1:
            overnight_day_code[start_pos + 1:end_pos] = day_code[day]

    # שלב 4: סכימת-נפח לפי (day_code, bin_idx) - groupby יחיד, וקטורי לגמרי.
    bin_idx = np.floor(close / BIN_WIDTH).astype(np.int64)
    valid_overnight = overnight_day_code >= 0
    vp = pd.DataFrame({
        "day_code": overnight_day_code[valid_overnight],
        "bin_idx": bin_idx[valid_overnight],
        "volume": volume[valid_overnight],
    })
    vol_by_bin = vp.groupby(["day_code", "bin_idx"], observed=True)["volume"].sum()

    # שלב 5: Value Area (70%) לכל day_code - לולאת-פייתון על קבוצות
    # ה-groupby בלבד (כ-1,900 איטרציות). ה-POC משמש רק כנקודת-ההתחלה של
    # הרחבת ה-Value Area (כנדרש באלגוריתם) - רק VAL (תחתית ה-Value Area)
    # נשמר לכל יום, כי זו הרמה היחידה שהכניסה תלויה בה כעת.
    val_price_by_code: dict = {}
    for dc, day_vols in vol_by_bin.groupby(level=0, observed=True):
        bin_vol = {b: v for (_, b), v in day_vols.items()}
        total_vol = sum(bin_vol.values())
        poc_bin = max(bin_vol, key=bin_vol.get)
        lo = hi = poc_bin
        cum = bin_vol[poc_bin]
        target = VALUE_AREA_PCT * total_vol
        while cum < target:
            next_lo = lo - 1 if (lo - 1) in bin_vol else None
            next_hi = hi + 1 if (hi + 1) in bin_vol else None
            if next_lo is None and next_hi is None:
                break
            elif next_lo is None:
                hi = next_hi
                cum += bin_vol[next_hi]
            elif next_hi is None:
                lo = next_lo
                cum += bin_vol[next_lo]
            elif bin_vol[next_hi] >= bin_vol[next_lo]:
                hi = next_hi
                cum += bin_vol[next_hi]
            else:
                lo = next_lo
                cum += bin_vol[next_lo]
        val_price_by_code[dc] = lo * BIN_WIDTH

    # שלבים 6-7: לכל יום - כניסה ב-VAL כש"מגיעים מלמעלה", יציאה ב-SL/TP/Close.
    stop_loss_pts_f = float(stop_loss_pts)
    take_profit_pts_f = float(take_profit_pts)

    entry_times, exit_times = [], []
    entry_prices, exit_prices = [], []
    entry_fill_idxs, exit_idxs = [], []

    for day in sorted_days:
        dc = day_code.get(day)
        if dc is None or dc not in val_price_by_code:
            continue
        val_level = val_price_by_code[dc]

        day_rth_idx = rth_idx_by_day[day]
        touch_pos = None
        for k, i in enumerate(day_rth_idx):
            # pattern1 - reclaim באותו בר: ה-Low יורד מתחת ל-VAL אבל ה-Close
            # חוזר מעליו. pattern2 - reclaim בבר הבא: הבר הקודם (בר-אחד-אחורה
            # בסדרה הרציפה, כולל פרה-מרקט אם i הוא הבר הראשון ב-RTH) נסגר
            # מתחת ל-VAL, והבר הזה נסגר מעליו. כל אחת משתי התבניות מספיקה.
            pattern1 = low[i] < val_level < close[i]
            pattern2 = close[i - 1] < val_level < close[i]
            if pattern1 or pattern2:
                touch_pos = k
                break
        if touch_pos is None:
            continue

        fill_i = int(day_rth_idx[touch_pos])
        entry_price = float(close[fill_i])  # מילוי ב-Close של בר-ה-reclaim עצמו, לא VAL
        sl_level = entry_price - stop_loss_pts_f
        tp_level = entry_price + take_profit_pts_f

        exit_i = None
        exit_price = None
        for i in day_rth_idx[touch_pos + 1:]:
            sl_hit = low[i] <= sl_level
            tp_hit = high[i] >= tp_level
            if sl_hit or tp_hit:
                exit_i = int(i)
                exit_price = sl_level if sl_hit else tp_level  # תיקו -> Stop Loss מנצח
                break
        if exit_i is None:
            exit_i = int(day_rth_idx[-1])
            exit_price = float(close[exit_i])

        if debug_log_touches:
            print(f"[overnight-val-debug] day={day} val_level={val_level} "
                  f"entry_time={times[fill_i]} exit_time={times[exit_i]}", flush=True)

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

    n_trades = len(entry_prices)
    raw_entry_prices_arr = np.asarray(entry_prices, dtype=float)
    raw_exit_prices_arr = np.asarray(exit_prices, dtype=float)
    entry_idx_arr = np.asarray(entry_fill_idxs, dtype=np.int64)
    exit_idx_arr = np.asarray(exit_idxs, dtype=np.int64)

    # סליפג' מוחל כאן על הכניסה והיציאה כאחד (כולל ה-Close-fallback) - נדרש
    # במפורש כהנחת-עלות שמרנית, בשונה מברירת-המחדל של מנועי-hammer. אותה
    # מוסכמה-כיוון בדיוק כמו run_down_streak_reversion_backtest.
    slippage_amount = (float(slippage_ticks) * float(tick_size)) if market_order else 0.0
    entry_prices_arr = raw_entry_prices_arr + slippage_amount
    exit_prices_arr = raw_exit_prices_arr - slippage_amount

    pnl = (exit_prices_arr - entry_prices_arr) * size * multiplier
    commission_total = float(commission_per_contract) * size * 2.0
    pnl = pnl - commission_total

    return_frac = np.divide(
        exit_prices_arr - entry_prices_arr, entry_prices_arr,
        out=np.zeros(n_trades), where=entry_prices_arr != 0,
    )

    # Equity מלא לכל בר (mark-to-market על עמדה פתוחה) - אותה טכניקה בדיוק
    # כמו שאר המנועים.
    position_delta = np.zeros(n)
    np.add.at(position_delta, entry_idx_arr, 1.0)
    np.add.at(position_delta, exit_idx_arr, -1.0)
    position_flag = np.cumsum(position_delta)

    entry_price_series = pd.Series(np.nan, index=np.arange(n))
    entry_price_series.iloc[entry_idx_arr] = entry_prices_arr
    entry_price_ffilled = entry_price_series.ffill().to_numpy()
    unrealized = np.where(position_flag > 0, (close - entry_price_ffilled) * size * multiplier, 0.0)

    realized_pnl_step = np.zeros(n)
    np.add.at(realized_pnl_step, exit_idx_arr, pnl)
    realized_pnl_cum = np.cumsum(realized_pnl_step)

    equity_curve = cash + realized_pnl_cum + unrealized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = np.where(running_max > 0, (equity_curve - running_max) / running_max * 100, 0.0)
    max_dd_pct = float(drawdown_pct.min()) if n else 0.0

    equity_final = cash + float(pnl.sum())
    return_total_pct = (equity_final - cash) / cash * 100 if cash else 0.0

    first_close = float(close[0]) if n else 0.0
    last_close = float(close[-1]) if n else 0.0
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

    trades_df = pd.DataFrame({
        "EntryTime": pd.to_datetime(entry_times),
        "ExitTime": pd.to_datetime(exit_times),
        "EntryPrice": entry_prices_arr,
        "ExitPrice": exit_prices_arr,
        "Size": np.full(n_trades, size),  # תמיד long
        "PnL": pnl,
        "ReturnPct": return_frac,
    })

    return {
        "Equity Final [$]": equity_final,
        "Return [%]": return_total_pct,
        "Max. Drawdown [%]": max_dd_pct,
        "Buy & Hold Return [%]": buy_hold_pct,
        "Sharpe Ratio": None,
        "Sortino Ratio": None,
        "_trades": trades_df,
    }


# ================================================ overnight_val_reversion - תצוגה בלבד
# פונקציה עצמאית נוספת (כפילות מכוונת, אותה מוסכמה כמו כל מנוע בקובץ הזה) -
# מחשבת את אותם POC/VAH/VAL של חלון-הלילה שה-strategy למעלה משתמש בהם, אבל
# בשביל אינדיקטור-תצוגה בלבד (ראו app.py:/api/overnight_levels ו-
# assets/indicators.js) - לא נקראת על ידי run_overnight_val_reversion_backtest
# ולא משפיעה עליו בשום צורה (ואין לה גישה/יכולת לשלוח הזמנות - מחזירה רק
# מספרים). קיימת בנפרד כדי לא לגעת במנוע-האסטרטגיה הקיים בכלל.
def compute_daily_overnight_value_areas(df: pd.DataFrame) -> pd.DataFrame:
    """מחזירה DataFrame אחד-לכל-יום (אינדקס: תאריך היום, tz-naive, שעון ET) עם
    העמודות poc/val/vah/poc_position (כמו בסטרטגיה) ו-rth_open_time/
    rth_close_time (חותמות-הזמן המדויקות של הבר הראשון/אחרון ב-RTH של אותו
    יום - נחוצות כדי לדעת היכן לצייר כל קו). ימים בלי סשן RTH, או בלי חלון-
    לילה קודם (היום הראשון בנתונים), או עם Value Area מנוונת (VAH==VAL) -
    פשוט לא מופיעים בתוצאה.

    האלגוריתם זהה לזה שבתוך run_overnight_val_reversion_backtest (bins של 5
    נקודות, נפח לפי Close, הרחבת Value Area בצעד-בין-יחיד עד 70% מהנפח,
    העדפת הבין העליון בתיקו) - קוד נפרד (לא קריאה לפונקציה המשותפת), באותה
    מוסכמה של כפילות-מכוונת כמו שאר הקובץ, כדי לא לגעת במנוע הקיים. אומת
    במפורש מול תוצאות המנוע עצמו (EntryPrice בפועל, עם slippage=0) בסקריפט
    בדיקה נפרד - לא רק "אותו קוד בערך".
    """
    if df is None or len(df) < 2:
        return pd.DataFrame(columns=["poc", "val", "vah", "poc_position",
                                     "rth_open_time", "rth_close_time"])

    BIN_WIDTH = 5.0
    VALUE_AREA_PCT = 0.70
    RTH_START_MIN = 9 * 60 + 30
    RTH_END_MIN = 16 * 60

    idx = df.index
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    times = idx.to_numpy()
    n = len(close)

    minutes_of_day = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    rth_mask = (minutes_of_day >= RTH_START_MIN) & (minutes_of_day < RTH_END_MIN)
    day_label = idx.normalize()

    rth_positions = np.flatnonzero(rth_mask)
    if len(rth_positions) == 0:
        return pd.DataFrame(columns=["poc", "val", "vah", "poc_position",
                                     "rth_open_time", "rth_close_time"])
    rth_series = pd.Series(rth_positions, index=pd.Index(day_label[rth_mask], name="day"))
    rth_idx_by_day: dict = {d: g.to_numpy() for d, g in rth_series.groupby(level=0, observed=True)}
    sorted_days = sorted(rth_idx_by_day.keys())

    day_code = {d: i for i, d in enumerate(sorted_days)}
    overnight_day_code = np.full(n, -1, dtype=np.int64)
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prior_day = sorted_days[i - 1]
        start_pos = int(rth_idx_by_day[prior_day][-1])
        end_pos = int(rth_idx_by_day[day][0])
        if end_pos > start_pos + 1:
            overnight_day_code[start_pos + 1:end_pos] = day_code[day]

    bin_idx = np.floor(close / BIN_WIDTH).astype(np.int64)
    valid_overnight = overnight_day_code >= 0
    vp = pd.DataFrame({
        "day_code": overnight_day_code[valid_overnight],
        "bin_idx": bin_idx[valid_overnight],
        "volume": volume[valid_overnight],
    })
    vol_by_bin = vp.groupby(["day_code", "bin_idx"], observed=True)["volume"].sum()

    rows = []
    for dc, day_vols in vol_by_bin.groupby(level=0, observed=True):
        bin_vol = {b: v for (_, b), v in day_vols.items()}
        total_vol = sum(bin_vol.values())
        poc_bin = max(bin_vol, key=bin_vol.get)
        poc_price = poc_bin * BIN_WIDTH + BIN_WIDTH / 2
        lo = hi = poc_bin
        cum = bin_vol[poc_bin]
        target = VALUE_AREA_PCT * total_vol
        while cum < target:
            next_lo = lo - 1 if (lo - 1) in bin_vol else None
            next_hi = hi + 1 if (hi + 1) in bin_vol else None
            if next_lo is None and next_hi is None:
                break
            elif next_lo is None:
                hi = next_hi
                cum += bin_vol[next_hi]
            elif next_hi is None:
                lo = next_lo
                cum += bin_vol[next_lo]
            elif bin_vol[next_hi] >= bin_vol[next_lo]:
                hi = next_hi
                cum += bin_vol[next_hi]
            else:
                lo = next_lo
                cum += bin_vol[next_lo]
        val_price = lo * BIN_WIDTH
        vah_price = hi * BIN_WIDTH + BIN_WIDTH
        if vah_price == val_price:
            continue  # Value Area מנוונת - לא כלולה (אותה מוסכמה כמו הסטרטגיה)
        day = sorted_days[dc]
        day_rth_idx = rth_idx_by_day[day]
        rows.append({
            "day": day, "poc": poc_price, "val": val_price, "vah": vah_price,
            "poc_position": (poc_price - val_price) / (vah_price - val_price),
            "rth_open_time": times[day_rth_idx[0]], "rth_close_time": times[day_rth_idx[-1]],
        })

    if not rows:
        return pd.DataFrame(columns=["poc", "val", "vah", "poc_position",
                                     "rth_open_time", "rth_close_time"])
    return pd.DataFrame(rows).set_index("day").sort_index()


# סגול אחיד לכניסה וליציאה - בדיוק כמו חצי הכניסה/יציאה של Strategy Tester ב-TradingView
# (שם צובעים את שני האירועים באותו צבע, ולא ירוק/אדום לפי סוג האירוע).
ENTRY_COLOR = "#C77DFF"
EXIT_COLOR = "#C77DFF"
LABEL_COLOR = "#FFFFFF"


def to_markers(trades: pd.DataFrame, epoch_fn) -> list[dict]:
    """נקודות הסימון על הגרף — אחת לכניסה ואחת ליציאה של כל עסקה.

    כל נקודה נושאת את **מחיר** הכניסה/היציאה בפועל, ולא רק את זמן הנר, כדי שהמשולש
    יצויר בדיוק על גובה המחיר שבו העסקה נפתחה או נסגרה. הציור עצמו נעשה ב-canvas
    ב-assets/chart.js: ל-Lightweight Charts יש רק סימונים שנצמדים לנר (aboveBar/
    belowBar/inBar) ואין דרך למקם סימון על מחיר מדויק.

    כניסה: משולש ירוק עם "long" (או "short") בלבן מתחתיו.
    יציאה: משולש אדום עם "exit" בלבן מתחתיו.
    חוד המשולש מצביע אל המחיר עצמו.
    """
    if trades is None or trades.empty:
        return []

    # ממירים את *כל* חותמות הזמן בבת אחת (וקטורי - אותה שיטה בדיוק כמו
    # to_chart_payload ב-data.py), לא בקריאה בודדת ל-epoch_fn לכל שורה בלולאה.
    # על "Full backtest" (כל ההיסטוריה - יכול לצאת 150K+ עסקאות) קריאה סקלרית
    # לכל שורה (epoch_fn בונה pd.DatetimeIndex חדש בכל קריאה) מדדנו שלוקחת
    # מעל 30 שניות; הגרסה הווקטורית הזו לוקחת עשיריות שנייה - זה בדיוק מה
    # שהפך מ"בקטסט מהיר אבל התצוגה עדיין תקועה" ל"מהיר מקצה לקצה".
    entry_epoch = to_display_naive(pd.DatetimeIndex(trades["EntryTime"])) \
        .astype("datetime64[s]").astype("int64").to_numpy()
    exit_epoch = to_display_naive(pd.DatetimeIndex(trades["ExitTime"])) \
        .astype("datetime64[s]").astype("int64").to_numpy()
    is_long_arr = (trades["Size"] > 0).to_numpy()
    entry_price_arr = trades["EntryPrice"].to_numpy()
    exit_price_arr = trades["ExitPrice"].to_numpy()

    points = []
    for i in range(len(trades)):
        is_long = bool(is_long_arr[i])
        points.append({
            "time": int(entry_epoch[i]), "price": float(entry_price_arr[i]),
            "color": ENTRY_COLOR, "dir": "up" if is_long else "down",
            "text": "long" if is_long else "short",
        })
        points.append({
            "time": int(exit_epoch[i]), "price": float(exit_price_arr[i]),
            "color": EXIT_COLOR, "dir": "down" if is_long else "up",
            "text": "exit",
        })

    points.sort(key=lambda m: m["time"])
    return points
