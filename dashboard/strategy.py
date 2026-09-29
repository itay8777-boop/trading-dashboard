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
    "STOP_LOSS_PTS": "stop_loss_pts",
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
                                    tick_size: float = MNQ_TICK_SIZE) -> dict:
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
# Hammer / Inverted-Hammer reversal, filtered by RSI extremes.
# Long:  RSI(previous bar) < RSI_OVERSOLD   AND current bar is a hammer.
# Short: RSI(previous bar) > RSI_OVERBOUGHT AND current bar is an inverted hammer.
# Exit:  stop-loss or take-profit, whichever the bar's high/low touches first.
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
                            take_profit_pts: float = 100.0) -> dict:
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

    long_signal = is_hammer & (prev_rsi < rsi_oversold)
    short_signal = is_inv_hammer & (prev_rsi > rsi_overbought)

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

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []

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
            hit_tp = search_high >= tp_level
            hit_sl = search_low <= sl_level
        else:
            hit_tp = search_low <= tp_level
            hit_sl = search_high >= sl_level
        hit_any = hit_tp | hit_sl

        if not hit_any.any():
            # לא נסגר עד סוף הדאטה - סוגרים על הסגירה האחרונה הידועה (כמו
            # backtesting.py: פוזיציה פתוחה בסוף הריצה נסגרת, לא נעלמת).
            exit_i = n - 1
            exit_price = float(close[exit_i])
        else:
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            # שני הרמות נחצו באותו בר - מניחים תרחיש פסימי (סטופ) כברירת מחדל
            # שמרנית, כי לא ידוע איזה מהם נגע קודם בפועל בתוך הבר.
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level
            exit_price = sl_level if sl_hit_here else tp_level

        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

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
                            allow_short: bool = True) -> dict:
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

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []
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
            hit_tp = high[fill_i + 1:] >= tp_level
            hit_sl = low[fill_i + 1:] <= sl_level
        else:
            sl_level = float(short_stop_lvl[sig_i])
            tp_level = float(short_target_lvl[sig_i])
            if not (tp_level < entry_price < sl_level):
                i_ptr += 1
                continue
            hit_tp = low[fill_i + 1:] <= tp_level
            hit_sl = high[fill_i + 1:] >= sl_level
        hit_any = hit_tp | hit_sl

        if not hit_any.any():
            exit_i = n - 1
            exit_price = float(close[exit_i])
        else:
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level
            exit_price = sl_level if sl_hit_here else tp_level   # שני הצדדים באותו בר: פסימי

        used_legs.add(leg_key)
        entry_times.append(times[fill_i])
        exit_times.append(times[exit_i])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i)
        exit_idxs.append(exit_i)

        i_ptr += 1
        while i_ptr < n_events and all_idx[i_ptr] <= exit_i:
            i_ptr += 1

    return _finalize_signed_trades(
        times=times, close=close, n=n, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        entry_times=entry_times, exit_times=exit_times,
        entry_prices=entry_prices, exit_prices=exit_prices, sizes=sizes,
        entry_fill_idxs=entry_fill_idxs, exit_idxs=exit_idxs,
    )


def _finalize_signed_trades(*, times, close, n, cash, multiplier, commission_per_contract,
                            entry_times, exit_times, entry_prices, exit_prices, sizes,
                            entry_fill_idxs, exit_idxs) -> dict:
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


def _simulate_signed_signals(*, times, open_, high, low, close, n,
                             sig_idx: np.ndarray, sig_is_long: np.ndarray,
                             sl_levels: np.ndarray, tp_levels: np.ndarray,
                             max_exit_idx: np.ndarray | None,
                             dedupe_key: np.ndarray | None,
                             size: float, slippage_amount: float,
                             cash: float, multiplier: float,
                             commission_per_contract: float) -> dict:
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

    entry_times, exit_times = [], []
    entry_prices, exit_prices, sizes = [], [], []
    entry_fill_idxs, exit_idxs = [], []
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
            hit_tp, hit_sl = seg_hi >= tp_level, seg_lo <= sl_level
        else:
            hit_tp, hit_sl = seg_lo <= tp_level, seg_hi >= sl_level
        hit_any = hit_tp | hit_sl
        if hit_any.size and hit_any.any():
            rel = int(np.argmax(hit_any))
            exit_i = fill_i + 1 + rel
            sl_hit_here = low[exit_i] <= sl_level if is_long else high[exit_i] >= sl_level
            exit_price = sl_level if sl_hit_here else tp_level
        else:
            exit_i = max(fill_i, last_i)     # יציאה כפויה (סוף סשן / סוף דאטה) בסגירה
            exit_price = float(close[exit_i])
        if key is not None:
            used.add(key)
        entry_times.append(times[fill_i]); exit_times.append(times[exit_i])
        entry_prices.append(entry_price); exit_prices.append(exit_price)
        sizes.append(size if is_long else -size)
        entry_fill_idxs.append(fill_i); exit_idxs.append(exit_i)
        i_ptr += 1
        while i_ptr < n_events and sig_idx[i_ptr] <= exit_i:
            i_ptr += 1

    return _finalize_signed_trades(
        times=times, close=close, n=n, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        entry_times=entry_times, exit_times=exit_times,
        entry_prices=entry_prices, exit_prices=exit_prices, sizes=sizes,
        entry_fill_idxs=entry_fill_idxs, exit_idxs=exit_idxs,
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
                     session_open: str = "09:30", exit_time: str = "15:55") -> dict:
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
    return _simulate_signed_signals(
        times=times, open_=open_, high=high, low=low, close=close, n=n,
        sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp,
        max_exit_idx=day_exit_idx[sig_idx], dedupe_key=day_codes[sig_idx],
        size=size, slippage_amount=slippage_amount, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
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
                             session_open: str = "09:30", exit_time: str = "15:55") -> dict:
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
                         allow_short: bool = True) -> dict:
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
    return _simulate_signed_signals(
        times=times, open_=open_, high=high, low=low, close=close, n=n,
        sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp,
        max_exit_idx=max_exit, dedupe_key=dedupe,
        size=size, slippage_amount=slippage_amount, cash=cash, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
    )


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
