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
