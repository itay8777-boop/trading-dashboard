"""עוטף קוד אסטרטגיה שכתב המשתמש (init/next בלבד) במחלקת Strategy של backtesting.py,
מוסיף VWAP אוטומטית, ומריץ את הבדיקה."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

DEFAULT_STRATEGY_CODE = """\
def init(self):
    # self.vwap מחושב אוטומטית וזמין תמיד
    # self.position_size מגיע מהגדרת "גודל פוזיציה" בסרגל הצד
    pass

def next(self):
    # דוגמה: כניסה לונג כשהמחיר חוצה מעל ה-VWAP, יציאה כשחוצה מתחתיו
    if not self.position and self.data.Close[-1] > self.vwap[-1]:
        self.buy(size=self.position_size)
    elif self.position and self.data.Close[-1] < self.vwap[-1]:
        self.position.close()
"""


class StrategyCompileError(Exception):
    """שגיאה בקומפילציה/הרצה של קוד האסטרטגיה שכתב המשתמש."""


def compute_vwap(df: pd.DataFrame) -> np.ndarray:
    high = pd.Series(df["High"]).reset_index(drop=True)
    low = pd.Series(df["Low"]).reset_index(drop=True)
    close = pd.Series(df["Close"]).reset_index(drop=True)
    volume = pd.Series(df["Volume"]).reset_index(drop=True)
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_vol_price = (typical * volume).cumsum()
    vwap = (cum_vol_price / cum_vol.replace(0, np.nan)).bfill().values
    return vwap


def _compile_user_strategy(code: str):
    namespace: dict = {"pd": pd, "np": np, "crossover": crossover}
    try:
        exec(code, namespace)
    except Exception as exc:
        raise StrategyCompileError(f"שגיאה בקוד האסטרטגיה: {exc}") from exc

    user_init = namespace.get("init")
    user_next = namespace.get("next")
    if not callable(user_init) or not callable(user_next):
        raise StrategyCompileError(
            "הקוד חייב להגדיר בדיוק שתי פונקציות: def init(self): ... ו-def next(self): ..."
        )
    return user_init, user_next


def build_strategy_class(code: str, position_size: int):
    user_init, user_next = _compile_user_strategy(code)

    class UserStrategy(Strategy):
        _user_position_size = position_size

        def init(self):
            vwap = compute_vwap(self.data.df)
            self.vwap = self.I(lambda: vwap, name="VWAP", overlay=True)
            self.position_size = self._user_position_size
            try:
                user_init(self)
            except Exception as exc:
                raise StrategyCompileError(f"שגיאה בפונקציית init: {exc}") from exc

        def next(self):
            try:
                user_next(self)
            except Exception as exc:
                raise StrategyCompileError(f"שגיאה בפונקציית next: {exc}") from exc

    return UserStrategy


def run_backtest(data: pd.DataFrame, code: str, cash: float, position_size: int, commission: float = 0.0):
    strategy_cls = build_strategy_class(code, position_size)
    bt = Backtest(data, strategy_cls, cash=cash, commission=commission, exclusive_orders=True)
    try:
        stats = bt.run()
    except StrategyCompileError:
        raise
    except Exception as exc:
        raise StrategyCompileError(f"שגיאה בהרצת הבדיקה: {exc}") from exc
    return stats
