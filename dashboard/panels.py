"""פאנל הביצועים שמתחת לגרף — בסגנון ה-Strategy Tester של TradingView.

ה-HTML נבנה על ידי ui_components.py שבשורש הפרויקט: אותם בוני טבלאות ואותן מחלקות
CSS שכבר משמשים את האפליקציה הקיימת, כדי שהמראה והמספרים יהיו זהים בשתי הסביבות.
מכיוון שהבקטסט כאן רץ על backtesting.py, אובייקט הסטטיסטיקות הוא בדיוק מה שהבונים
האלה מצפים לו — Sharpe, Sortino, Buy & Hold, Max Drawdown וכולי.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import to_display_naive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_components import (  # noqa: E402
    empty_state_html,
    key_stats_html,
    performance_summary_html,
    trades_table_html,
)


def key_stats(stats, cash: float) -> str:
    """שורת ה-Key stats: PnL, Max drawdown, אחוז עסקאות מרוויחות ו-Profit factor.

    ה-Profit factor ואחוז ההצלחה מחושבים מטבלת העסקאות עצמה ולא מ-backtesting.py,
    כדי שהכותרת והטבלה שמתחתיה יראו בדיוק את אותם מספרים.
    """
    trades: pd.DataFrame = stats["_trades"]
    n = len(trades)
    total_pnl = float(stats["Equity Final [$]"]) - cash
    dd_pct = abs(float(stats["Max. Drawdown [%]"]))

    if n:
        pnl = trades["PnL"]
        wins = int((pnl > 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(abs(pnl[pnl <= 0].sum()))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else None
        win_rate = wins / n * 100
    else:
        wins, win_rate, pf = 0, 0.0, None

    return key_stats_html(
        net_profit=total_pnl,
        net_profit_pct=float(stats["Return [%]"]),
        max_dd=dd_pct / 100 * cash,
        max_dd_pct=dd_pct,
        total_trades=n,
        profit_factor=pf,
        percent_profitable=win_rate,
        winning_trades=wins,
    )


def key_stats_for_period(trades: pd.DataFrame, cash: float, total_trades: int, period_label: str) -> str:
    """כמו key_stats, אבל מחושב רק מתת-הקבוצה שבחר המשתמש בפילטר התקופה מעל
    גרף ה-Performance (Overview) - לא מ-stats המקורי, שתמיד משקף את כל
    ההיסטוריה (הבקטסט עצמו לא רץ מחדש, ראו app.py: run_or_switch/PERIOD_LABELS).

    ה-Max Drawdown כאן מחושב מהעקומה המצטברת (cumsum) של *העסקאות הסגורות
    בלבד* בתת-הקבוצה - לא mark-to-market תוך-עסקה כמו stats["Max. Drawdown [%]"]
    המקורי (אין דרך זולה לשחזר את זה בלי להריץ מחדש בקטסט מלא רק על התקופה) -
    קירוב סביר, ועקבי עם איך שגרף ה-Performance עצמו מצייר את הקו המצטבר."""
    n = len(trades)
    if n == 0:
        return key_stats_html(
            net_profit=0.0, net_profit_pct=0.0, max_dd=0.0, max_dd_pct=0.0,
            total_trades=0, profit_factor=None, percent_profitable=0.0, winning_trades=0,
            period_note=f"{period_label} · 0 of {total_trades:,} trades",
        )
    pnl = trades["PnL"]
    total_pnl = float(pnl.sum())
    wins = int((pnl > 0).sum())
    win_rate = wins / n * 100
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl <= 0].sum()))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else None

    ordered_pnl = trades.sort_values("ExitTime")["PnL"].to_numpy()
    cum = np.cumsum(ordered_pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    max_dd_usd = float(drawdown.max()) if len(drawdown) else 0.0
    max_dd_pct = (max_dd_usd / cash * 100) if cash else 0.0
    net_profit_pct = (total_pnl / cash * 100) if cash else 0.0

    return key_stats_html(
        net_profit=total_pnl, net_profit_pct=net_profit_pct,
        max_dd=max_dd_usd, max_dd_pct=max_dd_pct,
        total_trades=n, profit_factor=pf, percent_profitable=win_rate, winning_trades=wins,
        period_note=f"{period_label} · {n:,} of {total_trades:,} trades",
    )


def performance_summary(stats, cash: float) -> str:
    trades: pd.DataFrame = stats["_trades"]
    if trades.empty:
        return empty_state_html("אין עסקאות לסיכום.")
    return performance_summary_html(stats, trades, cash)


def trades_list(stats, page: int = 0, page_size: int | None = None) -> str:
    """טבלת העסקאות. הזמנים מומרים לשעון התצוגה (ישראל) כדי שיתאימו לציר בגרף —
    טבלה בשעון בורסה לצד גרף בשעון ישראל הייתה מבלבלת. ההמרה היא לתצוגה בלבד;
    הבקטסט עצמו רץ על הזמנים המקוריים.

    page/page_size: ראו trades_table_html - מגבילים כמה שורות DOM נבנות בפועל
    (150K+ עסקאות בטבלה אחת הקפיאו את הדפדפן), בלי לפגוע במספור/במצטבר."""
    trades: pd.DataFrame = stats["_trades"]
    if trades.empty:
        return empty_state_html("לא בוצעו עסקאות.")

    shown = trades.copy()
    for col in ("EntryTime", "ExitTime"):
        shown[col] = to_display_naive(pd.DatetimeIndex(shown[col]))
    return trades_table_html(shown, page=page, page_size=page_size)


def empty(text: str) -> str:
    return empty_state_html(text)
