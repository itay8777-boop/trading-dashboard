"""
בדיקות אמינות למנועי הבקטסט (הרצה: .venv/bin/python -m pytest tests -q).

1. אי-הצצה לעתיד: מריצים כל מנוע על הדאטה עד זמן T ועל הדאטה המלא; כל העסקאות
   שנכנסו לפני T (עם מרווח ביטחון) חייבות להיות זהות - אחרת המנוע "ראה" נרות עתידיים.
2. הצלבה מול מדמה עצמאי: מדמה איטי, נר-אחר-נר, שנכתב בנפרד ובצורה ישירה, מקבל
   את אותם אותות וסטופ/יעד ומחזיר עסקאות; הן חייבות להיות זהות לאלה של
   _simulate_signed_signals הווקטורי (מילוי בפתיחה הבאה, סטופ/יעד לפי high/low,
   פסימי באותו נר, פערים, סליפג', יציאה כפויה, עסקה אחת למפתח).
3. חשבונאות: PnL = (יציאה-כניסה)·size·multiplier פחות עמלות, ו-Equity Final תואם.
4. תקינות אקראית: הפעלת כל מנוע על חלונות אקראיים לא מעלה חריגות, עסקאות
   מסודרות ולא חופפות, ומחירי כניסה/יציאה בתוך טווח הנרות.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import lab  # noqa: E402
from dashboard import strategy as S  # noqa: E402

CACHE = ROOT / "data_cache" / "MNQ_5m_ext.parquet"
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="אין קאש MNQ")
COMMON = dict(cash=100_000.0, size=1.0, multiplier=2.0, slippage_ticks=2.0, commission_per_contract=0.75,
              market_order=True, fill_mode="realistic", exit_slippage_ticks=3.0)


@pytest.fixture(scope="module")
def df():
    d = pd.read_parquet(CACHE)
    return d.iloc[-25000:]      # ~4 חודשים של נרות 5 דקות


def _trades(key, data):
    return lab.run_engine(key, data, COMMON, lab.code_overrides(lab.ENGINES[key]["code"]))["_trades"]


# ------------------------------------------------------------------ 1. אי-הצצה
@pytest.mark.parametrize("key", list(lab.ENGINES))
def test_no_lookahead(df, key):
    cut = int(len(df) * 0.7)
    T = df.index[cut]
    full = _trades(key, df)
    part = _trades(key, df.iloc[:cut])
    margin = pd.Timedelta(days=3)      # עסקאות שנסגרו לפני T פחות מרווח, כדי לא להיתקל בסגירה כפויה בסוף הדאטה החלקי
    f = full[full["ExitTime"] < T - margin].reset_index(drop=True)
    p = part[part["ExitTime"] < T - margin].reset_index(drop=True)
    assert len(f) == len(p), f"{key}: {len(f)} עסקאות במלא מול {len(p)} בחלקי"
    for col in ("EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size"):
        if col in f.columns:
            if f[col].dtype.kind == "f":
                assert np.allclose(f[col].to_numpy(), p[col].to_numpy()), f"{key}: {col} שונה"
            else:
                assert (f[col].to_numpy() == p[col].to_numpy()).all(), f"{key}: {col} שונה"


# ------------------------------------------------------------------ 2. מדמה עצמאי
def reference_sim(df, sig_idx, is_long, sl, tp, max_exit, dedupe, size, slip_in, slip_out, cash, mult, comm, mode="realistic", tick=0.25):
    """מדמה נר-אחר-נר, כתוב בפשטות (לולאות בלבד). אותה מדיניות כמו המנוע."""
    o, h, l, c = df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    n = len(o)
    order = np.argsort(sig_idx, kind="stable")
    events = [(int(sig_idx[k]), bool(is_long[k]), float(sl[k]), float(tp[k]), None if max_exit is None else int(max_exit[k]),
               None if dedupe is None else dedupe[k]) for k in order]
    used = set(); trades = []; pos_until = -1
    for si, long_, sl_lvl, tp_lvl, mx, key in events:
        if si <= pos_until or si + 1 >= n:
            continue
        if key is not None and (key, long_) in used:
            continue
        fi = si + 1
        entry = o[fi] + slip_in if long_ else o[fi] - slip_in
        ok = (sl_lvl < entry < tp_lvl) if long_ else (tp_lvl < entry < sl_lvl)
        if not ok:
            continue
        last = n - 1 if mx is None else min(n - 1, mx)
        exit_i, exit_p = None, None
        pad = tick if mode == "worst" else 0.0
        xs = slip_out * (2 if mode == "worst" else 1)
        for i in range(fi + 1, last + 1):
            if long_:
                hit_sl = l[i] <= sl_lvl; hit_tp = h[i] >= tp_lvl + pad
            else:
                hit_sl = h[i] >= sl_lvl; hit_tp = l[i] <= tp_lvl - pad
            if mode == "optimistic":   # ברמה בדיוק, בלי פערים ובלי סליפג'
                if hit_sl: exit_i, exit_p = i, sl_lvl; break
                if hit_tp: exit_i, exit_p = i, tp_lvl; break
                continue
            if hit_sl:        # פסימי: סטופ קודם
                base = (min(sl_lvl, o[i]) if o[i] < sl_lvl else sl_lvl) if long_ else (max(sl_lvl, o[i]) if o[i] > sl_lvl else sl_lvl)
                exit_i, exit_p = i, (base - xs if long_ else base + xs); break
            if hit_tp:
                exit_i = i
                if long_: exit_p = max(tp_lvl, o[i]) if o[i] > tp_lvl else tp_lvl
                else: exit_p = min(tp_lvl, o[i]) if o[i] < tp_lvl else tp_lvl
                break
        if exit_i is None:
            exit_i = max(fi, last)
            if mode == "optimistic": exit_p = c[exit_i]
            else: exit_p = c[exit_i] - xs if long_ else c[exit_i] + xs
        pnl = (exit_p - entry) * (size if long_ else -size) * mult - comm * size * 2
        trades.append((fi, exit_i, round(entry, 6), round(exit_p, 6), round(pnl, 6)))
        pos_until = exit_i
        if key is not None:
            used.add((key, long_))
    return trades


def _engine_sim(df, sig_idx, is_long, sl, tp, max_exit, dedupe, mode="realistic"):
    tick = 0.25
    res = S._simulate_signed_signals(times=df.index.to_numpy(), open_=df["Open"].to_numpy(float), high=df["High"].to_numpy(float),
                                     low=df["Low"].to_numpy(float), close=df["Close"].to_numpy(float), n=len(df),
                                     sig_idx=sig_idx, sig_is_long=is_long, sl_levels=sl, tp_levels=tp, max_exit_idx=max_exit, dedupe_key=dedupe,
                                     size=1.0, slippage_amount=2 * tick, cash=100_000.0, multiplier=2.0, commission_per_contract=0.75,
                                     exit_slippage_amount=3 * tick, fill_mode=mode, tick_size=tick)
    tr = res["_trades"]
    idx = {t: i for i, t in enumerate(df.index)}
    return [(idx[e], idx[x], round(ep, 6), round(xp, 6), round(p, 6)) for e, x, ep, xp, p in
            zip(tr["EntryTime"], tr["ExitTime"], tr["EntryPrice"], tr["ExitPrice"], tr["PnL"])], res


@pytest.mark.parametrize("mode", ["realistic", "worst", "optimistic"])
def test_simulator_matches_reference(df, mode):
    rng = np.random.default_rng(3)
    n = len(df)
    sig = np.sort(rng.choice(np.arange(50, n - 50), size=600, replace=False))
    is_long = rng.random(len(sig)) < 0.5
    close = df["Close"].to_numpy(float)
    atr = (df["High"] - df["Low"]).rolling(14).mean().bfill().to_numpy()
    base = close[np.minimum(sig + 1, n - 1)]
    risk = np.maximum(atr[sig] * rng.uniform(1, 4, len(sig)), 2.0)
    sl = np.where(is_long, base - risk, base + risk)
    tp = np.where(is_long, base + risk * 1.5, base - risk * 1.5)
    max_exit = sig + rng.integers(20, 300, len(sig))
    dedupe = (sig // 200).astype(int)
    eng, res = _engine_sim(df, sig, is_long, sl, tp, max_exit, dedupe, mode)
    ref = reference_sim(df, sig, is_long, sl, tp, max_exit, dedupe, 1.0, 0.5, 0.75, 100_000.0, 2.0, 0.75, mode)
    assert len(eng) == len(ref) and len(eng) > 50
    for a, b in zip(eng, ref):
        assert a[0] == b[0] and a[1] == b[1], f"אינדקסים שונים: {a} vs {b}"
        assert abs(a[2] - b[2]) < 1e-6 and abs(a[3] - b[3]) < 1e-6, f"מחירים שונים: {a} vs {b}"
        assert abs(a[4] - b[4]) < 1e-4, f"PnL שונה: {a} vs {b}"
    # 3. חשבונאות
    tr = res["_trades"]
    assert abs(res["Equity Final [$]"] - (100_000.0 + tr["PnL"].sum())) < 1e-6


# ------------------------------------------------------------------ 4. תקינות
@pytest.mark.parametrize("key", list(lab.ENGINES))
def test_trades_are_sane(df, key):
    tr = _trades(key, df)
    if tr.empty:
        pytest.skip("אין עסקאות בחלון")
    assert (tr["ExitTime"] >= tr["EntryTime"]).all()
    o = tr.sort_values("EntryTime")
    assert (o["EntryTime"].to_numpy()[1:] >= o["ExitTime"].to_numpy()[:-1]).all(), f"{key}: עסקאות חופפות"
    lo, hi = df["Low"].min(), df["High"].max()
    pad = 20.0
    assert tr["EntryPrice"].between(lo - pad, hi + pad).all() and tr["ExitPrice"].between(lo - pad, hi + pad).all()
    assert np.isfinite(tr["PnL"]).all()
