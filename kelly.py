"""
kelly.py — Fractional Kelly position sizing.
==========================================
Formula: f* = (p*b - q) / b,  b = avg_win / avg_loss
Half-Kelly default. Hard cap 3% account. Fallback 1% jika data < 20 trade.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

MIN_TRADES_FOR_KELLY = 20
DEFAULT_RISK_PCT = 1.0


def _log(level: int, msg: str, **kw) -> None:
    lg = logging.getLogger("xauusd")
    rec = lg.makeRecord(lg.name, level, "", 0, msg, (), None)
    rec.extra_data = kw
    lg.handle(rec)


def kelly_stats(R: List[float]) -> Optional[Tuple[float, float, float, int]]:
    """Return (win_rate, avg_win, avg_loss, n) or None."""
    n = len(R)
    if n < MIN_TRADES_FOR_KELLY:
        return None
    wins = [x for x in R if x > 0]
    losses = [abs(x) for x in R if x < 0]
    if not wins or not losses:
        return None
    p = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    if avg_loss <= 0:
        return None
    return p, avg_win, avg_loss, n


def kelly_fraction(R: List[float], fraction: float = 0.5, cap_pct: float = 3.0) -> Tuple[float, str]:
    """Return (risk_pct, reason)."""
    st = kelly_stats(R)
    if st is None:
        n = len(R)
        return DEFAULT_RISK_PCT, f"fallback {DEFAULT_RISK_PCT}% (n={n}<{MIN_TRADES_FOR_KELLY})"

    p, avg_win, avg_loss, n = st
    b = avg_win / avg_loss
    q = 1.0 - p
    f_full = (p * b - q) / b
    f_frac = max(f_full * fraction, 0.0) * 100.0
    capped = min(f_frac, cap_pct)
    reason = (f"kelly p={p:.3f} b={b:.2f} full={f_full*100:.2f}% "
              f"×{fraction}={f_frac:.2f}% cap={cap_pct}% n={n}")
    _log(logging.INFO, "kelly_computed", p=p, b=b, f_full=f_full,
         f_frac=f_frac / 100.0, n=n)
    return round(capped, 2), reason


def risk_pct_from_env() -> Tuple[float, float]:
    """Return (fraction, cap)."""
    return (
        float(os.getenv("KELLY_FRACTION", "0.5")),
        float(os.getenv("KELLY_CAP_PCT", "3.0")),
    )
