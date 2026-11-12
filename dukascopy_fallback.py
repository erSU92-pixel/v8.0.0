"""
dukascopy_fallback.py — Free, keyless XAUUSD data via Dukascopy.
Fetches OHLCV candles (H1, M15) for spot gold.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import dukascopy_python as duka
    from dukascopy_python.instruments import I_XAU_USD
except ImportError:
    duka = None

LOG = logging.getLogger("xauusd")


def _log(level, msg, **kw):
    rec = LOG.makeRecord(LOG.name, level, "", 0, msg, (), None)
    rec.extra_data = kw
    LOG.handle(rec)


def fetch_dukascopy(timeframe: str, limit: int = 600) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Dukascopy.
    timeframe: "1h" or "15m"
    Returns DataFrame with columns: time, open, high, low, close, volume.
    """
    if duka is None:
        _log(logging.WARNING, "dukascopy_unavailable", reason="lib not installed")
        return None

    # Mapping timeframe -> interval & unit
    if timeframe == "1h":
        interval, unit = 1, duka.TimeUnit.HOUR
    elif timeframe == "15m":
        interval, unit = 15, duka.TimeUnit.MINUTE
    else:
        _log(logging.ERROR, "dukascopy_bad_tf", tf=timeframe)
        return None

    end = datetime.now(timezone.utc)
    # Fetch enough bars: limit * interval + buffer
    if unit == duka.TimeUnit.HOUR:
        start = end - timedelta(hours=limit + 50)
    else:
        start = end - timedelta(minutes=limit * 15 + 50 * 15)

    try:
        candles = duka.fetch_candles(
            instrument=I_XAU_USD,
            interval=interval,
            time_unit=unit,
            offer_side=duka.OfferSide.BID,
            start=start,
            end=end,
            version=2,
        )
        if not candles:
            return None

        rows = []
        for c in candles:
            rows.append({
                "time": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume if hasattr(c, "volume") and c.volume else 1000.0,
            })
        df = pd.DataFrame(rows)
        # Ensure UTC naive (same format as bot expects)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df = df.sort_values("time").reset_index(drop=True)
        _log(logging.INFO, "dukascopy_ok", tf=timeframe, bars=len(df))
        return df
    except Exception as e:
        _log(logging.WARNING, "dukascopy_fail", tf=timeframe, err=str(e)[:120])
        return None
