"""
XAUUSD Quantitative Institutional Signal Bot — v9.0.0 PRODUCTION
================================================================
Changes v8.0.0 → v9.0.0 (all surgical, additive — no strategy change):
  [MACRO]    Tiered severity: OK / WARN / VETO / HALT
  [STALE]    Tighter H1 staleness gate (1.5h during market hours)
  [TP2]      TP2 distance capped at 4× ATR (feasibility for 72h hold)
  [PARAMS]   Range validation on params.json load
  [DEDUP]    Fingerprint includes regime
  [DAILY_R]  COALESCE(closed_at, ts) for robust daily R-sum
  [TELEGRAM] 3× exponential-backoff retry + status footer
  [LOG]      supabase-disabled risk, params-load-fail, gap-to-threshold
"""
from __future__ import annotations

import calendar
import html
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fuzzy_regime import blend_params, dominant, membership
from kelly import kelly_fraction, risk_pct_from_env
from news_calendar import NewsResult, check_news_guard as _live_news_guard
from supabase_client import SUPABASE, try_insert_signal

VERSION = "9.0.0"
RUN_ID = uuid.uuid4().hex[:8]
REGIMES = ("TRENDING", "WEAK_TREND", "RANGING")


# ============================================================================
# 1. CONFIG
# ============================================================================
@dataclass
class BotConfig:
    min_bars: int = 500
    retry_total: int = 3
    retry_backoff: float = 0.5
    http_timeout: int = 12
    telegram_timeout: int = 8
    telegram_retries: int = 3
    account_size_usd: float = 10_000.0
    risk_pct: float = 1.0
    signal_cooldown_min: int = 30
    spot_cache_min: int = 2
    macro_cache_min: int = 30
    news_enabled: bool = True
    notify_wait: bool = False
    # Stale data guard (v9: tightened)
    max_stale_h_market: float = 1.5
    max_stale_h_off: float = 72.0
    # MACRO severity thresholds (v9: tiered)
    dxy_warn: float = 0.3
    dxy_veto: float = 0.6
    dxy_halt: float = 1.2
    tnx_warn: float = 0.5
    tnx_veto: float = 1.5
    tnx_halt: float = 3.0
    # TP2 feasibility cap (v9)
    max_tp2_atr_mult: float = 4.0
    # Daily DD / circuit breaker
    daily_dd_limit_r: float = 3.0
    daily_dd_derisk_r: float = 2.0
    daily_dd_derisk_mult: float = 0.5
    cb_halt_consecutive: int = 5
    # Credentials
    twelve_data_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    csv_path: str = "data/xauusd_h1.csv"
    db_path: str = "data/quant_audit.db"

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            account_size_usd=float(os.getenv("ACCOUNT_SIZE_USD", "10000")),
            risk_pct=float(os.getenv("RISK_PCT", "1.0")),
            news_enabled=os.getenv("NEWS_ENABLED", "1") == "1",
            notify_wait=os.getenv("NOTIFY_WAIT", "0") == "1",
            daily_dd_limit_r=float(os.getenv("DAILY_DD_LIMIT_R", "3.0")),
            daily_dd_derisk_r=float(os.getenv("DAILY_DD_DERISK_R", "2.0")),
            daily_dd_derisk_mult=float(os.getenv("DAILY_DD_DERISK_MULT", "0.5")),
            cb_halt_consecutive=int(os.getenv("CB_HALT_CONSECUTIVE", "5")),
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            csv_path=os.getenv("OHLCV_CSV_PATH", "data/xauusd_h1.csv"),
            db_path=os.getenv("AUDIT_DB_PATH", "data/quant_audit.db"),
        )

    def warnings(self) -> List[str]:
        w = []
        if not self.twelve_data_api_key:
            w.append("TWELVE_DATA_API_KEY missing → yfinance fallback (GC=F, not spot)")
        if not self.telegram_bot_token or not self.telegram_chat_id:
            w.append("Telegram creds missing → stdout only")
        if not SUPABASE.enabled:
            w.append("Supabase disabled → SQLite-only, Kelly may never activate")
        return w


# ============================================================================
# 2. LOGGING
# ============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        p = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
             "level": record.levelname, "msg": record.getMessage()}
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            p.update(extra)
        if record.exc_info:
            p["exc"] = self.formatException(record.exc_info)
        return json.dumps(p, default=str)


def get_logger(name: str = "xauusd") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        lg.addHandler(h)
        lg.propagate = False
    return lg


LOG = get_logger()


def log_event(level: int = logging.INFO, event: str = "", **kw: Any) -> None:
    rec = LOG.makeRecord(LOG.name, level, "", 0, event, (), None)
    rec.extra_data = kw
    LOG.handle(rec)


# ============================================================================
# 3. INDICATORS
# ============================================================================
def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p, min_periods=p).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    loss = -d.clip(upper=0).ewm(alpha=1 / p, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))


def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, adjust=False).mean()


def adx(df: pd.DataFrame, p: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, p)
    p_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (tr + 1e-10)
    m_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (tr + 1e-10)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    return dx.ewm(alpha=1 / p, adjust=False).mean(), p_di, m_di


def volume_profile_poc(df: pd.DataFrame, bins: int = 25) -> float:
    if df["volume"].sum() <= 0:
        return float(df["close"].median())
    hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


# ============================================================================
# 4. HOLIDAYS
# ============================================================================
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    weeks = calendar.monthcalendar(year, month)
    days = [w[weekday] for w in weeks if w[weekday] != 0]
    return date(year, month, days[n - 1] if n > 0 else days[n])


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _holiday_set(year: int) -> Set[date]:
    h: Set[date] = {
        date(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _nth_weekday(year, 5, 0, -1),
        date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        date(year, 12, 25),
    }
    if year >= 2022:
        h.add(date(year, 6, 19))
    return h


_HOLIDAYS: Dict[int, Set[date]] = {y: _holiday_set(y) for y in range(2024, 2031)}


def is_holiday(d: date) -> bool:
    return d in _HOLIDAYS.get(d.year, set())


# ============================================================================
# 5. MARKET HOURS
# ============================================================================
def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    if is_holiday(now.date()):
        return False
    wd, h = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 6:
        return h >= 22
    if wd == 4:
        return h < 21
    return True


# ============================================================================
# 6. DATA QUALITY  (v9: tighter staleness)
# ============================================================================
def validate_ohlcv(df: pd.DataFrame, min_bars: int, label: str,
                   allow_gap: bool = False,
                   max_stale_market_h: float = 1.5,
                   max_stale_off_h: float = 72.0) -> None:
    req = {"time", "open", "high", "low", "close", "volume"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{label}: missing {miss}")
    if len(df) < min_bars:
        raise ValueError(f"{label}: {len(df)}<{min_bars}")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{label}: NaN OHLC")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{label}: high<low")
    if df["volume"].sum() <= 0:
        raise ValueError(f"{label}: zero volume")
    if not allow_gap and len(df) > 1:
        gaps = df["time"].diff()
        for i in range(1, len(df)):
            g = gaps.iloc[i]
            if g <= pd.Timedelta("4h"):
                continue
            prev_t = df["time"].iloc[i - 1]
            curr_t = df["time"].iloc[i]
            if g > pd.Timedelta("24h"):
                continue
            if prev_t.weekday() == 4 and prev_t.hour >= 20:
                continue
            if curr_t.weekday() in (6, 0):
                continue
            raise ValueError(f"{label}: internal gap {g} at {curr_t}")
    last_ts = pd.to_datetime(df["time"].iloc[-1])
    now = datetime.now(timezone.utc)
    age_h = (now.replace(tzinfo=None) - last_ts).total_seconds() / 3600
    # v9: tighter than v8's 6h
    max_age = max_stale_market_h if is_market_open(now) else max_stale_off_h
    if age_h > max_age:
        raise ValueError(f"{label}: stale {age_h:.2f}h (max {max_age}h)")


# ============================================================================
# 7. PARAMS  (v9: range validation)
# ============================================================================
DEFAULT_PER_REGIME_PARAMS: Dict[str, Dict[str, float]] = {
    "TRENDING":   {"threshold": 65.0, "sl_mult": 1.5, "tp1_mult": 2.0, "tp2_mult": 3.0, "max_hold": 72.0},
    "WEAK_TREND": {"threshold": 70.0, "sl_mult": 1.5, "tp1_mult": 1.5, "tp2_mult": 2.5, "max_hold": 48.0},
    "RANGING":    {"threshold": 78.0, "sl_mult": 1.0, "tp1_mult": 1.0, "tp2_mult": 2.0, "max_hold": 24.0},
}

_PARAM_BOUNDS = {
    "threshold": (50.0, 95.0),
    "sl_mult":   (0.5, 3.0),
    "tp1_mult":  (0.5, 5.0),
    "tp2_mult":  (1.0, 8.0),
    "max_hold":  (12.0, 168.0),
}


def _validate_params(params: Dict[str, Dict[str, float]]) -> Tuple[bool, str]:
    """v9: reject params outside reasonable bounds."""
    for regime in REGIMES:
        p = params.get(regime, {})
        for k, (lo, hi) in _PARAM_BOUNDS.items():
            v = p.get(k)
            if v is None or not isinstance(v, (int, float)):
                return False, f"{regime}.{k} missing or non-numeric"
            if not (lo <= v <= hi):
                return False, f"{regime}.{k}={v} outside [{lo}, {hi}]"
    return True, "ok"


def load_params() -> Tuple[Dict[str, Dict[str, float]], str]:
    path = os.getenv("PARAMS_FILE", "params.json")
    if not os.path.exists(path):
        return {r: dict(DEFAULT_PER_REGIME_PARAMS[r]) for r in REGIMES}, "builtin"
    try:
        with open(path) as f:
            j = json.load(f)
        schema = int(j.get("schema", 1))
        if schema >= 2:
            per = j.get("per_regime", {})
            merged = {r: {**DEFAULT_PER_REGIME_PARAMS[r], **per.get(r, {})} for r in REGIMES}
        else:
            flat = j.get("best_params", {})
            merged = {r: {**DEFAULT_PER_REGIME_PARAMS[r], **flat} for r in REGIMES}
        ok, msg = _validate_params(merged)
        if not ok:
            log_event(logging.WARNING, "params_invalid_fallback", reason=msg)
            return {r: dict(DEFAULT_PER_REGIME_PARAMS[r]) for r in REGIMES}, "builtin"
        return merged, "file"
    except Exception as e:
        log_event(logging.WARNING, "params_load_fail", err=str(e)[:120])
        return {r: dict(DEFAULT_PER_REGIME_PARAMS[r]) for r in REGIMES}, "builtin"


# ============================================================================
# 8. MARKET DATA HANDLER
# ============================================================================
def _retry_session(retries: int, backoff: float) -> requests.Session:
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff,
              status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": f"XAUUSD-Quant/{VERSION}"})
    return s


@dataclass
class MarketData:
    h1: pd.DataFrame
    m15: pd.DataFrame
    source: str
    m15_synthetic: bool = False


class MarketDataHandler:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.session = _retry_session(cfg.retry_total, cfg.retry_backoff)
        self._spot_cache: Optional[Tuple[datetime, float]] = None

    def _twelve(self, interval: str, size: int) -> Optional[pd.DataFrame]:
        if not self.cfg.twelve_data_api_key:
            return None
        url = (f"https://api.twelvedata.com/time_series?symbol=XAU/USD"
               f"&interval={interval}&outputsize={size}"
               f"&apikey={self.cfg.twelve_data_api_key}")
        r = self.session.get(url, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error" or j.get("code", 200) >= 400:
            raise RuntimeError(j.get("message", "TwelveData error"))
        vals = j.get("values") or []
        if not vals:
            return None
        df = pd.DataFrame(vals).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
        else:
            df["volume"] = 1000.0
        return df.sort_values("time").reset_index(drop=True)

    def _yf(self) -> MarketData:
        if yf is None:
            raise RuntimeError("yfinance missing")
        tk = yf.Ticker("GC=F")
        h1_raw = tk.history(period="60d", interval="1h")
        m15_raw = tk.history(period="60d", interval="15m")
        if h1_raw.empty:
            raise RuntimeError("Yahoo H1 empty")

        def norm(df: pd.DataFrame) -> pd.DataFrame:
            df = df.reset_index()
            tcol = "Datetime" if "Datetime" in df.columns else (
                "Date" if "Date" in df.columns else df.columns[0])
            df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"})
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
            else:
                df["volume"] = 1000.0
            return df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)

        h1 = norm(h1_raw)
        if not m15_raw.empty:
            m15 = norm(m15_raw)
            if len(m15) >= 50:
                return MarketData(h1, m15, "yfinance", m15_synthetic=False)
        m15 = h1.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(h1, m15, "yfinance+synthetic", m15_synthetic=True)

    def _csv(self) -> MarketData:
        p = self.cfg.csv_path
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            dummy = pd.date_range(
                end=datetime.now(timezone.utc).replace(tzinfo=None),
                periods=600, freq="h")
            pd.DataFrame({
                "time": dummy, "open": 2000.0, "high": 2001.0,
                "low": 1999.0, "close": 2000.0, "volume": 1000.0,
            }).to_csv(p, index=False)
            log_event(logging.WARNING, "csv_seeded_dummy", path=p)
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = "datetime" if "datetime" in df.columns else (
            "time" if "time" in df.columns else df.columns[0])
        df["time"] = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
        else:
            df["volume"] = 1000.0
        df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
        m15 = df.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(df, m15, "csv", m15_synthetic=True)

    def fetch(self) -> MarketData:
        try:
            h1 = self._twelve("1h", 500)
            m15 = self._twelve("15min", 200)
            if h1 is not None and len(h1) >= self.cfg.min_bars:
                validate_ohlcv(h1, self.cfg.min_bars, "12data-H1",
                               max_stale_market_h=self.cfg.max_stale_h_market,
                               max_stale_off_h=self.cfg.max_stale_h_off)
                if m15 is None or len(m15) < 50:
                    m15 = h1.set_index("time").resample("15min").ffill().reset_index()
                    m15["volume"] /= 4.0
                    synth = True
                else:
                    validate_ohlcv(m15, 50, "12data-M15")
                    synth = False
                log_event(event="data_ok", source="twelvedata", m15_synthetic=synth)
                return MarketData(h1, m15, "twelvedata", m15_synthetic=synth)
        except Exception as e:
            log_event(logging.WARNING, "twelvedata_fail", err=str(e)[:120])

        try:
            md = self._yf()
            validate_ohlcv(md.h1, self.cfg.min_bars, "yf-H1",
                           max_stale_market_h=self.cfg.max_stale_h_market)
            validate_ohlcv(md.m15, 50, "yf-M15")
            log_event(event="data_ok", source=md.source, m15_synthetic=md.m15_synthetic)
            return md
        except Exception as e:
            log_event(logging.WARNING, "yfinance_fail", err=str(e)[:120])

        md = self._csv()
        validate_ohlcv(md.h1, self.cfg.min_bars, "csv-H1", allow_gap=True,
                       max_stale_off_h=168.0)
        return md

    def fetch_spot(self, h1: pd.DataFrame) -> float:
        if self._spot_cache:
            ts, v = self._spot_cache
            if (datetime.now(timezone.utc) - ts).total_seconds() < self.cfg.spot_cache_min * 60:
                return v
        price = None
        if self.cfg.twelve_data_api_key:
            try:
                r = self.session.get(
                    f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.cfg.twelve_data_api_key}",
                    timeout=6)
                r.raise_for_status()
                p = float(r.json().get("price", 0) or 0)
                if p > 0:
                    price = p
            except Exception:
                pass
        if price is None and yf is not None:
            try:
                p = float(yf.Ticker("GC=F").fast_info["last_price"])
                if p > 0:
                    price = p
            except Exception:
                pass
        if price is None:
            price = float(h1["close"].iloc[-1])
        self._spot_cache = (datetime.now(timezone.utc), price)
        return price

    def get_basis(self, md: MarketData) -> float:
        fut = float(md.h1["close"].iloc[-1])
        spot = self.fetch_spot(md.h1)
        basis = spot - fut
        if abs(basis) > 100:
            log_event(logging.WARNING, "basis_anomaly", basis=round(basis, 2))
            return 0.0
        return basis


# ============================================================================
# 9. MACRO  (v9: tiered severity)
# ============================================================================
@dataclass
class MacroCheck:
    severity: str  # "OK", "WARN", "VETO", "HALT"
    reason: str
    dxy_chg: float = 0.0
    tnx_chg: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.severity in ("OK", "WARN")

    @property
    def blocks(self) -> bool:
        return self.severity in ("VETO", "HALT")


_macro_cache: Dict[str, Tuple[datetime, MacroCheck]] = {}


def check_macro_correlation(cfg: BotConfig) -> MacroCheck:
    now = datetime.now(timezone.utc)
    if "m" in _macro_cache:
        ts, val = _macro_cache["m"]
        if (now - ts).total_seconds() < cfg.macro_cache_min * 60:
            return val
    if yf is None:
        return MacroCheck("OK", "yfinance unavailable → bypass")

    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"]
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
        dxy_chg = 0.0
        tnx_chg = 0.0
        severity = "OK"
        notes = []

        if len(dxy) >= 2:
            dxy_chg = float((dxy.iloc[-1] - dxy.iloc[-2]) / dxy.iloc[-2] * 100)
            if abs(dxy_chg) >= cfg.dxy_halt:
                severity = "HALT"
                notes.append(f"DXY{'+' if dxy_chg > 0 else ''}{dxy_chg:.2f}%")
            elif abs(dxy_chg) >= cfg.dxy_veto:
                severity = max(severity, "VETO", key=lambda x: ["OK", "WARN", "VETO", "HALT"].index(x))
                notes.append(f"DXY{'+' if dxy_chg > 0 else ''}{dxy_chg:.2f}%")
            elif abs(dxy_chg) >= cfg.dxy_warn:
                severity = "WARN"
                notes.append(f"DXY±{dxy_chg:.2f}%")
        if len(tnx) >= 2:
            tnx_chg = float((tnx.iloc[-1] - tnx.iloc[-2]) / tnx.iloc[-2] * 100)
            if abs(tnx_chg) >= cfg.tnx_halt:
                severity = "HALT"
                notes.append(f"10Y{'+' if tnx_chg > 0 else ''}{tnx_chg:.2f}%")
            elif abs(tnx_chg) >= cfg.tnx_veto:
                if severity != "HALT":
                    severity = "VETO"
                notes.append(f"10Y{'+' if tnx_chg > 0 else ''}{tnx_chg:.2f}%")
            elif abs(tnx_chg) >= cfg.tnx_warn:
                if severity == "OK":
                    severity = "WARN"
                notes.append(f"10Y±{tnx_chg:.2f}%")

        reason = "normal" if severity == "OK" else " ".join(notes)
        result = MacroCheck(severity, reason, dxy_chg, tnx_chg)
        _macro_cache["m"] = (now, result)
        return result
    except Exception as e:
        result = MacroCheck("OK", f"bypass ({str(e)[:30]})")
        _macro_cache["m"] = (now, result)
        return result


# ============================================================================
# 10. SESSION PROFILE
# ============================================================================
def get_session_profile(cfg: BotConfig) -> Tuple[str, bool, float, float]:
    now = datetime.now(timezone.utc)
    if not is_market_open(now):
        return "CLOSED", False, 999.0, 1.0
    h = now.hour
    if 0 <= h < 7:   return "ASIA_RANGE",   True, 65.0, 1.0
    if 7 <= h < 12:  return "LONDON_TREND", True, 70.0, 1.3
    if 12 <= h < 16: return "NY_OVERLAP",   True, 75.0, 1.5
    if 16 <= h < 21: return "NY_PM",        True, 70.0, 1.3
    return "OFF_HOURS", False, 999.0, 1.0


# ============================================================================
# 11. SIGNAL GENERATOR
# ============================================================================
@dataclass
class EnsembleResult:
    action: str
    total_score: float
    voter_agreement: float
    forensic_score: float
    regime: str
    regime_weights: Dict[str, float]
    poc_level: float
    notes: List[str]
    atr_value: float
    atr_multiplier: float
    blended_params: Dict[str, float]
    threshold: float
    gap_to_threshold: float = 0.0
    params_source: str = "builtin"
    voters_debug: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    def __init__(self, params: Dict[str, Dict[str, float]], params_source: str):
        self.params = params
        self.params_source = params_source

    def _run_21_voters(self, df: pd.DataFrame) -> Tuple[float, int, int, Dict[str, int]]:
        close = df["close"]
        c = float(close.iloc[-1])
        v: List[int] = []
        dbg: Dict[str, int] = {}

        sma_votes = [1 if c > sma(close, p).iloc[-1] else -1 for p in (5, 10, 20, 50, 100, 200)]
        v.extend(sma_votes)
        dbg["sma"] = sum(sma_votes)

        e1 = 1 if ema(close, 9).iloc[-1] > ema(close, 21).iloc[-1] else -1
        e2 = 1 if ema(close, 12).iloc[-1] > ema(close, 26).iloc[-1] else -1
        e3 = 1 if ema(close, 50).iloc[-1] > ema(close, 200).iloc[-1] else -1
        v.extend([e1, e2, e3])
        dbg["ema"] = e1 + e2 + e3

        r = float(rsi(close, 14).iloc[-1])
        r_v = [1 if r > 50 else -1, 1 if r < 70 else -1, 1 if r > 30 else -1]
        v.extend(r_v)
        dbg["rsi"] = sum(r_v)

        adx_v, p_di, m_di = adx(df, 14)
        m_v = [
            1 if p_di.iloc[-1] > m_di.iloc[-1] else -1,
            1 if adx_v.iloc[-1] > 20 else -1,
            1 if c > close.iloc[-2] else -1,
            1 if c > close.iloc[-5] else -1,
            1 if c > close.iloc[-10] else -1,
        ]
        v.extend(m_v)
        dbg["momentum"] = sum(m_v)

        mp = 0
        for n in (5, 10):
            hi_n, lo_n = df["high"].iloc[-n:].max(), df["low"].iloc[-n:].min()
            pos = (c - lo_n) / max(hi_n - lo_n, 1e-10)
            vote = 1 if pos > 0.5 else -1
            v.append(vote)
            mp += vote
        dbg["micro"] = mp

        vol_v = 1 if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] else -1
        v.append(vol_v)
        dbg["volume"] = vol_v

        poc = volume_profile_poc(df)
        poc_v = 1 if c > poc else -1
        v.append(poc_v)
        dbg["poc"] = poc_v

        assert len(v) == 21
        bull = sum(1 for x in v if x == 1)
        bear = 21 - bull
        return max(bull, bear) / 21.0, bull, bear, dbg

    def generate(self, md: MarketData, news: NewsResult, macro: MacroCheck,
                 cfg: BotConfig) -> EnsembleResult:
        df, m15 = md.h1, md.m15
        close = df["close"]
        c_last = float(close.iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1])
        adx_val = float(adx(df, 14)[0].iloc[-1])

        weights = membership(adx_val)
        dom = dominant(weights)
        blended = blend_params(weights, self.params)
        threshold = blended.get("threshold", 70.0)
        atr_mult = blended.get("sl_mult", 1.5)

        poc = volume_profile_poc(df)
        session, allowed, _, _ = get_session_profile(cfg)
        ema200 = float(ema(close, 200).iloc[-1])

        last, prev = df.iloc[-1], df.iloc[-21:-1]
        op, hi, lo, cl = last["open"], last["high"], last["low"], last["close"]
        rng = max(hi - lo, 0.01)
        wick_pct = (max(hi - max(op, cl), min(op, cl) - lo) / rng) * 100.0
        sweep = "NONE"
        if hi > prev["high"].max() and cl < prev["high"].max():
            sweep = "BEAR"
        elif lo < prev["low"].min() and cl > prev["low"].min():
            sweep = "BULL"
        h1_vr = last["volume"] / (df["volume"].iloc[-21:-1].mean() + 1e-10)
        m15_vr = (m15["volume"].iloc[-1] / (m15["volume"].iloc[-20:].mean() + 1e-10)
                  if not md.m15_synthetic else 1.0)
        f_score = 30.0
        if sweep != "NONE":
            f_score += 35.0
        if wick_pct > 40.0:
            f_score += 20.0
        if h1_vr > 1.2 or m15_vr > 1.3:
            f_score += 15.0
        f_score = min(f_score, 100.0)

        m15_bull = ema(m15["close"], 9).iloc[-1] > ema(m15["close"], 21).iloc[-1]
        voter_agree, bull, bear, dbg = self._run_21_voters(df)
        combined = round((voter_agree * 0.6 + (f_score / 100.0) * 0.4) * 100, 1)
        intended = "BUY" if bull > bear else "SELL"
        macro_ok = ((intended == "BUY" and c_last > ema200)
                    or (intended == "SELL" and c_last < ema200))
        gap = round(combined - threshold, 1)

        notes, action = [], "WAIT"
        if not news.allowed:
            notes.append(f"NEWS: {news.reason}")
        elif macro.blocks:
            notes.append(f"MACRO[{macro.severity}]: {macro.reason}")
        elif not allowed:
            notes.append(f"SESSION: {session}")
        elif combined < threshold:
            notes.append(f"GATED[{dom}]: {combined}% under {threshold:.1f}% (gap {gap:+.1f})")
        elif not macro_ok:
            notes.append("VETO EMA200 H1")
        elif intended == "BUY" and m15_bull:
            action = "BUY"
            notes.append(f"OK[{dom}/{session}] {combined}%")
        elif intended == "SELL" and not m15_bull:
            action = "SELL"
            notes.append(f"OK[{dom}/{session}] {combined}%")
        else:
            notes.append("M15 misaligned")

        if macro.severity == "WARN":
            notes.append(f"macro_warn: {macro.reason}")

        dbg["news_source"] = news.source
        dbg["macro_severity"] = macro.severity
        return EnsembleResult(
            action=action, total_score=combined,
            voter_agreement=round(voter_agree, 4), forensic_score=round(f_score, 1),
            regime=dom, regime_weights=weights, poc_level=round(poc, 2),
            notes=notes, atr_value=round(atr_val, 4), atr_multiplier=round(atr_mult, 4),
            blended_params={k: round(v, 4) for k, v in blended.items()},
            threshold=round(threshold, 1), gap_to_threshold=gap,
            params_source=self.params_source, voters_debug=dbg,
        )


# ============================================================================
# 12. AUDIT DB
# ============================================================================
def _to_utc_iso(dt_or_str: Any) -> str:
    if isinstance(dt_or_str, str):
        d = datetime.fromisoformat(dt_or_str)
    elif isinstance(dt_or_str, datetime):
        d = dt_or_str
    elif isinstance(dt_or_str, pd.Timestamp):
        d = dt_or_str.to_pydatetime()
    else:
        d = datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat()


class AuditDB:
    def __init__(self, path: str):
        self.ok = False
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            with self.conn:
                self.conn.execute("""CREATE TABLE IF NOT EXISTS signals(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, action TEXT, spot REAL, score REAL,
                    regime TEXT, kelly_pct REAL, lot REAL, risk REAL, notes TEXT)""")
                self._migrate()
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts)")
            self.conn.commit()
            self.ok = True
        except Exception as e:
            log_event(logging.WARNING, "db_init_fail", err=str(e)[:80])

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(signals)").fetchall()}
        new_cols = [
            ("signal_id", "TEXT"),
            ("entry_price", "REAL"),
            ("sl", "REAL"),
            ("tp1", "REAL"),
            ("tp2", "REAL"),
            ("status", "TEXT DEFAULT 'OPEN'"),
            ("r_multiple", "REAL"),
            ("closed_at", "TEXT"),
            ("closed_price", "REAL"),
            ("max_hold_hours", "REAL DEFAULT 48"),
        ]
        for name, ctype in new_cols:
            if name not in cols:
                try:
                    self.conn.execute(f"ALTER TABLE signals ADD COLUMN {name} {ctype}")
                except Exception as e:
                    log_event(logging.WARNING, "db_migrate_col_fail", col=name, err=str(e)[:80])
        try:
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id)")
        except Exception:
            pass
        self.conn.commit()

    def record(self, signal_id: str, e: EnsembleResult, spot: float,
               levels: dict, max_hold_hours: float) -> None:
        if not self.ok or not levels:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO signals(signal_id, ts, action, spot, entry_price, sl, tp1, tp2,"
                    " score, regime, kelly_pct, lot, risk, notes, status, max_hold_hours)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?)",
                    (signal_id, _to_utc_iso(datetime.now(timezone.utc)),
                     e.action, spot,
                     levels["entry_price"], levels["sl_price"], levels["tp1_price"],
                     levels["tp2_price"], e.total_score, e.regime,
                     levels["risk_pct"], levels["lot_size"], levels["risk_amount_usd"],
                     "; ".join(e.notes), max_hold_hours))
        except Exception as ex:
            log_event(logging.WARNING, "db_write_fail", err=str(ex)[:80])

    def get_open_signals(self) -> List[dict]:
        if not self.ok:
            return []
        try:
            rows = self.conn.execute(
                "SELECT * FROM signals WHERE status='OPEN' AND action IN ('BUY','SELL')"
                " AND sl IS NOT NULL AND tp1 IS NOT NULL AND entry_price IS NOT NULL"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def close_signal(self, signal_id: str, status: str,
                     r_multiple: float, exit_price: float, exit_time: Any) -> None:
        if not self.ok:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "UPDATE signals SET status=?, r_multiple=?, closed_at=?, closed_price=?"
                    " WHERE signal_id=?",
                    (status, round(r_multiple, 4), _to_utc_iso(exit_time),
                     round(exit_price, 2), signal_id))
        except Exception as ex:
            log_event(logging.WARNING, "db_close_fail", err=str(ex)[:80])

    def get_recent_closed(self, limit: int = 10) -> List[dict]:
        if not self.ok:
            return []
        try:
            rows = self.conn.execute(
                "SELECT * FROM signals WHERE status IN ('WIN_TP1','WIN_TP2','LOSS','TIMEOUT')"
                " ORDER BY COALESCE(closed_at, ts) DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_recent_r(self, limit: int = 50) -> List[float]:
        if not self.ok:
            return []
        try:
            rows = self.conn.execute(
                "SELECT r_multiple FROM signals"
                " WHERE status IN ('WIN_TP1','WIN_TP2','LOSS','TIMEOUT')"
                " AND r_multiple IS NOT NULL"
                " ORDER BY COALESCE(closed_at, ts) DESC LIMIT ?", (limit,)
            ).fetchall()
            return [float(r[0]) for r in rows]
        except Exception:
            return []

    def get_today_r_sum(self) -> float:
        """v9: COALESCE(closed_at, ts) for robustness on legacy rows."""
        if not self.ok:
            return 0.0
        try:
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0).isoformat()
            row = self.conn.execute(
                "SELECT COALESCE(SUM(r_multiple), 0) FROM signals"
                " WHERE status IN ('WIN_TP1','WIN_TP2','LOSS','TIMEOUT')"
                " AND r_multiple IS NOT NULL"
                " AND COALESCE(closed_at, ts) >= ?", (today_start,)
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def last_action(self) -> Optional[Tuple[str, datetime, float, str]]:
        """Returns (action, ts, spot, regime)."""
        if not self.ok:
            return None
        try:
            row = self.conn.execute(
                "SELECT action, ts, spot, regime FROM signals WHERE action IN ('BUY','SELL')"
                " ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                ts = datetime.fromisoformat(row[1])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return row[0], ts, row[2], row[3] or ""
        except Exception:
            pass
        return None

    def is_dup(self, cooldown: int, action: str, spot: float, regime: str,
               tol: float = 2.0) -> bool:
        """v9: fingerprint includes regime."""
        last = self.last_action()
        if not last:
            return False
        a, ts, sp, rg = last
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        return (age < cooldown and a == action and rg == regime
                and abs(sp - spot) < tol)


# ============================================================================
# 13. RISK MANAGEMENT ENGINE  (v9: TP2 feasibility cap)
# ============================================================================
class RiskManagementEngine:
    def __init__(self, cfg: BotConfig, db: AuditDB):
        self.cfg = cfg
        self.db = db

    def _consecutive_losses(self) -> Tuple[int, str]:
        recent = self.db.get_recent_closed(self.cfg.cb_halt_consecutive)
        n = 0
        for s in recent:
            if s.get("status") == "LOSS":
                n += 1
            else:
                break
        if n >= self.cfg.cb_halt_consecutive:
            return n, f"HALT after {n} consecutive losses"
        if n >= 3:
            return n, f"derisk 50% ({n} consec losses)"
        if n >= 2:
            return n, f"derisk 25% ({n} consec losses)"
        return n, "normal"

    def _circuit_multiplier(self) -> Tuple[float, str]:
        n, reason = self._consecutive_losses()
        if n >= self.cfg.cb_halt_consecutive:
            return 0.0, reason
        if n >= 3:
            return 0.5, reason
        if n >= 2:
            return 0.75, reason
        return 1.0, reason

    def _daily_dd_multiplier(self) -> Tuple[float, str]:
        today_r = self.db.get_today_r_sum()
        if today_r <= -self.cfg.daily_dd_limit_r:
            return 0.0, f"DAILY_DD_HALT ({today_r:+.2f}R ≤ -{self.cfg.daily_dd_limit_r}R)"
        if today_r <= -self.cfg.daily_dd_derisk_r:
            return self.cfg.daily_dd_derisk_mult, f"DAILY_DD_DERISK ({today_r:+.2f}R)"
        return 1.0, f"daily OK ({today_r:+.2f}R)"

    def kelly_risk_pct(self) -> Tuple[float, str]:
        frac, cap = risk_pct_from_env()
        R_history: Optional[List[float]] = None
        if SUPABASE.enabled:
            R_history = SUPABASE.fetch_recent_r(50)
        if not R_history:
            R_history = self.db.get_recent_r(50)
        if R_history and len(R_history) >= 20:
            return kelly_fraction(R_history, fraction=frac, cap_pct=cap)
        n = len(R_history) if R_history else 0
        return self.cfg.risk_pct, f"fallback {self.cfg.risk_pct}% (n={n}<20)"

    def combined_multiplier(self) -> Tuple[float, str, str]:
        cb_mult, cb_r = self._circuit_multiplier()
        dd_mult, dd_r = self._daily_dd_multiplier()
        final = min(cb_mult, dd_mult)
        return final, cb_r, dd_r

    def compute_levels(self, sig: str, entry: float, atr_v: float, atr_mult: float,
                       blended: dict, kelly_risk_pct: float,
                       kelly_reason: str) -> Optional[dict]:
        if sig not in ("BUY", "SELL"):
            return None
        sl_d = max(atr_v * atr_mult, 1.0)
        risk_usd = self.cfg.account_size_usd * kelly_risk_pct / 100.0
        # XAUUSD: 1 lot = 100 oz; $1 move × 1 lot = $100 P/L
        raw_lot = risk_usd / (sl_d * 100.0)
        lot = round(raw_lot, 2)
        if lot < 0.01:
            lot = 0.01
        if lot * sl_d * 100.0 > risk_usd * 1.10 and lot > 0.01:
            lot = max(round(raw_lot - 0.01, 2), 0.01)
        actual_risk = lot * sl_d * 100.0

        tp1_r = blended.get("tp1_mult", 1.5) or 1.5
        tp2_r = blended.get("tp2_mult", 2.5) or 2.5

        # v9: TP2 feasibility cap
        tp2_dist_raw = sl_d * tp2_r
        tp2_dist_cap = atr_v * self.cfg.max_tp2_atr_mult
        tp2_dist = min(tp2_dist_raw, tp2_dist_cap)
        tp2_capped = tp2_dist < tp2_dist_raw

        if sig == "BUY":
            sl = entry - sl_d
            tp1 = entry + sl_d * tp1_r
            tp2 = entry + tp2_dist
            be = entry + sl_d
        else:
            sl = entry + sl_d
            tp1 = entry - sl_d * tp1_r
            tp2 = entry - tp2_dist
            be = entry - sl_d

        if tp2_capped:
            kelly_reason = f"{kelly_reason} | TP2 capped {tp2_dist_raw:.1f}→{tp2_dist:.1f} (4×ATR)"

        return {
            "entry_price": round(entry, 2), "sl_price": round(sl, 2),
            "tp1_price": round(tp1, 2), "tp2_price": round(tp2, 2),
            "break_even_trigger": round(be, 2),
            "lot_size": lot, "risk_amount_usd": round(risk_usd, 2),
            "actual_risk_usd": round(actual_risk, 2),
            "risk_pct": kelly_risk_pct, "kelly_reason": kelly_reason,
            "tp2_capped": tp2_capped,
        }


# ============================================================================
# 14. OUTCOME TRACKER
# ============================================================================
class OutcomeTracker:
    def __init__(self, db: AuditDB):
        self.db = db

    @staticmethod
    def _resolve(action: str, bar_low: float, bar_high: float,
                 sl: float, tp1: float) -> Optional[Tuple[str, float]]:
        if action == "BUY":
            if bar_low <= sl:
                return "LOSS", sl
            if bar_high >= tp1:
                return "WIN_TP1", tp1
        else:
            if bar_high >= sl:
                return "LOSS", sl
            if bar_low <= tp1:
                return "WIN_TP1", tp1
        return None

    @staticmethod
    def _compute_r(sig: dict, status: str, exit_px: float) -> float:
        entry = float(sig["entry_price"])
        sl = float(sig["sl"])
        sl_d = abs(entry - sl)
        if sl_d < 1e-6:
            return 0.0
        if status == "LOSS":
            return -1.0
        if status == "WIN_TP1":
            return abs(float(sig["tp1"]) - entry) / sl_d
        return (exit_px - entry) / sl_d if sig["action"] == "BUY" else (entry - exit_px) / sl_d

    @staticmethod
    def _parse_ts(v: Any) -> datetime:
        if isinstance(v, datetime):
            d = v
        else:
            d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if d.tzinfo is not None:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        return d

    def process(self, df_h1: pd.DataFrame) -> int:
        open_sigs = self.db.get_open_signals()
        if not open_sigs:
            return 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        df = df_h1.copy()
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
        closed = 0

        for sig in open_sigs:
            try:
                entry_t = self._parse_ts(sig["ts"])
                max_hold_h = float(sig.get("max_hold_hours") or 48.0)
                expiry_t = entry_t + timedelta(hours=max_hold_h)
                window_end = min(expiry_t, now)
                bars = df[(df["time"] > entry_t) & (df["time"] <= window_end)]

                if bars.empty:
                    if now > expiry_t:
                        last_px = float(df["close"].iloc[-1]) if not df.empty else float(sig["entry_price"])
                        r = self._compute_r(sig, "TIMEOUT", last_px)
                        self.db.close_signal(sig["signal_id"], "TIMEOUT", r, last_px,
                                             datetime.now(timezone.utc))
                        if SUPABASE.enabled:
                            SUPABASE.update_outcome(sig["signal_id"], "TIMEOUT", r, last_px)
                        log_event(logging.INFO, "outcome_timeout_no_bars",
                                  signal_id=sig["signal_id"], r=round(r, 3))
                        closed += 1
                    continue

                resolution = None
                resolution_bar = None
                for _, row in bars.iterrows():
                    res = self._resolve(sig["action"], float(row["low"]),
                                        float(row["high"]), float(sig["sl"]),
                                        float(sig["tp1"]))
                    if res:
                        resolution = res
                        resolution_bar = row
                        break

                if resolution:
                    status, px = resolution
                    r = self._compute_r(sig, status, float(px))
                    self.db.close_signal(sig["signal_id"], status, r, float(px),
                                         resolution_bar["time"])
                    if SUPABASE.enabled:
                        SUPABASE.update_outcome(sig["signal_id"], status, r, float(px))
                    log_event(logging.INFO, "outcome_resolved",
                              signal_id=sig["signal_id"], status=status,
                              r=round(r, 3), action=sig["action"])
                    closed += 1
                elif now >= expiry_t:
                    last = bars.iloc[-1]
                    exit_px = float(last["close"])
                    r = self._compute_r(sig, "TIMEOUT", exit_px)
                    self.db.close_signal(sig["signal_id"], "TIMEOUT", r, exit_px, last["time"])
                    if SUPABASE.enabled:
                        SUPABASE.update_outcome(sig["signal_id"], "TIMEOUT", r, exit_px)
                    log_event(logging.INFO, "outcome_timeout",
                              signal_id=sig["signal_id"], r=round(r, 3))
                    closed += 1
            except Exception as e:
                log_event(logging.WARNING, "outcome_eval_fail",
                          signal_id=sig.get("signal_id"), err=str(e)[:120])
        return closed


# ============================================================================
# 15. TELEGRAM NOTIFIER  (v9: retry + status footer)
# ============================================================================
def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


class TelegramNotifier:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def _post(self, payload: dict) -> bool:
        if not (self.cfg.telegram_bot_token and self.cfg.telegram_chat_id):
            return False
        if ":" not in self.cfg.telegram_bot_token:
            return False
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        for attempt in range(self.cfg.telegram_retries):
            try:
                r = requests.post(url, json=payload, timeout=self.cfg.telegram_timeout)
                if r.status_code == 200:
                    return True
                if r.status_code == 400 and "parse entities" in r.text:
                    plain = {"chat_id": payload["chat_id"],
                             "text": _strip_html(payload["text"]),
                             "disable_web_page_preview": True}
                    if "reply_markup" in payload:
                        plain["reply_markup"] = payload["reply_markup"]
                    r2 = requests.post(url, json=plain, timeout=self.cfg.telegram_timeout)
                    return r2.status_code == 200
                log_event(logging.WARNING, "telegram_retry",
                          attempt=attempt + 1, code=r.status_code, body=r.text[:200])
            except Exception as e:
                log_event(logging.WARNING, "telegram_exc",
                          attempt=attempt + 1, err=str(e)[:120])
            if attempt < self.cfg.telegram_retries - 1:
                time.sleep(2 ** attempt)
        return False

    def send_signal(self, e: EnsembleResult, spot: float, levels: Optional[dict],
                    cb_reason: str = "normal", dd_reason: str = "daily OK",
                    next_eta_min: int = 20) -> bool:
        if e.action == "WAIT" and not self.cfg.notify_wait:
            log_event(event="telegram_suppressed", action="WAIT",
                      score=e.total_score, reason=e.notes[0] if e.notes else "")
            return False

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        kb = {"inline_keyboard": [[
            {"text": "🔄 Refresh", "callback_data": "refresh"},
            {"text": "📊 Macro", "callback_data": "macro"},
        ]]}
        w = e.regime_weights
        fuzzy_str = html.escape(
            f"{e.regime} (R:{w.get('RANGING', 0):.2f} "
            f"T:{w.get('TRANSITION', 0):.2f} Tr:{w.get('TRENDING', 0):.2f})")
        safe_reason = html.escape(e.notes[0]) if e.notes else ""
        safe_kelly = html.escape(levels["kelly_reason"]) if levels else ""
        safe_cb = html.escape(cb_reason)
        safe_dd = html.escape(dd_reason)

        if e.action == "WAIT" or not levels:
            text = (f"🟡 <b>XAUUSD v{VERSION} — WAIT</b>\n"
                    f"⏰ <code>{now}</code>\n"
                    f"Score: <b>{e.total_score}%</b> | Regime: {fuzzy_str}\n"
                    f"Reason: {safe_reason}\n"
                    f"<i>CB: {safe_cb} | DD: {safe_dd}</i>\n"
                    f"<i>Next check: ~{next_eta_min}m</i>")
        else:
            em = "🟢" if e.action == "BUY" else "🔴"
            tp2_note = " (capped)" if levels.get("tp2_capped") else ""
            text = (f"{em} <b>XAUUSD v{VERSION} — {e.action}</b>\n"
                    f"⏰ <code>{now}</code>\n───────────────────\n"
                    f"Score: <b>{e.total_score}%</b>\n"
                    f"Regime: {fuzzy_str}\n"
                    f"POC: ${e.poc_level:.2f} | ATR: ${e.atr_value:.2f}\n"
                    f"<code>ENTRY : ${levels['entry_price']:.2f}\n"
                    f"SL    : ${levels['sl_price']:.2f}\n"
                    f"TP1   : ${levels['tp1_price']:.2f}\n"
                    f"TP2   : ${levels['tp2_price']:.2f}{tp2_note}\n"
                    f"BE @  : ${levels['break_even_trigger']:.2f}</code>\n"
                    f"Lot: <b>{levels['lot_size']}</b> | "
                    f"Risk: ${levels['actual_risk_usd']:.0f} ({levels['risk_pct']:.2f}%)\n"
                    f"<i>Kelly: {safe_kelly}</i>\n"
                    f"<i>CB: {safe_cb} | DD: {safe_dd}</i>")
        return self._post({"chat_id": self.cfg.telegram_chat_id, "text": text,
                           "parse_mode": "HTML", "reply_markup": kb,
                           "disable_web_page_preview": True})

    def send_failure(self, err: str) -> None:
        if not is_market_open():
            return
        self._post({
            "chat_id": self.cfg.telegram_chat_id,
            "text": f"⚠️ <b>BOT ERROR</b> [<code>{RUN_ID}</code>]\n"
                    f"<code>{html.escape(err[:400])}</code>",
            "parse_mode": "HTML",
        })


# ============================================================================
# 16. ORCHESTRATOR
# ============================================================================
class TradingBotOrchestrator:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.params, self.params_source = load_params()
        self.db = AuditDB(cfg.db_path)
        self.data = MarketDataHandler(cfg)
        self.signal_gen = SignalGenerator(self.params, self.params_source)
        self.risk = RiskManagementEngine(cfg, self.db)
        self.outcomes = OutcomeTracker(self.db)
        self.tg = TelegramNotifier(cfg)

    def _news_check(self) -> NewsResult:
        if not self.cfg.news_enabled:
            return NewsResult(allowed=True, reason="news disabled", source="bypass")
        try:
            return _live_news_guard()
        except Exception as e:
            log_event(logging.WARNING, "news_guard_fail", err=str(e)[:120])
            return NewsResult(allowed=True, reason="news guard error (bypass)",
                              source="error")

    def run(self) -> int:
        log_event(event="bot_start", version=VERSION,
                  params_source=self.params_source,
                  supabase_enabled=SUPABASE.enabled,
                  notify_wait=self.cfg.notify_wait,
                  warnings=self.cfg.warnings())

        if not SUPABASE.enabled:
            log_event(logging.WARNING, "supabase_disabled_risk",
                      note="Kelly will not activate without persistent R history")

        if not is_market_open():
            log_event(event="market_closed_or_holiday", skip=True)
            print("ACTION: SKIP — market closed or holiday")
            return 0

        try:
            md = self.data.fetch()
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
            log_event(logging.ERROR, "fatal_data", err=err)
            self.tg.send_failure(err)
            print(f"ACTION: ERROR — {err}")
            return 1

        try:
            n_closed = self.outcomes.process(md.h1)
            if n_closed:
                log_event(event="outcomes_closed", count=n_closed)
        except Exception as ex:
            log_event(logging.WARNING, "outcome_tracker_fail", err=str(ex)[:120])

        try:
            news = self._news_check()
            macro = check_macro_correlation(self.cfg)
            e = self.signal_gen.generate(md, news, macro, self.cfg)

            basis = self.data.get_basis(md)
            tradable = float(md.h1["close"].iloc[-1]) + basis

            mult, cb_r, dd_r = self.risk.combined_multiplier()
            if mult == 0.0:
                log_event(event="risk_halt", cb=cb_r, dd=dd_r)
                self.tg.send_signal(e, tradable, None, cb_r, dd_r)
                print(f"ACTION: HALT — CB:{cb_r} | DD:{dd_r}")
                return 0

            kelly_base, kelly_reason = self.risk.kelly_risk_pct()
            kelly_final = round(kelly_base * mult, 2)
            if mult < 1.0:
                kelly_reason = f"{kelly_reason} | mult={mult:.2f} | CB:{cb_r} | DD:{dd_r}"

            levels = self.risk.compute_levels(
                e.action, tradable, e.atr_value, e.atr_multiplier,
                e.blended_params, kelly_final, kelly_reason)

            if levels and self.db.is_dup(self.cfg.signal_cooldown_min, e.action,
                                         tradable, e.regime):
                log_event(event="signal_deduped", action=e.action,
                          regime=e.regime, spot=round(tradable, 2))
                print(f"ACTION: {e.action} (deduped) | Score: {e.total_score}%")
                return 0

            if levels and e.action in ("BUY", "SELL"):
                signal_id = uuid.uuid4().hex
                max_hold_h = float(e.blended_params.get("max_hold", 48.0))
                self.db.record(signal_id, e, tradable, levels, max_hold_h)
                try_insert_signal(
                    signal_id=signal_id, action=e.action, spot=tradable,
                    entry_price=levels["entry_price"],
                    sl=levels["sl_price"], tp1=levels["tp1_price"],
                    tp2=levels["tp2_price"],
                    score=e.total_score, regime=e.regime,
                    kelly_frac=kelly_final / 100.0, lot=levels["lot_size"],
                    risk_usd=levels["actual_risk_usd"],
                    max_hold_hours=max_hold_h,
                    notes="; ".join(e.notes))
            else:
                signal_id = None

            sent = self.tg.send_signal(e, tradable, levels, cb_r, dd_r)

            log_event(event="signal_emitted", action=e.action, score=e.total_score,
                      threshold=e.threshold, gap=e.gap_to_threshold,
                      spot=round(tradable, 2), basis=round(basis, 2),
                      source=md.source, m15_synthetic=md.m15_synthetic,
                      regime=e.regime, regime_weights=e.regime_weights,
                      macro_severity=macro.severity,
                      params_source=e.params_source,
                      kelly_base=kelly_base, kelly_final=kelly_final,
                      kelly_reason=kelly_reason, mult=mult, cb=cb_r, dd=dd_r,
                      signal_id=signal_id, tg=sent)
            print(f"ACTION: {e.action} | Score: {e.total_score}% "
                  f"(thr {e.threshold:.1f}, gap {e.gap_to_threshold:+.1f}) | "
                  f"Regime: {e.regime} | Kelly: {kelly_final:.2f}% (×{mult}) | "
                  f"Spot: {tradable:.2f} | TG: {sent} | CB: {cb_r} | DD: {dd_r}")
            return 0
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
            log_event(logging.ERROR, "fatal_engine", err=err)
            self.tg.send_failure(err)
            print(f"ACTION: ERROR — {err}")
            return 1


# ============================================================================
# 17. MAIN
# ============================================================================
def main() -> int:
    return TradingBotOrchestrator(BotConfig.from_env()).run()


if __name__ == "__main__":
    sys.exit(main())
