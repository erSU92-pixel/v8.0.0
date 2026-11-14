"""
XAUUSD & BTCUSD Quantitative Institutional Signal Bot — v12.3 APEX ADAPTIVE
==========================================================================
Dynamic Multi-Tier Adaptive Quantitative Engine:
  - Automated Regime-Based Mode Switching (SNIPER, BALANCED, GROWTH)
  - Multi-Timeframe Confluence Matrix (H4 + H1 + M15)
  - Robust SQLite WAL Auto-Migration & Fully Autonomous MetaApi Execution
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
import asyncio
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
    import metaapi_cloud_sdk
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    MetaApi = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fuzzy_regime import blend_params, dominant, membership
from kelly import kelly_fraction, risk_pct_from_env
from news_calendar import NewsResult, check_news_guard as _live_news_guard
from supabase_client import SUPABASE, try_insert_signal

VERSION = "12.3-APEX-ADAPTIVE"
RUN_ID = uuid.uuid4().hex[:8]
REGIMES = ("TRENDING", "WEAK_TREND", "RANGING")


# ============================================================================
# 1. CONFIGURATION & ASSET PROFILES
# ============================================================================
@dataclass
class AssetConfig:
    symbol: str
    twelve_symbol: str
    yf_ticker: str
    max_spread_pts: float
    max_slippage_pts: float
    is_crypto: bool
    min_lots: float

ASSET_PROFILES = {
    "XAUUSD": AssetConfig("XAUUSD", "XAU/USD", "GC=F", 35.0, 5.0, False, 0.01),
    "BTCUSD": AssetConfig("BTCUSD", "BTC/USD", "BTC-USD", 150.0, 50.0, True, 0.001)
}

@dataclass
class BotConfig:
    min_bars: int = 500
    retry_total: int = 3
    retry_backoff: float = 0.5
    http_timeout: int = 12
    telegram_timeout: int = 8
    telegram_retries: int = 3
    
    account_size_usd: float = 1_000.0
    risk_pct: float = 1.0
    max_lot_risk_pct: float = 2.5
    signal_cooldown_min: int = 30
    spot_cache_min: int = 2
    macro_cache_min: int = 30
    news_enabled: bool = True
    
    execute_trades: bool = False
    metaapi_token: Optional[str] = None
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD", "BTCUSD"])

    twelve_data_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    csv_dir: str = "data/"
    db_path: str = "data/quant_audit.db"

    @classmethod
    def from_env(cls) -> "BotConfig":
        syms_env = os.getenv("TRADING_SYMBOLS", "XAUUSD,BTCUSD")
        syms = [s.strip().upper() for s in syms_env.split(",") if s.strip()]
        return cls(
            account_size_usd=float(os.getenv("ACCOUNT_SIZE_USD", "1000")),
            risk_pct=float(os.getenv("RISK_PCT", "1.0")),
            max_lot_risk_pct=float(os.getenv("MAX_LOT_RISK_PCT", "2.5")),
            news_enabled=os.getenv("NEWS_ENABLED", "1") == "1",

            execute_trades=os.getenv("EXECUTE_TRADES", "0") == "1",
            metaapi_token=os.getenv("METAAPI_TOKEN"),
            mt5_login=os.getenv("MT5_LOGIN"),
            mt5_password=os.getenv("MT5_PASSWORD"),
            mt5_server=os.getenv("MT5_SERVER"),
            symbols=syms if syms else ["XAUUSD", "BTCUSD"],

            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            db_path=os.getenv("AUDIT_DB_PATH", "data/quant_audit.db"),
        )


# ============================================================================
# 2. LOGGING & UTILS
# ============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        p = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
             "level": record.levelname, "msg": record.getMessage()}
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict): p.update(extra)
        if record.exc_info: p["exc"] = self.formatException(record.exc_info)
        return json.dumps(p, default=str)

def get_logger(name: str = "apex-adaptive") -> logging.Logger:
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
def ema(s: pd.Series, p: int) -> pd.Series: return s.ewm(span=p, adjust=False).mean()
def sma(s: pd.Series, p: int) -> pd.Series: return s.rolling(p, min_periods=p).mean()
def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    loss = -d.clip(upper=0).ewm(alpha=1 / p, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))
def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
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
    if df["volume"].sum() <= 0: return float(df["close"].median())
    hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


# ============================================================================
# 4. SESSIONS & ADAPTIVE MULTI-TIER ENGINE
# ============================================================================
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    weeks = calendar.monthcalendar(year, month)
    days = [w[weekday] for w in weeks if w[weekday] != 0]
    return date(year, month, days[n - 1] if n > 0 else days[n])
def _easter(year: int) -> date:
    a = year % 19; b, c = divmod(year, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)
def _holiday_set(year: int) -> Set[date]:
    h = {date(year, 1, 1), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), _easter(year) - timedelta(days=2), _nth_weekday(year, 5, 0, -1), date(year, 7, 4), _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4), date(year, 12, 25)}
    if year >= 2022: h.add(date(year, 6, 19))
    return h
_HOLIDAYS: Dict[int, Set[date]] = {y: _holiday_set(y) for y in range(2024, 2031)}
def is_holiday(d: date) -> bool: return d in _HOLIDAYS.get(d.year, set())

def is_market_open(symbol: str, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    profile = ASSET_PROFILES.get(symbol)
    if profile and profile.is_crypto: return True
    if is_holiday(now.date()): return False
    wd, h = now.weekday(), now.hour
    if wd == 5: return False
    if wd == 6: return h >= 22
    if wd == 4: return h < 21
    return True

def get_adaptive_profile(symbol: str, adx_val: float) -> Tuple[str, str, float, float]:
    now = datetime.now(timezone.utc)
    if not is_market_open(symbol, now): return "CLOSED", "OFFLINE", 999.0, 1.0
    
    h = now.hour
    # Dynamic Tier Selection based on Market ADX & Session Liquidity
    if adx_val > 25.0 and h in range(7, 17):
        return "GROWTH", "TREND_RIDER", 63.5, 1.6  # Growth mode: Agile & Trend-focused
    elif adx_val < 18.0 or h in range(0, 7):
        return "SNIPER", "CONSERVATIVE", 71.0, 1.2 # Sniper mode: Ultra strict defense
    else:
        return "BALANCED", "SWEET_SPOT", 66.5, 1.4 # Balanced mode: Institutional standard


# ============================================================================
# 5. DATA INGESTION
# ============================================================================
def validate_ohlcv(df: pd.DataFrame, min_bars: int, label: str) -> None:
    req = {"time", "open", "high", "low", "close", "volume"}
    miss = req - set(df.columns)
    if miss: raise ValueError(f"{label}: missing {miss}")
    if len(df) < min_bars: raise ValueError(f"{label}: {len(df)}<{min_bars}")
    if df[["open", "high", "low", "close"]].isna().any().any(): raise ValueError(f"{label}: NaN OHLC")
    if (df["high"] < df["low"]).any(): raise ValueError(f"{label}: high<low")
    if df["volume"].sum() <= 0: raise ValueError(f"{label}: zero volume")

@dataclass
class MarketData:
    h4: pd.DataFrame
    h1: pd.DataFrame
    m15: pd.DataFrame
    source: str

class MarketDataHandler:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.session = requests.Session()
        r = Retry(total=cfg.retry_total, backoff_factor=cfg.retry_backoff, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        self.session.mount("https://", HTTPAdapter(max_retries=r))
        self.session.headers.update({"User-Agent": f"ApexAdaptive/{VERSION}"})

    def fetch(self, symbol: str) -> MarketData:
        prof = ASSET_PROFILES.get(symbol, ASSET_PROFILES["XAUUSD"])
        
        # 1. Try TwelveData
        if self.cfg.twelve_data_api_key:
            try:
                h1 = self._twelve(prof.twelve_symbol, "1h", 500)
                h4 = self._twelve(prof.twelve_symbol, "4h", 200)
                m15 = self._twelve(prof.twelve_symbol, "15min", 200)
                if h1 is not None and len(h1) >= self.cfg.min_bars:
                    validate_ohlcv(h1, self.cfg.min_bars, f"12data-{symbol}-H1")
                    if h4 is None or len(h4) < 50:
                        h4 = h1.set_index("time").resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
                    if m15 is None or len(m15) < 50:
                        m15 = h1.set_index("time").resample("15min").ffill().reset_index()
                        m15["volume"] /= 4.0
                    return MarketData(h4, h1, m15, "twelvedata")
            except Exception as e:
                log_event(logging.WARNING, "twelvedata_fail", symbol=symbol, err=str(e)[:100])

        # 2. Try Yahoo Finance
        if yf is not None:
            try:
                tk = yf.Ticker(prof.yf_ticker)
                h4_raw = tk.history(period="60d", interval="4h")
                h1_raw = tk.history(period="60d", interval="1h")
                m15_raw = tk.history(period="60d", interval="15m")
                if not h1_raw.empty:
                    def norm(df: pd.DataFrame) -> pd.DataFrame:
                        df = df.reset_index()
                        tcol = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
                        df = df.rename(columns={tcol: "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
                        for c in ["open", "high", "low", "close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
                        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0
                        return df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
                    
                    h1 = norm(h1_raw)
                    h4 = norm(h4_raw) if not h4_raw.empty else h1.set_index("time").resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
                    m15 = norm(m15_raw) if not m15_raw.empty else h1.set_index("time").resample("15min").ffill().reset_index()
                    validate_ohlcv(h1, self.cfg.min_bars, f"yf-{symbol}-H1")
                    return MarketData(h4, h1, m15, "yfinance")
            except Exception as e:
                log_event(logging.WARNING, "yfinance_fail", symbol=symbol, err=str(e)[:100])

        # 3. Fallback CSV
        csv_path = os.path.join(self.cfg.csv_dir, f"{symbol.lower()}_h1.csv")
        if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
            os.makedirs(self.cfg.csv_dir, exist_ok=True)
            dummy = pd.date_range(end=datetime.now(timezone.utc).replace(tzinfo=None), periods=600, freq="h")
            base_p = 3000.0 if symbol == "XAUUSD" else 65000.0
            pd.DataFrame({"time": dummy, "open": base_p, "high": base_p+1, "low": base_p-1, "close": base_p, "volume": 1000.0}).to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns else df["time"], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0
        h1 = df.dropna().sort_values("time").reset_index(drop=True)
        h4 = h1.set_index("time").resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
        m15 = h1.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(h4, h1, m15, "csv")

    def _twelve(self, sym_str: str, interval: str, size: int) -> Optional[pd.DataFrame]:
        if not self.cfg.twelve_data_api_key: return None
        url = f"https://api.twelvedata.com/time_series?symbol={sym_str}&interval={interval}&outputsize={size}&apikey={self.cfg.twelve_data_api_key}"
        r = self.session.get(url, timeout=self.cfg.http_timeout)
        j = r.json()
        if j.get("status") == "error": return None
        vals = j.get("values") or []
        if not vals: return None
        df = pd.DataFrame(vals).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0) if "volume" in df.columns else 1000.0
        return df.sort_values("time").reset_index(drop=True)


# ============================================================================
# 6. ADAPTIVE SIGNAL GENERATOR
# ============================================================================
class SignalGenerator:
    def __init__(self): pass

    def _run_voters(self, df: pd.DataFrame) -> Tuple[float, int, int]:
        close = df["close"]; c = float(close.iloc[-1]); v = []
        v.extend([1 if c > sma(close, p).iloc[-1] else -1 for p in (5, 10, 20, 50, 100, 200)])
        v.extend([1 if ema(close, 9).iloc[-1] > ema(close, 21).iloc[-1] else -1,
                  1 if ema(close, 12).iloc[-1] > ema(close, 26).iloc[-1] else -1,
                  1 if ema(close, 50).iloc[-1] > ema(close, 200).iloc[-1] else -1])
        r = float(rsi(close, 14).iloc[-1])
        v.extend([1 if r > 50 else -1, 1 if r < 75 else -1, 1 if r > 25 else -1])
        adx_v, p_di, m_di = adx(df, 14)
        v.extend([1 if p_di.iloc[-1] > m_di.iloc[-1] else -1, 1 if adx_v.iloc[-1] > 20 else -1,
                  1 if c > close.iloc[-2] else -1, 1 if c > close.iloc[-5] else -1, 1 if c > close.iloc[-10] else -1])
        v.append(1 if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] else -1)
        poc = volume_profile_poc(df)
        v.append(1 if c > poc else -1)
        bull = sum(1 for x in v if x == 1); bear = len(v) - bull
        return max(bull, bear) / len(v), bull, bear

    def generate(self, md: MarketData, symbol: str, news: NewsResult, cfg: BotConfig) -> EnsembleResult:
        df, h4, m15 = md.h1, md.h4, md.m15
        close = df["close"]
        atr_val = float(atr(df, 14).iloc[-1])
        adx_val = float(adx(df, 14)[0].iloc[-1])
        
        # Determine Adaptive Mode & Threshold dynamically
        mode, profile_desc, threshold, atr_mult = get_adaptive_profile(symbol, adx_val)
        if mode == "CLOSED":
            return EnsembleResult(action="WAIT", total_score=0.0, voter_agreement=0.0, forensic_score=0.0, regime="CLOSED", poc_level=0.0, notes=["Market Closed"], atr_value=atr_val, atr_multiplier=1.0, threshold=999.0, skip_reason="CLOSED")

        weights = membership(adx_val)
        dom = dominant(weights)

        h4_close = h4["close"]
        h4_bull = float(h4_close.iloc[-1]) > float(ema(h4_close, 50).iloc[-1])
        m15_bull = float(ema(m15["close"], 9).iloc[-1]) > float(ema(m15["close"], 21).iloc[-1])

        last = df.iloc[-1]
        rng = max(last["high"] - last["low"], 0.01)
        body_pct = abs(last["close"] - last["open"]) / rng * 100.0
        vol_spike = last["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 1.1

        f_score = 50.0
        if body_pct > 35.0: f_score += 25.0
        if vol_spike: f_score += 25.0
        f_score = min(f_score, 100.0)

        voter_agree, bull, bear = self._run_voters(df)
        combined = round((voter_agree * 0.55 + (f_score / 100.0) * 0.45) * 100, 1)
        intended = "BUY" if bull > bear else "SELL"

        notes, action, skip_reason = [f"MODE: {mode} ({profile_desc})"], "WAIT", None
        if not news.allowed: notes.append(f"NEWS: {news.reason}"); skip_reason = "NEWS"
        elif combined < threshold: notes.append(f"ADAPTIVE GATE: {combined}% < {threshold}%"); skip_reason = "GATE"
        elif intended == "BUY" and (h4_bull or m15_bull):
            action = "BUY"; notes.append(f"ADAPTIVE BUY OK [Score: {combined}%]")
        elif intended == "SELL" and (not h4_bull or not m15_bull):
            action = "SELL"; notes.append(f"ADAPTIVE SELL OK [Score: {combined}%]")
        else:
            notes.append("Momentum alignment unconfirmed"); skip_reason = "MOMENTUM"

        return EnsembleResult(
            action=action, total_score=combined, voter_agreement=round(voter_agree, 4),
            forensic_score=round(f_score, 1), regime=dom, poc_level=round(volume_profile_poc(df), 2),
            notes=notes, atr_value=round(atr_val, 4), atr_multiplier=atr_mult, threshold=threshold, skip_reason=skip_reason
        )

@dataclass
class EnsembleResult:
    action: str
    total_score: float
    voter_agreement: float
    forensic_score: float
    regime: str
    poc_level: float
    notes: List[str]
    atr_value: float
    atr_multiplier: float
    threshold: float
    skip_reason: Optional[str] = None


# ============================================================================
# 7. AUDIT DATABASE (WAL MODE)
# ============================================================================
class AuditDB:
    def __init__(self, path: str):
        self.ok = False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            with self.conn:
                self.conn.execute("""CREATE TABLE IF NOT EXISTS signals(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts TEXT, action TEXT, spot REAL, score REAL,
                    regime TEXT, kelly_pct REAL, lot REAL, risk REAL, status TEXT DEFAULT 'OPEN')""")
                self._migrate()
            self.conn.commit()
            self.ok = True
        except Exception as e: log_event(logging.WARNING, "db_init_fail", err=str(e)[:80])

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(signals)").fetchall()}
        if "symbol" not in cols:
            try: self.conn.execute("ALTER TABLE signals ADD COLUMN symbol TEXT DEFAULT 'XAUUSD'")
            except Exception: pass
        self.conn.commit()

    def record(self, symbol: str, signal_id: str, e: EnsembleResult, spot: float, levels: dict) -> None:
        if not self.ok or not levels: return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO signals(symbol, ts, action, spot, score, regime, kelly_pct, lot, risk, status) VALUES(?,?,?,?,?,?,?,?,?, 'OPEN')",
                    (symbol, datetime.now(timezone.utc).isoformat(), e.action, spot, e.total_score, e.regime, levels["risk_pct"], levels["lot_size"], levels["risk_amount_usd"]))
        except Exception as ex: log_event(logging.WARNING, "db_write_fail", err=str(ex)[:80])

    def is_dup(self, symbol: str, cooldown: int, action: str) -> bool:
        if not self.ok: return False
        try:
            row = self.conn.execute("SELECT ts FROM signals WHERE symbol=? AND action=? ORDER BY id DESC LIMIT 1", (symbol, action)).fetchone()
            if row:
                ts = datetime.fromisoformat(row[0])
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - ts).total_seconds() / 60 < cooldown
        except Exception: pass
        return False


# ============================================================================
# 8. RISK MANAGEMENT & METAAPI EXECUTION ENGINE
# ============================================================================
class RiskManagementEngine:
    def __init__(self, cfg: BotConfig): self.cfg = cfg

    def compute_levels(self, symbol: str, sig: str, entry: float, atr_v: float, atr_mult: float) -> Optional[dict]:
        if sig not in ("BUY", "SELL"): return None
        prof = ASSET_PROFILES.get(symbol, ASSET_PROFILES["XAUUSD"])
        sl_d = max(atr_v * atr_mult, entry * 0.002)
        risk_usd = self.cfg.account_size_usd * (self.cfg.risk_pct / 100.0)
        raw_lot = risk_usd / (sl_d * (100.0 if symbol == "XAUUSD" else 1.0))
        
        lot = max(round(raw_lot, 3 if prof.is_crypto else 2), prof.min_lots)
        actual_risk = lot * sl_d * (100.0 if symbol == "XAUUSD" else 1.0)
        
        sl = entry - sl_d if sig == "BUY" else entry + sl_d
        tp1 = entry + sl_d * 1.5 if sig == "BUY" else entry - sl_d * 1.5
        tp2 = entry + sl_d * 3.0 if sig == "BUY" else entry - sl_d * 3.0

        return {
            "skip": False, "entry_price": round(entry, 2), "sl_price": round(sl, 2),
            "tp1_price": round(tp1, 2), "tp2_price": round(tp2, 2),
            "lot_size": lot, "risk_amount_usd": round(actual_risk, 2), "risk_pct": self.cfg.risk_pct
        }


class MetaApiExecutionEngine:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.enabled = cfg.execute_trades and all([cfg.metaapi_token, cfg.mt5_login, cfg.mt5_password, cfg.mt5_server, MetaApi])

    async def _execute(self, symbol: str, action: str, levels: dict) -> str:
        if not self.enabled or not levels or levels.get("skip"): return "DISABLED/SKIPPED"
        prof = ASSET_PROFILES.get(symbol, ASSET_PROFILES["XAUUSD"])
        
        for attempt in range(1, 4):
            try:
                api = MetaApi(self.cfg.metaapi_token)
                accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
                account = None
                async for acc in accounts:
                    if str(acc.login) == str(self.cfg.mt5_login): account = acc; break
                
                if not account: return f"FAILED: Account login {self.cfg.mt5_login} not found"

                await account.wait_connected(timeout_in_seconds=60)
                connection = account.get_rpc_connection()
                await connection.connect()
                await connection.wait_synchronized()
                
                price_info = await connection.get_symbol_price(symbol)
                spread_pts = abs(float(price_info.get("ask", 0)) - float(price_info.get("bid", 0)))
                
                if spread_pts > prof.max_spread_pts * 3:
                    return f"ABORTED: SPREAD_TOO_WIDE ({spread_pts:.1f} pts)"

                if action == "BUY": res = await connection.create_market_buy_order(symbol, levels["lot_size"], levels["sl_price"], levels["tp2_price"])
                else: res = await connection.create_market_sell_order(symbol, levels["lot_size"], levels["sl_price"], levels["tp2_price"])
                
                order_id = res.get('orderId') or res.get('stringCode') or 'OK'
                return f"SUCCESS [ID: {order_id}]"
            except Exception as e:
                err_msg = str(e)
                log_event(logging.ERROR, "metaapi_exec_err", attempt=attempt, err=err_msg)
                if attempt == 3: return f"FAILED: {err_msg[:120]}"
                await asyncio.sleep(2 ** attempt)
        return "FAILED: Max retries exhausted"

    def execute_trade(self, symbol: str, action: str, levels: dict) -> str:
        if not self.enabled: return "DISABLED"
        try: return asyncio.run(self._execute(symbol, action, levels))
        except Exception as e: return f"ERROR: {str(e[:100])}"


# ============================================================================
# 9. ORCHESTRATOR & MAIN
# ============================================================================
class TelegramNotifier:
    def __init__(self, cfg: BotConfig): self.cfg = cfg

    def send_signal(self, symbol: str, e: EnsembleResult, spot: float, levels: Optional[dict], exec_status: str = "") -> bool:
        if not (self.cfg.telegram_bot_token and self.cfg.telegram_chat_id): return False
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        em = "🟢" if e.action == "BUY" else ("🔴" if e.action == "SELL" else "🟡")
        text = f"{em} <b>{symbol} v{VERSION} — {e.action}</b>\n⏰ <code>{now}</code>\n───────────────────\nScore: <b>{e.total_score}%</b> (Threshold: {e.threshold}%)\nRegime: {e.regime}\nPOC: ${e.poc_level:.2f}\n"
        if levels and not levels.get("skip"):
            text += f"<code>ENTRY : ${levels['entry_price']}\nSL    : ${levels['sl_price']}\nTP2   : ${levels['tp2_price']}</code>\nLot: <b>{levels['lot_size']}</b> | Risk: ${levels['risk_amount_usd']}\n⚡ <i>{exec_status or 'MANUAL'}</i>"
        else:
            text += f"Reason: {e.notes[0] if e.notes else 'Waiting for adaptive trigger'}"

        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": self.cfg.telegram_chat_id, "text": text, "parse_mode": "HTML"}, timeout=8)
            return r.status_code == 200
        except Exception: return False


class TradingBotOrchestrator:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.db = AuditDB(cfg.db_path)
        self.data = MarketDataHandler(cfg)
        self.signal_gen = SignalGenerator()
        self.risk = RiskManagementEngine(cfg)
        self.tg = TelegramNotifier(cfg)
        self.executor = MetaApiExecutionEngine(cfg)

    def run(self) -> int:
        log_event(event="bot_start", version=VERSION, symbols=self.cfg.symbols)
        news = NewsResult(allowed=True, reason="bypassed", source="bypass")

        for symbol in self.cfg.symbols:
            try:
                if not is_market_open(symbol):
                    log_event(event="market_closed", symbol=symbol)
                    continue

                md = self.data.fetch(symbol)
                e = self.signal_gen.generate(md, symbol, news, self.cfg)
                spot = float(md.h1["close"].iloc[-1])

                levels = self.risk.compute_levels(symbol, e.action, spot, e.atr_value, e.atr_multiplier)

                if levels and self.db.is_dup(symbol, self.cfg.signal_cooldown_min, e.action):
                    log_event(event="signal_deduped", symbol=symbol, action=e.action)
                    continue

                exec_status = ""
                if levels and e.action in ("BUY", "SELL"):
                    signal_id = uuid.uuid4().hex
                    self.db.record(symbol, signal_id, e, spot, levels)
                    if self.executor.enabled:
                        exec_status = self.executor.execute_trade(symbol, e.action, levels)

                self.tg.send_signal(symbol, e, spot, levels, exec_status)
                print(f"[{symbol}] ACTION: {e.action} | Score: {e.total_score}% (Thresh: {e.threshold}%) | Spot: {spot}")
            except Exception as ex:
                log_event(logging.ERROR, "symbol_loop_fail", symbol=symbol, err=str(ex)[:150])
        return 0


def main() -> int:
    return TradingBotOrchestrator(BotConfig.from_env()).run()

if __name__ == "__main__":
    sys.exit(main())
