"""
news_calendar.py — Live macro calendar (ForexFactory feed).
============================================================
Free, no auth. Weekly calendar with TTL cache.
Returns blackout windows around High-impact USD events.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Static fallback (recurring windows) — used when API unavailable
# ---------------------------------------------------------------------------
STATIC_BLACKOUTS: List[Tuple[int, int, int, str]] = [
    # (weekday, start_h_utc, end_h_utc, label)
    (2, 18, 20, "FOMC"),
    (1, 12, 13, "CPI"),
    (3, 12, 13, "PPI"),
    (4, 12, 14, "NFP"),
]

TIER1_KEYWORDS = ["FOMC", "FED", "NONFARM", "NFP", "CPI", "INTEREST RATE", "POWELL"]
STANDARD_BUFFER_MIN = 30
TIER1_BUFFER_MIN = 45


def _get_log() -> logging.Logger:
    return logging.getLogger("xauusd")


def _log_event(level: int, event: str, **kw) -> None:
    lg = _get_log()
    rec = lg.makeRecord(lg.name, level, "", 0, event, (), None)
    rec.extra_data = kw
    lg.handle(rec)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class NewsEvent:
    title: str
    country: str
    impact: str
    ts_utc: datetime

    def __str__(self) -> str:
        return f"{self.ts_utc.strftime('%a %H:%M')} {self.country}/{self.impact}: {self.title[:40]}"


@dataclass
class NewsBlackout:
    start: datetime
    end: datetime
    label: str


@dataclass
class NewsResult:
    allowed: bool
    reason: str
    source: str = "live"
    active_blackout: Optional[NewsBlackout] = None
    next_event: Optional[NewsEvent] = None


# ---------------------------------------------------------------------------
# Fetch & cache
# ---------------------------------------------------------------------------
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_PATH = os.getenv("NEWS_CACHE_PATH", "data/news_cache.json")
CACHE_TTL_MIN = int(os.getenv("NEWS_CACHE_TTL_MIN", "60"))


def _load_cache() -> Optional[Tuple[datetime, List[NewsEvent]]]:
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH) as f:
            j = json.load(f)
        ts = datetime.fromisoformat(j["fetched_at"])
        events = [
            NewsEvent(
                title=e["title"],
                country=e["country"],
                impact=e["impact"],
                ts_utc=datetime.fromisoformat(e["ts_utc"]),
            )
            for e in j["events"]
        ]
        return ts, events
    except Exception:
        return None


def _save_cache(events: List[NewsEvent]) -> None:
    try:
        d = os.path.dirname(CACHE_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "events": [
                    {"title": e.title, "country": e.country,
                     "impact": e.impact, "ts_utc": e.ts_utc.isoformat()}
                    for e in events
                ],
            }, f, indent=2)
    except Exception as ex:
        _log_event(logging.WARNING, "news_cache_save_fail", err=str(ex)[:80])


def _fetch_ff() -> List[NewsEvent]:
    r = requests.get(FF_URL, timeout=12, headers={"User-Agent": "XAUUSD-Quant/8.0"})
    r.raise_for_status()
    raw = r.json()
    out: List[NewsEvent] = []
    for e in raw:
        try:
            ts_str = e.get("date", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            ts_utc = ts.astimezone(timezone.utc).replace(tzinfo=None)
            out.append(NewsEvent(
                title=str(e.get("title", "")),
                country=str(e.get("country", "")),
                impact=str(e.get("impact", "")),
                ts_utc=ts_utc,
            ))
        except Exception:
            continue
    return out


def _load_events() -> Tuple[List[NewsEvent], str]:
    """Return (events, source_tag) where source_tag in {cache, live, static}."""
    cached = _load_cache()
    if cached:
        ts, events = cached
        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_min < CACHE_TTL_MIN:
            return events, "cache"

    try:
        events = _fetch_ff()
        if events:
            _save_cache(events)
            _log_event(logging.INFO, "news_fetched", count=len(events))
            return events, "live"
    except Exception as ex:
        _log_event(logging.WARNING, "news_fetch_fail", err=str(ex)[:120])

    if cached:
        return cached[1], "cache"

    return [], "static"


# ---------------------------------------------------------------------------
# Blackout logic
# ---------------------------------------------------------------------------
def _is_tier1(title: str) -> bool:
    t = title.upper()
    return any(k in t for k in TIER1_KEYWORDS)


def _blackout_for(event: NewsEvent) -> NewsBlackout:
    buf = TIER1_BUFFER_MIN if _is_tier1(event.title) else STANDARD_BUFFER_MIN
    return NewsBlackout(
        start=event.ts_utc - timedelta(minutes=buf),
        end=event.ts_utc + timedelta(minutes=buf),
        label=f"{event.impact} {event.country}: {event.title}",
    )


def _static_blackout(now: datetime) -> Optional[NewsBlackout]:
    for wd, s, e, label in STATIC_BLACKOUTS:
        if now.weekday() == wd and s <= now.hour < e:
            start = now.replace(hour=s, minute=0, second=0, microsecond=0)
            end = now.replace(hour=e, minute=0, second=0, microsecond=0)
            return NewsBlackout(start=start, end=end, label=f"STATIC {label}")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_news_guard(now: Optional[datetime] = None) -> NewsResult:
    """Live-first. Cache → static fallback. Only High-impact USD vetoes."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)

    events, source = _load_events()

    relevant = [
        e for e in events
        if e.country.upper() == "USD" and e.impact.lower() in ("high", "holiday")
    ]

    for e in relevant:
        bo = _blackout_for(e)
        if bo.start <= now <= bo.end:
            return NewsResult(
                allowed=False,
                reason=f"BLACKOUT {bo.label}",
                source=source,
                active_blackout=bo,
            )

    upcoming = sorted([e for e in relevant if e.ts_utc > now], key=lambda x: x.ts_utc)
    next_ev = (
        upcoming[0]
        if upcoming and (upcoming[0].ts_utc - now).total_seconds() < 6 * 3600
        else None
    )

    if source in ("live", "cache") and events:
        reason = "no USD-High blackout"
        if next_ev:
            reason += f" | next: {next_ev}"
        return NewsResult(allowed=True, reason=reason, source=source, next_event=next_ev)

    static_bo = _static_blackout(now)
    if static_bo:
        return NewsResult(
            allowed=False,
            reason=f"BLACKOUT {static_bo.label}",
            source="static",
            active_blackout=static_bo,
        )
    return NewsResult(allowed=True, reason="static windows clear", source="static")


def describe_next_events(limit: int = 3) -> str:
    """For Telegram 'Macro' button payload."""
    events, source = _load_events()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    upcoming = sorted(
        [e for e in events
         if e.country.upper() == "USD"
         and e.impact.lower() == "high"
         and e.ts_utc > now],
        key=lambda x: x.ts_utc,
    )[:limit]
    if not upcoming:
        return f"[{source}] No upcoming USD High events"
    lines = [f"[{source}] Upcoming USD High:"]
    for e in upcoming:
        delta_h = (e.ts_utc - now).total_seconds() / 3600
        lines.append(f"  • {e.ts_utc.strftime('%a %H:%M UTC')} ({delta_h:+.1f}h) {e.title[:40]}")
    return "\n".join(lines)
