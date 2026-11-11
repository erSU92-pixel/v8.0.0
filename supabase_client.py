"""
supabase_client.py — v8.0.0 with outcome updates.
Adds: signal_id, entry_price, status, closed_at/closed_price, max_hold_hours, update_outcome().
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _log(level: int, msg: str, **kw) -> None:
    lg = logging.getLogger("xauusd")
    rec = lg.makeRecord(lg.name, level, "", 0, msg, (), None)
    rec.extra_data = kw
    lg.handle(rec)


class SupabaseClient:
    def __init__(self) -> None:
        raw = (os.getenv("SUPABASE_URL") or "").strip()
        clean = raw.rstrip("/")
        if clean.endswith("/rest/v1"):
            clean = clean.replace("/rest/v1", "")
        if clean and not clean.startswith("http"):
            clean = f"https://{clean}"
        self.url = clean
        self.key = (os.getenv("SUPABASE_KEY") or "").strip()
        self.table = os.getenv("SUPABASE_TABLE", "signals")
        self.enabled = bool(self.url and self.key and Client)
        self.client = None
        if self.enabled:
            try:
                self.client = create_client(self.url, self.key)
            except Exception as e:
                _log(logging.ERROR, "supabase_init_fail", err=str(e)[:100])
                self.enabled = False

    def insert(self, row: Dict[str, Any]) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self.client.table(self.table).insert(row).execute()
            return True
        except Exception as e:
            _log(logging.ERROR, "supabase_insert_fail", err=str(e)[:200])
            return False

    def update_outcome(self, signal_id: str, status: str,
                       r_multiple: float, closed_price: float) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self.client.table(self.table).update({
                "status": status,
                "r_multiple": round(r_multiple, 4),
                "closed_price": round(closed_price, 2),
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("signal_id", signal_id).execute()
            return True
        except Exception as e:
            _log(logging.ERROR, "supabase_update_fail", err=str(e)[:200])
            return False

    def fetch_recent_r(self, limit: int = 50) -> Optional[List[float]]:
        if not self.enabled or not self.client:
            return None
        try:
            res = (self.client.table(self.table)
                   .select("r_multiple")
                   .in_("status", ["WIN_TP1", "WIN_TP2", "LOSS", "TIMEOUT"])
                   .not_.is_("r_multiple", "null")
                   .order("ts", desc=True)
                   .limit(limit)
                   .execute())
            return [float(x["r_multiple"]) for x in res.data
                    if x.get("r_multiple") is not None]
        except Exception:
            return None


SUPABASE = SupabaseClient()


def try_insert_signal(
    signal_id: str, action: str, spot: float, entry_price: float,
    sl: float, tp1: float, tp2: float, score: float, regime: str,
    kelly_frac: float, lot: float, risk_usd: float,
    max_hold_hours: float, notes: str,
) -> bool:
    row = {
        "signal_id": signal_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "spot": round(spot, 2),
        "entry_price": round(entry_price, 2),
        "sl": round(sl, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2),
        "score": round(score, 1),
        "regime": regime,
        "kelly_fraction": round(kelly_frac, 4),
        "lot_size": lot, "risk_usd": risk_usd,
        "max_hold_hours": max_hold_hours,
        "notes": notes[:500],
        "status": "OPEN",
    }
    ok = SUPABASE.insert(row)
    _log(logging.INFO if ok else logging.WARNING,
         "supabase_insert", ok=ok, action=action, score=score)
    return ok
