"""Which IG market the paper trader uses, and its contract details (cached on disk)."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from settings import Settings

log = logging.getLogger(__name__)

DEFAULT_SEARCH = "UK Natural Gas"
# Re-read the contract details twice a day: IG's gas CFD tracks a dated futures month
# (e.g. NOV-26), so the expiry used for orders changes when it rolls.
CACHE_SECONDS = 12 * 3600
# ...but re-check hourly while the market isn't open: IG reuses an epic for a later contract
# (EN.D.ICENG.FWS0.IP moved from NOV-26 to NOV-27, which is OFFLINE on demo), so a market that
# stays shut may need replacing rather than waiting for.
CLOSED_CACHE_SECONDS = 3600


def trading_dir(settings: Settings) -> Path:
    d = settings.data_dir / "trading"
    d.mkdir(parents=True, exist_ok=True)
    return d


def market_cache_path(settings: Settings) -> Path:
    return trading_dir(settings) / "market.json"


def load_market(settings: Settings) -> dict | None:
    p = market_cache_path(settings)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def contract_ended(info: dict | None, now: datetime | None = None) -> bool:
    """True once a dated contract can no longer be traded: its last dealing time has passed,
    or (if IG didn't give one) its delivery month has started."""
    from .ig import contract_month_ended

    if not info:
        return False
    now = now or datetime.now(timezone.utc)
    last = info.get("last_dealing")
    if last:
        try:
            ts = datetime.fromisoformat(str(last))
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return now > ts
        except ValueError:
            pass
    return contract_month_ended(info.get("expiry"), now)


def resolve_market(settings: Settings, client=None, refresh: bool = False) -> dict:
    """The traded market: `trading.epic` if set, else the best IG match for `trading.market_search`.
    Contract details (value per point, minimum size, expiry) are cached in data/trading/market.json.

    IG's UK gas CFD is a dated futures month (e.g. NOV-26) that stops trading late in the month
    before. When the cached contract has ended, the next one is found by searching again, so the
    trader doesn't sit on an expired market reporting "market closed" with no new candles."""
    from .ig import IGError

    cfg = settings.get("trading", {}) or {}
    cached = load_market(settings)
    pinned = cfg.get("epic")
    search = cfg.get("market_search", DEFAULT_SEARCH)
    max_age = CACHE_SECONDS if (cached or {}).get("market_status", "TRADEABLE") == "TRADEABLE" else CLOSED_CACHE_SECONDS
    fresh = cached and time.time() - float(cached.get("checked_at", 0)) < max_age
    ended = contract_ended(cached)
    if cached and fresh and not refresh and not ended and (not pinned or cached.get("epic") == pinned):
        return cached
    if client is None:
        from .ig import IGClient

        client = IGClient()

    epic = pinned
    if not epic and cached and not refresh and not ended:
        epic = cached.get("epic")  # keep the market we found before; just refresh its details
    info = None
    if epic:
        try:
            info = client.instrument(epic)
        except IGError:
            if pinned:
                raise
            info = None  # e.g. an expired contract IG no longer lists: search again below
        if info is not None and not pinned and contract_ended(info):
            info = None
        if info is not None and not pinned and (info.get("market_status") or "TRADEABLE") != "TRADEABLE":
            info = _better_market(client, search, info) or info
    if info is None:
        skip = (cached.get("epic"),) if cached and contract_ended(cached) else ()
        m = client.find_market(search, exclude=skip)
        info = client.instrument(m["epic"])
        log.info("IG market for %r: %s %s (%s)", search, m.get("instrumentName"), info.get("expiry"), m["epic"])
    elif pinned and contract_ended(info):
        log.warning("pinned IG market %s (%s) has ended; set trading.epic to the next contract", pinned,
                    info.get("expiry"))

    if cached and cached.get("epic") != info["epic"]:
        log.warning("IG contract rolled: %s (%s) -> %s (%s); candles from the new contract are appended "
                    "to the same price history", cached.get("epic"), cached.get("expiry"), info["epic"],
                    info.get("expiry"))
    elif cached and cached.get("expiry") != info.get("expiry"):
        log.info("IG market %s rolled from %s to %s", info["epic"], cached.get("expiry"), info.get("expiry"))
    info["checked_at"] = time.time()
    market_cache_path(settings).write_text(json.dumps(info, indent=2))
    return info


def _better_market(client, search: str, current: dict) -> dict | None:
    """The cached market isn't open. If IG lists an open version of the same market (or, when
    ours is OFFLINE, one that's merely shut for the night), return its details to switch to.
    Night-time and weekend closures find nothing better, so nothing changes then."""
    from .ig import IGError, _status_rank

    try:
        m = client.find_market(search, exclude=(current["epic"],))
    except IGError:
        return None
    if _status_rank(m.get("marketStatus")) >= _status_rank(current.get("market_status")):
        return None
    info = client.instrument(m["epic"])
    log.warning("IG market %s (%s) is %s; switching to %s (%s), which is %s", current["epic"],
                current.get("expiry"), current.get("market_status"), info["epic"], info.get("expiry"),
                info.get("market_status") or m.get("marketStatus"))
    return info


def value_per_point(settings: Settings, market: dict | None) -> float:
    """GBP per one-point move for one contract. `trading.value_per_point` overrides IG's figure."""
    override = (settings.get("trading", {}) or {}).get("value_per_point")
    if override:
        return float(override)
    v = (market or {}).get("value_per_point")
    return float(v) if v else 1.0
