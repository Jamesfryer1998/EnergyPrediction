"""Which IG market the paper trader uses, and its contract details (cached on disk)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from settings import Settings

log = logging.getLogger(__name__)

DEFAULT_SEARCH = "UK Natural Gas"
# Re-read the contract details twice a day: IG's gas CFD tracks a dated futures month
# (e.g. NOV-26), so the expiry used for orders changes when it rolls.
CACHE_SECONDS = 12 * 3600


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


def resolve_market(settings: Settings, client=None, refresh: bool = False) -> dict:
    """The traded market: `trading.epic` if set, else the best IG match for `trading.market_search`.
    Contract details (value per point, minimum size, expiry) are cached in data/trading/market.json."""
    cfg = settings.get("trading", {}) or {}
    cached = load_market(settings)
    epic = cfg.get("epic")
    fresh = cached and time.time() - float(cached.get("checked_at", 0)) < CACHE_SECONDS
    if cached and fresh and not refresh and (not epic or cached.get("epic") == epic):
        return cached
    if client is None:
        from .ig import IGClient

        client = IGClient()
    if not epic and cached and not refresh:
        epic = cached.get("epic")  # keep the market we found before; just refresh its details
    if not epic:
        m = client.find_market(cfg.get("market_search", DEFAULT_SEARCH))
        epic = m["epic"]
        log.info("IG market for %r: %s (%s)", cfg.get("market_search", DEFAULT_SEARCH), m.get("instrumentName"), epic)
    info = client.instrument(epic)
    if cached and cached.get("epic") == epic and cached.get("expiry") != info.get("expiry"):
        log.info("IG market %s rolled from %s to %s", epic, cached.get("expiry"), info.get("expiry"))
    info["checked_at"] = time.time()
    market_cache_path(settings).write_text(json.dumps(info, indent=2))
    return info


def value_per_point(settings: Settings, market: dict | None) -> float:
    """GBP per one-point move for one contract. `trading.value_per_point` overrides IG's figure."""
    override = (settings.get("trading", {}) or {}).get("value_per_point")
    if override:
        return float(override)
    v = (market or {}).get("value_per_point")
    return float(v) if v else 1.0
