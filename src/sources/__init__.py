"""Data sources. Importing this package registers every concrete source in base.REGISTRY."""
from . import carbon_intensity, certificates, commodities, elexon, events, ig_prices, weather  # noqa: F401  (register sources)
from .base import REGISTRY, DataSource, ManualCsvSource, SourceNotImplemented, UpdateResult
from .registry import build_source, enabled_sources, ingest_all, list_sources

__all__ = [
    "REGISTRY",
    "DataSource",
    "ManualCsvSource",
    "SourceNotImplemented",
    "UpdateResult",
    "build_source",
    "enabled_sources",
    "ingest_all",
    "list_sources",
]
