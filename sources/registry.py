"""Lookup and bulk operations over registered data sources."""
from __future__ import annotations

import logging
from dataclasses import asdict

from settings import Settings, get_settings
from storage import ParquetStore
from .base import REGISTRY, DataSource, ManualCsvSource, UpdateResult

log = logging.getLogger(__name__)


def list_sources() -> list[dict]:
    out = []
    for cls in sorted(REGISTRY.values(), key=lambda c: (c.category, c.name)):
        info = cls.info()
        info["mode"] = "manual-csv" if issubclass(cls, ManualCsvSource) else "api"
        out.append(info)
    return out


def build_source(name: str, region: str = "GB", store=None, settings: Settings | None = None) -> DataSource:
    if name not in REGISTRY:
        raise KeyError(f"unknown source {name!r}; known: {sorted(REGISTRY)}")
    settings = settings or get_settings()
    store = store or ParquetStore(settings.data_dir)
    return REGISTRY[name](region=region, store=store, settings=settings)


def enabled_sources(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    return [n for n in REGISTRY if settings.source_enabled(n)]


def ingest_all(
    region: str = "GB",
    names: list[str] | None = None,
    start=None,
    end=None,
    full: bool = False,
    settings: Settings | None = None,
    store: ParquetStore | None = None,
    progress=None,
) -> list[dict]:
    """Incrementally update sources concurrently. `progress(done, total, label)` is called as
    each source finishes."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    settings = settings or get_settings()
    store = store or ParquetStore(settings.data_dir)
    todo = names or enabled_sources(settings)
    workers = max(1, min(int(settings.get("ingest.workers", 4) or 4), len(todo) or 1))

    def run(name: str) -> UpdateResult:
        log.info("ingest %s/%s", name, region)
        try:
            return build_source(name, region, store, settings).update(start, end, full)
        except Exception as exc:
            log.exception("ingest %s failed", name)
            return UpdateResult(name, region, status="error", message=str(exc))

    results: dict[str, UpdateResult] = {}
    if progress:
        progress(0, len(todo), ", ".join(todo[:3]) + ("…" if len(todo) > 3 else ""))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest") as pool:
        futs = {pool.submit(run, n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            if progress:
                progress(i, len(todo), f"{futs[fut]} done")
    return [asdict(results[n]) for n in todo]
