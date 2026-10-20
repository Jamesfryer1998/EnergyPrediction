"""Settings loader: config/settings.yaml deep-merged with config/overrides.yaml."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"


DOTENV_KEYS: set[str] = set()  # keys that came from .env (rather than the shell)


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=value lines from the project's .env into the environment (existing values win).
    Keeps secrets such as the IG demo login out of the config files and out of git."""
    path = path or PROJECT_ROOT / ".env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            DOTENV_KEYS.add(key)


load_dotenv()


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Settings:
    """Thin wrapper around the merged settings dict with a few typed helpers."""

    def __init__(self, raw: dict[str, Any], path: Path | None = None):
        self.raw = raw
        self.path = path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data_dir(self) -> Path:
        env = os.environ.get("ENERGY_DATA_DIR")
        path = Path(env or self.raw.get("data_dir", "data"))
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def timezone(self) -> str:
        return self.raw.get("timezone", "Europe/London")

    def region(self, name: str) -> dict:
        return self.raw.get("regions", {}).get(name, {})

    def source_cfg(self, name: str) -> dict:
        return self.raw.get("sources", {}).get(name, {}) or {}

    def source_enabled(self, name: str) -> bool:
        return bool(self.source_cfg(name).get("enabled", False))

    @property
    def targets(self) -> dict[str, dict]:
        return self.raw.get("targets", {})


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    path = Path(path or os.environ.get("ENERGY_CONFIG") or CONFIG_DIR / "settings.yaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    override_path = path.parent / "overrides.yaml"
    if override_path.exists():
        with open(override_path) as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings(raw, path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def save_overrides(changes: dict, settings: Settings | None = None) -> Path:
    """Merge `changes` into config/overrides.yaml (used by the dashboard's settings panel)."""
    settings = settings or get_settings()
    override_path = (settings.path or CONFIG_DIR / "settings.yaml").parent / "overrides.yaml"
    current = {}
    if override_path.exists():
        current = yaml.safe_load(override_path.read_text()) or {}
    merged = _deep_merge(current, changes)
    override_path.write_text(yaml.safe_dump(merged, sort_keys=False))
    get_settings.cache_clear()
    return override_path
