from pathlib import Path

from settings import load_settings
from storage import ParquetStore


def make_env(tmp_path: Path, **overrides):
    base = {
        "data_dir": str(tmp_path / "data"),
        "training": {"train_days": 120, "validation_days": 14},
        "backtest": {"days": 5, "retrain_every_days": 5},
        "models": {"candidates": ["seasonal_naive", "hgb"]},
    }
    for k, v in overrides.items():
        base[k] = v
    settings = load_settings(overrides=base)
    return settings, ParquetStore(settings.data_dir)
