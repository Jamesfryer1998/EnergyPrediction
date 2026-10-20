"""Logging: console + rotating file under logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from settings import PROJECT_ROOT

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str = "INFO", logfile: str = "energy_prediction.log") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = RotatingFileHandler(log_dir / logfile, maxBytes=10_000_000, backupCount=5)
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)
    for noisy in ("urllib3", "matplotlib", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
