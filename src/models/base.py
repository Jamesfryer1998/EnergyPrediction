"""Abstract forecaster + registry. Every model sees the same tabular X / y."""
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

MODEL_REGISTRY: dict[str, type[Forecaster]] = {}


class Forecaster(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    tunable: ClassVar[bool] = True
    # True if the model holds out its own validation data for early stopping, so production
    # training can be a single fit on all data (no separate validate-then-refit pass).
    self_validating: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            MODEL_REGISTRY[cls.name] = cls

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.feature_names: list[str] = []

    # ── contract ─────────────────────────────────────────────────────────
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> None:
        """Train on X/y. X_val/y_val (may be None) are for early stopping."""

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one prediction per row of X."""

    @classmethod
    def search_space(cls, trial) -> dict[str, Any]:
        """Optuna search space. Return {} if not tunable."""
        return {}

    # ── public API ───────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> Forecaster:
        X = X[usable_columns(X)]
        self.feature_names = list(X.columns)
        self._fit(X, y, None if X_val is None else self._align(X_val), y_val)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict(self._align(X)), dtype=float)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_names) if self.feature_names else X

    def feature_importance(self) -> pd.Series | None:
        return None

    def refit_params(self) -> dict[str, Any]:
        """Params to use when refitting on train+validation after early stopping
        (e.g. the number of boosting rounds that early stopping chose)."""
        return {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        path.with_suffix(".json").write_text(
            json.dumps({"model": self.name, "params": self.params, "n_features": len(self.feature_names)}, default=str, indent=2)
        )

    @staticmethod
    def load(path: str | Path) -> Forecaster:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def usable_columns(X: pd.DataFrame) -> list[str]:
    """Columns that carry information in the training data. A source with no data yet (all
    NaN), or one constant over the whole training window, can't be learned from; worse, it
    breaks some models (sklearn's HistGradientBoosting fails on an all-NaN column, and a
    constant column has ~zero spread, so standardising later values blows up). Such columns
    are left out of the model until the data is there."""
    if X.empty:
        return list(X.columns)
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return keep or list(X.columns)


def create_model(name: str, **params) -> Forecaster:
    from . import zoo  # noqa: F401  (registers models)

    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


def available_models() -> list[str]:
    from . import zoo  # noqa: F401

    return sorted(MODEL_REGISTRY)
