"""Concrete forecasters. Heavy libraries are imported lazily so the rest of the
package (ingest, web) works even where torch/lightgbm aren't installed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """Baseline: value one week earlier (falls back to the shortest available lag)."""

    name = "seasonal_naive"
    description = "Same half-hour last week"
    tunable = False
    self_validating = True

    def _fit(self, X, y, X_val=None, y_val=None):
        lag_cols = [c for c in X.columns if "__lag" in c and c.startswith(f"{y.name}__")]
        by_lag = sorted(lag_cols, key=lambda c: abs(int(c.rsplit("lag", 1)[1]) - 336))
        self.col_ = by_lag[0] if by_lag else None
        self.mean_ = float(y.mean())

    def _predict(self, X):
        if self.col_ is None or self.col_ not in X:
            return np.full(len(X), self.mean_)
        return X[self.col_].fillna(self.mean_).to_numpy()


class LinearRidge(Forecaster):
    """Ridge (L2-regularised linear) regression: a fast, transparent benchmark between the
    seasonal-naive baseline and the tree/neural models. Inputs are standardised (missing values ->
    training median, extreme values clipped) so the coefficients are comparable and stable."""

    name = "linear"
    description = "Ridge linear regression (standardised inputs)"
    self_validating = True  # no early stopping: one fit on all the training data
    default_params = {"alpha": 10.0}
    Z_CLIP = 8.0

    @classmethod
    def search_space(cls, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e4, log=True)}

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.to_numpy(dtype=np.float64)
        arr = np.where(np.isnan(arr), self.med_, arr)
        return np.clip((arr - self.mu_) / self.sd_, -self.Z_CLIP, self.Z_CLIP)

    def _fit(self, X, y, X_val=None, y_val=None):
        from sklearn.linear_model import Ridge

        if X_val is not None and len(X_val):  # nothing to early-stop: use all of it
            X, y = pd.concat([X, X_val]), pd.concat([y, y_val])
        arr = X.to_numpy(dtype=np.float64)
        self.med_ = np.nan_to_num(np.nanmedian(arr, axis=0))
        filled = np.where(np.isnan(arr), self.med_, arr)
        self.mu_ = filled.mean(axis=0)
        sd = filled.std(axis=0)
        floor = np.maximum(1e-3 * np.abs(self.mu_), 1e-9)
        self.sd_ = np.where(sd > floor, sd, np.maximum(np.abs(self.mu_), 1.0))
        span = float(y.max() - y.min()) or 1.0
        self.ylim_ = (float(y.min()) - span, float(y.max()) + span)
        self.model_ = Ridge(alpha=float(self.params["alpha"]))
        self.model_.fit(self._prep(X), y.to_numpy(dtype=np.float64))

    def _predict(self, X):
        return np.clip(self.model_.predict(self._prep(X)), *self.ylim_)

    def feature_importance(self):
        return pd.Series(np.abs(self.model_.coef_), index=self.feature_names).sort_values(ascending=False)


class HistGradientBoosting(Forecaster):
    """scikit-learn HistGradientBoosting — LightGBM-style, zero extra dependencies."""

    name = "hgb"
    description = "sklearn HistGradientBoostingRegressor"
    self_validating = True  # early-stops on an internal 10% split
    # squared_error is ~1.3-2x faster than absolute_error; tuning can still pick absolute_error
    default_params = {"learning_rate": 0.08, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 40,
                      "l2_regularization": 0.1, "loss": "squared_error"}

    @classmethod
    def search_space(cls, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1500, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "loss": trial.suggest_categorical("loss", ["squared_error", "absolute_error"]),
        }

    def _fit(self, X, y, X_val=None, y_val=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model_ = HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.1,
                                                    n_iter_no_change=30, random_state=0, **self.params)
        self.model_.fit(X, y)

    def _predict(self, X):
        return self.model_.predict(X)


class LightGBM(Forecaster):
    name = "lightgbm"
    description = "LightGBM gradient boosted trees"
    default_params = {"objective": "l1", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 40,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "lambda_l2": 1.0, "n_estimators": 3000}

    @classmethod
    def search_space(cls, trial):
        return {
            "objective": trial.suggest_categorical("objective", ["l1", "l2", "huber"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10, log=True),
            "n_estimators": 2500,
        }

    def _fit(self, X, y, X_val=None, y_val=None):
        try:
            import lightgbm as lgb
        except OSError as exc:  # macOS pip wheel without the OpenMP runtime
            raise ImportError(
                "LightGBM is installed but can't load its OpenMP library (libomp). Fix with: "
                "conda activate EnergyPrediction && conda install -c conda-forge llvm-openmp"
            ) from exc

        self.model_ = lgb.LGBMRegressor(verbose=-1, random_state=0, **self.params)
        kw = {}
        if X_val is not None and len(X_val):
            kw = {"eval_set": [(X_val, y_val)], "callbacks": [lgb.early_stopping(100, verbose=False)]}
        self.model_.fit(X, y, **kw)

    def _predict(self, X):
        return self.model_.predict(X)

    def refit_params(self):
        best = getattr(self.model_, "best_iteration_", None)
        return {"n_estimators": max(50, int(best * 1.1))} if best else {}

    def feature_importance(self):
        return pd.Series(self.model_.booster_.feature_importance("gain"), index=self.feature_names).sort_values(ascending=False)


class TorchMLP(Forecaster):
    """PyTorch feed-forward net on the same tabular features (deep-learning comparison).
    Sequence models (LSTM / TFT) can be added as further Forecaster subclasses."""

    name = "torch_mlp"
    description = "PyTorch MLP (standardised inputs, early stopping)"
    self_validating = True  # carves the last 10% as validation when none is given
    default_params = {"hidden": 256, "layers": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5,
                      "batch_size": 512, "epochs": 60, "patience": 8}

    @classmethod
    def search_space(cls, trial):
        return {
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256, 512]),
            "layers": trial.suggest_int("layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "epochs": 80,
            "patience": 8,
        }

    Z_CLIP = 8.0  # inputs are clipped to ±8 standard deviations of the training data

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.to_numpy(dtype=np.float32)
        arr = (arr - self.mu_) / self.sd_
        # A feature far outside anything seen in training (e.g. a source that was nearly
        # constant until recently) would otherwise push the net to absurd outputs.
        arr = np.clip(arr, -self.Z_CLIP, self.Z_CLIP)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit(self, X, y, X_val=None, y_val=None):
        import torch
        from torch import nn

        torch.manual_seed(0)
        p = self.params
        arr = X.to_numpy(dtype=np.float64)
        self.mu_ = np.nan_to_num(np.nanmean(arr, axis=0)).astype(np.float32)
        sd = np.nan_to_num(np.nanstd(arr, axis=0))
        # near-constant columns: don't divide by ~0 (scale by their magnitude instead)
        floor = np.maximum(1e-3 * np.abs(self.mu_), 1e-6)
        self.sd_ = np.where(sd > floor, sd, np.maximum(np.abs(self.mu_), 1.0)).astype(np.float32)
        self.ymu_, self.ysd_ = float(y.mean()), float(y.std() or 1.0)
        span = float(y.max() - y.min()) or 1.0
        self.ylim_ = (float(y.min()) - span, float(y.max()) + span)  # outputs stay within a sane range

        if X_val is None or not len(X_val):  # carve the last 10% as validation
            cut = int(len(X) * 0.9)
            X, X_val, y, y_val = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

        def tensors(Xd, yd):
            return (torch.from_numpy(self._prep(Xd)),
                    torch.from_numpy(((yd.to_numpy(dtype=np.float32) - self.ymu_) / self.ysd_)).unsqueeze(1))

        Xt, yt = tensors(X, y)
        Xv, yv = tensors(X_val, y_val)
        layers, d = [], Xt.shape[1]
        for _ in range(int(p["layers"])):
            layers += [nn.Linear(d, int(p["hidden"])), nn.GELU(), nn.Dropout(float(p["dropout"]))]
            d = int(p["hidden"])
        layers.append(nn.Linear(d, 1))
        net = nn.Sequential(*layers)
        opt = torch.optim.AdamW(net.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
        loss_fn = nn.HuberLoss()
        best, best_state, bad = float("inf"), None, 0
        bs = int(p["batch_size"])
        for _ in range(int(p["epochs"])):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(loss_fn(net(Xv), yv))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(p["patience"]):
                    break
        if best_state:
            net.load_state_dict(best_state)
        self.net_ = net.eval()

    def _predict(self, X):
        import torch

        with torch.no_grad():
            out = self.net_(torch.from_numpy(self._prep(X))).squeeze(1).numpy()
        out = out * self.ysd_ + self.ymu_
        lo, hi = getattr(self, "ylim_", (-np.inf, np.inf))  # models saved before ylim_ existed
        return np.clip(out, lo, hi)
