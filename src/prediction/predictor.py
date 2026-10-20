"""Next-day predictions and buy/sell ranking (analysis only — no trading)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from settings import Settings, get_settings
from features import FeatureBuilder
from models import Forecaster
from storage import TS, ParquetStore
from training import Artifacts, fit_model, resolve_params
from compute import run_isolated
from models.ensemble import NAME as ENSEMBLE

log = logging.getLogger(__name__)

PRICE_TARGET_PREFIX = "price"


class Predictor:
    def __init__(self, settings: Settings | None = None, store: ParquetStore | None = None, region: str = "GB"):
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(self.settings.data_dir)
        self.region = region
        self.art = Artifacts(self.settings, region)
        self.tz = self.settings.timezone

    # ── helpers ──────────────────────────────────────────────────────────
    def day_bounds(self, day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(day).tz_localize(self.tz)
        end = (start + pd.Timedelta(days=1)).normalize()
        return start.tz_convert("UTC"), end.tz_convert("UTC")

    def tomorrow(self) -> date:
        return (pd.Timestamp.now(tz=self.tz) + pd.Timedelta(days=1)).date()

    def champion_name(self, target: str) -> str:
        champ = self.art.load_champion(target)
        if champ:
            return champ["model"]
        cands = self.settings.get("models.candidates", ["hgb"])
        return "hgb" if "hgb" in cands else cands[0]

    # ── data availability ────────────────────────────────────────────────
    def last_data(self, target: str) -> pd.Timestamp | None:
        return self.store.last_timestamp(self.settings.targets[target]["source"], self.region)

    def max_predictable_date(self, target: str) -> date | None:
        """Latest delivery day whose every half-hour still has all lag features.
        With min_lag = 96 that is roughly two days past the last data point."""
        last = self.last_data(target)
        if last is None:
            return None
        min_lag = int(self.settings.get("features.min_lag", 96))
        horizon_end = last + pd.Timedelta(minutes=30 * (min_lag + 1))  # exclusive end that is fully covered
        # days are fully covered up to the local midnight at or before horizon_end
        return (horizon_end.tz_convert(self.tz).normalize() - pd.Timedelta(hours=1)).date()

    # ── training ─────────────────────────────────────────────────────────
    def _training_features(self):
        """Feature matrix covering every target's training window, built once and reused
        (feature building is a large share of training time)."""
        lasts = [t for t in (self.last_data(n) for n in self.settings.targets) if t is not None]
        if not lasts:
            return None
        key = (min(lasts), max(lasts))
        if getattr(self, "_fs_key", None) != key:
            train_days = int(self.settings.get("training.train_days", 540))
            self._fs = FeatureBuilder(self.store, self.settings, self.region).build(
                key[0] - pd.Timedelta(days=train_days + 2), key[1] + pd.Timedelta(minutes=30)
            )
            self._fs_key = key
        return self._fs

    def _train(self, target: str, model_name: str) -> tuple[Forecaster, dict]:
        last = self.last_data(target)
        if last is None:
            raise RuntimeError(f"no data for {target}; run ingest first")
        train_end = last + pd.Timedelta(minutes=30)
        fs = self._training_features()
        model, metrics = fit_model(fs, target, model_name, train_end, self.settings,
                                   resolve_params(self.art, target, model_name), use_validation=False)
        return model, {"model": model_name, "val": metrics, "trained_to": train_end.isoformat()}

    @property
    def isolate(self) -> bool:
        """Fit/predict each model in its own process (see compute.run_isolated)."""
        return bool(self.settings.get("compute.isolate_models", True))

    def _train_and_save(self, target: str, model_name: str, path) -> dict:
        if not self.isolate:
            model, info = self._train(target, model_name)
            model.save(path)
            return info
        last = self.last_data(target)
        if last is None:
            raise RuntimeError(f"no data for {target}; run ingest first")
        return run_isolated(_fit_and_save, self._training_features(), self.settings.raw, self.region, target,
                            model_name, last + pd.Timedelta(minutes=30), str(path), label=f"train {target}/{model_name}")

    def train_champions(self, targets: list[str] | None = None) -> dict:
        targets = targets or list(self.settings.targets)
        out = {}
        for target in targets:
            if self.last_data(target) is None:
                out[target] = {"status": "no_data"}
                continue
            name = self.champion_name(target)
            if name == ENSEMBLE:
                out[target] = self._train_ensemble(target)
                continue
            path = self.art.model_path(target, "production")
            try:
                info = self._train_and_save(target, name, path)
            except ImportError as exc:
                log.warning("champion %s unavailable (%s); falling back to hgb", name, exc)
                info = self._train_and_save(target, "hgb", path)
            out[target] = {"status": "ok", **info}
            log.info("trained production %s with %s", target, info["model"])
        return out

    # ── ensemble (weighted blend of a few models; see models.ensemble) ────
    def uses_ensemble(self, target: str, model_name: str | None) -> bool:
        if model_name == ENSEMBLE:
            return True
        return model_name in (None, "", "champion") and self.champion_name(target) == ENSEMBLE

    def member_path(self, target: str, model: str):
        return self.art.model_path(target, f"ensemble_{model}")

    def _ensemble_weights(self, target: str) -> dict[str, float]:
        spec = self.art.load_ensemble(target)
        if not spec or not spec.get("weights"):
            raise RuntimeError(f"{target}: no ensemble yet — run a backtest first")
        return {m: float(w) for m, w in spec["weights"].items()}

    def _train_ensemble(self, target: str) -> dict:
        weights = self._ensemble_weights(target)
        trained, skipped = [], []
        for m in weights:
            try:
                self._train_and_save(target, m, self.member_path(target, m))
                trained.append(m)
            except ImportError as exc:
                log.warning("ensemble %s: member %s unavailable (%s); it will be left out", target, m, exc)
                skipped.append(m)
        if not trained:
            raise RuntimeError(f"{target}: none of the ensemble's models could be trained")
        log.info("trained production %s with ensemble (%s)", target,
                 " · ".join(f"{m} {w:.2f}" for m, w in weights.items() if m in trained))
        return {"status": "ok", "model": ENSEMBLE, "members": trained, "skipped": skipped, "weights": weights}

    def _predict_ensemble(self, target: str, X: pd.DataFrame, retrain: bool) -> np.ndarray:
        weights = self._ensemble_weights(target)
        preds, used = [], []
        for m, w in weights.items():
            path = self.member_path(target, m)
            try:
                if retrain or not path.exists():
                    self._train_and_save(target, m, path)
                if self.isolate:
                    y = run_isolated(_predict_with_file, str(path), X, label=f"predict {target}/{m}", timeout=3600)
                else:
                    y = np.asarray(Forecaster.load(path).predict(X), dtype=float)
            except ImportError as exc:
                self.warnings.append(f"{target}: ensemble member {m} unavailable ({exc}); left out")
                continue
            preds.append(y)
            used.append(w)
        if not preds:
            raise RuntimeError(f"{target}: no ensemble member could predict")
        wts = np.array(used) / sum(used)
        return np.sum([w * y for w, y in zip(wts, preds)], axis=0)

    def get_model(self, target: str, model_name: str | None = None, retrain: bool = False) -> Forecaster:
        """Production (champion) model by default; any other model is trained on demand
        on the latest data and cached separately so it never replaces the champion."""
        if model_name in (None, "", "champion"):
            path = self.art.model_path(target, "production")
            if retrain or not path.exists():
                self.train_champions([target])
            return Forecaster.load(path)
        path = self.art.model_path(target, f"adhoc_{model_name}")
        if retrain or not path.exists():
            self._train_and_save(target, model_name, path)
        return Forecaster.load(path)

    # ── predict ──────────────────────────────────────────────────────────
    def predict_day(
        self,
        day: date | None = None,
        targets: list[str] | None = None,
        model_name: str | None = None,
        retrain: bool = False,
    ) -> pd.DataFrame:
        """Predict every half-hour of `day` (default tomorrow). Sets self.warnings for
        targets that were skipped (no data, or day too far past the last data)."""
        day = day or self.tomorrow()
        targets = targets or list(self.settings.targets)
        start, end = self.day_bounds(day)
        issued = pd.Timestamp(datetime.now(timezone.utc))
        self.warnings: list[str] = []
        ok_targets = []
        for target in targets:
            max_day = self.max_predictable_date(target)
            if max_day is None:
                self.warnings.append(f"{target}: no data yet — run ingest first")
            elif day > max_day:
                self.warnings.append(f"{target}: latest data is {self.last_data(target):%Y-%m-%d %H:%M} UTC, "
                                     f"so the furthest day that can be predicted is {max_day}")
            else:
                ok_targets.append(target)
        for w in self.warnings:
            log.warning("predict %s: %s", day, w)
        if not ok_targets:
            return pd.DataFrame()
        fs = FeatureBuilder(self.store, self.settings, self.region).build(start, end)
        frames = []
        for target in ok_targets:
            X, _ = fs.xy(target, start, end, dropna=False)
            try:
                if self.uses_ensemble(target, model_name):
                    used, y_pred = ENSEMBLE, self._predict_ensemble(target, X, retrain)
                elif self.isolate:
                    used, y_pred = run_isolated(_load_and_predict, self.settings.raw, self.region, target, model_name,
                                                retrain, X, label=f"predict {target}", timeout=3600)
                else:
                    model = self.get_model(target, model_name, retrain)
                    used, y_pred = model.name, model.predict(X)
            except ImportError as exc:
                self.warnings.append(f"{target}: model '{model_name}' isn't installed ({exc})")
                log.warning("predict %s: %s", day, self.warnings[-1])
                continue
            frame = pd.DataFrame({
                TS: X.index, "target": target, "model": used, "y_pred": y_pred,
                "delivery_date": day.isoformat(), "issued_at": issued, "feature_coverage": X.notna().mean(axis=1).to_numpy(),
            })
            frames.append(frame)
            log.info("predicted %s %s with %s: %d periods, mean %.2f (min %.2f, max %.2f), feature coverage %.0f%%",
                     target, day, used, len(frame), frame["y_pred"].mean(), frame["y_pred"].min(),
                     frame["y_pred"].max(), 100 * frame["feature_coverage"].mean())
        if not frames:
            return pd.DataFrame()
        preds = pd.concat(frames, ignore_index=True)
        self.store.write("predictions", self.region, preds, key_cols=(TS, "target", "model", "delivery_date"))
        return preds

    def latest_predictions(self, day: date, model_name: str | None = None) -> pd.DataFrame:
        """Stored predictions for a day: one model per target (the given one, else the most recently issued)."""
        start, end = self.day_bounds(day)
        p = self.store.read("predictions", self.region, start, end)
        if p.empty:
            return p
        p = p[p["delivery_date"] == day.isoformat()]
        if model_name and model_name != "champion":
            return p[p["model"] == model_name]
        latest = p.sort_values("issued_at").groupby("target", as_index=False)["model"].last()
        return p.merge(latest, on=["target", "model"])

    # ── recommendations ──────────────────────────────────────────────────
    def recommend(self, preds: pd.DataFrame | None = None, day: date | None = None, persist: bool = True) -> dict:
        day = day or self.tomorrow()
        start, end = self.day_bounds(day)
        if preds is None:
            preds = self.latest_predictions(day)
        top_n = int(self.settings.get("recommendations.top_n_periods", 6))
        products, windows = [], []
        for target, g in preds.groupby("target") if not preds.empty else []:
            if not target.startswith(PRICE_TARGET_PREFIX):
                continue
            col = self.settings.targets[target]["column"]
            src = self.settings.targets[target]["source"]
            hist = self.store.read(src, self.region, start - pd.Timedelta(days=9), start, columns=[col])
            recent_mean = float(hist[col].tail(48 * 7).mean()) if not hist.empty else np.nan
            g = g.set_index(TS).sort_index()
            pred_mean = float(g["y_pred"].mean())
            champ = self.art.load_champion(target) or {}
            mae = (champ.get("metrics") or {}).get("mae")
            change = (pred_mean - recent_mean) / abs(recent_mean) * 100 if recent_mean == recent_mean and recent_mean else np.nan
            products.append({
                "target": target, "model": g["model"].iloc[0], "pred_mean": pred_mean,
                "pred_min": float(g["y_pred"].min()), "pred_max": float(g["y_pred"].max()),
                "pred_spread": float(g["y_pred"].max() - g["y_pred"].min()),
                "recent_7d_mean": recent_mean, "expected_change_pct": change, "backtest_mae": mae,
                # signal strength: expected move measured in backtest-MAE units
                "signal_to_noise": abs(pred_mean - recent_mean) / mae if mae and recent_mean == recent_mean else None,
            })
            local = g.index.tz_convert(self.tz)
            for kind, rows in (("buy", g["y_pred"].nsmallest(top_n)), ("sell", g["y_pred"].nlargest(top_n))):
                for rank, (ts, val) in enumerate(rows.items(), 1):
                    windows.append({"target": target, "action": kind, "rank": rank, TS: ts,
                                    "local_time": local[g.index.get_loc(ts)].strftime("%H:%M"), "pred": float(val)})
        products.sort(key=lambda p: (p["expected_change_pct"] if p["expected_change_pct"] == p["expected_change_pct"] else 0))
        for p in products:
            chg = p["expected_change_pct"]
            p["view"] = "cheap vs last 7d (buy)" if chg < -2 else "rich vs last 7d (sell)" if chg > 2 else "neutral"
        result = {"delivery_date": day.isoformat(), "products": products, "windows": windows}
        if windows and persist:
            w = pd.DataFrame(windows)
            w["delivery_date"] = day.isoformat()
            self.store.write("recommendations", self.region, w, key_cols=(TS, "target", "action", "delivery_date"))
        return result



# ── run in a fresh process (compute.run_isolated) ──────────────────────────

def _fit_and_save(fs, raw_settings, region, target, model_name, train_end, path) -> dict:
    settings = Settings(raw_settings)
    art = Artifacts(settings, region)
    model, metrics = fit_model(fs, target, model_name, train_end, settings, resolve_params(art, target, model_name),
                               use_validation=False)
    model.save(path)
    return {"model": model_name, "val": metrics, "trained_to": pd.Timestamp(train_end).isoformat()}


def _predict_with_file(path, X):
    return np.asarray(Forecaster.load(path).predict(X), dtype=float)


def _load_and_predict(raw_settings, region, target, model_name, retrain, X):
    pr = Predictor(Settings(raw_settings), region=region)
    pr.settings.raw.setdefault("compute", {})["isolate_models"] = False  # already in our own process
    model = pr.get_model(target, model_name, retrain)
    return model.name, np.asarray(model.predict(X), dtype=float)
