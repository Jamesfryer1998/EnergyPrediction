"""Fit a model for one target on a rolling window, and Optuna tuning."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from evaluation import regression_metrics
from settings import Settings, get_settings
from features import FeatureSet
from models import Forecaster, create_model
from .artifacts import Artifacts

log = logging.getLogger(__name__)


def resolve_params(art: Artifacts, target: str, model: str) -> dict:
    tuned = art.load_params(target, model)
    return dict(tuned["params"]) if tuned else {}


def split_window(fs: FeatureSet, target: str, train_end: pd.Timestamp, train_days: int, val_days: int):
    """Train on [train_end - train_days, train_end - val_days), validate on the last val_days."""
    train_end = pd.Timestamp(train_end)
    val_start = train_end - pd.Timedelta(days=val_days)
    tr_start = train_end - pd.Timedelta(days=train_days)
    X_tr, y_tr = fs.xy(target, tr_start, val_start)
    X_val, y_val = fs.xy(target, val_start, train_end)
    return X_tr, y_tr, X_val, y_val


def fit_model(
    fs: FeatureSet,
    target: str,
    model_name: str,
    train_end: pd.Timestamp,
    settings: Settings | None = None,
    params: dict | None = None,
    use_validation: bool = True,
) -> tuple[Forecaster, dict]:
    settings = settings or get_settings()
    tcfg = settings.get("training", {}) or {}
    train_days, val_days = int(tcfg.get("train_days", 540)), int(tcfg.get("validation_days", 28))
    X_tr, y_tr, X_val, y_val = split_window(fs, target, train_end, train_days, val_days)
    if len(X_tr) < 48 * 14:
        raise ValueError(f"{target}: only {len(X_tr)} training rows before {train_end}")
    model = create_model(model_name, **(params or {}))
    if not use_validation and model.self_validating and len(X_val):
        # Production fit: one pass on all data; the model early-stops on its own hold-out.
        model.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))
        return model, {}
    model.fit(X_tr, y_tr, X_val, y_val)
    metrics = regression_metrics(y_val, model.predict(X_val)) if len(X_val) else {}
    if not use_validation and len(X_val):
        # refit on train+val now that early-stopping told us the right size
        X_all, y_all = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
        model = create_model(model_name, **{**(params or {}), **model.refit_params()})
        model.fit(X_all, y_all)
    return model, metrics


def tuning_due(art: Artifacts, target: str, model: str, retune_every_days: int) -> bool:
    if retune_every_days <= 0:  # 0 = tune on every run
        return True
    tuned = art.load_params(target, model)
    if not tuned:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(tuned["tuned_at"])
    return age.days >= retune_every_days


def tune(
    fs: FeatureSet,
    target: str,
    model_name: str,
    train_end: pd.Timestamp,
    art: Artifacts,
    settings: Settings | None = None,
    n_trials: int | None = None,
    timeout_minutes: float | None = None,
    progress=None,
    train_days: int | None = None,
) -> dict:
    """Optuna search on the validation window; stores best params for the nightly run."""
    import optuna

    settings = settings or get_settings()
    cfg = settings.get("tuning", {}) or {}
    n_trials = int(n_trials or cfg.get("n_trials", 40))
    timeout = float(timeout_minutes or cfg.get("timeout_minutes", 45)) * 60
    model_cls = create_model(model_name).__class__
    if not model_cls.tunable:
        return {"status": "skipped", "reason": "not tunable"}

    tcfg = settings.get("training", {}) or {}
    # Trials train on a shorter recent window than production (tuning.train_days) so many more
    # of them fit in the time budget; hyper-parameters transfer well to the longer window.
    tune_days = int(train_days or cfg.get("train_days") or tcfg.get("train_days", 540))
    X_tr, y_tr, X_val, y_val = split_window(fs, target, train_end, tune_days, int(tcfg.get("validation_days", 28)))
    key = f"{target}/{model_name}"
    log.info("tuning %s: up to %d trials or %.0f min, %d training rows (%d days), %d validation rows",
             key, n_trials, timeout / 60, len(X_tr), tune_days, len(X_val))

    def objective(trial):
        params = model_cls.search_space(trial)
        t = time.time()
        try:
            m = create_model(model_name, **params)
            m.fit(X_tr, y_tr, X_val, y_val)
            mae = regression_metrics(y_val, m.predict(X_val))["mae"]
        except Exception as exc:
            log.warning("  %s trial %d failed after %.0fs: %s", key, trial.number, time.time() - t, exc)
            raise
        trial.set_user_attr("seconds", round(time.time() - t, 1))
        return mae

    def report(study, trial):
        if trial.value is None:
            return
        n = sum(1 for t in study.trials if t.datetime_start and t.datetime_start.timestamp() >= t0)
        best = min((t.value for t in study.trials if t.value is not None and t.datetime_start
                    and t.datetime_start.timestamp() >= t0), default=trial.value)
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in trial.params.items()}
        log.info("  %s trial %d/%d: val MAE %.3f (best %.3f) in %.0fs %s", key, n, n_trials, trial.value, best,
                 trial.user_attrs.get("seconds", 0), shown)
        if progress:
            progress(n, n_trials, f"{key} trial {n}: best MAE {best:.3f}")

    # One Optuna study per target/model carries on across nights, so the TPE sampler learns from
    # every earlier trial and proposes values near the ones that worked. Every
    # `tuning.reset_every_days` a fresh study starts (seeded with the current best), so trials scored
    # on old data stop steering the search.
    reset_days = int(cfg.get("reset_every_days", 7))
    study_name, fresh, age = art.study_generation(target, model_name, reset_days)
    study = optuna.create_study(
        study_name=study_name,
        storage=art.optuna_storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(time.time()) % 10_000),
    )
    earlier = [t for t in study.trials if t.value is not None]
    warm_after = int(cfg.get("warm_after", 20))
    if len(earlier) >= warm_after:
        # the sampler already knows the good region: fewer trials are needed
        n_trials = min(n_trials, int(cfg.get("warm_trials", 15)))
        log.info("tuning %s: study has %d earlier trials (%d days old); running %d trials around them",
                 key, len(earlier), age, n_trials)
    elif fresh:
        log.info("tuning %s: fresh study %s (reset every %d days); %d trials", key, study_name, reset_days, n_trials)
    # Re-score the current best and the study's top few on today's data first
    seeds = []
    current = art.load_params(target, model_name)
    if current:
        seeds.append(current["params"])
    for t in sorted(earlier, key=lambda t: t.value)[:3]:
        seeds.append(t.params)
    for params in seeds:
        try:
            study.enqueue_trial(params, skip_if_exists=True)
        except Exception:  # search space changed since
            log.debug("could not enqueue earlier params", exc_info=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,), callbacks=[report])
    # The study spans many nights; take the best trial from *this* session (today's data)
    session = [t for t in study.trials if t.datetime_start and t.datetime_start.timestamp() >= t0
               and t.value is not None]
    if not session:
        raise RuntimeError(f"no {key} trial finished within {timeout / 60:.0f} min (see warnings above)")
    best = min(session, key=lambda t: t.value)
    # Replay the space with the winning values so fixed (non-sampled) entries are kept too
    params = model_cls.search_space(optuna.trial.FixedTrial(best.params))
    art.save_params(target, model_name, params, best.value, len(session))
    log.info("tuned %s: val MAE %.3f over %d trials in %.1f min", key, best.value, len(session), (time.time() - t0) / 60)
    return {"status": "ok", "val_mae": best.value, "trials": len(session), "params": params}
