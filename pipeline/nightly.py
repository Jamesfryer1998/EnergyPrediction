"""Nightly orchestration: ingest -> tune (when due) -> backtest/compare -> train -> predict -> rank.

Run by the systemd timer (deploy/systemd/energy-prediction-nightly.timer) or
`python main.py nightly`. Each step is isolated so one failure is recorded
and the remaining steps still run where possible.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from backtest import Backtester
from settings import Settings, get_settings
from sources import ingest_all
from features import FeatureBuilder
from models import available_models
from prediction import Predictor
from storage import ParquetStore
from training import Artifacts, new_run_id, tune, tuning_due
from compute import run_isolated
from .runlog import capture

log = logging.getLogger(__name__)


class StepRecorder:
    def __init__(self, record: dict, save=None, logs=None):
        self.record = record
        self.save = save  # persists the record after each step, so a running pipeline shows progress
        self.logs = logs  # StepLogHandler: tags log lines with the running step

    def run(self, name: str, fn, *args, **kwargs):
        t0 = time.time()
        step = {"step": name, "started": _now(), "status": "running"}
        self.record["steps"].append(step)
        if self.logs is not None:
            self.logs.step = name
        self._save()
        try:
            return self._run(step, t0, fn, *args, **kwargs)
        finally:
            self._save()

    def _save(self) -> None:
        if self.save is not None:
            try:
                self.save({**self.record, "status": "running"})
            except Exception:
                log.exception("could not save run record")

    def _run(self, step: dict, t0: float, fn, *args, **kwargs):
        name = step["step"]
        try:
            step["result"] = fn(*args, **kwargs)
            step["status"] = "ok"
        except Exception as exc:
            log.exception("step %s failed", name)
            step["status"], step["error"] = "error", f"{exc}\n{traceback.format_exc(limit=5)}"
            step["result"] = None
        step["seconds"] = round(time.time() - t0, 1)
        return step["result"]


def run_tuning(settings: Settings, store: ParquetStore, region: str, art: Artifacts, force: bool = False,
               progress=None, tune_days: int | None = None) -> dict:
    """Optuna-tune every target x model. Each pair gets up to `tuning.timeout_minutes`, and the
    whole step at most `tuning.max_minutes` (shared out between the pairs still to do)."""
    cfg = settings.get("tuning", {}) or {}
    if not cfg.get("enabled", True) and not force:
        log.info("tuning is turned off (tuning.enabled)")
        return {"status": "disabled"}
    wanted = list(cfg.get("models", []))
    models = [m for m in wanted if m in available_models()]
    for m in set(wanted) - set(models):
        log.warning("tuning: unknown model %s skipped", m)
    every = int(cfg.get("retune_every_days", 0))
    pair_minutes = float(cfg.get("timeout_minutes", 15))
    max_minutes = float(cfg.get("max_minutes", 0) or 0)  # 0 = no overall cap
    tune_days = int(tune_days or cfg.get("train_days") or settings.get("training.train_days", 540))
    isolate = bool(settings.get("compute.isolate_models", True))
    out = {}
    pairs = []
    for target, tcfg in settings.targets.items():
        last = store.last_timestamp(tcfg["source"], region)
        if last is None:
            log.warning("tuning: no %s data yet, %s skipped", tcfg["source"], target)
            continue
        for model in models:
            key = f"{target}/{model}"
            if not force and not tuning_due(art, target, model, every):
                tuned = art.load_params(target, model) or {}
                log.info("tuning %s skipped: tuned %s, re-tuned every %d days", key, str(tuned.get("tuned_at"))[:16], every)
                out[key] = {"status": "fresh"}
                continue
            pairs.append((target, model, last + pd.Timedelta(minutes=30)))
    log.info("tuning %d target/model pair(s): %d trials or %.0f min each%s", len(pairs),
             int(cfg.get("n_trials", 40)), pair_minutes, f", {max_minutes:.0f} min in total" if max_minutes else "")
    t_start = time.time()
    fs_cache = None
    for i, (target, model, train_end) in enumerate(pairs):
        key = f"{target}/{model}"
        minutes = pair_minutes
        if max_minutes:
            left = max_minutes - (time.time() - t_start) / 60
            if left < 1:
                log.warning("tuning time budget (%.0f min) used up; %s and %d more pair(s) not tuned this run",
                            max_minutes, key, len(pairs) - i - 1)
                for t, m, _ in pairs[i:]:
                    out[f"{t}/{m}"] = {"status": "out_of_time"}
                break
            minutes = min(pair_minutes, left / (len(pairs) - i))
        if fs_cache is None:
            fs_cache = FeatureBuilder(store, settings, region).build(
                train_end - pd.Timedelta(days=tune_days + 1), train_end)

        def sub(d, t, label, i=i):
            if progress:
                progress(i, len(pairs), label)

        try:
            if isolate:
                # fresh process per pair (see compute.run_isolated); the hard limit stops a stuck
                # trial, which Optuna's own timeout can't (it only checks between trials)
                out[key] = run_isolated(_tune_pair, fs_cache, target, model, train_end, settings.raw, region, minutes,
                                        tune_days, progress=sub, label=f"tuning {key}",
                                        timeout=minutes * 60 * 1.5 + 300)
            else:
                out[key] = tune(fs_cache, target, model, train_end, art, settings, timeout_minutes=minutes,
                                progress=sub, train_days=tune_days)
        except ImportError as exc:
            log.warning("tuning %s skipped: %s", key, exc)
            out[key] = {"status": "unavailable", "reason": str(exc)}
        except Exception as exc:
            log.exception("tuning %s failed", key)
            out[key] = {"status": "error", "reason": str(exc)}
    return out


def _tune_pair(fs, target, model, train_end, raw_settings, region, minutes, tune_days, progress=None):
    settings = Settings(raw_settings)
    return tune(fs, target, model, train_end, Artifacts(settings, region), settings, timeout_minutes=minutes,
                progress=progress, train_days=tune_days)


PIPELINE_STEPS = ("ingest", "tune", "backtest", "train", "predict", "strategy")


def run_strategy(settings: Settings, region: str = "GB", progress=None, **overrides) -> dict:
    """Paper-trading strategy: walk-forward backtest, then refit the live model on the latest data.
    Skipped (not an error) when there's no IG price history, e.g. no IG demo login yet."""
    from sources import build_source
    from trading.ig import ig_configured
    from trading.strategy import load_candles

    store = ParquetStore(settings.data_dir)
    fetch_note = None
    if ig_configured():  # bring the IG price history up to date first (a few seconds when incremental)
        res = build_source("ig_gas", region, store=store, settings=settings).update()
        log.info("IG prices: %s, %d rows%s", res.status, res.rows, f" ({res.message})" if res.message else "")
        if res.status not in ("ok", "skipped"):
            fetch_note = res.message
    if load_candles(store, region).empty:
        reason = (f"no IG price history: {fetch_note}" if fetch_note else
                  "no IG price history" if ig_configured() else
                  "no IG demo login in .env (IG_API_KEY, IG_USERNAME, IG_PASSWORD)")
        log.warning("strategy skipped: %s", reason)
        return {"status": "skipped", "reason": reason}
    return run_isolated(_strategy_entry, settings.raw, region, overrides, progress=progress, label="strategy")


def _strategy_entry(raw_settings, region, overrides, progress=None):
    from trading import strategy

    settings = Settings(raw_settings)
    store = ParquetStore(settings.data_dir)
    bt = strategy.backtest(settings, store, region, progress=progress, **overrides)
    meta = strategy.train(settings, store, region)
    return {"status": "ok", "backtest": bt, "model": meta}


def run_nightly(
    settings: Settings | None = None,
    region: str = "GB",
    skip_ingest: bool = False,
    force_tune: bool = False,
    steps: list[str] | None = None,
    backtest_days: int | None = None,
    tune_days: int | None = None,
    trigger: str = "manual",
    progress=None,
) -> dict:
    """Full daily pipeline. Order matters: the backtest picks each target's champion model,
    training then refits that champion on all the latest data, and predict uses it."""
    settings = settings or get_settings()
    store = ParquetStore(settings.data_dir)
    art = Artifacts(settings, region)
    wanted = [s for s in PIPELINE_STEPS if s in (steps or PIPELINE_STEPS)]
    if skip_ingest and "ingest" in wanted:
        wanted.remove("ingest")
    run_id = new_run_id("nightly")
    record = {"run_id": run_id, "kind": "nightly", "trigger": trigger, "region": region, "started": _now(),
              "planned": wanted, "steps": []}
    with capture(settings.data_dir, run_id) as logs:
        rec = StepRecorder(record, art.save_run, logs)
        art.save_run({**record, "status": "running"})
        log.info("nightly %s started (%s): %s", run_id, trigger, ", ".join(wanted))

        def report(i: int, step: str, detail: str = "") -> None:
            if progress:
                progress({"done": i, "total": len(wanted), "label": step + (f": {detail}" if detail else ""),
                          "step": step, "run_id": run_id})

        def sub(i: int, step: str):  # progress callback for a step's own sub-progress
            return lambda d, t, label: report(i, step, f"{label} ({d}/{t})")

        bt = None
        predictor = Predictor(settings, store, region)
        for i, step in enumerate(wanted):
            report(i, step)
            if step == "ingest":
                rec.run("ingest", ingest_all, region, settings=settings, store=store, progress=sub(i, step))
            elif step == "tune":
                rec.run("tune", run_tuning, settings, store, region, art, force_tune, progress=sub(i, step),
                        tune_days=tune_days)
            elif step == "backtest":
                bt = rec.run("backtest", Backtester(settings, store, region).run, run_id=run_id,
                             days=backtest_days, progress=sub(i, step))
            elif step == "train":
                rec.run("train", predictor.train_champions)
            elif step == "strategy":
                rec.run("strategy", run_strategy, settings, region, progress=sub(i, step))
            elif step == "predict":
                preds = rec.run("predict", predictor.predict_day)
                rec.run("recommend", predictor.recommend, preds if preds is not None and len(preds) else None)
        report(len(wanted), "done")

        record["finished"] = _now()
        record["status"] = "ok" if all(s["status"] == "ok" for s in record["steps"]) else "partial"
        record["champions"] = (bt or {}).get("champions", {})
        art.save_run(record)
        logs.step = None
    log.info("nightly %s finished: %s", run_id, record["status"])
    return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
