"""Walk-forward backtest: simulate the nightly process over past days.

For each test day D, the model is trained only on data ending before D-1 (what would
have been available when the day-ahead prediction was made), then predicts all of D.
Models are refit every ``retrain_every_days``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pandas as pd

from settings import Settings, _deep_merge, get_settings
from features import FeatureBuilder, FeatureSet
from storage import TS, ParquetStore
from training import Artifacts, fit_model, new_run_id, resolve_params
from compute import ACTIVE_POOLS, cpu_count, limit_threads, resolve_workers
from evaluation import daily_shape_metrics, regression_metrics
from models.ensemble import NAME as ENSEMBLE
from models.ensemble import build as build_ensemble
from models.ensemble import describe as describe_weights

log = logging.getLogger(__name__)


class Backtester:
    def __init__(self, settings: Settings | None = None, store: ParquetStore | None = None, region: str = "GB"):
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(self.settings.data_dir)
        self.region = region
        self.art = Artifacts(self.settings, region)
        cfg = self.settings.get("backtest", {}) or {}
        self.days = int(cfg.get("days", 60))
        self.retrain_every = int(cfg.get("retrain_every_days", 7))

    def _local_midnight(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert(self.settings.timezone).normalize().tz_convert("UTC")

    def run(
        self,
        targets: list[str] | None = None,
        models: list[str] | None = None,
        end: pd.Timestamp | str | None = None,
        days: int | None = None,
        fs: FeatureSet | None = None,
        run_id: str | None = None,
        retrain_every: int | None = None,
        train_days: int | None = None,
        set_champions: bool = True,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Walk-forward test over the `days` local days ending at `end` (exclusive;
        default = the last fully observed day). Per-run overrides leave config untouched."""
        run_id = run_id or new_run_id("backtest")
        targets = targets or list(self.settings.targets)
        models = models or list(self.settings.get("models.candidates", ["seasonal_naive", "hgb"]))
        days = int(days or self.days)
        retrain_every = max(1, int(retrain_every or self.retrain_every))
        train_days = int(train_days or self.settings.get("training.train_days", 540))
        run_settings = Settings(_deep_merge(self.settings.raw, {"training": {"train_days": train_days}}),
                                self.settings.path)
        options = {"targets": targets, "models": models, "days": days, "retrain_every_days": retrain_every,
                   "train_days": train_days, "set_champions": set_champions}

        last = max(
            (self.store.last_timestamp(self.settings.targets[t]["source"], self.region) for t in targets),
            default=None,
            key=lambda x: x or pd.Timestamp(0, tz="UTC"),
        )
        if last is None:
            raise RuntimeError("no target data in the store; run ingest (or demo) first")
        latest_end = self._local_midnight(last)
        if end is None:
            end_ts = latest_end
        else:
            e = pd.Timestamp(end)
            end_ts = self._local_midnight(e.tz_localize(self.settings.timezone) if e.tzinfo is None else e)
            end_ts = min(end_ts, latest_end)
        # Count back whole calendar days in UK time. (end - N*24h lands on 01:00 BST, not midnight,
        # once a clock change is inside the window, and 01:00 on the October change day is ambiguous.)
        tz = self.settings.timezone
        end_day = end_ts.tz_convert(tz).date()
        start_ts = pd.Timestamp(end_day - timedelta(days=days)).tz_localize(tz).tz_convert("UTC")
        first = min((self.store.first_timestamp(self.settings.targets[t]["source"], self.region) for t in targets),
                    default=None, key=lambda x: x or pd.Timestamp.max.tz_localize("UTC"))
        if first is not None and start_ts - pd.Timedelta(days=30) < first:
            raise RuntimeError(f"not enough history: data starts {first:%Y-%m-%d}, the test window starts "
                               f"{start_ts:%Y-%m-%d} and models need at least ~30 days of training data before it")
        if fs is None:
            fs = FeatureBuilder(self.store, run_settings, self.region).build(
                start_ts - pd.Timedelta(days=train_days + 2), end_ts
            )

        # one local midnight per day (midnight is never ambiguous in the UK; the change is at 01:00/02:00)
        test_days = pd.date_range(end_day - timedelta(days=days), end_day, freq="D", inclusive="left").tz_localize(tz)
        all_preds, leaderboard = [], []
        self._leaderboard = leaderboard  # shared with _ensembles()
        tasks = [(t, m) for t in targets for m in models]
        params = {(t, m): resolve_params(self.art, t, m) for t, m in tasks}
        total = len(tasks)
        workers = resolve_workers(self.settings.get("backtest.workers", "auto"), total)
        log.info("backtest: %d target/model runs over %d days on %d worker(s)", total, days, workers)

        def collect(target, model_name, outcome):
            kind, payload = outcome
            if kind == "missing":
                log.warning("skip %s: %s", model_name, payload)
                leaderboard.append({"target": target, "model": model_name, "error": f"not installed: {payload}"})
                return
            if kind == "error":
                log.error("backtest %s/%s failed: %s", target, model_name, payload)
                leaderboard.append({"target": target, "model": model_name, "error": payload})
                return
            preds = payload
            if preds.empty:
                leaderboard.append({"target": target, "model": model_name, "error": "no test data"})
                return
            m = regression_metrics(preds["y_true"], preds["y_pred"])
            m.update(daily_shape_metrics(preds.set_index(TS)[["y_true", "y_pred"]],
                                         int(self.settings.get("recommendations.top_n_periods", 6)),
                                         self.settings.timezone))
            leaderboard.append({"target": target, "model": model_name, **m})
            preds["target"], preds["model"], preds["column"] = target, model_name, fs.target_cols[target]
            all_preds.append(preds)
            log.info("backtest %s/%s: MAE %.3f  sMAPE %.2f%%", target, model_name, m.get("mae", float("nan")), m.get("smape", float("nan")))

        # One process per walk (even with a single worker) unless compute.isolate_models is off:
        # LightGBM and PyTorch can deadlock when both are loaded into one process.
        if workers <= 1 and not self.settings.get("compute.isolate_models", True):
            for done, (target, model_name) in enumerate(tasks):
                if progress:
                    progress(done, total, f"{target} · {model_name}")
                collect(target, model_name, _run_walk(fs, target, model_name, test_days, run_settings,
                                                      retrain_every, params[(target, model_name)]))
        else:
            # Each (target, model) walk runs in a fresh worker process (one walk per process, so
            # native libraries such as LightGBM's and PyTorch's OpenMP never share a process).
            # Cores are split between workers so they don't oversubscribe.
            threads = max(1, cpu_count() // workers)
            args = {(t, m): (t, m, test_days, run_settings.raw, retrain_every, params[(t, m)]) for t, m in tasks}
            done = 0

            def on_done(key, outcome):
                nonlocal done
                done += 1
                collect(*key, outcome)
                if progress:
                    progress(done, total, f"finished {key[0]} · {key[1]}")

            if progress:
                progress(0, total, f"running {workers} in parallel")
            crashed = _run_pool(list(tasks), args, workers, fs, threads, on_done)
            if crashed:
                # A worker died (a native crash, e.g. an OpenMP clash). That breaks the whole pool,
                # so every unfinished walk failed with it. Re-run those walks one per process, so
                # only the walk that really crashes is lost.
                log.warning("a backtest worker process crashed; re-running %d walk(s) one per process: %s",
                            len(crashed), ", ".join(f"{t}/{m}" for t, m in crashed))
                for key in _run_isolated(crashed, args, workers, fs, threads, on_done):
                    on_done(key, ("error", "the worker process crashed (native library fault, e.g. OpenMP in "
                                           "LightGBM/PyTorch); see the server console for details"))
        if progress:
            progress(total, total, "saving")

        board = pd.DataFrame(leaderboard)
        now = pd.Timestamp(datetime.now(timezone.utc))
        ens = self._ensembles(board, all_preds, targets, save=set_champions) if all_preds else {}
        if ens:
            board = pd.DataFrame(leaderboard)  # _ensembles added rows / holdout scores

        champions = {}
        if set_champions and "mae" in board:
            ok = board[(board["model"] != ENSEMBLE)].dropna(subset=["mae"])
            for target, g in ok.groupby("target"):
                best = g.sort_values("mae").iloc[0]
                spec = ens.get(target)
                if spec and spec["beats_best_single"]:
                    row = board[(board["target"] == target) & (board["model"] == ENSEMBLE)].iloc[0]
                    champions[target] = ENSEMBLE
                    self.art.save_champion(target, ENSEMBLE, {**row.dropna().to_dict(), "weights": spec["weights"]},
                                           run_id)
                    log.info("champion %s: ensemble (%s), holdout MAE %.3f vs %s %.3f", target,
                             describe_weights(spec["weights"]), spec["holdout_mae"], spec["best_single"],
                             spec["best_single_holdout_mae"])
                else:
                    champions[target] = best["model"]
                    self.art.save_champion(target, best["model"], best.dropna().to_dict(), run_id)
        if not board.empty:
            board["champion"] = [champions.get(t) == m for t, m in zip(board["target"], board["model"])]

        if all_preds:
            p = pd.concat(all_preds, ignore_index=True)
            p["run_id"] = run_id
            self.store.write("backtest_predictions", self.region, p, key_cols=(TS, "run_id", "target", "model"))
        if not board.empty:
            b = board.copy()
            b[TS], b["run_id"] = now, run_id
            b["start"], b["end"] = start_ts.isoformat(), end_ts.isoformat()
            b["days"], b["retrain_every_days"], b["train_days"] = days, retrain_every, train_days
            self.store.write("backtest_metrics", self.region, b, key_cols=(TS, "run_id", "target", "model"))

        result = {
            "run_id": run_id,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "options": options,
            "leaderboard": board.to_dict(orient="records"),
            "champions": champions,
        }
        if not run_id.endswith("_nightly"):
            self.art.save_run({"run_id": run_id, "kind": "backtest", "region": self.region, "status": "ok",
                               "started": now.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
                               "options": options, "champions": champions,
                               "steps": [{"step": "backtest", "status": "ok"}]})
        return result


    def _ensembles(self, board: pd.DataFrame, all_preds: list, targets: list[str], save: bool = True) -> dict:
        """Blend the best few models per target (see models.ensemble). Appends an 'ensemble' row
        to the leaderboard and its holdout predictions to all_preds, and gives every model a
        `mae_holdout` on the same second-half days the blend is judged on."""
        cfg = self.settings.get("ensemble", {}) or {}
        if not cfg.get("enabled", True) or board.empty or "mae" not in board:
            return {}
        preds = pd.concat(all_preds, ignore_index=True)
        out = {}
        for target in targets:
            spec = build_ensemble(preds[preds["target"] == target], board, target,
                                  int(cfg.get("max_members", 3)), float(cfg.get("min_weight", 0.05)))
            if spec is None:
                continue
            hold = spec.pop("holdout_preds")
            hold_ts = set(hold[TS])
            col = preds.loc[preds["target"] == target, "column"].iloc[0]
            for row in self._rows(target):  # every single model, scored on exactly the blend's holdout periods
                g = preds[(preds["target"] == target) & (preds["model"] == row["model"]) & preds[TS].isin(hold_ts)]
                if len(g):
                    row["mae_holdout"] = float((g["y_pred"] - g["y_true"]).abs().mean())
            m = regression_metrics(hold["y_true"], hold["y_pred"])
            m.update(daily_shape_metrics(hold.set_index(TS)[["y_true", "y_pred"]],
                                         int(self.settings.get("recommendations.top_n_periods", 6)),
                                         self.settings.timezone))
            verdict = ("beats" if spec["beats_best_single"] else "doesn't beat")
            self._leaderboard.append({
                "target": target, "model": ENSEMBLE, **m, "mae_holdout": spec["holdout_mae"],
                "note": f"{describe_weights(spec['weights'])} · scored on 2nd half only; {verdict} "
                        f"{spec['best_single']} ({spec['best_single_holdout_mae']:.3f})"})
            hold = hold.assign(target=target, model=ENSEMBLE, column=col)
            all_preds.append(hold)
            if save:  # only runs that set champions change the production blend
                self.art.save_ensemble(target, spec)
            out[target] = spec
            log.info("ensemble %s: %s → holdout MAE %.3f (best single %s %.3f)", target,
                     describe_weights(spec["weights"]), spec["holdout_mae"], spec["best_single"],
                     spec["best_single_holdout_mae"])
        return out

    def _rows(self, target: str):
        return [r for r in self._leaderboard if r["target"] == target and r.get("model") != ENSEMBLE]


def _walk(fs: FeatureSet, target: str, model_name: str, test_days, settings: Settings,
          retrain_every: int, params: dict) -> pd.DataFrame:
    model, out = None, []
    for i, day in enumerate(test_days):
        day_utc = day.tz_convert("UTC")
        # next local midnight (calendar arithmetic, so clock-change days keep all 46/50 periods)
        nxt = (day.tz_localize(None).normalize() + pd.Timedelta(days=1)).tz_localize(day.tz).tz_convert("UTC")
        if model is None or i % retrain_every == 0:
            cutoff = day_utc - pd.Timedelta(days=1)  # info set at D-1 morning, conservatively
            model, _ = fit_model(fs, target, model_name, cutoff, settings, params)
        X, y = fs.xy(target, day_utc, nxt)
        if X.empty:
            continue
        out.append(pd.DataFrame({TS: X.index, "y_true": y.to_numpy(), "y_pred": model.predict(X)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _run_walk(fs, target, model_name, test_days, settings, retrain_every, params):
    """Returns ("ok", preds) | ("missing", reason) | ("error", message) — never raises."""
    try:
        return "ok", _walk(fs, target, model_name, test_days, settings, retrain_every, params)
    except ImportError as exc:
        return "missing", str(exc)
    except Exception as exc:
        log.exception("backtest %s/%s failed", target, model_name)
        return "error", str(exc)


# ── process pools ──────────────────────────────────────────────────────────

def _new_pool(workers: int, fs: FeatureSet, threads: int):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_init_worker, initargs=(fs, threads), max_tasks_per_child=1)
    ACTIVE_POOLS.append(pool)
    return pool


def _close_pool(pool) -> None:
    pool.shutdown(wait=True, cancel_futures=True)
    if pool in ACTIVE_POOLS:
        ACTIVE_POOLS.remove(pool)


def _run_pool(keys, args, workers, fs, threads, on_done) -> list:
    """Run walks in one shared pool. Returns the keys lost to a crashed (broken) pool."""
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool

    pool = _new_pool(workers, fs, threads)
    crashed = []
    try:
        futs = {pool.submit(_worker_walk, *args[k]): k for k in keys}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                outcome = fut.result()
            except BrokenProcessPool:
                crashed.append(key)
                continue
            except Exception as exc:
                outcome = ("error", str(exc))
            on_done(key, outcome)
    finally:
        _close_pool(pool)
    return crashed


def _run_isolated(keys, args, workers, fs, threads, on_done) -> list:
    """Run each walk in its own single-process pool, `workers` at a time. Returns keys that crashed."""
    from concurrent.futures import FIRST_COMPLETED, wait
    from concurrent.futures.process import BrokenProcessPool

    queue, running, crashed = list(keys), {}, []
    try:
        while queue or running:
            while queue and len(running) < workers:
                key = queue.pop(0)
                pool = _new_pool(1, fs, threads)
                running[pool.submit(_worker_walk, *args[key])] = (key, pool)
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                key, pool = running.pop(fut)
                try:
                    on_done(key, fut.result())
                except BrokenProcessPool:
                    crashed.append(key)
                except Exception as exc:
                    on_done(key, ("error", str(exc)))
                _close_pool(pool)
    finally:
        for _, pool in running.values():
            _close_pool(pool)
    return crashed


# ── worker-process side ────────────────────────────────────────────────────
_WORKER_FS: FeatureSet | None = None


def _init_worker(fs: FeatureSet, threads: int) -> None:
    global _WORKER_FS
    limit_threads(threads)
    _WORKER_FS = fs


def _worker_walk(target, model_name, test_days, raw_settings, retrain_every, params):
    import os

    if os.environ.get("ENERGY_TEST_CRASH") == f"{target}/{model_name}":  # tests: simulate a native crash
        os._exit(3)
    return _run_walk(_WORKER_FS, target, model_name, test_days, Settings(raw_settings), retrain_every, params)
