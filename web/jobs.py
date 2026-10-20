"""One-at-a-time background job runner for the dashboard's control buttons.

Each job runs in its own child process (spawn), not a thread inside the web server:
ML libraries (sklearn/LightGBM/PyTorch) use OpenMP thread pools that can hang when
driven from a web server's worker threads, a crash can't take the dashboard down,
and a running job can be cancelled.
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import queue
import threading
import traceback
from datetime import date, datetime, timezone

log = logging.getLogger(__name__)

JOBS = ("ingest", "tune", "backtest", "train", "predict", "nightly", "strategy")
_CTX = mp.get_context("spawn")


class JobRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self.current: dict | None = None
        self.history: list[dict] = []
        self._proc = None
        self._seq = 0

    def _shutdown(self) -> None:
        """On server exit, stop a running job (and its worker processes) instead of waiting for it.
        Quick, and a second Ctrl+C while it waits just stops everything at once."""
        proc = self._proc
        if proc is None or not proc.is_alive():
            return
        print("Stopping the running job…", flush=True)
        try:
            _signal_job(proc, "TERM")  # the job's SIGTERM handler also stops its backtest workers
            proc.join(timeout=5)
        except KeyboardInterrupt:
            pass
        if proc.is_alive():
            _signal_job(proc, "KILL")
            proc.join(timeout=2)

    def busy(self) -> bool:
        return self.current is not None and self.current["status"] == "running"

    def start(self, job: str, args: dict | None = None, settings=None) -> dict:
        if job not in JOBS:
            raise ValueError(f"unknown job {job!r}")
        if settings is None:
            from settings import get_settings

            settings = get_settings()
        with self._lock:
            if self.busy():
                raise RuntimeError(f"job '{self.current['job']}' is already running")
            self._seq += 1
            state = {"id": self._seq, "job": job, "args": args or {}, "status": "running", "started": _now()}
            q = _CTX.Queue()
            proc = _CTX.Process(target=_child, args=(job, state["args"], settings.raw, str(settings.path or ""), q),
                                daemon=False, name=f"job-{job}")
            # Not a daemon: daemon processes can't start children, and parallel backtests need
            # worker processes. Instead the server stops a running job itself when it exits.
            proc.start()
            # (Re-)register after start: multiprocessing installs its own exit hook that waits for
            # child processes, and exit hooks run newest-first, so ours must be the newest.
            atexit.unregister(self._shutdown)
            atexit.register(self._shutdown)
            self.current, self._proc = state, proc
            threading.Thread(target=self._watch, args=(state, proc, q), daemon=True, name=f"watch-{job}").start()
            return dict(state)

    def cancel(self) -> dict:
        with self._lock:
            if not self.busy() or self._proc is None:
                raise RuntimeError("no job is running")
            _signal_job(self._proc, "TERM")
            self.current["cancel_requested"] = True
            return dict(self.current)

    def _watch(self, state: dict, proc, q) -> None:
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except queue.Empty:
                if not proc.is_alive():
                    if state.get("cancel_requested"):
                        state["status"], state["error"] = "cancelled", "cancelled by user"
                    else:
                        state["status"] = "error"
                        state["error"] = f"job process exited unexpectedly (exit code {proc.exitcode})"
                    break
                continue
            if kind == "progress":
                state["progress"] = payload
            elif kind == "ok":
                state["result"], state["status"] = payload, "ok"
                break
            elif kind == "error":
                state["status"], state["error"] = "error", payload
                break
        proc.join(timeout=5)
        state["finished"] = _now()
        self.history = [state, *self.history][:20]

    def snapshot(self) -> dict:
        return {"current": self.current, "history": self.history}


def _child(job: str, args: dict, raw_settings: dict, settings_path: str, q) -> None:
    """Entry point of the job process."""
    import os
    import signal
    from pathlib import Path

    from compute import configure_threads, kill_active_pools

    configure_threads()  # before sklearn/LightGBM/torch are imported

    # Own process group: Ctrl+C in the server's terminal goes to the server only (which then
    # stops this job cleanly) instead of interrupting the job and its workers half-way.
    # The server signals the whole group, so worker processes are stopped with it.
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def on_cancel(signum, frame):  # stop any parallel backtest workers too
        kill_active_pools()
        os._exit(1)

    signal.signal(signal.SIGTERM, on_cancel)

    from logging_setup import setup_logging
    from settings import Settings

    setup_logging("INFO", "jobs.log")
    settings = Settings(raw_settings, Path(settings_path) if settings_path else None)
    try:
        result = _dispatch(job, args, settings, lambda p: q.put(("progress", p)))
        q.put(("ok", result))
    except Exception as exc:
        logging.getLogger(__name__).exception("job %s failed", job)
        q.put(("error", f"{exc}\n{traceback.format_exc(limit=4)}"))


def _signal_job(proc, which: str) -> None:
    """SIGTERM/SIGKILL the job and its whole process group (its parallel workers too)."""
    import os
    import signal

    sig = signal.SIGTERM if which == "TERM" else signal.SIGKILL
    try:
        if hasattr(os, "killpg") and proc.pid and os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, sig)
            return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    (proc.terminate if which == "TERM" else proc.kill)()


def _dispatch(job: str, args: dict, settings, report=None):
    region = args.get("region", "GB")
    if job == "ingest":
        from sources import ingest_all

        return ingest_all(region, args.get("sources"), args.get("start"), args.get("end"), settings=settings,
                          progress=lambda d, t, label: report and report({"done": d, "total": t, "label": label}))
    if job == "tune":
        from pipeline import run_tuning
        from storage import ParquetStore
        from training import Artifacts

        return run_tuning(settings, ParquetStore(settings.data_dir), region, Artifacts(settings, region),
                          bool(args.get("force")))
    if job == "backtest":
        from datetime import timedelta

        from backtest import Backtester

        def progress(done: int, total: int, label: str) -> None:
            if report is not None:
                report({"done": done, "total": total, "label": label})

        # UI sends the last test day inclusive; the backtester's `end` is exclusive
        end = (date.fromisoformat(args["end"]) + timedelta(days=1)).isoformat() if args.get("end") else None
        res = Backtester(settings, region=region).run(
            args.get("targets") or None, args.get("models") or None, end=end,
            days=_int(args.get("days")), retrain_every=_int(args.get("retrain_every_days")),
            train_days=_int(args.get("train_days")), set_champions=args.get("set_champions", True),
            progress=progress,
        )
        errors = [f"{r['target']}/{r['model']}: {r['error']}" for r in res["leaderboard"] if r.get("error") == r.get("error") and r.get("error")]
        return {k: res[k] for k in ("run_id", "champions", "start", "end", "options")} | {"errors": errors}
    if job == "train":
        from prediction import Predictor

        return Predictor(settings, region=region).train_champions(args.get("targets"))
    if job == "predict":
        from prediction import Predictor

        warnings: list[str] = []
        if args.get("refresh", True):
            # Pull the latest prices/demand/grid data and tomorrow's weather forecast first.
            from sources import ingest_all

            def ingest_progress(done: int, total: int, label: str) -> None:
                if report is not None:
                    report({"done": done, "total": total + 1, "label": f"fetching latest data ({done}/{total} sources)"})

            for r in ingest_all(region, settings=settings, progress=ingest_progress):
                if r["status"] == "error":
                    warnings.append(f"ingest {r['source']} failed: {r['message']}")
        if report is not None:
            report({"done": 1, "total": 1, "label": "predicting"})
        pr = Predictor(settings, region=region)
        day = date.fromisoformat(args["date"]) if args.get("date") else pr.tomorrow()
        model = args.get("model") or "champion"
        preds = pr.predict_day(day, args.get("targets") or None, model, bool(args.get("retrain")))
        warnings += pr.warnings
        rec = pr.recommend(preds if len(preds) else None, day)
        summary = []
        if len(preds):
            for (target, mdl), g in preds.groupby(["target", "model"]):
                summary.append({"target": target, "model": mdl, "mean": float(g["y_pred"].mean()),
                                "min": float(g["y_pred"].min()), "max": float(g["y_pred"].max()),
                                "coverage": float(g["feature_coverage"].mean())})
        if not summary:
            raise RuntimeError("; ".join(warnings) or "nothing was predicted")
        return {"date": day.isoformat(), "model": model, "summary": summary, "warnings": warnings,
                "products": rec["products"]}
    if job == "strategy":
        from pipeline.nightly import run_strategy

        return run_strategy(settings, region, progress=lambda d, t, label: report and report(
            {"done": d, "total": t, "label": label}), backtest_days=_int(args.get("backtest_days")),
            horizon=_int(args.get("horizon")), threshold_k=args.get("threshold_k") or None,
            model=args.get("model") or None)
    if job == "nightly":
        from pipeline import run_nightly

        rec = run_nightly(settings, region, bool(args.get("skip_ingest")), bool(args.get("force_tune")),
                          steps=args.get("steps"), backtest_days=_int(args.get("backtest_days")),
                          tune_days=_int(args.get("tune_days")),
                          trigger=args.get("trigger", "manual"), progress=report)
        return {k: rec.get(k) for k in ("run_id", "status", "champions", "trigger")} | {
            "steps": [{k: s.get(k) for k in ("step", "status", "seconds", "error")} for s in rec["steps"]]}
    raise ValueError(job)


def _int(v):
    return int(v) if v not in (None, "") else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
