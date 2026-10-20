"""CPU/thread helpers shared by the CLI, web jobs and parallel backtests."""
from __future__ import annotations

import os
import subprocess
import sys

ACTIVE_POOLS: list = []  # process pools to tear down if a job is cancelled
ACTIVE_PROCS: list = []  # single isolated processes (run_isolated), same reason


def cpu_count() -> int:
    """Cores worth giving to number crunching. On Apple Silicon that's the performance cores
    only: spreading OpenMP work onto the slower efficiency cores makes every step wait for them."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except Exception:
            pass
    return os.cpu_count() or 1


def configure_threads() -> None:
    """Call before importing sklearn / LightGBM / torch.

    Idle OpenMP threads normally busy-spin waiting for the next parallel step. That's fastest
    on a quiet Linux box, but on macOS (and on 1-2 core machines) it has been measured to make
    training many times slower, so there idle threads sleep instead and the thread count is
    capped to the performance cores. Anything already set in the environment wins.
    """
    if sys.platform == "darwin" or cpu_count() <= 2:
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count()))
        # LightGBM (conda's llvm-openmp) and PyTorch each ship a libomp. When both load into one
        # process, e.g. training champions of both kinds, Intel/LLVM OpenMP aborts the whole
        # process ("OMP: Error #15") unless told the duplicate is OK.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def limit_threads(n: int) -> None:
    """Cap every native thread pool in this process to n threads (worker processes)."""
    n = max(1, int(n))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:
        pass


def resolve_workers(setting, tasks: int) -> int:
    """'auto' -> about one worker per 2 cores (each worker still multi-threads), capped by tasks."""
    if setting in (None, "", "auto"):
        n = max(1, cpu_count() // 2)
    else:
        n = max(1, int(setting))
    return max(1, min(n, tasks))


def kill_active_pools() -> None:
    for pool in list(ACTIVE_POOLS):
        for proc in list(getattr(pool, "_processes", {}).values()):
            try:
                proc.terminate()
            except Exception:
                pass
    for proc in list(ACTIVE_PROCS):
        try:
            proc.terminate()
        except Exception:
            pass


# ── run one function in a fresh process ─────────────────────────────────────


class IsolatedError(RuntimeError):
    """The function raised in the child process (message and traceback carried over)."""


def run_isolated(fn, *args, timeout: float | None = None, label: str = "", progress=None, **kwargs):
    """Run fn(*args, **kwargs) in a fresh spawned process and return its result.

    Why: LightGBM and PyTorch each bring their own OpenMP runtime. Loaded into one process
    (e.g. tuning LightGBM, then PyTorch) they can deadlock, and a native crash would take the
    whole job down. A fresh process per model avoids both. The child's log records are
    re-emitted here (so they reach this process's log files and step logs), `progress` calls in
    the child are relayed, and `timeout` (seconds) kills a child that hangs.
    `fn` must be a module-level function; if it accepts `progress`, pass progress=... here.
    """
    import logging
    import multiprocessing as mp
    import queue as _queue
    import time

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_isolated_entry, args=(fn, args, kwargs, q, progress is not None),
                       daemon=False, name=f"isolated-{label or getattr(fn, '__name__', 'fn')}")
    proc.start()
    ACTIVE_PROCS.append(proc)
    t0 = time.time()
    try:
        while True:
            if timeout and time.time() - t0 > timeout:
                proc.kill()
                raise TimeoutError(f"{label or fn.__name__} did not finish within {timeout / 60:.0f} min "
                                   "(stopped; it was most likely stuck)")
            try:
                kind, payload = q.get(timeout=1.0)
            except _queue.Empty:
                if not proc.is_alive():
                    # drain anything sent just before exit
                    try:
                        kind, payload = q.get(timeout=1.0)
                    except _queue.Empty:
                        raise IsolatedError(f"{label or fn.__name__}: process crashed "
                                            f"(exit code {proc.exitcode})") from None
                else:
                    continue
            if kind == "log":
                logging.getLogger(payload.name).handle(payload)
            elif kind == "progress":
                if progress:
                    progress(*payload)
            elif kind == "ok":
                return payload
            elif kind == "error":
                exc_type, msg, tb = payload
                if exc_type in ("ImportError", "ModuleNotFoundError"):
                    raise ImportError(msg)
                raise IsolatedError(f"{msg}\n{tb}" if tb else msg)
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        if proc in ACTIVE_PROCS:
            ACTIVE_PROCS.remove(proc)


class _QueueLogHandler:
    """Minimal logging handler that ships records to the parent (message pre-formatted)."""

    def __new__(cls, q):
        import logging

        class H(logging.Handler):
            def emit(self, record):
                try:
                    if record.exc_info:
                        record.exc_text = logging.Formatter().formatException(record.exc_info)
                    record.msg, record.args, record.exc_info = record.getMessage(), None, None
                    if record.exc_text:
                        record.msg += "\n" + record.exc_text
                        record.exc_text = None
                    q.put(("log", record))
                except Exception:
                    self.handleError(record)

        return H(logging.INFO)


def _isolated_entry(fn, args, kwargs, q, with_progress) -> None:
    import logging
    import traceback

    configure_threads()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_QueueLogHandler(q))
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "matplotlib", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if with_progress:
        kwargs = {**kwargs, "progress": lambda *a: q.put(("progress", a))}
    try:
        q.put(("ok", fn(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        q.put(("error", (type(exc).__name__, str(exc), traceback.format_exc(limit=6))))
