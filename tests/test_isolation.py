import logging
import time

import isolated_helpers as h
from compute import ACTIVE_PROCS, IsolatedError, run_isolated


def test_run_isolated_relays_result_logs_and_progress():
    seen, records = [], []

    class Grab(logging.Handler):
        def emit(self, r):
            records.append((r.name, r.getMessage()))

    g = Grab()
    logging.getLogger().addHandler(g)
    try:
        assert run_isolated(h.add, 2, 3, progress=lambda *a: seen.append(a)) == 5
    finally:
        logging.getLogger().removeHandler(g)
    assert ("tests.child", "adding 2 and 3") in records
    assert seen == [(1, 2, "half way")]
    assert not ACTIVE_PROCS


def test_run_isolated_errors_and_timeouts():
    try:
        run_isolated(h.boom)
        raise AssertionError("expected an error")
    except IsolatedError as exc:
        assert "bad thing" in str(exc)
    try:
        run_isolated(h.missing)
        raise AssertionError("expected ImportError")
    except ImportError as exc:
        assert "torch" in str(exc)
    t = time.time()
    try:
        run_isolated(h.hang, timeout=3)
        raise AssertionError("expected TimeoutError")
    except TimeoutError:
        assert time.time() - t < 20
    assert not ACTIVE_PROCS
