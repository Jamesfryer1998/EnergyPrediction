"""Module-level functions for test_isolation (child processes must be able to import them)."""
import logging
import time


def add(a, b, progress=None):
    logging.getLogger("tests.child").info("adding %s and %s", a, b)
    if progress:
        progress(1, 2, "half way")
    return a + b


def boom():
    raise ValueError("bad thing")


def missing():
    raise ImportError("no module named torch")


def hang():
    time.sleep(60)
