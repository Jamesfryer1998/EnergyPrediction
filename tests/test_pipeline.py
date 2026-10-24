"""Features (no leakage), backtest, prediction and the web API on synthetic data."""
import numpy as np
import pandas as pd
from helpers import make_env

from backtest import Backtester
from sources.synthetic import populate
from evaluation import regression_metrics
from features import FeatureBuilder
from prediction import Predictor


def test_metrics_basic():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["n"] == 3 and abs(m["mae"] - 1 / 3) < 1e-9


def test_features_have_no_short_lags(tmp_path):
    settings, store = make_env(tmp_path)
    populate(store, days=60)
    end = store.last_timestamp("demand", "GB")
    fs = FeatureBuilder(store, settings).build(end - pd.Timedelta(days=10), end)
    lags = [int(c.rsplit("lag", 1)[1]) for c in fs.feature_cols if "__lag" in c]
    assert lags and min(lags) >= settings.get("features.min_lag")
    # a realised value at t must never appear as a same-time feature
    for col in fs.target_cols.values():
        assert col not in fs.feature_cols


def test_lag_feature_matches_shifted_target(tmp_path):
    settings, store = make_env(tmp_path)
    populate(store, days=40)
    end = store.last_timestamp("demand", "GB")
    fs = FeatureBuilder(store, settings).build(end - pd.Timedelta(days=5), end)
    f = fs.frame
    t = f.index[200]
    raw = store.read("demand", "GB").set_index("timestamp")["demand_itsdo"]
    assert np.isclose(f.loc[t, "demand_itsdo__lag96"], raw.loc[t - pd.Timedelta(minutes=30 * 96)])


def test_backtest_train_predict_end_to_end(tmp_path):
    settings, store = make_env(tmp_path)
    populate(store, days=200)
    res = Backtester(settings, store).run()
    assert set(res["champions"]) == set(settings.targets)
    board = pd.DataFrame(res["leaderboard"])
    assert (board["mae"] > 0).all()
    # the learned model should beat the naive baseline on price
    price = board[board.target == "price_mid"].set_index("model")["mae"]
    assert price["hgb"] < price["seasonal_naive"]

    pr = Predictor(settings, store)
    trained = pr.train_champions()
    assert all(v["status"] == "ok" for v in trained.values())
    preds = pr.predict_day()
    per_target = preds.groupby("target").size()
    assert set(per_target.index) == set(settings.targets)
    assert per_target.between(46, 50).all()  # 46/50 periods on clock-change days
    rec = pr.recommend(preds)
    assert rec["products"] and rec["windows"]


def test_web_api(tmp_path, monkeypatch):
    settings, store = make_env(tmp_path)
    populate(store, days=30)
    import web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    assert client.get("/").status_code == 200
    s = client.get("/api/status").get_json()
    assert {d["dataset"] for d in s["datasets"]} >= {"demand", "prices_mid"}
    assert client.get("/api/series?dataset=demand&days=3").get_json()["points"]
    assert client.post("/api/jobs/not-a-job").status_code == 409
    assert client.post("/api/settings", json={"values": {"web.port": 1}}).status_code == 400


def test_web_predict_job(tmp_path, monkeypatch):
    import time

    import settings as settings_mod
    import web

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_mod, "get_settings", lambda: settings)  # used by background jobs
    client = web.create_app().test_client()
    opts = client.get("/api/predict/options").get_json()
    assert opts["targets"][0]["max_date"] and "hgb" in opts["models"]

    def run(body):
        assert client.post("/api/jobs/predict", json=body).status_code == 200
        for _ in range(600):
            cur = client.get("/api/jobs").get_json()["current"]
            if cur["status"] != "running":
                return cur
            time.sleep(0.2)
        raise AssertionError("job timed out")

    ok = run({"date": opts["default_date"], "targets": ["price_mid"], "model": "seasonal_naive", "refresh": False})
    assert ok["status"] == "ok" and ok["result"]["summary"][0]["model"] == "seasonal_naive"
    too_far = run({"date": "2099-01-01", "targets": ["price_mid"], "refresh": False})
    assert too_far["status"] == "error" and "furthest day" in too_far["error"]
    f = client.get(f"/api/forecast?target=price_mid&date={opts['default_date']}").get_json()
    assert "seasonal_naive" in f["models"] and len(f["pred"]) >= 46


def test_web_backtest_job_and_cancel(tmp_path, monkeypatch):
    import time

    import web

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    opts = client.get("/api/backtest/options").get_json()
    assert opts["last_full_day"] and opts["defaults"]["days"] == 5

    def wait():
        for _ in range(900):
            cur = client.get("/api/jobs").get_json()["current"]
            if cur["status"] != "running":
                return cur
            time.sleep(0.2)
        raise AssertionError("job timed out")

    body = {"end": opts["last_full_day"], "days": 4, "retrain_every_days": 2, "train_days": 60,
            "targets": ["price_mid"], "models": ["seasonal_naive", "hgb"], "set_champions": False}
    assert client.post("/api/jobs/backtest", json=body).status_code == 200
    done = wait()
    assert done["status"] == "ok", done.get("error")
    assert done["result"]["options"]["train_days"] == 60 and done["result"]["champions"] == {}
    board = client.get(f"/api/backtests?run_id={done['result']['run_id']}").get_json()["leaderboard"]
    assert {r["model"] for r in board} - {"ensemble"} == {"seasonal_naive", "hgb"} and all(r["days"] == 4 for r in board)

    assert client.post("/api/jobs/backtest", json={**body, "days": 60}).status_code == 200
    time.sleep(1.5)
    assert client.post("/api/jobs/cancel").status_code == 200
    assert wait()["status"] == "cancelled"


def test_forecast_history_prefers_genuine_predictions(tmp_path, monkeypatch):
    import web

    settings, store = make_env(tmp_path)
    populate(store, days=40)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    actual = store.read("prices_mid", "GB").set_index("timestamp")["price_mid"]
    last_day = actual.index.max().tz_convert("Europe/London").date()

    rows = []
    for back, genuine, model in ((1, True, "hgb"), (2, False, "hgb"), (1, True, "lightgbm")):
        day = last_day - pd.Timedelta(days=back)
        start = pd.Timestamp(day).tz_localize("Europe/London").tz_convert("UTC")
        idx = pd.date_range(start, periods=48, freq="30min")
        issued = start - pd.Timedelta(hours=10 if model == "hgb" else 2) if genuine else start + pd.Timedelta(days=3)
        rows.append(pd.DataFrame({"timestamp": idx, "target": "price_mid", "model": model,
                                  "y_pred": actual.reindex(idx).to_numpy() + 5.0,  # always 5 too high
                                  "delivery_date": day.isoformat(), "issued_at": issued, "feature_coverage": 1.0}))
    store.write("predictions", "GB", pd.concat(rows), key_cols=("timestamp", "target", "model", "delivery_date"))

    client = web.create_app().test_client()
    h = client.get(f"/api/forecast/history?target=price_mid&date={last_day.isoformat()}&days=3").get_json()
    assert h["genuine_days"]["hgb"] == 1 and h["hindsight_days"]["hgb"] == 1
    assert abs(h["mae"]["hgb"] - 5.0) < 1e-6
    assert len(h["series"]["hgb"]) == 96 and h["actual"]
    # combined (overview): one line, each day from the latest forecast made before it began
    c = client.get(f"/api/forecast/history?target=price_mid&date={last_day.isoformat()}&days=3&combine=1").get_json()
    assert list(c["series"]) == ["previous"] and len(c["series"]["previous"]) == 96
    by_day = c["models_by_day"]
    assert by_day[(last_day - pd.Timedelta(days=1)).isoformat()] == "lightgbm"  # issued later, still before the day
    assert by_day[(last_day - pd.Timedelta(days=2)).isoformat()] == "hgb"


def test_heartbeat_version_changes_when_data_changes(tmp_path, monkeypatch):
    import time

    import web

    settings, store = make_env(tmp_path)
    populate(store, days=10)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    v1 = client.get("/api/heartbeat").get_json()["version"]
    time.sleep(0.05)
    populate(store, days=12)  # rewrites partitions, as an ingest would
    hb = client.get("/api/heartbeat").get_json()
    assert hb["version"] > v1 and hb["job"] is None


def test_scheduler_fires_once_per_day_and_waits_for_busy_runner(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from web.scheduler import Scheduler

    settings, _ = make_env(tmp_path)
    tz = ZoneInfo("Europe/London")

    class FakeRunner:
        def __init__(self):
            self.started, self.is_busy = [], False

        def busy(self):
            return self.is_busy

        def start(self, job, args, settings=None):
            self.started.append((job, args))
            return {"id": len(self.started)}

    runner = FakeRunner()
    sch = Scheduler(runner, lambda: settings)
    sch.update({"time": "02:30", "steps": ["ingest", "train", "predict"], "enabled": True, "refresh_enabled": False})
    cfg = sch.load()
    # pretend it was switched on yesterday evening
    cfg["last_slot"] = None
    sch._save(cfg)

    assert not sch.tick(datetime(2026, 10, 6, 2, 0, tzinfo=tz))          # too early
    runner.is_busy = True
    assert not sch.tick(datetime(2026, 10, 6, 2, 31, tzinfo=tz))         # waits for running job
    runner.is_busy = False
    assert sch.tick(datetime(2026, 10, 6, 2, 32, tzinfo=tz))             # fires
    assert runner.started[0] == ("nightly", {"steps": ["ingest", "train", "predict"], "backtest_days": None,
                                             "tune_days": None, "trigger": "schedule"})
    assert not sch.tick(datetime(2026, 10, 6, 3, 0, tzinfo=tz))          # only once per day
    assert sch.update({"tune_days": 90})["tune_days"] == 90 and sch.job_args(sch.load(), "x")["tune_days"] == 90
    try:
        sch.update({"tune_days": 2})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    sch.update({"tune_days": None})
    assert sch.tick(datetime(2026, 10, 7, 2, 30, tzinfo=tz))             # next day
    nxt = sch.next_run(now=datetime(2026, 10, 7, 9, 0, tzinfo=tz))
    assert nxt.isoformat().startswith("2026-10-08T02:30")
    sch.update({"enabled": False})
    assert not sch.tick(datetime(2026, 10, 8, 2, 30, tzinfo=tz)) and sch.next_run() is None


def test_nightly_runs_selected_steps_in_order(tmp_path):
    from pipeline import run_nightly

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    seen = []
    rec = run_nightly(settings, steps=["predict", "train", "backtest"], backtest_days=3,
                      progress=lambda p: seen.append(p["step"]))
    assert [s["step"] for s in rec["steps"]] == ["backtest", "train", "predict", "recommend"]
    assert rec["status"] == "ok" and rec["planned"] == ["backtest", "train", "predict"]
    assert seen[0] == "backtest" and seen[-1] == "done"


def test_parallel_backtest_matches_sequential(tmp_path):
    settings, store = make_env(tmp_path)
    populate(store, days=150)
    seq = Backtester(settings, store).run(models=["seasonal_naive", "hgb"], days=3, set_champions=False)
    settings.raw.setdefault("backtest", {})["workers"] = 2
    par = Backtester(settings, store).run(models=["seasonal_naive", "hgb"], days=3, set_champions=False)
    key = lambda b: sorted((r["target"], r["model"], round(r["mae"], 6)) for r in b["leaderboard"])  # noqa: E731
    assert key(seq) == key(par)


def test_ingest_fetches_windows_concurrently(tmp_path):
    import threading
    import time
    from datetime import date

    from sources import DataSource

    settings, store = make_env(tmp_path)
    active, peak = [0], [0]
    lock = threading.Lock()

    class SlowSource(DataSource):
        name = "_slow_test_source"
        max_days_per_request = 1
        parallel_windows = 4

        def fetch(self, start, end):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            return start

        def transform(self, d):
            idx = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min")
            return pd.DataFrame({"timestamp": idx, "v": 1.0})

    t0 = time.time()
    res = SlowSource(store=store, settings=settings).update(date(2026, 1, 1), date(2026, 1, 8))
    assert res.status == "ok" and res.windows == 8 and res.rows == 8 * 48
    assert peak[0] > 1 and time.time() - t0 < 1.2   # 8 x 0.2s sequential would be 1.6s


def test_parallel_backtest_runs_inside_dashboard_job(tmp_path, monkeypatch):
    """Jobs run in their own process; a parallel backtest must be able to start workers from it."""
    import time

    import web

    settings, store = make_env(tmp_path)
    settings.raw.setdefault("backtest", {})["workers"] = 2
    populate(store, days=150)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    body = {"days": 3, "retrain_every_days": 3, "train_days": 60, "targets": ["price_mid", "demand"],
            "models": ["seasonal_naive", "hgb"], "set_champions": False}
    assert client.post("/api/jobs/backtest", json=body).status_code == 200
    for _ in range(900):
        cur = client.get("/api/jobs").get_json()["current"]
        if cur["status"] != "running":
            break
        time.sleep(0.2)
    assert cur["status"] == "ok", cur.get("error")
    assert not cur["result"]["errors"]


def test_scheduler_refreshes_data_through_the_day(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from web.scheduler import Scheduler

    settings, _ = make_env(tmp_path)
    tz = ZoneInfo("Europe/London")

    class FakeRunner:
        def __init__(self):
            self.started, self.is_busy = [], False

        def busy(self):
            return self.is_busy

        def start(self, job, args, settings=None):
            self.started.append((job, args))
            return {"id": len(self.started)}

    runner = FakeRunner()
    sch = Scheduler(runner, lambda: settings)
    sch.update({"refresh_enabled": True, "refresh_minutes": 30})   # daily schedule stays off
    t = datetime(2026, 10, 8, 9, 0, tzinfo=tz)
    assert sch.tick(t) and runner.started[-1] == ("ingest", {"trigger": "refresh"})
    assert not sch.tick(t.replace(minute=20))                         # not due yet
    runner.is_busy = True
    assert not sch.tick(t.replace(minute=31))                         # waits for a running job
    runner.is_busy = False
    assert sch.tick(t.replace(minute=32)) and len(runner.started) == 2
    # the daily pipeline takes priority over a due refresh, and its ingest counts as one
    sch.update({"enabled": True, "time": "10:00", "steps": ["ingest", "predict"]})
    cfg = sch.load()
    cfg["last_slot"] = None  # as if it was switched on before 10:00 (update() uses the real clock)
    sch._save(cfg)
    assert sch.tick(datetime(2026, 10, 8, 10, 5, tzinfo=tz)) and runner.started[-1][0] == "nightly"
    assert not sch.tick(datetime(2026, 10, 8, 10, 20, tzinfo=tz))
    sch.update({"refresh_enabled": False})
    assert not sch.tick(datetime(2026, 10, 8, 12, 0, tzinfo=tz)) and sch.next_refresh() is None


def test_fetch_now_runs_ingest_and_resets_refresh_timer(tmp_path, monkeypatch):
    import time

    import web

    # only an offline (manual CSV) source enabled, so no network is touched
    sources = {n: {"enabled": False} for n in ["prices_mid", "prices_system", "demand", "grid_generation",
                                               "grid_carbon_intensity", "weather", "calendar_events",
                                               "weather_events", "gas_prices", "carbon_allowances", "ig_gas", "certificates"]}
    sources["certificate_prices"] = {"enabled": True}
    settings, _ = make_env(tmp_path, sources=sources)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    app = web.create_app()
    client = app.test_client()
    r = client.post("/api/fetch-now")
    assert r.status_code == 200 and r.get_json()["job"] == "ingest"
    for _ in range(300):
        cur = client.get("/api/jobs").get_json()["current"]
        if cur["status"] != "running":
            break
        time.sleep(0.1)
    assert cur["status"] == "ok" and cur["args"]["trigger"] == "fetch"
    assert [x["source"] for x in cur["result"]] == ["certificate_prices"]
    sched = client.get("/api/schedule").get_json()
    assert sched["config"]["last_refresh"]           # timer restarted from this fetch


def test_features_use_storms_on_the_day_and_lag_gas_two_days(tmp_path):
    settings, store = make_env(tmp_path)
    populate(store, days=40)
    end = store.last_timestamp("demand", "GB")
    day = (end - pd.Timedelta(days=5)).tz_convert("Europe/London").normalize()
    store.write("weather_events", "GB", pd.DataFrame({"timestamp": [day.tz_convert("UTC")], "storm_name": ["Amy"],
                                                      "named_storm": [1]}), ("timestamp", "storm_name"))
    gas_days = pd.date_range(day - pd.Timedelta(days=10), day, freq="1D").tz_convert("UTC")
    store.write("gas_prices", "GB", pd.DataFrame({"timestamp": gas_days, "gas_sap_pkwh": range(len(gas_days))}))
    fs = FeatureBuilder(store, settings).build(end - pd.Timedelta(days=10), end)
    f = fs.frame
    noon = (day + pd.Timedelta(hours=12)).tz_convert("UTC")
    assert f.loc[noon, "evt__named_storm"] == 1 and f.loc[noon - pd.Timedelta(days=1), "evt__named_storm"] == 0
    # on storm day D the newest gas value usable is D-2's
    assert f.loc[noon, "gas_prices__gas_sap_pkwh"] == len(gas_days) - 3


def test_nightly_writes_step_logs_served_by_the_dashboard(tmp_path, monkeypatch):
    import logging

    from pipeline import run_nightly

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    rec = run_nightly(settings, steps=["backtest", "train", "predict"], backtest_days=3)
    assert all(s.get("started") for s in rec["steps"])
    assert not any(isinstance(h, logging.FileHandler) and "runs" in str(getattr(h, "baseFilename", ""))
                   for h in logging.getLogger().handlers)  # handler removed afterwards

    import web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    run_id = rec["run_id"]
    bt = client.get(f"/api/runs/{run_id}/logs?step=backtest").get_json()
    assert bt["source"] == "run" and bt["total"] > 0
    assert all(x["step"] == "backtest" for x in bt["lines"])
    assert any("backtest" in x["msg"] for x in bt["lines"])
    # polling from `next` returns only new lines
    again = client.get(f"/api/runs/{run_id}/logs?step=backtest&after={bt['next']}").get_json()
    assert again["lines"] == [] and again["total"] == bt["total"]
    everything = client.get(f"/api/runs/{run_id}/logs").get_json()
    assert everything["total"] >= bt["total"]
    warn = client.get(f"/api/runs/{run_id}/logs?level=WARNING").get_json()
    assert all(x["level"] in ("WARNING", "ERROR", "CRITICAL") for x in warn["lines"])
    assert client.get("/api/runs/nope/logs").status_code == 404
    sched = client.get("/api/schedule").get_json()
    assert sched["runs"][0]["steps"][0]["started"]


def test_step_logs_fall_back_to_jobs_log_for_old_runs(tmp_path):
    from datetime import datetime, timezone

    from pipeline.runlog import read_step_logs

    def local(iso):  # jobs.log is written in local time
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "jobs.log").write_text("\n".join([
        f"{local('2026-10-09T01:00:00+00:00')},000 INFO    sources.registry | before the run",
        f"{local('2026-10-09T01:31:50+00:00')},000 INFO    sources.registry | ingest demand/GB",
        f"{local('2026-10-09T01:32:00+00:00')},000 ERROR   source.gas_prices | window failed",
        "Traceback (most recent call last):",
        "  boom",
        f"{local('2026-10-09T02:00:00+00:00')},000 INFO    training.trainer | tuned demand/hgb",
        f"{local('2026-10-09T03:00:00+00:00')},000 INFO    sources.registry | a later refresh",
    ]) + "\n")
    run = {"run_id": "20261009T013146_nightly", "started": "2026-10-09T01:31:46+00:00",
           "finished": "2026-10-09T02:30:00+00:00",
           "steps": [{"step": "ingest", "started": "2026-10-09T01:31:46+00:00"},
                     {"step": "tune", "started": "2026-10-09T01:40:00+00:00"}]}
    ing = read_step_logs(tmp_path / "data", run, "ingest", logs_root=logs)
    assert ing["source"] == "jobs.log"
    assert [x["msg"].split("\n")[0] for x in ing["lines"]] == ["ingest demand/GB", "window failed"]
    assert "boom" in ing["lines"][1]["msg"]
    tune = read_step_logs(tmp_path / "data", run, "tune", logs_root=logs)
    assert [x["msg"] for x in tune["lines"]] == ["tuned demand/hgb"]
    # no step times recorded: whole run, filtered by logger
    bare = {**run, "steps": []}
    assert [x["msg"] for x in read_step_logs(tmp_path / "data", bare, "tune", logs_root=logs)["lines"]] == ["tuned demand/hgb"]
    assert ing["lines"][1]["level"] == "ERROR"
    assert datetime.fromisoformat(ing["lines"][0]["t"]) == datetime(2026, 10, 9, 1, 31, 50, tzinfo=timezone.utc)


def test_backtest_survives_a_crashing_worker(tmp_path, monkeypatch):
    settings, store = make_env(tmp_path)
    populate(store, days=150)
    settings.raw.setdefault("backtest", {})["workers"] = 2
    monkeypatch.setenv("ENERGY_TEST_CRASH", "price_mid/hgb")  # that walk's process dies
    res = Backtester(settings, store).run(models=["seasonal_naive", "hgb"], days=3, set_champions=False)
    board = {(r["target"], r["model"]): r for r in res["leaderboard"]}
    assert "crashed" in str(board[("price_mid", "hgb")].get("error"))
    ok = [k for k, r in board.items() if r.get("mae") == r.get("mae") and r.get("mae") is not None]
    assert len(ok) == len(board) - 1  # every other walk still finished


def test_models_ignore_empty_and_constant_columns():
    import numpy as np
    import pandas as pd

    from models import create_model

    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "empty": np.nan, "const": 5.0})  # e.g. a source with no data yet
    y = pd.Series(3 * X["a"] - X["b"] + rng.normal(scale=0.1, size=n))
    m = create_model("hgb").fit(X, y)  # sklearn fails on an all-NaN column without the filter
    assert m.feature_names == ["a", "b"]
    X_new = X.head(5).assign(const=1e6, empty=42.0)  # the source starts reporting
    assert np.isfinite(m.predict(X_new)).all()


def test_torch_inputs_are_robustly_scaled():
    import numpy as np
    import pandas as pd

    from models.zoo import TorchMLP

    m = TorchMLP.__new__(TorchMLP)
    X = pd.DataFrame({"a": np.r_[np.zeros(999), 1.0], "b": np.linspace(0, 1, 1000)})
    # replicate the scaling from _fit
    arr = X.to_numpy(dtype=np.float64)
    m.mu_ = np.nan_to_num(np.nanmean(arr, axis=0)).astype(np.float32)
    sd = np.nanstd(arr, axis=0)
    floor = np.maximum(1e-3 * np.abs(m.mu_), 1e-6)
    m.sd_ = np.where(sd > floor, sd, np.maximum(np.abs(m.mu_), 1.0)).astype(np.float32)
    z = m._prep(pd.DataFrame({"a": [1e9], "b": [0.5]}))
    assert np.abs(z).max() <= TorchMLP.Z_CLIP


def test_tuning_runs_every_time_and_logs_each_trial(tmp_path):
    import logging

    import pytest

    pytest.importorskip("optuna")
    from pipeline import run_tuning
    from training import Artifacts

    settings, store = make_env(tmp_path, tuning={"models": ["hgb"], "n_trials": 2, "timeout_minutes": 5,
                                                  "train_days": 60, "retune_every_days": 0})
    populate(store, days=120)
    records = []

    class Grab(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    h = Grab(level=logging.INFO)
    root = logging.getLogger()
    old = root.level
    root.setLevel(logging.INFO)
    root.addHandler(h)
    try:
        art = Artifacts(settings)
        first = run_tuning(settings, store, "GB", art)
        again = run_tuning(settings, store, "GB", art, tune_days=30)  # retune_every_days 0: tunes again
    finally:
        root.removeHandler(h)
        root.setLevel(old)
    keys = [k for k in first if k.endswith("/hgb")]
    assert keys and all(first[k]["status"] == "ok" and again[k]["status"] == "ok" for k in keys)
    assert any("trial 1/2" in m for m in records) and any(m.startswith("tuned ") for m in records)
    assert any("(60 days)" in m for m in records) and any("(30 days)" in m for m in records)  # schedule override
    assert art.load_params(keys[0].split("/")[0], "hgb")["n_trials"] == 2


def test_linear_model_fits_and_backtests(tmp_path):
    import numpy as np

    from models import create_model

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "c": np.where(rng.random(500) < .2, np.nan, 1.0)})
    y = pd.Series(4 * X["a"] - 2 * X["b"] + rng.normal(scale=0.1, size=500))
    m = create_model("linear", alpha=0.01).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.99 and np.isfinite(m.predict(X.assign(a=1e9))).all()
    assert list(m.feature_importance().index[:2]) == ["a", "b"]

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    res = Backtester(settings, store).run(models=["seasonal_naive", "linear"], days=3, set_champions=False)
    lin = [r for r in res["leaderboard"] if r["model"] == "linear"]
    assert lin and all(r.get("mae") == r.get("mae") and r.get("mae") is not None for r in lin)


def test_ensemble_weights_blend_and_champion(tmp_path):
    import json

    import numpy as np

    from models.ensemble import fit_weights, prune
    from prediction import Predictor
    from training import Artifacts

    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    good, ok_, bad = y + rng.normal(scale=.3, size=400), y + rng.normal(scale=.6, size=400), rng.normal(size=400)
    w = fit_weights(np.c_[good, ok_, bad], y)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all() and w[0] > w[1] > w[2] - 1e-9
    assert prune({"a": .7, "b": .27, "c": .03}, .05) == {"a": .7 / .97, "b": .27 / .97}

    settings, store = make_env(tmp_path)
    populate(store, days=150)
    res = Backtester(settings, store).run(models=["seasonal_naive", "linear", "hgb"], days=6)
    board = pd.DataFrame(res["leaderboard"])
    ens = board[board["model"] == "ensemble"]
    assert len(ens) >= 1 and ens["note"].str.contains("2nd half").all()
    art = Artifacts(settings)
    for t in settings.targets:
        champ = art.load_champion(t)["model"]
        assert board[(board["target"] == t) & (board["champion"])]["model"].tolist() == [champ]
        spec = art.load_ensemble(t)
        if t not in set(ens["target"]):  # one model took all the weight: no blend
            assert champ != "ensemble"
            continue
        assert board[board["target"] == t]["mae_holdout"].notna().all()
        assert abs(sum(spec["weights"].values()) - 1) < 1e-9 and len(spec["weights"]) >= 2
        assert (champ == "ensemble") == bool(spec["beats_best_single"])

    # production: force the ensemble as champion and predict with it
    t = ens["target"].iloc[0]
    art.save_champion(t, "ensemble", {"mae": 1.0}, "test")
    pr = Predictor(settings, store)
    out = pr.train_champions([t])
    assert out[t]["model"] == "ensemble" and set(out[t]["members"]) == set(art.load_ensemble(t)["weights"])
    day = pr.max_predictable_date(t)
    preds = pr.predict_day(day, [t])
    assert set(preds["model"]) == {"ensemble"} and preds["y_pred"].notna().all()
    weights = art.load_ensemble(t)["weights"]
    X, _ = __import__("features").FeatureBuilder(store, settings).build(*pr.day_bounds(day)).xy(t, *pr.day_bounds(day), dropna=False)
    from models import Forecaster
    manual = sum(w * Forecaster.load(pr.member_path(t, m)).predict(X) for m, w in weights.items())
    assert np.allclose(preds["y_pred"].to_numpy(), manual)
    json.dumps(art.load_champion(t))


def test_tuning_study_generations_reset(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone

    from training import Artifacts

    settings, _ = make_env(tmp_path)
    art = Artifacts(settings)
    name, new, _ = art.study_generation("demand", "hgb", 7)
    assert new and name == "GB:demand:hgb"           # generation 1 keeps the old study name
    name2, new2, _ = art.study_generation("demand", "hgb", 7)
    assert name2 == name and not new2
    p = settings.data_dir / "optuna" / "generations.json"
    gens = json.loads(p.read_text())
    gens["GB:demand:hgb"]["started"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    p.write_text(json.dumps(gens))
    name3, new3, _ = art.study_generation("demand", "hgb", 7)
    assert new3 and name3 == "GB:demand:hgb:g2"
    assert not art.study_generation("demand", "hgb", 0)[1]  # 0 = never reset


def test_day_bounds_cover_clock_change_days(tmp_path):
    from datetime import date

    from prediction.predictor import Predictor

    settings, store = make_env(tmp_path)
    pr = Predictor(settings, store)
    half_hours = {d: (lambda s, e: (e - s) / pd.Timedelta(minutes=30))(*pr.day_bounds(d))
                  for d in (date(2026, 10, 24), date(2026, 10, 25), date(2026, 3, 29))}
    # normal day, clocks go back (25 hours), clocks go forward (23 hours)
    assert half_hours == {date(2026, 10, 24): 48, date(2026, 10, 25): 50, date(2026, 3, 29): 46}
    start, end = pr.day_bounds(date(2026, 10, 25))
    assert str(start) == "2026-10-24 23:00:00+00:00" and str(end) == "2026-10-26 00:00:00+00:00"


def test_predict_on_the_day_the_clocks_go_back(tmp_path, monkeypatch):
    from datetime import date

    import prediction.predictor as predmod
    from prediction.predictor import Predictor

    settings, store = make_env(tmp_path)
    populate(store, days=200)
    pr = Predictor(settings, store)
    pr.train_champions()
    day = date(2026, 10, 25)  # clocks go back: 50 half-hours
    monkeypatch.setattr(predmod.Predictor, "max_predictable_date", lambda self, t: day)
    preds = pr.predict_day(day)
    assert not preds.empty
    assert (preds.groupby("target").size() == 50).all()
