"""Flask dashboard: forecasts, backtest leaderboard, data health, job controls, settings."""
from __future__ import annotations

import math
import os
from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from settings import get_settings, save_overrides
from storage import TS, ParquetStore
from .jobs import JOBS, JobRunner
from .scheduler import STEPS, Scheduler
from .trading_runner import TradingRunner

# Settings the dashboard may edit (dotted path -> type). Everything else stays in YAML.
EDITABLE = {
    "models.candidates": list,
    "training.train_days": int,
    "training.validation_days": int,
    "tuning.enabled": bool,
    "tuning.n_trials": int,
    "tuning.timeout_minutes": float,
    "tuning.max_minutes": float,
    "tuning.train_days": int,
    "tuning.retune_every_days": int,
    "tuning.warm_trials": int,
    "tuning.reset_every_days": int,
    "ensemble.enabled": bool,
    "ensemble.max_members": int,
    "backtest.days": int,
    "backtest.retrain_every_days": int,
    "recommendations.top_n_periods": int,
}

# Hover help for the Config page.
EDITABLE_HELP = {
    "models.candidates": "Models the backtest compares. The best per target (or the ensemble) becomes the champion.",
    "training.train_days": "Days of history each production model and backtest refit trains on.",
    "training.validation_days": "Most recent days held back for early stopping and validation when training.",
    "tuning.enabled": "Run Optuna hyper-parameter tuning in the daily pipeline's Tune step.",
    "tuning.n_trials": "Trials per target/model in a fresh study (before it has learnt from earlier runs).",
    "tuning.timeout_minutes": "Time limit per target/model pair.",
    "tuning.max_minutes": "Time limit for the whole Tune step, shared between the pairs still to do (0 = none).",
    "tuning.train_days": "Days of history each tuning trial trains on. Shorter = faster trials.",
    "tuning.retune_every_days": "Only re-tune a pair when its settings are older than this many days (0 = every run).",
    "tuning.warm_trials": "Trials per pair once its study already has enough earlier trials to learn from.",
    "tuning.reset_every_days": "Start a fresh Optuna study this often, seeded with the current best, so old "
                               "trials stop steering the search (0 = never).",
    "ensemble.enabled": "After each backtest, try a weighted blend of the best models; it becomes champion "
                        "only if it beats the best single model.",
    "ensemble.max_members": "How many of the best single models the blend may combine.",
    "backtest.days": "Default number of days the walk-forward backtest tests over.",
    "backtest.retrain_every_days": "How often models are refitted while walking through the backtest window.",
    "recommendations.top_n_periods": "How many cheapest/dearest half-hours to list for each day.",
}


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy/pandas scalars -> python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].map(lambda t: t.isoformat() if pd.notna(t) else None)
    return _clean(df.to_dict(orient="records"))


class _QuietPolling:
    """Drop access-log lines for the dashboard's 2-second polling so the log stays readable."""

    def filter(self, record) -> bool:
        msg = record.getMessage()
        return not (" 200 " in msg and ("/api/heartbeat" in msg or "/logs?" in msg))


def create_app() -> Flask:
    import logging

    wz = logging.getLogger("werkzeug")
    if not any(isinstance(f, _QuietPolling) for f in wz.filters):
        wz.addFilter(_QuietPolling())
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev")
    runner = JobRunner()
    trader = TradingRunner(lambda: get_settings())
    scheduler = Scheduler(runner, lambda: get_settings(), trading=trader)
    # With `--debug` Flask's reloader runs a watcher process plus the real server; only the
    # real server (WERKZEUG_RUN_MAIN=true) may run the schedule, or jobs would start twice.
    if os.environ.get("ENERGY_NO_SCHEDULER") != "1" and (
        os.environ.get("ENERGY_WEB_RELOADER") != "1" or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    ):
        scheduler.start()
    app.extensions["energy_scheduler"] = scheduler

    def ctx():
        s = get_settings()
        return s, ParquetStore(s.data_dir)

    @app.get("/")
    def index():
        s, _ = ctx()
        return render_template("index.html", targets=list(s.targets), jobs=JOBS)

    @app.get("/api/status")
    def status():
        from training import Artifacts

        s, store = ctx()
        region = request.args.get("region", "GB")
        art = Artifacts(s, region)
        champions = {t: art.load_champion(t) for t in s.targets}
        runs = _mark_interrupted(art.list_runs(1), runner)
        return jsonify(_clean({
            "region": region,
            "targets": list(s.targets),
            "champions": champions,
            "last_run": runs[0] if runs else None,
            "jobs": runner.snapshot(),
            "datasets": store.summary(region),
        }))

    @app.get("/api/sources")
    def sources():
        from sources import list_sources

        s, store = ctx()
        region = request.args.get("region", "GB")
        summary = {d["dataset"]: d for d in store.summary(region)}
        out = []
        for src in list_sources():
            info = summary.get(src["name"], {})
            out.append({**src, "enabled": s.source_enabled(src["name"]), "first": info.get("first"),
                        "last": info.get("last"), "bytes": info.get("bytes", 0)})
        return jsonify(out)

    @app.get("/api/series")
    def series():
        _, store = ctx()
        ds, col = request.args["dataset"], request.args.get("column")
        days = int(request.args.get("days", 7))
        end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
        df = store.read(ds, request.args.get("region", "GB"), end - pd.Timedelta(days=days + 2), end)
        if df.empty:
            return jsonify({"columns": [], "points": []})
        num_cols = [c for c in df.select_dtypes("number").columns]
        cols = [col] if col in num_cols else num_cols[:6]
        if len(df) > 2000:  # keep payload light
            df = df.set_index(TS)[cols].resample(f"{max(1, len(df) // 2000) * 30}min").mean().reset_index()
        return jsonify({"columns": num_cols, "shown": cols, "points": _records(df[[TS, *cols]])})

    @app.get("/api/forecast")
    def forecast():
        """Predictions for one target/day: every model that has predicted it, plus actuals."""
        s, store = ctx()
        region = request.args.get("region", "GB")
        target = request.args.get("target") or next(iter(s.targets))
        day = request.args.get("date")
        preds = store.read("predictions", region)
        empty = {"target": target, "dates": [], "pred": [], "actual": [], "series": {}, "models": []}
        if preds.empty:
            return jsonify(empty)
        preds = preds[preds["target"] == target]
        dates = sorted(preds["delivery_date"].unique(), reverse=True)
        day = day or (dates[0] if dates else None)
        p = preds[preds["delivery_date"] == day].sort_values(TS)
        if p.empty:
            return jsonify({**empty, "date": day, "dates": list(dates[:90])})
        latest_model = p.sort_values("issued_at")["model"].iloc[-1]
        series = {m: _records(g[[TS, "y_pred"]]) for m, g in p.groupby("model")}
        issued = {m: g["issued_at"].max().isoformat() for m, g in p.groupby("model")}
        tcfg = s.targets[target]
        start = p[TS].min() - pd.Timedelta(days=2)
        end = p[TS].max() + pd.Timedelta(minutes=30)
        actual = store.read(tcfg["source"], region, start, end, columns=[tcfg["column"]])
        actual = actual.rename(columns={tcfg["column"]: "y"}) if not actual.empty else actual
        return jsonify({"target": target, "date": day, "dates": list(dates[:90]), "model": latest_model,
                        "models": sorted(series), "issued": issued, "series": series,
                        "pred": series[latest_model], "actual": _records(actual)})

    @app.get("/api/forecast/history")
    def forecast_history():
        """Earlier predictions for the `days` delivery days before `date`, stitched into one line per
        model, plus actuals over the same window. For each day it prefers the latest prediction
        issued *before* that day started (a genuine forecast); otherwise it falls back to the latest
        one and counts the day as hindsight."""
        s, store = ctx()
        region = request.args.get("region", "GB")
        target = request.args.get("target") or next(iter(s.targets))
        end_day = date.fromisoformat(request.args["date"])
        days = max(1, min(int(request.args.get("days", 7)), 90))
        tz = s.timezone
        win_start = pd.Timestamp(end_day - pd.Timedelta(days=days)).tz_localize(tz).tz_convert("UTC")
        win_end = (pd.Timestamp(end_day).tz_localize(tz) + pd.Timedelta(days=1)).normalize().tz_convert("UTC")
        out = {"target": target, "window_start": win_start.isoformat(), "series": {}, "hindsight_days": {},
               "genuine_days": {}, "mae": {}, "actual": []}
        tcfg = s.targets[target]
        actual = store.read(tcfg["source"], region, win_start, win_end, columns=[tcfg["column"]])
        if not actual.empty:
            out["actual"] = _records(actual.rename(columns={tcfg["column"]: "y"}))
        p = store.read("predictions", region, win_start, win_end)
        if p.empty:
            return jsonify(out)
        p = p[(p["target"] == target) & (p["delivery_date"] < end_day.isoformat())]
        if p.empty:
            return jsonify(out)
        p = p.copy()
        p["issued_at"] = pd.to_datetime(p["issued_at"], utc=True)
        day_start = pd.to_datetime(p["delivery_date"]).dt.tz_localize(tz).dt.tz_convert("UTC")
        p["genuine"] = p["issued_at"] < day_start
        if request.args.get("combine"):
            # One line across models: for each day, the latest forecast made before that day began
            # (whichever model made it, e.g. the champion of the time), else the latest one at all.
            chosen, used = [], {}
            for day, d in p.groupby("delivery_date"):
                pool = d[d["genuine"]] if d["genuine"].any() else d
                latest = pool[pool["issued_at"] == pool["issued_at"].max()]
                model = sorted(latest["model"].unique())[0]
                chosen.append(latest[latest["model"] == model])
                used[day] = model
            c = pd.concat(chosen).sort_values(TS)
            p = c.assign(model="previous")
            out["models_by_day"] = used
        actual_by_ts = (actual.set_index(TS)[tcfg["column"]] if not actual.empty else pd.Series(dtype=float))
        for model, g in p.groupby("model"):
            chosen = []
            hindsight = genuine = 0
            for _, d in g.groupby("delivery_date"):
                pool = d[d["genuine"]] if d["genuine"].any() else d
                latest = pool["issued_at"].max()
                chosen.append(pool[pool["issued_at"] == latest])
                genuine += int(d["genuine"].any())
                hindsight += int(not d["genuine"].any())
            c = pd.concat(chosen).sort_values(TS)
            out["series"][model] = _records(c[[TS, "y_pred"]])
            out["hindsight_days"][model], out["genuine_days"][model] = hindsight, genuine
            joined = c.set_index(TS)["y_pred"].to_frame().join(actual_by_ts.rename("y"), how="inner").dropna()
            out["mae"][model] = float((joined["y_pred"] - joined["y"]).abs().mean()) if len(joined) else None
        return jsonify(_clean(out))

    @app.get("/api/predict/options")
    def predict_options():
        """What the Predict form needs: targets, models, and how far ahead each target can go."""
        from models import available_models
        from prediction import Predictor

        s, store = ctx()
        pr = Predictor(s, store, request.args.get("region", "GB"))
        targets = []
        for t in s.targets:
            last = pr.last_data(t)
            champ = pr.art.load_champion(t)
            targets.append({
                "name": t,
                "last_data": last.isoformat() if last is not None else None,
                "max_date": (md.isoformat() if (md := pr.max_predictable_date(t)) else None),
                "champion": champ["model"] if champ else None,
                "has_production_model": pr.art.model_path(t, "production").exists(),
            })
        models = available_models()
        if any(pr.art.load_ensemble(t) for t in s.targets):
            models = [*models, "ensemble"]
        return jsonify({"targets": targets, "models": models, "default_date": pr.tomorrow().isoformat()})

    @app.get("/api/predictions/index")
    def predictions_index():
        _, store = ctx()
        p = store.read("predictions", request.args.get("region", "GB"))
        if p.empty:
            return jsonify([])
        idx = (p.groupby(["delivery_date", "target", "model"])
                .agg(issued_at=("issued_at", "max"), mean=("y_pred", "mean"), n=("y_pred", "size"))
                .reset_index().sort_values(["delivery_date", "issued_at"], ascending=False).head(200))
        return jsonify(_records(idx))

    @app.get("/api/recommendations")
    def recommendations():
        from prediction import Predictor

        s, store = ctx()
        day = request.args.get("date")
        pr = Predictor(s, store, request.args.get("region", "GB"))
        try:
            d = date.fromisoformat(day) if day else _latest_delivery(store, pr.region) or pr.tomorrow()
            preds = pr.latest_predictions(d, request.args.get("model"))
            rec = pr.recommend(preds if len(preds) else None, day=d, persist=False)
        except Exception as exc:
            return jsonify({"error": str(exc), "products": [], "windows": []})
        return jsonify(_clean({**rec, "windows": _records(pd.DataFrame(rec["windows"]))}))

    @app.get("/api/backtests")
    def backtests():
        _, store = ctx()
        m = store.read("backtest_metrics", request.args.get("region", "GB"))
        if m.empty:
            return jsonify({"runs": [], "leaderboard": []})
        runs = m.sort_values(TS, ascending=False)["run_id"].drop_duplicates().tolist()
        run = request.args.get("run_id") or runs[0]
        board = m[m["run_id"] == run].drop(columns=[TS])
        if "mae" in board:
            board = board.sort_values(["target", "mae"])
        return jsonify({"runs": runs[:50], "run_id": run, "leaderboard": _records(board)})

    @app.get("/api/backtest/options")
    def backtest_options():
        """Defaults and data range for the Backtest form."""
        from models import available_models

        s, store = ctx()
        region = request.args.get("region", "GB")
        firsts, lasts = [], []
        for t in s.targets.values():
            f, last = store.first_timestamp(t["source"], region), store.last_timestamp(t["source"], region)
            if f is not None:
                firsts.append(f)
            if last is not None:
                lasts.append(last)
        tz = s.timezone
        last_full_day = ((max(lasts).tz_convert(tz).normalize() - pd.Timedelta(hours=1)).date().isoformat()
                         if lasts else None)
        return jsonify({
            "targets": list(s.targets),
            "models": available_models(),
            "candidates": s.get("models.candidates", []),
            "defaults": {"days": s.get("backtest.days", 60), "retrain_every_days": s.get("backtest.retrain_every_days", 7),
                         "train_days": s.get("training.train_days", 540)},
            "data_start": min(firsts).tz_convert(tz).date().isoformat() if firsts else None,
            "last_full_day": last_full_day,
        })

    @app.get("/api/backtest_series")
    def backtest_series():
        _, store = ctx()
        run, target = request.args["run_id"], request.args["target"]
        p = store.read("backtest_predictions", request.args.get("region", "GB"))
        if p.empty:
            return jsonify({})
        p = p[(p["run_id"] == run) & (p["target"] == target)]
        out = {"actual": _records(p.drop_duplicates(TS)[[TS, "y_true"]].sort_values(TS))}
        out["models"] = {mdl: _records(g[[TS, "y_pred"]].sort_values(TS)) for mdl, g in p.groupby("model")}
        return jsonify(out)

    @app.get("/api/runs")
    def runs():
        from training import Artifacts

        s, _ = ctx()
        return jsonify(_clean(Artifacts(s, request.args.get("region", "GB")).list_runs(30)))

    @app.get("/api/runs/<run_id>/logs")
    def run_logs(run_id):
        """Log lines of one pipeline run, optionally for one step. Poll with ?after=<next>."""
        import json as _json

        from pipeline.runlog import read_step_logs
        from training import Artifacts

        s, _ = ctx()
        try:
            path = Artifacts(s, request.args.get("region", "GB")).runs_dir / f"{run_id}.json"
            if "/" in run_id or ".." in run_id or not path.exists():
                return jsonify({"error": f"no run {run_id}"}), 404
            run = _json.loads(path.read_text())
            return jsonify(read_step_logs(
                s.data_dir, run, request.args.get("step") or None, int(request.args.get("after", 0)),
                request.args.get("level", "INFO"), int(request.args.get("limit", 2000))))
        except (ValueError, _json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/schedule")
    def get_schedule():
        from training import Artifacts

        s, _ = ctx()
        cfg = scheduler.load()
        nxt = scheduler.next_run(cfg)
        runs = [r for r in Artifacts(s, request.args.get("region", "GB")).list_runs(60) if r.get("kind") == "nightly"][:20]
        runs = _mark_interrupted(runs, runner)
        for r in runs:  # keep the payload small
            r["steps"] = [{k: st.get(k) for k in ("step", "status", "seconds", "error", "started")}
                          for st in r.get("steps", [])]
        return jsonify(_clean({
            "config": {k: cfg.get(k) for k in ("enabled", "time", "steps", "backtest_days", "tune_days", "last_started",
                                                "refresh_enabled", "refresh_minutes", "last_refresh")},
            "next_run": nxt.isoformat() if nxt else None,
            "next_refresh": (nr.isoformat() if (nr := scheduler.next_refresh(cfg)) else None),
            "timezone": s.timezone,
            "all_steps": list(STEPS),
            "default_backtest_days": s.get("backtest.days", 60),
            "default_tune_days": s.get("tuning.train_days") or s.get("training.train_days", 540),
            "runs": runs,
        }))

    @app.post("/api/schedule")
    def set_schedule():
        try:
            cfg = scheduler.update(request.get_json(force=True) or {})
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        nxt = scheduler.next_run(cfg)
        return jsonify({"config": cfg, "next_run": nxt.isoformat() if nxt else None})

    @app.post("/api/fetch-now")
    def fetch_now():
        try:
            return jsonify(scheduler.fetch_now())
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/schedule/run-now")
    def schedule_run_now():
        try:
            return jsonify(scheduler.run_now())
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 409

    @app.get("/api/heartbeat")
    def heartbeat():
        """Polled by the dashboard every few seconds: job state plus a data version that changes
        whenever stored data, models or runs change (including from the CLI or the nightly timer)."""
        s, _ = ctx()
        cur = runner.current
        job = None if cur is None else {k: cur.get(k) for k in ("id", "job", "status", "progress", "started", "finished")}
        if job is not None:
            job["trigger"] = (cur.get("args") or {}).get("trigger")
        tc = trader.current
        return jsonify({"job": job, "version": _data_version(s.data_dir),
                        "trading": {"busy": tc is not None, "started": (tc or {}).get("started")}})

    # ── paper trading ────────────────────────────────────────────────────
    @app.get("/api/trading/status")
    def trading_status():
        from trading.paper import status as paper_status

        s, _ = ctx()
        st = paper_status(s)
        st["runner"] = trader.snapshot()
        st["next_cycle"] = trader.next_cycle(st["state"])
        return jsonify(_clean(st))

    @app.post("/api/trading/settings")
    def trading_settings():
        from trading.paper import update_state

        s, _ = ctx()
        try:
            return jsonify(_clean(update_state(s, request.get_json(force=True) or {})))
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/trading/cycle")
    def trading_cycle():
        body = request.get_json(silent=True) or {}
        try:
            if body.get("close"):
                return jsonify(trader.start("manual", force_target=0.0, reason="closed from the dashboard"))
            return jsonify(trader.start("manual"))
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/trading/reset")
    def trading_reset():
        from trading.paper import Ledger

        s, _ = ctx()
        if trader.busy():
            return jsonify({"error": "a trading cycle is running"}), 409
        Ledger(s).reset()
        return jsonify({"reset": True})

    @app.get("/api/trading/chart")
    def trading_chart():
        from trading.paper import Ledger, candles_for_chart

        s, _ = ctx()
        days = int(request.args.get("days", 7))
        c = candles_for_chart(s, days=days)
        ledger = Ledger(s)
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()
        out = {"candles": [] if c.empty else _records(pd.DataFrame({
            "ts": c.index, "bid": c["bid_close"].to_numpy(), "ask": c["ask_close"].to_numpy(),
            "mid": ((c["bid_close"] + c["ask_close"]) / 2).to_numpy()}))}
        out["fills"] = [f for f in ledger.rows("fills", 500) if f["ts"] >= since]
        out["equity"] = [e for e in ledger.rows("equity", 5000) if e["ts"] >= since]
        out["decisions"] = ledger.rows("decisions", 50)
        out["all_fills"] = ledger.rows("fills", 100)
        return jsonify(_clean(out))

    @app.get("/api/trading/backtests")
    def trading_backtests():
        from trading.strategy import list_backtests

        s, _ = ctx()
        return jsonify(_clean(list_backtests(s)))

    @app.get("/api/trading/backtest/<run_id>")
    def trading_backtest(run_id):
        from trading.strategy import load_backtest

        s, _ = ctx()
        b = load_backtest(s, run_id)
        if b is None:
            return jsonify({"error": f"no backtest {run_id}"}), 404
        return jsonify(_clean(b))

    @app.get("/api/jobs")
    def jobs():
        return jsonify(_clean(runner.snapshot()))

    @app.post("/api/jobs/cancel")
    def cancel_job():
        try:
            return jsonify(runner.cancel())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/jobs/<job>")
    def start_job(job):
        try:
            return jsonify(runner.start(job, request.get_json(silent=True) or {}, get_settings()))
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 409

    @app.get("/api/settings")
    def get_editable():
        from models import available_models

        s, _ = ctx()
        from models import MODEL_REGISTRY
        from sources import REGISTRY

        names = s.get("sources") or {}
        return jsonify({"values": {k: s.get(k) for k in EDITABLE},
                        "sources": {n: s.source_enabled(n) for n in names},
                        "help": EDITABLE_HELP,
                        "source_help": {n: REGISTRY[n].description for n in names if n in REGISTRY},
                        "model_help": {n: MODEL_REGISTRY[n].description for n in available_models()},
                        "available_models": available_models()})

    @app.post("/api/settings")
    def set_editable():
        body = request.get_json(force=True) or {}
        changes: dict = {}
        for key, value in (body.get("values") or {}).items():
            if key not in EDITABLE:
                return jsonify({"error": f"{key} is not editable"}), 400
            typ = EDITABLE[key]
            try:
                value = typ(value) if typ is not list else list(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"bad value for {key}"}), 400
            node = changes
            *parents, leaf = key.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        for name, enabled in (body.get("sources") or {}).items():
            changes.setdefault("sources", {}).setdefault(name, {})["enabled"] = bool(enabled)
        path = save_overrides(changes)
        return jsonify({"saved": str(path.name)})

    return app


def _mark_interrupted(runs: list[dict], runner) -> list[dict]:
    """A run record still saying "running" when the dashboard isn't running a pipeline was cut
    short (server restarted, job cancelled or crashed): show it as interrupted."""
    cur = runner.current
    live = cur is not None and cur.get("status") == "running" and cur.get("job") == "nightly"
    live_id = ((cur or {}).get("progress") or {}).get("run_id") if live else None
    out = []
    for r in runs:
        if r.get("status") == "running" and r.get("kind") == "nightly" and not (live and (live_id in (None, r.get("run_id")))):
            r = {**r, "status": "interrupted"}
        out.append(r)
    return out


def _data_version(data_dir) -> float:
    """Cheap change detector: newest mtime across store partitions, model artefacts and runs.
    Store writes go via temp file + rename, which bumps the partition directory's mtime."""
    newest = 0.0
    store = data_dir / "store"
    try:
        for ds in os.scandir(store):
            if ds.is_dir():
                for part in os.scandir(ds.path):
                    newest = max(newest, part.stat().st_mtime)
    except FileNotFoundError:
        pass
    for sub in ("runs", "trading", "trading/backtests"):
        try:
            for f in os.scandir(data_dir / sub):
                newest = max(newest, f.stat().st_mtime)
        except FileNotFoundError:
            pass
    models = data_dir / "models"
    if models.exists():
        for root, _, files in os.walk(models):
            for f in files:
                try:
                    newest = max(newest, os.stat(os.path.join(root, f)).st_mtime)
                except FileNotFoundError:
                    pass
    return newest


def _latest_delivery(store: ParquetStore, region: str):
    p = store.read("predictions", region, columns=["delivery_date"])
    if p.empty:
        return None
    return date.fromisoformat(sorted(p["delivery_date"].unique())[-1])


# gunicorn entry point (run from the project root): gunicorn "web:create_app()"
