"""CLI:  python src/main.py <command> [options]

  sources                      list registered data sources
  probe <source> [--days 30]   fetch + parse a source live and print it (nothing is stored)
  ingest [--source X] [--start YYYY-MM-DD] [--end ...] [--full]
  demo [--days 400]            fill the store with synthetic data (offline testing)
  tune [--force]               Optuna tuning for targets/models that are due
  backtest [--days N] [--models a,b] [--targets a,b]
  train                        fit production (champion) models on latest data
  predict [--date YYYY-MM-DD] [--targets a,b] [--model champion|hgb|...] [--retrain] [--ingest]
                               predict a day (default tomorrow) + buy/sell ranking
  nightly [--skip-ingest] [--force-tune]
  ig-check                     log in to the IG demo account and show the trading market + quote
  strategy [--days N]          gas paper-trading strategy: walk-forward backtest, then train the live model
  trade [--close]              one paper-trading cycle now (or close the paper position)
  web                          run the Flask dashboard (dev server; use gunicorn in prod)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from settings import get_settings
from logging_setup import setup_logging
from compute import configure_threads

# Thread settings must be in place before any ML library is imported (they're imported lazily below).
configure_threads()


def _csv(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="GB")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources")
    s = sub.add_parser("probe", help="download one source's recent data and show it, without storing")
    s.add_argument("source")
    s.add_argument("--days", type=int, default=30)
    s = sub.add_parser("ingest")
    s.add_argument("--source", help="comma-separated source names (default: all enabled)")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--full", action="store_true", help="re-download from history_start")
    s = sub.add_parser("demo")
    s.add_argument("--days", type=int, default=400)
    s = sub.add_parser("tune")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("backtest")
    s.add_argument("--days", type=int)
    s.add_argument("--models")
    s.add_argument("--targets")
    sub.add_parser("train")
    s = sub.add_parser("predict")
    s.add_argument("--date")
    s.add_argument("--targets")
    s.add_argument("--model", default="champion")
    s.add_argument("--retrain", action="store_true")
    s.add_argument("--ingest", action="store_true", help="fetch the latest data first")
    s = sub.add_parser("nightly")
    s.add_argument("--skip-ingest", action="store_true")
    s.add_argument("--force-tune", action="store_true")
    sub.add_parser("ig-check")
    s = sub.add_parser("strategy")
    s.add_argument("--days", type=int, help="backtest window (default trading.backtest_days)")
    s = sub.add_parser("trade")
    s.add_argument("--close", action="store_true", help="close the paper position instead of following the model")
    s = sub.add_parser("web")
    s.add_argument("--port", type=int)
    s.add_argument("--debug", action="store_true")
    a = p.parse_args(argv)

    setup_logging(a.log_level, "web.log" if a.cmd == "web" else "energy_prediction.log")
    settings = get_settings()

    if a.cmd == "sources":
        from sources import list_sources

        for s in list_sources():
            flag = "on " if settings.source_enabled(s["name"]) else "off"
            print(f"[{flag}] {s['name']:<24} {s['mode']:<11} {s['category']:<12} {s['description']}")
        return 0

    if a.cmd == "probe":
        from datetime import timedelta

        import pandas as pd

        from sources import build_source

        src = build_source(a.source, a.region, settings=settings)
        end = date.today()
        start = end - timedelta(days=a.days)
        print(f"probing {a.source} {start}..{end}")
        df = src.validate(src.transform(src.fetch(start, end)))
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(f"{len(df)} rows; columns: {list(df.columns)}")
            print(df.head(5).to_string())
            print("...")
            print(df.tail(5).to_string())
        return 0 if len(df) else 1

    if a.cmd == "ingest":
        from sources import ingest_all

        res = ingest_all(a.region, _csv(a.source), a.start, a.end, a.full, settings)
        print(json.dumps(res, indent=2))
        return 0 if all(r["status"] != "error" for r in res) else 1

    if a.cmd == "demo":
        from sources.synthetic import populate
        from storage import ParquetStore

        print(populate(ParquetStore(settings.data_dir), a.region, a.days))
        return 0

    if a.cmd == "tune":
        from pipeline import run_tuning
        from storage import ParquetStore
        from training import Artifacts

        res = run_tuning(settings, ParquetStore(settings.data_dir), a.region, Artifacts(settings, a.region), a.force)
        print(json.dumps(res, indent=2, default=str))
        return 0

    if a.cmd == "backtest":
        from backtest import Backtester

        res = Backtester(settings, region=a.region).run(_csv(a.targets), _csv(a.models), days=a.days)
        print(json.dumps({k: res[k] for k in ("run_id", "start", "end", "champions")}, indent=2))
        for row in res["leaderboard"]:
            print(f"  {row['target']:<14} {row['model']:<16} MAE={row.get('mae', float('nan')):.3f}")
        return 0

    if a.cmd == "train":
        from prediction import Predictor

        print(json.dumps(Predictor(settings, region=a.region).train_champions(), indent=2, default=str))
        return 0

    if a.cmd == "predict":
        from prediction import Predictor

        if a.ingest:
            from sources import ingest_all

            ingest_all(a.region, settings=settings)
        pr = Predictor(settings, region=a.region)
        day = date.fromisoformat(a.date) if a.date else None
        preds = pr.predict_day(day, _csv(a.targets), a.model, a.retrain)
        for w in pr.warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(json.dumps(pr.recommend(preds if len(preds) else None, day), indent=2, default=str))
        return 0

    if a.cmd == "nightly":
        from pipeline import run_nightly

        rec = run_nightly(settings, a.region, a.skip_ingest, a.force_tune)
        print(json.dumps({k: rec[k] for k in ("run_id", "status", "champions")}, indent=2))
        return 0 if rec["status"] == "ok" else 1

    if a.cmd == "ig-check":
        import os

        from settings import DOTENV_KEYS
        from trading.ig import IGClient, IGError
        from trading.market import resolve_market

        def masked(k: str) -> str:
            v = os.environ.get(k, "")
            if not v:
                return "(not set)"
            shown = f"{v[:2]}{'*' * max(0, len(v) - 4)}{v[-2:]}" if k != "IG_PASSWORD" else "*" * len(v)
            return f"{shown}  [{len(v)} chars, from {'.env' if k in DOTENV_KEYS else 'the shell environment'}]"

        print("IG demo check (https://demo-api.ig.com)")
        for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
            print(f"  {k:14} {masked(k)}")
        client = IGClient()
        try:
            acct = client.login()
        except (IGError, OSError) as exc:  # OSError covers requests' connection errors
            print(f"\nlogin failed: {exc}", file=sys.stderr)
            return 1
        market = resolve_market(settings, client, refresh=True)
        quote = client.quote(market["epic"])
        print(json.dumps({"account": acct, "market": market, "quote": quote,
                          "history_allowance": client.last_allowance}, indent=2, default=str))
        return 0

    if a.cmd == "strategy":
        from pipeline.nightly import run_strategy

        print(json.dumps(run_strategy(settings, a.region, backtest_days=a.days), indent=2, default=str))
        return 0

    if a.cmd == "trade":
        from trading.paper import run_cycle

        res = run_cycle(settings, a.region, force_target=0.0 if a.close else None,
                        reason="closed from the command line" if a.close else "")
        print(json.dumps(res, indent=2, default=str))
        return 0

    if a.cmd == "web":
        from web import create_app

        if a.debug:
            import os

            os.environ["ENERGY_WEB_RELOADER"] = "1"  # tells create_app which process may run the schedule
        app = create_app()
        app.run(host=settings.get("web.host", "0.0.0.0"), port=a.port or settings.get("web.port", 5050), debug=a.debug)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
