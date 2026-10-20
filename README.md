# EnergyPrediction

Energy data aggregation, model training and comparison, and backtesting for day-ahead
**demand** and **price** forecasts. It starts with GB, and the code takes the region as a
parameter so more markets can be added. There is no trading: the "buy/sell" output is a
relative-value ranking for analysis only.

```
 data sources ──► Parquet store ──► feature builder ──► models ──► walk-forward backtest
 (one class each,  (DuckDB for SQL)   (leak-free lags,    (LightGBM,   │
  shared ABC)                          weather, calendar)  PyTorch,    ▼
                                                           sklearn) champion per target ──► next-day
                                                                                            prediction
 daily schedule (dashboard) runs: ingest → tune (Optuna) → backtest → train → predict → rank ──► Flask dashboard
```

## Quick start (Mac)

```bash
conda activate EnergyPrediction          # always activate before pip installs
conda install -c conda-forge lightgbm    # conda's build bundles OpenMP on macOS
pip install -r requirements.txt

python src/main.py sources      # list every data source and whether it's on
python src/main.py demo         # synthetic data, so everything runs offline
python src/main.py nightly --skip-ingest
python src/main.py web          # http://localhost:5050
pytest
```

With real data:

```bash
python src/main.py ingest --full          # backfill from history_start (takes a while)
python src/main.py nightly --force-tune
```

Delete `data/store` before switching from demo data to real data.

## Data sources

Every source subclasses `sources.base.DataSource` and implements only
`fetch(start, end)` (raw download) and `transform(raw)` (tidy frame with a UTC `timestamp`).
The base class handles everything else: chunking to the API's window limit, retries,
validation, de-duplication, incremental resume with a revision look-back, and storage.
Subclasses register themselves automatically.

| Stream | Source class | Feed |
|---|---|---|
| Historical prices | `ElexonMarketIndexPrices`, `ElexonSystemPrices` | Elexon BMRS (live) |
| Energy usage | `ElexonDemandOutturn` | Elexon BMRS (live) |
| Grid metrics | `ElexonGenerationByFuel`, `CarbonIntensity` | Elexon FUELHH, NESO carbon API (live) |
| Weather | `OpenMeteoWeather` | Open-Meteo ERA5 archive + forecast, 8 GB points (live) |
| Calendar events | `UKBankHolidays` | gov.uk (live) |
| Weather events | `WeatherEvents` | Named storms per impact day, scraped from the Met Office storm centre season pages |
| Gas prices | `GasPrices` | GB System Average Price of gas (p/kWh and p/therm, daily) from National Gas's data API, falling back to the ONS SAP spreadsheets |
| Carbon allowances | `CarbonAllowances` | EU allowances from EEX auction results (EUR/t, auction days); UK allowances from the GOV.UK UK ETS monthly average (GBP/t); EUR converted with ECB reference rates. Includes a monthly UK-minus-EU spread |
| Gas CFD quotes (paper trading) | `IGGasPrices` | IG demo API: hourly bid/ask candles for UK natural gas (needs an IG demo login in `.env`) |
| Renewable certificates | `OfgemCertificates` | REGOs and ROCs issued per month by technology (wind, solar, hydro, bio, other) from Ofgem's Renewable Electricity Register summary reports. Only the small "by Technology and Month of Issue" reports are downloaded and only monthly totals stored. Each month counts from the 1st of the next month |
| Certificate prices | `CertificatePrices` | Manual CSV: Ofgem publishes REGO volumes, not prices |

Gas, carbon and storm data are also available ahead of time or with a publication lag, and
the feature builder accounts for that: daily commodity prices are lagged two days, UK
monthly carbon averages are only used from the 15th of the following month, and named
storms count from the day they hit, since they're named in advance.

Test a source against the live site without storing anything:
`python src/main.py probe gas_prices` (or `carbon_allowances`, `weather_events`, …).

See `src/sources/manual_data.md` for the CSV formats (the CSVs go in `data/manual/`). To automate a manual source, override
its `fetch`/`transform` and keep the same output columns.

Adding a source:

```python
class MySource(DataSource):
    name = "my_source"; category = "prices"; max_days_per_request = 7
    def fetch(self, start, end): return self.http_get_json(URL, {...})
    def transform(self, raw): return pd.DataFrame({"timestamp": ..., "my_value": ...})
```

Then add `my_source: {enabled: true}` under `sources:` in `src/config/settings.yaml`.

## ML

* **Models** (`models/zoo.py`, all share the `Forecaster` ABC):
  `seasonal_naive` (baseline), `linear` (ridge regression), `hgb` (sklearn), `lightgbm`, `torch_mlp` (PyTorch).
  Gradient boosting is the workhorse for tabular energy data. PyTorch is there as the
  deep-learning comparison, and sequence models (LSTM/TFT) can be added as new subclasses.
* **Features**: weather and calendar at time *t*. Realised series (prices, demand,
  generation) only enter as lags of at least `features.min_lag` (96 periods = 2 days), so
  every input is known when tomorrow is predicted.
* **Tuning**: Optuna, on every pipeline run by default (`tuning.retune_every_days: 0`).
  Each target/model pair keeps one study across nights, so the TPE sampler learns from every
  earlier trial and proposes values near the ones that worked. Once a study has
  `tuning.warm_after` trials, each night runs only `tuning.warm_trials` (default 15) instead of
  `tuning.n_trials`, starting with the current best and the study's top 3 re-scored on today's
  data. Every `tuning.reset_every_days` (default 7) a fresh study starts, seeded with the current
  best, so trials scored on old data stop steering the search. The whole step is capped at
  `tuning.max_minutes`, and trials train on the last `tuning.train_days`. Every trial is logged
  (Tune step log on the Schedule tab).
* **Backtest**: walk-forward over the last `backtest.days`, retraining every
  `retrain_every_days`. It reports MAE, RMSE, sMAPE, bias and R², plus
  intraday ranking metrics: the share of the actual cheapest/dearest half-hours that were
  predicted, and whether the day-on-day direction was right. The best model per target
  becomes the champion.
* **Ensemble**: after each backtest the top `ensemble.max_members` models per target are blended
  as a weighted average. The weights are non-negative, sum to 1, and minimise MAE on the models'
  saved backtest predictions, so nothing extra is trained. The weights are fitted on the first half
  of the window and the blend is scored on the second half, alongside every single model on the
  same periods (the leaderboard's "MAE 2ND HALF"). The blend becomes champion only if it beats the
  best single model there. Production then uses weights refitted on the whole window and trains
  each member.
* **Prediction**: the champion is refit on the latest data and predicts tomorrow's 48
  half-hours. The ranking lists each price product's expected move against its 7-day
  mean, scaled by backtest MAE, plus the cheapest and dearest half-hours.

## Dashboard

`python src/main.py web` gives a Flask app in the same dark bento theme as
BitcoinPolymarket, with these tabs:

* **Overview**: forecast against actuals, champions, relative value, best half-hours, and
  job buttons for ingest, tune, backtest, train, predict and nightly.
* **Predict**: pick a delivery date, targets and a model (the champion or any specific
  model), then run. By default it fetches the latest data first (incremental, a few
  seconds), and it can optionally retrain. Results are charted against actuals (all models
  that predicted that day are overlaid), with a summary and a history you can click back
  through. The **Previous predictions** toggle overlays earlier predictions for the last
  3–30 days (dashed) with their error against actuals, preferring forecasts that were
  issued before each day began. Days only predicted afterwards are counted as hindsight.
* **Backtest**: run a walk-forward backtest with your own test window (last day + number of
  days), retrain frequency, training window, targets and models, and choose whether it
  updates the champions. Shows live progress; the leaderboard (and model-vs-actual chart)
  records each run's settings so runs can be compared.
* **Schedule**: run the daily pipeline at a set time (UK time) with START/STOP. You choose
  the steps (ingest → tune → backtest → train → predict, always in that order) and the
  backtest length. It shows the run in progress step by step and a history of runs. Click
  a step to see its log underneath (live while it runs, with a warnings-only filter), or
  click a row in the history to look at an older run. Step logs are kept in
  `data/runs/logs/<run_id>.jsonl`; runs from before that are read from `logs/jobs.log`.
  **RUN NOW** runs it straight away. It runs inside the dashboard server, so the server
  must be running.
* **Data**: source freshness and a series explorer.
* **Runs**: nightly pipeline history.
* **Config**: edit default training and backtest parameters and turn sources on or off. Changes
  are written to `src/config/overrides.yaml`.

Jobs started from the dashboard run in a separate worker process, one at a time. The
header shows what's running, and a CANCEL button stops it.

## Speed

* Ingest downloads sources concurrently, each source fetches its API windows in parallel,
  and weather points are fetched in parallel. A full backfill is several times faster;
  a daily incremental update takes a few seconds.
* Training builds the feature matrix once for all targets. Models that early-stop on their
  own hold-out data (hgb, torch_mlp) are fitted once rather than fit-then-refit.
* Backtests run target/model combinations in parallel worker processes (`backtest.workers`,
  `auto` = about one per two CPU cores), each with its share of threads.
* hgb defaults to squared-error loss, which is quicker than absolute error; tuning can still
  choose absolute error.
* Each model is tuned, trained and used for prediction in its own short-lived process
  (`compute.isolate_models`). LightGBM and PyTorch each bring an OpenMP runtime, and with
  both loaded in one process PyTorch can hang. Logs from those processes still go to the
  job log and the step logs. A tuning run that hangs is stopped after 1.5× its time limit plus 5 minutes.
* LightGBM is usually the fastest model. If it reports a libomp error on macOS, run
  `conda install -c conda-forge llvm-openmp` in the env.

## Paper trading (UK natural gas CFD, IG demo)

Simulation only: no real money. The only broker connection is IG's **demo** gateway, and the
code has no setting that points it anywhere else.

* **Setup**: create an IG demo account, generate an API key while logged in to it (My IG >
  Settings > API keys), and put `IG_API_KEY`, `IG_USERNAME` and `IG_PASSWORD` in `.env` (see
  `src/.env.example`; `.env` lives in the repo root). `python src/main.py ig-check` logs in and shows the market it found ("UK Natural
  Gas" by default; pin another with `trading.epic`), its value per point and a live quote.
* **Data**: the `ig_gas` source downloads hourly bid/ask candles: on the first run the latest
  `trading.history_points` candles (one request), afterwards only what's new since the last
  candle. IG limits requests per minute (~30 on demo) and candles per week (~10,000 on demo), so all
  IG calls on the machine are spaced 2 seconds apart, and only one download runs at a time.
* **Model** (`trading/strategy.py`): predicts the mid-price move over the next
  `trading.horizon_hours` candles from recent returns, volatility, spread, time of day, and slow
  GB fundamentals as known at the time (gas SAP and EU carbon, both lagged two days, and the last
  24h of temperature and wind). Any model from the zoo can be used (`trading.model`).
* **Rule**: go long or short `size` contracts when the predicted move is larger than
  `threshold_k` × the typical spread. With a weaker signal it keeps a position that agrees with
  the signal and closes one that doesn't.
* **Backtest**: walk-forward over `trading.backtest_days`, refitting every `retrain_every_days`.
  It decides at each candle close and fills at the next candle's open (buy at the ask, sell at
  the bid), and charges overnight funding at each 22:00 UK cut-off. It reports P&L, Sharpe,
  drawdown, win rate, direction hit rate, spread and funding paid, and buy & hold over the same
  window. It runs as the STRATEGY step of the daily pipeline, or with BACKTEST + TRAIN on the Trading
  tab, or with `python src/main.py strategy`.
* **Live paper trading** (Trading tab): every N minutes on weekdays it fetches new candles and a live
  quote, predicts, and trades a paper account (`data/trading/paper.db`) at the live bid/offer.
  With **Mirror trades to IG demo** switched on, each trade is also sent as a market order to the
  IG demo account, and the demo fill is stored next to the paper fill. Cycles run separately from the
  pipeline jobs, so a long nightly run doesn't hold them up. `python src/main.py trade [--close]`
  runs one cycle from the command line.

## Deployment

* `.github/workflows/ci.yml` runs lint and tests on every push.
* `.github/workflows/deploy.yml` runs after CI passes on `main`. Its job runs on the server
  through a self-hosted runner and calls `src/deploy/deploy.sh`, which syncs the code, updates
  the conda env when dependencies change, installs the systemd units and restarts
  `energy-prediction-web`.
* `src/deploy/systemd/` holds the web service, plus an optional nightly service and timer for running without the dashboard
  (02:30 Europe/London).

One-off server setup is in `src/deploy/README.md`.

## Layout

```
src/                              all the code (on the Python path; run as python src/main.py)
  main.py                         CLI entry point: python src/main.py <command>
  settings.py                     loads config/settings.yaml (+ overrides) and the root .env
  config/settings.yaml            all tunables (overrides.yaml layered on top, git-ignored)
  sources/base.py                 DataSource ABC (+ ManualCsvSource)
  sources/*.py                    one class per stream (elexon, weather, events, ...)
  storage/parquet_store.py        Parquet partitions + DuckDB SQL
  features/builder.py             leak-free feature matrix
  models/                         Forecaster ABC, model zoo, ensemble
  training/                       fitting, Optuna tuning, artefact paths
  backtest/                       walk-forward backtester
  evaluation.py                   metrics
  prediction/                     next-day prediction + relative-value ranking
  pipeline/                       daily pipeline orchestration, step logs
  trading/                        IG demo client, gas strategy + backtest, paper trader
  web/                            Flask app, templates, static
  deploy/                         deploy script, systemd units, server setup
  .env.example                    template for the root .env (secrets)
tests/                            pytest suite
README.md, requirements.txt, environment.yml, pyproject.toml, .github/, .gitignore

Not in git (created when it runs): data/ (store, models, runs, trading ledger, manual CSVs),
logs/, .env
```
