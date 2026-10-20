# Manual data drops

Sources without a free price feed (currently only REGO certificate prices) read `<source_name>.csv` from `data/manual/` in the repo root (create it if needed; it is git-ignored). Each needs a
`timestamp` (or `date`) column plus numeric value columns. Times without a timezone are
treated as UTC.

| File | Columns |
|---|---|
| `certificate_prices.csv` | date, rego_price (GBP/MWh) |

Turn the source on under `sources:` in `config/settings.yaml` (or on the dashboard's
CONFIG tab), then run `python src/main.py ingest --source <name>`.
