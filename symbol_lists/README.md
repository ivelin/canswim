# Symbol list CSVs (checked in)

Light reference ticker universes used by gather / train / forecast.
Historical market data (parquet) lives under `data/` (gitignored) and is refreshed via FMP/yfinance.

Source of truth on HF (optional CSV-only sync): `ivelin/canswim` dataset `data-3rd-party/*.csv`.

Do **not** store multi-GB price history here.
