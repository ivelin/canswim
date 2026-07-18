# Command-line interface

Operator recipes for `python -m canswim <task>`.

**Flag details:** run `python -m canswim -h` — argparse help in `src/canswim/__main__.py` is the source of truth for flags. This page covers workflows and when to use each task.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Install

```bash
# package
pip install canswim

# dev checkout
pip install -e ./

# recommended: project conda env (used by ./scripts/ci-local.sh)
conda activate canswim
```

Local work should keep **`hfhub_sync=False`** (default) unless you intentionally sync Hugging Face data.

## Task map

| Task | Purpose | Common flags |
|------|---------|----------------|
| `dashboard` | Gradio UI: Charts, Scans, Run, Advanced | `--same_data True` to reuse DuckDB |
| `gatherdata` | Get market data (full universe or scoped) | `--tickers`, `--no_covariates` |
| `forecast` | Run forecasts (full or scoped) | `--tickers`, `--forecast_start_date`, `--dry_run` |
| `resolve_start` | Print which forecast start date would be used | `--forecast_start_date` |
| `mcp` | MCP server for clients | env `MCP_ALLOW_RUNS=1` for writes; `--http --host --port` for Streamable HTTP |
| `train` | Continuous model training (full history) | `--new_model` |
| `modelsearch` | Hyperparameter search | — |
| `downloaddata` | Download train/forecast data from HF Hub | needs `hfhub_sync` / tokens as configured |
| `uploaddata` | Upload local train/forecast data to HF Hub | same |

```bash
python -m canswim -h
```

## Day-to-day path (short list of symbols)

Same backend as the dashboard **Run** tab and MCP write tools. Policy detail: [run_triggers.md](run_triggers.md). Storage: [data_store.md](data_store.md).

```bash
# 1) Get market data (~3y, missing-only; includes fundamentals unless --no_covariates)
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"

# 2) Check start date (optional)
python -m canswim resolve_start
python -m canswim resolve_start --forecast_start_date 2026-03-05

# 3) Dry-run forecast (symbols + start only; no model)
python -m canswim forecast --tickers AAPL --forecast_start_date 2026-03-05 --dry_run

# 4) Forecast
python -m canswim forecast --tickers "AAPL,MSFT" --forecast_start_date 2026-03-05

# 5) Open UI (reuse search DB after first build)
python -m canswim dashboard --same_data True
```

Without `--tickers`, `gatherdata` / `forecast` keep **full-universe / train-style** behavior (longer history, not the lean 2y scoped path).

### Flags (scoped gather / forecast)

| Flag | Tasks | Meaning |
|------|-------|---------|
| `--tickers "AAPL, MSFT"` | `gatherdata`, `forecast` | Scoped run via `run_triggers` (≤50 symbols) |
| `--forecast_start_date YYYY-MM-DD` | `forecast`, `resolve_start` | Origin date; week-aligned per [run_triggers.md](run_triggers.md) |
| `--dry_run` | `forecast --tickers` | Validate only (no torch) |
| `--no_covariates` | `gatherdata --tickers` | Prices (+ broad market path) only; skip earnings, ownership, etc. |
| `--same_data True` | `dashboard` | Reuse DuckDB search DB (faster start) |
| `--new_model True` | `train` | Train a new model instead of continuing |
| `--http` | `mcp` | Streamable HTTP (gateway) instead of stdio |
| `--transport …` | `mcp` | `stdio` / `streamable-http` / `http` / `sse` |
| `--host` / `--port` | `mcp` | Bind for HTTP/SSE transports |

## Environment variables (common)

| Env | Default | Meaning |
|-----|---------|---------|
| `hfhub_sync` | `False` | Full HF dataset/model sync off |
| `SYNC_SYMBOL_LISTS` | `False` | If `True`, fetch light symbol CSVs from HF once |
| `YFINANCE_USE_CACHE` | `False` | Avoid multi‑GB yfinance SQLite cache hang |
| `MCP_ALLOW_RUNS` / `CANSWIM_ALLOW_RUNS` | unset | Enable MCP gather/forecast tools only |
| `MCP_INIT_DB` | unset | If set, MCP may build DuckDB from parquet on start |
| `CANSWIM_MCP_TRANSPORT` / `MCP_TRANSPORT` | `stdio` | MCP transport when flags omitted |
| `CANSWIM_MCP_HOST` / `MCP_HOST` | `127.0.0.1` | HTTP/SSE bind host |
| `CANSWIM_MCP_PORT` / `MCP_PORT` | `8000` | HTTP/SSE bind port |
| `data_dir` | `data` | Data root |
| `db_file` | (dashboard/MCP defaults) | DuckDB filename under `data_dir` |
| `stock_tickers_list` | list CSV name | Universe for full gather/forecast when not using `--tickers` |
| `LOG_LEVEL` / `LOGURU_LEVEL` | `INFO` | Logging |

## Failure patterns (scoped forecast)

| Situation | What to do |
|-----------|------------|
| Incomplete OHLCV history | `gatherdata --tickers "…"` then retry forecast |
| Prices OK but covariates fail | Gather again **without** `--no_covariates` (includes ownership, estimates, …) |
| Forecast already on file for that start | Skipped on purpose; delete partition or use another start to re-run |
| Start after last local bar | Update market data or pick an earlier start |

Exact copy and policy: [run_triggers.md](run_triggers.md).

## Infrequent / heavy tasks

- **`train`** — full-history training; not the same as scoped `forecast --tickers`.
- **`modelsearch`** — hyperparameter search; long-running.
- **`downloaddata` / `uploaddata`** — HF Hub sync of train/forecast artifacts; use only when you mean to sync the cloud dataset.

## Related docs

- [run_triggers.md](run_triggers.md) — gather/forecast contract (CLI · GUI · MCP)
- [mcp.md](mcp.md) — MCP server
- [data_store.md](data_store.md) — parquet vs DuckDB
- [Home](index.md) — landing page
