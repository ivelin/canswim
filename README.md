# canswim
Developer toolkit for CANSLIM investment style practitioners

For a brief introduction read [this blog post](https://medium.com/@ivelin.atanasoff.ivanov/canswim-a-deep-learning-tool-for-canslim-practitioners-2c9740bb0d3d).

[Here is also a video recording](https://www.youtube.com/watch?v=GfC-H0uxXvk&ab_channel=AustinPythonMeetup) of a CANSWIM presentation for the [Python Austin Meetup](https://www.meetup.com/austinpython/).

Example forecast chart below:

![image](https://github.com/user-attachments/assets/7235004c-e92e-4731-85bf-03d67c12b2d9)


# Setup


```
pip install canswim
```


## Install canswim package in dev mode

```
pip install -e ./
```

## Local-first market data (gather)

By default **`gatherdata` does not download or upload the full Hugging Face dataset**.
That HF snapshot step is slow and was a common reason the CLI “hung”. Instead:

1. Use **local** parquet under `data/data-3rd-party/` (created/updated by gather).
2. Refresh from **FMP / yfinance** APIs as needed.
3. Resolve ticker universes from checked-in **`symbol_lists/*.csv`** (light reference files).

```bash
# full-universe local gather (no HF dataset sync)
hfhub_sync=False python -m canswim gatherdata

# scoped list via env (legacy full gather path)
stock_tickers_list=few_stocks.csv hfhub_sync=False python -m canswim gatherdata

# scoped list via --tickers (same pipeline as Dashboard Run + MCP gather_tickers)
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"
```

Optional HF:

| Env | Default | Meaning |
|-----|---------|---------|
| `hfhub_sync` | `False` | Full dataset/model sync off |
| `SYNC_SYMBOL_LISTS` | `False` | If `True`, fetch only light CSVs from the HF dataset once |
| `YFINANCE_USE_CACHE` | `False` | Avoid multi‑GB SQLite cache hang |
| `MCP_ALLOW_RUNS` | unset | Enable MCP gather/forecast tools (CLI/GUI do not need this) |

Train/forecast **skip tickers without complete ground-truth OHLCV** (no synthetic price fill).

## Gather & forecast: one contract (CLI · GUI · MCP)

Scoped runs share **`canswim.run_triggers`** (tickers + week-aligned start). Full detail: **[docs/run_triggers.md](docs/run_triggers.md)**.

| Surface | Gather | Forecast | Preview start |
|---------|--------|----------|---------------|
| **CLI** | `gatherdata --tickers "…"` | `forecast --tickers "…" [--forecast_start_date] [--dry_run]` | `resolve_start [--forecast_start_date]` |
| **GUI** | Dashboard → **Run** → Gather | **Run** → Forecast | **Preview start date** |
| **MCP** | `gather_tickers` (`MCP_ALLOW_RUNS=1`) | `forecast_tickers` (`MCP_ALLOW_RUNS=1`) | `resolve_forecast_start` (always) |

**Start-date policy (all three):** past dates → first NYSE session of that market week (holiday Monday → next open that week); empty/today → live origin after latest week-end close; no pure-future origins.

```bash
# examples
python -m canswim resolve_start --forecast_start_date 2026-03-05
python -m canswim forecast --tickers AAPL --forecast_start_date 2026-03-05 --dry_run
python -m canswim forecast --tickers "AAPL,MSFT" --forecast_start_date 2026-03-05
```

Without `--tickers`, `gatherdata` / `forecast` keep legacy full-universe behavior.

## Command line interface

```bash
python -m canswim -h
```

Main tasks: `dashboard`, `gatherdata`, `forecast`, `train`, `modelsearch`, `downloaddata`, `uploaddata`, `mcp`, `resolve_start`.

Useful flags:

| Flag | Used by | Meaning |
|------|---------|---------|
| `--tickers` | `gatherdata`, `forecast` | Scoped run via shared orchestration |
| `--forecast_start_date` | `forecast`, `resolve_start` | Origin date (week-aligned when using `--tickers` / resolve) |
| `--dry_run` | `forecast --tickers` | Resolve start + validate only |
| `--no_covariates` | `gatherdata --tickers` | Prices (+ broad market) only |
| `--same_data` | `dashboard` | Reuse DuckDB search DB |
| `--new_model` | `train` | Fresh model vs continue |

## MCP server (read-only by default)

Expose precomputed TiDE forecasts and local market data to MCP clients (Claude Desktop, Cursor, etc.) over the **same DuckDB search database** used by the dashboard.

By default the process stays **read-only**. Write tools are listed for discoverability but only execute when `MCP_ALLOW_RUNS=1` (or `CANSWIM_ALLOW_RUNS=1`). CLI `--tickers` and the Dashboard **Run** tab do **not** use that flag.

### Prerequisites

1. Local data + forecasts (e.g. `python -m canswim downloaddata` / `forecast` as usual).
2. Build the search DB once via the dashboard (or set `MCP_INIT_DB=1`):

```bash
python -m canswim dashboard --same_data True
# or first run without --same_data to (re)build DuckDB from parquet
```

### Run

```bash
python -m canswim mcp
# or canswim-mcp / python -m canswim.mcp

# enable gather/forecast tools
MCP_ALLOW_RUNS=1 python -m canswim mcp
```

Configure paths via `.env` (`data_dir`, `db_file`) — same as the dashboard.

### Example client config (stdio)

```json
{
  "mcpServers": {
    "canswim": {
      "command": "python",
      "args": ["-m", "canswim", "mcp"],
      "cwd": "/path/to/canswim",
      "env": {
        "data_dir": "data",
        "db_file": "canswim_local.duckdb"
      }
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `health_check` | DB path / readiness |
| `get_server_info` | Version, read-only / runs_allowed flags, tool list |
| `list_tickers` | Symbols in search DB |
| `get_forecast` | Quantile forecast rows for a symbol |
| `get_reward_risk` | Reward/risk for latest forecast (confidence 80/95/99) |
| `scan_forecasts` | Universe scan (dashboard Scans tab) |
| `get_close_price` | Historical closes |
| `get_backtest_error` | Forecast vs actual error |
| `run_select` | Single SELECT only (Advanced Queries) |
| `resolve_forecast_start` | Preview week-aligned start (≡ CLI `resolve_start`) |
| `gather_tickers` | Scoped gather (≡ `gatherdata --tickers`; needs `MCP_ALLOW_RUNS=1`) |
| `forecast_tickers` | Scoped forecast (≡ `forecast --tickers`; needs `MCP_ALLOW_RUNS=1`) |

### Dashboard Run tab

Paste tickers → **Gather data** / **Run forecast** / **Preview start date**. Same `run_triggers` path as CLI `--tickers` and MCP write tools (see [docs/run_triggers.md](docs/run_triggers.md)).

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**
