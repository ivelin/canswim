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
# typical local gather (no HF dataset sync)
hfhub_sync=False python -m canswim gatherdata

# optional: only a few symbols
stock_tickers_list=few_stocks.csv hfhub_sync=False python -m canswim gatherdata
```

Optional HF:

| Env | Default | Meaning |
|-----|---------|---------|
| `hfhub_sync` | `False` | Full dataset/model sync off |
| `SYNC_SYMBOL_LISTS` | `False` | If `True`, fetch only light CSVs from the HF dataset once |
| `YFINANCE_USE_CACHE` | `False` | Avoid multi‑GB SQLite cache hang |

Train/forecast **skip tickers without complete ground-truth OHLCV** (no synthetic price fill).

## Command line interface

```
$  python -m canswim -h
usage: canswim [-h] [--forecast_start_date FORECAST_START_DATE] [--new_model NEW_MODEL] [--same_data SAME_DATA] {dashboard,gatherdata,downloaddata,uploaddata,modelsearch,train,forecast}

CANSWIM is a toolkit for CANSLIM style investors. Aims to complement the Simple Moving Average and other technical indicators.

positional arguments:
  {dashboard,gatherdata,downloaddata,uploaddata,modelsearch,train,forecast}
                        Which canswim task to run: `dashboard` for stock charting and scans of recorded forecasts. 'gatherdata` to gather 3rd party stock market data and save to HF Hub. 'downloaddata` download model training and forecast
                        data from HF Hub to local data storage. 'uploaddata` upload to HF Hub any interim changes to local train and forecast data. `modelsearch` to find and save optimal hyperparameters for model training. `train` for
                        continuous model training. `forecast` to run forecast on stocks and upload dataset to HF Hub.

options:
  -h, --help            show this help message and exit
  --forecast_start_date FORECAST_START_DATE
                        Optional argument for the `forecast` task. Indicate forecast start date in YYYY-MM-DD format. If not specified, forecast will start from the end of the target series.
  --new_model NEW_MODEL
                        Optional argument for the `train` task. Whether to train a newly created model or continue training an existing pre-trained model.
  --same_data SAME_DATA
                        Optional argument for the `dashboard` task. Whether to reuse previously created search database (faster start time) or update with new forecast data (slower start time).

NOTE: NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.
```

## MCP server (read-only)

Expose precomputed TiDE forecasts and local market data to MCP clients (Claude Desktop, Cursor, etc.) over the **same DuckDB search database** used by the dashboard. The MCP process does **not** load the torch model and does not run train/gather/forecast.

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
# or
canswim-mcp
# or
python -m canswim.mcp
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
| `get_server_info` | Version, read-only flag, tool list |
| `list_tickers` | Symbols in search DB |
| `get_forecast` | Quantile forecast rows for a symbol |
| `get_reward_risk` | Reward/risk for latest forecast (confidence 80/95/99) |
| `scan_forecasts` | Universe scan (dashboard Scans tab) |
| `get_close_price` | Historical closes |
| `get_backtest_error` | Forecast vs actual error |
| `run_select` | Single SELECT only (Advanced Queries) |

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**
