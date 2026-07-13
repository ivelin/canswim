# canswim

Developer toolkit for CANSLIM-style investors: **CLI**, Gradio **GUI**, and **MCP** over a local search database.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

Intro: [blog post](https://medium.com/@ivelin.atanasoff.ivanov/canswim-a-deep-learning-tool-for-canslim-practitioners-2c9740bb0d3d) ·
[Austin Python Meetup video](https://www.youtube.com/watch?v=GfC-H0uxXvk&ab_channel=AustinPythonMeetup) ·
[GitHub repo](https://github.com/ivelin/canswim)

## Documentation

| Page | Contents |
|------|----------|
| [CLI](cli.md) | Tasks, recipes, env vars (`python -m canswim -h` is flag SoT) |
| [Gather & forecast](run_triggers.md) | CLI · GUI · MCP shared contract; stocks vs IPOs vs ETFs |
| [MCP](mcp.md) | Tools, read-only default, `MCP_ALLOW_RUNS` |
| [Data store](data_store.md) | Parquet (system of record) vs DuckDB (search/UI) |

This site is built from the `main` branch `docs/` folder. Edit docs on `main`; Pages redeploys automatically.

## Setup

```bash
pip install canswim
# or from a checkout
pip install -e ./
conda activate canswim   # recommended for this repo
```

Local work: keep **`hfhub_sync=False`** (default) unless you intentionally sync Hugging Face data. Full forecast path needs market data APIs (e.g. FMP). Default TiDE checkpoint is public on Hugging Face and can be swapped.

## Get market data & run forecasts

| | Get market data | Run a forecast | Check start date |
|--|-----------------|----------------|------------------|
| **CLI** | `gatherdata --tickers "…"` | `forecast --tickers "…" …` | `resolve_start` |
| **GUI** | **Update market data** | **Run forecast** | **Check start date** |
| **MCP** | `gather_tickers`* | `forecast_tickers`* | `resolve_forecast_start` |

\*MCP write tools need `MCP_ALLOW_RUNS=1`.

```bash
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"
python -m canswim forecast --tickers AAPL --dry_run
python -m canswim dashboard --same_data True
```

Details: [run_triggers.md](run_triggers.md) · [cli.md](cli.md)

## Dashboard

```bash
python -m canswim dashboard --same_data True
```

| Tab | Purpose |
|-----|---------|
| **Charts** | Price + forecast bands |
| **Scans** | Filter by as-of, reward, risk, confidence |
| **Run** | Update market data / Run forecast / Check start date |
| **Advanced** | Read-only SQL |

![Charts](images/charts.png)

![Scans](images/scans.png)

![Run](images/run.png)

## MCP (quick)

```bash
python -m canswim mcp
MCP_ALLOW_RUNS=1 python -m canswim mcp
```

Full guide: [mcp.md](mcp.md)
