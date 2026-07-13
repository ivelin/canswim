# MCP server

Expose precomputed TiDE forecasts and local market data to MCP clients (Claude Desktop, Cursor, etc.) over the **same DuckDB search database** used by the dashboard.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Behavior

| Mode | How | Tools |
|------|-----|--------|
| **Read-only (default)** | `python -m canswim mcp` | health, list, forecast/scan/price queries, `run_select` (SELECT only) |
| **Runs allowed** | `MCP_ALLOW_RUNS=1` (or `CANSWIM_ALLOW_RUNS=1`) | also `gather_tickers`, `forecast_tickers`, `refresh_tickers` |

CLI `--tickers` and the dashboard **Run** tab do **not** need `MCP_ALLOW_RUNS`. Write tools share the same backend as CLI/GUI: [run_triggers.md](run_triggers.md).

## Prerequisites

1. Local parquet under `data/data-3rd-party/` and forecasts under `data/forecast/` (or your `data_dir`).
2. A search DuckDB built at least once:

```bash
python -m canswim dashboard --same_data True
# first time: omit --same_data (or False) to (re)build from parquet
# or: MCP_INIT_DB=1 when starting MCP
```

Paths: `.env` / env `data_dir`, `db_file` — same as the dashboard. See [data_store.md](data_store.md).

## Run

```bash
python -m canswim mcp
# equivalents: canswim-mcp / python -m canswim.mcp

MCP_ALLOW_RUNS=1 python -m canswim mcp
```

## Example client config (stdio)

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

Add `"MCP_ALLOW_RUNS": "1"` under `env` only if the client should be allowed to gather/forecast.

## Tools

Canonical registration: `src/canswim/mcp/server.py`. Update this table in the **same PR** as any tool add/rename/remove.

| Tool | Description | Runs gate |
|------|-------------|-----------|
| `health_check` | DB path / readiness | — |
| `get_server_info` | Version, read-only / runs_allowed, tool list | — |
| `list_tickers` | Symbols in search DB | — |
| `get_forecast` | Quantile forecast rows for a symbol | — |
| `get_reward_risk` | Reward/risk for a forecast (confidence 80/95/99) | — |
| `scan_forecasts` | Universe scan (≡ dashboard Scans) | — |
| `get_close_price` | Historical closes | — |
| `get_backtest_error` | Forecast vs actual error (mean abs log-error) | — |
| `run_select` | Single `SELECT` only (≡ Advanced Queries) | — |
| `resolve_forecast_start` | Preview week-aligned start (≡ CLI `resolve_start`) | — |
| `gather_tickers` | Scoped gather (≡ `gatherdata --tickers`) | `MCP_ALLOW_RUNS=1` |
| `forecast_tickers` | Scoped forecast; blank start = monthly catch-up + live | `MCP_ALLOW_RUNS=1` |
| `refresh_tickers` | Gather + catch-up forecast (≡ dashboard **Refresh symbols**) | `MCP_ALLOW_RUNS=1` |

## Related docs

- [cli.md](cli.md) — CLI recipes
- [run_triggers.md](run_triggers.md) — gather/forecast policy
- [data_store.md](data_store.md) — parquet vs DuckDB
