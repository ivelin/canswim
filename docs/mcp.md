# MCP server

Expose precomputed TiDE forecasts and local market data to MCP clients (Claude Desktop, Cursor, etc.) over the **same DuckDB search database** used by the dashboard.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Behavior

| Mode | How | Tools |
|------|-----|--------|
| **Read-only (default)** | `python -m canswim mcp` | health, list, forecast/scan/price queries, `get_db_schema`, `run_select` (SELECT / WITH…SELECT only) |
| **Runs allowed** | `MCP_ALLOW_RUNS=1` (or `CANSWIM_ALLOW_RUNS=1`) | also `gather_tickers`, `forecast_tickers`, `refresh_tickers`, `refresh_job_start` |

CLI `--tickers` and the dashboard **Run** tab do **not** need `MCP_ALLOW_RUNS`. Write tools share the same backend as CLI/GUI: [run_triggers.md](run_triggers.md).

### Server version (client rediscovery)

MCP initialize and **`get_server_info`** expose package version from `setup.cfg` (via `canswim.version`). **Any** MCP tool add/rename/remove or behavior change must bump that version in the same PR so remote clients refresh tool discovery instead of caching a stale list.

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

### Streamable HTTP (production / mcp-gateway)

Default transport is **stdio** (desktop clients). For a reverse-proxy gateway (Tailscale Funnel + Caddy apikey), run Streamable HTTP bound to localhost:

```bash
python -m canswim mcp --http --host 127.0.0.1 --port 3472
# equivalent:
python -m canswim mcp --transport streamable-http --host 127.0.0.1 --port 3472
```

| Flag / env | Meaning |
|------------|---------|
| `--http` | Shorthand for Streamable HTTP transport |
| `--transport stdio\|streamable-http\|http\|sse` | Explicit transport (`http` ≡ `streamable-http`) |
| `--host` / `CANSWIM_MCP_HOST` / `MCP_HOST` | Bind address (default `127.0.0.1`) |
| `--port` / `CANSWIM_MCP_PORT` / `MCP_PORT` | Bind port (default `8000`) |
| `CANSWIM_MCP_TRANSPORT` / `MCP_TRANSPORT` | Env override for transport |

FastMCP serves the MCP endpoint at **`/mcp`** on that host:port. Public clients use the gateway path with `?apikey=` (never expose the bind port directly on the public internet).

Full production layout (user systemd, Tailscale-only Gradio UI, Funnel + Caddy apikey): **[deploy_service.md](deploy_service.md)**.

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
| `get_db_schema` | Tables, columns, indexes, row counts + markdown (for agent SQL) | — |
| `run_select` | Single read-only `SELECT` or `WITH…SELECT` (≡ Advanced Queries) | — |
| `resolve_forecast_start` | Preview week-aligned start (≡ CLI `resolve_start`) | — |
| `gather_tickers` | Scoped gather (≡ `gatherdata --tickers`) | `MCP_ALLOW_RUNS=1` |
| `forecast_tickers` | Scoped forecast; blank start = monthly catch-up + live | `MCP_ALLOW_RUNS=1` |
| `refresh_tickers` | Gather + catch-up forecast (blocking; ≡ dashboard **Refresh data & forecasts**) | `MCP_ALLOW_RUNS=1` |
| `refresh_job_start` | **Async** refresh: same pipeline as `refresh_tickers`, returns `job_id` immediately | `MCP_ALLOW_RUNS=1` |
| `refresh_job_status` | Poll a job from `refresh_job_start` (`progress_pct`, `message`, `result`) | — |

### Async refresh (preferred for SuperGrok / short tool timeouts)

Long refreshes can run for many minutes. Clients that **time out** mid-call should use the job pair instead of blocking `refresh_tickers`:

1. **`refresh_job_start`** with the ticker list → immediate `{ok, data: {job_id, status, poll_after_seconds, client_hint, …}}`.
2. Sleep about `poll_after_seconds` (typically 5–15s).
3. **`refresh_job_status`** with that `job_id` until `status` is `succeeded` or `failed` (`done=true`).
4. On success, use `data.result` and/or read tools (`get_forecast`, `list_tickers`) to verify. **Do not claim completion while status is `queued` or `running`.**

Job state is file-backed under `{data_dir}/mcp_jobs/` so status survives **client** disconnects. Only **one** refresh job runs at a time; a second start returns an error pointing at the active `job_id`. If the MCP process restarts mid-job, status is marked failed and the client should start again.

### Progress streaming (blocking long runs)

`refresh_tickers`, `forecast_tickers`, and `gather_tickers` stream **live progress** while they run (and while the tool call stays open):

| Channel | When the client sees it |
|---------|-------------------------|
| MCP `notifications/progress` | Client includes a `progressToken` in the tool call request meta (standard MCP progress protocol). Values are **0–100** with `total=100` and a human `message` (e.g. “Step 2/2: catch-up forecasts…”, origin/symbol stages). |
| MCP log (`info`) | Same stage messages, when the client supports logging notifications. |

Progress is the **same pipeline** as the dashboard Run-tab bar (`run_triggers` → `progress_cb`). Work runs off the MCP event loop so notifications can flush mid-run. Final tool result is still the usual `{ok, data|error}` payload when the job finishes.

Clients that omit `progressToken` get only the final result (no error). Async jobs use **file status** instead of mid-call progress notifications.

**Host diagnostics:** with `MCP_PROGRESS_DEBUG=1` (default when unset), the MCP process
logs to the journal whether each long tool saw a `progressToken` and each
progress emit (`MCP progress: …` / `MCP progress emit: …`). Set
`MCP_PROGRESS_DEBUG=0` to silence. Example:

```bash
journalctl --user -u canswim-mcp -f | rg 'MCP progress'
```

### Custom SQL (read-only)

1. Call **`get_db_schema`** so the client AI sees tables, types, and indexes.
2. Call **`run_select`** with one `SELECT` or `WITH … SELECT` statement.
3. Guards:
   - Statement must start with `SELECT` or `WITH` (and contain `SELECT`).
   - DDL/DML keywords, multi-statement `;`, `PRAGMA`, `ATTACH`, `COPY`, etc. are **rejected**.
   - DuckDB is opened **read-only** (`connect_readonly`).
   - Results are wrapped with `LIMIT` (default 5000).
4. **Writes are never free-form SQL** — only gated tools (`gather_tickers`, `forecast_tickers`, `refresh_tickers`, `refresh_job_start`) when `MCP_ALLOW_RUNS=1`.

## Related docs

- [cli.md](cli.md) — CLI recipes
- [run_triggers.md](run_triggers.md) — gather/forecast policy
- [data_store.md](data_store.md) — parquet vs DuckDB
