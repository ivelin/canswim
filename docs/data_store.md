# Data store: parquet vs DuckDB

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two layers

| Layer | Location (defaults) | Role |
|-------|---------------------|------|
| **Parquet (system of record)** | `data/data-3rd-party/*.parquet`, `data/forecast/` | Ground-truth prices, fundamentals, model forecasts. Written by gather/forecast/train pipelines. |
| **DuckDB (search / UI cache)** | `data/<db_file>` e.g. `canswim_local.duckdb` | Charts dropdown, Scans, Advanced SQL, MCP reads. **Derived** from parquet. |

If the UI or MCP looks stale or missing a symbol, the usual fix is: ensure parquet is updated, then **rebuild or sync** the search DB—not invent rows in DuckDB alone.

## What each is used for

- **Gather** writes/merges parquet (prices, earnings, ownership, estimates, …).
- **Forecast** writes forecast partitions under `data/forecast/` (and may sync rows into DuckDB).
- **Dashboard** builds or reuses DuckDB (`--same_data True` reuses; first open / rebuild loads from parquet).
- **MCP** reads DuckDB; optional `MCP_INIT_DB=1` can build it on start.

Scoped gather/forecast also **sync** symbols and forecast rows into DuckDB so Charts/Scans pick up new tickers without a full rebuild.

## Operator tips

```bash
# Reuse existing search DB (fast)
python -m canswim dashboard --same_data True

# Rebuild search tables from local parquet (slower; use after major data changes)
python -m canswim dashboard --same_data False
```

- Prefer **`hfhub_sync=False`** for local work so gather does not pull the full HF dataset snapshot.
- Do not commit gitignored `data/` artifacts or slimmed local-only symbol lists used for experiments (see `AGENTS.md`).

## Related docs

- [cli.md](cli.md)
- [run_triggers.md](run_triggers.md)
- [mcp.md](mcp.md)
