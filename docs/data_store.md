# Data store: parquet vs DuckDB

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two layers

| Layer | Location (defaults) | Role |
|-------|---------------------|------|
| **Parquet (system of record)** | `data/data-3rd-party/*.parquet`, `data/forecast/` | Ground-truth prices, fundamentals, model forecasts. Written by gather/forecast/train pipelines. |
| **DuckDB (search / UI cache)** | `data/<db_file>` e.g. `canswim_local.duckdb` | Charts dropdown, Scans, Advanced SQL, MCP reads. **Derived** from parquet. |

If the UI or MCP looks stale or missing a symbol, the usual fix is: ensure parquet is updated, then **rebuild or sync** the search DB—not invent rows in DuckDB alone.

## What each is used for

- **Gather** writes/merges parquet (prices, earnings, ownership, estimates, company profiles, …).
- **Forecast** writes forecast partitions under `data/forecast/` (and may sync rows into DuckDB).
- **Dashboard** builds or reuses DuckDB (`--same_data True` reuses; first open / rebuild loads from parquet).
- **MCP** reads DuckDB; optional `MCP_INIT_DB=1` can build it on start.

Scoped gather/forecast also **sync** symbols, forecast rows, and **company profiles** into DuckDB so Charts/Scans pick up new tickers and name/sector/industry without a full rebuild.

| Parquet file | DuckDB table | Used by |
|--------------|--------------|---------|
| `data-3rd-party/company_profile.parquet` | `company_profile` | Charts company blurb; Scans `company_name` / `sector` / `industry` columns |

`company_profile` is optional for `--same_data True` reuse (core scan tables can exist without it); gather/sync or a full rebuild populates it when the parquet is present.

## Operator tips

```bash
# Reuse existing search DB (fast)
python -m canswim dashboard --same_data True

# Rebuild search tables from local parquet (slower; use after major data changes)
python -m canswim dashboard --same_data False
```

- The dashboard shows a **search DB status** banner (path, reuse vs rebuild, row counts, whether parquet sources exist).
- On the **Run** tab under More options, **Rebuild Charts database** rebuilds DuckDB without restarting (same as `--same_data False`).
- With `--same_data True`, optional tables such as `company_profile` are **repaired lazily** if missing (no full wipe).
- Prefer **`hfhub_sync=False`** for local work so gather does not pull the full HF dataset snapshot.
- Do not commit gitignored `data/` artifacts or slimmed local-only symbol lists used for experiments (see `AGENTS.md`).

### If Charts look empty but parquet has data

1. Confirm the status banner DB path matches the data you expect.
2. Click **Rebuild Charts database** on the Run tab under More options (or restart with `--same_data False`).
3. Charts still read **DuckDB only** — editing parquet alone does not update the UI until sync/rebuild.

## Related docs

- [cli.md](cli.md)
- [run_triggers.md](run_triggers.md)
- [mcp.md](mcp.md)
