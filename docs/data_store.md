# Data store: parquet vs DuckDB

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two layers

| Layer | Location (defaults) | Role |
|-------|---------------------|------|
| **Parquet (system of record)** | `data/data-3rd-party/*.parquet`, `data/forecast/` | Ground-truth prices, fundamentals, model forecasts. Written by gather/forecast/train pipelines. |
| **DuckDB (search / UI cache)** | `data/<db_file>` e.g. `canswim_local.duckdb` | Charts dropdown, Scans, Advanced SQL, MCP reads. **Derived** from parquet. |

If the UI or MCP looks stale or missing a symbol, the usual fix is: ensure parquet is updated, then **rebuild or migrate** the search DB—not invent rows in DuckDB alone.

## What each is used for

- **Gather** writes/merges parquet (prices, earnings, ownership, estimates, company profiles, …).
- **Forecast** writes forecast partitions under `data/forecast/` (and may sync rows into DuckDB).
- **Dashboard** builds or reuses DuckDB (`--same_data True` reuses + **runs migrations**; first open / rebuild loads from parquet).
- **MCP** reads DuckDB; optional `MCP_INIT_DB=1` can build it on start.

Scoped gather/forecast also **sync** symbols, forecast rows, and **company profiles** into DuckDB so Charts/Scans pick up new tickers and name/sector/industry without a full rebuild.

## Search DB schema (DuckDB)

**Code:** `canswim.db` (tables), `canswim.db_migrations` (version + upgrades).  
**Runtime export:** MCP `get_db_schema` / `describe_search_schema()` (live columns, types, indexes, `schema_version`).  
**Current schema version:** `CURRENT_SCHEMA_VERSION` in `src/canswim/db_migrations.py` (v**1** as of this doc).

### Core tables (required for reuse)

| Table | Role | Typical keys / columns |
|-------|------|-------------------------|
| `stock_tickers` | Charts/Scans symbol universe | `symbol` (or `Symbol`) |
| `forecast` | Quantile paths | `symbol`, `start_date` (origin), `date` (horizon day), `close_quantile_*` |
| `latest_forecast` | Per-symbol latest origin | `symbol`, `date` (= max start_date) |
| `close_price` | Historical closes | `Date`, `Symbol`/`symbol`, `Close` |
| `backtest_error` | Mean abs log-error of median vs close | `symbol`, `start_date`, `mal_error` |

### Optional tables

| Table | Role |
|-------|------|
| `company_profile` | Name, sector, industry, exchange, mkt_cap, website, description (one row per symbol) |
| `canswim_schema_meta` | Key/value meta: **`schema_version`**, migration stamps |

### Parquet → DuckDB map

| Parquet / path | DuckDB table | Notes |
|----------------|--------------|--------|
| `symbol_lists/*.csv` ∪ price ∪ forecast hive | `stock_tickers` | Union on rebuild; expand on reuse |
| `data/forecast/**/*.parquet` (hive) | `forecast`, `latest_forecast` | Rebuild from hive partitions |
| `data-3rd-party/all_stocks_price_hist_1d.parquet` | `close_price` | Target column (default Close) |
| (derived from forecast + close) | `backtest_error` | Computed on rebuild / refreshed on forecast sync |
| `data-3rd-party/company_profile.parquet` | `company_profile` | Optional; lazy repair on reuse |

### Indexes (created on full rebuild)

- `forecast_symd_idx` on `forecast (symbol, start_date, date)`
- `latest_forecast_symd_idx` on `latest_forecast (symbol, date)`
- `close_price_symd_idx` on `close_price (symbol, date)`
- `backtest_error_sym_start_idx` on `backtest_error (symbol, start_date)`

Column casing may vary (`Symbol` vs `symbol`); prefer `upper(symbol)` joins in custom SQL.

---

## Upgrading between app versions (required path)

**Going forward, every schema-affecting change must ship with:**

1. **Code** — a new entry in `MIGRATIONS` + bump `CURRENT_SCHEMA_VERSION` (`db_migrations.py`)
2. **Docs** — a row in the **Migration log** below (this file)
3. **Tests** — coverage in `tests/canswim/test_db_migrations.py`

### Operator upgrade steps

1. **Install** the new `canswim` package (pip/conda).
2. **Keep** local `data/data-3rd-party/` and `data/forecast/` (parquet SoT).
3. **Start dashboard or rebuild:**
   - Prefer: `python -m canswim dashboard --same_data True`  
     → reuses DuckDB and **applies pending migrations automatically**.
   - Or full rebuild: Run tab **Rebuild Charts database**, or  
     `python -m canswim dashboard --same_data False`  
     → recreates tables from parquet and **stamps current schema version**.
4. **Confirm** (optional): MCP `get_db_schema` / health — `schema_version` should equal app `schema_version_current`.
5. If migrations fail or the DB looks corrupt: **full rebuild** from parquet (step 3, `--same_data False`). Parquet is never rewritten by search migrations.

### Escape hatch

| Situation | Action |
|-----------|--------|
| Migration error / unknown old file | Rebuild: `--same_data False` or **Rebuild Charts database** |
| Charts empty but parquet full | Rebuild search DB |
| Only missing `company_profile` | Reuse path repairs it; or re-gather profiles |

---

## Migration log

| Version | Name | App note | What the migration does |
|---------|------|----------|-------------------------|
| **1** | `baseline_search_v1` | Introduced with schema versioning | Ensures `canswim_schema_meta`; ensures optional `company_profile`; stamps `schema_version=1`. Legacy DBs with core tables but no meta are treated as v0 → v1. |

*When adding v2+: append a row here in the same PR as the code migration.*

### Developer checklist (new schema change)

```text
[ ] Bump CURRENT_SCHEMA_VERSION
[ ] Add Migration(version=N, name=..., upgrade=...)
[ ] Document in Migration log (this file)
[ ] Test: apply_migrations from N-1 → N (and legacy → N if needed)
[ ] Prefer additive changes; destructive changes require rebuild docs + test
```

---

## Operator tips

```bash
# Reuse existing search DB (fast) + run migrations
python -m canswim dashboard --same_data True

# Rebuild search tables from local parquet (slower; stamps current schema version)
python -m canswim dashboard --same_data False
```

- The dashboard shows a **search DB status** banner (path, reuse vs rebuild, row counts, whether parquet sources exist).
- On the **Run** tab under More options, **Rebuild Charts database** rebuilds DuckDB without restarting (same as `--same_data False`).
- With `--same_data True`, optional tables such as `company_profile` are **repaired lazily** if missing (no full wipe); **schema migrations** run first.
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
