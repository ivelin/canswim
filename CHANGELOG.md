# Changelog

All notable releases are documented here. Versioning follows date-style `0.0.YYYYMMDD` unless noted.

## 0.0.20260711

Operator-focused release: install locally, use Gradio GUI or MCP with a public TiDE checkpoint and your own market-data API keys (e.g. FMP).

### Highlights

- **Dashboard Run tab** — **Update market data** / **Run forecast** / **Check start date** (same backend as CLI and MCP)
- **Scoped CLI** — `gatherdata --tickers`, `forecast --tickers`, `resolve_start`, `--dry_run`, `--no_covariates`
- **MCP server** — read-only by default; `gather_tickers` / `forecast_tickers` with `MCP_ALLOW_RUNS=1`
- **Lean gather** — ~2y missing-only history for scoped runs; train path keeps long history
- **Hard-fail forecast** — no invented prices; clear messages for incomplete OHLCV vs missing covariates
- **Skip re-forecast** when a partition already exists for the start date
- **DuckDB sync** after gather/forecast so Charts/Scans see new symbols
- **Docs site** — https://ivelin.github.io/canswim/ (from `main` / MkDocs)
- Operator guides: CLI, MCP, data store (parquet vs DuckDB), run triggers

### Requirements for end users

- Python ≥ 3.10
- Optional but typical: **FMP API key** (and related env) for full fundamentals/covariates
- Default model checkpoint on **Hugging Face** (public; swappable)
- Local disk for parquet + DuckDB under `data/`

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

### Install

```bash
pip install canswim==0.0.20260711
python -m canswim -h
python -m canswim dashboard --same_data True
python -m canswim mcp
```

## 0.0.20250107

Previous PyPI release (baseline prior to 2026 operator tooling wave).
