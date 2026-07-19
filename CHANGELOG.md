# Changelog

All notable releases are documented here. Versioning follows date-style `0.0.YYYYMMDD` unless noted.

## 0.0.20260722

SuperGrok called `canswim___get_chart_data` and got **Unknown tool** (connector prefix not stripped), then fell back to incomplete charts.

### Highlights

- **Connector prefix aliases** — CallTool names `canswim___…` / `canswim/…` resolve to bare tool names
- **`get_forecast(as_chart=true)`** and **`get_close_price(as_chart=true)`** return the same full chart payload as `get_chart_data` (works when chart tools are missing from a stale connector list)
- `chart_guidance` documents the as_chart fallback

### Install

```bash
pip install canswim==0.0.20260722
```

## 0.0.20260721

SuperGrok kept claiming `get_chart_data` was unavailable and fell back to `get_close_price` + `get_forecast` (latest-only, no backtests).

### Highlights

- **`plot_chart`** alias of `get_chart_data` (same payload) for connectors that miss one name
- Tool descriptions assert **PRIMARY / AVAILABLE**; `get_forecast` / `get_close_price` redirect to chart tools
- **`chart_guidance`** on `get_server_info` — do not claim unavailable if listed; do not stitch prices+forecast for full charts
- Coerce string confidence / history_years / bool args from loose clients

### Install

```bash
pip install canswim==0.0.20260721
```

## 0.0.20260720

Remote clients kept inventing a local DuckDB they could open; MCP text leaked host paths and engine details.

### Highlights

- **Client access boundary** — `get_server_info` / `health_check` / schema & SQL tools no longer return host `db_path`, `data_dir`, or “open local DuckDB” language
- Explicit **`access`** message: data only via MCP tools; no client-side database file
- Prefer purpose-built tools (`get_chart_data`, …); `run_select` framed as optional MCP analytics, not a private DB connection

### Install

```bash
pip install canswim==0.0.20260720
```

## 0.0.20260719

MCP clients needed three tools and custom grouping to recreate the dashboard Charts tab.

### Highlights

- **`get_chart_data`** — one-shot read-only tool: ~2y actual closes, **all** in-window forecast overlays (monthly backtests + live) with median + confidence band (80/95/99), reward/risk rows, and **plot_hints** / `client_recipe` for one-call plotting
- Pure DuckDB (no torch/Gradio); matches Charts confidence mapping (high band 0.95)

### Install

```bash
pip install canswim==0.0.20260719
```

## 0.0.20260718

Successful refresh still left `get_forecast` empty (TSM job succeeded, DuckDB had 0 rows).

### Highlights

- **Root cause:** `sync_forecasts_to_search_db` DELETE then INSERT failed on mixed hive parquet (`date` vs legacy `time` columns)
- **Fix:** `union_by_name` + schema-aware date expr; probe rows before DELETE; surface sync errors in forecast messages

### Install

```bash
pip install canswim==0.0.20260718
```

## 0.0.20260717

Catch-up forecasts were failing with a misleading “covariate misalignment” error.

### Highlights

- **Root cause:** scoped gather kept only ~**2 years** of prices; monthly catch-up origins need **≥336** pre-start bars, so origins like 2025-07/09/11 only had ~261–330 bars
- **Fix:** forecast-scoped lookback **~3 years** (`FORECAST_LOOKBACK_YEARS=3`); refresh re-downloads longer history when local window starts too late
- **Honest errors:** short pre-start history reports `fail_reason=short_history` (not covariates)

### Install

```bash
pip install canswim==0.0.20260717
```

## 0.0.20260716

SuperGrok still called blocking `refresh_tickers` and disconnected mid-run. **Default path is now async.**

### Highlights

- **`refresh_tickers` defaults to background job** (`wait=false`) — returns `job_id` immediately; poll `refresh_job_status`
- **`wait=true`** keeps the old blocking path (opt-in only)
- Stronger `client_hint` / `refresh_guidance` so start response is never treated as completion

### Install

```bash
pip install canswim==0.0.20260716
```

## 0.0.20260715

MCP surface bump so clients rediscover tools after async refresh jobs + portfolio-safe limits.

### Highlights

- **MCP async refresh** — `refresh_job_start` + `refresh_job_status` (file-backed jobs under `{data_dir}/mcp_jobs/`); preferred for SuperGrok / short tool timeouts
- **Portfolio jobs** — async max **~200** symbols, internal batches of ~20, `coverage` + strict `client_hint` (no full-account overclaim)
- **No silent truncate** — over-max ticker lists return `ok=false` with `omitted_tickers` / `recommended_tool` (blocking max remains ~50)
- **`get_server_info.refresh_guidance`** — preferred workflow for Schwab/large lists
- **MCP progress diagnostics** — journal `progressToken` PRESENT/MISSING + emit lines (`MCP_PROGRESS_DEBUG`)
- **Version discovery** — MCP `get_server_info` / protocol version read **checkout** `setup.cfg` when running via `PYTHONPATH`
- **Rule** — bump package version on every MCP tool/behavior change so clients refresh discovery

### Install

```bash
pip install canswim==0.0.20260715
python -m canswim -h
python -m canswim mcp
```

## 0.0.20260714

Operator UX and data-refresh wave on top of 0.0.20260713 (Run tab, catch-up forecasts, MCP schema/SQL, progress).

### Highlights

- **Refresh data & forecasts** — all-in-one gather + ~12 monthly catch-up origins + live (GUI · CLI · MCP `refresh_tickers`)
- **Run tab** — one primary path; short help + “What this does” accordion; single progress bar; Charts replot after refresh
- **Charts universe** — dropdown from CSV ∪ price parquet ∪ forecast hive (not `few_stocks` only); symbols added after refresh
- **Search DB handshake** — rebuild Charts/Scans DuckDB from parquet when needed
- **MCP** — `get_db_schema` + hardened read-only `run_select`; **live progress streaming** (`notifications/progress` + info logs) on `refresh_tickers` / `forecast_tickers` / `gather_tickers`
- **Cross-tab Gradio fix** — Run handlers no longer depend on Charts-tab inputs (fixes hard Error toast on refresh)
- **Gather status UX** — multi-bucket ready / short-history / IPO messaging
- **Torch ≥2.6 checkpoint load** — `darts_torch_load_compat()` around trusted Darts full-model pickles (`weights_only=False`); CPU and any CUDA GPU; no host/arch hardcoding
- **Portable GPU use** — still `torch.cuda.is_available()` / Lightning `accelerator="auto"`; install a torch wheel that matches the machine (see pytorch.org)
- **Deploy home layout** — canonical per-user state is **`~/.canswim/`** (`data/`, optional `service/` for user-systemd wrappers); not a separate `~/.canswim-dashboard` tree

### Requirements for end users

- Python ≥ 3.10
- Optional but typical: **FMP API key** for fundamentals, ownership, estimates, and company profiles
- Default model checkpoint on **Hugging Face** (public; swappable)
- Local disk for parquet + DuckDB under `data/` (or `~/.canswim/data` when using the prod home layout)
- Optional CUDA: any GPU supported by your installed PyTorch build

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

### Install

```bash
pip install canswim==0.0.20260714
python -m canswim -h
python -m canswim dashboard --same_data True
python -m canswim mcp
```

## 0.0.20260713

Stability and quality release on top of the 0.0.20260711 operator toolkit (GUI / MCP / scoped gather·forecast).

### Highlights

- **Company profiles (#50)** — FMP profile → parquet / DuckDB; Charts name·sector·industry blurb; Scans columns for company name, sector, industry
- **Train key-metrics hard fails (#75)** — collapse dirty weekend / duplicate index paths before train; skip bad symbols instead of aborting the whole loop
- **Forecast parquet sanity (#32)** — validate forecast frames before save (no empty / all-null quantile dumps)
- **Fund covariate imputation (#33)** — zero-fill missing fundamentals for IPO / short-history symbols so train·forecast dimensions match
- **Held-out test metrics (#31)** — report test MAE/MAPE after train fit
- **IPO / short-history UX** — consumer-friendly gather messages (not raw rate-limit noise)
- **Gradio dashboard launch** — works on restricted / non-localhost hosts
- **Publish tooling** — twine≥6 for Metadata-Version 2.4 PyPI uploads

### Requirements for end users

- Python ≥ 3.10
- Optional but typical: **FMP API key** for fundamentals, ownership, estimates, **and company profiles**
- Default model checkpoint on **Hugging Face** (public; swappable)
- Local disk for parquet + DuckDB under `data/`

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

### Install

```bash
pip install canswim==0.0.20260713
python -m canswim -h
python -m canswim dashboard --same_data True
python -m canswim mcp
```

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
