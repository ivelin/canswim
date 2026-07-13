# Get market data & run forecasts (CLI · GUI · MCP)

User-facing actions for a **short list of symbols**. Implementation: `canswim.run_triggers`,
`canswim.gather_policy`, `canswim.calendar_weeks`.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two separate steps

| Step | What it does | CLI | GUI | MCP |
|------|----------------|-----|-----|-----|
| **Refresh data & forecasts** | Gather + catch-up forecasts (**default** GUI path) | MCP `refresh_tickers` | **Refresh data & forecasts** | `refresh_tickers` |
| **Get market data** | Update local prices **and** model fundamentals for listed symbols | `gatherdata --tickers "AAPL,MSFT"` | **Update market data** | `gather_tickers` |
| **Run a forecast** | Forecast those symbols (blank start = monthly catch-up + live) | `forecast --tickers "AAPL" …` | **Run forecast** | `forecast_tickers` |
| **Check start date** | Show which forecast start date will be used | `resolve_start` | **Check start date** | `resolve_forecast_start` |
| **Rebuild Charts database** | Rebuild DuckDB Charts/Scans cache from parquet | `dashboard --same_data False` | **Rebuild Charts database** | (rebuild via dashboard / `MCP_INIT_DB`) |

MCP write tools need `MCP_ALLOW_RUNS=1`. CLI and dashboard do not.

Without `--tickers`, CLI `gatherdata` / `forecast` keep **full-universe / train-style** behavior.

## Get market data (lean & rate-limit aware)

For scoped runs (`--tickers` / dashboard / MCP):

- Target about the **last 2 years** of prices (enough to forecast)—not multi-decade history.
- **Skip remote download** when local history is already complete and recent.
- If history is short or gappy, download only the **missing window**.
- If history is complete but stale, download only a **short tail** refresh.
- **Train** mode (`gatherdata` without `--tickers`) still uses full `train_date_start` history.

### Fundamentals (covariates), not only OHLCV

Unless `--no_covariates` / GUI equivalent, scoped gather also refreshes model inputs such as:

- earnings calendar, key metrics  
- institutional ownership, analyst estimates  
- dividends, splits  
- broad market / sector series as needed  

Scoped writers **merge** into existing parquet by symbol so a short ticker list does not wipe other symbols’ fundamentals.

After a successful gather, symbols are **synced into the DuckDB search DB** so Charts/Scans dropdowns include them. See [data_store.md](data_store.md).

## Refresh data & forecasts (recommended)

All-in-one for a short list (portfolio / new names). GUI label: **Refresh data & forecasts**.

1. **Market data** — prices + fundamentals (missing-only, ~2y).  
2. **Catch-up forecasts** — ~12 monthly origins + live for symbols that are ready.  
3. **Charts list + DuckDB** — symbols appear in Charts; forecasts and backtest errors sync for Scans.

**Skipped:** work already on file; short-history / IPO names (reported in status).

MCP tool name remains `refresh_tickers` (same pipeline).

## Run a forecast

- Forecasts never invent prices. If OHLCV history is incomplete, the run **fails** and asks you to update market data first (or use **Refresh data & forecasts**).
- If prices look fine but ownership/estimates (or alignment) fail, the run fails with a **covariates** message—run **Update market data** again (with fundamentals), then retry.
- Symbols that **already have a saved forecast** for a given start are **skipped** (no re-run).
- After a successful forecast, rows are **synced into DuckDB** (including **backtest_error** refresh) for Charts/Scans.
- Live starts may be **clamped** to the next open session after the last available local bar when broad/covariate calendars lag the requested week start.

### Catch-up mode (blank start date)

When **start date is blank** (GUI / MCP / CLI scoped forecast):

- Origins = **first market week of each of the last ~12 calendar months** (env `CATCHUP_MONTHS`, default 12) **plus** the live week start.
- Monthly origin = first NYSE session of that month, snapped to that week’s first session (at most **one forecast per ISO week**).
- Already-saved symbol×start pairs are skipped.
- Charts/Scans then have history for reward/risk and backtest quality, not only the latest live path.

Explicit `YYYY-MM-DD` still means **single-origin** mode (week-aligned).

## Start date rules (enforced in code)

| You enter | System uses |
|-----------|-------------|
| Blank | **Catch-up**: monthly origins (~12 months) + live week start |
| Today / default live only (via resolve) | Next market-week start after the latest completed trading week |
| A past date | Start of that market week (first open session; if Monday is a holiday, next open day that week) |
| A future date past the allowed default | Rejected |

Operator detail only—primary UI uses plain language.

## Examples

```bash
# Update market data for two symbols (missing-only, ~2y + fundamentals)
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"

# See start date
python -m canswim resolve_start
python -m canswim forecast --tickers AAPL --forecast_start_date 2026-03-05 --dry_run

# Forecast (fails if data incomplete; skips if already saved for that start)
python -m canswim forecast --tickers "AAPL,MSFT" --forecast_start_date 2026-03-05
```

More CLI recipes: [cli.md](cli.md). MCP: [mcp.md](mcp.md).

## Missing fundamentals (IPO / thin coverage)

The model stack uses **zero-fill / sentinel fill** when a symbol has prices but no
earnings, key metrics, institutional ownership, or analyst estimates (common for
recent IPOs and small caps). That keeps feature dimensionality fixed for train
and forecast so those names are not dropped solely for missing fundamentals.

- Prices remain ground-truth (no invented OHLCV). Short price history still fails
  readiness checks (~2y for forecast-scoped runs).
- Ownership fill: `0`. Earnings / estimates: same padding style as sparse known
  series (`-1` / zero columns aligned to the price calendar when a template exists).

## Design rules

1. One orchestration for CLI / GUI / MCP.
2. Missing-only remote calls for forecast-scoped gather; train stays full-history.
3. Fail closed on incomplete forecast data (prices or covariates).
4. Consumer copy in the product; policy detail in this doc.
5. Parquet is the system of record; DuckDB is the search/UI cache ([data_store.md](data_store.md)).
6. Impute missing optional fundamentals rather than excluding symbols (train + forecast).
