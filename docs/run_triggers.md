# Get market data & run forecasts (CLI · GUI · MCP)

User-facing actions for a **short list of symbols**. Implementation: `canswim.run_triggers`,
`canswim.gather_policy`, `canswim.calendar_weeks`.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two separate steps

| Step | What it does | CLI | GUI | MCP |
|------|----------------|-----|-----|-----|
| **Refresh data & forecasts** | Gather + catch-up forecasts (**default** GUI path) | MCP `refresh_tickers` or **async** `refresh_job_start` + `refresh_job_status` | **Refresh data & forecasts** | `refresh_tickers` / `refresh_job_*` |
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

### When remote APIs fail (network / key / plan)

Gather and **Refresh data & forecasts** (GUI · MCP · CLI) classify provider failures
via `canswim.remote_api_errors` and return a **gentle checklist** instead of a raw
stack trace:

| Kind | Typical cause |
|------|----------------|
| `network` | Offline, DNS, firewall, VPN, provider outage |
| `auth` | Invalid / rotated / revoked API key |
| `subscription` | Plan expired, tier too low for endpoint |
| `rate_limit` | Too many calls (429) |
| `timeout` | Slow link or overloaded provider |
| `missing_key` | `FMP_API_KEY` not set in this process |

**MCP:** failed write tools include `error` (human text) plus structured
`remote_api` (`kind`, `checklist`, `provider`, `detail`).  
**GUI:** Run tab status shows the same checklist; Technical log keeps full JSON.

Operators should verify internet access, that **FMP_API_KEY** (or other tokens)
are loaded after restart, and that the data plan is active.

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

MCP: blocking tool `refresh_tickers` (same pipeline), or **async** `refresh_job_start` + `refresh_job_status` for clients that time out on long tool calls (see [mcp.md](mcp.md)).

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

## Symbol classes: stocks, IPOs, and ETFs (same model)

canswim uses **one TiDE checkpoint** and a **fixed feature layout** for every
symbol at train and at inference. The model does not have separate ETF/IPO
heads. What changes is **how much real CANSLIM-style fund data** is available
and what we **impute** so the tensor width still matches training.

### One model, fixed feature width

| Layer | Role | Always required? |
|-------|------|------------------|
| **Target** | Stock/ETF **Close** (or configured target column) | **Yes** — ground-truth bars only (no invented OHLCV) |
| **Past covariates** | Own OHLC+volume, earnings, key metrics, ownership, splits, broad market / sectors / industry funds | **Yes as a block** — missing *fund* slices are zero-/sentinel-filled |
| **Future covariates** | Dividends, analyst estimate paths, holidays | **Yes as a block** — missing estimates zero-filled |

Training and forecast both call the same covariate stack (`canswim.covariates`).
If a column that existed at train is missing at forecast, Darts raises a
**dimensionality** error. That is why fund-thin names must **impute columns**,
not drop them.

### Three operator-facing classes (MECE)

| Class | Examples | What “rich CANSLIM data” means here | Typical gaps |
|-------|----------|--------------------------------------|--------------|
| **A. Covered stocks** | LLY, AAPL, MSFT | Full(ish) price history **and** corporate fundamentals: earnings calendar, key metrics, institutional ownership, sell-side estimates | Occasional sparse fields only |
| **B. IPOs / thin equities** | Recent listings | Price history often short; fundamentals **late or empty** | Not enough **sessions** for min history; fund rows missing until coverage catches up |
| **C. ETFs / funds** | XLF, SPY, sector & theme ETFs | **Prices + market context** matter; no corporate EPS / key metrics / equity research “by design” | Fund rows **never** appear (empty filter for that symbol) |

These classes share **market-context** past covariates (broad indexes, sectors,
industry funds) and the **own-price** past block. They differ on **issuer-level**
fundamentals.

### Data requirements: train vs inference

Same rules on both paths unless noted.

| Requirement | Covered stocks (A) | IPOs / thin (B) | ETFs / funds (C) |
|-------------|--------------------|-----------------|------------------|
| **OHLCV history** | Full train window or ~2y scoped | Must eventually reach ~**2 years of sessions** for forecast-scoped readiness | Same price floor as stocks (~2y scoped) |
| **Own OHLC+volume as past covs** | Real | Real when listed | Real (ETF prints) |
| **Earnings / key metrics / ownership** | Real when present | **Zero-fill** missing (#33) | **Zero-fill** missing (same mechanism; expected permanent) |
| **Analyst estimates (future)** | Real when present | **Zero-fill** missing | **Zero-fill** missing |
| **Broad / sector / industry funds** | Shared series (all symbols) | Shared series | Shared series (often the informative path for ETFs) |
| **Dividends / splits / holidays** | Real or empty-padded | Same | Same |
| **Train inclusion** | Preferred “rich” examples | Included if prices + imputed fund dims work | Can be included the same way; model still learns price+market features |
| **Forecast / Refresh** | Default path | Fail **history** if too short; else impute fund | Fail **history** if too short; else impute fund (empty batch OK) |

**Hard fail (cannot invent):** insufficient **price** history for the forecast
window / min samples. Status talks about short history / IPOs.

**Soft gap (impute, do not drop columns):** missing earnings, key metrics,
ownership, estimates — whether temporary (IPO) or structural (ETF).

### How imputation works (train + inference)

1. Build real series per symbol when parquet has rows.
2. If some symbols in the batch lack a block, **copy the column template** from a
   peer that has it and fill with `0` (ownership) or `-1` / zeros (earn, kms,
   estimates), aligned to that symbol’s **price calendar**.
3. If the **entire batch** has no fund rows (e.g. refresh **XLF** alone), there is
   no peer template:
   - **Earnings:** fixed train schema columns (always emit the same names/order).
   - **Key metrics / analyst estimates:** load a **disk template** from covered
     large caps (e.g. AAPL family) and zero-fill those columns for the thin name.
4. Stack into past/future covariates with the **same width** as training so one
   checkpoint works for A/B/C.

Implementation: `canswim.covariates` (issue #33 for IPOs; empty-batch / ETF path
extends the same idea).

### What this means for operators

| You want to… | Expectation |
|--------------|-------------|
| Refresh **LLY / AAPL** | Market data + real fundamentals when APIs have them; catch-up forecasts as usual |
| Refresh a **new IPO** | May **stop** until ~2y of sessions; fundamentals imputed once prices are ready |
| Refresh **XLF** or a sector ETF | Prices + market funds load; **no corporate fund rows** — imputed automatically; forecast should not fail on dimensionality alone |
| Mix ETF + stocks in one list | Peers can supply templates; still fine if only ETFs (empty-batch path) |

**Interpretation note:** For ETFs (and heavily imputed IPOs), the model is driven
mainly by **price path + broad/sector context**, not issuer fundamentals. That is
intentional with a single shared head—not a second “ETF model.”

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Design rules

1. One orchestration for CLI / GUI / MCP.
2. Missing-only remote calls for forecast-scoped gather; train stays full-history.
3. Fail closed on incomplete **price** history; **impute** optional fundamentals so feature width stays fixed.
4. Consumer copy in the product; policy detail in this doc.
5. Parquet is the system of record; DuckDB is the search/UI cache ([data_store.md](data_store.md)).
6. Same model for covered stocks, IPOs, and ETFs — different real-data density, same tensor schema.
7. Impute missing optional fundamentals rather than excluding symbols (train + forecast).
