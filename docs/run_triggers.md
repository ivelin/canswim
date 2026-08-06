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

## Weekend host job (all DB symbols)

CLI task **`weekend`** (`./weekend.sh`, `canswim-weekend.timer`) loads every symbol
from DuckDB **`stock_tickers`**, batches gather + forecast, and defaults to the
**live week start** from blank `resolve_start`. Optional `--catchup` uses blank
forecast start (monthly catch-up + live). Host operator path (`force_allow`), not
an MCP tool. See [cli.md](cli.md) and [deploy_service.md](deploy_service.md).

## Get market data (lean & rate-limit aware)

For scoped runs (`--tickers` / dashboard / MCP):

- Target about the **last 3 years** of prices (model lookback + ~12 monthly catch-up origins)—not multi-decade history.
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

1. **Market data** — prices + fundamentals (missing-only, ~3y).  
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
# Update market data for two symbols (missing-only, ~3y + fundamentals)
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
| **Past covariates** | Own OHLC+volume, earnings, key metrics, ownership, splits, broad market / sectors / industry funds | **Forecast:** real fund rows required (earnings + key metrics + estimates). **Train only:** missing fund slices may be zero-filled (#33) so feature width matches the checkpoint. |
| **Future covariates** | Dividends, analyst estimate paths, holidays | **Forecast:** real analyst estimates required. **Train only:** may zero-fill missing estimates. |

Training and forecast both call the same covariate stack (`canswim.covariates`).
If a column that existed at train is missing at forecast, Darts raises a
**dimensionality** error. **Forecast/backtest never invent fundamentals** —
symbols without real local fund data are skipped (`fail_reason=fundamentals`).
Train may still impute fund-thin names so one checkpoint trains on mixed batches.

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
| **OHLCV history** | Full train window or ~3y scoped | Must eventually reach ~**3 years of sessions** for forecast-scoped readiness | Same price floor as stocks (~3y scoped) |
| **Own OHLC+volume as past covs** | Real | Real when listed | Real (ETF prints) |
| **Earnings / key metrics / ownership** | Real on disk for forecast | **Forecast:** hard-fail if missing. **Train:** may zero-fill (#33) | **Forecast:** hard-fail if no real fund rows (typical for pure ETFs until data exists). **Train:** may zero-fill |
| **Analyst estimates (future)** | Real on disk for forecast | Same as above | Same as above |
| **Broad / sector / industry funds** | Shared series (all symbols) | Shared series | Shared series (often the informative path for ETFs) |
| **Dividends / splits / holidays** | Real or empty-padded | Same | Same |
| **Train inclusion** | Preferred “rich” examples | Included if prices + imputed fund dims work | Can be included the same way; model still learns price+market features |
| **Forecast / Refresh** | Real prices **and** real fund files | Fail if **history** short **or** fund files missing | Fail if fund files missing (no invented estimates) |

**Hard fail (cannot invent) — forecast/backtest:**

1. Insufficient **price** history (ground-truth OHLCV only; no synthetic bars).
2. Missing **real fundamentals** on disk: earnings calendar, key metrics, and
   analyst estimates (annual **or** quarterly). Zero-filled placeholders are
   **not** accepted for inference.

**Train-only soft gap (impute, do not drop columns):** missing earnings, key
metrics, ownership, estimates — temporary (IPO) or structural (ETF) — so the
feature width still matches the checkpoint. Controlled by
`allow_fundamentals_imputation=True` on the train path only.

### How train imputation works (not used for forecast)

1. Build real series per symbol when parquet has rows.
2. If some symbols in the batch lack a block **and** train imputation is on,
   **copy the column template** from a peer and fill with `0` / `-1`.
3. Forecast path **refuses** that step and skips the symbol instead.

Implementation: `canswim.eligibility` (`fundamentals_are_ready`,
`partition_by_fundamentals`) + `canswim.covariates` +
`forecast_for_tickers` gate. Operator cleanup:
`scripts/purge_forecasts_without_fundamentals.py`.

### What this means for operators

| You want to… | Expectation |
|--------------|-------------|
| Refresh **LLY / AAPL** | Market data + real fundamentals when APIs have them; catch-up forecasts as usual |
| Refresh a **new IPO** | May **stop** until enough sessions (~3y floor for catch-up) **and** real fund rows exist on disk |
| Refresh **XLF** or a sector ETF | Prices + market funds load; **no corporate fund rows** — imputed automatically; forecast should not fail on dimensionality alone |
| Mix ETF + stocks in one list | Peers can supply templates; still fine if only ETFs (empty-batch path) |

**Interpretation note:** For ETFs (and heavily imputed IPOs), the model is driven
mainly by **price path + broad/sector context**, not issuer fundamentals. That is
intentional with a single shared head—not a second “ETF model.”

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Design rules

1. One orchestration for CLI / GUI / MCP.
2. Missing-only remote calls for forecast-scoped gather; train stays full-history.
3. Fail closed on incomplete **price** history; **forecast** fails closed without real fundamentals; **train** may impute fund width only.
4. Consumer copy in the product; policy detail in this doc.
5. Parquet is the system of record; DuckDB is the search/UI cache ([data_store.md](data_store.md)).
6. Same model for covered stocks, IPOs, and ETFs — different real-data density, same tensor schema.
7. Impute missing optional fundamentals only on **train**; **forecast** excludes symbols without real fund data.
