# Get market data & run forecasts (CLI · GUI · MCP)

User-facing actions for a **short list of symbols**. Implementation: `canswim.run_triggers`,
`canswim.gather_policy`, `canswim.calendar_weeks`.

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Two separate steps

| Step | What it does | CLI | GUI | MCP |
|------|----------------|-----|-----|-----|
| **Get market data** | Update local prices for listed symbols | `gatherdata --tickers "AAPL,MSFT"` | **Update market data** | `gather_tickers` |
| **Run a forecast** | Forecast those symbols | `forecast --tickers "AAPL" …` | **Run forecast** | `forecast_tickers` |
| **Check start date** | Show which start date will be used | `resolve_start` | **Check start date** | `resolve_forecast_start` |

MCP write tools need `MCP_ALLOW_RUNS=1`. CLI and dashboard do not.

Without `--tickers`, CLI `gatherdata` / `forecast` keep **full-universe / train-style** behavior.

## Get market data (lean & rate-limit aware)

For scoped runs (`--tickers` / dashboard / MCP):

- Target about the **last 2 years** of prices (enough to forecast)—not multi-decade history.
- **Skip remote download** when local history is already complete and recent.
- If history is short or gappy, download only the **missing window**.
- If history is complete but stale, download only a **short tail** refresh.
- **Train** mode (`gatherdata` without `--tickers`) still uses full `train_date_start` history.

Forecasts never invent prices. If history is incomplete, **Run a forecast** fails and asks you to update market data first.

## Start date rules (enforced in code)

| You enter | System uses |
|-----------|-------------|
| Blank / today | Next market-week start after the latest completed trading week |
| A past date | Start of that market week (first open session; if Monday is a holiday, next open day that week) |
| A future date past the allowed default | Rejected |

Operator detail only—primary UI uses plain language.

## Examples

```bash
# Update market data for two symbols (missing-only, ~2y)
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"

# See start date
python -m canswim resolve_start
python -m canswim forecast --tickers AAPL --forecast_start_date 2026-03-05 --dry_run

# Forecast (fails if data incomplete)
python -m canswim forecast --tickers "AAPL,MSFT" --forecast_start_date 2026-03-05
```

## Design rules

1. One orchestration for CLI / GUI / MCP.
2. Missing-only remote calls for forecast-scoped gather; train stays full-history.
3. Fail closed on incomplete forecast data.
4. Consumer copy in the product; policy detail in this doc.
