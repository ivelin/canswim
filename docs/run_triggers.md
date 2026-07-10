# Gather & forecast triggers (CLI · GUI · MCP)

One shared contract for **scoped** data gather and TiDE forecast runs.
Implementation: `canswim.run_triggers` (+ `canswim.calendar_weeks` for dates).

**NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.**

## Surfaces (aligned)

| Surface | How to invoke | Gate |
|---------|----------------|------|
| **CLI** | `python -m canswim gatherdata --tickers "AAPL,MSFT"` | Always (local CLI) |
| **CLI** | `python -m canswim forecast --tickers "AAPL" [--forecast_start_date …] [--dry_run]` | Always |
| **CLI** | `python -m canswim resolve_start [--forecast_start_date …]` | Always |
| **GUI** | Dashboard → **Run** tab | Always (in-process UI) |
| **MCP** | tools `gather_tickers`, `forecast_tickers` | **`MCP_ALLOW_RUNS=1`** (or `CANSWIM_ALLOW_RUNS=1`) |
| **MCP** | tool `resolve_forecast_start` | Read-only; always available |

Without `--tickers`, CLI `gatherdata` / `forecast` keep **legacy full-universe** behavior
(`stock_tickers_list` / all stocks; forecast default = next open after latest bar).
That path is intentionally separate so weekend batch jobs stay stable.

## Shared parameters

### Tickers
- Comma, semicolon, whitespace, and/or **newlines**
- Uppercased; duplicates reported; invalid tokens rejected
- Soft max **50** symbols per run (`DEFAULT_MAX_TICKERS`)

### Forecast start date (`YYYY-MM-DD` or empty)

| Input | Resolved start |
|-------|----------------|
| Empty / omitted | **Live default**: first NYSE session **after** the last completed week-end close (usually Monday after Friday), using local latest close when known |
| Today (calendar) | Same as live default |
| Past calendar day | First NYSE session of that **ISO market week** (usually Monday) |
| Holiday Monday (e.g. Memorial Day) | First **open** session of that week (often Tuesday) — **not** prior week |
| After allowed live origin | **Rejected** (no pure-future origins) |

Preview without running the model:
- CLI: `python -m canswim resolve_start --forecast_start_date 2026-03-05`
- GUI: **Preview start date**
- MCP: `resolve_forecast_start`

## Examples

```bash
# Gather two symbols (local-first; same as Run tab / MCP gather_tickers)
hfhub_sync=False python -m canswim gatherdata --tickers "AAPL, MSFT"

# Preview week-aligned start only
python -m canswim resolve_start
python -m canswim forecast --tickers AAPL --forecast_start_date 2026-03-05 --dry_run

# Scoped forecast (week-aligned)
python -m canswim forecast --tickers "AAPL,MSFT" --forecast_start_date 2026-03-05

# MCP write tools (default server is read-only)
MCP_ALLOW_RUNS=1 python -m canswim mcp
```

## Result shape

Structured JSON (CLI prints it; GUI shows it; MCP wraps in `{ok, data|error}`):

- **Gather:** `ok`, `tickers`, `rejected`, `messages`, optional `error`
- **Forecast:** `ok`, `tickers`, `resolved_start`, `forecasted`, `skipped`, `messages`, optional `dry_run` / `error`

## Design rules

1. **One orchestration** — do not reimplement parse/snap in GUI or MCP.
2. **MCP stays safe by default** — write tools listed for discoverability, gated at invoke time.
3. **Small lists** — keep runs short; full-universe stays on legacy CLI without `--tickers`.
4. **Local-first** — `hfhub_sync` remains off unless explicitly enabled.
