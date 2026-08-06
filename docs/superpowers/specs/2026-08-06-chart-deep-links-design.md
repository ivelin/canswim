# Chart deep links (shareable GUI URLs)

**Status:** Approved design  
**Date:** 2026-08-06  
**Scope:** Gradio dashboard Charts tab only (v1)

## Problem

The CANSWIM Gradio playground exposes a single top-level URL (e.g. `http://host:7860/`). Selecting a stock symbol, confidence level, or other UI state does not change the browser URL, so users cannot copy a link that opens the same chart.

## Goals

1. Selecting a ticker (and confidence) updates the browser address bar with a shareable, human-readable query string.
2. Opening that URL loads the Charts tab with the same ticker and confidence and renders the forecast chart.
3. Links remain valid across Gradio process restarts (no server-side snapshot dependency).

## Non-goals (v1)

- Encoding active tab (Scans / Run / Advanced)
- Scans filters, forecast start date, Advanced query text
- Gradio `DeepLinkButton` / opaque `deep_link=` hashes
- Path-based multi-page routing (`/charts`, etc.)
- MCP / CLI changes or version bump

## URL contract

| Param   | Required | Values                         | Notes |
|---------|----------|--------------------------------|--------|
| `ticker` | No      | Uppercase symbol string        | Must be in the Charts dropdown choices (symbols known to the search DB). Case-insensitive on input; normalized to uppercase. |
| `lowq`   | No      | `80`, `95`, or `99`            | Confidence % for lowest close price. Invalid/missing → `80`. |

**Examples**

```
http://spark-9045.tail39d5a.ts.net:7860/?ticker=AAPL
http://spark-9045.tail39d5a.ts.net:7860/?ticker=AAPL&lowq=95
```

**Invalid / missing behavior**

- Missing both params → keep current behavior (random default ticker from available symbols, `lowq=80`).
- Unknown `ticker` (not in choices) → fall back to current default ticker; do not crash.
- Invalid `lowq` → use `80`.
- Preserve unrelated query params if present when updating the URL (only set/replace `ticker` and `lowq`).

## Approach (chosen)

**Readable query params + live `history.replaceState`.**

Rejected alternatives:

- **Gradio DeepLinkButton** — opaque hashes, server-side state files, not always in the address bar, fragile across restarts.
- **Multi-page Gradio routes** — larger refactor than needed for charts-only share links.

## Architecture

### 1. Pure helpers (testable)

Small module or functions (e.g. under `src/canswim/dashboard/url_state.py` or next to charts):

- `parse_chart_query(query_params: Mapping) -> tuple[str | None, int]`  
  - Normalize ticker (strip, upper); validate `lowq` ∈ {80, 95, 99}.
- `resolve_chart_ticker(requested: str | None, choices: Sequence[str], default: str | None) -> str | None`  
  - Return requested if in choices (case-insensitive match to canonical choice), else default.
- `chart_query_js` string (or builder) for client-side URL sync — keep JS minimal and documented.

### 2. Load path

In `CanswimPlayground.launch` / existing `demo.load`:

- Accept `gr.Request` (or read query via Gradio’s request injection).
- On page load, parse `ticker` / `lowq` from `request.query_params`.
- Resolve against current ticker choices.
- Outputs must update: `tickerDropdown`, `lowq`, plot, reward/risk table, company markdown (same outputs as today’s load + plot path).
- Prefer query values over component-constructed defaults when valid.

### 3. Live URL sync

When `tickerDropdown` or `lowq` changes:

- Existing Python `plot_forecast` handlers stay responsible for chart data.
- Attach client-side JS (Gradio `js=` on the change events, or a thin wrapper) that:
  1. Reads current ticker and lowq from the event inputs.
  2. Updates `URLSearchParams` for `ticker` and `lowq` only.
  3. Calls `history.replaceState` (no navigation / no reload).

No full page reload on ticker change.

### 4. Component defaults

- Keep random default ticker when no query param (current UX for casual browsing).
- When query provides a valid ticker, dropdown initial value should match so load does not flash a wrong symbol longer than necessary (best-effort: set `value=` after parse if we can read query only at load-time via request; initial render may still use random until load runs — acceptable Gradio limitation).

## Error handling

- Parse/resolve never raises into Gradio UI; invalid input → defaults.
- Missing DB / empty ticker list → existing empty-state behavior.
- JS failures must not break plot updates (plot remains Python-driven).

## Testing

1. **Unit tests** for parse/resolve helpers:
   - case normalization (`aapl` → `AAPL`)
   - invalid lowq → 80
   - unknown ticker → default
   - empty / missing params
2. No browser E2E required for v1; optional manual check on the live dashboard.

## Documentation

- Short note for operators/users: share chart via address bar `?ticker=&lowq=`.
  - Prefer a brief addition near dashboard mentions in `README.md` or `docs/cli.md` / deploy docs only if those already describe the GUI URL — keep docs proportional.
- Do **not** invent a parallel long design tree in user-facing docs; this file is the design record.

## Implementation plan outline (for later planning skill)

1. Add pure URL helpers + unit tests.
2. Wire `demo.load` to request query params → component updates + plot.
3. Wire ticker/lowq change events with JS `replaceState`.
4. Run `./scripts/ci-local.sh`.
5. Manual smoke: open `/?ticker=…`, change ticker, confirm address bar and reload.

## Success criteria

- [ ] `/?ticker=AAPL&lowq=95` opens Charts with AAPL at 95% confidence and plots.
- [ ] Changing ticker or confidence updates the address bar without reload.
- [ ] Copy-paste of that URL in a new tab restores the same chart controls.
- [ ] Unknown ticker / bad lowq does not error; falls back safely.
- [ ] `./scripts/ci-local.sh` passes.
- [ ] No MCP version bump (GUI-only).
