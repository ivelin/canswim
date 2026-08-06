"""Shareable chart URL helpers for the Gradio dashboard.

Query contract (Charts tab v1):
  ?ticker=AAPL
  ?ticker=AAPL&lowq=95

``lowq`` is confidence % for the lowest close price: 80 | 95 | 99.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

VALID_LOWQ = frozenset({80, 95, 99})
DEFAULT_LOWQ = 80

# Side-effect only: update address bar without reload. Used with fn=None listeners.
# Inputs: (ticker, lowq). Preserve unrelated query params.
CHART_URL_SYNC_JS = """
(ticker, lowq) => {
  try {
    const url = new URL(window.location.href);
    if (ticker) {
      url.searchParams.set('ticker', String(ticker));
    } else {
      url.searchParams.delete('ticker');
    }
    if (lowq !== null && lowq !== undefined && lowq !== '') {
      url.searchParams.set('lowq', String(lowq));
    }
    window.history.replaceState({}, '', url);
  } catch (e) {
    console.error('canswim chart URL sync failed', e);
  }
}
"""


def parse_chart_query(query_params: Mapping[str, Any] | None) -> tuple[str | None, int]:
    """Parse chart deep-link query params.

    Returns
    -------
    ticker :
        Uppercased symbol string, or ``None`` if missing/blank.
    lowq :
        Confidence percent in ``VALID_LOWQ``; invalid or missing → ``DEFAULT_LOWQ``.
    """
    if not query_params:
        return None, DEFAULT_LOWQ

    raw_ticker = query_params.get("ticker")
    ticker: str | None = None
    if raw_ticker is not None:
        s = str(raw_ticker).strip().upper()
        if s:
            ticker = s

    lowq = DEFAULT_LOWQ
    raw_lowq = query_params.get("lowq")
    if raw_lowq is not None and str(raw_lowq).strip() != "":
        try:
            n = int(float(str(raw_lowq).strip()))
            if n in VALID_LOWQ:
                lowq = n
        except (TypeError, ValueError):
            pass

    return ticker, lowq


def resolve_chart_ticker(
    requested: str | None,
    choices: Sequence[str] | None,
    default: str | None,
) -> str | None:
    """Map a requested ticker to a canonical choice, or fall back to ``default``."""
    if not requested:
        return default
    if not choices:
        return default
    want = str(requested).strip().upper()
    if not want:
        return default
    for c in choices:
        if c is None:
            continue
        if str(c).strip().upper() == want:
            return str(c)
    return default
