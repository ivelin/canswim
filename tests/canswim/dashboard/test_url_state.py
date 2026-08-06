"""Unit tests for chart deep-link URL helpers."""

from canswim.dashboard.url_state import (
    CHART_URL_SYNC_JS,
    DEFAULT_LOWQ,
    parse_chart_query,
    resolve_chart_ticker,
)


def test_url_sync_js_uses_replace_state():
    assert "history.replaceState" in CHART_URL_SYNC_JS
    assert "searchParams" in CHART_URL_SYNC_JS
    assert "ticker" in CHART_URL_SYNC_JS
    assert "lowq" in CHART_URL_SYNC_JS


class TestParseChartQuery:
    def test_missing_params(self):
        assert parse_chart_query({}) == (None, DEFAULT_LOWQ)
        assert parse_chart_query(None) == (None, DEFAULT_LOWQ)

    def test_ticker_normalized_upper(self):
        t, q = parse_chart_query({"ticker": "aapl"})
        assert t == "AAPL"
        assert q == DEFAULT_LOWQ

    def test_ticker_stripped(self):
        t, _ = parse_chart_query({"ticker": "  msft  "})
        assert t == "MSFT"

    def test_blank_ticker_is_none(self):
        t, _ = parse_chart_query({"ticker": "   "})
        assert t is None

    def test_lowq_valid(self):
        assert parse_chart_query({"lowq": "95"})[1] == 95
        assert parse_chart_query({"lowq": 99})[1] == 99
        assert parse_chart_query({"lowq": "80"})[1] == 80

    def test_lowq_invalid_falls_back(self):
        assert parse_chart_query({"lowq": "50"})[1] == DEFAULT_LOWQ
        assert parse_chart_query({"lowq": "nope"})[1] == DEFAULT_LOWQ
        assert parse_chart_query({"lowq": ""})[1] == DEFAULT_LOWQ

    def test_ticker_and_lowq_together(self):
        t, q = parse_chart_query({"ticker": "nvda", "lowq": "95"})
        assert t == "NVDA"
        assert q == 95

    def test_unrelated_params_ignored(self):
        t, q = parse_chart_query({"foo": "bar", "ticker": "IBM"})
        assert t == "IBM"
        assert q == DEFAULT_LOWQ


class TestResolveChartTicker:
    choices = ["AAPL", "MSFT", "GOOG"]

    def test_exact_match(self):
        assert resolve_chart_ticker("AAPL", self.choices, "MSFT") == "AAPL"

    def test_case_insensitive_match(self):
        assert resolve_chart_ticker("aapl", self.choices, "MSFT") == "AAPL"
        assert resolve_chart_ticker("MsFt", self.choices, "AAPL") == "MSFT"

    def test_unknown_falls_back_to_default(self):
        assert resolve_chart_ticker("NOTREAL", self.choices, "AAPL") == "AAPL"

    def test_none_requested_uses_default(self):
        assert resolve_chart_ticker(None, self.choices, "GOOG") == "GOOG"

    def test_empty_choices_uses_default(self):
        assert resolve_chart_ticker("AAPL", [], "X") == "X"
        assert resolve_chart_ticker("AAPL", None, "X") == "X"

    def test_blank_requested_uses_default(self):
        assert resolve_chart_ticker("  ", self.choices, "MSFT") == "MSFT"
