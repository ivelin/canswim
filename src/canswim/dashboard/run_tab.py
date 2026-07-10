"""Dashboard Run tab: gather data + week-aligned forecast for user tickers.

Uses the same ``canswim.run_triggers`` orchestration as:
- CLI: ``python -m canswim gatherdata|forecast --tickers …``
- MCP: ``gather_tickers`` / ``forecast_tickers`` (opt-in)
"""

from __future__ import annotations

import json

import gradio as gr
from loguru import logger

from canswim.run_triggers import (
    DATE_POLICY_SUMMARY,
    FORECAST_START_HELP,
    TICKERS_HELP,
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
    resolve_start_for_run,
)


def _fmt_result(payload: dict) -> str:
    return f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"


class RunTab:
    def __init__(self, canswim_model=None, db_path=None):
        # model/db unused for triggers but kept for dashboard consistency
        self.canswim_model = canswim_model
        self.db_path = db_path

        gr.Markdown(
            f"""
### Run gather & forecast
Same pipeline as **CLI** (`--tickers`) and **MCP** (`gather_tickers` / `forecast_tickers`).

{DATE_POLICY_SUMMARY}

**Gather** → local market data for the listed symbols (not the full universe).  
**Forecast** → TiDE for that list at the resolved week-aligned start.  
**Preview start date** → see the snap/default without running the model.
            """
        )
        with gr.Row():
            self.tickers = gr.Textbox(
                label="Tickers",
                lines=4,
                placeholder="AAPL, MSFT\nNVDA",
                info=TICKERS_HELP,
            )
        with gr.Row():
            self.forecastDate = gr.Textbox(
                label="Forecast start date (optional YYYY-MM-DD)",
                value="",
                placeholder="Leave empty for live week-start default",
                info=FORECAST_START_HELP,
            )
            self.resolveBtn = gr.Button("Preview start date", scale=0)
        with gr.Row():
            self.gatherBtn = gr.Button("Gather data", variant="secondary")
            self.forecastBtn = gr.Button("Run forecast", variant="primary")
        self.status = gr.Markdown(
            value=(
                "_Enter tickers, then Gather and/or Run forecast. "
                "CLI equivalent: `python -m canswim gatherdata --tickers '…'` / "
                "`forecast --tickers '…' [--forecast_start_date YYYY-MM-DD]`._"
            )
        )

        self.resolveBtn.click(
            fn=self.preview_start,
            inputs=[self.forecastDate],
            outputs=[self.status],
        )
        self.gatherBtn.click(
            fn=self.do_gather,
            inputs=[self.tickers],
            outputs=[self.status],
        )
        self.forecastBtn.click(
            fn=self.do_forecast,
            inputs=[self.tickers, self.forecastDate],
            outputs=[self.status],
        )

    def preview_start(self, forecast_date):
        info = resolve_start_for_run(forecast_date or None)
        if info.get("ok"):
            return (
                f"**Resolved start:** `{info.get('start')}` "
                f"({info.get('reason')}) · live default `{info.get('live_default')}` · "
                f"latest_close used `{info.get('latest_close_used')}`\n\n"
                + _fmt_result(info)
            )
        return f"**Could not resolve start:** {info.get('error')}\n\n{_fmt_result(info)}"

    def do_gather(self, tickers_text):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            return f"**Gather aborted:** {parsed.get('error')}\n\n{_fmt_result(parsed)}"
        logger.info(f"Dashboard gather for {parsed['tickers']}")
        # Dashboard is an explicit local user action
        result = gather_for_tickers(tickers_text, force_allow=True)
        if result.get("ok"):
            return (
                f"**Gather OK** for `{', '.join(result.get('tickers') or [])}`\n\n"
                + _fmt_result(result)
            )
        return f"**Gather failed:** {result.get('error')}\n\n{_fmt_result(result)}"

    def do_forecast(self, tickers_text, forecast_date):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            return f"**Forecast aborted:** {parsed.get('error')}\n\n{_fmt_result(parsed)}"
        start_preview = resolve_start_for_run(forecast_date or None)
        if not start_preview.get("ok"):
            return (
                f"**Forecast aborted (start date):** {start_preview.get('error')}\n\n"
                + _fmt_result(start_preview)
            )
        logger.info(
            f"Dashboard forecast for {parsed['tickers']} start={start_preview.get('start')}"
        )
        result = forecast_for_tickers(
            tickers_text,
            forecast_start_date=forecast_date or None,
            force_allow=True,
        )
        if result.get("ok"):
            rs = (result.get("resolved_start") or {}).get("start")
            return (
                f"**Forecast OK** · resolved start `{rs}` · "
                f"forecasted `{result.get('forecasted')}` · "
                f"skipped `{result.get('skipped')}`\n\n"
                + _fmt_result(result)
            )
        return f"**Forecast failed:** {result.get('error')}\n\n{_fmt_result(result)}"
