"""Dashboard Run tab: gather data + week-aligned forecast for user tickers."""

from __future__ import annotations

import json

import gradio as gr
from loguru import logger

from canswim.run_triggers import (
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
            """
### Run gather & forecast
Enter one or more tickers (comma or newline separated).  
**Gather** pulls local market data for those symbols.  
**Forecast** runs TiDE for that list with a **week-aligned** start date
(snapped to the first NYSE session of the market week; default / today → live week origin).
            """
        )
        with gr.Row():
            self.tickers = gr.Textbox(
                label="Tickers",
                lines=4,
                placeholder="AAPL, MSFT\nNVDA",
                info="Comma and/or newline separated. Max 50 symbols per run.",
            )
        with gr.Row():
            self.forecastDate = gr.Textbox(
                label="Forecast start date (optional YYYY-MM-DD)",
                value="",
                placeholder="Leave empty for live week-start default",
                info=(
                    "Past dates snap to the market week start on or before the pick. "
                    "Empty or today → live default (week start after latest week-end close). "
                    "Future dates beyond the live default are rejected."
                ),
            )
            self.resolveBtn = gr.Button("Preview start date", scale=0)
        with gr.Row():
            self.gatherBtn = gr.Button("Gather data", variant="secondary")
            self.forecastBtn = gr.Button("Run forecast", variant="primary")
        self.status = gr.Markdown(
            value="_Enter tickers, then Gather and/or Run forecast._"
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
