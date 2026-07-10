"""Dashboard: separate Get market data vs Run a forecast panels.

Uses the same ``canswim.run_triggers`` orchestration as CLI and MCP.
"""

from __future__ import annotations

import json

import gradio as gr
from loguru import logger

from canswim.run_triggers import (
    FORECAST_BUTTON,
    FORECAST_SECTION_HELP,
    FORECAST_SECTION_TITLE,
    FORECAST_START_HELP,
    GATHER_BUTTON,
    GATHER_SECTION_HELP,
    GATHER_SECTION_TITLE,
    PREVIEW_START_BUTTON,
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
        self.canswim_model = canswim_model
        self.db_path = db_path

        gr.Markdown(
            """
## Update data & run forecasts
Two separate steps: **get market data**, then **run a forecast**.  
Use the same stock symbols in both when you want a full path from download to prediction.
            """
        )

        # ---- Gather panel ----
        with gr.Group():
            gr.Markdown(f"### {GATHER_SECTION_TITLE}\n{GATHER_SECTION_HELP}")
            self.gatherTickers = gr.Textbox(
                label="Symbols to update",
                lines=3,
                placeholder="AAPL, MSFT",
                info=TICKERS_HELP,
            )
            self.gatherBtn = gr.Button(GATHER_BUTTON, variant="secondary")
            self.gatherStatus = gr.Markdown(
                value="_Enter symbols, then update market data._"
            )

        gr.Markdown("---")

        # ---- Forecast panel ----
        with gr.Group():
            gr.Markdown(f"### {FORECAST_SECTION_TITLE}\n{FORECAST_SECTION_HELP}")
            self.forecastTickers = gr.Textbox(
                label="Symbols to forecast",
                lines=3,
                placeholder="AAPL, MSFT",
                info=TICKERS_HELP,
            )
            with gr.Row():
                self.forecastDate = gr.Textbox(
                    label="Start date (optional)",
                    value="",
                    placeholder="YYYY-MM-DD — leave blank for the default",
                    info=FORECAST_START_HELP,
                )
                self.resolveBtn = gr.Button(PREVIEW_START_BUTTON, scale=0)
            self.forecastBtn = gr.Button(FORECAST_BUTTON, variant="primary")
            self.forecastStatus = gr.Markdown(
                value="_Enter symbols, optionally check the start date, then run a forecast._"
            )

        self.gatherBtn.click(
            fn=self.do_gather,
            inputs=[self.gatherTickers],
            outputs=[self.gatherStatus],
        )
        self.resolveBtn.click(
            fn=self.preview_start,
            inputs=[self.forecastDate],
            outputs=[self.forecastStatus],
        )
        self.forecastBtn.click(
            fn=self.do_forecast,
            inputs=[self.forecastTickers, self.forecastDate],
            outputs=[self.forecastStatus],
        )

    def preview_start(self, forecast_date):
        info = resolve_start_for_run(forecast_date or None)
        if info.get("ok"):
            return (
                f"**Start date:** `{info.get('start')}`  \n"
                f"Default if left blank: `{info.get('live_default')}`\n\n"
                + _fmt_result(info)
            )
        return (
            f"**Could not use that date:** {info.get('error')}\n\n"
            + _fmt_result(info)
        )

    def do_gather(self, tickers_text):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            return (
                f"**Update failed:** {parsed.get('error')}\n\n"
                + _fmt_result(parsed)
            )
        logger.info(f"Dashboard gather for {parsed['tickers']}")
        result = gather_for_tickers(tickers_text, force_allow=True)
        if result.get("ok"):
            skipped = result.get("skipped_remote") or []
            fetched = result.get("fetched") or []
            summary = (
                f"**Market data updated** for `{', '.join(result.get('tickers') or [])}`"
            )
            if skipped:
                summary += f"  \nAlready local: {', '.join(skipped)}"
            if fetched:
                summary += f"  \nDownloaded/refreshed: {', '.join(fetched)}"
            return summary + "\n\n" + _fmt_result(result)
        return f"**Update failed:** {result.get('error')}\n\n{_fmt_result(result)}"

    def do_forecast(self, tickers_text, forecast_date):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            return (
                f"**Forecast failed:** {parsed.get('error')}\n\n"
                + _fmt_result(parsed)
            )
        start_preview = resolve_start_for_run(forecast_date or None)
        if not start_preview.get("ok"):
            return (
                f"**Forecast failed (start date):** {start_preview.get('error')}\n\n"
                + _fmt_result(start_preview)
            )
        logger.info(
            f"Dashboard forecast for {parsed['tickers']} "
            f"start={start_preview.get('start')}"
        )
        result = forecast_for_tickers(
            tickers_text,
            forecast_start_date=forecast_date or None,
            force_allow=True,
        )
        if result.get("ok"):
            rs = (result.get("resolved_start") or {}).get("start")
            return (
                f"**Forecast complete** · start `{rs}` · "
                f"symbols `{result.get('forecasted')}`\n\n"
                + _fmt_result(result)
            )
        need = result.get("need_gather")
        prefix = "**Need more market data**" if need else "**Forecast failed**"
        return f"{prefix}: {result.get('error')}\n\n{_fmt_result(result)}"
