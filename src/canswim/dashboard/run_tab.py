"""Dashboard: separate Get market data vs Run a forecast panels.

Uses the same ``canswim.run_triggers`` orchestration as CLI and MCP.
Consumer UI shows a short status line; full JSON is under Advanced details.
"""

from __future__ import annotations

import json

import gradio as gr
from loguru import logger

from canswim.db import (
    format_search_db_status_markdown,
    init_search_db,
    list_tickers,
    search_db_status,
)
from canswim.run_triggers import (
    FORECAST_BUTTON,
    FORECAST_SECTION_HELP,
    FORECAST_SECTION_TITLE,
    FORECAST_START_HELP,
    GATHER_BUTTON,
    GATHER_SECTION_HELP,
    GATHER_SECTION_TITLE,
    PREVIEW_START_BUTTON,
    REFRESH_SEARCH_BUTTON,
    REFRESH_SEARCH_SECTION_HELP,
    REFRESH_SEARCH_SECTION_TITLE,
    TICKERS_HELP,
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
    resolve_start_for_run,
)


def _json_details(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


def _gather_summary(result: dict) -> str:
    if not result.get("ok"):
        err = result.get("error") or "Unknown error."
        short = result.get("short_history") or []
        none = result.get("no_history") or []
        if short or none or "IPO" in err or "trading history" in err:
            return f"❌ **Could not finish for every symbol.**  \n{err}"
        return f"❌ **Could not update market data.**  \n{err}"
    tickers = result.get("tickers") or []
    ready = result.get("ready") or []
    incomplete = result.get("incomplete") or []
    skipped = result.get("skipped_remote") or []
    fetched = result.get("fetched") or []
    added = (result.get("db_sync") or {}).get("added") or []
    short = result.get("short_history") or []
    if result.get("partial") and incomplete:
        show = ", ".join(ready) if ready else "—"
        lines = [
            f"⚠️ **Partial success** — ready: {show}.",
            result.get("error")
            or (
                f"Not ready (often recent IPOs / short history): "
                f"{', '.join(incomplete)}."
            ),
        ]
        # Prefer full friendly text from messages if present
        for m in result.get("messages") or []:
            if "trading history" in m or "No usable price" in m:
                lines = [
                    f"⚠️ **Partial success** — ready: {show}.",
                    m,
                ]
                break
        if short:
            lines.append(
                "Tip: drop recent IPOs from the list, then run **Update market data** again."
            )
    else:
        show = ", ".join(ready or tickers) or "—"
        lines = [f"✅ **Market data ready** for {show}."]
    if skipped and not fetched:
        lines.append("Already up to date locally — no download needed.")
    elif skipped:
        lines.append(f"Already local: {', '.join(skipped)}.")
    if fetched:
        lines.append(f"Downloaded or refreshed: {', '.join(fetched)}.")
    if added:
        lines.append(f"Added to Charts list: {', '.join(added)}.")
    return "  \n".join(lines)


def _forecast_summary(result: dict) -> str:
    start = (result.get("resolved_start") or {}).get("start") or "—"
    if result.get("dry_run"):
        already = result.get("already_have_forecast") or []
        if already:
            return (
                f"ℹ️ **Check only** — start date `{start}`.  \n"
                f"Already on file (would skip): {', '.join(already)}."
            )
        return f"ℹ️ **Check only** — start date `{start}`. No model run."
    if result.get("already_saved") and not result.get("forecasted"):
        already = result.get("already_have_forecast") or result.get("tickers") or []
        return (
            f"✅ **Already done** for {', '.join(already)} "
            f"(start `{start}`).  \n"
            "Skipped re-run — no new forecast files written."
        )
    if not result.get("ok"):
        if result.get("need_covariates"):
            head = "❌ **Forecast inputs incomplete.**"
        elif result.get("need_gather"):
            head = "❌ **Need more market data first.**"
        else:
            head = "❌ **Forecast failed.**"
        return f"{head}  \n{result.get('error') or 'Unknown error.'}"
    forecasted = result.get("forecasted") or []
    already = result.get("already_have_forecast") or []
    lines = [f"✅ **Forecast complete** (start `{start}`)."]
    if forecasted:
        lines.append(f"New: {', '.join(forecasted)}.")
    if already:
        lines.append(f"Already on file (skipped): {', '.join(already)}.")
    return "  \n".join(lines)


def _start_summary(info: dict) -> str:
    if info.get("ok"):
        return (
            f"✅ **Start date:** `{info.get('start')}`  \n"
            f"Default if left blank: `{info.get('live_default')}`."
        )
    return f"❌ **Could not use that date.**  \n{info.get('error') or ''}"


class RunTab:
    def __init__(
        self,
        canswim_model=None,
        db_path=None,
        charts_ticker_dropdown=None,
        db_status_banner=None,
    ):
        self.canswim_model = canswim_model
        self.db_path = db_path
        self.charts_ticker_dropdown = charts_ticker_dropdown
        self.db_status_banner = db_status_banner

        gr.Markdown(
            """
## Update data & run forecasts
Two separate steps: **get market data**, then **run a forecast**.
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
            self.gatherBtn = gr.Button(GATHER_BUTTON, variant="primary")
            self.gatherStatus = gr.Markdown(
                value="_Enter symbols, then click Update market data._"
            )
            with gr.Accordion("Advanced details", open=False):
                self.gatherDetails = gr.Code(
                    label="Technical log",
                    language="json",
                    lines=8,
                    interactive=False,
                    value="",
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
            with gr.Accordion("Advanced details", open=False):
                self.forecastDetails = gr.Code(
                    label="Technical log",
                    language="json",
                    lines=8,
                    interactive=False,
                    value="",
                )

        gr.Markdown("---")

        # ---- Search DB refresh (parquet → DuckDB) ----
        with gr.Group():
            gr.Markdown(
                f"### {REFRESH_SEARCH_SECTION_TITLE}\n{REFRESH_SEARCH_SECTION_HELP}"
            )
            self.refreshDbBtn = gr.Button(REFRESH_SEARCH_BUTTON, variant="secondary")
            initial_status = ""
            if self.db_path:
                try:
                    initial_status = format_search_db_status_markdown(
                        search_db_status(self.db_path)
                    )
                except Exception as e:
                    logger.debug(f"initial search db status: {e}")
                    initial_status = "_Search DB status unavailable._"
            self.refreshDbStatus = gr.Markdown(value=initial_status)
            with gr.Accordion("Advanced details", open=False):
                self.refreshDbDetails = gr.Code(
                    label="Technical log",
                    language="json",
                    lines=8,
                    interactive=False,
                    value="",
                )

        gather_outputs = [self.gatherStatus, self.gatherDetails]
        if self.charts_ticker_dropdown is not None:
            gather_outputs.append(self.charts_ticker_dropdown)

        self.gatherBtn.click(
            fn=self.do_gather,
            inputs=[self.gatherTickers],
            outputs=gather_outputs,
        )
        self.resolveBtn.click(
            fn=self.preview_start,
            inputs=[self.forecastDate],
            outputs=[self.forecastStatus, self.forecastDetails],
        )
        self.forecastBtn.click(
            fn=self.do_forecast,
            inputs=[self.forecastTickers, self.forecastDate],
            outputs=[self.forecastStatus, self.forecastDetails],
        )

        refresh_outputs = [self.refreshDbStatus, self.refreshDbDetails]
        if self.db_status_banner is not None:
            refresh_outputs.append(self.db_status_banner)
        if self.charts_ticker_dropdown is not None:
            refresh_outputs.append(self.charts_ticker_dropdown)
        self.refreshDbBtn.click(
            fn=self.do_refresh_search_db,
            inputs=[],
            outputs=refresh_outputs,
        )

    def _charts_dropdown_update(self):
        if self.charts_ticker_dropdown is None or not self.db_path:
            return None
        try:
            choices = sorted(list_tickers(self.db_path))
            current = choices[0] if choices else None
            return gr.update(choices=choices, value=current)
        except Exception as e:
            logger.warning(f"Could not refresh Charts dropdown: {e}")
            return gr.update()

    def preview_start(self, forecast_date):
        info = resolve_start_for_run(forecast_date or None)
        return _start_summary(info), _json_details(info)

    def do_gather(self, tickers_text):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            status = (
                f"❌ **Could not update market data.**  \n"
                f"{parsed.get('error') or 'Invalid symbols.'}"
            )
            details = _json_details(parsed)
            if self.charts_ticker_dropdown is not None:
                return status, details, gr.update()
            return status, details
        logger.info(f"Dashboard gather for {parsed['tickers']}")
        result = gather_for_tickers(tickers_text, force_allow=True)
        status = _gather_summary(result)
        details = _json_details(result)
        if self.charts_ticker_dropdown is not None:
            return status, details, self._charts_dropdown_update()
        return status, details

    def do_forecast(self, tickers_text, forecast_date):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            return (
                f"❌ **Forecast failed.**  \n{parsed.get('error') or 'Invalid symbols.'}",
                _json_details(parsed),
            )
        start_preview = resolve_start_for_run(forecast_date or None)
        if not start_preview.get("ok"):
            return (
                f"❌ **Forecast failed (start date).**  \n"
                f"{start_preview.get('error') or ''}",
                _json_details(start_preview),
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
        return _forecast_summary(result), _json_details(result)

    def do_refresh_search_db(self):
        """Rebuild DuckDB search cache from local parquet (full refresh)."""
        banner = "_Search DB status unavailable._"
        if not self.db_path:
            status = "❌ **No database path configured.**"
            details = _json_details({"ok": False, "error": "db_path missing"})
            return self._refresh_outputs(status, details, banner, dropdown=False)

        target_col = "Close"
        if self.canswim_model is not None:
            target_col = getattr(self.canswim_model, "target_column", None) or "Close"

        logger.info(f"Refreshing search DB from parquet → {self.db_path}")
        try:
            result = init_search_db(
                self.db_path,
                same_data=False,
                target_column=target_col,
            )
            st = result.get("status") or search_db_status(self.db_path)
            banner = format_search_db_status_markdown(st, mode="rebuilt")
            if st.get("ok"):
                head = f"✅ **Search DB rebuilt** from parquet → `{self.db_path}`."
            else:
                head = (
                    f"⚠️ **Rebuild finished but search DB looks incomplete.**  \n"
                    f"Path: `{self.db_path}`"
                )
            status = head + "  \n\n" + banner
            details = _json_details(result)
        except Exception as e:
            logger.exception("Search DB refresh failed")
            result = {"ok": False, "error": str(e)}
            try:
                st = search_db_status(self.db_path)
                banner = format_search_db_status_markdown(st)
            except Exception:
                banner = f"_Could not read status: {e}_"
            status = f"❌ **Could not rebuild search DB.**  \n{e}  \n\n{banner}"
            details = _json_details(result)

        return self._refresh_outputs(status, details, banner, dropdown=True)

    def _refresh_outputs(self, status, details, banner, *, dropdown: bool):
        outs = [status, details]
        if self.db_status_banner is not None:
            outs.append(banner)
        if self.charts_ticker_dropdown is not None:
            outs.append(
                self._charts_dropdown_update() if dropdown else gr.update()
            )
        return tuple(outs)
