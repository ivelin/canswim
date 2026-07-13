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


def _norm_syms(symbols) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols or []:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _fmt_syms(symbols, *, limit: int = 10) -> str:
    """Compact ticker list for primary UI (full list lives in Advanced details)."""
    syms = _norm_syms(symbols)
    if not syms:
        return "—"
    if len(syms) <= limit:
        return ", ".join(syms)
    head = ", ".join(syms[:limit])
    return f"{head} … +{len(syms) - limit} more"


def _copy_line(symbols) -> str:
    """Single pasteable line for the forecast box."""
    syms = _norm_syms(symbols)
    return ", ".join(syms) if syms else ""


def _gather_summary(result: dict) -> str:
    """Consumer-facing gather status: buckets, counts, clear next steps."""
    tickers = _norm_syms(result.get("tickers") or [])
    ready = _norm_syms(result.get("ready") or [])
    incomplete = _norm_syms(result.get("incomplete") or [])
    short = _norm_syms(result.get("short_history") or [])
    no_hist = _norm_syms(result.get("no_history") or [])
    skipped = _norm_syms(result.get("skipped_remote") or [])
    fetched = _norm_syms(result.get("fetched") or [])
    added = _norm_syms((result.get("db_sync") or {}).get("added") or [])
    n_req = len(tickers) or (len(ready) + len(incomplete))
    n_ready = len(ready)
    n_not = len(incomplete) if incomplete else len(short) + len(no_hist)

    # ---- hard failure (nothing forecast-ready) ----
    if not result.get("ok"):
        err = (result.get("error") or "").strip()
        is_hist = bool(
            short
            or no_hist
            or "IPO" in err
            or "trading history" in err.lower()
            or "not enough" in err.lower()
        )
        lines: list[str] = []
        if is_hist:
            lines.append(
                f"❌ **None of these symbols are ready for forecasts yet** "
                f"({n_not or n_req} checked)."
            )
            if short:
                lines.append(
                    f"**Short history / recent listings ({len(short)}):** "
                    f"{_fmt_syms(short)}"
                )
            if no_hist:
                lines.append(
                    f"**No usable price history ({len(no_hist)}):** "
                    f"{_fmt_syms(no_hist)}"
                )
            if not short and not no_hist and incomplete:
                lines.append(
                    f"**Not ready ({len(incomplete)}):** {_fmt_syms(incomplete)}"
                )
            lines.append("")
            lines.append("**What this means**")
            lines.append(
                "Forecasts need about **two years** of trading sessions. "
                "Recent IPOs and new listings usually cannot be forecasted yet—"
                "even if some prices were saved."
            )
            lines.append("")
            lines.append("**What you can do next**")
            lines.append(
                "1. **Recommended:** Remove those names from the list and "
                "**Update market data** only for symbols with longer history, "
                "then **Run forecast**."
            )
            lines.append(
                "2. Keep watching the short-history names in Charts later—"
                "they become usable as more sessions accumulate."
            )
            lines.append(
                "3. Full lists and technical notes are under **Advanced details**."
            )
        else:
            lines.append("❌ **Could not update market data.**")
            if err:
                # Keep error short in primary UI
                short_err = err if len(err) <= 220 else err[:217] + "…"
                lines.append(short_err)
            lines.append("")
            lines.append("**What you can do next**")
            lines.append(
                "1. **Recommended:** Retry **Update market data** with a shorter list."
            )
            lines.append(
                "2. Check API keys / network, then open **Advanced details** for the log."
            )
        return "\n\n".join(lines)

    # ---- success (all or partial) ----
    partial = bool(result.get("partial") and incomplete)
    lines = []
    if partial:
        lines.append(
            f"⚠️ **Partial update** — **{n_ready} of {n_req or (n_ready + n_not)}** "
            f"symbols are ready for forecasts."
        )
    else:
        show_n = n_ready or len(tickers)
        lines.append(
            f"✅ **Market data ready** — **{show_n}** symbol"
            f"{'s' if show_n != 1 else ''} can be forecasted."
        )

    # Buckets (counts first; compact symbol lists)
    if ready:
        lines.append(
            f"**Ready for forecasts ({n_ready}):** {_fmt_syms(ready)}"
        )
        if n_ready > 1:
            lines.append(f"Paste into forecast box: `{_copy_line(ready)}`")

    if incomplete:
        why = "often recent IPOs or short listings"
        if short and not no_hist:
            why = "short trading history / recent listings"
        elif no_hist and not short:
            why = "no usable price history"
        lines.append(
            f"**Not ready for forecasts yet ({len(incomplete)}):** "
            f"{_fmt_syms(incomplete)}  \n"
            f"_Why: {why}. Prices may still be on file; forecasts need ~2 years of sessions._"
        )

    # Download activity as counts (avoid repeating every ticker twice)
    dl_bits = []
    if skipped and not fetched:
        dl_bits.append(f"already up to date locally ({len(skipped)})")
    else:
        if skipped:
            dl_bits.append(f"already local, no download ({len(skipped)})")
        if fetched:
            dl_bits.append(f"downloaded or refreshed ({len(fetched)})")
    if added:
        dl_bits.append(f"new on Charts list: {_fmt_syms(added, limit=8)}")
    if dl_bits:
        lines.append("**Downloads:** " + " · ".join(dl_bits))

    lines.append("**What you can do next**")
    if ready and incomplete:
        lines.append(
            "1. **Recommended:** Copy the ready line above into "
            "**Symbols to forecast** and click **Run forecast**."
        )
        lines.append(
            "2. Leave the not-ready names off the forecast list for now "
            "(try again in months as history grows)."
        )
        lines.append(
            "3. Optional: trim them from **Symbols to update** before the next gather "
            "to keep the status quieter."
        )
    elif ready:
        lines.append(
            "1. **Recommended:** Enter the same symbols under **Run a forecast** "
            "and click **Run forecast**."
        )
        lines.append(
            "2. Or open **Charts** to review a symbol visually first."
        )
    else:
        lines.append(
            "1. **Recommended:** Narrow the list to longer-history names and "
            "run **Update market data** again."
        )
    lines.append(
        "Full symbol lists and the technical log are under **Advanced details**."
    )
    return "\n\n".join(lines)


def _forecast_summary(result: dict) -> str:
    start = (result.get("resolved_start") or {}).get("start") or "—"
    if result.get("dry_run"):
        already = _norm_syms(result.get("already_have_forecast") or [])
        if already:
            return (
                f"ℹ️ **Check only** — start date `{start}`.\n\n"
                f"**Would skip (already on file, {len(already)}):** "
                f"{_fmt_syms(already)}\n\n"
                "No model run. Click **Run forecast** when ready."
            )
        return (
            f"ℹ️ **Check only** — start date `{start}`.\n\n"
            "No model run. Click **Run forecast** when ready."
        )
    if result.get("already_saved") and not result.get("forecasted"):
        already = _norm_syms(
            result.get("already_have_forecast") or result.get("tickers") or []
        )
        return (
            f"✅ **Already done** for **{len(already)}** symbol"
            f"{'s' if len(already) != 1 else ''} (start `{start}`).\n\n"
            f"{_fmt_syms(already)}\n\n"
            "Skipped re-run — no new forecast files written.\n\n"
            "**Next:** open **Charts** or **Scans** to review results."
        )
    if not result.get("ok"):
        if result.get("need_covariates"):
            head = "❌ **Forecast inputs incomplete.**"
            nexts = (
                "1. **Recommended:** **Update market data** for these symbols "
                "(includes fundamentals), then **Run forecast** again.\n\n"
                "2. See **Advanced details** for which inputs failed."
            )
        elif result.get("need_gather"):
            head = "❌ **Need more market data first.**"
            nexts = (
                "1. **Recommended:** **Update market data** for these symbols, "
                "then **Run forecast**.\n\n"
                "2. Recent IPOs may never be ready until enough history exists."
            )
        else:
            head = "❌ **Forecast failed.**"
            nexts = (
                "1. **Recommended:** Retry with a shorter list.\n\n"
                "2. Open **Advanced details** for the error log."
            )
        err = (result.get("error") or "Unknown error.").strip()
        short_err = err if len(err) <= 280 else err[:277] + "…"
        return f"{head}\n\n{short_err}\n\n**What you can do next**\n\n{nexts}"

    forecasted = _norm_syms(result.get("forecasted") or [])
    already = _norm_syms(result.get("already_have_forecast") or [])
    lines = [f"✅ **Forecast complete** (start `{start}`)."]
    if forecasted:
        lines.append(
            f"**New forecasts ({len(forecasted)}):** {_fmt_syms(forecasted)}"
        )
    if already:
        lines.append(
            f"**Already on file, skipped ({len(already)}):** {_fmt_syms(already)}"
        )
    lines.append(
        "**Next:** open **Charts** for a symbol or **Scans** for the whole set."
    )
    return "\n\n".join(lines)


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
