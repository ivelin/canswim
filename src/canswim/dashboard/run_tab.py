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
    REFRESH_SYMBOLS_BUTTON,
    REFRESH_SYMBOLS_SECTION_HELP,
    REFRESH_SYMBOLS_SECTION_TITLE,
    TICKERS_HELP,
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
    refresh_symbols,
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
            "1. **Recommended:** Paste the ready line into **Refresh symbols** "
            "(or **Run forecast** with blank start for catch-up)."
        )
        lines.append(
            "2. Leave the not-ready names off for now "
            "(try again as history grows)."
        )
        lines.append(
            "3. Optional: trim them from the list next time to keep status quieter."
        )
    elif ready:
        lines.append(
            "1. **Recommended:** **Refresh symbols** or **Run forecast** "
            "(blank start = monthly catch-up + live)."
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
    mode = result.get("mode") or "single"
    origins = result.get("origins") or []
    start = (result.get("resolved_start") or {}).get("start") or "—"
    n_origins = len(origins) if origins else 1

    if result.get("dry_run"):
        need = 0
        for v in (result.get("by_start") or {}).values():
            need += len(v.get("to_run") or [])
        if mode == "catchup":
            return (
                f"ℹ️ **Check only — catch-up** ({n_origins} monthly/live origins).\n\n"
                f"**Jobs that would run:** {need} symbol×start pair(s).\n\n"
                "No model run. Click **Run forecast** or **Refresh symbols** when ready."
            )
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
        head = (
            f"✅ **Already caught up** ({n_origins} origin(s))."
            if mode == "catchup"
            else f"✅ **Already done** (start `{start}`)."
        )
        return (
            f"{head}\n\n{_fmt_syms(already)}\n\n"
            "Skipped re-run — no new forecast files written.\n\n"
            "**Next:** open **Charts** or **Scans** to review results."
        )
    if not result.get("ok"):
        if result.get("need_covariates"):
            head = "❌ **Forecast inputs incomplete.**"
            nexts = (
                "1. **Recommended:** **Refresh symbols** or **Update market data**, "
                "then try again.\n\n"
                "2. See **Advanced details** for which inputs failed."
            )
        elif result.get("need_gather"):
            head = "❌ **Need more market data first.**"
            nexts = (
                "1. **Recommended:** **Refresh symbols** (gather + forecast) "
                "or **Update market data** then **Run forecast**.\n\n"
                "2. Recent IPOs may need more history first."
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
    incomplete_starts = result.get("incomplete_starts") or []
    n_miss = len(incomplete_starts)
    if mode == "catchup":
        if result.get("partial") or n_miss:
            n_ok = max(n_origins - n_miss, 0)
            lines = [
                f"⚠️ **Catch-up mostly done** — **{n_ok} of {n_origins}** "
                f"monthly/live origins OK"
                + (f" ({n_miss} still missing)." if n_miss else ".")
            ]
        else:
            lines = [
                f"✅ **Catch-up complete** — all **{n_origins}** monthly/live origins OK."
            ]
    else:
        lines = [f"✅ **Forecast complete** (start `{start}`)."]
        if result.get("partial"):
            lines[0] = "⚠️ " + lines[0].replace("✅ ", "")
    if forecasted:
        lines.append(f"**Symbols with new forecast data:** {_fmt_syms(forecasted)}")
    elif not n_miss:
        lines.append("No new files needed — everything was already on disk.")
    if n_miss and n_miss <= 5:
        lines.append(f"**Still missing:** {', '.join(incomplete_starts)}")
    elif n_miss:
        lines.append(f"**Still missing:** {n_miss} symbol×start pairs (see Advanced).")
    lines.append("**Next:** open **Charts** for that symbol, or **Scans** to rank.")
    return "\n\n".join(lines)


def _refresh_summary(result: dict) -> str:
    """Primary status for all-in-one Refresh symbols."""
    gather = result.get("gather") or {}
    forecast = result.get("forecast") or {}
    ready = _norm_syms(result.get("ready") or gather.get("ready") or [])
    incomplete = _norm_syms(
        result.get("incomplete") or gather.get("incomplete") or []
    )
    lines: list[str] = []

    if result.get("dry_run"):
        lines.append("ℹ️ **Check only — Refresh symbols** (no downloads / model).")
        lines.append(f"**Would gather then catch-up:** {_fmt_syms(ready)}")
        if incomplete:
            lines.append(f"**Would skip (not ready):** {_fmt_syms(incomplete)}")
        return "\n\n".join(lines)

    if not result.get("ok") and not forecast.get("forecasted"):
        g_part = _gather_summary(gather) if gather else ""
        head = "❌ **Refresh could not finish.**"
        if incomplete and not ready:
            head = "❌ **Refresh stopped — no symbols ready for forecast yet.**"
        return (
            f"{head}\n\n"
            f"{result.get('error') or ''}\n\n"
            f"{g_part}\n\n"
            "**What you can do next**\n\n"
            "1. **Recommended:** Drop short-history / IPO names and "
            "**Refresh symbols** again.\n\n"
            "2. See **Advanced details** for the full log."
        )

    n_orig = len(forecast.get("origins") or [])
    new_fc = _norm_syms(forecast.get("forecasted") or [])
    if result.get("partial") or incomplete:
        lines.append(
            f"⚠️ **Refresh partial** — **{len(ready)}** ready, "
            f"**{len(incomplete)}** not forecast-ready."
        )
    else:
        lines.append(
            f"✅ **Refresh complete** for **{len(ready)}** symbol"
            f"{'s' if len(ready) != 1 else ''}."
        )
    if ready:
        lines.append(f"**Forecast-ready:** {_fmt_syms(ready)}")
    if incomplete:
        lines.append(
            f"**Not ready (short history etc.):** {_fmt_syms(incomplete)}"
        )
    if n_orig:
        lines.append(
            f"**Catch-up origins:** {n_orig} monthly/live starts "
            f"(skipped ones already on file)."
        )
    if new_fc:
        lines.append(f"**New forecast files:** {_fmt_syms(new_fc)}")
    elif forecast.get("already_saved"):
        lines.append("Forecasts were already on file for all ready origins.")
    lines.append("**What you can do next**")
    lines.append(
        "1. **Recommended:** Open **Scans** or **Charts** — backtest history "
        "and the live forecast should be available for ready names."
    )
    if incomplete:
        lines.append(
            "2. Leave not-ready names for later (need more trading history)."
        )
    lines.append("Full log under **Advanced details**.")
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

        # ---- Primary path only (everything else under More options) ----
        with gr.Group():
            gr.Markdown(
                f"### {REFRESH_SYMBOLS_SECTION_TITLE}\n"
                f"{REFRESH_SYMBOLS_SECTION_HELP}"
            )
            self.refreshTickers = gr.Textbox(
                label="Symbols",
                lines=2,
                placeholder="LLY, AAPL, MSFT",
                info=TICKERS_HELP,
            )
            self.refreshBtn = gr.Button(REFRESH_SYMBOLS_BUTTON, variant="primary")
            self.refreshStatus = gr.Markdown(
                value="_Enter symbols → **Refresh symbols**. That is usually all you need._"
            )
            with gr.Accordion("Technical log", open=False):
                self.refreshDetails = gr.Code(
                    label="JSON log",
                    language="json",
                    lines=6,
                    interactive=False,
                    value="",
                )

        with gr.Accordion("More options (rarely needed)", open=False):
            with gr.Group():
                gr.Markdown(
                    f"#### {GATHER_SECTION_TITLE}\n{GATHER_SECTION_HELP}"
                )
                self.gatherTickers = gr.Textbox(
                    label="Symbols",
                    lines=2,
                    placeholder="AAPL, MSFT",
                )
                self.gatherBtn = gr.Button(GATHER_BUTTON, variant="secondary")
                self.gatherStatus = gr.Markdown(value="")
                with gr.Accordion("Gather log", open=False):
                    self.gatherDetails = gr.Code(
                        language="json",
                        lines=4,
                        interactive=False,
                        value="",
                    )

            with gr.Group():
                gr.Markdown(
                    f"#### {FORECAST_SECTION_TITLE}\n{FORECAST_SECTION_HELP}"
                )
                self.forecastTickers = gr.Textbox(
                    label="Symbols",
                    lines=2,
                    placeholder="LLY",
                )
                with gr.Row():
                    self.forecastDate = gr.Textbox(
                        label="Start date (optional)",
                        value="",
                        placeholder="Blank = monthly catch-up + live",
                        info=FORECAST_START_HELP,
                    )
                    self.resolveBtn = gr.Button(PREVIEW_START_BUTTON, scale=0)
                self.forecastBtn = gr.Button(FORECAST_BUTTON, variant="secondary")
                self.forecastStatus = gr.Markdown(value="")
                with gr.Accordion("Forecast log", open=False):
                    self.forecastDetails = gr.Code(
                        language="json",
                        lines=4,
                        interactive=False,
                        value="",
                    )

            with gr.Group():
                gr.Markdown(
                    f"#### {REFRESH_SEARCH_SECTION_TITLE}\n"
                    f"{REFRESH_SEARCH_SECTION_HELP}"
                )
                self.refreshDbBtn = gr.Button(
                    REFRESH_SEARCH_BUTTON, variant="secondary"
                )
                initial_status = ""
                if self.db_path:
                    try:
                        initial_status = format_search_db_status_markdown(
                            search_db_status(self.db_path)
                        )
                    except Exception as e:
                        logger.debug(f"initial search db status: {e}")
                        initial_status = ""
                self.refreshDbStatus = gr.Markdown(value=initial_status)
                with gr.Accordion("Search DB log", open=False):
                    self.refreshDbDetails = gr.Code(
                        language="json",
                        lines=4,
                        interactive=False,
                        value="",
                    )

        # ---- wire events ----
        refresh_outputs = [self.refreshStatus, self.refreshDetails]
        if self.charts_ticker_dropdown is not None:
            refresh_outputs.append(self.charts_ticker_dropdown)
        self.refreshBtn.click(
            fn=self.do_refresh_symbols,
            inputs=[self.refreshTickers],
            outputs=refresh_outputs,
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

        db_refresh_outputs = [self.refreshDbStatus, self.refreshDbDetails]
        if self.db_status_banner is not None:
            db_refresh_outputs.append(self.db_status_banner)
        if self.charts_ticker_dropdown is not None:
            db_refresh_outputs.append(self.charts_ticker_dropdown)
        self.refreshDbBtn.click(
            fn=self.do_refresh_search_db,
            inputs=[],
            outputs=db_refresh_outputs,
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

    def do_refresh_symbols(self, tickers_text):
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            status = (
                f"❌ **Could not refresh symbols.**  \n"
                f"{parsed.get('error') or 'Invalid symbols.'}"
            )
            details = _json_details(parsed)
            if self.charts_ticker_dropdown is not None:
                return status, details, gr.update()
            return status, details
        logger.info(f"Dashboard refresh_symbols for {parsed['tickers']}")
        result = refresh_symbols(tickers_text, force_allow=True)
        status = _refresh_summary(result)
        details = _json_details(result)
        if self.charts_ticker_dropdown is not None:
            return status, details, self._charts_dropdown_update()
        return status, details

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
