"""Dashboard: separate Get market data vs Run a forecast panels.

Uses the same ``canswim.run_triggers`` orchestration as CLI and MCP.
Consumer UI shows a short status line; full JSON is under Advanced details.
"""

from __future__ import annotations

import json

import gradio as gr
from loguru import logger

from canswim.db import (
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
    REFRESH_SYMBOLS_SECTION_DETAILS,
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
        remote = result.get("remote_api") or {}
        is_remote = bool(
            remote
            or result.get("fail_reason") == "remote_api"
            or "market-data" in err.lower()
            or "api key" in err.lower()
            or "could not reach" in err.lower()
        )
        is_hist = bool(
            not is_remote
            and (
                short
                or no_hist
                or "IPO" in err
                or "trading history" in err.lower()
                or "not enough" in err.lower()
            )
        )
        lines: list[str] = []
        if is_remote:
            lines.append("❌ **Could not update market data from the remote provider.**")
            if err:
                # Gentle multi-line message already includes checklist
                lines.append(err)
            else:
                lines.append(
                    "Check internet access, API key, and subscription status "
                    "(see checklist below)."
                )
            if not err or "Please check:" not in err:
                lines.append("")
                lines.append("**What you can do next**")
                lines.append(
                    "1. **Recommended:** Confirm **internet access** and that "
                    "**FMP_API_KEY** (or your data provider token) is valid and active."
                )
                lines.append(
                    "2. If your plan expired or the key was rotated, update `.env` "
                    "and **restart** the dashboard, then retry."
                )
                lines.append(
                    "3. Open **Technical log** for the raw provider error."
                )
            return "\n\n".join(lines)
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
                "Forecasts need about **three years** of trading sessions. "
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
            f"_Why: {why}. Prices may still be on file; forecasts need ~3 years of sessions._"
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
            "1. **Recommended:** Paste the ready line and click "
            f"**{REFRESH_SYMBOLS_BUTTON}**."
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
            f"1. **Recommended:** **{REFRESH_SYMBOLS_BUTTON}** "
            "(or **Run forecast** under More options)."
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
                f"No model run. Click **{REFRESH_SYMBOLS_BUTTON}** when ready."
            )
        already = _norm_syms(result.get("already_have_forecast") or [])
        if already:
            return (
                f"ℹ️ **Check only** — start date `{start}`.\n\n"
                f"**Would skip (already on file, {len(already)}):** "
                f"{_fmt_syms(already)}\n\n"
                f"No model run. Click **{REFRESH_SYMBOLS_BUTTON}** when ready."
            )
        return (
            f"ℹ️ **Check only** — start date `{start}`.\n\n"
            f"No model run. Click **{REFRESH_SYMBOLS_BUTTON}** when ready."
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
                f"1. **Recommended:** **{REFRESH_SYMBOLS_BUTTON}**, "
                "then try again.\n\n"
                "2. See the technical log for details."
            )
        elif result.get("need_gather"):
            head = "❌ **Need more market data first.**"
            nexts = (
                f"1. **Recommended:** **{REFRESH_SYMBOLS_BUTTON}**.\n\n"
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
    """Primary status for all-in-one Refresh data & forecasts."""
    gather = result.get("gather") or {}
    forecast = result.get("forecast") or {}
    ready = _norm_syms(result.get("ready") or gather.get("ready") or [])
    incomplete = _norm_syms(
        result.get("incomplete") or gather.get("incomplete") or []
    )
    lines: list[str] = []
    btn = REFRESH_SYMBOLS_BUTTON

    if result.get("dry_run"):
        lines.append(f"ℹ️ **Check only — {btn}** (no downloads / model).")
        lines.append(f"**Would update then forecast:** {_fmt_syms(ready)}")
        if incomplete:
            lines.append(f"**Would skip (not ready):** {_fmt_syms(incomplete)}")
        return "\n\n".join(lines)

    if not result.get("ok") and not forecast.get("forecasted"):
        head = f"❌ **{btn} could not finish.**"
        if incomplete and not ready:
            head = (
                f"❌ **{btn} stopped** — no symbols ready for forecast yet."
            )
        err = (result.get("error") or forecast.get("error") or "").strip()
        err_l = err.lower()
        remote = (
            result.get("remote_api")
            or gather.get("remote_api")
            or forecast.get("remote_api")
            or {}
        )
        is_remote = bool(
            remote
            or result.get("fail_reason") == "remote_api"
            or gather.get("fail_reason") == "remote_api"
            or "market-data" in err_l
            or "api key" in err_l
            or "could not reach" in err_l
            or "subscription" in err_l
        )
        is_cov = bool(
            not is_remote
            and (
                forecast.get("need_covariates")
                or result.get("fail_reason") == "covariates"
                or "dimensionality" in err_l
                or "feature mismatch" in err_l
                or "fund-thin" in err_l
                or "past_covariates" in err_l
            )
        )
        is_hist = bool(incomplete and not ready and not is_cov and not is_remote)

        lines = [head]
        if is_remote:
            lines.append(
                "Market data could not be refreshed from the remote provider."
            )
            if err:
                lines.append(err)
            if not err or "Please check:" not in err:
                lines.append("**What you can do next**")
                lines.append(
                    "1. **Recommended:** Check **internet**, **FMP_API_KEY** "
                    "(valid / not revoked), and that your **data plan is active**."
                )
                lines.append(
                    "2. Restart the dashboard after updating keys, then retry."
                )
                lines.append("3. Open **Technical log** for the provider detail.")
            return "\n\n".join(lines)

        if err:
            lines.append(err)
        # Only restate gather hard-fail; skip green gather success + IPO advice
        # when market data was fine and forecast covariates failed.
        if incomplete and not ready:
            g_part = _gather_summary(gather) if gather else ""
            if g_part:
                lines.append(g_part)
        elif ready:
            lines.append(
                f"Market data looked ready for: {_fmt_syms(ready)}. "
                "The forecast step could not finish."
            )

        lines.append("**What you can do next**")
        if is_cov:
            lines.append(
                "1. **Recommended:** Try again after **Update market data** "
                "(fundamentals). ETFs and other fund-thin names use zero-filled "
                "fund inputs like IPOs—if it still fails, open **Technical log**."
            )
            lines.append(
                "2. Or forecast a regular stock (e.g. LLY) to confirm the model path."
            )
        elif is_hist:
            lines.append(
                "1. **Recommended:** Drop short-history / IPO names and try again."
            )
            lines.append("2. Open **Technical log** for details.")
        else:
            lines.append("1. Open **Technical log** for details.")
            lines.append("2. Retry, or try fewer symbols.")
        return "\n\n".join(lines)

    n_orig = len(forecast.get("origins") or [])
    new_fc = _norm_syms(forecast.get("forecasted") or [])
    if result.get("partial") or incomplete:
        lines.append(
            f"⚠️ **{btn} partial** — **{len(ready)}** ready, "
            f"**{len(incomplete)}** not forecast-ready."
        )
    else:
        lines.append(
            f"✅ **{btn} complete** for **{len(ready)}** symbol"
            f"{'s' if len(ready) != 1 else ''}."
        )
    if ready:
        lines.append(f"**Ready for Charts / Scans:** {_fmt_syms(ready)}")
    if incomplete:
        lines.append(
            f"**Not ready yet (short history etc.):** {_fmt_syms(incomplete)}"
        )
    if n_orig:
        lines.append(
            f"**Forecast history:** {n_orig} monthly/live starts "
            f"(already-saved starts were skipped)."
        )
    if new_fc:
        lines.append(f"**New forecasts written for:** {_fmt_syms(new_fc)}")
    elif forecast.get("already_saved"):
        lines.append("Forecasts were already up to date for ready symbols.")
    lines.append("**What you can do next**")
    lines.append(
        "1. **Recommended:** Open **Charts** (already refreshed for the first "
        "ready symbol) or **Scans** to rank by reward/risk and backtest history."
    )
    if incomplete:
        lines.append(
            "2. Leave not-ready names for later (need more trading history)."
        )
    lines.append("Details under **Technical log** if needed.")
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
        charts_tab=None,
        db_status_banner=None,  # unused; kept for call-site compatibility
    ):
        self.canswim_model = canswim_model
        self.db_path = db_path
        # Prefer full charts_tab (replot after refresh); fallback to dropdown only
        self.charts_tab = charts_tab
        self.charts_ticker_dropdown = (
            charts_tab.tickerDropdown
            if charts_tab is not None
            else charts_ticker_dropdown
        )
        self.db_status_banner = None  # no global debug banner

        # ---- Primary path only (everything else under More options) ----
        with gr.Group():
            gr.Markdown(f"### {REFRESH_SYMBOLS_SECTION_TITLE}")
            gr.Markdown(REFRESH_SYMBOLS_SECTION_HELP.strip())
            with gr.Accordion("What this does", open=False):
                gr.Markdown(REFRESH_SYMBOLS_SECTION_DETAILS.strip())
            self.refreshTickers = gr.Textbox(
                label="Symbols to refresh",
                lines=1,
                max_lines=3,
                placeholder="LLY, AAPL, MSFT",
                info=TICKERS_HELP,
            )
            self.refreshBtn = gr.Button(REFRESH_SYMBOLS_BUTTON, variant="primary")
            # Single progress UI: gr.Progress() on the handler (no separate Progress textbox —
            # that doubled the bar with Gradio's built-in progress).
            self.refreshStatus = gr.Markdown(
                value=(
                    f"_Enter symbols, then **{REFRESH_SYMBOLS_BUTTON}**._"
                )
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
                self.refreshDbStatus = gr.Markdown(
                    value="_Only needed if Charts stay empty after a data refresh._"
                )
                with gr.Accordion("Technical log", open=False):
                    self.refreshDbDetails = gr.Code(
                        language="json",
                        lines=4,
                        interactive=False,
                        value="",
                    )

        # ---- wire events ----
        # IMPORTANT: do NOT take Charts-tab widgets as *inputs* here.
        # Cross-tab inputs are not delivered by Gradio (handler gets fewer
        # values than declared → hard Error toast; refresh never runs).
        # Replot uses default confidence (80) via handler defaults.
        refresh_outputs = [
            self.refreshStatus,
            self.refreshDetails,
        ]
        if self.charts_ticker_dropdown is not None:
            refresh_outputs.append(self.charts_ticker_dropdown)
        if self.charts_tab is not None:
            refresh_outputs.extend(
                [
                    self.charts_tab.plotComponent,
                    self.charts_tab.rrTable,
                    self.charts_tab.companyInfo,
                ]
            )
        self.refreshBtn.click(
            fn=self.do_refresh_symbols,
            inputs=[self.refreshTickers],
            outputs=refresh_outputs,
            # One progress chrome only (handler uses gr.Progress for %).
            show_progress="full",
        )

        gather_outputs = [self.gatherStatus, self.gatherDetails]
        if self.charts_ticker_dropdown is not None:
            gather_outputs.append(self.charts_ticker_dropdown)
        if self.charts_tab is not None:
            gather_outputs.extend(
                [
                    self.charts_tab.plotComponent,
                    self.charts_tab.rrTable,
                    self.charts_tab.companyInfo,
                ]
            )
        self.gatherBtn.click(
            fn=self.do_gather,
            inputs=[self.gatherTickers],
            outputs=gather_outputs,
            show_progress="full",
        )
        self.resolveBtn.click(
            fn=self.preview_start,
            inputs=[self.forecastDate],
            outputs=[self.forecastStatus, self.forecastDetails],
        )
        forecast_outputs = [self.forecastStatus, self.forecastDetails]
        if self.charts_ticker_dropdown is not None:
            forecast_outputs.append(self.charts_ticker_dropdown)
        if self.charts_tab is not None:
            forecast_outputs.extend(
                [
                    self.charts_tab.plotComponent,
                    self.charts_tab.rrTable,
                    self.charts_tab.companyInfo,
                ]
            )
        self.forecastBtn.click(
            fn=self.do_forecast,
            inputs=[self.forecastTickers, self.forecastDate],
            outputs=forecast_outputs,
            show_progress="full",
        )

        db_refresh_outputs = [self.refreshDbStatus, self.refreshDbDetails]
        if self.charts_ticker_dropdown is not None:
            db_refresh_outputs.append(self.charts_ticker_dropdown)
        if self.charts_tab is not None:
            db_refresh_outputs.extend(
                [
                    self.charts_tab.plotComponent,
                    self.charts_tab.rrTable,
                    self.charts_tab.companyInfo,
                ]
            )
        self.refreshDbBtn.click(
            fn=self.do_refresh_search_db,
            inputs=[],
            outputs=db_refresh_outputs,
            show_progress="full",
        )

    def _charts_dropdown_update(self, prefer=None):
        """Refresh Charts symbol list; prefer selecting the first preferred ticker."""
        if self.charts_ticker_dropdown is None or not self.db_path:
            return None
        try:
            choices = sorted(list_tickers(self.db_path))
            prefer_list = [
                str(p).strip().upper()
                for p in (prefer or [])
                if p and str(p).strip()
            ]
            current = None
            for p in prefer_list:
                if p in choices:
                    current = p
                    break
            if current is None:
                current = choices[0] if choices else None
            return gr.update(choices=choices, value=current)
        except Exception as e:
            logger.warning(f"Could not refresh Charts dropdown: {e}")
            return gr.update()

    def _replot_charts(self, prefer=None, lowq=80):
        """Force Charts plot/RR/company to reload after data changes.

        Gradio does not re-fire dropdown.change when value stays the same, so
        switching back to Charts after a Run-tab refresh would show a stale chart.
        """
        if self.charts_tab is None:
            return ()
        prefer_list = [
            str(p).strip().upper()
            for p in (prefer or [])
            if p and str(p).strip()
        ]
        ticker = None
        try:
            choices = list_tickers(self.db_path) if self.db_path else []
            choice_set = set(choices)
            for p in prefer_list:
                if p in choice_set:
                    ticker = p
                    break
            if ticker is None and choices:
                ticker = choices[0]
            if ticker is None and prefer_list:
                ticker = prefer_list[0]
        except Exception:
            ticker = prefer_list[0] if prefer_list else None
        if not ticker:
            return gr.update(), gr.update(), gr.update()
        try:
            lowq_i = int(lowq) if lowq is not None else 80
        except (TypeError, ValueError):
            lowq_i = 80
        try:
            out = self.charts_tab.plot_forecast(ticker, lowq_i)
            if isinstance(out, dict):
                return (
                    out.get(self.charts_tab.plotComponent),
                    out.get(self.charts_tab.rrTable),
                    out.get(self.charts_tab.companyInfo)
                    or self.charts_tab._company_md(ticker),
                )
            company = (
                out[2]
                if len(out) > 2
                else self.charts_tab._company_md(ticker)
            )
            return out[0], out[1], company
        except Exception as e:
            logger.warning(f"Charts replot after Run action failed: {e}")
            return gr.update(), gr.update(), gr.update()

    def _pack_run_outputs(
        self,
        *core,
        prefer=None,
        lowq=80,
        replot: bool = False,
        dropdown_update=None,
    ):
        """Append optional dropdown + chart replot outputs for wired handlers."""
        outs = list(core)
        if self.charts_ticker_dropdown is not None:
            if dropdown_update is not None:
                outs.append(dropdown_update)
            elif replot or prefer is not None:
                outs.append(self._charts_dropdown_update(prefer=prefer))
            else:
                outs.append(gr.update())
        if self.charts_tab is not None:
            if replot:
                outs.extend(self._replot_charts(prefer=prefer, lowq=lowq))
            else:
                outs.extend((gr.update(), gr.update(), gr.update()))
        return tuple(outs)

    def preview_start(self, forecast_date):
        info = resolve_start_for_run(forecast_date or None)
        return _start_summary(info), _json_details(info)

    def _charts_lowq_default(self) -> int:
        """Confidence for post-run replot without reading Charts-tab inputs.

        Cross-tab Gradio inputs are unreliable; default matches Charts Radio.
        """
        return 80

    def do_refresh_symbols(self, tickers_text, progress=gr.Progress()):
        """Run refresh with a single Gradio progress bar + Charts replot on finish.

        Not a generator: intermediate yields + gr.Progress both draw bars (double UI).
        Live % comes only from ``progress``; status/details update once at the end.
        """
        lowq = self._charts_lowq_default()
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            status = (
                f"❌ **Could not start {REFRESH_SYMBOLS_BUTTON}.**  \n"
                f"{parsed.get('error') or 'Invalid symbols.'}"
            )
            details = _json_details(parsed)
            return self._pack_run_outputs(status, details, replot=False)

        logger.info(f"Dashboard refresh_symbols for {parsed['tickers']}")
        progress(0.0, desc="Starting…")

        def progress_cb(frac: float, desc: str = "") -> None:
            try:
                progress(frac, desc=desc or "Working…")
            except Exception:
                pass

        try:
            result = refresh_symbols(
                tickers_text,
                force_allow=True,
                progress_cb=progress_cb,
            )
        except Exception as e:
            logger.exception("refresh_symbols failed")
            result = {"ok": False, "error": str(e), "tickers": parsed["tickers"]}

        progress(1.0, desc="Done")
        status = _refresh_summary(result)
        details = _json_details(result)
        prefer = result.get("ready") or result.get("forecasted") or parsed["tickers"]
        # Replot so Charts tab shows new forecasts without re-selecting the symbol
        return self._pack_run_outputs(
            status,
            details,
            prefer=prefer,
            lowq=lowq,
            replot=True,
        )

    def do_gather(self, tickers_text):
        lowq = self._charts_lowq_default()
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            status = (
                f"❌ **Could not update market data.**  \n"
                f"{parsed.get('error') or 'Invalid symbols.'}"
            )
            details = _json_details(parsed)
            return self._pack_run_outputs(status, details, replot=False)
        logger.info(f"Dashboard gather for {parsed['tickers']}")
        result = gather_for_tickers(tickers_text, force_allow=True)
        status = _gather_summary(result)
        details = _json_details(result)
        prefer = result.get("ready") or parsed["tickers"]
        return self._pack_run_outputs(
            status, details, prefer=prefer, lowq=lowq, replot=True
        )

    def do_forecast(self, tickers_text, forecast_date, progress=gr.Progress()):
        lowq = self._charts_lowq_default()
        parsed = parse_ticker_list(tickers_text)
        if not parsed["ok"]:
            status = (
                f"❌ **Forecast failed.**  \n"
                f"{parsed.get('error') or 'Invalid symbols.'}"
            )
            details = _json_details(parsed)
            return self._pack_run_outputs(status, details, replot=False)
        # catch-up if blank; single origin if date set
        if forecast_date and str(forecast_date).strip():
            start_preview = resolve_start_for_run(forecast_date)
            if not start_preview.get("ok"):
                status = (
                    f"❌ **Forecast failed (start date).**  \n"
                    f"{start_preview.get('error') or ''}"
                )
                details = _json_details(start_preview)
                return self._pack_run_outputs(status, details, replot=False)
        logger.info(
            f"Dashboard forecast for {parsed['tickers']} "
            f"start={forecast_date or 'catch-up'}"
        )

        def progress_cb(frac: float, desc: str = "") -> None:
            try:
                progress(frac, desc=desc or "Forecasting…")
            except Exception:
                pass

        progress(0.0, desc="Starting forecast…")
        result = forecast_for_tickers(
            tickers_text,
            forecast_start_date=forecast_date or None,
            force_allow=True,
            progress_cb=progress_cb,
        )
        progress(1.0, desc="Done")
        status = _forecast_summary(result)
        details = _json_details(result)
        prefer = result.get("forecasted") or parsed["tickers"]
        return self._pack_run_outputs(
            status, details, prefer=prefer, lowq=lowq, replot=True
        )

    def do_refresh_search_db(self):
        """Rebuild DuckDB search cache from local parquet (full refresh)."""
        lowq = self._charts_lowq_default()
        if not self.db_path:
            status = "❌ **No database path configured.**"
            details = _json_details({"ok": False, "error": "db_path missing"})
            return self._pack_run_outputs(status, details, replot=False)

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
            n_sym = (st.get("counts") or {}).get("stock_tickers")
            if st.get("ok"):
                status = (
                    f"✅ **Charts database rebuilt.** "
                    f"{n_sym or '—'} symbols available in Charts / Scans."
                )
            else:
                status = (
                    "⚠️ **Rebuild finished, but some Charts data may be missing.** "
                    "Try **Refresh data & forecasts** for your symbols."
                )
            details = _json_details(result)
        except Exception as e:
            logger.exception("Search DB refresh failed")
            result = {"ok": False, "error": str(e)}
            status = (
                "❌ **Could not rebuild Charts database.** "
                "See technical log for details."
            )
            details = _json_details(result)

        return self._pack_run_outputs(
            status, details, prefer=None, lowq=lowq, replot=True
        )
