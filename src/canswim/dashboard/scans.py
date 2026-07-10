from loguru import logger
from canswim.model import CanswimModel
import gradio as gr
from canswim.db import (
    list_forecast_start_date_choices,
    scan_forecasts as db_scan_forecasts,
)


def _normalize_start_value(current) -> str | None:
    """Map Gradio dropdown / state payload to ISO YYYY-MM-DD."""
    if current is None:
        return None
    # Index payloads (type=index or accidental int) are not usable alone
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return None
    s = str(current).strip()
    if not s:
        return None
    # Prefer leading ISO date (handles full label if Gradio sends it)
    return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else s


class ScanTab:

    def __init__(self, canswim_model: CanswimModel = None, db_path=None):
        assert canswim_model is not None
        assert db_path is not None
        self.canswim_model = canswim_model
        self.db_path = db_path
        # Horizon for open-vs-completed labels (TiDE output_chunk_length when loaded)
        try:
            self._pred_horizon = int(self.canswim_model.pred_horizon)
        except Exception:
            self._pred_horizon = 42

        # Preload so the Dropdown is never empty (Gradio is flaky with choices=[]
        # then later gr.update — selected value can arrive as None).
        init_pairs = list_forecast_start_date_choices(
            self.db_path, pred_horizon_bdays=self._pred_horizon
        )
        init_labels = [lab for lab, _v in init_pairs]
        init_iso = init_pairs[0][1] if init_pairs else None
        init_label = init_labels[0] if init_labels else None

        # Stable selection store — backup if Dropdown payload is empty
        self.selectedStart = gr.State(value=init_iso)

        with gr.Row():
            # Plain string choices (label == value). Avoid (label, value) tuples:
            # frontend can send None / label / value inconsistently.
            self.forecastStart = gr.Dropdown(
                choices=init_labels,
                value=init_label,
                label="Forecast start date (as-of origin)",
                info=(
                    "Newest first. Latest may be an **open-horizon live forecast**; "
                    "older dates with a full elapsed window are **completed backtests**. "
                    "Changing the date or filters updates results immediately."
                ),
                allow_custom_value=False,
                filterable=True,
            )
            self.refreshStartsBtn = gr.Button(
                value="↻ dates",
                scale=0,
                min_width=80,
                variant="secondary",
            )
        with gr.Row():
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Low-price confidence used in reward/risk",
            )
            self.reward = gr.Radio(
                choices=[5, 10, 15, 20, 25],
                value=5,
                label="Minimum probabilistic price gain (%)",
                info="Median forecast high vs prior close",
            )
            self.rr = gr.Radio(
                choices=[1, 1.5, 3, 5, 10],
                value=1,
                label="Minimum reward / risk ratio",
                info="Upside vs downside (low-confidence quantile)",
            )

        with gr.Row():
            self.scanStatus = gr.Markdown(
                value="_Pick filters — results update automatically._"
            )

        with gr.Row():
            self.scanResult = gr.Dataframe()

        scan_inputs = [
            self.lowq,
            self.reward,
            self.rr,
            self.forecastStart,
            self.selectedStart,
        ]
        scan_outputs = [self.selectedStart, self.scanStatus, self.scanResult]

        # Auto-scan on any control change (same pattern as Charts tab)
        self.forecastStart.change(
            fn=self.apply_and_scan,
            inputs=scan_inputs,
            outputs=scan_outputs,
        )
        self.lowq.change(
            fn=self.apply_and_scan,
            inputs=scan_inputs,
            outputs=scan_outputs,
        )
        self.reward.change(
            fn=self.apply_and_scan,
            inputs=scan_inputs,
            outputs=scan_outputs,
        )
        self.rr.change(
            fn=self.apply_and_scan,
            inputs=scan_inputs,
            outputs=scan_outputs,
        )
        # Optional: reload date list only (new forecast origins in DB)
        self.refreshStartsBtn.click(
            fn=self.refresh_start_dates,
            inputs=[self.forecastStart, self.selectedStart],
            outputs=[self.forecastStart, self.selectedStart],
        ).then(
            fn=self.apply_and_scan,
            inputs=scan_inputs,
            outputs=scan_outputs,
        )

    def apply_and_scan(
        self, lowq, reward, rr, forecast_start_date=None, stored_start=None
    ):
        """Update selection state + run scan. Used by control change handlers."""
        asof = (
            _normalize_start_value(forecast_start_date)
            or _normalize_start_value(stored_start)
        )
        status, table = self._run_scan(lowq=lowq, reward=reward, rr=rr, asof=asof)
        return asof, status, table

    def refresh_start_dates(self, current=None, stored=None):
        """Reload start dates from DB; keep selection if still valid, else newest.

        Prefer the Dropdown value (``current``) over State (``stored``).

        Returns ``(dropdown_update, selected_iso)`` for Dropdown + State.
        """
        pairs = list_forecast_start_date_choices(
            self.db_path, pred_horizon_bdays=self._pred_horizon
        )
        labels = [lab for lab, _v in pairs]
        by_iso = {v: lab for lab, v in pairs}
        isos = [v for _lab, v in pairs]

        cur = _normalize_start_value(current) or _normalize_start_value(stored)
        if cur and cur in by_iso:
            selected_iso = cur
            selected_label = by_iso[cur]
        else:
            selected_iso = isos[0] if isos else None
            selected_label = labels[0] if labels else None

        logger.info(
            f"Scan start-date picker: {len(pairs)} origins; "
            f"selected={selected_iso} (current={current!r}, stored={stored!r})"
        )
        return gr.update(choices=labels, value=selected_label), selected_iso

    def initial_scan(self):
        """Page-load helper: (dropdown_update, iso, status, table)."""
        dd, iso = self.refresh_start_dates(current=None, stored=None)
        # Defaults match Radio initial values
        status, table = self._run_scan(lowq=80, reward=5, rr=1, asof=iso)
        return dd, iso, status, table

    def _run_scan(self, lowq, reward, rr, asof):
        df = db_scan_forecasts(
            self.db_path,
            lowq=lowq,
            reward=reward,
            rr=rr,
            forecast_start_date=asof,
        )
        n = 0 if df is None else len(df)
        logger.info(
            f"Scan as-of={asof} lowq={lowq} reward={reward} rr={rr} hits={n}"
        )
        if df is None or df.empty:
            status = (
                f"**No matches** for as-of `{asof or 'latest'}` · "
                f"reward ≥ {reward}% · R/R ≥ {rr} · low conf {lowq}%. "
                f"Try lower thresholds or another start date."
            )
            return status, df
        status = (
            f"**{n} symbol{'s' if n != 1 else ''}** · as-of `{asof}` · "
            f"reward ≥ {reward}% · R/R ≥ {rr} · low conf {lowq}%"
        )
        styled = df.style.format(
            precision=2,
            thousands=",",
            decimal=".",
        )
        return status, styled
