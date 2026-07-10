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
        # then later gr.update — selected value can arrive as None on Refresh).
        init_pairs = list_forecast_start_date_choices(
            self.db_path, pred_horizon_bdays=self._pred_horizon
        )
        init_labels = [lab for lab, _v in init_pairs]
        init_iso = init_pairs[0][1] if init_pairs else None
        init_label = init_labels[0] if init_labels else None

        # Stable selection store — do not rely only on Dropdown I/O (same component
        # as input+output often loses the user pick in Gradio 4.x).
        self.selectedStart = gr.State(value=init_iso)

        with gr.Row():
            # Plain string choices (label == value). Avoid (label, value) tuples:
            # frontend can send None / label / value inconsistently on refresh.
            self.forecastStart = gr.Dropdown(
                choices=init_labels,
                value=init_label,
                label="Forecast start date (as-of origin)",
                info=(
                    "Queried from the search DB (newest first). Default = most recent. "
                    "Includes the latest run, which may still be an **open-horizon live "
                    "forecast** (not a finished backtest) if today is inside the "
                    "prediction window. Older dates with a fully elapsed horizon are "
                    "**completed backtests**. Scan scores only that origin. "
                    "Refresh reloads the list but keeps your current selection when possible."
                ),
                allow_custom_value=False,
                filterable=True,
            )
            self.refreshStartsBtn = gr.Button(
                value="Refresh dates",
                scale=0,
            )
        with gr.Row():
            self.lowq = gr.Radio(
                choices=[80, 95, 99],
                value=80,
                label="Confidence level for lowest close price",
                info="Choose low price confidence percentage",
            )
            self.reward = gr.Radio(
                choices=[5, 10, 15, 20, 25],
                value=5,
                label="Minimum probabilistic price gain (%)",
                info="Median forecast high vs prior close. Lower if the table is empty.",
            )
            self.rr = gr.Radio(
                choices=[1, 1.5, 3, 5, 10],
                value=1,
                label="Minimum reward / risk ratio",
                info="Upside to downside (low-confidence quantile). Lower if empty.",
            )

        with gr.Row():
            self.scanBtn = gr.Button(value="Scan", variant="primary")

        with gr.Row():
            self.scanResult = gr.Dataframe()

        # Keep State in sync whenever the user picks a date
        self.forecastStart.change(
            fn=self._on_start_change,
            inputs=[self.forecastStart],
            outputs=[self.selectedStart],
        )
        self.forecastStart.input(
            fn=self._on_start_change,
            inputs=[self.forecastStart],
            outputs=[self.selectedStart],
        )
        # Refresh: prefer Dropdown (what user sees), fall back to State
        self.refreshStartsBtn.click(
            fn=self.refresh_start_dates,
            inputs=[self.forecastStart, self.selectedStart],
            outputs=[self.forecastStart, self.selectedStart],
        )
        # Scan: prefer Dropdown payload, fall back to State
        self.scanBtn.click(
            fn=self.scan_forecasts,
            inputs=[
                self.lowq,
                self.reward,
                self.rr,
                self.forecastStart,
                self.selectedStart,
            ],
            outputs=[self.scanResult],
        )

    def _on_start_change(self, current):
        return _normalize_start_value(current)

    def refresh_start_dates(self, current=None, stored=None):
        """Reload start dates from DB; keep selection if still valid, else newest.

        Prefer the Dropdown value (``current``) over State (``stored``). State alone
        can lag if change events were skipped, which previously forced a reset to
        the default newest origin.

        Returns ``(dropdown_update, selected_iso)`` for Dropdown + State.
        """
        pairs = list_forecast_start_date_choices(
            self.db_path, pred_horizon_bdays=self._pred_horizon
        )
        labels = [lab for lab, _v in pairs]
        by_iso = {v: lab for lab, v in pairs}
        isos = [v for _lab, v in pairs]

        # Dropdown first (authoritative UI selection), then State backup
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

    def scan_forecasts(
        self, lowq, reward, rr, forecast_start_date=None, stored_start=None
    ):
        asof = (
            _normalize_start_value(forecast_start_date)
            or _normalize_start_value(stored_start)
        )
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
            gr.Warning(
                f"No symbols met reward ≥ {reward}% and R/R ≥ {rr} for as-of "
                f"{asof or 'latest'}. "
                f"Try lower thresholds (e.g. 5% / 1.0) or another start date. "
                f"Latest open-horizon forecasts are often milder than past backtests."
            )
            return df
        return df.style.format(
            precision=2,
            thousands=",",
            decimal=".",
        )
