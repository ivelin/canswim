"""TimesFM-3 quantile mapping and parquet model tagging (no GPU / no weights)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canswim import constants
from canswim.db import FORECAST_MODEL_TIDE, FORECAST_MODEL_TIMESFM3, _normalize_forecast_model
from canswim.timesfm3_model import TIMESFM3_QUANTILES, map_timesfm_quantiles_to_canswim


def test_normalize_forecast_model():
    assert _normalize_forecast_model(None) == FORECAST_MODEL_TIDE
    assert _normalize_forecast_model("TiDE") == FORECAST_MODEL_TIDE
    assert _normalize_forecast_model("timesfm3") == FORECAST_MODEL_TIMESFM3
    assert _normalize_forecast_model("timesfm-3") == FORECAST_MODEL_TIMESFM3


def test_map_timesfm_quantiles_shape_and_order():
    rng = np.random.default_rng(0)
    horizon = 8
    # Increasing dummy quantile fan around 100
    base = np.linspace(90, 110, len(TIMESFM3_QUANTILES))
    q = np.tile(base, (horizon, 1)) + rng.normal(0, 0.01, size=(horizon, 9))
    mapped = map_timesfm_quantiles_to_canswim(q)
    cols = [f"close_quantile_{qname}" for qname in constants.quantiles]
    assert list(mapped) == cols
    stacked = np.stack([mapped[c] for c in cols], axis=1)
    assert stacked.shape == (horizon, len(constants.quantiles))
    diffs = np.diff(stacked, axis=1)
    assert (diffs >= -1e-9).all()
    # Median-ish 0.5 should land near TimesFM 0.5 (index 4)
    np.testing.assert_allclose(mapped["close_quantile_0.5"], q[:, 4], atol=0.5)


def test_map_rejects_bad_shape():
    with pytest.raises(ValueError):
        map_timesfm_quantiles_to_canswim(np.zeros((5, 7)))


def test_timesfm3_parquet_keeps_model_column(tmp_path, monkeypatch):
    from canswim.forecast import validate_forecast_dataframe
    from canswim.timesfm3_forecast import save_timesfm3_forecast
    from canswim.db import DataPaths

    idx = pd.bdate_range("2026-01-05", periods=42)
    df = pd.DataFrame(
        {f"close_quantile_{q}": np.linspace(100, 110, 42) for q in constants.quantiles},
        index=idx,
    )
    df.index.name = "date"
    df["symbol"] = "NVDA"
    df["model"] = FORECAST_MODEL_TIMESFM3
    df["forecast_start_year"] = 2026
    df["forecast_start_month"] = 1
    df["forecast_start_day"] = 5
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=42)
    assert "model" in cleaned.columns
    assert cleaned["model"].iloc[0] == FORECAST_MODEL_TIMESFM3

    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("timesfm3_forecast_subdir", "forecast-timesfm3/")
    paths = DataPaths.from_env()
    out = save_timesfm3_forecast(cleaned, paths=paths)
    files = list((tmp_path / "forecast-timesfm3").rglob("*.parquet"))
    assert files, out
    back = pd.read_parquet(files[0])
    # hive partition may put symbol in path; model should remain a column
    assert "model" in back.columns or FORECAST_MODEL_TIMESFM3 in str(files[0])
    if "model" in back.columns:
        assert (back["model"] == FORECAST_MODEL_TIMESFM3).all()
