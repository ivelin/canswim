"""Issue #31: held-out test MAE/MAPE helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from canswim.model import CanswimModel


def test_median_mae_mape_perfect():
    idx = pd.bdate_range("2024-01-02", periods=5)
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
    mae, mape = CanswimModel.median_mae_mape(s, s)
    assert mae == pytest.approx(0.0)
    assert mape == pytest.approx(0.0)


def test_median_mae_mape_known_error():
    idx = pd.bdate_range("2024-01-02", periods=4)
    pred = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    actual = pd.Series([11.0, 22.0, 33.0, 44.0], index=idx)
    mae, mape = CanswimModel.median_mae_mape(pred, actual)
    # abs errors 1,2,3,4 → mean 2.5
    assert mae == pytest.approx(2.5)
    # each relative error is 1/11 → ~9.0909%
    assert mape == pytest.approx(100.0 / 11.0)


def test_median_mae_mape_no_overlap():
    a = pd.Series([1.0], index=pd.to_datetime(["2024-01-02"]))
    b = pd.Series([1.0], index=pd.to_datetime(["2024-01-03"]))
    with pytest.raises(ValueError, match="overlapping"):
        CanswimModel.median_mae_mape(a, b)


def test_evaluate_skips_without_test_split():
    """No GPU/train: empty splits → empty metrics, no raise."""
    from unittest.mock import MagicMock

    m = CanswimModel.__new__(CanswimModel)
    m.test_series = {}
    m.train_series = {}
    out = m.evaluate_and_log_test_metrics()
    assert out == {}
