"""Zero-shot TimesFM-3 adapter (research / non-commercial weights).

Does **not** replace the live TiDE model. Maps TimesFM-3's 9 quantiles
(0.1 … 0.9) onto CANSWIM close_quantile_{0.01,0.05,0.2,0.5,0.8,0.95,0.99}.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger
from pandas.tseries.offsets import BDay

from canswim import constants
from canswim.db import FORECAST_MODEL_TIMESFM3, DataPaths

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESFM3_QUANTILES = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64
)
# Match TiDE output_chunk_length used in Charts/Scans labeling.
DEFAULT_PRED_HORIZON = 42
DEFAULT_CONTEXT = 512
MAX_PAST_ONLY_CHANNELS = 8
MAX_PAST_FUTURE_CHANNELS = 6

_KEYMETRIC_COLS = (
    "peRatio",
    "pbRatio",
    "revenuePerShare",
    "freeCashFlowPerShare",
    "earningsYield",
    "currentRatio",
)

_forecaster = None


def hf_cache_dir() -> str:
    return os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME") or str(
        Path.home() / ".cache" / "huggingface"
    )


def map_timesfm_quantiles_to_canswim(q_hz_k: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate TimesFM-3 quantiles onto CANSWIM quantile columns.

    ``q_hz_k`` shape (horizon, 9) at TIMESFM3_QUANTILES. Values outside
    [0.1, 0.9] are linearly extrapolated from the nearest two points.
    """
    arr = np.asarray(q_hz_k, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(TIMESFM3_QUANTILES):
        raise ValueError(
            f"expected (horizon, {len(TIMESFM3_QUANTILES)}) quantiles, got {arr.shape}"
        )
    src_q = TIMESFM3_QUANTILES
    out: dict[str, np.ndarray] = {}
    for q in constants.quantiles:
        mapped = np.empty(arr.shape[0], dtype=np.float64)
        for i in range(arr.shape[0]):
            mapped[i] = _interp_extrapolate(float(q), src_q, arr[i])
        # Enforce weak monotonicity across mapped quantiles later (row-wise).
        out[f"close_quantile_{q}"] = mapped
    # Row-wise sort to keep CANSWIM quantile order after extrapolation noise.
    stacked = np.stack([out[f"close_quantile_{q}"] for q in constants.quantiles], axis=1)
    stacked.sort(axis=1)
    for j, q in enumerate(constants.quantiles):
        out[f"close_quantile_{q}"] = stacked[:, j]
    return out


def _interp_extrapolate(q: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if q <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + slope * (q - xs[0]))
    if q >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (q - xs[-1]))
    return float(np.interp(q, xs, ys))


def get_forecaster(*, device: Optional[str] = None, local_files_only: bool = True):
    """Load TimesFM-3 once per process (research NC weights)."""
    global _forecaster
    if _forecaster is not None:
        return _forecaster
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = hf_cache_dir()
    logger.info(
        f"Loading TimesFM-3 checkpoint {TIMESFM3_CHECKPOINT} device={device} "
        f"cache={cache} (research / non-commercial / non-production)"
    )
    kwargs = dict(
        checkpoint_path=TIMESFM3_CHECKPOINT,
        per_core_batch_size=4,
        device=device,
        cache_dir=cache,
        local_files_only=local_files_only,
    )
    try:
        _forecaster = TimesFM3Evaluator(ModelConfig(**kwargs))
    except Exception as e:
        if local_files_only:
            logger.warning(f"TimesFM-3 local load failed ({e}); retrying with hub")
            kwargs["local_files_only"] = False
            _forecaster = TimesFM3Evaluator(ModelConfig(**kwargs))
        else:
            raise
    return _forecaster


def load_close_history(
    symbol: str,
    *,
    asof: Optional[pd.Timestamp] = None,
    min_samples: int = DEFAULT_CONTEXT,
    paths: Optional[DataPaths] = None,
) -> pd.Series:
    """Daily Close strictly before ``asof`` (or full history if asof is None/live)."""
    paths = paths or DataPaths.from_env()
    sym = str(symbol).strip().upper()
    df = pd.read_parquet(
        paths.stocks_price_path,
        filters=[("Symbol", "in", [sym])],
        dtype_backend="numpy_nullable",
    )
    if df.empty:
        raise ValueError(f"no price history for {sym}")
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("Symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    if "Close" not in df.columns:
        raise ValueError(f"{sym}: Close column missing")
    close = df["Close"].astype(float)
    close = close[np.isfinite(close) & (close > 0)]
    if asof is not None:
        cutoff = pd.Timestamp(asof).tz_localize(None).normalize()
        close = close[close.index < cutoff]
    if len(close) < min_samples:
        raise ValueError(
            f"{sym}: only {len(close)} close bars before asof "
            f"(need >= {min_samples})"
        )
    return close


def load_volume_history(symbol: str, index: pd.DatetimeIndex) -> Optional[np.ndarray]:
    paths = DataPaths.from_env()
    sym = str(symbol).strip().upper()
    try:
        df = pd.read_parquet(
            paths.stocks_price_path,
            filters=[("Symbol", "in", [sym])],
            columns=["Volume"],
        )
    except Exception as e:
        logger.debug(f"volume load {sym}: {e}")
        return None
    if df.empty or "Volume" not in df.columns:
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("Symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    vol = df["Volume"].reindex(index).astype(float)
    vol = vol.ffill().bfill().fillna(0.0)
    return vol.to_numpy(dtype=np.float32).reshape(1, -1)


def load_keymetrics_aligned(
    symbol: str, index: pd.DatetimeIndex
) -> Optional[np.ndarray]:
    paths = DataPaths.from_env()
    data_3rd = os.getenv("data-3rd-party", "data-3rd-party")
    km_path = Path(paths.data_dir) / data_3rd / "keymetrics_history.parquet"
    if not km_path.is_file():
        return None
    sym = str(symbol).strip().upper()
    try:
        df = pd.read_parquet(
            km_path,
            filters=[("Symbol", "in", [sym])],
            columns=[c for c in _KEYMETRIC_COLS],
        )
    except Exception as e:
        logger.debug(f"keymetrics {sym}: {e}")
        return None
    if df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("Symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    cols = [c for c in _KEYMETRIC_COLS if c in df.columns][:MAX_PAST_ONLY_CHANNELS]
    if not cols:
        return None
    aligned = df[cols].reindex(index).ffill().bfill().fillna(0.0)
    arr = aligned.to_numpy(dtype=np.float32).T  # (channels, T)
    return arr


def calendar_past_future(
    context_index: pd.DatetimeIndex, horizon_index: pd.DatetimeIndex
) -> np.ndarray:
    """Known future calendar features: dow, month, day-of-month (scaled)."""
    full = context_index.append(horizon_index)
    dow = full.dayofweek.to_numpy(dtype=np.float32) / 6.0
    month = (full.month.to_numpy(dtype=np.float32) - 1.0) / 11.0
    dom = (full.day.to_numpy(dtype=np.float32) - 1.0) / 30.0
    return np.stack([dow, month, dom], axis=0)  # (3, T+H)


def horizon_sessions(start: pd.Timestamp, n: int = DEFAULT_PRED_HORIZON) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    start = pd.Timestamp(start).tz_localize(None).normalize()
    days = nyse.valid_days(
        start_date=start, end_date=start + BDay(n * 3 + 10), tz=None
    )
    if days is None or len(days) == 0:
        raise ValueError(f"no NYSE sessions from {start.date()}")
    idx = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(None).normalize()
    return idx[: int(n)]


def _finite_2d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def forecast_symbol(
    symbol: str,
    asof: pd.Timestamp,
    *,
    horizon: int = DEFAULT_PRED_HORIZON,
    context: int = DEFAULT_CONTEXT,
    include_covariates: bool = True,
    forecaster=None,
) -> pd.DataFrame:
    """Zero-shot TimesFM-3 forecast for one symbol as-of ``asof``.

    Returns a dataframe indexed by forecast date with CANSWIM quantile columns,
    symbol, partition keys, and model=timesfm3.
    """
    asof = pd.Timestamp(asof).tz_localize(None).normalize()
    close = load_close_history(symbol, asof=asof, min_samples=min(60, context // 4))
    ctx = close.iloc[-int(context) :] if len(close) > context else close
    ctx_vals = ctx.to_numpy(dtype=np.float32)
    hz_idx = horizon_sessions(asof, n=horizon)
    past_only = None
    past_future = None
    if include_covariates:
        parts = []
        vol = load_volume_history(symbol, ctx.index)
        if vol is not None:
            parts.append(vol)
        km = load_keymetrics_aligned(symbol, ctx.index)
        if km is not None:
            parts.append(km)
        if parts:
            past_only = np.concatenate(parts, axis=0)
            if past_only.shape[0] > MAX_PAST_ONLY_CHANNELS:
                past_only = past_only[:MAX_PAST_ONLY_CHANNELS]
        past_future = calendar_past_future(ctx.index, hz_idx)
        if past_future.shape[0] > MAX_PAST_FUTURE_CHANNELS:
            past_future = past_future[:MAX_PAST_FUTURE_CHANNELS]
    model = forecaster or get_forecaster()
    outputs = list(
        model.predict_batch(
            contexts=[ctx_vals],
            horizon=int(horizon),
            past_only_covariates=[_finite_2d(past_only)] if past_only is not None else None,
            past_future_covariates=[_finite_2d(past_future)]
            if past_future is not None
            else None,
            ts_ids=[str(symbol).upper()],
            return_quantiles=True,
            use_symmetric_averaging=True,
            make_positive=True,
            sort_quantiles=True,
        )
    )
    if not outputs:
        raise RuntimeError(f"{symbol}: TimesFM-3 returned no outputs")
    q = outputs[0].quantiles
    if q is None:
        raise RuntimeError(f"{symbol}: TimesFM-3 returned no quantiles")
    q = np.asarray(q)
    # Univariate → (horizon, 9); multivariate target would be (1, H, 9)
    if q.ndim == 3:
        q = q[0]
    if q.ndim != 2:
        raise RuntimeError(f"{symbol}: unexpected quantile shape {q.shape}")
    n = min(len(hz_idx), q.shape[0])
    mapped = map_timesfm_quantiles_to_canswim(q[:n])
    out = pd.DataFrame(mapped, index=hz_idx[:n])
    out.index.name = "date"
    # Floor at a tiny positive (prices)
    for c in out.columns:
        out[c] = np.maximum(out[c].astype(float), 0.01)
    out["symbol"] = str(symbol).strip().upper()
    out["model"] = FORECAST_MODEL_TIMESFM3
    out["forecast_start_year"] = int(asof.year)
    out["forecast_start_month"] = int(asof.month)
    out["forecast_start_day"] = int(asof.day)
    return out
