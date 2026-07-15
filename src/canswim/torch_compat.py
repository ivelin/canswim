"""PyTorch compatibility helpers for Darts checkpoints (CPU and all GPUs)."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def darts_torch_load_compat():
    """Allow Darts full-model pickles under torch>=2.6 (any device).

    Darts ``TiDEModel.load`` calls ``torch.load`` without ``weights_only=``.
    Torch 2.6+ defaults ``weights_only=True``, which rejects pickled model
    classes. Local/HF checkpoints are trusted operator artifacts. This is
    independent of GPU arch (works on CPU, Ampere, Ada, etc.).
    """
    orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return orig(*args, **kwargs)
        except TypeError as e:
            # Older torch without weights_only kwarg
            if "weights_only" not in str(e):
                raise
            kwargs.pop("weights_only", None)
            return orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig
