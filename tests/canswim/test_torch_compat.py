"""Tests for torch/Darts load compatibility helpers."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from canswim.torch_compat import darts_torch_load_compat


def test_darts_torch_load_compat_sets_weights_only_false():
    """Darts full pickles need weights_only=False under torch>=2.6."""
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    with patch.object(torch, "load", fake_load):
        with darts_torch_load_compat():
            result = torch.load("checkpoint.pt", map_location="cpu")
        # restored after context
        assert torch.load is fake_load

    assert result == "ok"
    assert captured.get("weights_only") is False
    assert captured.get("map_location") == "cpu"


def test_darts_torch_load_compat_restores_original_on_error():
    orig = torch.load
    try:
        with darts_torch_load_compat():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert torch.load is orig


def test_darts_torch_load_compat_preserves_explicit_weights_only():
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return None

    with patch.object(torch, "load", fake_load):
        with darts_torch_load_compat():
            torch.load("x.pt", weights_only=True)

    assert captured.get("weights_only") is True


def test_load_model_uses_darts_torch_load_compat():
    """CanswimModel.load_model must load under the weights_only compat context."""
    from canswim.model import CanswimModel

    entered = {"n": 0}

    @contextmanager
    def tracking_compat():
        entered["n"] += 1
        yield

    fake_model = MagicMock(name="TiDEModel")
    with (
        patch("canswim.model.darts_torch_load_compat", tracking_compat),
        patch("canswim.model.TiDEModel.load", return_value=fake_model) as load,
        patch("canswim.model.torch.cuda.is_available", return_value=False),
        patch.object(CanswimModel, "__init__", lambda self, *a, **k: None),
    ):
        m = CanswimModel.__new__(CanswimModel)
        m.model_name = "canswim_model.pt"
        m.torch_model = None
        ok = m.load_model()

    assert ok is True
    assert entered["n"] == 1
    load.assert_called_once_with(path="canswim_model.pt", map_location="cpu")
    assert m.torch_model is fake_model


def test_hfhub_download_model_uses_darts_torch_load_compat():
    """HF model load path also wraps TiDEModel.load for torch>=2.6."""
    from canswim.hfhub import HFHub

    entered = {"n": 0}

    @contextmanager
    def tracking_compat():
        entered["n"] += 1
        yield

    fake_cls = MagicMock()
    fake_cls.load.return_value = MagicMock(
        model_name="canswim_model.pt",
        model_params={},
        model_created=True,
    )

    hub = HFHub.__new__(HFHub)
    hub.hfhub_sync = True
    hub.HF_TOKEN = "x"
    hub.repo_id = "ivelin/canswim"

    with (
        patch("canswim.hfhub.darts_torch_load_compat", tracking_compat),
        patch("canswim.hfhub.snapshot_download"),
        patch("canswim.hfhub.tempfile.TemporaryDirectory") as td,
        patch("canswim.hfhub.os.listdir", return_value=["canswim_model.pt"]),
        patch("canswim.hfhub.torch.cuda.is_available", return_value=False),
    ):
        td.return_value.__enter__.return_value = "/tmp/fake-hf"
        td.return_value.__exit__.return_value = None
        out = hub.download_model(
            model_name="canswim_model.pt",
            model_class=fake_cls,
        )

    assert entered["n"] == 1
    fake_cls.load.assert_called_once()
    assert out is fake_cls.load.return_value
