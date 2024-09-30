import pytest
from canswim.model import CanswimModel


def test_canswim_model_initialization():
    model = CanswimModel()
    assert isinstance(model, CanswimModel)
    assert model.model is None


def test_canswim_model_build():
    model = CanswimModel()
    model._CanswimModel__build_model()
    assert model.model is not None
