import numpy as np
import pytest

from tensor_broth.broth import CHANNELS, SpatialEngineNumPy, make_engine

torch = pytest.importorskip("torch", reason="torch extra not installed")
from tensor_broth.broth import SpatialEngine  # noqa: E402


@pytest.mark.parametrize("cls", [SpatialEngine, SpatialEngineNumPy])
def test_shape_preserved(cls):
    eng = cls(cube_size=12, seed=0)
    before = eng.space.shape
    eng.step(2)
    assert eng.space.shape == before


@pytest.mark.parametrize("cls", [SpatialEngine, SpatialEngineNumPy])
def test_state_bounded_by_tanh(cls):
    eng = cls(cube_size=10, seed=1)
    eng.step(3)
    arr = eng.space if isinstance(eng.space, np.ndarray) else eng.space.numpy()
    assert np.abs(arr).max() <= 1.0 + 1e-6


def test_torch_determinism():
    a = SpatialEngine(cube_size=8, seed=7)
    b = SpatialEngine(cube_size=8, seed=7)
    a.step(2)
    b.step(2)
    assert torch.allclose(a.space, b.space)


def test_activity_in_zero_one():
    eng = SpatialEngine(cube_size=8, seed=3)
    eng.step(1)
    assert 0.0 <= eng.activity() <= 1.0


def test_factory_numpy():
    eng = make_engine(backend="numpy", cube_size=6)
    assert eng.backend == "numpy"
    eng.step(1)
    assert eng.space.shape == (CHANNELS, 6, 6, 6)


def test_factory_auto_picks_something():
    eng = make_engine(backend="auto", cube_size=6)
    eng.step(1)
    assert eng.backend in ("torch", "numpy")
