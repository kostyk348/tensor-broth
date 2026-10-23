"""The 3D spatial dataflow engine — "tensor broth".

The bet: a program is not a sequence of instructions but a *space* where
data lives. Each voxel holds a state; a local 3x3x3 interaction kernel lets
neighbouring voxels transform each other. There is no fetch-decode loop,
no program counter, no branches — just physics on a lattice.

The primary backend is PyTorch (CPU or CUDA). A pure-NumPy fallback keeps
the idea honest on machines without torch.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional extra
    torch = None

CHANNELS = 3  # value, affinity mask, shift gradient


def _torch_available() -> bool:
    return torch is not None


class SpatialEngine:
    """Torch-backed 3D spatial engine (GPU or CPU)."""

    backend = "torch"

    def __init__(self, cube_size: int = 64, seed: int = 0,
                 device: Optional[str] = None, dt: float = 0.05) -> None:
        if not _torch_available():
            raise ImportError("PyTorch is not installed: pip install tensor-broth[torch]")
        self.cube_size = cube_size
        self.dt = dt
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.space = torch.rand(
            (1, CHANNELS, cube_size, cube_size, cube_size),
            generator=gen, dtype=torch.float32,
        ).to(self.device)
        self.kernel = (torch.randn(
            (CHANNELS, CHANNELS, 3, 3, 3), generator=gen, dtype=torch.float32
        ) * 0.1).to(self.device)

    def step(self, n: int = 1) -> None:
        """Advance the soup ``n`` ticks: local 3D convolution + nonlinearity."""
        for _ in range(n):
            interaction = torch.nn.functional.conv3d(self.space, self.kernel, padding=1)
            self.space = torch.tanh(self.space + self.dt * interaction)

    def run(self, n_steps: int) -> float:
        """Run ``n_steps`` ticks, return elapsed seconds (synchronized)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.step(n_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    def voxel_ops(self, n_steps: int) -> int:
        """Theoretical MAC-ish op count: 27-tap conv over the whole volume."""
        return n_steps * (self.cube_size ** 3) * 27 * CHANNELS

    def activity(self) -> float:
        """Mean |state| — how 'hot' the broth is right now."""
        return float(self.space.abs().mean().item())


class SpatialEngineNumPy:
    """Dependency-free fallback: same semantics, sliding-window convolution.

    Honest note: pure NumPy convolution is *slower* per voxel than torch's
    vectorized kernels on CPU. It exists so the paradigm is testable
    everywhere, not to win benchmarks.
    """

    backend = "numpy"

    def __init__(self, cube_size: int = 64, seed: int = 0, dt: float = 0.05) -> None:
        self.cube_size = cube_size
        self.dt = dt
        rng = np.random.default_rng(seed)
        self.space = rng.random((CHANNELS, cube_size, cube_size, cube_size)).astype(np.float32)
        self.kernel = (rng.standard_normal((CHANNELS, CHANNELS, 3, 3, 3)) * 0.1).astype(np.float32)

    def step(self, n: int = 1) -> None:
        from numpy.lib.stride_tricks import sliding_window_view

        for _ in range(n):
            padded = np.pad(self.space, ((0, 0), (1, 1), (1, 1), (1, 1)))
            # windows: (C_in, D, H, W, 3, 3, 3)
            windows = sliding_window_view(padded, (3, 3, 3), axis=(1, 2, 3))
            # contract C_in and the 3 tap axes -> (D, H, W, C_out)
            interaction = np.tensordot(windows, self.kernel, axes=([0, 4, 5, 6], [1, 2, 3, 4]))
            self.space = np.tanh(
                self.space + self.dt * interaction.transpose(3, 0, 1, 2)
            ).astype(np.float32)

    def run(self, n_steps: int) -> float:
        start = time.perf_counter()
        self.step(n_steps)
        return time.perf_counter() - start

    def voxel_ops(self, n_steps: int) -> int:
        return n_steps * (self.cube_size ** 3) * 27 * CHANNELS

    def activity(self) -> float:
        return float(np.abs(self.space).mean())


def make_engine(backend: str = "auto", cube_size: int = 64, seed: int = 0):
    """Factory: 'auto' -> torch if importable, else numpy."""
    if backend == "auto":
        backend = "torch" if _torch_available() else "numpy"
    if backend == "torch":
        return SpatialEngine(cube_size=cube_size, seed=seed)
    if backend == "numpy":
        return SpatialEngineNumPy(cube_size=cube_size, seed=seed)
    raise ValueError(f"unknown backend: {backend!r}")


__all__ = ["SpatialEngine", "SpatialEngineNumPy", "make_engine", "CHANNELS"]
