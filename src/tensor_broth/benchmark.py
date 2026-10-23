"""Benchmark harness: von Neumann VM vs spatial dataflow engine."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List, Optional

from .broth import make_engine
from .von_neumann import run_von_neumann_vm


@dataclass
class BenchResult:
    backend: str
    device: str
    vm_steps: int
    vm_seconds: float
    vm_mops: float
    cube: int
    steps: int
    soup_seconds: float
    soup_gops: float
    soup_activity: float

    def row(self) -> str:
        return (f"| {self.backend:<6} | {self.device:<8} | {self.vm_mops:10.2f} | "
                f"{self.soup_gops:10.2f} | {self.soup_activity:8.4f} |")


@dataclass
class Report:
    results: List[BenchResult] = field(default_factory=list)
    vm_baseline: Optional[BenchResult] = None

    def markdown(self) -> str:
        lines = [
            "| backend | device | VM Mops/sec | broth GOPS (effective) | broth activity |",
            "|:--------|:-------|------------:|----------------------:|---------------:|",
        ]
        if self.vm_baseline:
            r = self.vm_baseline
            lines.append(f"| vm      | {r.device:<8} | {r.vm_mops:10.2f} | "
                         f"{'-':>21} | {'-':>14} |")
        for r in self.results:
            lines.append(r.row())
        return "\n".join(lines)


def compare(cube: int = 32, steps: int = 200, vm_steps: int = 500_000,
            backend: str = "auto", seed: int = 0) -> Report:
    """Run both paradigms and collect one report."""
    report = Report()

    vm = run_von_neumann_vm(n_steps=vm_steps, seed=seed)
    report.vm_baseline = BenchResult(
        backend="vm", device="cpu", vm_steps=vm_steps,
        vm_seconds=vm.elapsed, vm_mops=vm.mops_per_sec,
        cube=cube, steps=steps, soup_seconds=0.0, soup_gops=0.0, soup_activity=0.0,
    )

    engine = make_engine(backend=backend, cube_size=cube, seed=seed)
    elapsed = engine.run(steps)
    gops = (engine.voxel_ops(steps) / elapsed) / 1e9
    device = str(getattr(engine, "device", "cpu"))
    report.results.append(BenchResult(
        backend=engine.backend, device=device, vm_steps=vm_steps,
        vm_seconds=vm.elapsed, vm_mops=vm.mops_per_sec,
        cube=cube, steps=steps, soup_seconds=elapsed,
        soup_gops=gops, soup_activity=engine.activity(),
    ))
    return report


def environment_line() -> str:
    torch_info = "n/a"
    try:
        import torch
        torch_info = torch.__version__
    except ImportError:
        pass
    return (f"python {platform.python_version()} | numpy {platform.python_implementation()} "
            f"| torch {torch_info}")
