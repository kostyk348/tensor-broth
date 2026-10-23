"""tensor-broth CLI."""

from __future__ import annotations

import argparse
import json

from .benchmark import compare, environment_line
from .broth import make_engine
from .von_neumann import run_von_neumann_vm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensor-broth",
        description="Benchmark a von Neumann VM against a 3D spatial dataflow engine.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bench", help="run the paradigm comparison")
    p.add_argument("--cube", type=int, default=32, help="soup edge length (default 32)")
    p.add_argument("--steps", type=int, default=200, help="soup ticks (default 200)")
    p.add_argument("--vm-steps", type=int, default=500_000, help="VM instructions (default 500k)")
    p.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    p2 = sub.add_parser("sim", help="run only the soup and report its final activity")
    p2.add_argument("--cube", type=int, default=32)
    p2.add_argument("--steps", type=int, default=200)
    p2.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")

    ns = parser.parse_args(argv)

    if ns.cmd == "bench":
        report = compare(cube=ns.cube, steps=ns.steps, vm_steps=ns.vm_steps, backend=ns.backend)
        if ns.json:
            base = report.vm_baseline
            print(json.dumps({
                "environment": environment_line(),
                "vm": {"steps": base.vm_steps, "seconds": round(base.vm_seconds, 4),
                       "mops_per_sec": round(base.vm_mops, 2)},
                "broth": [{"backend": r.backend, "device": r.device, "cube": r.cube,
                           "steps": r.steps, "seconds": round(r.soup_seconds, 4),
                           "gops_effective": round(r.soup_gops, 2),
                           "activity": round(r.soup_activity, 4)} for r in report.results],
            }, indent=2))
        else:
            print(f"tensor-broth | {environment_line()}")
            print(report.markdown())
        return 0

    engine = make_engine(backend=ns.backend, cube_size=ns.cube)
    elapsed = engine.run(ns.steps)
    print(f"backend={engine.backend} cube={ns.cube}^3 steps={ns.steps} "
          f"elapsed={elapsed:.3f}s activity={engine.activity():.4f} "
          f"gops={(engine.voxel_ops(ns.steps) / elapsed) / 1e9:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
