#!/usr/bin/env python3
"""von Neumann VM vs spatial dataflow engine, side by side."""

from tensor_broth.benchmark import compare, environment_line

print(f"tensor-broth | {environment_line()}")
report = compare(cube=32, steps=100, vm_steps=500_000)
print(report.markdown())
