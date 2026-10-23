"""The traditional side of the benchmark: a von Neumann VM.

Fetch-decode-execute over a small data memory with a program counter.
This is the "how we've been doing it for 80 years" baseline that
:mod:`tensor_broth.broth` is compared against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

OP_ADD, OP_MUL, OP_NONLIN, OP_MOVE = 0, 1, 2, 3
OP_NAMES = {OP_ADD: "ADD", OP_MUL: "MUL", OP_NONLIN: "NONLIN", OP_MOVE: "MOVE"}

MEM_SIZE = 256


@dataclass
class VMResult:
    elapsed: float
    steps: int
    memory: np.ndarray

    @property
    def mops_per_sec(self) -> float:
        return (self.steps / self.elapsed) / 1e6


def make_program(n_steps: int, seed: int = 0) -> np.ndarray:
    """Random instruction stream: (n, 3) int32 — [opcode, r1, r2]."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n_steps, 3), dtype=np.int32)


def run_von_neumann_vm(n_steps: int = 1_000_000, seed: int = 0) -> VMResult:
    """Execute ``n_steps`` instructions over a 256-cell data memory."""
    data_mem = np.zeros(MEM_SIZE, dtype=np.float32)
    instructions = make_program(n_steps, seed=seed)

    start = time.perf_counter()
    for pc in range(n_steps):
        op, r1, r2 = instructions[pc]
        i, j = r1 % MEM_SIZE, r2 % MEM_SIZE
        if op == OP_ADD:
            data_mem[i] += data_mem[j] + 0.1
        elif op == OP_MUL:
            data_mem[i] = (data_mem[i] * 0.99) + 0.01
        elif op == OP_NONLIN:
            data_mem[i] = np.sin(data_mem[j])
        else:
            data_mem[i] = data_mem[j]
    elapsed = time.perf_counter() - start
    return VMResult(elapsed=elapsed, steps=n_steps, memory=data_mem)
