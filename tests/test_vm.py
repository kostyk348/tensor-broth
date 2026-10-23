import numpy as np
import pytest

from tensor_broth.von_neumann import MEM_SIZE, make_program, run_von_neumann_vm


def test_program_shape():
    prog = make_program(100, seed=1)
    assert prog.shape == (100, 3)
    assert prog[:, 0].min() >= 0 and prog[:, 0].max() <= 3


def test_vm_runs_and_is_deterministic():
    r1 = run_von_neumann_vm(n_steps=2_000, seed=5)
    r2 = run_von_neumann_vm(n_steps=2_000, seed=5)
    assert r1.memory.shape == (MEM_SIZE,)
    assert np.array_equal(r1.memory, r2.memory)


def test_vm_throughput_positive():
    r = run_von_neumann_vm(n_steps=5_000, seed=0)
    assert r.elapsed > 0
    assert r.mops_per_sec > 0
