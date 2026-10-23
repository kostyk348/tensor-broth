<div align="center">

```
 _________________________
< T E N S O R   B R O T H >
 -------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

# tensor-broth

**Computing without a program counter: a 3D spatial dataflow engine vs
the 80-year-old fetch-decode VM.**

[![CI](https://github.com/example/tensor-broth/actions/workflows/ci.yml/badge.svg)](https://github.com/example/tensor-broth/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green)]()

*README на русском: [README.ru.md](README.ru.md)*

</div>

---

## TL;DR

Two engines. One benchmark. A paradigm fight.

| engine                 | what it is                                                    |
|------------------------|---------------------------------------------------------------|
| `von_neumann`          | classic fetch-decode-execute loop over a 256-cell data memory |
| `broth`                | a 3D lattice of voxels that transform each other locally — no PC, no branches, no instructions |

```console
$ tensor-broth bench --cube 24 --steps 60 --vm-steps 200000
tensor-broth | python 3.12.15 | torch 2.14.1+cpu
| backend | device | VM Mops/sec | broth GOPS (effective) | broth activity |
|:--------|:-------|------------:|----------------------:|---------------:|
| vm      | cpu    |        0.88 |                     - |              - |
| torch   | cpu    |        0.88 |       1.15            |         0.2682 |
```

The VM pushes ~1 million *instructions* per second through a Python
fetch-decode loop. The broth pushes ~10⁹ *voxel×tap interactions* per
second through one fused `conv3d` — because the data structure **is** the
program, and the hardware it wants is a GPU.

## The idea

A von Neumann machine spends its life shuttling between memory and ALU:
fetch, decode, execute, repeat. The broth abolishes the trip. State lives
in a 3D tensor `(value, affinity, gradient) × D × H × W`, and one tick is:

```
space ← tanh(space + dt · conv3d(space, kernel))
```

Every voxel reads its 3×3×3 neighborhood and updates itself. That's the
whole instruction set. Complexity per tick is `O(volume)` but it is
*data-parallel all the way down* — the same program text scales from a
CPU tensor to a CUDA tensor to (conceptually) an optical or
memristive lattice, with zero code changes.

This is the substrate thinking behind cellular automata, continuous
reservoir computers and "physics processing units": if your data is
spatial, maybe your *computer* should be too.

## The three channels

| channel | role                                                                  |
|---------|-----------------------------------------------------------------------|
| 0       | value — the actual signal living in the soup                          |
| 1       | affinity — local mask modulating how strongly neighbors couple        |
| 2       | shift gradient — directional bias propagating through the lattice     |

The interaction kernel is a random 3×3×3×3×3 tensor (seeded, so runs are
reproducible). Swap it, and you get a different "physics".

## Install

```bash
pip install tensor-broth            # numpy-only core
pip install "tensor-broth[torch]"   # + PyTorch backend (CPU or CUDA)
```

No torch? A pure-NumPy engine (`sliding_window_view` + `tensordot`)
implements the same semantics so the paradigm is testable anywhere.
Honest note: the NumPy fallback is slower per voxel — it exists for
correctness and curiosity, not for winning benchmarks.

## Usage

### Python

```python
from tensor_broth import make_engine, run_von_neumann_vm

# the old world
vm = run_von_neumann_vm(n_steps=1_000_000)
print(f"VM: {vm.mops_per_sec:.2f} Mops/sec")

# the soup
engine = make_engine(backend="auto", cube_size=64)   # torch if available
elapsed = engine.run(1000)
print(f"broth: {engine.voxel_ops(1000) / elapsed / 1e9:.2f} GOPS effective")
print(f"activity: {engine.activity():.4f}")
```

### CLI

```
tensor-broth bench --cube 64 --steps 1000 --backend auto [--json]
tensor-broth sim   --cube 32 --steps 200
```

`bench` runs both paradigms and prints (or JSON-dumps) the comparison.
`sim` runs only the soup and reports final activity and throughput.

## Repository layout

```
tensor-broth/
├── src/tensor_broth/
│   ├── von_neumann.py    the fetch-decode baseline
│   ├── broth.py          SpatialEngine (torch) + SpatialEngineNumPy (fallback)
│   ├── benchmark.py      compare() + markdown reports
│   └── cli.py            bench / sim commands
├── tests/                13 pytest cases: determinism, bounds, throughput
├── examples/basic_bench.py
├── notebooks/            the original research notebook (Colab)
└── .github/workflows/ci.yml
```

## Where this could go

- Train the kernel (the "enzyme") instead of random — gradient descent over
  `conv3d` weights is just a 3D CNN with unusual semantics.
- Sparse lattices / octrees: most of the volume is boring; make the soup
  work only where it's interesting.
- Alternative nonlinearity — `tanh` today, but cellular-automaton
  thresholds (`sign`, `round`) would make it fully non-float. See our
  sibling project **timelock-ca** for exactly that obsession.
- Multi-channel reaction-diffusion (Gray-Scott in this framework is
  ~10 lines).

## Related work

- [Wolfram Physics](https://www.wolframphysics.org) — the space-is-the-program worldview.
- Lattice-gas automata & LBM — local rules, global physics (see **timelock-ca**).
- Reservoir computing — untrained recurrent substrates doing useful work.

## License

Apache-2.0 — see [LICENSE](LICENSE).
