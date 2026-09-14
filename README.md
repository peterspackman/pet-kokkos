# pet-kokkos

Pure C++/Kokkos evaluation of **PET** (Point Edge Transformer) machine-learning
interatomic potentials: energy, conservative forces and the virial, with **no
libtorch at run time**. One source tree runs on CPU (serial or OpenMP), NVIDIA
(CUDA) and AMD (HIP).

> **Pre-alpha.** The API will change. See [PLAN.md](PLAN.md) for where this is
> going.

```bash
pet-eval pet-mad-xs structure.xyz --forces
# frame 0: N=8  E = -77.0418057655 eV  (-9.6302257207 eV/atom)
#   max|F| = 1.58410661e-02 eV/A
#   stress (eV/A^3):
#      1.08528671e-02   2.01368111e-03  -1.55941820e-03
#      ...
```

```cpp
#include <pet/calculator.hpp>

Kokkos::initialize(argc, argv);
{
    pet::Calculator calc("pet-mad-xs");
    pet::System water;
    water.n_atoms = 3;
    water.atomic_numbers = {8, 1, 1};
    water.positions = {0, 0, 0.11926, 0, 0.76323, -0.47704, 0, -0.76323, -0.47704};

    pet::Results r = calc.compute(water);
    // r.energy[0] == -15.2855578467 eV;  r.forces is [N*3] eV/Angstrom
}
Kokkos::finalize();
```

## Why

PET is a non-equivariant transformer over per-atom neighbour environments. It
has no spherical harmonics, no Clebsch–Gordan products and no irreps: the
geometry enters through a single `Linear(4 -> d_pet)` on
`[edge_vector(xyz), distance]`, and everything after that is dense transformer
math — QKV attention and MLPs. That is what makes a portable Kokkos
implementation practical, and what this is.

The reference implementation is Python on PyTorch
([`lab-cosmo/upet`](https://github.com/lab-cosmo/upet),
[`metatensor/metatrain`](https://github.com/metatensor/metatrain)), and their
LAMMPS integration calls a TorchScript model through libtorch. This evaluates
the network directly against safetensors weights, with a hand-written analytic
backward. The two share an architecture and a weight format, not an
implementation.

What you get for that:

- **No libtorch**, anywhere, at run time. The dependencies are Kokkos and
  nlohmann/json, both fetched at configure time.
- **Determinism.** A PET evaluation here is bit-exactly reproducible run to run,
  on CPU and on GPU. See below — it is not free, and it is not an accident.
- **Batching.** Many small structures evaluate in one pass, so a GPU is as busy
  on a thousand 8-atom cells as on one big one.
- **A device-resident path.** Positions in, forces out, without the geometry
  ever crossing the bus — which is what a relaxer or an MD driver actually wants.

## Building

Needs CMake ≥ 3.18 and a C++20 compiler. Kokkos, nlohmann/json and Catch2 are
fetched automatically.

```bash
cmake --preset serial && cmake --build build-serial -j
ctest --preset serial
```

Other backends:

```bash
cmake --preset openmp && cmake --build build-openmp -j
cmake --preset cuda   && cmake --build build-cuda   -j   # needs nvcc on PATH
cmake --preset hip    && cmake --build build-hip    -j
```

For CUDA you may also want your architecture, e.g. `-DKokkos_ARCH_ADA89=ON` for
an RTX 40-series card. The `cuda` preset points `CMAKE_CXX_COMPILER` at Kokkos'
`nvcc_wrapper`; on a first configure that file does not exist yet, so configure
once with the `serial` preset (which fetches Kokkos) or pass an explicit path.

### Options

| option | default | |
|---|---|---|
| `PET_BACKEND` | `serial` | `serial` \| `openmp` \| `cuda` \| `hip` |
| `PET_PRECISION` | `mixed` | `mixed` \| `fp32` \| `fp64` — see below |
| `PET_BUILD_TESTS` | on if top-level | |
| `PET_BUILD_APPS` | on if top-level | the `pet-eval` CLI |
| `PET_ARCH_NATIVE` | `ON` | `-march=native`; turn off if build and run hosts differ |

As a dependency (`add_subdirectory`, CPM, FetchContent) tests and apps default
off, and an existing `Kokkos::kokkos` target is reused rather than a second
Kokkos being fetched.

## Models

**Model weights are not in this repository, and must never be committed** — some
checkpoints are distributed under terms that do not permit redistribution.

A model is a pair of files, `<name>.json` (hyperparameters, species map,
composition energies, energy scale) and `<name>.safetensors` (the weights).
Convert one from a metatrain checkpoint or a named `upet` model:

```bash
uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs
uv run tools/convert_pet.py --ckpt my-model.ckpt --variant pbe0 --out models/my-model
```

`pet-eval --models` prints where a named model is looked for:

1. `$PET_MODEL_DIR` (`:`-separated) — the knob for batch jobs
2. `./models` and `.` — a self-contained run directory
3. `$XDG_DATA_HOME/pet/models`, else `~/.local/share/pet/models`
4. this source tree's `models/` (git-ignored)

A directory only matches if it holds **both** halves of the pair.

## Architectures supported

Two validated families, each covering the opposite branch of every config axis.
The loader rejects any other mix rather than silently running the wrong path.

| | **pet-mad-xs** | **pbe0-pet** | **pet-mad-xs v1.6** | **pet-attn2**¹ |
|---|---|---|---|---|
| featurizer | feedforward | residual | feedforward | feedforward |
| transformer | PreLN | PostLN | PreLN | PreLN |
| normalization | RMSNorm | LayerNorm | RMSNorm | RMSNorm |
| activation | SwiGLU | SiLU | SwiGLU | SwiGLU |
| central token | expanded | non-expanded | expanded | expanded |
| cutoff | Bump, adaptive **grid** | Cosine, fixed | Bump, adaptive **solver** | Bump, adaptive solver |
| attention layers | 1 | 2 | 1 | **2** |

¹ Synthetic. Every published upet model uses `num_attention_layers = 1` — the
larger ones add GNN layers instead — while metatrain's *default* is 2, so a
locally trained model can easily need it and there is nothing real to validate
against. `tools/make_multilayer_checkpoint.py` builds a genuine two-block
metatrain model from a one-block one (with the blocks deliberately made
non-identical, so an implementation that read block 0's weights for every block
would fail), and its goldens come from metatrain like all the others.

Both adaptive-cutoff schemes are implemented — `"grid"` (the legacy
probe-grid average) and `"solver"` (Newton–bisection root find, metatrain's
current default). A checkpoint carrying no `adaptive_cutoff_method` predates the
choice and is read as `"grid"`, which is what the metatrain that produced it
used; anything else is refused by name rather than defaulted, because the two
schemes choose different per-atom cutoffs and picking the wrong one is silently
wrong energies, not an error.

Both featurizers handle any `num_attention_layers >= 1`. Two limits are enforced
at load time rather than assumed:

- the residual path has no adaptive-cutoff chain rule, so a residual model with
  `num_neighbors_adaptive > 0` is rejected — its forces would be silently
  inconsistent with its energy;
- `zbl` and long-range models are rejected.

## Precision

Compile-time, via `PET_PRECISION`:

- **mixed** (default) — `float` network, `double` geometry, energy, force and
  virial accumulation. Matches metatrain's fp32 weights while keeping the
  conserved quantities in fp64. This is what the goldens are validated against.
- **fp32** — everything single. Fastest on consumer GPUs.
- **fp64** — everything double. A correctness instrument: on a GeForce card fp64
  runs at 1/64 of fp32, so do not read its timings as performance.

## Determinism

A PET evaluation is bit-exactly reproducible run to run, which matters because a
relaxation is a chaotic map — last-bit force noise grows into multi-kJ/mol
differences in relaxed energies and reshuffled rankings. Getting there meant
removing float atomics from every reduction the energy depends on:

- neighbour slots are assigned by a per-atom walk in edge-list order, not by an
  `atomic_fetch_add` in thread-arrival order (this was by far the largest source);
- per-edge force gradients are **gathered** per atom through the reverse-edge map
  instead of scattered with two atomics per edge;
- the per-structure energy and virial are ordered segmented sums over each
  structure's contiguous atom range, not atomic scatters;
- the attention backward's cutoff adjoint is summed over heads in index order in
  a single thread.

Integer counts are order-independent in value, so the count-plus-scan that
recovers those ranges is itself safe. `tests/test_determinism.cpp` checks this
against the model, needing no golden.

TF32 tensor-core GEMMs (`PET_TF32=1`, or `Options::allow_tf32`) are **off by
default** and must stay off for anything compared against a reference. They are
not a free speedup: on the 8-atom `pet-mad-xs` crystal golden, turning them on
moves the total energy by 9.4 meV — 1.2 meV/atom, about 2800x the fp32 noise the
same golden otherwise sits at. The setting is process-global, because cuBLAS
fixes a handle's math mode when the handle is created.

## Virial convention

This library returns the **symmetric** virial in Voigt order
`[xx, yy, zz, xy, xz, yz]` — the physical `W = V·σ`, which its goldens are
validated against. A strain-DOF optimizer usually wants `dE/dε`, whose
off-diagonals are `W_xy + W_yx = 2·W_xy`. Apply that factor at your own seam;
doing it here would make the returned quantity something other than a stress.

## Testing

```bash
ctest --preset serial          # or openmp / cuda
```

| suite | needs a model? | what it holds |
|---|---|---|
| `neighbors` | no | periodic images, NEF packing, the reverse-edge map, adaptive cutoff |
| `io` | no | extended-XYZ reading and writing |
| `golden` | yes | energy, per-atom energy, forces and stress against metatrain's own evaluation |
| `finite_diff` | yes | `F = -dE/dx` and `W = dE/dε` against **this model's** energy — no reference implementation involved |
| `determinism` | yes | bit-exact repeats; a batch equals the structures evaluated alone |
| `paths_agree` | yes | host vs device neighbour builds; fresh and reused Verlet cache |

Model-dependent suites **skip** when no model is installed, so a fresh clone is
green. Put a model on the search path and they start running. Goldens for all
three architectures above are shipped; reproduce the models with

```bash
uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs-v1.6
```

`$PET_TEST_MODELS` and `$PET_TEST_GOLDEN_EXTRA` (both `:`-separated) add model
names and golden directories at run time, for a model whose weights cannot live
in this tree.

## Provenance and licence

BSD 3-Clause, matching upstream PET/metatrain.

This code was extracted from [klasp](https://github.com/peterspackman/klasp)'s
`src/pet`, which began life in `peterspackman/lammps-pet-kokkos` as
`lib/pet-kokkos`. All of it is the same author's. The GPLv2 on that LAMMPS
repository attaches to the pair style in its `src/ML-PET`, not to this library,
which never included a LAMMPS header; klasp is GPL-3 and this is a deliberate
relicence by its author to match the ecosystem it plugs into.

PET the architecture is published work
([Pozdnyakov & Ceriotti 2023](https://arxiv.org/abs/2305.19302)), and the
reference implementation is `metatensor/metatrain` (BSD-3-Clause) — the same
models `tools/convert_pet.py` converts. Nothing here is a port of their C++,
because there is none to port.
