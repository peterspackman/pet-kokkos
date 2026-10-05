# pet-kokkos

C++/Kokkos evaluation of **PET** (Point Edge Transformer) machine-learning
interatomic potentials — the [upet](https://github.com/lab-cosmo/upet) /
PET-MAD family — with energies, analytic forces and the virial, and **no
libtorch at run time**. One source tree runs on CPUs (serial, OpenMP), NVIDIA
GPUs (CUDA) and AMD GPUs (HIP).

> **Pre-release.** The API may still change.

```bash
pet-eval pet-mad-xs diamond.xyz --forces
# frame 0: N=8  E = -77.1825019678 eV  (-9.6478127460 eV/atom)
#   max|F| = 1.21209333e-02 eV/A
#   stress (eV/A^3):
#      1.71035128e-02  -2.06989277e-03   1.27728266e-03
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
  pet::Results r = calc.compute(water);  // r.energy[0] in eV, r.forces [N*3] in eV/A
}
Kokkos::finalize();
```

## What it does

- **Every published upet model.** All 20 checkpoints on `lab-cosmo/upet` agree
  with upstream's own evaluation (metatomic) to ~1e-7 relative in energy and
  ~1e-5 eV/Å in forces: both featurizers, 1–3 attention layers, both adaptive
  cutoff schemes, charge/spin conditioning. `tools/test_all_models.py` checks it.
- **Fast.** Against upet 0.3.0 on PyTorch, one evaluation with forces,
  neighbour list included, on an RTX 4080 SUPER:

  | model | 64-atom diamond | 648-atom water | 1728-atom diamond |
  |---|---|---|---|
  | pet-mad-xs | 3.1 ms (9.1×) | 7.5 ms (4.1×) | 18.4 ms (3.1×) |
  | pet-mad-s | 4.9 ms (8.5×) | 27.7 ms (2.2×) | 63.6 ms (1.9×) |
  | pet-omat-l | 23.6 ms (1.7×) | 206 ms (1.6×) | |

- **Batches.** Many structures evaluate in one pass; each gets the answer it
  would get alone.
- **MD engines.** An engine can hand over its own neighbour list once per
  rebuild and step positions on the device, getting forces, per-atom energies,
  per-atom virials and per-edge gradients back (`set_neighbors`,
  `compute_step`, `compute_edges`). [`lammps/`](lammps/README.md) is a LAMMPS
  package built on it: `pair_style pet` and `pet/kk`, across MPI ranks with
  GPU-aware MPI.
- **Deterministic.** Repeated evaluations are bit-identical, on CPU and GPU.

## Building

CMake ≥ 3.18 and a C++20 compiler (C++17 when building against Kokkos 4).
Kokkos 5, nlohmann/json and Catch2 are fetched at configure time.

```bash
cmake --preset serial && cmake --build build-serial -j
ctest --preset serial
```

```bash
cmake --preset openmp && cmake --build build-openmp -j
cmake --preset cuda   && cmake --build build-cuda   -j   # nvcc on PATH
cmake --preset hip    && cmake --build build-hip    -j
```

For CUDA, set your architecture, e.g. `-DKokkos_ARCH_ADA89=ON` for an RTX 40
series card. The `cuda` preset uses Kokkos' `nvcc_wrapper` as the compiler, which
exists only once Kokkos has been fetched: configure once with the `serial`
preset first, or pass the compiler explicitly.

| option | default | |
|---|---|---|
| `PET_BACKEND` | `serial` | `serial`, `openmp`, `cuda`, `hip`; taken from an existing `Kokkos::kokkos` target if there is one |
| `PET_PRECISION` | `mixed` | `mixed`, `fp32`, `fp64` (below) |
| `PET_WITH_CUTLASS` | `AUTO` | fused GEMM kernels on CUTLASS, CUDA only |
| `PET_WITH_VESIN` | `AUTO` | [vesin](https://github.com/Luthaf/vesin) for the host neighbour search |
| `PET_ARCH_NATIVE` | `ON` | `-march=native`; turn off when build and run hosts differ |
| `PET_BUILD_TESTS`, `PET_BUILD_APPS` | on when top-level | |

As a dependency (`add_subdirectory`, CPM, FetchContent) the tests and apps
default off and an existing `Kokkos::kokkos` target is used.

## Models

**Model weights are not in this repository and must never be committed**: some
checkpoints are distributed under terms that do not permit redistribution.

A model is a pair of files: `<name>.json` (hyperparameters, species,
composition energies, energy scale) and `<name>.safetensors` (weights). Convert
one from a published upet model or a metatrain checkpoint:

```bash
uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs
uv run tools/convert_pet.py --ckpt my-model.ckpt --out models/my-model
uv run tools/convert_pet.py --ckpt pbe0.ckpt --variant pbe0 --out models/pbe0-pet  # a non-default output head
```

A named model is looked for, in order, in `$PET_MODEL_DIR` (`:`-separated),
`./models` and `.`, `$XDG_DATA_HOME/pet/models` (else
`~/.local/share/pet/models`), and this source tree's `models/`. A directory
counts only if it holds both files. `pet-eval --models` prints the list.

## Numerics

**Precision**, fixed at build time by `PET_PRECISION`:

- `mixed` (default): fp32 network, fp64 geometry and energy, force and virial
  accumulation. What the reference tests are validated against.
- `fp32`: everything single.
- `fp64`: everything double, for checking. On GPUs with slow fp64,
  `PET_GEMM_MODE=ozaki` runs its matrix products to fp64 accuracy on integer
  tensor cores (the Ozaki scheme).

**Determinism.** No floating-point atomics feed the energy or forces: neighbour
slots are assigned in edge-list order, per-edge gradients are gathered through
the reverse-edge map rather than scattered, and per-structure sums are ordered.
Repeated runs are bit-identical; `tests/test_determinism.cpp` checks it.

**TF32** tensor-core GEMMs (`PET_TF32=1` or `Options::allow_tf32`) are off by
default. They are ~15% faster on an RTX 4080 and move energies by ~0.1 meV/atom
and forces by a few meV/Å.

**Virial convention.** The virial is the symmetric `W = V·σ` in Voigt order
`[xx, yy, zz, xy, xz, yz]`. A strain-gradient optimizer wants `dE/dε`, whose
off-diagonals are `2·W_xy`; apply that factor on your side.

## Runtime switches

| variable | |
|---|---|
| `PET_MODEL_DIR` | extra directories to search for models |
| `PET_TF32=1` | TF32 GEMMs |
| `PET_CUTLASS=0` | plain cuBLAS instead of the fused CUTLASS kernels |
| `PET_GRAPHS=0` | no CUDA/HIP graph replay (for profilers that time each kernel) |
| `PET_DEVICE_SEARCH=auto\|cells\|brute` | the device neighbour search |
| `PET_NEIGHBORS=builtin` | the built-in host search instead of vesin |
| `PET_GEMM_MODE=ozaki`, `PET_OZAKI_SLICES` | fp64 GEMM emulation (fp64 builds) |
| `PET_WS_POISON=1` | fill unwritten scratch with NaN, to catch reads before writes |

The neighbour searches and the cache are all checked against each other in the
tests; they find the same edges and differ only in the last bits.

## Testing

```bash
ctest --preset serial          # or openmp / cuda
```

| suite | needs a model | |
|---|---|---|
| `neighbors` | no | periodic images, neighbour packing, the reverse-edge map, adaptive cutoffs |
| `io` | no | extended XYZ |
| `cell_list` | yes | the device cell list against the brute-force search, and run to run |
| `golden` | yes | energies, per-atom energies, forces and stress against metatrain |
| `finite_diff` | yes | forces and virial against the model's own energy |
| `determinism` | yes | bit-identical repeats; a batch equals its structures alone |
| `paths_agree` | yes | host and device neighbour builds, the Verlet cache, a shrinking batch |
| `edges` | yes | engine neighbour lists: half and full, ghosts, MD stepping, exchange across ranks |
| `ozaki` | no | fp64 GEMM emulation against native fp64 |

Suites that need a model skip when none is installed. Reference values ship
in `tests/golden/` for four models: pet-mad-xs (v1.0 and v1.6), pbe0-pet, and a
synthetic two-attention-layer model built by
`tools/make_multilayer_checkpoint.py`. `$PET_TEST_MODELS` and
`$PET_TEST_GOLDEN_EXTRA` add models and golden directories.

Against the whole published catalogue:

```bash
uv run tools/test_all_models.py           # models up to 150 MB
uv run tools/test_all_models.py --all     # all 20, ~13 GB of downloads
```

## Licence

BSD 3-Clause, as upstream PET and metatrain.

PET is described in [Pozdnyakov & Ceriotti 2023](https://arxiv.org/abs/2305.19302);
the reference implementation is [metatrain](https://github.com/metatensor/metatrain).
This is an independent implementation that reads the same checkpoints.

This code began inside the author's klasp and lammps-pet-kokkos projects and is
relicensed here by its sole author. The LAMMPS package under `lammps/` is
GPL-2.0, as LAMMPS itself.
