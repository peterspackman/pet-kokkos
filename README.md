# pet-kokkos

A C++ implementation of PET (Point Edge Transformer) machine-learning
interatomic potentials, the [upet](https://github.com/lab-cosmo/upet) / PET-MAD
family, built on Kokkos. It computes energies, forces and the virial without
needing libtorch, and runs on CPUs (serial or OpenMP), NVIDIA GPUs (CUDA) and
AMD GPUs (HIP).

The API may still change before a first release.

```bash
$ pet-eval pet-mad-xs diamond.xyz --forces
frame 0: N=8  E = -77.1825019678 eV  (-9.6478127460 eV/atom)
  max|F| = 1.21209333e-02 eV/A
  ...
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
  pet::Results r = calc.compute(water);  // r.energy[0] in eV, r.forces in eV/A
}
Kokkos::finalize();
```

All 20 models published in lab-cosmo/upet agree with upstream's own evaluation
to about 1e-7 relative in energy and 1e-5 eV/Å in forces
(`tools/test_all_models.py`). Several structures can be evaluated in one batch,
and results are bit-identical from run to run on both CPU and GPU.

MD codes can pass in their own neighbour list and get back forces, per-atom
energies and virials, or per-edge gradients (`set_neighbors`, `compute_step`,
`compute_edges`). The LAMMPS package in [`lammps/`](lammps/README.md) uses this
to provide `pair_style pet` and `pet/kk`.

## Performance

One evaluation with forces, neighbour list included. On an RTX 4080 SUPER, with
the speed-up over upet 0.3.0 (PyTorch) in brackets:

| model | 64-atom diamond | 648-atom water | 1728-atom diamond |
|---|---|---|---|
| pet-mad-xs | 3.1 ms (9.1×) | 7.5 ms (4.1×) | 18.4 ms (3.1×) |
| pet-mad-s | 4.9 ms (8.5×) | 27.7 ms (2.2×) | 63.6 ms (1.9×) |
| pet-omat-l | 23.6 ms (1.7×) | 206 ms (1.6×) | |

On one GCD of an MI250X (Setonix):

| model | 64-atom diamond | 648-atom water | 1728-atom diamond |
|---|---|---|---|
| pet-mad-xs | 4.5 ms | 7.2 ms | 15.5 ms |
| pet-mad-s | 5.9 ms | 27.1 ms | 72.4 ms |
| pet-mad-m | 16.0 ms | 121 ms | 274 ms |

## Building

You need CMake 3.18 or newer and a C++20 compiler (C++17 if you build against
Kokkos 4). Kokkos 5, nlohmann/json and Catch2 are downloaded when you configure.

```bash
cmake --preset serial && cmake --build build-serial -j
ctest --preset serial
```

The other presets are `openmp`, `cuda` and `hip`. For CUDA, also set your GPU
architecture, e.g. `-DKokkos_ARCH_ADA89=ON` for an RTX 40 series card. The
`cuda` preset compiles with Kokkos' `nvcc_wrapper`, which doesn't exist until
Kokkos has been downloaded, so configure the `serial` preset once first or pass
the compiler yourself.

CMake options:

- `PET_BACKEND`: `serial`, `openmp`, `cuda` or `hip`. If a `Kokkos::kokkos`
  target already exists, its backend is used.
- `PET_PRECISION`: `mixed` (default), `fp32` or `fp64`; see below.
- `PET_WITH_CUTLASS`: fused GEMM kernels with CUTLASS (CUDA only; on if found).
- `PET_WITH_VESIN`: use [vesin](https://github.com/Luthaf/vesin) for the host
  neighbour search (on if found).
- `PET_ARCH_NATIVE`: build with `-march=native` (default on). Turn it off if you
  build on one machine and run on another.
- `PET_BUILD_TESTS`, `PET_BUILD_APPS`.

When pet-kokkos is pulled in as a dependency (`add_subdirectory`, CPM or
FetchContent), the tests and apps are off by default and an existing
`Kokkos::kokkos` target is reused.

## Models

Model weights are not included here, and shouldn't be added, because some
checkpoints don't allow redistribution. Convert them yourself from a published
upet model or a metatrain checkpoint:

```bash
uv run tools/convert_pet.py --model pet-mad-xs --out models/pet-mad-xs
uv run tools/convert_pet.py --ckpt my-model.ckpt --out models/my-model
```

`--version` picks a published version and `--variant` another output head.

Each model is two files, `<name>.json` (hyperparameters, species and
composition energies) and `<name>.safetensors` (weights). When a model is
asked for by name, pet-kokkos searches `$PET_MODEL_DIR` (a `:`-separated list),
then `./models`, `.`, `$XDG_DATA_HOME/pet/models` (or
`~/.local/share/pet/models`), and finally `models/` in the source tree.
`pet-eval --models` lists what it can find.

## Numerics

`PET_PRECISION` sets the precision when you build:

- `mixed` (default): the network runs in fp32. Geometry is fp64, and so are the
  energy, force and virial sums. This is the build the reference tests check.
- `fp32`: single precision throughout.
- `fp64`: double precision throughout, for checking results. On GPUs with slow
  fp64, `PET_GEMM=ozaki` computes the matrix products to fp64 accuracy using
  integer tensor cores (the Ozaki scheme).

The network's matrix products can also run in lower precision on tensor cores,
set with `PET_GEMM` (`tf32`, `fp16`, `bf16` or `fp8`) or `Options::gemm`. This
is off by default. `tf32` and `fp16` are 10–20% faster on an RTX 4080 and
change forces by about 1 meV/Å. `fp8` needs an Ada or Hopper GPU and is only
good enough for screening. On AMD GPUs, only `bf16` and `fp16` are supported.

The virial is `W = V·σ`, a symmetric tensor in Voigt order
`[xx, yy, zz, xy, xz, yz]`. Optimizers that work with strain gradients need
`dE/dε`, whose off-diagonal terms are `2·W_xy`.

## Environment variables

- `PET_MODEL_DIR`: extra directories to search for models.
- `PET_GEMM`: `native`, `tf32`, `fp16`, `bf16`, `fp8` or `ozaki`.
- `PET_CUTLASS=0`: use plain cuBLAS instead of the CUTLASS kernels.
- `PET_GRAPHS=0`: don't use CUDA/HIP graphs (useful when profiling).
- `PET_DEVICE_SEARCH`: `auto`, `cells` or `brute`, for the GPU neighbour search.
- `PET_NEIGHBORS=builtin`: use the built-in host neighbour search instead of
  vesin.

## Testing

```bash
ctest --preset serial          # or openmp / cuda / hip
```

Most tests need models and are skipped without them;
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) shows how to make the
ones the reference values in `tests/golden/` were computed with.
`uv run tools/test_all_models.py` checks against every published model.

## Licence

pet-kokkos is BSD 3-Clause, the same as PET and metatrain. The LAMMPS package in
`lammps/` is GPL-2.0, the same as LAMMPS.

PET is described in [Pozdnyakov & Ceriotti 2023](https://arxiv.org/abs/2305.19302),
and its reference implementation is
[metatrain](https://github.com/metatensor/metatrain). pet-kokkos is a separate
implementation that reads the same checkpoints. It started as part of the
author's klasp and lammps-pet-kokkos projects.
