# pet-kokkos

A C++ implementation of [PET](https://arxiv.org/abs/2305.19302) (Point Edge Transformer) machine-learning
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
to provide `pair_style pet` and `pet/kk`. Engines that load
[metatomic](https://github.com/metatensor/metatomic) plugins can use the plugin
in [`metatomic/`](#metatomic-plugin).

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
- `PET_USE_BLAS`: serial and OpenMP builds run their GEMMs through a CBLAS
  (OpenBLAS, MKL, BLIS, ...) when one is found (default on). Without one they
  fall back to a plain Kokkos loop, 12-20x slower on pet-mad. With OpenMP, an
  OpenMP-threaded BLAS shares Kokkos's threads.
- `PET_BUILD_TESTS`, `PET_BUILD_APPS`.
- `PET_BUILD_METATOMIC`: build the metatomic plugin (default off; see below).

When pet-kokkos is pulled in as a dependency (`add_subdirectory`, CPM or
FetchContent), the tests and apps are off by default and an existing
`Kokkos::kokkos` target is reused.

## metatomic plugin

`-DPET_BUILD_METATOMIC=ON` builds `metatomic/pet-kokkos-metatomic.so`, a
[metatomic](https://github.com/metatensor/metatomic) plugin. It needs metatomic's
C++ library and metatensor. metatomic-core is not on PyPI yet: build it from a
metatomic checkout (it needs Rust), and its Python package then ships the CMake
config, as metatensor's does:

```bash
pip install "metatensor-core>=0.2.4,<0.3" setuptools packaging cmake
pip install --no-build-isolation ./python/metatomic_core   # in the metatomic checkout
```

```bash
cmake --preset cuda -DKokkos_ARCH_ADA89=ON -DPET_BUILD_METATOMIC=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import metatomic; print(metatomic.utils.cmake_prefix_path)');$(python -c 'import metatensor; print(metatensor.utils.cmake_prefix_path)')"
```

```python
import metatomic as mta

mta.load_plugin("build-cuda/metatomic/pet-kokkos-metatomic.so")
model = mta.load_model("pet-mad-xs", {"gemm": "native"}, plugin_name="pet-kokkos")
```

`load_model` takes a model name or path prefix, found as `pet::Calculator`
finds it. The options, all strings, are `gemm` (as `PET_GEMM`; process-wide),
`attention` (as `PET_ATTN`; process-wide),
`recompute` (`auto`, `never`, `wide` or `layers`), `memory_budget_gib`, and
`device_id` (by default each node-local MPI rank takes its own GPU).

The model computes `energy` per system, with `positions` and `strain`
gradients, and per atom. It asks for a full pair list at the model's cutoff, and
for `charge` and `spin_multiplicity` if the model is charge-conditioned
(pet-omol). With `selected_atoms`, a system's energy is the selected atoms'
share and its gradients are with respect to every atom, so an engine can pass
ghost atoms out to the model's `interaction_range` (one cutoff per
message-passing layer, plus one for the adaptive cutoff) and select its own.

The plugin carries its own Kokkos, with its symbols hidden, and starts it with
the first model.

A CUDA build lists `cuda` first among its devices. Systems in GPU memory are
read where they are -- positions, types and the pair list -- and the outputs,
values and labels, stay on the GPU as DLPack-exportable arrays; nothing but the
cell, a selection and a few labels crosses the bus, and a call costs what
`pet-eval` does. Systems on the CPU work with any build; the evaluation still
runs on the build's device, and copying the pair list there adds a few
milliseconds per call on a few thousand atoms. What metatomic itself cannot do
yet:

- Check device outputs: `check_consistency` builds its expected labels on the
  CPU and fails on device blocks. Run without it on the GPU.
- Take custom data on the GPU, so a charge-conditioned model (pet-omol) needs
  its systems on the CPU.
- Execute on ROCm arrays at all, so a HIP build takes CPU systems only.

`metatomic/test_plugin.py` checks it against the goldens through metatomic's
Python API (see the file for what it needs). No engine loads metatomic plugins
yet -- metatomic's ASE and torch-sim engines take TorchScript models --
so [`metatomic/examples/ase_calculator.py`](metatomic/examples/ase_calculator.py)
is a small ASE calculator over the C API, with vesin's neighbour lists, for
MD and relaxations on either path.

## Models

Every published upet model, already converted, is on Hugging Face at
[peterspackman/pet-kokkos-models](https://huggingface.co/peterspackman/pet-kokkos-models):

```bash
hf download peterspackman/pet-kokkos-models --local-dir ~/.local/share/pet/models
hf download peterspackman/pet-kokkos-models --include "pet-mad-s.*" --local-dir models
```

Model weights aren't kept in this repository. To convert a model yourself,
from upet or a metatrain checkpoint:

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
set with `PET_GEMM` (`tf32`, `fp16`, `bf16` or `fp8`) or `Options::gemm`. On
NVIDIA GPUs from compute capability 8.0 `tf32` is the default: 10-20% faster on
an RTX 4080 and up to 2x on an H200, 0.1-0.25 meV/atom from fp32 and the same
NVE energy drift; `PET_GEMM=native` turns it off. `fp16` is about as accurate;
`bf16` is not (about 1 meV/atom, 4x the drift). `fp8` needs an Ada or Hopper GPU
and is only good enough for screening. On AMD GPUs, only `bf16` and `fp16` are
supported, and their fp32 already runs on the matrix cores.

The virial is `W = V·σ`, a symmetric tensor in Voigt order
`[xx, yy, zz, xy, xz, yz]`. Optimizers that work with strain gradients need
`dE/dε`, whose off-diagonal terms are `2·W_xy`.

## Environment variables

- `PET_MODEL_DIR`: extra directories to search for models.
- `PET_GEMM`: `native`, `tf32`, `fp16`, `bf16`, `fp8` or `ozaki`. Unset, NVIDIA
  GPUs from compute capability 8.0 (A100, H100/H200, RTX 30xx/40xx) use `tf32`
  (0.1-0.25 meV/atom from fp32, up to 2x faster) and everything else `native`;
  `PET_GEMM=native` keeps plain fp32 everywhere.
- `PET_ATTN`: `native` (default) or `tf32`. With `tf32`, attention's own
  products (scores, weights times values, and their adjoints) run on TF32
  tensor cores, on NVIDIA GPUs from compute capability 8.0 (elsewhere it is
  ignored). Separate from `PET_GEMM`, so the two can be measured apart: on its
  own 0.01-0.13 meV/atom and 1-6 meV/A from fp32 on the published models.
  Also `Options::attention`, and the metatomic option `attention`.
- `PET_CUTLASS=0`: use plain cuBLAS instead of the CUTLASS kernels.
- `PET_GRAPHS=0`: don't use CUDA/HIP graphs (useful when profiling).
- `PET_RECOMPUTE`: `never`, `wide` or `layers` forces how much of the forward
  the backward recomputes; by default the cheapest tier whose estimate fits the
  device (less a reserve of 10% or 1.5 GiB) is taken, and a pass that runs out
  of memory anyway is rerun a tier lower. `pet-eval --memory` shows the choice.
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
