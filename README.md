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

One evaluation with forces, neighbour list included, in fp32 (the default; see
[Faster modes](#faster-modes) for what tensor cores add). On an RTX 4080 SUPER,
with the speed-up over upet 0.3.0 (PyTorch) in brackets:

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

## Python

```bash
uv pip install pet-kokkos                 # CPU (not on PyPI yet: built by the Wheels workflow)
uv pip install "pet-kokkos[cuda12]"       # + the NVIDIA GPU core (compute capability 8.0+)
uv pip install "pet-kokkos[rocm6]"        # + the AMD Instinct core (MI200, MI300; needs ROCm 6)
uv build --wheel                          # the CPU wheel from this tree
```

```python
import pet_kokkos
from pet_kokkos.ase import PETCalculator

atoms.calc = PETCalculator("pet-mad-s")          # downloads the model on first use
atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()

calc = pet_kokkos.Calculator("pet-mad-s", gemm="tf32", attention="tf32")
r = calc.compute(numbers, positions, cell, pbc=True)   # energy, energies, forces, virial
rs = calc.compute_batch([(numbers, positions, cell, True), ...])
```

A model name not found locally (see Models) is downloaded from the Hugging
Face repository into `~/.local/share/pet/models`. The ASE calculator keeps a
Verlet cache of the neighbour topology across steps (`cache_neighbors=True`)
and reads a charge and spin multiplicity from `atoms.info["charge"]` and
`atoms.info["spin"]` for models conditioned on them. A wheel built with
`-Ccmake.define.PET_BUILD_METATOMIC=ON` also carries the metatomic plugin, at
`pet_kokkos.metatomic_plugin()`.

CPU wheels use OpenMP and OpenBLAS on Linux and Kokkos's C++ threads with
Accelerate on macOS (Apple's compiler has no OpenMP). The GPU cores are
separate packages (`python/gpu/`), as JAX's are: `pet_kokkos` uses one when it
is installed, and `PET_KOKKOS_DEVICE=cpu|cuda|rocm` chooses explicitly.

- `pet-kokkos-cuda12`: one wheel for every NVIDIA GPU from compute capability
  8.0 (A100, RTX 30xx/40xx, H100/H200 and later). Kokkos builds one
  architecture, sm_80; `NVCC_APPEND_FLAGS` adds sm_89 and sm_90 code and
  compute_90 PTX, so each GPU runs code for itself (an sm_80-only build was 6%
  slower in fp32 on an RTX 4080). The CUDA runtime and cuBLAS come from
  NVIDIA's pip wheels (`nvidia-cublas-cu12`, ...), the ones PyTorch and JAX
  use, found through the module's RPATH.
- `pet-kokkos-rocm6`: MI200 (gfx90a) and MI300 (gfx942). It links the
  system's ROCm 6 (HIP, rocBLAS, hipBLASLt), which must be installed, as with
  CuPy's ROCm wheels.

To build one yourself, e.g. for another architecture or toolkit:
`cd python/gpu/cuda12 && PET_NVCC_WRAPPER=<kokkos>/bin/nvcc_wrapper uv build
--wheel`. `python/tests` checks the package (`uv run --with pytest --with ase
pytest python/tests` with it installed).

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

The virial is `W = V·σ`, a symmetric tensor in Voigt order
`[xx, yy, zz, xy, xz, yz]`. Optimizers that work with strain gradients need
`dE/dε`, whose off-diagonal terms are `2·W_xy`.

### Faster modes

Everything runs in fp32 unless you ask otherwise: the answers are the
reference's, to fp32 rounding, on every device. Two independent switches trade
a little accuracy for speed on tensor cores. Each is process-wide and can be
set by environment variable, by `pet::Options`, or as a metatomic option:

| switch | what it changes | where it runs | env / `Options` / metatomic |
|---|---|---|---|
| GEMM mode | the dense layers' matrix products | see below | `PET_GEMM` / `gemm` / `gemm` |
| attention mode | attention's own products (scores, weights times values, their adjoints) | NVIDIA, compute capability 8.0+ | `PET_ATTN` / `attention` / `attention` |

**For MD or relaxations on a recent NVIDIA GPU** (A100, H100/H200, RTX 30xx/40xx),
`PET_GEMM=tf32 PET_ATTN=tf32` is the one to use. Measured against fp32, on
pet-mad-xs, pet-mad-s, pbe0-pet (pet-mols-s) and pet-omat-l, 648-atom water and
1728-atom diamond:

| | energy | largest force error |
|---|---|---|
| `PET_GEMM=tf32` | 0.08-0.37 meV/atom | 1.7-5 meV/Å |
| `PET_ATTN=tf32` | 0.01-0.13 meV/atom | 0.7-6 meV/Å |
| both | 0.02-0.31 meV/atom | 1.4-8 meV/Å |

all far below the models' own errors. With `PET_GEMM=tf32` the NVE energy drift
is the same as fp32's; it has not yet been measured with `PET_ATTN=tf32`.

What they buy, ms per evaluation with forces:

| | fp32 | `PET_GEMM=tf32` | + `PET_ATTN=tf32` |
|---|---|---|---|
| H200, pet-mad-s, 5184-atom water | 92.0 | 37.6 | 33.2 |
| H200, pet-mols-s, 5184-atom water | 98.5 | 51.0 | 39.0 |
| H200, pet-omat-l, 2592-atom water | 439 | 138 | 116 |
| RTX 4080 SUPER, pet-mad-s, 648-atom water | 24.5 | 17.3 | 16.7 |
| RTX 4080 SUPER, pbe0-pet, 1728-atom diamond | 98.9 | 85.6 | 71.9 |

An H200's fp32 runs on its CUDA cores, several times slower than its TF32
tensor cores, so it gains the most; an RTX 4080's TF32 and fp32 peaks are about
equal, and what it gains is from better-used kernels.

The other GEMM modes:

- `fp16`: about as accurate as `tf32` (0.08-0.37 meV/atom, 1-6 meV/Å on the
  models above); NVIDIA from compute capability 7.0, and AMD's matrix cores (on
  an MI250X, about 10% faster than fp32).
- `bf16`: much less accurate (up to 5 meV/atom and 7-42 meV/Å from fp32, 4x
  the NVE drift): its 7-bit mantissa is 8x coarser than fp16's. NVIDIA 8.0+
  and AMD. Not faster than `tf32` on NVIDIA.
- `fp8`: Ada or Hopper only, good enough for screening and nothing more.
- `ozaki`: for the fp64 build, fp64-accurate products on integer tensor cores.

On AMD GPUs fp32 already runs on the matrix cores, so there is no `tf32`, and
attention already uses them in fp32. A mode the device cannot run falls back
to fp32.

## Environment variables

- `PET_MODEL_DIR`: extra directories to search for models.
- `PET_GEMM`: `native` (default, fp32), `tf32`, `fp16`, `bf16`, `fp8` or
  `ozaki`; see [Faster modes](#faster-modes).
- `PET_ATTN`: `native` (default, fp32) or `tf32`; see
  [Faster modes](#faster-modes).
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
