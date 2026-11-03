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
uv pip install pet-kokkos                 # CPU
uv build --wheel                          # the CPU wheel from this tree
```

GPU builds are the same package with a local version label, as PyTorch's are
(`pet-kokkos 0.1.0+cu12`), served from the GitHub releases through an index
for each, since PyPI takes no local versions:

```bash
uv pip install pet-kokkos --extra-index-url https://peterspackman.github.io/pet-kokkos/cu12
```

| Index | GPUs | Needs |
|---|---|---|
| `cu12` | NVIDIA, compute capability 8.0+ (A100, RTX 30xx-50xx, H100/H200, B200) | driver for CUDA 12; the CUDA runtime and cuBLAS come from NVIDIA's pip wheels |
| `cu13` | as `cu12` | driver for CUDA 13 (580+) |
| `rocm6` | AMD Instinct MI200 (gfx90a), MI300 (gfx942) | ROCm 6 installed |
| `rocm7` | as `rocm6`, and MI350 (gfx950) | ROCm 7 installed |

In a uv project, pin the package to one:

```toml
[[tool.uv.index]]
name = "pet-cu12"
url = "https://peterspackman.github.io/pet-kokkos/cu12"
explicit = true

[tool.uv.sources]
pet-kokkos = { index = "pet-cu12" }
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
Accelerate on macOS (Apple's compiler has no OpenMP). A GPU wheel carries
only its GPU's core: `pet_kokkos.backend()` says which.

- CUDA: one wheel for every NVIDIA GPU from compute capability 8.0. Kokkos
  builds one architecture, sm_80; `NVCC_APPEND_FLAGS` adds code for sm_89,
  sm_90, sm_100 and sm_120 and PTX for later GPUs, so each runs code for itself
  (an sm_80-only build was 6% slower in fp32 on an RTX 4080). The CUDA runtime
  and cuBLAS are NVIDIA's pip wheels, the ones PyTorch and JAX use, found
  through the module's RPATH.
- ROCm: CDNA GPUs only (the attention's matrix-core kernels). It links the
  system's ROCm (HIP, rocBLAS, hipBLASLt), as CuPy's ROCm wheels do.

`PET_KOKKOS_VARIANT=cu12|cu13|rocm6|rocm7 uv build --wheel` builds one
yourself, e.g. for another architecture or toolkit (pyproject.toml's overrides
say what each needs); `python/wheels/` has CI's toolchains. `python/tests`
checks the package (`uv run --with pytest --with ase pytest python/tests` with
it installed).

### Second derivatives

`pet_kokkos.properties` computes Hessians, vibrational modes, phonons (via
phonopy, `pip install pet-kokkos[phonons]`) and elastic tensors. A structure is
a `(numbers, positions, cell, pbc)` tuple, a chmpy `System`, `Crystal` or
`Molecule`, or anything else with those four attributes. None of it needs
ASE.

```python
from pet_kokkos import properties
vib = properties.vibrations(calc, molecule)              # cm^-1, rigid modes projected out
el = properties.elastic_tensor(calc, crystal)            # relaxed-ion C, GPa, no relaxations
ph = properties.phonons(calc, crystal, [2, 2, 2])        # a phonopy.Phonopy with force constants
H = properties.hessian(calc, atoms, method="fd", points=4, step=0.005)
```

They are all built on the derivatives of the forces and virial along a set of
directions (atomic displacements, strains), which come one of two ways:

- `method="exact"`: a Hessian-vector product. The network's forward and its
  hand-written backward are differentiated along every direction at once
  (forward-over-reverse), so there is no step and no truncation error. The
  K directions run as extra rows of the same GEMMs. `Calculator.response`
  exposes it directly. It works for every published model: both featurizers
  (pet-mad, pet-omol; pet-mols-s), and the solver, grid or no adaptive cutoff.
- `method="fd"`: central differences of the forces, every displaced structure
  of a property in one batched evaluation. The stencil can be set: `points`
  (2, 4 or 6), `step` (Å), `strain_step`. It works with any model.

`method="auto"`, the default, takes the exact product, and falls back to
finite differences for a model it cannot differentiate.
Measured against the exact product in an fp64 build:

| | Hessian | frequencies | elastic constants |
|---|---|---|---|
| exact, fp32 | ~3e-5 eV/Å² | ~0.0005 cm⁻¹ | <0.0001 GPa |
| `fd`, `points=2, step=0.0025` | ~5e-3 eV/Å² | ~0.1 cm⁻¹ | ~0.02 GPa |
| `fd`, `points=4, step=0.005` | ~1e-3 eV/Å² | ~0.02 cm⁻¹ | ~0.01 GPa |
| exact, TF32 GEMMs and attention | ~3e-2 eV/Å² | ~0.1–0.4 cm⁻¹ | ~0.02 GPa |
| `fd`, TF32 GEMMs or attention | | 1–20 cm⁻¹ | 1.5–8 GPa |

These were measured on benzene, the CO2 crystal and silicon with pet-mad-s.
fp32's own noise is not what limits the finite differences; the stencil's
truncation is. Near soft or unstable modes the relaxed-ion elastic tensor
divides by small eigenvalues, and finite-difference errors grow there: on an
unrelaxed acetic acid cell they reached 0.2–0.6 GPa, and moved with the step.
With finite differences, TF32 is refused unless `allow_reduced_precision=True`.
The exact product has no step to amplify TF32's noise, so it is allowed there.
(For forces, TF32 is 1.6–2× faster on an H200; it hasn't been timed for the
product.)

The cosine cutoff (pet-mols-s) is smooth only to first order: its second
derivative jumps at both ends of the taper (4.3 and 4.5 Å), so the Hessian
itself jumps where a pair distance crosses one. A finite difference whose
stencil straddles such a distance is off by up to ~0.04 eV/Å². Benzene's meta
H–H distance, 4.2996 Å, is one such case. The exact product gives the Hessian
at the geometry itself.

`smooth_cutoff=True` (`pet::Options::smooth_cutoff`, `pet-eval
--smooth-cutoff`) swaps the cosine taper for 1 − (35x⁴ − 84x⁵ + 70x⁶ − 20x⁷).
It has the same ends, with its first three derivatives zero at both, so the
Hessian and its strain derivative are continuous. That matters for
quasi-harmonic free energies, which differentiate force constants with respect
to strain: with the cosine, pet-mols-s's dF_vib/dε for naphthalene moved by up
to 50% between steps of 3e-4 and 3e-3; with the smooth taper it holds to about
0.5%. It does change the model, slightly. Over the X23 crystals, energies move
by at most 0.05 meV/atom and forces by about 6e-4 eV/Å RMS. Stresses move by
under 0.02 GPa, except oxalic acid α (0.05 GPa) and urea (0.14 GPa). The option
does nothing for a bump-tapered model (pet-mad, pet-omol).

The exact Hessian is symmetric, and the response to a translation is zero, to
rounding. It also carries the solver adaptive cutoff's full second derivative.
The forward treats that cutoff's root and slope as constants, as metatrain
does, which is right for forces; a double backward through that graph, as torch
would do it, leaves out terms that are 4% of the virial's response to a strain.

With pet-mad-s on an RTX 4080 SUPER:

| | exact | fd, 2-point | fd, 4-point |
|---|---|---|---|
| benzene crystal, 48 atoms: Hessian | 0.35 s | 0.51 s | 1.03 s |
| anthracene, 48 atoms: Hessian | 0.39 s | 0.57 s | 1.19 s |
| anthracene: elastic tensor | 0.42 s | 0.55 s | 1.11 s |
| anthracene 2×2×2: phonons (36 displacements of 384 atoms) | 1.11 s | 1.19 s | |

The same anthracene elastic tensor computed one structure per call takes
1.96 s.

How a direction of the exact product compares with a force evaluation, with
pet-mad-s (`pet-eval STRUCTURE --hvp K` times it):

| | RTX 4080 SUPER | MI250X (one GCD) |
|---|---|---|
| diamond, 64 atoms | 2.9 ms per direction (forces 4.9 ms) | 2.5 ms (forces 4.9 ms) |
| water, 648 atoms | 41 ms (forces 21.7 ms) | 26 ms (forces 21 ms) |

So a direction costs about one evaluation in small cells, and up to two at
several hundred atoms. That is at most what a 2-point difference costs per
direction, with none of its error. On AMD's CDNA GPUs the attention's tangents
run on the matrix cores, as its forward and backward do.

`properties.HessianOperator(calc, structure)` wraps the exact product as a
matrix-free operator (a scipy `LinearOperator` via `as_linear_operator()`;
mass-weighted and with rigid modes projected out if asked). One block of k
vectors is one call with k directions, so Krylov methods such as `cg`,
`eigsh` and Newton-CG pay only for the products they take.

`elastic_tensor` uses it from 100 atoms up (`solver="auto"`). It solves for
the internal relaxation by block CG, six products per iteration, preconditioned
with a pair Hessian (Packwood et al.'s "Exp"). That took 25–40 iterations on
molecular crystals from 12 to 384 atoms. For anthracene 2×2×2 (384 atoms) it
is 6.6 s and 247 products, against 1,158 products to build the Hessian, and it
reproduces the unit cell's tensor to <1e-4 GPa. If the atoms are not at a
minimum, it falls back to building the Hessian.

Finding the lowest few modes iteratively does not pay off for molecular
crystals. LOBPCG needed 400–7,500 products for the six lowest modes, against
3N for the whole Hessian: the spectrum spans a stiffness ratio of ~10⁴, and a
pair preconditioner does not capture the soft intermolecular modes those six
are made of. For those, take the full exact Hessian.

PET is not exactly rotationally equivariant, and that shows up in second
derivatives at a level well above the finite-difference error:

- A molecule's rotations come out at 6–18 cm⁻¹ rather than zero, even in fp64,
  so `vibrations` projects them out.
- Silicon's cubic elastic constants differ from one another by ~0.5 GPa.
- Silicon's triply degenerate optical mode is split by ~1.5 cm⁻¹.

Translations are exact: the model sees only edge vectors.

`symmetry=True` (in `hessian`, `vibrations` and `elastic_tensor`; needs
spglib) averages the result over the crystal's space group, as phonopy does for
its force constants. It is opt-in, because it changes the numbers. Silicon then
comes out exactly cubic, with a degenerate optical mode. The same symmetry
breaking moves atoms during an unconstrained relaxation: X23 crystals relaxed
from their symmetric cells drifted 0.01–0.05 Å off their space groups. So relax
with the space group held first if you want to symmetrise: chmpy's
`relax(crystal, calc, symmetry=True)` varies only the asymmetric unit.

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
pet-mad-xs, pet-mad-s, pet-mols-s and pet-omat-l, 648-atom water and
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
| RTX 4080 SUPER, pet-mols-s, 1728-atom diamond | 98.9 | 85.6 | 71.9 |

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

### Non-conservative forces

Most published models also have heads that predict forces and stress
directly. `pet::Options::non_conservative` (`non_conservative=True` in Python,
`--non-conservative` for `pet-eval`) takes forces and virial from them instead
of from the energy's gradient, as upet's `non_conservative=True` does. There is
no backward pass, so an evaluation with forces costs about what the energy
alone does. With pet-mad-s on an RTX 4080 SUPER:

| | conservative | non-conservative |
|---|---|---|
| 624-atom Cu-BTC | 18.0 ms | 8.9 ms |
| 384-atom anthracene supercell | 16.7 ms | 8.7 ms |

Each structure's net force is removed, as metatomic does (except over several
ranks, where no rank sees the whole structure). Edge gradients and per-atom
virials need the backward pass, so they are not available in this mode.

These forces are not the gradient of the energy. They differ from it by about
0.1 eV/Å RMS on the structures above. They also aren't exactly rotationally
equivariant, and they don't conserve energy in NVE; upet uses them with a
conservative correction every few steps. Check the stress head before relying on
it: pet-mad-s's gives about 0.15 GPa for an unrelaxed anthracene cell whose
conservative stress is up to 20 GPa (metatomic gives the same), so it is no use
for cell relaxations or NPT there. A model with a force head and no stress head
(pet-omol) returns a zero virial in this mode.

Models converted before the converter kept these heads don't have them:
convert them again.

### Uncertainty

pet-mad (xs, s and m) and pet-mols-s come with metatrain's LLPR uncertainty.
`pet::Options::uncertainty` (`uncertainty=True` in Python, `--uncertainty` for
`pet-eval`) returns it alongside the energy:

- `energy_uncertainty`: each structure's, a standard deviation in eV (upet's
  `get_energy_uncertainty()`);
- `per_atom_energy_uncertainty`: each atom's, the one upet's
  `uncertainty_threshold` warns on;
- `energy_ensemble`: the 128 members of the model's energy ensemble, which
  average to the energy.

It is computed from the last-layer features the forward pass already has, so
it costs little: 0.5 ms on top of 18.9 ms for 624-atom Cu-BTC with pet-mad-s,
forces included. The Cholesky factor it uses is badly conditioned (about 4e8
for pet-mad-s), so pet-kokkos applies its inverse in double precision; the
results match metatomic's float64 evaluation to about 1e-6. The force and
stress uncertainties upet derives from ensembles are not implemented.

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
