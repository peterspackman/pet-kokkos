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

Grimme's D3 dispersion correction can be added to any model's results, or
evaluated on its own, on the device ([D3 dispersion](#d3-dispersion)).

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

Everything runs fp32-accurate unless you ask otherwise: the answers are the
reference's, to fp32 rounding, on every device. The one exception is on by
default on AMD GPUs and changes the answer by less than fp32's own error (the
fast sigmoid, below). Two independent switches trade
a little accuracy for speed on tensor cores. Each is process-wide and can be
set by environment variable, by `pet::Options`, or as a metatomic option:

| switch | what it changes | where it runs | env / `Options` / metatomic |
|---|---|---|---|
| GEMM mode | the dense layers' matrix products | see below | `PET_GEMM` / `gemm` / `gemm` |
| attention mode | attention's own products (scores, weights times values, their adjoints) | NVIDIA, compute capability 8.0+ | `PET_ATTN` / `attention` / `attention` |

**The fast sigmoid (AMD GPUs, on by default).** On an AMD GPU in an fp32
build, the sigmoid in SiLU and SwiGLU, forward and backward, is the hardware's
exp and reciprocal (1 ulp each) instead of the library exp and the IEEE
division. That saves about 20 instructions a value, which matters because the
activations are computed in the matrix-core GEMMs' epilogues: the SiLU GEMMs
run 5-15% faster. `PET_FAST_SIGMOID=0` turns it off and gives back the accurate
sigmoid's answer to the bit. The forward and backward use the same sigmoid, so
forces stay the gradient of the energy computed, and results still repeat to
the bit and do not depend on the batch. Elsewhere (NVIDIA, CPUs, fp64 builds)
the switch does nothing. Measured on an MI250X against the fp64 build, on
pet-mad-s, pet-mad-xs and pet-omat-s, 648- and 5184-atom water and 1728-atom
diamond:

| | energy | largest force error | RMS force error |
|---|---|---|---|
| accurate sigmoid against fp64 | 1e-6-2.1e-4 meV/atom | 6.0-7.2e-3 meV/Å | 1.4-2.0e-3 meV/Å |
| fast sigmoid against fp64 | 1.3e-5-2.0e-4 meV/atom | 5.6-7.6e-3 meV/Å | 1.4-2.0e-3 meV/Å |
| fast against accurate | 1.3-3.0e-5 meV/atom | 2.1-3.5e-3 meV/Å | 0.5-1.0e-3 meV/Å |

That is, the change is below fp32's own error. Over 1000 NVE steps (648-atom
water, 0.5 fs) the energy drifts 0.093 meV/atom with it against 0.090 without
(pet-mad-s), 0.112 against 0.105 (pet-mad-xs). On one MI250X GCD it takes
pet-mad-s at 5184-atom water from 117.8 to 116.7 ms, pet-omat-s from 85.7 to
85.0, pet-mad-xs from 28.2 to 27.6, and diamond 1728 with pet-mad-xs from
12.9 to 12.7.

**fp32 on tensor cores (3xTF32, A100, H100/H200 and B200).**
On these GPUs the TF32 tensor cores run 7-8x the rate of the fp32 CUDA cores,
so with `PET_GEMM=3xtf32 PET_ATTN=3xtf32` the GEMMs and
attention's products run there with each fp32 operand split into two TF32
parts, both rounded to nearest, and three of their four products summed in
fp32. That carries fp32's 24 bits, not TF32's 11. Measured against the fp64
build on pet-mad-xs, pet-mad-s, pet-omat-s and pet-mols-s, 648- and 5184-atom
water and 1728-atom diamond:

| against fp64 | energy | largest force error | RMS force error |
|---|---|---|---|
| fp32 on the CUDA cores (the default) | 1e-5-2.2e-4 meV/atom | 3.4-9.6e-3 meV/Å | 0.8-2.2e-3 meV/Å |
| `PET_GEMM=3xtf32 PET_ATTN=3xtf32` | 5e-5-2.2e-3 meV/atom | 3.6-11e-3 meV/Å | 1.0-2.5e-3 meV/Å |
| `PET_GEMM=tf32 PET_ATTN=tf32` | 0.02-0.31 meV/atom | 1.4-9.3 meV/Å | 0.3-2.1 meV/Å |

The forces are as accurate as CUDA-core fp32's; the energy's error, though up
to 15x fp32's, is under 2.2e-6 eV per atom, about 3e-7 of the energy per atom:
an fp32 ulp or two. Results repeat to the bit, do not depend on the batch or
on which tensor-core tile the GEMMs time fastest, but differ from the CUDA-core
path's at rounding level. On an H200 it is 23-25% faster than the CUDA cores
(the table below). (CUTLASS's own 3xTF32 truncates the big part, which biases
the energy by up to 5e-3 meV/atom; rounding both parts to nearest removes most
of that.)
It is opt-in, not the default, for that energy error. On RTX and workstation
GPUs TF32 runs at about the fp32 rate, so `3xtf32` runs there (from compute
capability 8.0) but is not faster.

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

What they buy, ms per evaluation with forces. On an H200 (best of 20):

| | fp32, CUDA cores (default) | `3xtf32` | `PET_GEMM=tf32 PET_ATTN=tf32` |
|---|---|---|---|
| pet-mad-s, 5184-atom water | 87.9 | 65.7 | 30.7 |
| pet-omat-s, 5184-atom water | 60.9 | 46.0 | 21.8 |
| pet-mad-xs, 5184-atom water | 17.7 | 13.7 | 9.0 |

and, measured earlier (2026-10, CUDA-core fp32 as the default then):

| | fp32 | `PET_GEMM=tf32` | + `PET_ATTN=tf32` |
|---|---|---|---|
| H200, pet-mols-s, 5184-atom water | 98.5 | 51.0 | 39.0 |
| H200, pet-omat-l, 2592-atom water | 439 | 138 | 116 |
| RTX 4080 SUPER, pet-mad-s, 648-atom water | 24.5 | 17.3 | 16.7 |
| RTX 4080 SUPER, pet-mols-s, 1728-atom diamond | 98.9 | 85.6 | 71.9 |

An H200's fp32 runs on its CUDA cores, several times slower than its TF32
tensor cores, so it gains the most; an RTX 4080's TF32 and fp32 peaks are about
equal, and what it gains is from better-used kernels (and there the default
stays on the CUDA cores).

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

## D3 dispersion

A model trained on a DFT functional without dispersion misses the van der
Waals attraction that functional misses. pet-kokkos can add Grimme's DFT-D3
correction to every result (energy, per-atom energy, forces, virial), computed
on the device alongside the model:

```bash
pet-eval pet-mad-s water.xyz -f --d3 pbesol             # rational (BJ) damping
pet-eval pet-mad-s water.xyz -f --d3 pbesol --d3-atm    # with the three-body term
pet-eval none water.xyz -f --d3 pbesol                  # D3 alone
```

```cpp
pet::Options opts;
opts.d3 = pet::D3Settings::functional("pbesol");        // or pet::D3 on its own
pet::Calculator calc("pet-mad-s", opts);
```

```python
calc = pet_kokkos.Calculator("pet-mad-s", d3="pbesol")
d3 = pet_kokkos.D3("pbesol", atm=True)                  # alone: d3.compute(numbers, positions, cell, pbc)
```

In LAMMPS: `pair_style pet pet-mad-s d3 pbesol` (see [lammps/](lammps/README.md)).

**Which functional** is a question about the model's training data, and
pet-kokkos leaves it to you: D3 is off unless asked for. Use the functional the
reference energies were computed with: PET-MAD was trained on PBEsol, so
`pbesol`; a model trained on PBE data, `pbe`. A model trained on a functional
that already includes dispersion must not get D3 on top: the OMol models'
ωB97M-V has VV10, a nonlocal dispersion term.

**Damping**: rational (Becke-Johnson, `bj`, the default) or the original zero
damping (`zero`), with s-dftd3's published parameters for about 160 functionals
(`pet-eval --d3-list`, `pet_kokkos.d3_functionals()`), or your own
(`--d3-params s6=..,s8=..,a1=..,a2=..`, `pet::D3Params`; a2 in bohr as
published on the command line and in Python). The Axilrod-Teller-Muto
three-body term is off unless asked for (`--d3-atm`: s9 = 1). The reference C6
table and parameters are tad-dftd3's (Apache-2.0; see [NOTICE](NOTICE)).

**Accuracy against s-dftd3.** At s-dftd3's cutoffs (`--d3-reference`: pairs to
95 bohr, CN and three-body to 40 bohr), energies agree with s-dftd3 1.7.0 to
1e-10 eV, forces to 1e-10 eV/Å and virials to 2e-9 eV, for both dampings with
and without the three-body term, on molecules, small periodic cells (one atom of
Cu or W, which takes tens of thousands of images) and a slab
(`tests/test_d3.cpp`, references by `tools/make_d3_golden.py`).

**Cutoffs.** Those cutoffs are slow, so by default the pair sum stops at 12 Å,
the coordination numbers at 8 Å and the three-body sum at 8 Å, each through a
smooth taper 2 Å wide (C3: energy and forces stay continuous in MD). The pair
energy falls off as 1/r^6, but its sum beyond a cutoff only as 1/r^3, and at
12 Å what is left is too much: 0.5 meV/atom in water and 15-28 meV/atom in a
transition metal, with a matching error in the pressure. So for a 3D-periodic
structure the default adds the sum beyond the taper back as a uniform continuum
(g(r) = 1), the tail correction MD codes use for Lennard-Jones: its energy, its
pressure, and its forces through the coordination numbers. It costs O(N) (the
C6 sum factorises over each element's reference weights). Errors against a
converged sum (pairs to 106 Å, PBE, BJ), per atom, with the forces' largest
difference and the pressure's:

| | E(D3) | default (12 Å, tail) | no tail | 15 Å, tail | s-dftd3's cutoffs |
|---|---|---|---|---|---|
| water, 648 atoms | −33.4 meV | −0.008 meV, 0.02 meV/Å, 0.0002 GPa | +0.47 meV, 0.008 GPa | −0.007 meV | +0.005 meV |
| CO2 crystal, 96 | −69.5 meV | −0.03 meV, 0.03 meV/Å, 0.001 GPa | +1.1 meV, 0.011 GPa | −0.009 meV | +0.012 meV |
| diamond, 64 | −191 meV | −0.01 meV, 0.02 meV/Å, 0.003 GPa | +3.1 meV, 0.085 GPa | +0.001 meV | +0.03 meV |
| Cu fcc, 108 | −583 meV | +0.12 meV, 0.18 meV/Å, 0.03 GPa | +15.4 meV, 0.17 GPa | −0.03 meV | +0.15 meV |
| W bcc, 128 | −696 meV | +0.39 meV, 0.39 meV/Å, 0.06 GPa | +28.3 meV, 0.35 GPa | +0.06 meV | +0.25 meV |

The default is as close to the converged energy as s-dftd3's own cutoffs are;
its forces are within 0.4 meV/Å. The coordination numbers' cutoff hardly
matters (6, 8 or 21 Å agree to 0.001 meV/atom). Metals keep a pressure error of
a few hundredths of a GPa from their structure past 10 Å; `--d3-cutoff 15`
halves it (at about twice the pair sum's cost). The tail needs the whole
periodic cell, so it is for `compute`, `compute_batch`, `compute_device`, an
engine's periodic list (shifts or images: LAMMPS mode images), and not for
molecules, slabs or an engine's ghost atoms (`D3Cutoffs::tail`,
`--d3-no-tail`, `d3_tail no`).

The three-body term converges more slowly, and costs much more per cutoff (its
triangles grow as r^6). Against 21 Å, with the 2 Å taper:

| three-body cutoff | water | CO2 | diamond | Cu |
|---|---|---|---|---|
| E(ATM) at 21 Å | +1.60 meV | +2.96 meV | +14.9 meV | +70.3 meV |
| 8 Å (default) | −0.18 meV | −0.53 meV | −4.2 meV | −35 meV |
| 10 Å | −0.06 meV | −0.15 meV | −1.4 meV | −11 meV |
| 12 Å | −0.03 meV | −0.07 meV | −0.6 meV | −4.5 meV |
| 16 Å | −0.006 meV | −0.016 meV | −0.15 meV | −1.1 meV |

**Cost.** On an RTX 4080 SUPER (whose double precision is 1/64 of its single),
the 5184-atom water box (`water_648` replicated 2x2x2), forces included:

| | pet-eval, one evaluation | LAMMPS `pet/kk`, per MD step |
|---|---|---|
| pet-mad-xs | 21.7 ms | 20.4 ms |
| pet-mad-xs + D3 | 25.8 ms | 24.4 ms |
| pet-mad-xs + D3, `mixed_precision` | 24.5 ms | 23.1 ms |
| pet-mad-s | 143.7 ms | 154.2 ms |
| pet-mad-s + D3 | 148.4 ms | 158.6 ms |
| D3 alone (default cutoffs) | 5.6 ms | |
| D3 alone, + three-body (8 Å) | 112 ms (44 ms mixed) | |
| D3 alone, s-dftd3's cutoffs | 157 ms | |
| pet-mad-xs + D3, s-dftd3's cutoffs | 180 ms | |

So D3 at the default cutoffs adds about a fifth to a pet-mad-xs step and 3% to
a pet-mad-s one on this card; on GPUs with full-rate double precision (A100,
H100/H200, MI250X) its share is smaller. The three-body term is the expensive
part: each atom sums its ~(n^2)/2 neighbour pairs, so it is opt-in and keeps
a short cutoff. `mixed_precision` (`--d3-mixed`, `d3_mixed yes`) does each
pair's and triangle's arithmetic in float and keeps the sums in double: within
~1e-7 relative of the default, and 2.5x faster for the three-body term; off by
default.

**How it runs.** No pair list is stored. Each structure is binned once per
call (the PET search's counting sort, `sort_into_bins`, at a bin width chosen
for D3's radii), positions are gathered into bin order, and three passes walk
the bins on the fly, a warp per atom: the coordination numbers; the pair
energy, forces, virial and dE/dCN; and the coordination numbers' chain rule.
Each atom sums its own full neighbourhood in a fixed order, the hits compacted
so every lane works, and the warp combines its lanes in a fixed tree: no
atomics, so results are the same to the bit run to run and whichever batch a
structure is in. An engine's atoms are binned each step into a box kept between
rebuilds (a graph replays the step). The three-body term gathers each atom's
neighbours into a list in scratch memory and shares out its pairs.

**Engines.** D3 searches the atoms itself, so an engine's list only tells it
how to see them (`EdgeListView::periodic` and friends): a periodic list (shifts
or images) is the owned atoms in their cell; owned atoms and ghosts without an
exchange need ghosts out to the pair plus CN cutoff (`ghost_cutoff()` includes
it); with an Exchange, ghosts only to the pair cutoff
(`exchange_ghost_cutoff()`), their coordination numbers forwarded from their
owners and their dE/dCN sent home. Only the owned atoms' energy counts; every
atom gets the force of it. D3 is in the forces and virial `compute_edges`
returns but not in its `edge_gradient`, which is the model's (its per-atom
virial share is `Results::atom_virial`). `response()` takes D3's part by
central differences of its forces (to ~1e-8; D3 is double and smooth).

## Environment variables

- `PET_MODEL_DIR`: extra directories to search for models.
- `PET_GEMM`: `native` (fp32 on the CUDA cores or CPU), `3xtf32` (as
  accurate on the forces, on tensor cores; faster on A100, H100/H200 and B200),
  `tf32`, `fp16`, `bf16`, `fp8` or `ozaki`; see [Faster modes](#faster-modes).
- `PET_ATTN`: `native` (the default), `3xtf32` or
  `tf32`; see [Faster modes](#faster-modes).
- `PET_TF32_TILE=0..7`: pin the CUTLASS tensor-core tile (`tf32` on Ampere and
  Ada, `3xtf32`) instead of timing them; for tests and profiling.
- `PET_CUTLASS=0`: use plain cuBLAS instead of the CUTLASS kernels.
- `PET_GRAPHS=0`: don't use CUDA graphs (useful when profiling). On HIP graphs
  are off unless `PET_GRAPHS=1`: under ROCm 7.0.1 a replayed graph gave wrong
  forces in a stepping loop, and they never made an MI250X evaluation faster.
- `PET_MFMA_GEMM=0`: on AMD CDNA GPUs, use rocBLAS/hipBLASLt instead of the
  built-in fp32 matrix-core GEMMs and their fused epilogues.
- `PET_FAST_SIGMOID=0`: on AMD GPUs, use the accurate sigmoid in SiLU and
  SwiGLU instead of the hardware exp and reciprocal (the default there; no
  effect elsewhere); see [Faster modes](#faster-modes).
- `PET_RECOMPUTE`: `never`, `wide` or `layers` forces how much of the forward
  the backward recomputes; by default the cheapest tier whose estimate fits the
  device (less a reserve of 10% or 1.5 GiB) is taken, and a pass that runs out
  of memory anyway is rerun a tier lower. `pet-eval --memory` shows the choice.
  Under `wide` and `layers` the edge MLP (and, on AMD CDNA GPUs, the
  attention) runs over chunks of rows, an eighth of the edges and at least 64k
  rows, so its widest temporaries are a chunk's; the answer is the same to the
  bit. `PET_EDGE_CHUNK=rows` sets the chunk, for tests.
- `PET_DEVICE_SEARCH`: `auto`, `cells` or `brute`, for the GPU neighbour search.
- `PET_NEIGHBORS=builtin`: use the built-in host neighbour search instead of
  vesin.
- `PET_D3_PROFILE=1`: print each D3 pass's time (fenced). `PET_D3_BIN=width`
  sets D3's bin width in Å (default half its longest cutoff);
  `PET_D3_ATM_CAP=n` caps the three-body neighbour list, for tests.

## Testing

```bash
ctest --preset serial          # or openmp / cuda / hip
```

Most tests need models and are skipped without them;
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) shows how to make the
ones the reference values in `tests/golden/` were computed with.
`uv run tools/test_all_models.py` checks against every published model.
