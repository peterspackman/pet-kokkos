# pair_style pet for LAMMPS

PET machine-learning potentials (the upet / PET-MAD family) in LAMMPS through
pet-kokkos: package `ML-PET`, with `pet/kk` for the KOKKOS package.

## Install into a LAMMPS tree

```sh
lammps/patch_lammps.sh /path/to/lammps          # copy the sources in
lammps/patch_lammps.sh /path/to/lammps --link   # or symlink them, to work on them
```

This puts `src/ML-PET`, `src/KOKKOS/pair_pet_kokkos.*` and
`cmake/Modules/Packages/ML-PET.cmake` in place, records where pet-kokkos is, and
registers the package in `cmake/CMakeLists.txt` (running it again is harmless).
CMake builds only.

## Configure

On a GPU, with LAMMPS's KOKKOS package (pet-kokkos builds on LAMMPS's Kokkos,
for its device; `pet/kk` keeps every step on the GPU):

```sh
cmake -S cmake -B build -C cmake/presets/kokkos-cuda.cmake \
      -D Kokkos_ARCH_ADA89=ON -D PKG_ML-PET=ON -D BUILD_MPI=ON
```

Without the KOKKOS package (pet-kokkos fetches its own Kokkos 5; LAMMPS's
arrays stay on the host and cross each step):

```sh
cmake -S cmake -B build -D PKG_ML-PET=ON -D PET_BACKEND=cuda -D Kokkos_ARCH_ADA89=ON \
      -D CMAKE_CXX_STANDARD=20 -D CMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper
```

| variable | meaning |
|---|---|
| `PKG_ML-PET` | build the package |
| `PET_KOKKOS_DIR` | the pet-kokkos source tree; defaults to the one that ran `patch_lammps.sh` |
| `PET_BACKEND` | without `PKG_KOKKOS`: `cuda`, `hip`, `openmp` or `serial`. With it, taken from LAMMPS's Kokkos |
| `PET_PRECISION` | `mixed` (default: fp32 network, fp64 geometry), `fp32`, `fp64` |
| `PET_WITH_CUTLASS` | fused GEMM kernels on CUDA: `AUTO` (default), `ON`, `OFF` |
| `CPM_SOURCE_CACHE` | where pet-kokkos's downloads (Kokkos, CUTLASS, json) are kept |

## Run

```
units       metal
pair_style  pet pet-mad-s            # or a path prefix to MODEL.json / MODEL.safetensors
pair_coeff  * * C H O                # an element per atom type
```

```sh
lmp -in in.lmp                                  # pair_style pet
lmp -k on g 1 -sf kk -in in.lmp                 # pair_style pet/kk, on the GPU
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in in.lmp
```

(pet/kk asks for its own full list; `neigh half` only satisfies the KOKKOS
package's rule that full lists go with `newton off`.)

Models are found on `PET_MODEL_DIR` (and `./models`, `~/.local/share/pet/models`).

`pair_style pet MODEL mode images|exchange|ghosts`:
- `images` (the default on one MPI rank): every ghost is a periodic image of an
  owned atom, so PET evaluates only the owned atoms. Exact, and the cheapest.
- `exchange` (the default on several): each rank evaluates its owned atoms with
  ghosts one cutoff deep, and at every message-passing layer the ranks swap the
  rows of the edges that cross between them (and, in the backward, their
  adjoints): ghost cutoffs by LAMMPS's comm, edge rows straight to the owning
  rank by MPI_Alltoallv, only for edges PET keeps. Needs `newton on`.
- `ghosts`: no exchange; ghosts as deep as the message passing reaches (several
  cutoffs), all evaluated. Kept for comparison.

In `exchange` mode the edge rows cross straight from device memory when MPI is
GPU-aware (the log says so), else through pinned host buffers. `pair_style pet
MODEL gpu_aware auto|yes|no` overrides the check (`auto` asks
`MPIX_Query_cuda_support`/`_rocm_support`, or on Cray MPICH reads
`MPICH_GPU_SUPPORT_ENABLED`); for `pet/kk`, LAMMPS's `-pk kokkos gpu/aware`
decides, and its atom comm follows `-pk kokkos comm`. Open MPI 5 built against an
external PMIx may point `mca_base_component_path` at PMIx's plugins and never load
its own CUDA ones (`MPIX_Query_cuda_support` then says 0); pass
`--mca mca_base_component_path $OMPI/lib/openmpi:$PMIX_PLUGINS`.

`pair_style pet` needs `newton on` (its half list); `pet/kk` in `images` mode
does not. Per-atom energy and virial (`compute pe/atom`, `stress/atom`) are
supported; the global virial is PET's symmetric one (PET is not exactly
rotation invariant, so LAMMPS's f.r would keep one side of an antisymmetric
part). Runtime switches: `PET_CUTLASS=0`, `PET_GRAPHS=0`.

## Check it

`examples/` has structures as LAMMPS data files and as extxyz, and `compare.py`
checks LAMMPS's energy, per-atom energy, forces and pressure against `pet-eval`
on the same structure:

```sh
cd examples
lmp -in in.pet -var data diamond_216 -var model pet-mad-s -var elements C -var nsteps 0
python3 compare.py diamond_216 /path/to/pet-eval pet-mad-s log.lammps
```
