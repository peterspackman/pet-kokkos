# ML-PET: pair_style pet (and pet/kk) on pet-kokkos. Put in place by
# pet-kokkos/lammps/patch_lammps.sh.
#
#   PET_KOKKOS_DIR  the pet-kokkos source tree (default: the one that patched this tree)
#
# With PKG_KOKKOS, pet-kokkos builds on LAMMPS's Kokkos, for its device, and
# pair_style pet/kk keeps each step on it; C++17 is enough. Without, pet-kokkos
# fetches its own Kokkos 5: set PET_BACKEND (cuda, hip, openmp, serial), C++20,
# and for CUDA nvcc_wrapper as the compiler. Other pet-kokkos options
# (PET_PRECISION, PET_WITH_CUTLASS, ...) pass through.
set(PET_KOKKOS_DIR "@PET_KOKKOS_DIR@" CACHE PATH "pet-kokkos source tree")
if(NOT EXISTS "${PET_KOKKOS_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "ML-PET: PET_KOKKOS_DIR (${PET_KOKKOS_DIR}) is not a pet-kokkos source tree")
endif()
if(NOT PKG_KOKKOS AND CMAKE_CXX_STANDARD LESS 20)
  message(FATAL_ERROR "ML-PET without PKG_KOKKOS fetches Kokkos 5: needs -D CMAKE_CXX_STANDARD=20")
endif()
set(PET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PET_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(PET_INSTALL OFF CACHE BOOL "" FORCE)

# Included after the KOKKOS package (patch_lammps.sh puts it there), so with
# PKG_KOKKOS pet-kokkos finds LAMMPS's Kokkos instead of fetching one.
add_subdirectory(${PET_KOKKOS_DIR} ${CMAKE_BINARY_DIR}/pet-kokkos)
target_link_libraries(lammps PRIVATE pet::kokkos)
