# pet-kokkos — pure C++/Kokkos evaluation of PET (Point Edge Transformer)
# machine-learning interatomic potentials: energy, conservative forces and the
# virial, with no libtorch at run time.
#
# Unlike the klasp subdirectory this grew out of, this project OWNS its
# dependencies: Kokkos, nlohmann/json and the warning set are fetched and
# configured here.
cmake_minimum_required(VERSION 3.18)

project(pet_kokkos
    VERSION 0.1.0
    DESCRIPTION "Kokkos evaluation of PET machine-learning interatomic potentials"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Whether we are the top-level project or a consumer's add_subdirectory/CPM
# dependency. Tests, apps and installation default OFF for a consumer, which is
# how klasp will take this once the extraction is finished.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PET_IS_TOP_LEVEL ON)
else()
    set(PET_IS_TOP_LEVEL OFF)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- options -----------------------------------------------------------------

option(PET_BUILD_TESTS   "Build the test suite"            ${PET_IS_TOP_LEVEL})
option(PET_BUILD_APPS    "Build the command-line tools"    ${PET_IS_TOP_LEVEL})
option(PET_BUILD_PYTHON  "Build the nanobind Python module" OFF)
option(PET_INSTALL       "Generate install/export targets" ${PET_IS_TOP_LEVEL})
option(PET_BUILD_METATOMIC_PLUGIN
       "Build the metatomic backend plugin (prototype; needs metatensor + metatomic)" OFF)

# Optional vesin backend for the host neighbour search. The built-in search is an
# O(N^2 x images) brute force, which is fine for small cells and dominates
# everything else for large ones -- 95% of the runtime on a 1728-atom supercell.
# vesin's cell list does the same work in O(N).
#
# AUTO uses it when it can be found (the `vesin` Python wheel ships the header
# and the library, so a venv counts) and silently does without otherwise; ON
# fails the configure if it is missing; OFF never looks.
set(PET_WITH_VESIN "AUTO" CACHE STRING "Use vesin for neighbour lists: AUTO | ON | OFF")
set_property(CACHE PET_WITH_VESIN PROPERTY STRINGS AUTO ON OFF)
set(PET_WITH_CUTLASS "AUTO" CACHE STRING "Fused GEMM kernels on CUTLASS (CUDA only): AUTO | ON | OFF")
set_property(CACHE PET_WITH_CUTLASS PROPERTY STRINGS AUTO ON OFF)

# Tune for the building machine's CPU. A real win on the host kernels, but it
# bakes THIS machine's instruction set into the objects: turn it off when the
# build host and the run host may differ, or when objects are cached and reused
# elsewhere (ccache hashes the literal "-march=native", not its expansion, and
# will happily reuse an object on a CPU that cannot run it). Never for CUDA --
# nvcc does not accept the flag.
option(PET_ARCH_NATIVE "Optimize for the building machine's CPU (-march=native)" ON)

set(PET_BACKEND "serial" CACHE STRING "Kokkos backend: serial, openmp, cuda, hip")
set_property(CACHE PET_BACKEND PROPERTY STRINGS serial openmp cuda hip)

# Precision mode. PUBLIC on the library target so a consumer sees the same
# Net/Real types the library was compiled with.
#   mixed (default) : Net=float,  Real=double -- fp32 network, fp64 geometry,
#                     energy, force and virial accumulation. What the goldens
#                     are validated against.
#   fp32            : everything single. Fastest on consumer GPUs.
#   fp64            : everything double. Slow on consumer GPUs (1/64 rate on
#                     GeForce); a correctness instrument, not a speed mode.
set(PET_PRECISION "mixed" CACHE STRING "PET precision: mixed | fp32 | fp64")
set_property(CACHE PET_PRECISION PROPERTY STRINGS mixed fp32 fp64)

if(PET_ARCH_NATIVE AND NOT PET_BACKEND STREQUAL "cuda")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native PET_HAS_MARCH_NATIVE)
    if(PET_HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# --- CPM ---------------------------------------------------------------------

set(CPM_DOWNLOAD_VERSION 0.40.8)
if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
endif()
include(${CMAKE_BINARY_DIR}/cmake/CPM.cmake)

# --- Kokkos ------------------------------------------------------------------
#
# CPM's OPTIONS are not promoted to cache variables, so anything that later does
# find_package(Kokkos) or reads Kokkos_ENABLE_* cannot see them. Set every flag
# as a FORCE'd cache var up front so Kokkos and any downstream consumer agree.
#
# Only fetch when the consumer has not already provided Kokkos: klasp (and any
# other Kokkos application) owns one already, and two Kokkos configurations in
# one build tree do not end well.
if(NOT TARGET Kokkos::kokkos)
    set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_HIP    OFF CACHE BOOL "" FORCE)

    if(PET_BACKEND STREQUAL "openmp")
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    elseif(PET_BACKEND STREQUAL "cuda")
        set(Kokkos_ENABLE_CUDA           ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_LAMBDA    ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_OPENMP         ON CACHE BOOL "" FORCE)
    elseif(PET_BACKEND STREQUAL "hip")
        set(Kokkos_ENABLE_HIP    ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    elseif(NOT PET_BACKEND STREQUAL "serial")
        message(FATAL_ERROR "PET_BACKEND must be one of: serial, openmp, cuda, hip")
    endif()

    CPMAddPackage(
        NAME Kokkos
        GITHUB_REPOSITORY kokkos/kokkos
        GIT_TAG 5.2.2
    )
else()
    message(STATUS "pet-kokkos: using the consumer's existing Kokkos::kokkos")
endif()

# --- nlohmann/json -----------------------------------------------------------

if(NOT TARGET nlohmann_json::nlohmann_json)
    CPMAddPackage(
        NAME nlohmann_json
        VERSION 3.11.3
        GITHUB_REPOSITORY nlohmann/json
        OPTIONS "JSON_BuildTests OFF"
    )
endif()

# --- CUTLASS (optional, CUDA) -----------------------------------------------
# Header-only: its fp32 GEMM main loop carries the fused kernels in
# src/fused_gemm.cpp. Without it those operations run unfused.

set(PET_HAVE_CUTLASS OFF)
if(PET_BACKEND STREQUAL "cuda" AND NOT PET_WITH_CUTLASS STREQUAL "OFF")
    CPMAddPackage(
        NAME cutlass
        GITHUB_REPOSITORY NVIDIA/cutlass
        GIT_TAG v4.8.0
        DOWNLOAD_ONLY YES
    )
    if(cutlass_ADDED)
        set(PET_HAVE_CUTLASS ON)
        message(STATUS "pet-kokkos: fused GEMMs on CUTLASS from ${cutlass_SOURCE_DIR}")
    elseif(PET_WITH_CUTLASS STREQUAL "ON")
        message(FATAL_ERROR "pet-kokkos: PET_WITH_CUTLASS=ON but CUTLASS could not be fetched")
    endif()
endif()

# --- vesin (optional) --------------------------------------------------------

set(PET_HAVE_VESIN OFF)
if(NOT PET_WITH_VESIN STREQUAL "OFF")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    if(PET_WITH_VESIN STREQUAL "ON")
        find_package(Vesin REQUIRED)
    else()
        find_package(Vesin QUIET)
    endif()
    if(Vesin_FOUND)
        set(PET_HAVE_VESIN ON)
        message(STATUS "pet-kokkos: vesin neighbour lists from ${Vesin_LIBRARY}")
    elseif(PET_WITH_VESIN STREQUAL "AUTO")
        message(STATUS "pet-kokkos: vesin not found; using the built-in neighbour search")
    endif()
endif()

# --- warnings ----------------------------------------------------------------
#
# Applied to our own targets only, never to Kokkos. Nothing here is an error:
# -Werror in a project whose kernels are compiled by three different vendor
# toolchains breaks a build over a warning one of them invented.
#
# sign-compare and unused-parameter are off because Kokkos' own idioms trip them
# in our code.
add_library(pet_warnings INTERFACE)
if(MSVC)
    target_compile_options(pet_warnings INTERFACE /W3)
else()
    target_compile_options(pet_warnings INTERFACE
        -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
endif()

# --- the library -------------------------------------------------------------

add_subdirectory(src)

if(PET_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(PET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PET_BUILD_METATOMIC_PLUGIN)
    add_subdirectory(plugins/metatomic)
endif()

# --- summary -----------------------------------------------------------------

message(STATUS "pet-kokkos ${PROJECT_VERSION}")
message(STATUS "  backend    : ${PET_BACKEND}")
message(STATUS "  precision  : ${PET_PRECISION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  tests      : ${PET_BUILD_TESTS}   apps: ${PET_BUILD_APPS}")
message(STATUS "  vesin      : ${PET_HAVE_VESIN}")
