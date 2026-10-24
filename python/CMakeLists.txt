# The Python package's compiled module, _core (nanobind). It carries
# pet-kokkos and Kokkos inside, their symbols kept to itself, like the metatomic
# plugin. A wheel build (pyproject.toml, scikit-build-core) installs it, and the
# plugin when that is built too, into PET_PYTHON_PACKAGE: pet_kokkos for the
# CPU wheel, or a GPU package's own (pet_kokkos_cuda12, ...; python/gpu/).

set(PET_PYTHON_PACKAGE "pet_kokkos" CACHE STRING "Python package the module installs into")
# A CUDA package finds the CUDA runtime and cuBLAS in NVIDIA's pip wheels
# (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12), installed next to it in
# site-packages, rather than bundling them.
option(PET_PYTHON_NVIDIA_WHEELS "Find the CUDA libraries in NVIDIA's pip wheels" OFF)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE: not -Os, which nvcc (a CUDA build compiles this with nvcc_wrapper) rejects.
nanobind_add_module(_core NB_STATIC STABLE_ABI NOMINSIZE core.cpp)
target_link_libraries(_core PRIVATE pet_kokkos pet_warnings)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_core PRIVATE "LINKER:--exclude-libs,ALL")
endif()
if(PET_PYTHON_NVIDIA_WHEELS)
    set(_nv "$ORIGIN/../nvidia")
    set_target_properties(_core PROPERTIES INSTALL_RPATH
        "${_nv}/cublas/lib;${_nv}/cuda_runtime/lib;${_nv}/cu13/lib")
endif()

install(TARGETS _core LIBRARY DESTINATION ${PET_PYTHON_PACKAGE} COMPONENT python)
if(PET_BUILD_METATOMIC)
    install(TARGETS pet_metatomic LIBRARY DESTINATION ${PET_PYTHON_PACKAGE} COMPONENT python)
endif()
