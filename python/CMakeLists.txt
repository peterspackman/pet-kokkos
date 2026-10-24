# The Python package's compiled module, _core (nanobind). It carries
# pet-kokkos and Kokkos inside, their symbols kept to itself, like the metatomic
# plugin. A wheel build (pyproject.toml, scikit-build-core) installs it, and the
# plugin when that is built too, into pet_kokkos.

# A CUDA wheel finds the CUDA runtime and cuBLAS in NVIDIA's pip wheels
# (nvidia/cu12's under cublas/ and cuda_runtime/, CUDA 13's under cu13/),
# installed next to it in site-packages, rather than bundling them.
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

install(TARGETS _core LIBRARY DESTINATION pet_kokkos COMPONENT python)
if(PET_BUILD_METATOMIC)
    install(TARGETS pet_metatomic LIBRARY DESTINATION pet_kokkos COMPONENT python)
endif()
