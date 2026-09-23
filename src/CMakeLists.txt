# The PET network itself, one static library. Each piece of the network keeps
# its forward and its analytic backward side by side in one file -- they are
# written against each other, and apart they drift.
#   ops.cpp          dense layers, norms, SwiGLU, compress.0, energy assembly
#   attention.cpp    attention
#   feedforward.cpp  the feedforward featurizer
#   residual.cpp     the residual featurizer
#   forces.cpp       edge gradients -> forces and virial, adaptive cutoff
#   model.cpp        loading, upload, packing, readout, compute()

add_library(pet_kokkos STATIC
    model.cpp
    ops.cpp
    attention.cpp
    feedforward.cpp
    residual.cpp
    forces.cpp
    device_geometry.cpp
    device_cell_list.cpp
    calculator.cpp
    io/extxyz.cpp
    ozaki.cpp
    vesin_neighbors.cpp
)
add_library(pet::kokkos ALIAS pet_kokkos)

target_include_directories(pet_kokkos PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pet_kokkos
    PUBLIC  Kokkos::kokkos nlohmann_json::nlohmann_json
    PRIVATE pet_warnings
)

set_target_properties(pet_kokkos PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
)

# --- precision ---------------------------------------------------------------
# PUBLIC: a consumer must see the same Net/Real types the library was built with,
# since they appear in the View typedefs on the public API.
if(PET_PRECISION STREQUAL "fp32")
    target_compile_definitions(pet_kokkos PUBLIC PET_KOKKOS_FP32)
elseif(PET_PRECISION STREQUAL "fp64")
    target_compile_definitions(pet_kokkos PUBLIC PET_KOKKOS_FP64)
elseif(NOT PET_PRECISION STREQUAL "mixed")
    message(FATAL_ERROR "PET_PRECISION must be one of: mixed, fp32, fp64")
endif()

# --- vesin (optional) --------------------------------------------------------
# PUBLIC: the switch appears in a public header's preprocessor conditionals, so a
# consumer compiling against these headers has to see the same definition.
if(PET_HAVE_VESIN)
    target_link_libraries(pet_kokkos PUBLIC vesin::vesin)
    target_compile_definitions(pet_kokkos PUBLIC PET_HAVE_VESIN)
endif()

# --- dense GEMM --------------------------------------------------------------
# The only BLAS PET needs. gemm.hpp routes it to the vendor library of the
# active Kokkos backend; on host it uses a portable Kokkos fallback.
if(PET_BACKEND STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pet_kokkos PUBLIC CUDA::cublas CUDA::cublasLt)
    message(STATUS "pet_kokkos: GEMM via cuBLAS")
elseif(PET_BACKEND STREQUAL "hip")
    find_package(rocblas REQUIRED)
    target_link_libraries(pet_kokkos PUBLIC roc::rocblas)
    message(STATUS "pet_kokkos: GEMM via rocBLAS")
else()
    message(STATUS "pet_kokkos: GEMM via portable Kokkos fallback (host)")
endif()

# In-tree model directory, the last entry in the runtime model search path.
# Weights are never committed (see README.md); this only lets a developer drop a
# <name>.json / <name>.safetensors pair into models/ and have it found.
target_compile_definitions(pet_kokkos PUBLIC
    PET_KOKKOS_MODEL_DIR="${PROJECT_SOURCE_DIR}/models")

# --- install -----------------------------------------------------------------
if(PET_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS pet_kokkos pet_warnings
            EXPORT pet-kokkos-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/pet
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
