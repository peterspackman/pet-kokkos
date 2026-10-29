# The PET network itself, one static library. Each piece of the network keeps
# its forward and its analytic backward side by side in one file -- they are
# written against each other, and apart they drift.
#   ops.cpp          dense layers, norms, SwiGLU, compress.0, energy assembly
#   attention.cpp    attention
#   feedforward.cpp  the feedforward featurizer
#   residual.cpp     the residual featurizer
#   forces.cpp       edge gradients -> forces and virial, adaptive cutoff
#   model.cpp        loading, upload, packing, readout, compute()
#   checkpoint.cpp   the model JSON and safetensors weights
#   neighbors.cpp    the host neighbour list and adaptive cutoff
#   device_neighbors.cpp  the same on the device
#   device_geometry.cpp   staged batches, the device search, the Verlet cache
#   device_cell_list.cpp  the device cell list
#   vesin_neighbors.cpp   the host search through vesin
#   calculator.cpp   the public entry point
#   io/extxyz.cpp    extended XYZ
#   fused_gemm.cpp   GEMMs with their elementwise neighbours fused (CUTLASS)
#   ozaki.cpp        fp64-accurate GEMMs on integer tensor cores
#   gemm_lowp.cpp    GEMMs on FP8, BF16 and FP16 operands (tensor cores)

add_library(pet_kokkos STATIC
    model.cpp
    ops.cpp
    neighbors.cpp
    device_neighbors.cpp
    attention.cpp
    feedforward.cpp
    residual.cpp
    forces.cpp
    device_geometry.cpp
    device_cell_list.cpp
    calculator.cpp
    checkpoint.cpp
    io/extxyz.cpp
    ozaki.cpp
    gemm_lowp.cpp
    vesin_neighbors.cpp
    fused_gemm.cpp
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

set_target_properties(pet_kokkos PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(pet_kokkos PUBLIC cxx_std_17)

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

# --- CUTLASS (optional) ------------------------------------------------------
# PRIVATE: only fused_gemm.cpp sees it.
if(PET_HAVE_CUTLASS)
    target_include_directories(pet_kokkos SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include)
    target_compile_definitions(pet_kokkos PRIVATE PET_HAVE_CUTLASS)
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
# active Kokkos backend; on host to a CBLAS when there is one (PET_USE_BLAS),
# else to a portable Kokkos loop, many times slower.
if(PET_BACKEND STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pet_kokkos PUBLIC CUDA::cublas CUDA::cublasLt)
    message(STATUS "pet_kokkos: GEMM via cuBLAS")
elseif(PET_BACKEND STREQUAL "hip")
    find_package(rocblas REQUIRED)
    target_link_libraries(pet_kokkos PUBLIC roc::rocblas)
    # hipBLASLt, where ROCm has it, puts the bias in the GEMM's epilogue.
    find_package(hipblaslt QUIET)
    if(hipblaslt_FOUND)
        target_link_libraries(pet_kokkos PUBLIC roc::hipblaslt)
        target_compile_definitions(pet_kokkos PUBLIC PET_HAVE_HIPBLASLT)
        message(STATUS "pet_kokkos: bias GEMMs via hipBLASLt")
    endif()
    message(STATUS "pet_kokkos: GEMM via rocBLAS")
elseif(PET_BACKEND STREQUAL "metal")
    message(STATUS "pet_kokkos: GEMM via Kokkos::Experimental::metal_gemm (Metal Performance Shaders)")
else()
    set(_pet_cblas FALSE)
    if(PET_USE_BLAS)
        # Any BLAS that also has the C interface: OpenBLAS, MKL, BLIS, ... --
        # not the reference libblas, whose CBLAS is a separate library.
        find_package(BLAS QUIET)
        find_path(PET_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas blis)
        if(BLAS_FOUND AND PET_CBLAS_INCLUDE_DIR)
            include(CheckFunctionExists)
            set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
            check_function_exists(cblas_sgemm PET_BLAS_HAS_CBLAS)
            unset(CMAKE_REQUIRED_LIBRARIES)
            set(_pet_cblas ${PET_BLAS_HAS_CBLAS})
        endif()
    endif()
    if(_pet_cblas)
        target_link_libraries(pet_kokkos PUBLIC BLAS::BLAS)
        target_include_directories(pet_kokkos PUBLIC $<BUILD_INTERFACE:${PET_CBLAS_INCLUDE_DIR}>)
        target_compile_definitions(pet_kokkos PUBLIC PET_HAVE_CBLAS)
        message(STATUS "pet_kokkos: GEMM via CBLAS (${BLAS_LIBRARIES})")
    else()
        message(STATUS "pet_kokkos: GEMM via portable Kokkos fallback (host; no CBLAS found, or PET_USE_BLAS=OFF)")
    endif()
endif()

# In-tree model directory, the last entry in the runtime model search path.
# Weights are never committed (see README.md); this only lets a developer drop a
# <name>.json / <name>.safetensors pair into models/ and have it found.
target_compile_definitions(pet_kokkos PRIVATE
    PET_KOKKOS_MODEL_DIR="${PROJECT_SOURCE_DIR}/models")

# --- install -----------------------------------------------------------------
# The library and its headers only: no CMake package config, since the fetched
# dependencies are not installed with it. Consumers take this as a subproject.
if(PET_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS pet_kokkos
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/pet
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
