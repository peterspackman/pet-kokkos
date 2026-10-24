# pet_kokkos._core: the Python package's compiled module (nanobind). It carries
# pet-kokkos and Kokkos inside, their symbols kept to itself, like the metatomic
# plugin. A wheel build (pyproject.toml, scikit-build-core) installs it, and the
# plugin when that is built too, into the package directory.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC STABLE_ABI core.cpp)
target_link_libraries(_core PRIVATE pet_kokkos pet_warnings)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_core PRIVATE "LINKER:--exclude-libs,ALL")
endif()

install(TARGETS _core LIBRARY DESTINATION pet_kokkos COMPONENT python)
if(PET_BUILD_METATOMIC)
    install(TARGETS pet_metatomic LIBRARY DESTINATION pet_kokkos COMPONENT python)
endif()
