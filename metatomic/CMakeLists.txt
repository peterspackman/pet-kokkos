# pet-kokkos as a metatomic plugin: one shared module for mta_load_plugin.
#
# It links libmetatomic (and through it metatensor) dynamically -- the host
# process's copy -- and carries pet-kokkos and Kokkos inside, their symbols kept
# to itself so a host with its own Kokkos does not see them.
#
# metatomic and metatensor are found with find_package; their Python wheels
# ship the CMake configs:
#   -DCMAKE_PREFIX_PATH="$(python -c 'import metatomic; print(metatomic.utils.cmake_prefix_path)');$(python -c 'import metatensor; print(metatensor.utils.cmake_prefix_path)')"

find_package(metatomic CONFIG REQUIRED)

add_library(pet_metatomic MODULE pet_metatomic.cpp)
target_link_libraries(pet_metatomic PRIVATE pet_kokkos metatomic pet_warnings)
target_compile_features(pet_metatomic PRIVATE cxx_std_17)

# The same file name everywhere, so engines can be pointed at it plainly.
set_target_properties(pet_metatomic PROPERTIES
    OUTPUT_NAME pet-kokkos-metatomic
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Only mta_plugin_init leaves the module; the static Kokkos and pet-kokkos
# inside stay private.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pet_metatomic PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Find libmetatomic where it was found at build time.
get_target_property(_pet_mta_location metatomic::shared IMPORTED_LOCATION)
if(_pet_mta_location)
    get_filename_component(_pet_mta_dir "${_pet_mta_location}" DIRECTORY)
    set_target_properties(pet_metatomic PROPERTIES
        BUILD_RPATH "${_pet_mta_dir}"
        INSTALL_RPATH "${_pet_mta_dir}")
endif()

if(PET_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS pet_metatomic LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
