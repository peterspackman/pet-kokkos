# Locate an installed vesin (https://github.com/Luthaf/vesin).
#
# vesin ships a C API (vesin.h) and a shared library. The Python wheel carries
# both, so a venv with `pip install vesin` will do.
#
# Sets: Vesin_FOUND, Vesin_INCLUDE_DIR, Vesin_LIBRARY, and the imported target
# vesin::vesin.

# Ask a Python interpreter where its vesin lives, if there is one. Explicit hints
# (VESIN_ROOT, CMAKE_PREFIX_PATH) still win -- this is only the fallback, and it
# is why a plain `cmake -B build` inside a venv picks vesin up with no arguments.
if(NOT VESIN_ROOT)
    find_program(_vesin_python NAMES python3 python)
    if(_vesin_python)
        execute_process(
            COMMAND ${_vesin_python} -c
                    "import os,vesin;print(os.path.dirname(vesin.__file__))"
            OUTPUT_VARIABLE _vesin_pkg
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_vesin_pkg)
            set(VESIN_ROOT "${_vesin_pkg}")
        endif()
    endif()
endif()

find_path(Vesin_INCLUDE_DIR
    NAMES vesin.h
    HINTS ${VESIN_ROOT} ENV VESIN_ROOT
    PATH_SUFFIXES include)

find_library(Vesin_LIBRARY
    NAMES vesin
    HINTS ${VESIN_ROOT} ENV VESIN_ROOT
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vesin
    REQUIRED_VARS Vesin_INCLUDE_DIR Vesin_LIBRARY)

if(Vesin_FOUND AND NOT TARGET vesin::vesin)
    add_library(vesin::vesin UNKNOWN IMPORTED)
    set_target_properties(vesin::vesin PROPERTIES
        IMPORTED_LOCATION "${Vesin_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Vesin_INCLUDE_DIR}")
endif()

mark_as_advanced(Vesin_INCLUDE_DIR Vesin_LIBRARY)
