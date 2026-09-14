# Test suite.
#
# Model weights are not in this repository, so the tests that need one SKIP
# rather than fail when it is absent -- a fresh clone must still produce a green
# run. Put a <name>.json / <name>.safetensors pair anywhere on the model search
# path (PET_MODEL_DIR, ./models, ~/.local/share/pet/models, this tree's models/)
# and they start running.

CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION 3.5.2
)

add_executable(pet_tests
    test_main.cpp
    test_neighbors.cpp
    test_io.cpp
    test_golden.cpp
    test_fd.cpp
    test_determinism.cpp
    test_device_vs_host.cpp
)

target_link_libraries(pet_tests PRIVATE
    pet_kokkos
    pet_warnings
    Catch2::Catch2
    nlohmann_json::nlohmann_json
)

target_include_directories(pet_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(pet_tests PRIVATE
    PET_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

# One CTest entry per Catch2 tag group, so a failure names which contract broke
# rather than just "pet_tests failed".
add_test(NAME neighbors    COMMAND pet_tests "[neighbors]")
add_test(NAME io           COMMAND pet_tests "[io]")
add_test(NAME golden       COMMAND pet_tests "[golden]")
add_test(NAME finite_diff  COMMAND pet_tests "[fd]")
add_test(NAME determinism  COMMAND pet_tests "[determinism]")
add_test(NAME paths_agree  COMMAND pet_tests "[paths]")

# The finite-difference test evaluates the model a few hundred times.
set_tests_properties(finite_diff PROPERTIES TIMEOUT 900)
set_tests_properties(golden determinism paths_agree PROPERTIES TIMEOUT 600)
