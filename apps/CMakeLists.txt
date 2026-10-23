# Command-line tools. Off when pet-kokkos is consumed as a subproject.

add_executable(pet_eval pet_eval.cpp)
set_target_properties(pet_eval PROPERTIES OUTPUT_NAME pet-eval)
target_link_libraries(pet_eval PRIVATE pet_kokkos pet_warnings)

if(PET_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS pet_eval RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(pet_batch_probe batch_probe.cpp)
set_target_properties(pet_batch_probe PROPERTIES OUTPUT_NAME pet-batch-probe)
target_link_libraries(pet_batch_probe PRIVATE pet_kokkos pet_warnings)
