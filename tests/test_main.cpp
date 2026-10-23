// Catch2's main, inside Kokkos::initialize/finalize.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

#include "pet/gemm.hpp"

#include <cstdlib>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  // The tests check the implementation against fp32 references, so they run the
  // GEMMs in fp32 (the default anyway); test_gemm_modes measures the other
  // modes. PET_GEMM still overrides.
  if (!std::getenv("PET_GEMM")) pet::gemm_mode() = pet::GemmMode::Native;
  int rc = 0;
  {
    rc = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return rc;
}
