// Catch2's main, inside Kokkos::initialize/finalize.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

#include "pet/gemm.hpp"

#include <cstdlib>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  // The tests check the implementation against fp32 references, in the default
  // modes (Native); test_gemm_modes and
  // the determinism tests measure the others. PET_GEMM and PET_ATTN override.
  int rc = 0;
  {
    rc = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return rc;
}
