// Catch2's main, inside Kokkos::initialize/finalize.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    rc = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return rc;
}
