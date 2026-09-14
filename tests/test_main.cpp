// Catch2 entry point.
//
// A custom main rather than Catch2WithMain, because Kokkos has to be
// initialized before any test constructs a model and finalized after the last
// one releases its Views. The library never does this itself -- an application
// owns the Kokkos lifetime -- so the test binary is that application.
#include <catch2/catch_session.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    // Scoped so every Kokkos::View a test created is destroyed before finalize.
    rc = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return rc;
}
