// Bit-exact reproducibility, run to run.
//
// This is not a nicety. A relaxation is a chaotic map: last-bit force noise
// grows into multi-kJ/mol differences in relaxed energies and reshuffled
// rankings. Getting here meant removing float atomics from every reduction the
// energy depends on -- neighbour slots assigned by a per-atom walk in edge-list
// order rather than by atomic_fetch_add in thread-arrival order (by far the
// largest source); per-edge force gradients GATHERED per atom via the
// reverse-edge map instead of scattered with two atomics per edge;
// per-structure energy and virial as ordered segmented sums; the attention
// backward's cutoff adjoint summed over heads in index order in a single
// thread.
//
// It needs no golden: it compares the model against itself. Any change that
// reintroduces an order-dependent reduction fails here and nowhere else, which
// is exactly why it has to stay in CI -- and why it must run with TF32 off,
// since a tensor-core GEMM is its own source of run-to-run variation.
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// Every evaluation of the same geometry must agree to the BIT, not to a
// tolerance. A tolerance here would pass the very bug this test exists to catch.
void require_bit_identical(const pet::Results& a, const pet::Results& b, const char* what) {
  INFO(what);
  REQUIRE(a.energy.size() == b.energy.size());
  for (std::size_t i = 0; i < a.energy.size(); ++i) REQUIRE(a.energy[i] == b.energy[i]);
  REQUIRE(a.forces.size() == b.forces.size());
  for (std::size_t i = 0; i < a.forces.size(); ++i) REQUIRE(a.forces[i] == b.forces[i]);
  REQUIRE(a.virial.size() == b.virial.size());
  for (std::size_t i = 0; i < a.virial.size(); ++i) REQUIRE(a.virial[i] == b.virial[i]);
  REQUIRE(a.per_atom_energy.size() == b.per_atom_energy.size());
  for (std::size_t i = 0; i < a.per_atom_energy.size(); ++i)
    REQUIRE(a.per_atom_energy[i] == b.per_atom_energy[i]);
}

constexpr int kRepeats = 4;

}  // namespace

TEST_CASE("repeated evaluation is bit-identical", "[model][determinism]") {
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) {
      WARN("model '" << model << "' is not installed; skipping its determinism check");
      continue;
    }
    pet::Calculator calc(found->first, found->second);
    REQUIRE_FALSE(calc.options().allow_tf32);  // see the header comment

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results first = calc.compute(g.system, true);
        for (int k = 1; k < kRepeats; ++k)
          require_bit_identical(first, calc.compute(g.system, true), "repeat evaluation");
      }
    }
  }
}

TEST_CASE("a batch gives each structure the same answer as evaluating it alone",
          "[model][determinism]") {
  // PET's network is per-atom and per-edge, and edges never cross a structure
  // boundary, so concatenating structures into one NEF must not change any
  // per-atom energy or force. Padding slots carry cutoff_factor = 0, so the
  // attention softmax skips them and the real neighbour slots keep their order.
  //
  // The batch runs on the HOST neighbour path here, because the single-structure
  // entry point always does: comparing a device-built batch against a host-built
  // single would measure the two builders against each other rather than the
  // concatenation, and that comparison has its own test
  // (test_device_vs_host.cpp).
  //
  // Not bit equality. Concatenation pads every row to the batch-wide M and the
  // energy reduction becomes an ordered segmented sum over each structure's
  // atom range instead of a whole-array sum, so the fp32 network sees a
  // different summation order. The bound is relative to the largest force in
  // the structure: an absolute one says nothing without knowing that scale.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Options host_nef;
    host_nef.device_neighbors = false;
    pet::Calculator calc(found->first, found->second, host_nef);

    std::vector<Golden> goldens;
    for (const auto& path : golden_paths(model)) {
      Golden g = load_golden(path);
      if (g.periodic) goldens.push_back(std::move(g));  // batch path assumes cells
    }
    if (goldens.size() < 2) continue;

    DYNAMIC_SECTION(model << " / batch of " << goldens.size()) {
      std::vector<pet::System> systems;
      for (const auto& g : goldens) systems.push_back(g.system);
      const pet::Results batched = calc.compute(systems, true);

      REQUIRE(batched.energy.size() == goldens.size());
      std::size_t foff = 0;
      for (std::size_t b = 0; b < goldens.size(); ++b) {
        const pet::Results alone = calc.compute(goldens[b].system, true);
        INFO("structure " << b << " (" << goldens[b].name << ")");

        const double de = std::abs(batched.energy[b] - alone.energy[0]);
        INFO("|dE| = " << de << " eV on E = " << alone.energy[0]);
        CHECK(de <= 1e-5 * std::max(1.0, std::abs(alone.energy[0])));

        double max_f = 0.0, max_df = 0.0;
        for (std::size_t i = 0; i < alone.forces.size(); ++i) {
          max_f = std::max(max_f, std::abs(alone.forces[i]));
          max_df = std::max(max_df, std::abs(batched.forces[foff + i] - alone.forces[i]));
        }
        INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
        CHECK(max_df <= 1e-5 * std::max(max_f, 1.0));
        foff += alone.forces.size();
      }
    }
  }
}
