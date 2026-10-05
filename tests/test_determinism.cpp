// Bit-exact reproducibility, run to run. A relaxation is chaotic: last-bit
// force noise becomes different relaxed structures. So no float atomics feed the
// energy: neighbour slots follow the edge list, force gradients are gathered per
// atom, and per-structure sums are ordered. The model is compared against
// itself, with TF32 off.
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/gemm.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// To the bit: a tolerance would pass the bug this exists to catch.
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
    // The process-global setting, which PET_GEMM could have changed.
    REQUIRE(pet::gemm_mode() == pet::GemmMode::Native);

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
  // Edges never cross structures, so batching must not change any atom's energy
  // or force. Both sides run on the host neighbour path (these structures are
  // small), so this tests the concatenation, not the builders. Not to the bit:
  // the batch pads to its own M and sums per segment, a different fp32 order.
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
        CHECK(de <= 1e-5 * worst(1.0, std::abs(alone.energy[0])));

        double max_f = 0.0, max_df = 0.0;
        for (std::size_t i = 0; i < alone.forces.size(); ++i) {
          max_f = worst(max_f, std::abs(alone.forces[i]));
          max_df = worst(max_df, std::abs(batched.forces[foff + i] - alone.forces[i]));
        }
        // Alone takes the host neighbour path and the batch the device one, and
        // the vendor BLAS picks its kernels by size: fp32 noise, which a model
        // with several attention layers amplifies (~2e-5 on an MI250X).
        INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
        CHECK(max_df <= 1e-4 * std::max(max_f, 1.0));
        foff += alone.forces.size();
      }
    }
  }
}

TEST_CASE("a reused Calculator gives the same batched answer every time",
          "[model][determinism]") {
  // Repeated evaluation, with and without forces, must not move: state carried
  // between calls (an unzeroed counter, say) shows up here.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Calculator calc(found->first, found->second);

    std::vector<pet::System> systems;
    for (const auto& path : golden_paths(model)) {
      Golden g = load_golden(path);
      if (g.periodic) systems.push_back(g.system);
    }
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      for (bool forces : {false, true}) {
        const char* what = forces ? "with forces" : "energy only";
        INFO(what);
        const pet::Results first = calc.compute(systems, forces);
        REQUIRE(first.energy.size() == systems.size());
        // Catches that failure even if the first call already had it.
        for (std::size_t b = 0; b < systems.size(); ++b) {
          const double per_atom = first.energy[b] / std::max(1, systems[b].n_atoms);
          INFO("structure " << b << ": " << per_atom << " eV/atom");
          CHECK(std::abs(per_atom) > 1e-6);
        }
        for (int k = 0; k < 3; ++k) {
          const pet::Results again = calc.compute(systems, forces);
          for (std::size_t i = 0; i < first.energy.size(); ++i) {
            INFO("repeat " << k << ", structure " << i);
            CHECK(first.energy[i] == again.energy[i]);
          }
          REQUIRE(first.forces.size() == again.forces.size());
          for (std::size_t i = 0; i < first.forces.size(); ++i)
            CHECK(first.forces[i] == again.forces[i]);
        }
      }
    }
  }
}

TEST_CASE("recomputing activations in the backward changes nothing", "[model][determinism]") {
  // Every recompute tier re-runs the same kernels on the same data: bit-identical.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    auto calc = [&](pet::Recompute r) {
      pet::Options o;
      o.recompute = r;
      return std::make_unique<pet::Calculator>(found->first, found->second, o);
    };
    auto keep = calc(pet::Recompute::Never), wide = calc(pet::Recompute::Wide),
         layers = calc(pet::Recompute::Layers);
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results ref = keep->compute(g.system, true);
        require_bit_identical(ref, wide->compute(g.system, true), "Recompute::Wide");
        require_bit_identical(ref, layers->compute(g.system, true), "Recompute::Layers");
      }
    }
  }
}

TEST_CASE("a replayed CUDA graph gives the eager answer", "[model][determinism]") {
  // Eager, recording and replaying evaluations all equal one without graphs.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Options eager;
    eager.graphs = false;
    pet::Calculator a(found->first, found->second, eager), b(found->first, found->second);
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results ref = a.compute(g.system, true);
        for (int k = 0; k < kRepeats; ++k) require_bit_identical(ref, b.compute(g.system, true), "graph");
      }
    }
  }
}
