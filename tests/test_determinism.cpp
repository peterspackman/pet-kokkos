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

#include <memory>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/gemm.hpp"
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
    // The GLOBAL, not the Options field: TF32 is a property of the process's
    // cuBLAS handle, so checking the field would pass while a stray PET_TF32=1
    // in the environment quietly made this test meaningless.
    REQUIRE_FALSE(pet::tf32_enabled());

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
        CHECK(de <= 1e-5 * worst(1.0, std::abs(alone.energy[0])));

        double max_f = 0.0, max_df = 0.0;
        for (std::size_t i = 0; i < alone.forces.size(); ++i) {
          max_f = worst(max_f, std::abs(alone.forces[i]));
          max_df = worst(max_df, std::abs(batched.forces[foff + i] - alone.forces[i]));
        }
        INFO("max|dF| = " << max_df << " eV/A on max|F| = " << max_f << " eV/A");
        CHECK(max_df <= 1e-5 * std::max(max_f, 1.0));
        foff += alone.forces.size();
      }
    }
  }
}

TEST_CASE("a reused Calculator gives the same batched answer every time",
          "[model][determinism]") {
  // The batched path keeps per-structure counters in the pooled workspace, and
  // those are atomic accumulators: if one is not zeroed between calls it carries
  // the previous call's count. That is not hypothetical -- energy_forces_batch's
  // segmentation counter relied on the pool's zeroing policy, which compute()
  // turns OFF for the forward and only back on for the backward. An energy-only
  // batch never runs a backward, so the policy stayed off and the second call
  // put every atom in structure 0: the first structure got the whole batch's
  // energy and the rest got nothing.
  //
  // Correct on the first call and silently wrong on the second is exactly the
  // shape no single-shot test can see, so this one evaluates repeatedly, with
  // and without forces, and demands the answer not move.
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
        // Every structure must have a sane per-atom energy. The failure this
        // guards against gave structure 0 the sum of the whole batch and the
        // others zero, which a same-vs-same comparison alone would miss if the
        // very first call were already wrong.
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
  // Every recompute tier re-runs the same kernels on the same data, so the answer
  // must be bit-identical to keeping everything -- anything else means a rebuild
  // is not the forward it claims to be.
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
