// Golden-reference regression: evaluate each shipped structure against the
// model it was generated with and compare energy, forces and stress to what
// metatrain's own evaluation produced.
//
// This is the contract the extraction from klasp was measured against, and it
// is the contract every later change is measured against. The references were
// produced by tools/make_golden.py through
//   metatrain.utils.io.load_model(ckpt).export() -> metatomic AtomisticModel
//   -> MetatomicCalculator.compute_energy(atoms, compute_forces_and_stresses=True)
// so they test the whole pipeline: neighbour list, adaptive cutoff, NEF packing,
// the network, and the energy assembly (scale then composition).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// Tolerances for the default mixed precision (fp32 network, fp64 geometry and
// accumulation).
//
// The energy bound is relative-OR-absolute deliberately. fp32 gives ~1e-7
// relative accuracy, so for composition-heavy absolute energies -- the pbe0
// model sits near -1036 eV/atom, because its composition energies are full
// atomic DFT totals -- an absolute-only bound is far tighter than the reference
// itself is. The reference is float32 end to end, and one fp32 ulp there is
// 1.2e-4 eV/atom; see tools/make_fp64_golden.py, which exists precisely to lift
// that floor when a tighter comparison is wanted.
double energy_tolerance(double e_ref) { return std::max(1e-3, 2e-6 * std::fabs(e_ref)); }
constexpr double kForceTolerance = 1e-2;    // eV/Angstrom
constexpr double kStressTolerance = 1e-4;   // eV/Angstrom^3

}  // namespace

TEST_CASE("goldens reproduce the reference energy, forces and stress", "[model][golden]") {
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) {
      WARN("model '" << model << "' is not installed; skipping its goldens");
      continue;
    }
    pet::Calculator calc(found->first, found->second);

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const pet::Results r = calc.compute(g.system, /*compute_forces=*/true);

        REQUIRE(r.energy.size() == 1);
        const double de = std::fabs(r.energy[0] - g.total_energy);
        INFO("E = " << r.energy[0] << " eV, reference " << g.total_energy
                    << " eV, |dE| = " << de);
        CHECK(de <= energy_tolerance(g.total_energy));

        REQUIRE(r.forces.size() == g.forces.size());
        double max_df = 0.0;
        for (std::size_t i = 0; i < r.forces.size(); ++i)
          max_df = std::max(max_df, std::fabs(r.forces[i] - g.forces[i]));
        INFO("max|dF| = " << max_df << " eV/A");
        CHECK(max_df <= kForceTolerance);

        // Per-atom energies, when the reference carries them. They sum to the
        // total, so this catches a per-atom redistribution that leaves the sum
        // right -- which a broken neighbour list can absolutely produce.
        if (!g.per_atom_energies.empty()) {
          REQUIRE(r.per_atom_energy.size() == g.per_atom_energies.size());
          double max_dea = 0.0;
          for (std::size_t i = 0; i < r.per_atom_energy.size(); ++i)
            max_dea = std::max(max_dea, std::fabs(r.per_atom_energy[i] - g.per_atom_energies[i]));
          INFO("max|dE_atom| = " << max_dea << " eV");
          CHECK(max_dea <= energy_tolerance(g.total_energy));
        }

        // Stress, for the periodic goldens. We return the symmetric virial
        // W = V*sigma in Voigt order, so sigma = W / V directly -- no shear
        // factor, unlike a strain-gradient consumer.
        //
        // The reference is SYMMETRIZED before comparison. The stress a stress
        // tensor describes is symmetric, but autograd's is not exactly: its
        // antisymmetric part is noise, and comparing against a single
        // off-diagonal element measures that noise rather than this library.
        // It used to dominate the number -- max|S-S^T|/2 matched the reported
        // deviation to four digits on every periodic golden.
        if (g.stress && g.volume > 0.0) {
          REQUIRE(r.virial.size() == 6);
          constexpr int vi[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
          double max_ds = 0.0;
          for (int t = 0; t < 6; ++t) {
            const double s_pet = r.virial[t] / g.volume;
            const double s_ref =
                0.5 * ((*g.stress)[vi[t][0] * 3 + vi[t][1]] + (*g.stress)[vi[t][1] * 3 + vi[t][0]]);
            max_ds = std::max(max_ds, std::fabs(s_pet - s_ref));
          }
          INFO("max|dStress| = " << max_ds << " eV/A^3");
          CHECK(max_ds <= kStressTolerance);
        }
      }
    }
  }
}

TEST_CASE("energy() and energy_forces() agree", "[model][golden]") {
  // The forward pass is shared, so asking for forces must not change the energy.
  // It is the cheapest possible check that the backward has not corrupted a
  // forward buffer it was only supposed to read -- which the Workspace's
  // zeroing policy makes a live risk every time a buffer key is reused.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    pet::Calculator calc(found->first, found->second);
    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        const double e_only = calc.compute(g.system, false).energy.at(0);
        const double e_grad = calc.compute(g.system, true).energy.at(0);
        CHECK(e_only == e_grad);
      }
    }
  }
}

TEST_CASE("the loader refuses an architecture it does not implement", "[model][golden]") {
  // A model whose hypers name a scheme this build does not have must be
  // REFUSED, not evaluated on the nearest thing available. The specific trap:
  // metatrain's adaptive_cutoff_method default changed from "grid" to "solver",
  // the two choose different per-atom cutoffs, and a checkpoint from before the
  // change carries no such field at all. So "field absent" has to mean "grid"
  // while "field says solver" has to be an error -- and the difference between
  // those two behaviours is silently wrong energies.
  const auto found = find_model("pet-mad-xs");
  if (!found) {
    WARN("no adaptive model installed; skipping the capability-gate check");
    return;
  }

  // Take a real model's metadata, flip the one field, and point the loader at
  // it with the genuine weights alongside.
  std::ifstream in(found->first);
  REQUIRE(in);
  nlohmann::json meta;
  in >> meta;
  REQUIRE(meta.at("hypers").at("num_neighbors_adaptive").is_number());  // adaptive model

  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / "pet_kokkos_gate_test";
  std::filesystem::create_directories(tmp);
  const std::string json_path = (tmp / "gated.json").string();

  SECTION("an unimplemented method is refused by name") {
    meta["hypers"]["adaptive_cutoff_method"] = "solver";
    std::ofstream(json_path) << meta.dump(2);
    REQUIRE_THROWS_WITH(pet::Calculator(json_path, found->second),
                        Catch::Matchers::ContainsSubstring("solver"));
  }
  SECTION("an unrecognized method is refused too, rather than defaulted") {
    meta["hypers"]["adaptive_cutoff_method"] = "something-new";
    std::ofstream(json_path) << meta.dump(2);
    REQUIRE_THROWS(pet::Calculator(json_path, found->second));
  }
  SECTION("an absent method means grid, and still loads") {
    meta["hypers"].erase("adaptive_cutoff_method");
    std::ofstream(json_path) << meta.dump(2);
    REQUIRE_NOTHROW(pet::Calculator(json_path, found->second));
  }

  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}
