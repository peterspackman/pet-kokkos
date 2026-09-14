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

TEST_CASE("adaptive_cutoff_method is read, defaulted and validated", "[model][golden]") {
  // metatrain's adaptive_cutoff_method default changed from "grid" to "solver",
  // and the two choose different per-atom cutoffs -- so this field decides which
  // energy a model returns, and getting its DEFAULT wrong is silently wrong
  // results rather than a crash.
  //
  // A checkpoint converted before the field existed carries no such key, and
  // must be read as "grid", because that is the scheme the metatrain that
  // produced it used. Both named schemes are implemented now; anything else has
  // to be an error rather than a fallback.
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

  SECTION("both named schemes load") {
    for (const char* method : {"grid", "solver"}) {
      meta["hypers"]["adaptive_cutoff_method"] = method;
      std::ofstream(json_path) << meta.dump(2);
      INFO(method);
      REQUIRE_NOTHROW(pet::Calculator(json_path, found->second));
    }
  }
  SECTION("an unrecognized method is refused, rather than defaulted") {
    meta["hypers"]["adaptive_cutoff_method"] = "something-new";
    std::ofstream(json_path) << meta.dump(2);
    REQUIRE_THROWS_WITH(pet::Calculator(json_path, found->second),
                        Catch::Matchers::ContainsSubstring("something-new"));
  }
  SECTION("an absent method means grid, and gives grid's answer") {
    // Not just "loads": the whole point is WHICH scheme it silently picks. This
    // model's shipped metadata has no adaptive_cutoff_method, so the default has
    // to reproduce the golden -- and an explicit "solver" has to differ from it,
    // or the two schemes are not actually distinct in this build.
    meta["hypers"].erase("adaptive_cutoff_method");
    std::ofstream(json_path) << meta.dump(2);
    pet::Calculator defaulted(json_path, found->second);

    meta["hypers"]["adaptive_cutoff_method"] = "grid";
    std::ofstream(json_path) << meta.dump(2);
    pet::Calculator grid(json_path, found->second);

    meta["hypers"]["adaptive_cutoff_method"] = "solver";
    std::ofstream(json_path) << meta.dump(2);
    pet::Calculator solver(json_path, found->second);

    const auto goldens = golden_paths("pet-mad-xs");
    REQUIRE_FALSE(goldens.empty());
    const Golden g = load_golden(goldens.front());
    const double e_def = defaulted.compute(g.system, false).energy.at(0);
    const double e_grid = grid.compute(g.system, false).energy.at(0);
    const double e_solver = solver.compute(g.system, false).energy.at(0);
    INFO("default " << e_def << ", grid " << e_grid << ", solver " << e_solver);
    CHECK(e_def == e_grid);
    CHECK(e_def != e_solver);
  }

  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}
