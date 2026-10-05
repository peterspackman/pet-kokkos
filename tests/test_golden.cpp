// Golden references: each shipped structure against metatrain's own evaluation
// of it (tools/make_golden.py, through metatomic's calculator), so the whole
// pipeline is under test: neighbour list, adaptive cutoff, network, assembly.
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

// Tolerances for mixed precision. The energy's is relative or absolute: the
// references are fp32 end to end, and for pbe0's ~-1036 eV/atom one fp32 ulp is
// 1.2e-4 eV/atom (tools/make_fp64_golden.py lifts that floor).
double energy_tolerance(double e_ref) { return std::max(1e-3, 2e-6 * std::fabs(e_ref)); }
constexpr double kForceTolerance = 1e-2;    // eV/Angstrom
constexpr double kStressTolerance = 1e-4;   // eV/Angstrom^3

}  // namespace

TEST_CASE("goldens reproduce the reference energy, forces and stress", "[model][golden]") {
  for (const auto& model : golden_models()) {
    pet::Calculator* calcp = shared_calculator(model);
    if (!calcp) {
      WARN("model '" << model << "' is not installed; skipping its goldens");
      continue;
    }
    pet::Calculator& calc = *calcp;

    for (const auto& path : golden_paths(model)) {
      const Golden g = load_golden(path);
      DYNAMIC_SECTION(model << " / " << g.name) {
        require_matching_model(g, calc);
        const pet::Results r = calc.compute(g.system, /*compute_forces=*/true);

        REQUIRE(r.energy.size() == 1);
        const double de = std::fabs(r.energy[0] - g.total_energy);
        INFO("E = " << r.energy[0] << " eV, reference " << g.total_energy
                    << " eV, |dE| = " << de);
        CHECK(de <= energy_tolerance(g.total_energy));

        REQUIRE(r.forces.size() == g.forces.size());
        double max_df = 0.0;
        for (std::size_t i = 0; i < r.forces.size(); ++i)
          max_df = worst(max_df, std::fabs(r.forces[i] - g.forces[i]));
        INFO("max|dF| = " << max_df << " eV/A");
        CHECK(max_df <= kForceTolerance);

        // Per-atom energies catch a redistribution that leaves the sum right.
        if (!g.per_atom_energies.empty()) {
          REQUIRE(r.per_atom_energy.size() == g.per_atom_energies.size());
          double max_dea = 0.0;
          for (std::size_t i = 0; i < r.per_atom_energy.size(); ++i)
            max_dea = worst(max_dea, std::fabs(r.per_atom_energy[i] - g.per_atom_energies[i]));
          INFO("max|dE_atom| = " << max_dea << " eV");
          CHECK(max_dea <= energy_tolerance(g.total_energy));
        }

        // Stress = W / V. The reference is symmetrized first: autograd's
        // antisymmetric part is noise, and large enough to dominate.
        if (g.stress && g.volume > 0.0) {
          REQUIRE(r.virial.size() == 6);
          constexpr int vi[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
          double max_ds = 0.0;
          for (int t = 0; t < 6; ++t) {
            const double s_pet = r.virial[t] / g.volume;
            const double s_ref =
                0.5 * ((*g.stress)[vi[t][0] * 3 + vi[t][1]] + (*g.stress)[vi[t][1] * 3 + vi[t][0]]);
            max_ds = worst(max_ds, std::fabs(s_pet - s_ref));
          }
          INFO("max|dStress| = " << max_ds << " eV/A^3");
          CHECK(max_ds <= kStressTolerance);
        }
      }
    }
  }
}

TEST_CASE("energy() and energy_forces() agree", "[model][golden]") {
  // Asking for forces must not change the energy: the backward must not have
  // written a buffer the forward result still lives in.
  for (const auto& model : golden_models()) {
    pet::Calculator* calcp = shared_calculator(model);
    if (!calcp) continue;
    pet::Calculator& calc = *calcp;
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
  // Grid and solver choose different cutoffs, so the field decides the energy. A
  // checkpoint without it predates it and is grid; an unknown value is an error.
  const auto found = find_model("pet-mad-xs");
  if (!found) {
    WARN("no adaptive model installed; skipping the capability-gate check");
    return;
  }

  // A real model's metadata with the one field changed, beside its weights.
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
    // The default must reproduce the golden, and "solver" must differ from it.
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

TEST_CASE("charge and spin change a conditioned model's answer", "[model][golden]") {
  // A conditioned model must give different energies for different charge and
  // spin, which conditioning loaded but not applied would not. Runs only where
  // such a model (pet-omol-s) is installed.
  const auto found = find_model("pet-omol-s");
  if (!found) {
    WARN("no conditioned model installed; skipping the charge/spin check");
    return;
  }
  pet::Calculator calc(found->first, found->second);
  if (!calc.hypers().system_conditioning) {
    WARN("pet-omol-s resolved to a model without system_conditioning");
    return;
  }

  pet::System water;
  water.n_atoms = 3;
  water.atomic_numbers = {8, 1, 1};
  water.positions = {0.0, 0.0, 0.11926, 0.0, 0.76323, -0.47704, 0.0, -0.76323, -0.47704};

  const double neutral = calc.compute(water, false).energy.at(0);

  pet::System cation = water;
  cation.charge = 1;
  cation.spin_multiplicity = 2;
  const double charged = calc.compute(cation, false).energy.at(0);

  INFO("neutral singlet " << neutral << " eV, +1 doublet " << charged << " eV");
  CHECK(neutral != charged);

  // A System that says nothing is the neutral singlet, exactly.
  pet::System explicit_neutral = water;
  explicit_neutral.charge = 0;
  explicit_neutral.spin_multiplicity = 1;
  CHECK(calc.compute(explicit_neutral, false).energy.at(0) == neutral);

  // A batch carries each structure's own state, not the first one's.
  const pet::Results batched = calc.compute(std::vector<pet::System>{water, cation}, true);
  REQUIRE(batched.energy.size() == 2);
  INFO("batched: " << batched.energy[0] << ", " << batched.energy[1]);
  CHECK(std::fabs(batched.energy[0] - neutral) < 1e-4);
  CHECK(std::fabs(batched.energy[1] - charged) < 1e-4);
}
