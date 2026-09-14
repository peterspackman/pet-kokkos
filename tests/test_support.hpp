// Shared test helpers: locating models and loading golden references.
//
// Model weights are not in this repository (see README.md), so every test that
// needs one has to be able to SKIP rather than fail when it is absent. A fresh
// clone must produce a green test run; "green except the model tests, which
// cannot run here" is what that means in practice, and it is reported as skips
// so the difference is visible.
#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/neighbors.hpp"

namespace pet_test {

// Directory holding the golden reference structures, baked in at configure time.
inline const char* golden_dir() { return PET_TEST_GOLDEN_DIR; }

// Resolve a named model, or nullopt when it is not installed on this machine.
inline std::optional<std::pair<std::string, std::string>> find_model(const std::string& name) {
  try {
    std::string j, w;
    pet::resolve_model(name, j, w);
    return std::make_pair(j, w);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// A golden reference: the structure plus what metatrain's own evaluation of it
// produced. Energies eV, positions Angstrom, forces eV/Angstrom, stress
// eV/Angstrom^3.
struct Golden {
  std::string model;
  std::string name;
  pet::System system;
  bool periodic = false;
  double total_energy = 0.0;
  double volume = 0.0;
  // The energy scale of the model this golden was generated from, when it
  // recorded one. Used to refuse a mismatched model -- see require_matching_model.
  std::optional<double> model_energy_scale;
  std::vector<double> per_atom_energies;   // [N]
  std::vector<double> forces;              // [N*3]
  std::optional<std::array<double, 9>> stress;  // row-major 3x3, if periodic
};

inline Golden load_golden(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open golden '" + path + "'");
  nlohmann::json g;
  f >> g;

  Golden out;
  out.model = g.value("model", std::string{});
  out.name = g.value("case", std::string{});
  out.periodic = g.value("periodic", false);
  out.total_energy = g.at("total_energy").get<double>();
  out.volume = g.value("volume", 0.0);
  if (g.contains("model_metadata") && g.at("model_metadata").contains("energy_scale"))
    out.model_energy_scale = g.at("model_metadata").at("energy_scale").get<double>();

  const int N = g.at("n_atoms").get<int>();
  out.system.n_atoms = N;
  out.system.atomic_numbers = g.at("atomic_numbers").get<std::vector<int>>();
  out.system.positions.reserve(static_cast<std::size_t>(N) * 3);
  for (const auto& p : g.at("positions"))
    for (int c = 0; c < 3; ++c) out.system.positions.push_back(p[c].get<double>());

  // Golden cells store lattice vectors as ROWS, which is what pet::System wants.
  int r = 0;
  for (const auto& row : g.at("cell")) {
    for (int c = 0; c < 3; ++c) out.system.cell[r * 3 + c] = row[c].get<double>();
    ++r;
  }
  const auto pbc = g.at("pbc").get<std::vector<bool>>();
  for (int d = 0; d < 3 && d < (int) pbc.size(); ++d) out.system.pbc[d] = pbc[d];

  if (g.contains("per_atom_energies") && !g.at("per_atom_energies").is_null())
    out.per_atom_energies = g.at("per_atom_energies").get<std::vector<double>>();

  for (const auto& fv : g.at("forces"))
    for (int c = 0; c < 3; ++c) out.forces.push_back(fv[c].get<double>());

  if (g.contains("stress") && !g.at("stress").is_null()) {
    std::array<double, 9> s{};
    int i = 0;
    for (const auto& row : g.at("stress")) {
      for (int c = 0; c < 3; ++c) s[i * 3 + c] = row[c].get<double>();
      ++i;
    }
    out.stress = s;
  }
  return out;
}

// Directories scanned for goldens: the shipped ones, plus anything in
// $PET_TEST_GOLDEN_EXTRA (':'-separated). The extra path exists because a model
// whose weights cannot be redistributed still needs its goldens run somewhere --
// generate them next to the weights and point this at the directory.
inline std::vector<std::string> golden_dirs() {
  std::vector<std::string> dirs{golden_dir()};
  if (const char* e = std::getenv("PET_TEST_GOLDEN_EXTRA")) {
    const std::string s(e);
    for (std::size_t p = 0; p <= s.size();) {
      const std::size_t q = s.find(':', p);
      std::string d = s.substr(p, (q == std::string::npos ? s.size() : q) - p);
      if (!d.empty()) dirs.push_back(std::move(d));
      if (q == std::string::npos) break;
      p = q + 1;
    }
  }
  return dirs;
}

// Refuse to compare a golden against a model it was not generated from.
//
// Model NAMES are not unique across time: `pet-mad-xs` means one checkpoint to
// the goldens shipped here and a different one on HuggingFace today (different
// weights, different cutoff width, grid vs solver). Both legitimately answer to
// that name, so whichever directory comes first on the search path wins -- and
// the symptom is a golden failing by 0.14 eV with no hint that it is comparing
// two different models.
//
// energy_scale is the fingerprint: a per-model fitted constant, recorded in the
// golden's model_metadata, and different for any two distinct checkpoints.
inline void require_matching_model(const Golden& g, const pet::Calculator& calc) {
  if (!g.model_energy_scale) return;  // an older golden with no fingerprint
  const double want = *g.model_energy_scale, got = calc.energy_scale();
  INFO("golden '" << g.model << "/" << g.name << "' was generated from a model with "
                  << "energy_scale " << want << ", but the model resolved for it has " << got
                  << ". A different checkpoint is answering to the same name -- check "
                     "PET_MODEL_DIR ordering.");
  REQUIRE(std::fabs(want - got) <= 1e-9 * std::max(1.0, std::fabs(want)));
}

// Every golden available for a given model.
inline std::vector<std::string> golden_paths(const std::string& model) {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& dir : golden_dirs()) {
    if (!std::filesystem::is_directory(dir, ec)) continue;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string fn = e.path().filename().string();
      if (fn.rfind(model + "_", 0) == 0 && e.path().extension() == ".json")
        out.push_back(e.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The models the goldens cover. Kept here so a new architecture lands as one
// entry plus its goldens, not as an edit to every test. Extra names can be added
// at run time through $PET_TEST_MODELS (':'-separated), which is how a model
// that cannot be committed still gets exercised -- see golden_dirs().
inline const std::vector<std::string>& golden_models() {
  static const std::vector<std::string> m = [] {
    // pet-mad-xs   -- 2026.1-era checkpoint, adaptive_cutoff_method = grid
    // pbe0-pet     -- residual featurizer, PostLN, LayerNorm, SiLU, fixed cosine cutoff
    // pet-mad-xs-v1.6 -- current release, adaptive_cutoff_method = solver. Its
    //                 goldens are shipped; reproduce the model itself with
    //                 "uv run tools/convert_pet.py --model pet-mad-xs
    //                  --out models/pet-mad-xs-v1.6".
    // pet-attn2  -- synthetic, num_attention_layers = 2 (see its goldens' notes,
    //               and tools/make_multilayer_checkpoint.py). No published upet
    //               model uses A > 1, but metatrain defaults to it.
    std::vector<std::string> v{"pet-mad-xs", "pbe0-pet", "pet-mad-xs-v1.6", "pet-attn2"};
    if (const char* e = std::getenv("PET_TEST_MODELS")) {
      const std::string s(e);
      for (std::size_t p = 0; p <= s.size();) {
        const std::size_t q = s.find(':', p);
        std::string n = s.substr(p, (q == std::string::npos ? s.size() : q) - p);
        if (!n.empty()) v.push_back(std::move(n));
        if (q == std::string::npos) break;
        p = q + 1;
      }
    }
    return v;
  }();
  return m;
}

}  // namespace pet_test
