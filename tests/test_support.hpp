// Shared test helpers: finding models and loading golden references. Weights
// are not in the repository, so a test without its model skips, visibly.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/neighbors.hpp"

namespace pet_test {

// A build without a GPU runs the network on the CPU, a hundred or more times
// slower; the heaviest checks take a sample there rather than everything.
inline constexpr bool kHostBackend = std::is_same_v<pet::MemSpace, Kokkos::HostSpace>;

// A running maximum in which NaN wins: std::max(m, NaN) is m, which lets an
// all-NaN result pass every tolerance.
inline double worst(double m, double x) { return (std::isnan(x) || x > m) ? x : m; }

inline const char* golden_dir() { return PET_TEST_GOLDEN_DIR; }

// The ':'-separated entries of an environment variable.
inline std::vector<std::string> env_list(const char* name) {
  const char* e = std::getenv(name);
  const std::string s = e ? e : "";
  std::vector<std::string> out;
  for (std::size_t p = 0, q; p <= s.size(); p = q + 1) {
    q = std::min(s.find(':', p), s.size());
    if (q > p) out.push_back(s.substr(p, q - p));
  }
  return out;
}

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

// A structure and metatrain's evaluation of it: eV, Angstrom, eV/A, eV/A^3.
struct Golden {
  std::string model;
  std::string name;
  pet::System system;
  bool periodic = false;
  double total_energy = 0.0;
  double volume = 0.0;
  std::optional<double> model_energy_scale;  // see require_matching_model
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

  // Lattice vectors as rows, like pet::System.
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

// The shipped goldens, plus $PET_TEST_GOLDEN_EXTRA: goldens for a model whose
// weights cannot be redistributed live next to those weights.
inline std::vector<std::string> golden_dirs() {
  std::vector<std::string> dirs{golden_dir()};
  for (auto& d : env_list("PET_TEST_GOLDEN_EXTRA")) dirs.push_back(d);
  return dirs;
}

// Refuse a golden made from a different checkpoint of the same name (pet-mad-xs
// has several published versions). energy_scale, a fitted constant, is the
// fingerprint.
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

// The models the goldens cover, plus $PET_TEST_MODELS:
//   pet-mad-xs       v1.5.0, grid adaptive cutoff
//   pet-mad-xs-v1.6  v1.6.0, solver adaptive cutoff
//   pet-mols-s-v1.0  v1.0.0, the residual featurizer: PostLN, LayerNorm, SiLU,
//                    cosine cutoff
//   pet-attn2        synthetic, two attention layers, as every published model
//                    from size m up has (tools/make_multilayer_checkpoint.py)
// .github/workflows/ci.yml shows how each is made.
inline const std::vector<std::string>& golden_models() {
  static const std::vector<std::string> m = [] {
    std::vector<std::string> v{"pet-mad-xs", "pet-mad-xs-v1.6", "pet-mols-s-v1.0", "pet-attn2"};
    for (auto& n : env_list("PET_TEST_MODELS")) v.push_back(n);
    return v;
  }();
  return m;
}

// The models a test of the machinery around the network runs -- neighbour
// lists, caches, stepping, an engine's ghosts -- which is the same for every
// architecture: every golden model on a GPU, and on the host only the first
// installed one, the CPU being slow and the per-architecture tests (goldens,
// finite differences, determinism) covering the rest.
inline std::vector<std::string> plumbing_models() {
  if (!kHostBackend) return golden_models();
  for (const auto& m : golden_models())
    if (find_model(m)) return {m};
  return {};
}

// One default-options Calculator per model, shared across test cases (loading
// is the slow part). A test needing other Options builds its own. Leaked: freed
// at static destruction, the Views would outlive Kokkos.
inline pet::Calculator* shared_calculator(const std::string& model) {
  static auto* cache = new std::map<std::string, pet::Calculator*>();
  const auto it = cache->find(model);
  if (it != cache->end()) return it->second;
  const auto found = find_model(model);
  if (!found) {
    cache->emplace(model, nullptr);
    return nullptr;
  }
  auto* calc = new pet::Calculator(found->first, found->second);
  cache->emplace(model, calc);
  return calc;
}

}  // namespace pet_test
