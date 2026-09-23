// A path and its fallback must agree, to the precision of whatever differs
// between them:
//   host vs device neighbour lists   same edges and order, fp32 noise
//   fresh Verlet cache vs uncached   same edges, double round-off
//   reused cache vs uncached         unwrapped vs wrapped images, fp32 noise
// The fresh-cache bound is the one a dropped edge cannot pass: the cache once
// lost edges within 0.1 A of the cutoff, an energy error below the force noise.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "pet/calculator.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// Largest absolute difference, and the scale to judge it against.
struct Deviation {
  double abs = 0.0;
  double scale = 0.0;
  double relative() const { return abs / std::max(scale, 1e-30); }
};

Deviation deviation(const std::vector<double>& a, const std::vector<double>& b) {
  Deviation d;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    d.abs = worst(d.abs, std::fabs(a[i] - b[i]));
    d.scale = worst(d.scale, std::fabs(b[i]));
  }
  return d;
}

// Two orderings of the fp32 network's arithmetic; more means different edges.
constexpr double kFp32Noise = 1e-5;

// Double round-off: only the summation order may differ.
constexpr double kFp64Noise = 1e-10;

std::vector<pet::System> periodic_systems(const std::string& model) {
  std::vector<pet::System> out;
  for (const auto& path : golden_paths(model)) {
    const Golden g = load_golden(path);
    if (g.periodic) out.push_back(g.system);
  }
  return out;
}

}  // namespace

TEST_CASE("the host and device neighbour builders agree", "[model][paths]") {
  // The builders issue the edge arithmetic differently, and the fp32 network
  // turns a double ulp into an fp32 one: ~3e-8 relative measured. Relative,
  // because the energies span -15 to -8293 eV.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) {
      WARN("model '" << model << "' is not installed; skipping its path comparison");
      continue;
    }
    const auto systems = periodic_systems(model);
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      pet::Options on_device;
      on_device.device_neighbors = true;
      pet::Options on_host;
      on_host.device_neighbors = false;

      pet::Calculator dev(found->first, found->second, on_device);
      pet::Calculator hst(found->first, found->second, on_host);

      const pet::Results rd = dev.compute(systems, true);
      const pet::Results rh = hst.compute(systems, true);

      REQUIRE(rd.energy.size() == rh.energy.size());
      REQUIRE(rd.forces.size() == rh.forces.size());
      const Deviation e = deviation(rd.energy, rh.energy);
      const Deviation f = deviation(rd.forces, rh.forces);
      const Deviation w = deviation(rd.virial, rh.virial);
      INFO("energy " << e.abs << " eV (rel " << e.relative() << "), force " << f.abs
                     << " eV/A (rel " << f.relative() << "), virial " << w.abs << " eV (rel "
                     << w.relative() << ")");
      CHECK(e.relative() <= kFp32Noise);
      CHECK(f.relative() <= kFp32Noise);
      CHECK(w.relative() <= kFp32Noise);
    }
  }
}

TEST_CASE("a freshly built Verlet cache reproduces the uncached search", "[model][paths]") {
  // The same neighbours from the same coordinates: ~4e-16 relative measured.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    const auto systems = periodic_systems(model);
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      pet::Options cached;
      cached.cache_neighbors = true;
      pet::Calculator with_cache(found->first, found->second, cached);
      pet::Calculator no_cache(found->first, found->second);

      const pet::Results a = with_cache.compute(systems, true);
      const pet::Results b = no_cache.compute(systems, true);

      REQUIRE(a.energy.size() == b.energy.size());
      REQUIRE(a.forces.size() == b.forces.size());
      const Deviation e = deviation(a.energy, b.energy);
      const Deviation f = deviation(a.forces, b.forces);
      INFO("energy " << e.abs << " eV (rel " << e.relative() << "), force " << f.abs
                     << " eV/A (rel " << f.relative() << ")");
      CHECK(e.relative() <= kFp64Noise);
      CHECK(f.relative() <= kFp64Noise);
    }
  }
}

TEST_CASE("a reused Verlet cache stays within the network's precision", "[model][paths]") {
  // The cache states images in unwrapped coordinates (so a stored edge survives
  // an atom crossing a face), the search in wrapped ones: the same vectors,
  // rounded differently. ~9e-9 relative on the energy, ~2e-7 on the forces.
  for (const auto& model : golden_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    auto systems = periodic_systems(model);
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      pet::Options cached;
      cached.cache_neighbors = true;
      pet::Calculator with_cache(found->first, found->second, cached);
      pet::Calculator no_cache(found->first, found->second);

      // Round 0 builds the cache; later rounds move the atoms well inside the skin.
      for (int round = 0; round < 3; ++round) {
        if (round > 0)
          for (auto& s : systems)
            for (std::size_t i = 0; i < s.positions.size(); ++i)
              s.positions[i] += 0.01 * ((i % 3) == 0 ? 1.0 : -1.0);

        const pet::Results a = with_cache.compute(systems, true);
        const pet::Results b = no_cache.compute(systems, true);

        const Deviation e = deviation(a.energy, b.energy);
        const Deviation f = deviation(a.forces, b.forces);
        INFO("round " << round << ": energy " << e.abs << " eV (rel " << e.relative()
                      << "), force " << f.abs << " eV/A (rel " << f.relative() << ")");
        CHECK(e.relative() <= kFp32Noise);
        CHECK(f.relative() <= kFp32Noise);
      }
    }
  }
}
