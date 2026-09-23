// A path and its fallback must agree. A difference between them is a bug, not a
// setting.
//
// Two pairs are checked:
//
//   host NEF        vs  device NEF     (Options::device_neighbors)
//   uncached device vs  Verlet-cached  (Options::cache_neighbors)
//
// Neither pair is bit-identical, and the reasons differ -- see each test. What
// matters is that both agree to within the fp32 network's own resolution, and
// that a fresh cache agrees to DOUBLE round-off, which is the check a dropped
// edge cannot survive.
//
// That last one is not hypothetical: the cached path was once caught dropping
// edges within 0.1 A of the cutoff, because it reused image ranges sized for the
// bare cutoff while caching out to cutoff+skin, so an edge in the outermost
// shell was silently absent for every round the cache was reused. The energy
// error was below the fp32 force noise, which is precisely why only an
// exact-agreement test finds it.
//
// These claims used to be checked by hand through environment variables. They
// are load-bearing enough to be tests.
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

// The network runs in fp32 (mixed precision), so any two orderings of the same
// arithmetic differ by a few fp32 ulps. Anything at or below this is the
// precision of the model, not a disagreement between paths; anything above it
// means the two paths saw different EDGES.
constexpr double kFp32Noise = 1e-5;

// Double round-off. A fresh cache re-derives the same edges from the same
// coordinates, so only the summation order can differ at all.
constexpr double kFp64Noise = 1e-10;

// Two or more periodic goldens, so the batched (device) path has something to do.
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
  // Not bit-identical, and cannot be: the host builder walks images and atoms in
  // nested loops on one thread, while the device builder distributes the same
  // walk across threads and recovers per-atom ranges through a count-plus-scan.
  // The edge SET and its per-atom ORDER are the same -- that is what the scan
  // guarantees -- but the arithmetic that produces each edge vector is issued
  // differently, and the fp32 network turns a double-precision ulp into an fp32
  // one. Measured: ~3e-8 relative on the energy, i.e. below fp32 epsilon.
  //
  // So the bound is RELATIVE and sits at the network's own resolution. An
  // absolute bound would be meaningless here: these models' total energies span
  // -15 eV to -8293 eV.
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
  // This is the strong one. On the round that builds it, the cache has just
  // enumerated the same neighbours from the same coordinates as the uncached
  // search would, so the only possible difference is summation order in double
  // precision. Measured: ~4e-16 relative, i.e. exact for every practical
  // purpose.
  //
  // A dropped or spurious edge cannot hide under a bound this tight -- which is
  // the whole reason to test the fresh cache separately from the reused one.
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
  // Once the geometry has moved, the cached path is no longer doing the same
  // arithmetic as the uncached one, and it is worth being precise about why --
  // it is not a tolerance chosen to make a test pass.
  //
  // The uncached search runs on WRAPPED coordinates and states each edge's image
  // in wrapped terms, so its edge vector is posw(j) + shift*cell - posw(i). The
  // cache stores its images in UNWRAPPED terms (that is what lets a stored
  // triple survive an atom crossing a cell face, and it is why the cache is
  // useful at all), so its edge vector is pos(j) + cshift*cell - pos(i). Those
  // are the same vector in exact arithmetic and different roundings in floating
  // point, and the fp32 network amplifies the difference to a few fp32 ulps.
  //
  // Measured: ~9e-9 relative on the energy, ~2e-7 on the forces -- at or below
  // fp32 epsilon. The edge SET is identical; only the last bits of each vector
  // differ. (An earlier version of this code kept `pos` wrapped in place and
  // could claim bit-identity; that version also rebuilt the cache on 76% of
  // rounds, because a face crossing looked like a lattice-vector jump to the
  // skin test. The current trade is the right one, but it does cost exactness.)
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

      // Round 0 builds the cache; the rest reuse it after a displacement well
      // inside the skin, which is the state that actually matters.
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
