// A path and its fallback must agree, to the precision of whatever differs
// between them:
//   host vs device neighbour lists   same edges and order, fp32 noise
//   fresh Verlet cache vs uncached   same edges, double round-off
//   reused cache vs uncached         unwrapped vs wrapped images, fp32 noise
// The fresh-cache bound is tight enough that dropping an edge near the cutoff
// fails it, an error the force tolerance alone would miss.
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
  for (const auto& model : plumbing_models()) {
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
  for (const auto& model : plumbing_models()) {
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
  for (const auto& model : plumbing_models()) {
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

namespace {

// Results of the leading `B` structures of a device evaluation, on the host.
pet::Results to_host(const pet::BatchResult& br, int B, int Ntot) {
  pet::Results r;
  auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), br.energy);
  auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), br.forces);
  for (int b = 0; b < B; ++b) r.energy.push_back(e(b));
  for (int i = 0; i < Ntot; ++i)
    for (int c = 0; c < 3; ++c) r.forces.push_back(f(i, c));
  return r;
}

}  // namespace

TEST_CASE("staged geometry follows moves, strains and a shrinking batch", "[model][paths]") {
  // What a relaxation driver does between steps: move atoms and cell in place,
  // and drop finished structures off the end. Each state must match evaluating
  // it from scratch.
  for (const auto& model : plumbing_models()) {
    const auto found = find_model(model);
    if (!found) continue;
    auto systems = periodic_systems(model);
    if (systems.size() < 2) continue;

    DYNAMIC_SECTION(model) {
      pet::Options o;
      o.cache_neighbors = true;
      pet::Calculator calc(found->first, found->second, o), ref(found->first, found->second);
      pet::DeviceGeom g = calc.stage(systems);
      int Ntot = 0;
      for (auto& s : systems) Ntot += s.n_atoms;

      auto check = [&](int B, int N, const char* what) {
        const std::vector<pet::System> head(systems.begin(), systems.begin() + B);
        const pet::Results a = to_host(calc.compute_device(g), B, N), b = ref.compute(head, true);
        const Deviation e = deviation(a.energy, b.energy), f = deviation(a.forces, b.forces);
        INFO(what << ": energy rel " << e.relative() << ", force rel " << f.relative());
        CHECK(e.relative() <= kFp32Noise);
        CHECK(f.relative() <= kFp32Noise);
      };
      check(g.B, Ntot, "staged");

      // Move every atom, and strain every cell by 15% so it needs other images.
      for (auto& s : systems) {
        for (std::size_t i = 0; i < s.positions.size(); ++i) s.positions[i] = 0.85 * s.positions[i] + 0.01 * (i % 5);
        for (auto& c : s.cell) c *= 0.85;
      }
      auto hp = Kokkos::create_mirror_view(g.pos);
      auto hc = Kokkos::create_mirror_view(g.scell);
      for (int b = 0, gi = 0; b < (int) systems.size(); ++b) {
        for (int e = 0; e < 9; ++e) hc(b, e) = systems[b].cell[e];
        for (int i = 0; i < systems[b].n_atoms; ++i, ++gi)
          for (int d = 0; d < 3; ++d) hp(gi, d) = systems[b].positions[3 * i + d];
      }
      Kokkos::deep_copy(g.pos, hp), Kokkos::deep_copy(g.scell, hc);
      check(g.B, Ntot, "moved and strained");

      // The first structure finishes: the rest are packed to the front, as a
      // relaxation driver does, and `staged` says where each went. The cache
      // reassembles their pairs rather than searching again.
      if (systems.size() >= 2) {
        const std::vector<pet::System> rest(systems.begin() + 1, systems.end());
        auto hsid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.sid);
        auto hspec = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.spec);
        auto hsoff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.soff);
        auto hscnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.scnt);
        auto hsper = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.sper);
        auto hst = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.staged);
        const int skip = systems[0].n_atoms;
        int N = 0;
        for (int b = 1, gi = 0; b < (int) systems.size(); ++b) {
          const int from = hsoff(b), n = hscnt(b);
          for (int e = 0; e < 9; ++e) hc(b - 1, e) = systems[b].cell[e];
          hsoff(b - 1) = gi, hscnt(b - 1) = n, hsper(b - 1) = hsper(b), hst(b - 1) = b;
          for (int i = 0; i < n; ++i, ++gi) {
            for (int d = 0; d < 3; ++d) hp(gi, d) = hp(from + i, d);
            hspec(gi) = hspec(from + i), hsid(gi) = b - 1;
          }
          N = gi;
        }
        REQUIRE(N == Ntot - skip);
        Kokkos::deep_copy(g.pos, hp), Kokkos::deep_copy(g.scell, hc), Kokkos::deep_copy(g.sid, hsid);
        Kokkos::deep_copy(g.spec, hspec), Kokkos::deep_copy(g.soff, hsoff), Kokkos::deep_copy(g.scnt, hscnt);
        Kokkos::deep_copy(g.sper, hsper), Kokkos::deep_copy(g.staged, hst);
        g.B = (int) rest.size(), g.Ntot = N;
        const pet::Results a = to_host(calc.compute_device(g), g.B, N), b = ref.compute(rest, true);
        const Deviation e = deviation(a.energy, b.energy), f = deviation(a.forces, b.forces);
        INFO("packed after the first finished: energy rel " << e.relative() << ", force rel " << f.relative());
        CHECK(e.relative() <= kFp32Noise);
        CHECK(f.relative() <= kFp32Noise);
        // Restore the full batch for what follows.
        g = calc.stage(systems);
      }

      // The leading structure alone must be exactly what staging it alone gives.
      g.B = 1, g.Ntot = systems[0].n_atoms;
      pet::Calculator solo(found->first, found->second);
      const pet::Results p = to_host(calc.compute_device(g), 1, g.Ntot);
      const pet::Results q = to_host(solo.compute_device(solo.stage({systems[0]})), 1, g.Ntot);
      const Deviation e = deviation(p.energy, q.energy), f = deviation(p.forces, q.forces);
      INFO("leading structure only: energy rel " << e.relative() << ", force rel " << f.relative());
      CHECK(e.relative() <= kFp32Noise);
      CHECK(f.relative() <= kFp32Noise);
    }
  }
}

TEST_CASE("a slab is periodic along two axes only, on both paths", "[model][paths]") {
  const auto found = find_model("pet-mad-xs");
  if (!found) return;
  // A 2x2 layer of a small cubic lattice with a vacuum axis whose cell vector is
  // shorter than the layer is thick: wrapping or imaging along it would be wrong.
  pet::System s;
  s.cell = {6.0, 0, 0, 0, 6.0, 0, 0, 0, 3.0};
  s.pbc = {true, true, false};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        s.atomic_numbers.push_back(6);
        s.positions.insert(s.positions.end(), {2.0 * i + 0.1 * k, 2.0 * j, 1.6 * k - 0.5});
      }
  s.n_atoms = (int) s.atomic_numbers.size();
  pet::Options host;
  host.device_neighbors = false;
  host.neighbors = pet::Options::Neighbors::Builtin;  // vesin refuses mixed periodicity
  pet::Calculator dev(found->first, found->second), hst(found->first, found->second, host);
  const std::vector<pet::System> two{s, s};  // a batch, so the device path runs
  const pet::Results a = dev.compute(two, true), b = hst.compute(two, true);
  const Deviation e = deviation(a.energy, b.energy), f = deviation(a.forces, b.forces);
  INFO("energy rel " << e.relative() << ", force rel " << f.relative());
  CHECK(e.relative() <= kFp32Noise);
  CHECK(f.relative() <= kFp32Noise);
}
