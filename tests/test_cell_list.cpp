// The device cell list against the brute-force search: the same edges, which
// the edge sets show directly, and through a whole evaluation, equal to fp32
// noise (the order differs). Cases are the ways a cell list goes wrong: a box
// narrower than the cutoff (one bin, several images), two bins (one bin reached
// with two shifts), a skewed cell (spacing well below the vector length), a
// mixed batch.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/device_cell_list.hpp"
#include "test_support.hpp"

using namespace pet_test;

namespace {

// n^3 cells of an 8-atom diamond cube, optionally sheared so the cell is
// strongly non-orthogonal.
pet::System diamond(int n, double a, double shear = 0.0) {
  pet::System s;
  const double base[8][3] = {{0, 0, 0},       {0, .5, .5},    {.5, 0, .5},    {.5, .5, 0},
                             {.25, .25, .25}, {.25, .75, .75}, {.75, .25, .75}, {.75, .75, .25}};
  const double L = n * a;
  s.pbc = {true, true, true};
  // Tilting c toward a shrinks the interplanar spacing but no vector's length.
  s.cell = {L, 0, 0, 0, L, 0, shear * L, 0, L};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        for (int t = 0; t < 8; ++t) {
          const double fa = (i + base[t][0]) / n, fb = (j + base[t][1]) / n,
                       fc = (k + base[t][2]) / n;
          s.atomic_numbers.push_back(6);
          s.positions.push_back(fa * s.cell[0] + fb * s.cell[3] + fc * s.cell[6]);
          s.positions.push_back(fa * s.cell[1] + fb * s.cell[4] + fc * s.cell[7]);
          s.positions.push_back(fa * s.cell[2] + fb * s.cell[5] + fc * s.cell[8]);
        }
  s.n_atoms = (int) s.atomic_numbers.size();
  return s;
}

struct SearchGuard {
  pet::DeviceSearch saved = pet::device_search();
  ~SearchGuard() { pet::device_search() = saved; }
  void use(pet::DeviceSearch s) { pet::device_search() = s; }
};

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    m = worst(m, std::fabs(a[i] - b[i]));
  return m;
}

using Edge = std::tuple<int, int, int, int, int, int>;  // structure, i, j, image shift

// The device cell list's edges for `systems`, within `cutoff`, as (structure,
// local i, local j, shift), in the order it lists them. diamond() puts every
// atom inside its cell, so the staged positions are already the wrapped ones.
std::vector<Edge> cell_list_edges(const std::vector<pet::System>& systems, double cutoff) {
  pet::Workspace ws;
  std::vector<int> species_to_index(7, -1);
  species_to_index[6] = 0;
  const pet::DeviceGeom g = pet::stage_systems(ws, systems, species_to_index);
  pet::RView2D cinv("cinv", g.B, 9);
  auto hc = Kokkos::create_mirror_view(cinv);
  for (int b = 0; b < g.B; ++b) {  // rows are lattice vectors: fractional = r H^-1
    const auto& h = systems[b].cell;
    const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]);
    const double inv[9] = {(h[4] * h[8] - h[5] * h[7]) / det, (h[2] * h[7] - h[1] * h[8]) / det,
                           (h[1] * h[5] - h[2] * h[4]) / det, (h[5] * h[6] - h[3] * h[8]) / det,
                           (h[0] * h[8] - h[2] * h[6]) / det, (h[2] * h[3] - h[0] * h[5]) / det,
                           (h[3] * h[7] - h[4] * h[6]) / det, (h[1] * h[6] - h[0] * h[7]) / det,
                           (h[0] * h[4] - h[1] * h[3]) / det};
    for (int k = 0; k < 9; ++k) hc(b, k) = inv[k];
  }
  Kokkos::deep_copy(cinv, hc);
  const pet::RawEdges re = pet::build_raw_edges_cells(ws, g, g.pos, cinv, cutoff);
  auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), re.i);
  auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), re.j);
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), re.shift);
  std::vector<int> first(systems.size() + 1, 0), owner;
  for (std::size_t b = 0; b < systems.size(); ++b) {
    first[b + 1] = first[b] + systems[b].n_atoms;
    owner.insert(owner.end(), systems[b].n_atoms, (int) b);
  }
  std::vector<Edge> out;
  for (int e = 0; e < re.count; ++e) {
    const int b = owner[hi(e)];
    out.emplace_back(b, hi(e) - first[b], hj(e) - first[b], hs(e, 0), hs(e, 1), hs(e, 2));
  }
  return out;
}

// The host brute-force search's, as a set.
std::set<Edge> brute_force_edges(const std::vector<pet::System>& systems, double cutoff) {
  std::set<Edge> out;
  for (std::size_t b = 0; b < systems.size(); ++b)
    for (const auto& e : pet::detail::build_raw_edges(systems[b], cutoff))
      out.emplace((int) b, e.i, e.j, e.sa, e.sb, e.sc);
  return out;
}

constexpr double kCutoff = 7.5;  // PET-MAD's raw cutoff

struct Setup {
  const char* name;
  std::vector<pet::System> systems;
};

std::vector<Setup> setups() {
  std::vector<Setup> s;
  // 2x2x2 of a 3.567 A cube is 7.13 A across against a 7.5 A cutoff: the grid
  // collapses to a single bin per axis and the search must reach past it.
  s.push_back({"box narrower than the cutoff", {diamond(2, 3.567)}});
  // 3x3x3 is 10.7 A: two bins per axis, so a bin is reached twice with
  // different shifts and must count as two distinct images.
  s.push_back({"two bins per axis", {diamond(3, 3.567)}});
  // 5x5x5 is comfortably multi-bin, the regime the grid is actually for.
  s.push_back({"many bins", {diamond(5, 3.567)}});
  // Sheared, and big enough to stay multi-bin (4x4x4 would collapse to one bin).
  s.push_back({"strongly skewed cell", {diamond(6, 3.567, 0.5)}});
  // A batch of different sizes, which exercises the per-structure grid offsets.
  s.push_back({"mixed batch", {diamond(2, 3.567), diamond(4, 3.567), diamond(3, 3.567)}});
  return s;
}

}  // namespace

TEST_CASE("the device cell list finds exactly the brute-force search's edges", "[neighbors][cells]") {
  for (const auto& s : setups()) {
    DYNAMIC_SECTION(s.name) {
      const std::vector<Edge> cells = cell_list_edges(s.systems, kCutoff);
      const std::set<Edge> brute = brute_force_edges(s.systems, kCutoff);
      const std::set<Edge> cell_set(cells.begin(), cells.end());
      INFO(cells.size() << " cell-list edges, " << cell_set.size() << " distinct, " << brute.size()
                        << " by brute force");
      CHECK(cells.size() == cell_set.size());  // none listed twice
      CHECK(cell_set == brute);
    }
  }
}

TEST_CASE("the cell list is reproducible run to run", "[neighbors][cells][determinism]") {
  // Bins fill through an atomic counter; the per-bin sort makes the order fixed.
  const std::vector<pet::System> sys = {diamond(4, 3.567), diamond(3, 3.567)};
  const std::vector<Edge> first = cell_list_edges(sys, kCutoff);
  for (int k = 0; k < 3; ++k) REQUIRE(cell_list_edges(sys, kCutoff) == first);
}

TEST_CASE("an evaluation is the same on the cell list as by brute force", "[model][cells]") {
  const auto found = find_model("pet-mad-xs");
  if (!found) {
    WARN("model 'pet-mad-xs' is not installed; skipping the cell-list evaluation");
    return;
  }
  // A small mixed batch: the edge sets above are the thorough check, this the
  // whole pipeline on top of them.
  const std::vector<pet::System> sys = {diamond(2, 3.567), diamond(3, 3.567)};
  pet::Calculator calc(found->first, found->second);
  SearchGuard guard;
  auto run = [&](pet::DeviceSearch mode) {
    guard.use(mode);
    return calc.compute(sys, /*compute_forces=*/true);
  };
  const pet::Results brute = run(pet::DeviceSearch::BruteForce);
  const pet::Results cells = run(pet::DeviceSearch::CellList);
  REQUIRE(brute.energy.size() == cells.energy.size());
  REQUIRE(brute.forces.size() == cells.forces.size());
  REQUIRE_FALSE(brute.forces.empty());

  double escale = 0.0, fscale = 0.0;
  for (double e : brute.energy) escale = worst(escale, std::fabs(e));
  for (double f : brute.forces) fscale = worst(fscale, std::fabs(f));
  const double de = max_abs_diff(brute.energy, cells.energy);
  const double df = max_abs_diff(brute.forces, cells.forces);
  const double dw = max_abs_diff(brute.virial, cells.virial);
  INFO("dE = " << de << " (scale " << escale << "), dF = " << df << " (scale " << fscale << "), dW = " << dw);
  CHECK(de <= 1e-5 * std::max(escale, 1.0));
  CHECK(df <= 1e-5 * std::max(fscale, 1.0));
  CHECK(dw <= 1e-5 * std::max(escale, 1.0));
}
