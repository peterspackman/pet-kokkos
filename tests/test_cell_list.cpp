// The device cell list against the brute-force search it replaces.
//
// These compare the two device searches through a whole evaluation, because the
// raw edge list is internal to build_nef_device and the thing that actually
// matters is that the energy and forces come out the same. The edge SET has to
// be identical; the ORDER legitimately differs (bins and shifts versus images
// and atoms), so the comparison is to fp32 noise, exactly as for the host and
// device builders.
//
// The cases are chosen for the ways a cell list goes wrong rather than for
// realism: a box narrower than the cutoff (where the grid collapses to one bin
// and the search has to reach several images), a two-bin box (where the same bin
// is reached twice with different shifts), a strongly skewed cell (where the
// interplanar spacing is much smaller than the lattice vector length), and a
// batch mixing sizes.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
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
  // Row-major, lattice vectors as rows. The shear tilts c toward a, which
  // shrinks the interplanar spacing without changing any vector's length --
  // precisely the case a grid sized from |c| rather than the spacing gets wrong.
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

}  // namespace

TEST_CASE("the device cell list agrees with the brute-force search", "[model][cells]") {
  const auto found = find_model("pet-mad-xs");
  if (!found) {
    WARN("model 'pet-mad-xs' is not installed; skipping the cell-list comparison");
    return;
  }

  struct Setup {
    const char* name;
    std::vector<pet::System> systems;
  };
  std::vector<Setup> setups;
  // 2x2x2 of a 3.567 A cube is 7.13 A across against a 7.5 A cutoff: the grid
  // collapses to a single bin per axis and the search must reach past it.
  setups.push_back({"box narrower than the cutoff", {diamond(2, 3.567)}});
  // 3x3x3 is 10.7 A: two bins per axis, so a bin is reached twice with
  // different shifts and must count as two distinct images.
  setups.push_back({"two bins per axis", {diamond(3, 3.567)}});
  // 5x5x5 is comfortably multi-bin, the regime the grid is actually for.
  setups.push_back({"many bins", {diamond(5, 3.567)}});
  // Sheared, so the interplanar spacing is well below the vector length. Sized
  // to be genuinely multi-bin: at 4x4x4 the spacing is still under two cutoffs
  // and the grid collapses to one bin, which would make this a duplicate of the
  // narrow-box case rather than a test of the skew.
  setups.push_back({"strongly skewed cell", {diamond(6, 3.567, 0.5)}});
  // A batch of different sizes, which exercises the per-structure grid offsets.
  setups.push_back({"mixed batch", {diamond(2, 3.567), diamond(4, 3.567), diamond(3, 3.567)}});

  pet::Calculator calc(found->first, found->second);
  SearchGuard guard;

  for (const auto& s : setups) {
    DYNAMIC_SECTION(s.name) {
      auto run = [&](pet::DeviceSearch mode) {
        guard.use(mode);
        // A vector of two or more goes through the device batch path, which is
        // what build_nef_device serves; a single structure takes the host path
        // and would not exercise this at all.
        std::vector<pet::System> sys = s.systems;
        if (sys.size() == 1) sys.push_back(sys[0]);
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
      INFO("dE = " << de << " (scale " << escale << "), dF = " << df << " (scale " << fscale
                   << "), dW = " << dw);
      // A missing or duplicated edge moves the energy far more than this. The
      // bound is the fp32 network's own resolution, which is all the difference
      // in edge ordering can produce.
      CHECK(de <= 1e-5 * std::max(escale, 1.0));
      CHECK(df <= 1e-5 * std::max(fscale, 1.0));
      CHECK(dw <= 1e-5 * std::max(escale, 1.0));
    }
  }
}

TEST_CASE("the cell list is reproducible run to run", "[model][cells][determinism]") {
  // Bin membership is filled with an atomic counter, so without the per-bin sort
  // the neighbour order -- and the energy's last bits -- would vary between runs.
  // This is the test that the sort is doing its job.
  const auto found = find_model("pet-mad-xs");
  if (!found) {
    WARN("model 'pet-mad-xs' is not installed; skipping");
    return;
  }
  pet::Calculator calc(found->first, found->second);
  SearchGuard guard;
  guard.use(pet::DeviceSearch::CellList);

  const std::vector<pet::System> sys = {diamond(4, 3.567), diamond(3, 3.567)};
  const pet::Results first = calc.compute(sys, true);
  for (int k = 0; k < 3; ++k) {
    const pet::Results again = calc.compute(sys, true);
    REQUIRE(first.energy.size() == again.energy.size());
    for (std::size_t i = 0; i < first.energy.size(); ++i) REQUIRE(first.energy[i] == again.energy[i]);
    for (std::size_t i = 0; i < first.forces.size(); ++i) REQUIRE(first.forces[i] == again.forces[i]);
  }
}
