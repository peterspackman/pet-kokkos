// Neighbour-list construction, independent of any model.
//
// These need no weights, so they run on a fresh clone. They cover the
// preprocessing that reproduces metatrain's structures.py / adaptive_cutoff.py:
// periodic image enumeration, NEF packing, the reverse-edge map, and the
// adaptive cutoff.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <map>
#include <set>
#include <tuple>

#include "pet/neighbors.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// A simple cubic lattice of `n`x`n`x`n` atoms with spacing `a`.
pet::System simple_cubic(int n, double a) {
  pet::System s;
  s.n_atoms = n * n * n;
  s.pbc = {true, true, true};
  s.cell = {n * a, 0, 0, 0, n * a, 0, 0, 0, n * a};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        s.atomic_numbers.push_back(1);
        s.positions.insert(s.positions.end(), {i * a, j * a, k * a});
      }
  return s;
}

pet::Hypers fixed_cutoff_hypers(double cutoff) {
  pet::Hypers h;
  h.cutoff = cutoff;
  h.cutoff_width = 0.5;
  h.cutoff_function = pet::CutoffFunction::Bump;
  h.num_neighbors_adaptive = -1.0;  // disabled
  return h;
}

// Identity species map: Z -> Z, wide enough for hydrogen.
std::vector<int> identity_species(int max_z = 8) {
  std::vector<int> m(max_z + 1);
  for (int z = 0; z <= max_z; ++z) m[z] = z;
  return m;
}

}  // namespace

TEST_CASE("a simple cubic lattice has the coordination number it should", "[neighbors]") {
  // Spacing 2.0 A, cutoff 2.5 A: only the 6 face neighbours are inside. The
  // next shell is at 2*sqrt(2) = 2.83 A.
  const double a = 2.0;
  const pet::System s = simple_cubic(4, a);
  const pet::Hypers h = fixed_cutoff_hypers(2.5);
  const pet::EdgeData ed = pet::build_edge_data(s, h, identity_species());

  REQUIRE(ed.n_atoms == s.n_atoms);
  for (int i = 0; i < ed.n_atoms; ++i) {
    INFO("atom " << i);
    CHECK(ed.num_neigh[i] == 6);
  }

  // Every real edge is exactly one lattice spacing.
  for (int i = 0; i < ed.n_atoms; ++i)
    for (int m = 0; m < ed.max_neighbors; ++m) {
      const std::size_t k = static_cast<std::size_t>(i) * ed.max_neighbors + m;
      if (!ed.mask[k]) continue;
      CHECK_THAT(ed.edge_dist[k], WithinRel(a, 1e-12));
    }
}

TEST_CASE("padding slots are masked and carry a zero cutoff factor", "[neighbors]") {
  // Two atoms with very different environments in one cell, so the NEF has to
  // pad. The padding must be invisible to the network: mask 0 and, critically,
  // cutoff_factor 0, which is what makes the attention softmax skip it.
  pet::System s;
  s.n_atoms = 3;
  s.pbc = {false, false, false};
  s.cell = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.atomic_numbers = {1, 1, 1};
  s.positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0};

  const pet::Hypers h = fixed_cutoff_hypers(3.0);
  const pet::EdgeData ed = pet::build_edge_data(s, h, identity_species());

  REQUIRE(ed.num_neigh[0] == 1);
  REQUIRE(ed.num_neigh[1] == 1);
  REQUIRE(ed.num_neigh[2] == 0);  // 9 A away, outside a 3 A cutoff

  for (int i = 0; i < ed.n_atoms; ++i)
    for (int m = ed.num_neigh[i]; m < ed.max_neighbors; ++m) {
      const std::size_t k = static_cast<std::size_t>(i) * ed.max_neighbors + m;
      INFO("atom " << i << " padding slot " << m);
      CHECK(ed.mask[k] == 0);
      CHECK(ed.cutoff_factor[k] == 0.0);
      CHECK(ed.reverse_index[k] == -1);
    }
}

TEST_CASE("the reverse-edge map points both ways", "[neighbors]") {
  // The backward GATHERS per-edge force gradients through this map instead of
  // scattering them with atomics, so a wrong entry is a wrong force -- and a
  // silently non-deterministic one if it were absent. Every real edge must have
  // a partner, and following the map twice must return to where it started.
  const pet::System s = simple_cubic(3, 2.0);
  const pet::Hypers h = fixed_cutoff_hypers(2.5);
  const pet::EdgeData ed = pet::build_edge_data(s, h, identity_species());
  const int M = ed.max_neighbors;

  int checked = 0;
  for (int i = 0; i < ed.n_atoms; ++i)
    for (int m = 0; m < M; ++m) {
      const std::size_t k = static_cast<std::size_t>(i) * M + m;
      if (!ed.mask[k]) continue;
      const int rev = ed.reverse_index[k];
      INFO("edge (atom " << i << ", slot " << m << ") -> " << rev);
      REQUIRE(rev >= 0);
      REQUIRE(ed.mask[rev] != 0);
      // The partner's own reverse index comes back to this slot.
      CHECK(ed.reverse_index[rev] == static_cast<int>(k));
      // And the partner's edge vector is the negative of this one.
      for (int c = 0; c < 3; ++c)
        CHECK_THAT(ed.edge_vec[static_cast<std::size_t>(rev) * 3 + c],
                   WithinAbs(-ed.edge_vec[k * 3 + c], 1e-12));
      ++checked;
    }
  CHECK(checked == 27 * 6);
}

TEST_CASE("the cutoff function is smooth and vanishes at the cutoff", "[neighbors]") {
  const double rc = 5.0, w = 0.5;
  SECTION("bump") {
    CHECK(pet::detail::bump_cutoff(1.0, rc, w) == 1.0);          // far inside
    CHECK(pet::detail::bump_cutoff(rc - w - 0.01, rc, w) == 1.0);  // just inside the taper
    CHECK(pet::detail::bump_cutoff(rc, rc, w) == 0.0);           // at the cutoff
    CHECK(pet::detail::bump_cutoff(rc + 1.0, rc, w) == 0.0);     // outside
    // Monotone decreasing across the taper.
    double prev = 1.0;
    for (double d = rc - w; d <= rc; d += w / 50.0) {
      const double f = pet::detail::bump_cutoff(d, rc, w);
      CHECK(f <= prev + 1e-15);
      prev = f;
    }
  }
  SECTION("cosine") {
    CHECK(pet::detail::cosine_cutoff(1.0, rc, w) == 1.0);
    CHECK_THAT(pet::detail::cosine_cutoff(rc, rc, w), WithinAbs(0.0, 1e-12));
    CHECK(pet::detail::cosine_cutoff(rc + 1.0, rc, w) == 0.0);
  }
}

TEST_CASE("the adaptive cutoff tracks the target neighbour count", "[neighbors]") {
  // A sparse structure and a dense one, with the same hypers. The adaptive
  // scheme exists so both see roughly num_neighbors_adaptive neighbours, which
  // a fixed cutoff cannot do.
  pet::Hypers h = fixed_cutoff_hypers(7.5);
  h.num_neighbors_adaptive = 8.0;
  REQUIRE(h.adaptive());

  const auto species = identity_species();
  const pet::EdgeData dense = pet::build_edge_data(simple_cubic(4, 2.0), h, species);
  const pet::EdgeData sparse = pet::build_edge_data(simple_cubic(4, 3.5), h, species);

  auto mean_neigh = [](const pet::EdgeData& ed) {
    double t = 0;
    for (int i = 0; i < ed.n_atoms; ++i) t += ed.num_neigh[i];
    return t / ed.n_atoms;
  };

  const double d = mean_neigh(dense), sp = mean_neigh(sparse);
  INFO("dense mean neighbours " << d << ", sparse " << sp);
  // Not exact -- the scheme targets a SMOOTHED count, and the hard count within
  // the chosen radius sits near it rather than on it. The point is that the two
  // land in the same ballpark where a fixed 7.5 A cutoff would differ several-fold.
  CHECK(d > 4.0);
  CHECK(d < 20.0);
  CHECK(sp > 4.0);
  CHECK(sp < 20.0);
}

TEST_CASE("an isolated molecule is not wrapped into a phantom cell", "[neighbors]") {
  // A non-periodic system has no cell, and enumerating images of one would tear
  // the molecule apart. The distances must be the plain Cartesian ones.
  pet::System s;
  s.n_atoms = 3;
  s.pbc = {false, false, false};
  s.cell = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.atomic_numbers = {8, 1, 1};
  s.positions = {0.0, 0.0, 0.11926, 0.0, 0.76323, -0.47704, 0.0, -0.76323, -0.47704};

  const pet::Hypers h = fixed_cutoff_hypers(5.0);
  const pet::EdgeData ed = pet::build_edge_data(s, h, identity_species());

  REQUIRE(ed.num_neigh[0] == 2);
  const double oh = std::sqrt(0.76323 * 0.76323 + 0.5963 * 0.5963);
  for (int m = 0; m < 2; ++m) CHECK_THAT(ed.edge_dist[m], WithinAbs(oh, 1e-4));
}

TEST_CASE("concatenating edge data preserves every structure's edges", "[neighbors]") {
  // The batched path concatenates per-structure NEFs into one global index
  // space. Atom-index references (reverse_index, raw_center/raw_neigh) are
  // re-encoded in the combined M, and getting that wrong silently mixes
  // structures together.
  const auto species = identity_species();
  const pet::Hypers h = fixed_cutoff_hypers(2.5);
  const pet::EdgeData a = pet::build_edge_data(simple_cubic(2, 2.0), h, species);
  const pet::EdgeData b = pet::build_edge_data(simple_cubic(3, 2.0), h, species);

  std::vector<int> struct_id;
  const pet::EdgeData c = pet::concat_edge_data({a, b}, struct_id);

  REQUIRE(c.n_atoms == a.n_atoms + b.n_atoms);
  REQUIRE(static_cast<int>(struct_id.size()) == c.n_atoms);
  CHECK(c.max_neighbors == std::max(a.max_neighbors, b.max_neighbors));
  for (int i = 0; i < a.n_atoms; ++i) CHECK(struct_id[i] == 0);
  for (int i = a.n_atoms; i < c.n_atoms; ++i) CHECK(struct_id[i] == 1);

  // Neighbour counts survive, and no edge crosses the boundary.
  for (int i = 0; i < a.n_atoms; ++i) CHECK(c.num_neigh[i] == a.num_neigh[i]);
  for (int i = 0; i < b.n_atoms; ++i) CHECK(c.num_neigh[a.n_atoms + i] == b.num_neigh[i]);
  for (int e = 0; e < c.n_raw; ++e) {
    const bool ci = c.raw_center[e] < a.n_atoms;
    const bool nj = c.raw_neigh[e] < a.n_atoms;
    INFO("raw edge " << e);
    CHECK(ci == nj);
  }
  // And the re-encoded reverse map still points at a real edge of the same atom.
  for (int i = 0; i < c.n_atoms; ++i)
    for (int m = 0; m < c.max_neighbors; ++m) {
      const std::size_t k = static_cast<std::size_t>(i) * c.max_neighbors + m;
      if (!c.mask[k]) continue;
      REQUIRE(c.reverse_index[k] >= 0);
      CHECK(c.reverse_index[c.reverse_index[k]] == static_cast<int>(k));
    }
}
