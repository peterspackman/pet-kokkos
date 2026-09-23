#include "pet/vesin_neighbors.hpp"

#include "pet/calculator.hpp"
#include "pet/neighbors.hpp"

#ifdef PET_HAVE_VESIN
#include <vesin.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pet {

bool vesin_available() {
#ifdef PET_HAVE_VESIN
  return true;
#else
  return false;
#endif
}

NeighborBackend& neighbor_backend() {
  static NeighborBackend b = [] {
    const char* e = std::getenv("PET_NEIGHBORS");
    if (e && std::string(e) == "builtin") return NeighborBackend::Builtin;
    return NeighborBackend::Vesin;
  }();
  return b;
}

namespace detail {

// Vesin if selected and built in, else the built-in search.
std::vector<RawEdge> build_raw_edges_dispatch(const System& sys, double cutoff) {
#ifdef PET_HAVE_VESIN
  if (neighbor_backend() == NeighborBackend::Vesin) return build_raw_edges_vesin(sys, cutoff);
#endif
  return build_raw_edges(sys, cutoff);
}

#ifdef PET_HAVE_VESIN

namespace {

struct ListGuard {
  VesinNeighborList list{};
  ~ListGuard() { vesin_free(&list); }
};

}  // namespace

std::vector<RawEdge> build_raw_edges_vesin(const System& sys, double cutoff) {
  const int N = sys.n_atoms;
  std::vector<RawEdge> edges;
  if (N == 0) return edges;

  // Lattice vectors as rows, like System.
  double box[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) box[r][c] = sys.cell[r * 3 + c];
  bool periodic[3] = {sys.pbc[0], sys.pbc[1], sys.pbc[2]};

  // Mixed periodicity is refused rather than guessed at.
  const bool any = periodic[0] || periodic[1] || periodic[2];
  const bool all = periodic[0] && periodic[1] && periodic[2];
  if (any && !all)
    throw std::runtime_error(
        "pet: the vesin neighbour backend does not handle partially periodic systems; "
        "build with PET_WITH_VESIN=OFF or set Options::neighbors to builtin");

  VesinOptions opts{};
  opts.cutoff = cutoff;
  opts.full = true;    // both i->j and j->i, for the reverse map
  opts.sorted = true;  // grouped by i
  opts.algorithm = VesinAutoAlgorithm;
  opts.skin = 0.0;
  opts.n_threads = 0;
  opts.return_shifts = true;
  opts.return_distances = true;
  opts.return_vectors = true;

  VesinDevice device{VesinCPU, 0};
  ListGuard g;
  const char* err = nullptr;
  const int rc = vesin_neighbors(reinterpret_cast<const double(*)[3]>(sys.positions.data()),
                                 static_cast<std::size_t>(N), box, periodic, device, opts, &g.list,
                                 &err);
  if (rc != 0)
    throw std::runtime_error(std::string("pet: vesin_neighbors failed: ") +
                             (err ? err : "(no message)"));

  edges.resize(g.list.length);
  for (std::size_t e = 0; e < g.list.length; ++e) {
    RawEdge& o = edges[e];
    o.i = static_cast<int>(g.list.pairs[e][0]);
    o.j = static_cast<int>(g.list.pairs[e][1]);
    o.sa = g.list.shifts[e][0];
    o.sb = g.list.shifts[e][1];
    o.sc = g.list.shifts[e][2];
    o.vx = g.list.vectors[e][0];
    o.vy = g.list.vectors[e][1];
    o.vz = g.list.vectors[e][2];
    o.dist = g.list.distances[e];
  }

  std::sort(edges.begin(), edges.end(), [](const RawEdge& a, const RawEdge& b) {
    return std::tie(a.i, a.j, a.sa, a.sb, a.sc) < std::tie(b.i, b.j, b.sa, b.sb, b.sc);
  });
  return edges;
}

#endif  // PET_HAVE_VESIN

}  // namespace detail
}  // namespace pet
