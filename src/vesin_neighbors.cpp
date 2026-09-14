#include "pet/vesin_neighbors.hpp"

#include "pet/calculator.hpp"
#include "pet/neighbors.hpp"

#ifdef PET_HAVE_VESIN
#include <vesin.h>
#endif

#include <cstdlib>
#include <string>

#include <algorithm>
#include <cstring>
#include <stdexcept>

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
    return NeighborBackend::Vesin;  // used only when compiled in; see dispatch
  }();
  return b;
}

namespace detail {

// Route to whichever search is selected AND available. Asking for vesin in a
// build without it is not an error -- the two are meant to agree, so falling
// back silently is the right behaviour and `pet::vesin_available()` is how a
// caller finds out which it got.
std::vector<RawEdge> build_raw_edges_dispatch(const System& sys, double cutoff) {
#ifdef PET_HAVE_VESIN
  if (neighbor_backend() == NeighborBackend::Vesin) return build_raw_edges_vesin(sys, cutoff);
#endif
  return build_raw_edges(sys, cutoff);
}

#ifdef PET_HAVE_VESIN

namespace {

// vesin allocates into a VesinNeighborList and expects vesin_free to release it.
// Scoped so an exception between the call and the copy-out cannot leak it.
struct ListGuard {
  VesinNeighborList list{};
  ~ListGuard() { vesin_free(&list); }
};

}  // namespace

std::vector<RawEdge> build_raw_edges_vesin(const System& sys, double cutoff) {
  const int N = sys.n_atoms;
  std::vector<RawEdge> edges;
  if (N == 0) return edges;

  // vesin takes the cell with lattice vectors as ROWS, which is how System
  // stores it, so this is a straight copy rather than a transpose.
  double box[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) box[r][c] = sys.cell[r * 3 + c];
  bool periodic[3] = {sys.pbc[0], sys.pbc[1], sys.pbc[2]};

  // A system periodic in some directions but not others is not something vesin
  // models as a per-axis flag in the same way the built-in search does, and PET
  // itself only ever sees fully periodic or fully isolated systems. Refuse the
  // mixed case rather than quietly treating it as one or the other.
  const bool any = periodic[0] || periodic[1] || periodic[2];
  const bool all = periodic[0] && periodic[1] && periodic[2];
  if (any && !all)
    throw std::runtime_error(
        "pet: the vesin neighbour backend does not handle partially periodic systems; "
        "build with PET_WITH_VESIN=OFF or set Options::neighbors to builtin");

  VesinOptions opts{};
  opts.cutoff = cutoff;
  opts.full = true;    // both i->j and j->i; the NEF's reverse-edge map needs it
  opts.sorted = true;  // grouped by i, which is what the per-atom scan assumes
  opts.algorithm = VesinAutoAlgorithm;
  opts.skin = 0.0;     // caching is handled a level up, not here
  opts.n_threads = 0;  // vesin's own default
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

  // Canonical order. vesin guarantees grouping by `i` and explicitly leaves the
  // order within a group unspecified; it is also threaded, so that order is not
  // something to depend on across runs, builds or thread counts. The NEF packer
  // assigns neighbour slots in list order and the slot order fixes the attention
  // softmax's summation order, so an unstable order here is an unstable energy
  // in the last bits -- which is precisely what the determinism work removed
  // everywhere else. A stable sort on (i, j, shift) costs one pass and makes the
  // result reproducible by construction.
  std::sort(edges.begin(), edges.end(), [](const RawEdge& a, const RawEdge& b) {
    if (a.i != b.i) return a.i < b.i;
    if (a.j != b.j) return a.j < b.j;
    if (a.sa != b.sa) return a.sa < b.sa;
    if (a.sb != b.sb) return a.sb < b.sb;
    return a.sc < b.sc;
  });
  return edges;
}

#endif  // PET_HAVE_VESIN

}  // namespace detail
}  // namespace pet
