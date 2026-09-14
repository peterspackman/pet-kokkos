// Host-side neighbor-list construction, adaptive cutoff, and neighbor-edge-format
// (NEF) packing. This reproduces metatrain's structures.py / adaptive_cutoff.py
// preprocessing so the network sees identical inputs.
//
// This host path builds the full neighbour list itself, brute force over periodic
// images. It backs the single-structure entry points and the goldens. A caller
// running many evaluations wants device_neighbors.hpp instead, which shares the
// NEF packing and adaptive-cutoff logic but never touches the host.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "pet/config.hpp"

namespace pet {

// A single structure to evaluate.
struct System {
  int n_atoms = 0;
  std::vector<int> atomic_numbers;     // [N]
  std::vector<double> positions;       // [N*3] cartesian, Angstrom
  std::array<double, 9> cell{};        // row-major lattice vectors a,b,c (rows)
  std::array<bool, 3> pbc{false, false, false};

  // Electronic state, for a model trained with system_conditioning. Ignored by
  // every other model. The defaults are metatrain's own fallback for a system
  // that carries no such information: neutral singlet.
  int charge = 0;
  int spin_multiplicity = 1;  // 2S+1, so 1 is a singlet
};

// Edges in neighbor-edge-format. Row-major NEF arrays are [N, M] (or [N, M, 3]).
struct EdgeData {
  int n_atoms = 0;
  int max_neighbors = 0;  // M
  std::vector<int> species;            // [N] species index
  std::vector<int> num_neigh;          // [N]
  std::vector<double> edge_vec;        // [N*M*3]  (r_j - r_i + shift)
  std::vector<double> edge_dist;       // [N*M]
  std::vector<int> neigh_species;      // [N*M]
  std::vector<double> cutoff_factor;   // [N*M]
  std::vector<double> pair_cutoff;     // [N*M] per-edge cutoff radius (adaptive or fixed)
  std::vector<char> mask;              // [N*M] 1=real, 0=padding
  std::vector<int> reverse_index;      // [N*M] flat (j*M+slot) of reverse edge, -1 if none

  // Per-STRUCTURE electronic state for a conditioned model, in the order the
  // structures were concatenated (length 1 for a single system). Empty when the
  // model is not conditioned.
  std::vector<int> charge;
  std::vector<int> spin_multiplicity;

  // SOLVER adaptive cutoff only: [N] the root of the smoothed neighbour count and
  // [N] dn/dr there, both computed by the forward and consumed by the backward's
  // implicit-function step. Empty for the grid method and for a fixed cutoff. A
  // zero in adapt_dn marks an atom whose cutoff hit a clamp bound (no gradient).
  //
  // They are carried rather than recomputed for the same reason the device build
  // keeps them: the backward would otherwise redo the entire Newton solve, and it
  // would have to do it with atomics (it has no per-atom edge segmentation of its
  // own), which would put run-to-run noise into a quantity the forces depend on.
  std::vector<double> adapt_r;
  std::vector<double> adapt_dn;

  // Full raw edge list within cutoff (BEFORE adaptive masking). Needed for the
  // adaptive-cutoff chain rule in the backward pass, since the per-atom adaptive
  // cutoff depends on every neighbor within the search radius, including edges
  // that are subsequently dropped.
  int n_raw = 0;
  std::vector<int> raw_center;          // [E]
  std::vector<int> raw_neigh;           // [E]
  std::vector<double> raw_dist;         // [E]
  std::vector<double> raw_vec;          // [E*3]
};

// Concatenate per-structure EdgeData into one combined EdgeData for batched
// evaluation. Atoms are concatenated into a single global index space; NEF rows
// are padded to M = max over structures (extra slots are masked padding, exactly
// as a sparse atom already is); every atom-index reference (reverse_index,
// raw_center/raw_neigh) is offset by the running atom count. Edges
// never cross a structure boundary, so PET's per-atom network produces identical
// per-atom energies/forces -- only the energy reduction and virial are
// per-structure. `struct_id[atom]` (filled, length N) records the owning
// structure. The padded keys carry cutoff_factor=0, so the attention softmax skips
// them and real neighbor slots stay in the same order -> results match the
// per-structure runs to round-off.
inline EdgeData concat_edge_data(const std::vector<EdgeData>& parts, std::vector<int>& struct_id) {
  EdgeData out;
  int N = 0, M = 0, E = 0;
  for (const auto& p : parts) {
    N += p.n_atoms;
    M = std::max(M, p.max_neighbors);
    E += p.n_raw;
  }
  out.n_atoms = N;
  out.max_neighbors = M;
  out.n_raw = E;
  const std::size_t NM = (std::size_t) N * M;
  out.species.resize(N);
  out.num_neigh.resize(N);
  out.edge_vec.assign(NM * 3, 0.0);
  out.edge_dist.assign(NM, 0.0);
  out.neigh_species.assign(NM, 0);
  out.cutoff_factor.assign(NM, 0.0);
  out.pair_cutoff.assign(NM, 0.0);
  out.mask.assign(NM, 0);
  out.reverse_index.assign(NM, -1);
  out.raw_center.resize(E);
  out.raw_neigh.resize(E);
  out.raw_dist.resize(E);
  out.raw_vec.resize((std::size_t) E * 3);
  struct_id.assign(N, 0);

  // Solver adaptive state, when the parts carry it. Per-atom and independent of
  // any other structure, so concatenation is a straight copy into the global atom
  // index space -- but it does have to happen: the backward cannot rebuild it,
  // and dropping it here is how a batched solver evaluation ends up with no
  // gradient path at all.
  const bool have_adapt = !parts.empty() && !parts.front().adapt_r.empty();
  if (have_adapt) {
    out.adapt_r.assign(N, 0.0);
    out.adapt_dn.assign(N, 0.0);
  }
  // Per-structure, so concatenation appends rather than copies per atom.
  for (const auto& p : parts) {
    out.charge.insert(out.charge.end(), p.charge.begin(), p.charge.end());
    out.spin_multiplicity.insert(out.spin_multiplicity.end(), p.spin_multiplicity.begin(),
                                 p.spin_multiplicity.end());
  }

  int off_n = 0, off_e = 0;
  for (std::size_t b = 0; b < parts.size(); ++b) {
    const EdgeData& p = parts[b];
    const int Mb = p.max_neighbors;
    for (int i = 0; i < p.n_atoms; ++i) {
      const int gi = off_n + i;
      out.species[gi] = p.species[i];
      out.num_neigh[gi] = p.num_neigh[i];
      struct_id[gi] = (int) b;
      if (have_adapt && i < (int) p.adapt_r.size()) {
        out.adapt_r[gi] = p.adapt_r[i];
        out.adapt_dn[gi] = p.adapt_dn[i];
      }
      for (int m = 0; m < Mb; ++m) {
        const std::size_t src = (std::size_t) i * Mb + m;
        const std::size_t dst = (std::size_t) gi * M + m;
        out.mask[dst] = p.mask[src];
        out.neigh_species[dst] = p.neigh_species[src];
        out.edge_dist[dst] = p.edge_dist[src];
        out.cutoff_factor[dst] = p.cutoff_factor[src];
        out.pair_cutoff[dst] = p.pair_cutoff[src];
        out.edge_vec[dst * 3 + 0] = p.edge_vec[src * 3 + 0];
        out.edge_vec[dst * 3 + 1] = p.edge_vec[src * 3 + 1];
        out.edge_vec[dst * 3 + 2] = p.edge_vec[src * 3 + 2];
        const int rev = p.reverse_index[src];
        if (rev >= 0)  // decode in the part's M, re-encode in the combined M
          out.reverse_index[dst] = (off_n + rev / Mb) * M + (rev % Mb);
      }
    }
    for (int e = 0; e < p.n_raw; ++e) {
      const int ge = off_e + e;
      out.raw_center[ge] = p.raw_center[e] + off_n;
      out.raw_neigh[ge] = p.raw_neigh[e] + off_n;
      out.raw_dist[ge] = p.raw_dist[e];
      out.raw_vec[(std::size_t) ge * 3 + 0] = p.raw_vec[(std::size_t) e * 3 + 0];
      out.raw_vec[(std::size_t) ge * 3 + 1] = p.raw_vec[(std::size_t) e * 3 + 1];
      out.raw_vec[(std::size_t) ge * 3 + 2] = p.raw_vec[(std::size_t) e * 3 + 2];
    }
    off_n += p.n_atoms;
    off_e += p.n_raw;
  }
  return out;
}

namespace detail {

inline double bump_cutoff(double d, double rc, double width) {
  const double scaled = (d - (rc - width)) / width;
  if (scaled <= 0.0) return 1.0;
  if (scaled >= 1.0) return 0.0;
  return 0.5 * (1.0 + std::tanh(1.0 / std::tan(M_PI * scaled)));
}

inline double cosine_cutoff(double d, double rc, double width) {
  const double scaled = (d - (rc - width)) / width;
  const double x = std::min(std::max(scaled, 0.0), 1.0);
  return 0.5 * (1.0 + std::cos(M_PI * x));
}

inline double cutoff_value(const Hypers& h, double d, double rc) {
  return h.cutoff_function == CutoffFunction::Bump ? bump_cutoff(d, rc, h.cutoff_width)
                                                   : cosine_cutoff(d, rc, h.cutoff_width);
}

// A raw directed edge before NEF packing.
struct RawEdge {
  int i, j;
  int sa, sb, sc;       // integer cell shift applied to j
  double vx, vy, vz;    // r_j + shift - r_i
  double dist;
};

inline std::array<int, 3> image_ranges(const System& sys, double cutoff) {
  std::array<int, 3> n{0, 0, 0};
  const auto& c = sys.cell;
  // lattice vectors (rows)
  const double a[3] = {c[0], c[1], c[2]};
  const double b[3] = {c[3], c[4], c[5]};
  const double cc[3] = {c[6], c[7], c[8]};
  auto cross = [](const double* u, const double* v, double* w) {
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
  };
  auto norm = [](const double* u) { return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
  double bc[3], ca[3], ab[3];
  cross(b, cc, bc);
  cross(cc, a, ca);
  cross(a, b, ab);
  const double vol = std::fabs(a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]);
  if (vol < 1e-12) return n;  // degenerate / non-periodic cell
  const double da = vol / norm(bc);  // interplanar spacing perpendicular to a
  const double db = vol / norm(ca);
  const double dc = vol / norm(ab);
  if (sys.pbc[0]) n[0] = static_cast<int>(std::ceil(cutoff / da));
  if (sys.pbc[1]) n[1] = static_cast<int>(std::ceil(cutoff / db));
  if (sys.pbc[2]) n[2] = static_cast<int>(std::ceil(cutoff / dc));
  return n;
}

// Full, strict neighbor list within `cutoff` (matches vesin full_list=True).
inline std::vector<RawEdge> build_raw_edges(const System& sys, double cutoff) {
  const int N = sys.n_atoms;
  const auto& c = sys.cell;
  const auto rng = image_ranges(sys, cutoff);
  const double cutoff2 = cutoff * cutoff;
  std::vector<RawEdge> edges;
  edges.reserve(static_cast<std::size_t>(N) * 16);

  for (int i = 0; i < N; ++i) {
    const double xi = sys.positions[3 * i + 0];
    const double yi = sys.positions[3 * i + 1];
    const double zi = sys.positions[3 * i + 2];
    for (int j = 0; j < N; ++j) {
      const double xj = sys.positions[3 * j + 0];
      const double yj = sys.positions[3 * j + 1];
      const double zj = sys.positions[3 * j + 2];
      for (int sa = -rng[0]; sa <= rng[0]; ++sa)
        for (int sb = -rng[1]; sb <= rng[1]; ++sb)
          for (int sc = -rng[2]; sc <= rng[2]; ++sc) {
            const double shx = sa * c[0] + sb * c[3] + sc * c[6];
            const double shy = sa * c[1] + sb * c[4] + sc * c[7];
            const double shz = sa * c[2] + sb * c[5] + sc * c[8];
            const double vx = xj + shx - xi;
            const double vy = yj + shy - yi;
            const double vz = zj + shz - zi;
            const double d2 = vx * vx + vy * vy + vz * vz;
            if (d2 < 1e-24) continue;       // skip the self/zero edge
            if (d2 > cutoff2) continue;
            RawEdge e;
            e.i = i;
            e.j = j;
            e.sa = sa; e.sb = sb; e.sc = sc;
            e.vx = vx; e.vy = vy; e.vz = vz;
            e.dist = std::sqrt(d2);
            edges.push_back(e);
          }
    }
  }
  return edges;
}

}  // namespace detail

// A raw directed edge supplied to build_edge_data_from_raw. `center` and
// `neigh_owner` are local atom indices in [0,N) (neigh_owner is the *owner* of a
// neighbor, i.e. the local atom a periodic/MPI ghost is an image of — used for
// adaptive-cutoff lookup and force accumulation). `tag_*` are global atom ids,
// used only to match an edge with its reverse (j->i).
struct RawEdgeIn {
  int center;
  int neigh_owner;
  long long tag_center;
  long long tag_neigh;
  int sa, sb, sc;     // integer cell shift of the neighbor image (for reverse matching)
  double vx, vy, vz;  // r_j(+image) - r_i
  double dist;
};

namespace detail {

// d/d(rc) of bump_cutoff at fixed distance -- the host twin of
// detail::dev_bump_dcutoff_dr (device_neighbors.hpp), which carries the
// derivation and the reason for the clamp.
inline double bump_dcutoff_dr(double d, double rc, double width) {
  const double scaled = (d - (rc - width)) / width;
  if (scaled <= 0.0 || scaled >= 1.0) return 0.0;
  const double safe = std::min(std::max(scaled, 1e-6), 1.0 - 1e-6);
  const double ps = M_PI * safe;
  const double si = std::sin(ps);
  const double tt = std::tanh(std::cos(ps) / si);
  return 0.5 * (1.0 - tt * tt) * (M_PI / (si * si)) / width;
}

// Solver adaptive cutoff: Newton-bisection on n_total(r) = target, plus one
// implicit-function step. The host twin of the pet_adapt_solver kernel in
// device_neighbors.hpp, which carries the full commentary; the two must agree,
// and tests/test_device_vs_host.cpp is what holds them to it.
inline std::vector<double> adaptive_cutoffs_solver(int N, const Hypers& h,
                                                   const std::vector<RawEdgeIn>& edges,
                                                   std::vector<double>* out_r = nullptr,
                                                   std::vector<double>* out_dn = nullptr) {
  const double rmax = h.cutoff;
  const double inv_rmax = 1.0 / rmax;
  const double width = h.cutoff_width_adaptive;
  const double target = h.num_neighbors_adaptive;
  const double lo_bound = rmax * (1.0 / 16.0);
  constexpr int kIters = 10;
  constexpr double kDnFloor = 1e-6;

  // Group edges by centre so each atom's sums accumulate in a fixed order.
  std::vector<std::vector<double>> dist_of(N);
  for (const auto& e : edges) dist_of[e.center].push_back(e.dist);

  std::vector<double> out(N, rmax);
  if (out_r) out_r->assign(N, rmax);
  if (out_dn) out_dn->assign(N, 0.0);
  for (int a = 0; a < N; ++a) {
    const auto& ds = dist_of[a];
    double r_lo = 0.0, r_hi = rmax, r = 0.5 * rmax;
    for (int it = 0; it < kIters; ++it) {
      double n = 0.0, dn = 0.0;
      for (double d : ds) {
        n += bump_cutoff(d, r, width);
        dn += bump_dcutoff_dr(d, r, width);
      }
      const double x = r * inv_rmax;
      n += target * x * x * x;
      dn += 3.0 * target * x * x * inv_rmax;

      const double f = n - target;
      if (f <= 0.0) r_lo = r; else r_hi = r;
      const double r_newton = r - f / std::max(dn, kDnFloor);
      r = (r_newton >= r_lo && r_newton <= r_hi) ? r_newton : 0.5 * (r_lo + r_hi);
    }
    double n = 0.0, dn = 0.0;
    for (double d : ds) {
      n += bump_cutoff(d, r, width);
      dn += bump_dcutoff_dr(d, r, width);
    }
    const double x = r * inv_rmax;
    n += target * x * x * x;
    dn += 3.0 * target * x * x * inv_rmax;

    const double dn_root = std::max(dn, kDnFloor);
    const double adapted = r - (n - target) / dn_root;
    const double clamped = std::min(std::max(adapted, lo_bound), rmax);
    out[a] = clamped;
    if (out_r) (*out_r)[a] = r;
    // Zero signals "clamped, so no gradient" -- the same convention the device
    // forward uses, so the backward needs only one rule.
    if (out_dn) (*out_dn)[a] = (adapted > lo_bound && adapted < rmax) ? dn_root : 0.0;
  }
  return out;
}

// Adaptive per-atom cutoff, GRID method (adaptive_cutoff.py's legacy
// get_adaptive_cutoffs_grid). Returns atomic_cutoffs[N].
inline std::vector<double> adaptive_cutoffs(int N, const Hypers& h,
                                            const std::vector<RawEdgeIn>& edges) {
  const double min_cutoff = 0.5;
  const double max_cutoff = h.cutoff;
  // The taper of the SMOOTHED NEIGHBOUR COUNT, which metatrain passes as
  // cutoff_width_adaptive -- not the edge cutoff factor's own width.
  const double width = h.cutoff_width_adaptive;
  const double target = h.num_neighbors_adaptive;  // 8
  const double probe_spacing = width / 4.0;        // 0.125

  std::vector<double> probes;
  for (double p = min_cutoff; p < max_cutoff - 1e-12; p += probe_spacing) probes.push_back(p);
  const int P = static_cast<int>(probes.size());
  if (P == 0) return std::vector<double>(N, max_cutoff);

  // effective neighbor counts: eff[atom][p] = sum over atom's edges of bump(d, probe_p, width)
  std::vector<std::vector<double>> eff(N, std::vector<double>(P, 0.0));
  for (const auto& e : edges) {
    for (int p = 0; p < P; ++p)
      eff[e.center][p] += bump_cutoff(e.dist, probes[p], width);
  }

  std::vector<double> out(N, max_cutoff);
  std::vector<double> diff(P), grad(P), logw(P);
  for (int a = 0; a < N; ++a) {
    for (int p = 0; p < P; ++p) {
      const double x = (P > 1) ? static_cast<double>(p) / (P - 1) : 0.0;
      diff[p] = eff[a][p] - target + target * x * x * x;  // baseline = target * x^3
    }
    // torch.gradient along p, unit spacing: centered interior, one-sided edges
    if (P == 1) {
      grad[0] = std::fabs(diff[0]) * 0.5 + 1e-12;
    } else {
      for (int p = 0; p < P; ++p) {
        double g;
        if (p == 0) g = diff[1] - diff[0];
        else if (p == P - 1) g = diff[P - 1] - diff[P - 2];
        else g = 0.5 * (diff[p + 1] - diff[p - 1]);
        grad[p] = std::max(std::fabs(g), 1e-12);
      }
    }
    // logw = -0.5*(diff/width_p)^2 ; softmax over p (global max cancels in normalization)
    double mx = -1e300;
    for (int p = 0; p < P; ++p) {
      logw[p] = -0.5 * (diff[p] / grad[p]) * (diff[p] / grad[p]);
      mx = std::max(mx, logw[p]);
    }
    double sum = 0.0;
    for (int p = 0; p < P; ++p) { logw[p] = std::exp(logw[p] - mx); sum += logw[p]; }
    double acc = 0.0;
    for (int p = 0; p < P; ++p) acc += probes[p] * (logw[p] / sum);
    out[a] = acc;
  }
  return out;
}

}  // namespace detail

// Build NEF edge data from a pre-supplied raw edge list and per-atom species
// indices (already mapped Z -> [0,n_species)). Shared by the standalone System
// path and any caller supplying its own neighbour list. `edges` must be the FULL list within the
// search cutoff (before adaptive masking).
inline EdgeData build_edge_data_from_raw(int N, const std::vector<int>& species, const Hypers& h,
                                         const std::vector<RawEdgeIn>& edges) {
  using namespace detail;
  EdgeData ed;
  ed.n_atoms = N;
  ed.species = species;

  // keep the full raw edge list for the adaptive-cutoff backward
  ed.n_raw = static_cast<int>(edges.size());
  ed.raw_center.resize(edges.size());
  ed.raw_neigh.resize(edges.size());
  ed.raw_dist.resize(edges.size());
  ed.raw_vec.resize(edges.size() * 3);
  for (std::size_t e = 0; e < edges.size(); ++e) {
    ed.raw_center[e] = edges[e].center;
    ed.raw_neigh[e] = edges[e].neigh_owner;
    ed.raw_dist[e] = edges[e].dist;
    ed.raw_vec[3 * e + 0] = edges[e].vx;
    ed.raw_vec[3 * e + 1] = edges[e].vy;
    ed.raw_vec[3 * e + 2] = edges[e].vz;
  }

  // adaptive cutoff masking + cutoff factors
  std::vector<double> acut;
  if (h.adaptive()) {
    if (h.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver)
      acut = adaptive_cutoffs_solver(N, h, edges, &ed.adapt_r, &ed.adapt_dn);
    else
      acut = adaptive_cutoffs(N, h, edges);
  }

  std::vector<const RawEdgeIn*> kept;
  std::vector<double> kept_pair_cutoff, kept_factor;
  kept.reserve(edges.size());
  for (const auto& e : edges) {
    double rc;
    if (h.adaptive()) {
      rc = 0.5 * (acut[e.center] + acut[e.neigh_owner]);
      if (e.dist > rc) continue;  // drop edge beyond adaptive pair cutoff
    } else {
      rc = h.cutoff;
    }
    kept.push_back(&e);
    kept_pair_cutoff.push_back(rc);
    kept_factor.push_back(cutoff_value(h, e.dist, rc));
  }

  // per-atom neighbor counts and M
  ed.num_neigh.assign(N, 0);
  for (const auto* e : kept) ed.num_neigh[e->center]++;
  int M = 0;
  for (int i = 0; i < N; ++i) M = std::max(M, ed.num_neigh[i]);
  M = std::max(M, 1);  // guard
  ed.max_neighbors = M;

  const std::size_t NM = static_cast<std::size_t>(N) * M;
  ed.edge_vec.assign(NM * 3, 0.0);
  ed.edge_dist.assign(NM, 0.0);
  ed.neigh_species.assign(NM, 0);
  ed.cutoff_factor.assign(NM, 0.0);
  ed.pair_cutoff.assign(NM, h.cutoff);
  ed.mask.assign(NM, 0);
  ed.reverse_index.assign(NM, -1);

  // assign edges to NEF slots (stable insertion order = argsort-stable on centers).
  // reverse edge (j->i) matched exactly by (tag_j, tag_i, -cell_shift).
  auto ekey = [](long long ta, long long tb, int sa, int sb, int sc) {
    return std::make_tuple(ta, tb, sa, sb, sc);
  };
  std::vector<int> slot(N, 0);
  std::map<std::tuple<long long, long long, int, int, int>, int> edge_to_slot;
  std::vector<int> flat_of_edge(kept.size());

  for (std::size_t k = 0; k < kept.size(); ++k) {
    const auto& e = *kept[k];
    const int s = slot[e.center]++;
    const std::size_t flat = static_cast<std::size_t>(e.center) * M + s;
    flat_of_edge[k] = static_cast<int>(flat);
    ed.edge_vec[flat * 3 + 0] = e.vx;
    ed.edge_vec[flat * 3 + 1] = e.vy;
    ed.edge_vec[flat * 3 + 2] = e.vz;
    ed.edge_dist[flat] = e.dist;
    ed.neigh_species[flat] = ed.species[e.neigh_owner];
    ed.cutoff_factor[flat] = kept_factor[k];
    ed.pair_cutoff[flat] = kept_pair_cutoff[k];
    ed.mask[flat] = 1;
    edge_to_slot[ekey(e.tag_center, e.tag_neigh, e.sa, e.sb, e.sc)] = static_cast<int>(flat);
  }

  for (std::size_t k = 0; k < kept.size(); ++k) {
    const auto& e = *kept[k];
    auto it = edge_to_slot.find(ekey(e.tag_neigh, e.tag_center, -e.sa, -e.sb, -e.sc));
    ed.reverse_index[flat_of_edge[k]] = (it != edge_to_slot.end()) ? it->second : -1;
  }

  return ed;
}

// Build NEF edge data from a System (standalone path: builds its own full
// periodic neighbor list).
inline EdgeData build_edge_data(const System& sys, const Hypers& h,
                                const std::vector<int>& species_to_index) {
  using namespace detail;
  const int N = sys.n_atoms;
  std::vector<int> species(N);
  for (int i = 0; i < N; ++i) {
    const int Z = sys.atomic_numbers[i];
    const int s = (Z >= 0 && Z < (int)species_to_index.size()) ? species_to_index[Z] : -1;
    if (s < 0) throw std::runtime_error("unsupported atomic number in system");
    species[i] = s;
  }

  auto raw = build_raw_edges(sys, h.cutoff);
  std::vector<RawEdgeIn> edges(raw.size());
  for (std::size_t e = 0; e < raw.size(); ++e) {
    edges[e].center = raw[e].i;
    edges[e].neigh_owner = raw[e].j;
    edges[e].tag_center = raw[e].i;
    edges[e].tag_neigh = raw[e].j;
    edges[e].sa = raw[e].sa;
    edges[e].sb = raw[e].sb;
    edges[e].sc = raw[e].sc;
    edges[e].vx = raw[e].vx;
    edges[e].vy = raw[e].vy;
    edges[e].vz = raw[e].vz;
    edges[e].dist = raw[e].dist;
  }
  EdgeData ed = build_edge_data_from_raw(N, species, h, edges);
  // One structure, so one entry. Only a conditioned model reads these; carrying
  // them unconditionally keeps build_edge_data free of a model-shape branch, and
  // two ints per structure is not worth a conditional.
  ed.charge = {sys.charge};
  ed.spin_multiplicity = {sys.spin_multiplicity};
  return ed;
}

}  // namespace pet
