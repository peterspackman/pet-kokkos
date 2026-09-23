// The host neighbour list: see neighbors.hpp. The device builder
// (device_neighbors.hpp) mirrors this, and tests/test_device_vs_host.cpp holds
// the two to agreement.
#include "pet/neighbors.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace pet {

namespace detail {

double bump_cutoff(double d, double rc, double width) {
  const double x = (d - (rc - width)) / width;
  if (x <= 0.0) return 1.0;
  if (x >= 1.0) return 0.0;
  return 0.5 * (1.0 + std::tanh(1.0 / std::tan(M_PI * x)));
}

double cosine_cutoff(double d, double rc, double width) {
  const double x = std::min(std::max((d - (rc - width)) / width, 0.0), 1.0);
  return 0.5 * (1.0 + std::cos(M_PI * x));
}

// The derivative runs off to infinity at both ends of the taper, where the
// function itself is flat; the clamp keeps it finite there.
double bump_dcutoff_dr(double d, double rc, double width) {
  const double x = (d - (rc - width)) / width;
  if (x <= 0.0 || x >= 1.0) return 0.0;
  const double ps = M_PI * std::min(std::max(x, 1e-6), 1.0 - 1e-6);
  const double si = std::sin(ps), tt = std::tanh(std::cos(ps) / si);
  return 0.5 * (1.0 - tt * tt) * (M_PI / (si * si)) / width;
}

// The number of images is the cutoff over the interplanar spacing along each
// periodic direction; a degenerate cell has none.
std::array<int, 3> image_ranges(const System& sys, double cutoff) {
  const auto& c = sys.cell;
  const double a[3] = {c[0], c[1], c[2]}, b[3] = {c[3], c[4], c[5]}, cc[3] = {c[6], c[7], c[8]};
  auto cross = [](const double* u, const double* v, double* w) {
    w[0] = u[1] * v[2] - u[2] * v[1], w[1] = u[2] * v[0] - u[0] * v[2], w[2] = u[0] * v[1] - u[1] * v[0];
  };
  auto norm = [](const double* u) { return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
  double bc[3], ca[3], ab[3];
  cross(b, cc, bc), cross(cc, a, ca), cross(a, b, ab);
  const double vol = std::fabs(a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]);
  std::array<int, 3> n{0, 0, 0};
  if (vol < 1e-12) return n;
  const double* face[3] = {bc, ca, ab};
  for (int d = 0; d < 3; ++d)
    if (sys.pbc[d]) n[d] = (int) std::ceil(cutoff / (vol / norm(face[d])));
  return n;
}

std::vector<RawEdge> build_raw_edges(const System& sys, double cutoff) {
  const int N = sys.n_atoms;
  const auto& c = sys.cell;
  const auto rng = image_ranges(sys, cutoff);
  const double cutoff2 = cutoff * cutoff;
  const double* x = sys.positions.data();
  std::vector<RawEdge> edges;
  edges.reserve((std::size_t) N * 16);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int sa = -rng[0]; sa <= rng[0]; ++sa)
        for (int sb = -rng[1]; sb <= rng[1]; ++sb)
          for (int sc = -rng[2]; sc <= rng[2]; ++sc) {
            const double vx = x[3 * j + 0] + (sa * c[0] + sb * c[3] + sc * c[6]) - x[3 * i + 0];
            const double vy = x[3 * j + 1] + (sa * c[1] + sb * c[4] + sc * c[7]) - x[3 * i + 1];
            const double vz = x[3 * j + 2] + (sa * c[2] + sb * c[5] + sc * c[8]) - x[3 * i + 2];
            const double d2 = vx * vx + vy * vy + vz * vz;
            if (d2 < 1e-24 || d2 > cutoff2) continue;  // the atom itself, or too far
            edges.push_back({i, j, sa, sb, sc, vx, vy, vz, std::sqrt(d2)});
          }
  return edges;
}

// The solver adaptive cutoff: Newton-bisection for the r where the smoothed
// neighbour count n_total(r) = sum_j bump(d_j, r) + target (r/rmax)^3 reaches
// target, then one implicit-function step for the backward to differentiate.
// The host twin of the pet_adapt_solver kernel, which explains more.
std::vector<double> adaptive_cutoffs_solver(int N, const Hypers& h, const std::vector<RawEdgeIn>& edges,
                                            std::vector<double>& out_r, std::vector<double>& out_dn) {
  const double rmax = h.cutoff, inv_rmax = 1.0 / rmax, width = h.cutoff_width_adaptive;
  const double target = h.num_neighbors_adaptive, lo_bound = rmax / 16.0;
  constexpr int kIters = 10;
  constexpr double kDnFloor = 1e-6;
  std::vector<std::vector<double>> dist_of(N);  // grouped by centre, in edge order
  for (const auto& e : edges) dist_of[e.center].push_back(e.dist);

  auto count = [&](const std::vector<double>& ds, double r, double& n, double& dn) {
    n = dn = 0.0;
    for (double d : ds) n += bump_cutoff(d, r, width), dn += bump_dcutoff_dr(d, r, width);
    const double x = r * inv_rmax;
    n += target * x * x * x;
    dn += 3.0 * target * x * x * inv_rmax;
  };
  std::vector<double> out(N, rmax);
  out_r.assign(N, rmax);
  out_dn.assign(N, 0.0);
  for (int a = 0; a < N; ++a) {
    double r_lo = 0.0, r_hi = rmax, r = 0.5 * rmax, n, dn;
    for (int it = 0; it < kIters; ++it) {
      count(dist_of[a], r, n, dn);
      const double f = n - target;
      if (f <= 0.0) r_lo = r; else r_hi = r;
      const double r_newton = r - f / std::max(dn, kDnFloor);
      r = (r_newton >= r_lo && r_newton <= r_hi) ? r_newton : 0.5 * (r_lo + r_hi);
    }
    count(dist_of[a], r, n, dn);
    const double dn_root = std::max(dn, kDnFloor), adapted = r - (n - target) / dn_root;
    out[a] = std::min(std::max(adapted, lo_bound), rmax);
    out_r[a] = r;
    out_dn[a] = (adapted > lo_bound && adapted < rmax) ? dn_root : 0.0;  // 0: clamped
  }
  return out;
}

// The grid adaptive cutoff (metatrain's get_adaptive_cutoffs_grid): a softmax-
// weighted average of probe radii, each weighted by how close its smoothed
// neighbour count is to target.
std::vector<double> adaptive_cutoffs_grid(int N, const Hypers& h, const std::vector<RawEdgeIn>& edges) {
  const double width = h.cutoff_width_adaptive, target = h.num_neighbors_adaptive;
  std::vector<double> probes;
  for (double p = 0.5; p < h.cutoff - 1e-12; p += width / 4.0) probes.push_back(p);
  const int P = probes.size();
  if (P == 0) return std::vector<double>(N, h.cutoff);

  std::vector<std::vector<double>> eff(N, std::vector<double>(P, 0.0));
  for (const auto& e : edges)
    for (int p = 0; p < P; ++p) eff[e.center][p] += bump_cutoff(e.dist, probes[p], width);

  std::vector<double> out(N), diff(P), grad(P), w(P);
  for (int a = 0; a < N; ++a) {
    for (int p = 0; p < P; ++p) {
      const double x = (P > 1) ? (double) p / (P - 1) : 0.0;
      diff[p] = eff[a][p] - target + target * x * x * x;
    }
    // torch.gradient: centred inside, one-sided at the ends.
    if (P == 1)
      grad[0] = std::fabs(diff[0]) * 0.5 + 1e-12;
    else
      for (int p = 0; p < P; ++p) {
        const double g = p == 0 ? diff[1] - diff[0] : p == P - 1 ? diff[P - 1] - diff[P - 2]
                                                                 : 0.5 * (diff[p + 1] - diff[p - 1]);
        grad[p] = std::max(std::fabs(g), 1e-12);
      }
    double mx = -1e300, sum = 0.0, acc = 0.0;
    for (int p = 0; p < P; ++p) mx = std::max(mx, w[p] = -0.5 * (diff[p] / grad[p]) * (diff[p] / grad[p]));
    for (int p = 0; p < P; ++p) sum += (w[p] = std::exp(w[p] - mx));
    for (int p = 0; p < P; ++p) acc += probes[p] * (w[p] / sum);
    out[a] = acc;
  }
  return out;
}

}  // namespace detail

EdgeData build_edge_data_from_raw(int N, const std::vector<int>& species, const Hypers& h,
                                  const std::vector<RawEdgeIn>& edges) {
  EdgeData ed;
  ed.n_atoms = N;
  ed.species = species;
  ed.n_raw = edges.size();
  for (const auto& e : edges) {
    ed.raw_center.push_back(e.center);
    ed.raw_neigh.push_back(e.neigh_owner);
    ed.raw_dist.push_back(e.dist);
    ed.raw_vec.insert(ed.raw_vec.end(), {e.vx, e.vy, e.vz});
  }

  // The adaptive cutoff, and which edges it keeps: those within the mean of
  // their two atoms' cutoffs.
  std::vector<double> acut;
  if (h.adaptive())
    acut = h.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver
               ? detail::adaptive_cutoffs_solver(N, h, edges, ed.adapt_r, ed.adapt_dn)
               : detail::adaptive_cutoffs_grid(N, h, edges);
  struct Kept {
    const RawEdgeIn* e;
    double rc;
  };
  std::vector<Kept> kept;
  ed.num_neigh.assign(N, 0);
  for (const auto& e : edges) {
    const double rc = h.adaptive() ? 0.5 * (acut[e.center] + acut[e.neigh_owner]) : h.cutoff;
    if (h.adaptive() && e.dist > rc) continue;
    kept.push_back({&e, rc});
    ++ed.num_neigh[e.center];
  }
  const int M = std::max(1, N ? *std::max_element(ed.num_neigh.begin(), ed.num_neigh.end()) : 0);
  ed.max_neighbors = M;

  const std::size_t NM = (std::size_t) N * M;
  ed.edge_vec.assign(NM * 3, 0.0);
  ed.edge_dist.assign(NM, 0.0);
  ed.neigh_species.assign(NM, 0);
  ed.cutoff_factor.assign(NM, 0.0);
  ed.pair_cutoff.assign(NM, h.cutoff);
  ed.mask.assign(NM, 0);
  ed.reverse_index.assign(NM, -1);

  // Slots in edge order per atom; the reverse of (i, j, s) is (j, i, -s), found
  // by its tags.
  using Key = std::tuple<long long, long long, int, int, int>;
  std::map<Key, int> slot_of;
  std::vector<int> slot(N, 0), flat(kept.size());
  for (std::size_t k = 0; k < kept.size(); ++k) {
    const RawEdgeIn& e = *kept[k].e;
    const std::size_t f = (std::size_t) e.center * M + slot[e.center]++;
    flat[k] = f;
    ed.edge_vec[f * 3 + 0] = e.vx, ed.edge_vec[f * 3 + 1] = e.vy, ed.edge_vec[f * 3 + 2] = e.vz;
    ed.edge_dist[f] = e.dist;
    ed.neigh_species[f] = ed.species[e.neigh_owner];
    ed.cutoff_factor[f] = h.cutoff_function == CutoffFunction::Bump
                              ? detail::bump_cutoff(e.dist, kept[k].rc, h.cutoff_width)
                              : detail::cosine_cutoff(e.dist, kept[k].rc, h.cutoff_width);
    ed.pair_cutoff[f] = kept[k].rc;
    ed.mask[f] = 1;
    slot_of[{e.tag_center, e.tag_neigh, e.sa, e.sb, e.sc}] = f;
  }
  for (std::size_t k = 0; k < kept.size(); ++k) {
    const RawEdgeIn& e = *kept[k].e;
    const auto it = slot_of.find({e.tag_neigh, e.tag_center, -e.sa, -e.sb, -e.sc});
    ed.reverse_index[flat[k]] = it != slot_of.end() ? it->second : -1;
  }
  return ed;
}

EdgeData build_edge_data(const System& sys, const Hypers& h, const std::vector<int>& species_to_index) {
  const int N = sys.n_atoms;
  std::vector<int> species(N);
  for (int i = 0; i < N; ++i) {
    const int Z = sys.atomic_numbers[i];
    species[i] = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
    if (species[i] < 0) throw std::runtime_error("unsupported atomic number in system");
  }
  std::vector<RawEdgeIn> edges;
  for (const auto& r : detail::build_raw_edges_dispatch(sys, h.cutoff))
    edges.push_back({r.i, r.j, r.i, r.j, r.sa, r.sb, r.sc, r.vx, r.vy, r.vz, r.dist});
  EdgeData ed = build_edge_data_from_raw(N, species, h, edges);
  ed.charge = {sys.charge};
  ed.spin_multiplicity = {sys.spin_multiplicity};
  return ed;
}

EdgeData concat_edge_data(const std::vector<EdgeData>& parts, std::vector<int>& struct_id) {
  EdgeData out;
  int N = 0, M = 0, E = 0;
  for (const auto& p : parts) N += p.n_atoms, M = std::max(M, p.max_neighbors), E += p.n_raw;
  out.n_atoms = N, out.max_neighbors = M, out.n_raw = E;
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
  struct_id.assign(N, 0);
  const bool solver = !parts.empty() && !parts.front().adapt_r.empty();
  if (solver) out.adapt_r.assign(N, 0.0), out.adapt_dn.assign(N, 0.0);

  int n0 = 0;
  for (std::size_t b = 0; b < parts.size(); ++b) {
    const EdgeData& p = parts[b];
    const int Mb = p.max_neighbors;
    out.charge.insert(out.charge.end(), p.charge.begin(), p.charge.end());
    out.spin_multiplicity.insert(out.spin_multiplicity.end(), p.spin_multiplicity.begin(), p.spin_multiplicity.end());
    for (int i = 0; i < p.n_atoms; ++i) {
      const int gi = n0 + i;
      out.species[gi] = p.species[i];
      out.num_neigh[gi] = p.num_neigh[i];
      struct_id[gi] = b;
      if (solver && i < (int) p.adapt_r.size()) out.adapt_r[gi] = p.adapt_r[i], out.adapt_dn[gi] = p.adapt_dn[i];
      for (int m = 0; m < Mb; ++m) {
        const std::size_t s = (std::size_t) i * Mb + m, d = (std::size_t) gi * M + m;
        out.mask[d] = p.mask[s];
        out.neigh_species[d] = p.neigh_species[s];
        out.edge_dist[d] = p.edge_dist[s];
        out.cutoff_factor[d] = p.cutoff_factor[s];
        out.pair_cutoff[d] = p.pair_cutoff[s];
        for (int c = 0; c < 3; ++c) out.edge_vec[d * 3 + c] = p.edge_vec[s * 3 + c];
        const int r = p.reverse_index[s];  // in the part's M; re-encode in the batch's
        if (r >= 0) out.reverse_index[d] = (n0 + r / Mb) * M + r % Mb;
      }
    }
    for (int e = 0; e < p.n_raw; ++e) {
      out.raw_center.push_back(p.raw_center[e] + n0);
      out.raw_neigh.push_back(p.raw_neigh[e] + n0);
      out.raw_dist.push_back(p.raw_dist[e]);
      for (int c = 0; c < 3; ++c) out.raw_vec.push_back(p.raw_vec[(std::size_t) e * 3 + c]);
    }
    n0 += p.n_atoms;
  }
  return out;
}

}  // namespace pet
