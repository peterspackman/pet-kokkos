// The device neighbour builder: see device_neighbors.hpp. It mirrors the host
// builder (neighbors.cpp), and tests/test_device_vs_host.cpp holds the two to it.
#include "pet/device_neighbors.hpp"

#include <Kokkos_Sort.hpp>

#include "pet/neighbors.hpp"

#include <cstdlib>
#include <stdexcept>

namespace pet {

namespace {

using AtomTeams = TeamPolicy;  // one warp per atom, lanes on its edges
using Atom = AtomTeams::member_type;

// The solver adaptive cutoff: Newton-bisection for the r where the smoothed
// neighbour count n_total(r) = sum_j bump(d_j, r) + target (r/rmax)^3 reaches
// target. The cubic baseline makes n_total monotone on [0, rmax], so that
// interval brackets the one root from the start. A final implicit-function step
// puts a differentiable expression at the root, so the backward never unrolls
// the iterations; ar and adn keep r and the slope there for it (adn = 0:
// clamped, no gradient).
void solver_cutoffs(const Hypers& h, int N, IView1D roff, RView1D re_dist, RView1D acut, RView1D ar,
                    RView1D adn) {
  const double target = h.num_neighbors_adaptive, rmax = h.cutoff, inv_rmax = 1.0 / rmax;
  const double width = h.cutoff_width_adaptive, lo_bound = rmax * detail::PET_SOLVER_MIN_CUTOFF_FACTOR;
  Kokkos::parallel_for(
      "pet_adapt_solver", AtomTeams(N, 1, kLanes), KOKKOS_LAMBDA(const Atom& t) {
        const int a = t.league_rank();
        auto count = [&](double r, double& n, double& dn) {
          n = dn = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(t, roff(a), roff(a + 1)),
              [&](int e, double& sn, double& sdn) {
                sn += detail::dev_bump_cutoff(re_dist(e), r, width);
                sdn += detail::dev_bump_dcutoff_dr(re_dist(e), r, width);
              },
              n, dn);
          const double x = r * inv_rmax;
          n += target * x * x * x;
          dn += 3.0 * target * x * x * inv_rmax;
        };
        double r_lo = 0.0, r_hi = rmax, r = 0.5 * rmax, n, dn;
        for (int it = 0; it < detail::PET_SOLVER_ITERS; ++it) {
          count(r, n, dn);
          const double f = n - target;
          if (f <= 0.0) r_lo = r; else r_hi = r;
          // Newton, unless it would leave the bracket (on a flat shoulder): bisect.
          const double r_newton = r - f / Kokkos::fmax(dn, detail::PET_SOLVER_DN_FLOOR);
          r = (r_newton >= r_lo && r_newton <= r_hi) ? r_newton : 0.5 * (r_lo + r_hi);
        }
        count(r, n, dn);
        const double dn_root = Kokkos::fmax(dn, detail::PET_SOLVER_DN_FLOOR);
        const double adapted = Kokkos::fmin(Kokkos::fmax(r - (n - target) / dn_root, lo_bound), rmax);
        Kokkos::single(Kokkos::PerThread(t), [&] {
          acut(a) = adapted;
          ar(a) = r;
          adn(a) = (adapted > lo_bound && adapted < rmax) ? dn_root : 0.0;
        });
      });
}

// The grid adaptive cutoff: a softmax-weighted average of probe radii, each
// weighted by how close its smoothed neighbour count eff is to target. eff is
// kept for the backward.
//
// bump(d, probe) is exactly 1 for probes at or past d + width and 0 at or below
// d, smooth only in between, and the grid is uniform: so each edge marks where
// its run of 1s starts and adds the few smooth values, and a per-atom prefix sum
// turns the marks into counts -- O(E + N*P), not O(E*P).
RView2D grid_cutoffs(Workspace& ws, const Hypers& h, int N, IView1D roff, RView1D re_dist, RView1D probes,
                     int P, RView1D acut) {
  const double target = h.num_neighbors_adaptive, width = h.cutoff_width_adaptive;
  Kokkos::deep_copy(ExecSpace(), acut, h.cutoff);
  if (P == 0) return RView2D();
  RView2D eff = ws.r2("nef:eff", N, P), effp = ws.r2("nef:effp", N, P);
  Kokkos::parallel_for(
      "pet_adapt_eff_scatter", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        const double p0 = probes(0), dp = (P > 1) ? (probes(1) - probes(0)) : 1.0;
        for (int e = roff(a); e < roff(a + 1); ++e) {
          const double d = re_dist(e);
          const int p_full = Kokkos::clamp((int) Kokkos::ceil((d + width - p0) / dp), 0, P);  // first 1
          const int p_start = Kokkos::max((int) Kokkos::floor((d - p0) / dp) + 1, 0);        // first > 0
          if (p_full < P) eff(a, p_full) += 1.0;
          for (int p = p_start; p < p_full && p < P; ++p) effp(a, p) += detail::dev_bump_cutoff(d, probes(p), width);
        }
      });
  Kokkos::parallel_for(
      "pet_adapt_eff_scan", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        double run = 0.0;
        for (int p = 0; p < P; ++p) {
          run += eff(a, p);
          eff(a, p) = run + effp(a, p);
        }
      });
  RView2D diff = ws.r2("nef:diffv", N, P), grad = ws.r2("nef:gradv", N, P);
  Kokkos::parallel_for(
      "pet_adapt_diff", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {
        const int a = i / P, p = i % P;
        const double x = (P > 1) ? (double) p / (P - 1) : 0.0;
        diff(a, p) = eff(a, p) - target + target * x * x * x;
      });
  Kokkos::parallel_for(
      "pet_adapt_grad", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {  // torch.gradient
        const int a = i / P, p = i % P;
        if (P == 1) return (void) (grad(a, 0) = Kokkos::fabs(diff(a, 0)) * 0.5 + 1e-12);
        const double g = p == 0       ? diff(a, 1) - diff(a, 0)
                         : p == P - 1 ? diff(a, P - 1) - diff(a, P - 2)
                                      : 0.5 * (diff(a, p + 1) - diff(a, p - 1));
        grad(a, p) = Kokkos::fmax(Kokkos::fabs(g), 1e-12);
      });
  Kokkos::parallel_for(
      "pet_adapt_softmax", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        double mx = -1e300, sum = 0.0, acc = 0.0;
        for (int p = 0; p < P; ++p) {
          const double r = diff(a, p) / grad(a, p), lw = -0.5 * r * r;
          mx = (lw > mx) ? lw : mx;
        }
        for (int p = 0; p < P; ++p) {
          const double r = diff(a, p) / grad(a, p), w = Kokkos::exp(-0.5 * r * r - mx);
          sum += w;
          acc += probes(p) * w;
        }
        acut(a) = acc / sum;
      });
  return eff;
}

}  // namespace

void edge_topology(Workspace& ws, EdgeMap& edge_map, DeviceEdgeData& dev, int N, IView1D species, IView1D re_i,
                   IView1D re_j, IView2D re_shift, RView2D re_vec, RView1D re_dist, int E) {
  dev.n_atoms = N;
  dev.n_raw = E;
  dev.species = species;
  dev.raw_center = re_i, dev.raw_neigh = re_j, dev.raw_dist = re_dist, dev.raw_vec = re_vec;

  // Each atom's range of the raw list: a count and a scan, since re_i is sorted.
  IView1D roff = ws.i1("nef:roff", N + 1), rcount = ws.i1("nef:rcnt", N);
  Kokkos::deep_copy(ExecSpace(), rcount, 0);
  Kokkos::parallel_for(
      "pet_raw_count", RangePolicy(0, E), KOKKOS_LAMBDA(int e) { Kokkos::atomic_inc(&rcount(re_i(e))); });
  Kokkos::parallel_scan(
      "pet_raw_scan", RangePolicy(0, N), KOKKOS_LAMBDA(int a, int& upd, bool final) {
        if (final) roff(a) = upd;
        upd += rcount(a);
        if (final && a == N - 1) roff(N) = upd;
      });
  if (N == 0) Kokkos::deep_copy(ExecSpace(), roff, 0);
  dev.raw_off = roff;

  // Each raw edge's partner (j, i, -shift), through a hash of the list. The keep
  // test below is symmetric in i and j, so the kept edges' reverse map is read
  // off this one.
  IView1D raw_rev = ws.i1("nef:rawrev", E);
  const uint32_t need = E > 0 ? 2 * E + 16 : 16;
  if (edge_map.capacity() < need) edge_map.rehash(need);
  edge_map.clear();
  auto key = KOKKOS_LAMBDA(int e, int sign) {
    return sign > 0 ? detail::pet_pack_key(re_i(e), re_j(e), re_shift(e, 0), re_shift(e, 1), re_shift(e, 2))
                    : detail::pet_pack_key(re_j(e), re_i(e), -re_shift(e, 0), -re_shift(e, 1), -re_shift(e, 2));
  };
  Kokkos::parallel_for("pet_raw_hash", RangePolicy(0, E), KOKKOS_LAMBDA(int e) { edge_map.insert(key(e, 1), e); });
  Kokkos::parallel_for(
      "pet_raw_reverse", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        const uint32_t idx = edge_map.find(key(e, -1));
        raw_rev(e) = edge_map.valid_at(idx) ? edge_map.value_at(idx) : -1;
      });
  dev.raw_reverse = raw_rev;

  // Edges with no partner, by (target atom, edge): the force folds gather each
  // edge's share from its partner, so these are added on their own (forces.cpp).
  // Only an engine's list with neighbourless ghosts has any.
  int n_orphan = 0;
  Kokkos::parallel_reduce(
      "pet_orphans", RangePolicy(0, E), KOKKOS_LAMBDA(int e, int& c) { c += raw_rev(e) < 0; }, n_orphan);
  if (n_orphan > 0) {
    U64View1D keys = ws.u64("nef:orph_keys", n_orphan);
    IView1D ooff = dev.orphan_off = ws.i1("nef:orph_off", N + 1), oedge = dev.orphan_edge = ws.i1("nef:orph_edge", n_orphan);
    Kokkos::parallel_scan(
        "pet_orph_keys", RangePolicy(0, E), KOKKOS_LAMBDA(int e, int& k, bool final) {
          if (raw_rev(e) >= 0) return;
          if (final) keys(k) = uint64_t(re_j(e)) << 32 | uint32_t(e);
          ++k;
        });
    Kokkos::sort(ExecSpace(), keys);
    // orphan_off(a) = the first orphan pointing at a or beyond.
    Kokkos::parallel_for(
        "pet_orph_off", RangePolicy(0, N + 1), KOKKOS_LAMBDA(int a) {
          int lo = 0, hi = n_orphan;
          while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (int(keys(mid) >> 32) < a) lo = mid + 1;
            else hi = mid;
          }
          ooff(a) = lo;
        });
    Kokkos::parallel_for(
        "pet_orph_edge", RangePolicy(0, n_orphan), KOKKOS_LAMBDA(int k) { oedge(k) = int(keys(k) & 0xffffffffu); });
  } else {
    dev.orphan_off = IView1D(), dev.orphan_edge = IView1D();
  }
}

void edge_geometry(Workspace& ws, DeviceEdgeData& dev, const Hypers& h, RView1D probes, int P, int& m_high,
                   int m_fixed, IView1D overflow) {
  const bool adaptive = h.adaptive(), bump = h.cutoff_function == CutoffFunction::Bump;
  const double width = h.cutoff_width, cutoff = h.cutoff;
  const int N = dev.n_atoms, NC = dev.centres(), E = dev.n_raw;
  auto roff = dev.raw_off, raw_rev = dev.raw_reverse, re_i = dev.raw_center, re_j = dev.raw_neigh;
  auto re_dist = dev.raw_dist;
  auto re_vec = dev.raw_vec;
  auto species = dev.species;

  // The adaptive cutoff of each atom. Both schemes taper the neighbour count with
  // the bump, whatever the model's own cutoff function, as metatrain does.
  RView1D acut = ws.r1("nef:acut", N);
  if (adaptive && h.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver) {
    dev.adapt_r = ws.r1("nef:adapt_r", N), dev.adapt_dn = ws.r1("nef:adapt_dn", N);
    solver_cutoffs(h, N, roff, re_dist, acut, dev.adapt_r, dev.adapt_dn);
  } else if (adaptive) {
    dev.adapt_eff = grid_cutoffs(ws, h, N, roff, re_dist, probes, P, acut);
  }
  // A ghost has no list here: its cutoff comes from the rank that owns it.
  if (adaptive && dev.exchange) dev.exchange->atoms_forward(acut);

  // Which edges are kept -- those within the mean of their two atoms' cutoffs --
  // their smooth factors, and each atom's count.
  RView1D rcv = ws.r1("nef:rcv", E), factor = ws.r1("nef:factorv", E);
  IView1D keep = ws.i1("nef:keepv", E), count = ws.i1("nef:count", NC);
  Kokkos::deep_copy(ExecSpace(), count, 0);
  Kokkos::parallel_for(
      "pet_keep", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        const double rc = adaptive ? 0.5 * (acut(re_i(e)) + acut(re_j(e))) : cutoff;
        rcv(e) = rc;
        keep(e) = re_dist(e) <= rc;
        if (keep(e)) {
          factor(e) = detail::dev_cutoff_value(re_dist(e), rc, width, bump);
          Kokkos::atomic_inc(&count(re_i(e)));
        }
      });
  // M: this geometry's largest count, or a stepping loop's fixed capacity -- then
  // every slot is an edge row (padding is zero-weight), so the shapes do not
  // change between its neighbour-list rebuilds, and an atom past it raises
  // `overflow` for the caller to grow M and redo.
  int M = m_fixed;
  if (m_fixed > 0) {
    dev.n_edges = NC * M, dev.padded = true;
    Kokkos::parallel_for(
        "pet_overflow", RangePolicy(0, NC), KOKKOS_LAMBDA(int a) {
          if (count(a) > m_fixed) overflow(0) = 1;
        });
  } else {
    // Both into device memory, then one readback.
    IView1D mn = ws.i1("nef:m_n", 2);
    Kokkos::parallel_reduce(
        "pet_maxM", RangePolicy(0, NC), KOKKOS_LAMBDA(int a, int& m) { m = count(a) > m ? count(a) : m; },
        Kokkos::Max<int, MemSpace>(Kokkos::subview(mn, 0)));
    Kokkos::parallel_reduce(
        "pet_nkept", RangePolicy(0, NC), KOKKOS_LAMBDA(int a, int& c) { c += count(a); },
        Kokkos::Sum<int, MemSpace>(Kokkos::subview(mn, 1)));
    const auto hmn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mn);
    M = hmn(0), dev.n_edges = hmn(1);
    M = std::max(M, 1);
    if (m_high > 0) M = m_high = std::max(M, m_high);
    dev.padded = false;
  }
  const int S = M + 1, NM = NC * M;
  dev.max_neighbors = M;

  // The kept edges into their atoms' slots, in edge order: slot = the number of
  // the atom's kept edges before this one, an ordered warp scan.
  auto neigh_species = dev.neigh_species = ws.i1("nef:out:neigh_species", NM);
  auto reverse = dev.reverse_index = ws.i1("nef:out:reverse", NM);
  auto edge_vec = dev.edge_vec = ws.r2("nef:out:edge_vec", NM, 3);
  auto dist = dev.dist = ws.r1("nef:out:dist", NM);
  auto mask = dev.mask = ws.r1("nef:out:mask", NM);
  auto pcut = dev.pair_cutoff = ws.r1("nef:out:pcut", NM);
  auto cut = dev.cutoff_factor = ws.n1("nef:out:cutoff", NM);
  auto cf_seq = dev.cf_seq = ws.n2("nef:out:cf_seq", NC, S);
  IView1D flat = dev.raw_slot = ws.i1("nef:flat_edge", E);
  Kokkos::deep_copy(ExecSpace(), reverse, -1);
  Kokkos::deep_copy(ExecSpace(), flat, -1);
  Kokkos::parallel_for(
      "pet_scatter", AtomTeams(NC, 1, kLanes), KOKKOS_LAMBDA(const Atom& t) {
        const int i = t.league_rank(), e0 = roff(i);
        // A zero-based range, offset by hand: the CUDA vector scan of some Kokkos
        // releases (5.0.x) ignores a ThreadVectorRange's begin.
        Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, roff(i + 1) - e0), [&](int q, int& s, bool final) {
          const int e = e0 + q;
          if (final && keep(e) && s < M) {
            const int f = flat(e) = i * M + s;
            for (int c = 0; c < 3; ++c) edge_vec(f, c) = re_vec(e, c);
            dist(f) = re_dist(e);
            neigh_species(f) = species(re_j(e));
            cut(f) = static_cast<Net>(factor(e));
            pcut(f) = rcv(e);
            mask(f) = 1.0;
          }
          s += keep(e);
        });
      });
  Kokkos::parallel_for(
      "pet_reverse", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        if (flat(e) >= 0) reverse(flat(e)) = raw_rev(e) >= 0 ? flat(raw_rev(e)) : -1;
      });
  // The attention bias source: 1 for the central token, then the factors.
  Kokkos::parallel_for(
      "pet_cfseq", RangePolicy(0, NC * S), KOKKOS_LAMBDA(int i) {
        const int n = i / S, s = i % S;
        cf_seq(n, s) = s == 0 ? static_cast<Net>(1.0) : cut(n * M + s - 1);
      });
}

DeviceEdgeData build_device_edge_data(Workspace& ws, EdgeMap& edge_map, int& m_high, RView1D probes,
                                      int P, int N, IView1D species, IView1D re_i, IView1D re_j,
                                      IView2D re_shift, RView2D re_vec, RView1D re_dist, int E,
                                      const Hypers& h) {
  DeviceEdgeData dev;
  edge_topology(ws, edge_map, dev, N, species, re_i, re_j, re_shift, re_vec, re_dist, E);
  edge_geometry(ws, dev, h, probes, P, m_high, 0, IView1D());
  return dev;
}

DeviceEdgeData build_device_edge_data(const System& sys, const Hypers& h,
                                      const std::vector<int>& species_to_index) {
  const int N = sys.n_atoms;
  if (N > detail::PET_KEY_ATOM_MAX) throw std::runtime_error("build_device_edge_data: too many atoms for the edge key");
  std::vector<int> sp(N);
  for (int i = 0; i < N; ++i) {
    const int Z = sys.atomic_numbers[i];
    sp[i] = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
    if (sp[i] < 0) throw std::runtime_error("unsupported atomic number in system");
  }
  const auto raw = detail::build_raw_edges(sys, h.cutoff);
  const int E = raw.size();
  std::vector<int> i_(E), j_(E), sh(3 * E);
  std::vector<double> v(3 * E), d(E);
  for (int e = 0; e < E; ++e) {
    const auto& r = raw[e];
    if (std::abs(r.sa) > detail::PET_KEY_SHIFT_BIAS || std::abs(r.sb) > detail::PET_KEY_SHIFT_BIAS ||
        std::abs(r.sc) > detail::PET_KEY_SHIFT_BIAS)
      throw std::runtime_error("build_device_edge_data: cell shift too large for the edge key");
    i_[e] = r.i, j_[e] = r.j, d[e] = r.dist;
    sh[3 * e] = r.sa, sh[3 * e + 1] = r.sb, sh[3 * e + 2] = r.sc;
    v[3 * e] = r.vx, v[3 * e + 1] = r.vy, v[3 * e + 2] = r.vz;
  }
  auto up = [](const auto& host, auto dev) {
    Kokkos::deep_copy(dev, Kokkos::View<const typename decltype(dev)::value_type*, Kokkos::HostSpace>(
                               host.data(), host.size()));
    return dev;
  };
  IView1D species = up(sp, IView1D("species", N)), re_i = up(i_, IView1D("re_i", E)), re_j = up(j_, IView1D("re_j", E));
  IView2D re_shift("re_shift", E, 3);
  RView2D re_vec("re_vec", E, 3);
  RView1D re_dist = up(d, RView1D("re_dist", E));
  up(sh, IView1D(re_shift.data(), 3 * E));
  up(v, RView1D(re_vec.data(), 3 * E));

  std::vector<double> ph;
  for (double p = 0.5; p < h.cutoff - 1e-12; p += h.cutoff_width_adaptive / 4.0) ph.push_back(p);
  const int P = ph.size();
  RView1D probes("probes", std::max(P, 1));
  up(ph, RView1D(probes.data(), P));

  Workspace ws;
  EdgeMap edge_map(E > 0 ? 2 * E + 16 : 16);
  int m_high = 0;
  return build_device_edge_data(ws, edge_map, m_high, probes, P, N, species, re_i, re_j, re_shift, re_vec,
                                re_dist, E, h);
}

}  // namespace pet
