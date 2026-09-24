// The device-resident raw neighbour search: see device_geometry.hpp.
#include "pet/device_cell_list.hpp"
#include "pet/device_geometry.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pet {

namespace {

using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

// The brute-force search for one atom: every image within `rng` of every atom
// of its structure, images outermost, in a fixed order. hit(j, sa, sb, sc, v, d2)
// for each pair closer than sqrt(r2) (the atom itself excluded).
struct BruteSearch {
  RView2D posw, scell;
  IView1D sid, soff, scnt, sper;

  template <class Hit>
  KOKKOS_INLINE_FUNCTION void visit(int gi, const int rng[3], double r2, Hit&& hit) const {
    const int b = sid(gi), j0 = soff(b), j1 = soff(b) + scnt(b);
    const double xi = posw(gi, 0), yi = posw(gi, 1), zi = posw(gi, 2);
    for (int sa = -rng[0]; sa <= rng[0]; ++sa)
      for (int sb = -rng[1]; sb <= rng[1]; ++sb)
        for (int sc = -rng[2]; sc <= rng[2]; ++sc) {
          const double shx = sa * scell(b, 0) + sb * scell(b, 3) + sc * scell(b, 6) - xi;
          const double shy = sa * scell(b, 1) + sb * scell(b, 4) + sc * scell(b, 7) - yi;
          const double shz = sa * scell(b, 2) + sb * scell(b, 5) + sc * scell(b, 8) - zi;
          for (int j = j0; j < j1; ++j) {
            const double v[3] = {posw(j, 0) + shx, posw(j, 1) + shy, posw(j, 2) + shz};
            const double d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (d2 >= 1e-24 && d2 <= r2) hit(j, sa, sb, sc, v, d2);
          }
        }
  }
  // Image ranges for `radius` in structure b's cell, now: a cell relaxation can
  // need more images than the starting cell did.
  KOKKOS_INLINE_FUNCTION void ranges(int b, double radius, int rng[3]) const {
    const double c[9] = {scell(b, 0), scell(b, 1), scell(b, 2), scell(b, 3), scell(b, 4),
                         scell(b, 5), scell(b, 6), scell(b, 7), scell(b, 8)};
    image_ranges_rows(c, radius, sper(b), rng);
  }
};

// offsets(i) = the sum of counts before atom i, and the total.
int prefix_sum(IView1D counts, IView1D offsets, int N) {
  Kokkos::parallel_scan(
      "pet_edge_scan", RangePolicy(0, N), KOKKOS_LAMBDA(int i, int& upd, bool final) {
        if (final) offsets(i) = upd;
        upd += counts(i);
        if (final && i == N - 1) offsets(N) = upd;
      });
  int total = 0;
  if (N > 0) Kokkos::deep_copy(total, Kokkos::subview(offsets, N));
  return total;
}

}  // namespace

DeviceEdgeData build_nef_device(const DeviceGeom& g, const Hypers& h, const RView1D& probes, int P,
                                Workspace& ws, EdgeMap& edge_map, int& m_high, NefCache* cache) {
  const int N = g.Ntot, B = g.B;
  auto pos = g.pos, scell = g.scell;
  auto sid = g.sid, sper = g.sper;
  const double cutoff = h.cutoff, cutoff2 = cutoff * cutoff;

  // Wrap positions into each cell for the search, into a separate buffer, and
  // record the lattice cell each atom was folded by. pos stays unwrapped, which
  // is what lets the Verlet cache state its images in unwrapped terms, where they
  // survive an atom crossing a cell face. Only periodic axes are wrapped.
  RView2D cinv = ws.r2("pet_cinv", B, 9), posw = ws.r2("pet_posw", N, 3);
  IView2D wrapi = ws.i2("pet_wrapi", N, 3);
  Kokkos::parallel_for(
      "pet_cellinv", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
        const double a0 = scell(b, 0), a1 = scell(b, 1), a2 = scell(b, 2), b0 = scell(b, 3), b1 = scell(b, 4),
                     b2 = scell(b, 5), c0 = scell(b, 6), c1 = scell(b, 7), c2 = scell(b, 8);
        const double det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
        const double id = (det != 0.0) ? 1.0 / det : 0.0;
        cinv(b, 0) = (b1 * c2 - b2 * c1) * id, cinv(b, 1) = (a2 * c1 - a1 * c2) * id, cinv(b, 2) = (a1 * b2 - a2 * b1) * id;
        cinv(b, 3) = (b2 * c0 - b0 * c2) * id, cinv(b, 4) = (a0 * c2 - a2 * c0) * id, cinv(b, 5) = (a2 * b0 - a0 * b2) * id;
        cinv(b, 6) = (b0 * c1 - b1 * c0) * id, cinv(b, 7) = (a1 * c0 - a0 * c1) * id, cinv(b, 8) = (a0 * b1 - a1 * b0) * id;
      });
  Kokkos::parallel_for(
      "pet_wrap", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
        const int b = sid(i);
        if (!sper(b)) {
          for (int d = 0; d < 3; ++d) posw(i, d) = pos(i, d), wrapi(i, d) = 0;
          return;
        }
        const double x = pos(i, 0), y = pos(i, 1), z = pos(i, 2);
        double f[3];
        for (int d = 0; d < 3; ++d) {
          f[d] = x * cinv(b, d) + y * cinv(b, 3 + d) + z * cinv(b, 6 + d);  // fractional
          const double w = (sper(b) >> d & 1) ? Kokkos::floor(f[d]) : 0.0;
          f[d] -= w;
          wrapi(i, d) = (int) w;
        }
        for (int d = 0; d < 3; ++d) posw(i, d) = f[0] * scell(b, d) + f[1] * scell(b, 3 + d) + f[2] * scell(b, 6 + d);
      });

  // The raw edge list: from the cell list, a brute-force search, or the Verlet
  // cache. All three give each atom's edges contiguously, in a fixed order.
  IView1D re_i, re_j;
  IView2D re_shift;
  RView2D re_vec;
  RView1D re_dist;
  int E = 0;
  auto alloc = [&](int n) {
    re_i = ws.i1("pet_re_i", n), re_j = ws.i1("pet_re_j", n), re_shift = ws.i2("pet_re_shift", n, 3);
    re_vec = ws.r2("pet_re_vec", n, 3), re_dist = ws.r1("pet_re_dist", n);
  };
  const BruteSearch bs{posw, scell, sid, g.soff, g.scnt, sper};
  const bool use_cells = !cache && (device_search() == DeviceSearch::CellList ||
                                    (device_search() == DeviceSearch::Auto && N >= kCellListMinAtoms));

  if (use_cells) {
    RawEdges re = build_raw_edges_cells(ws, g, posw, cinv, cutoff);
    re_i = re.i, re_j = re.j, re_shift = re.shift, re_vec = re.vec, re_dist = re.dist, E = re.count;
  } else if (!cache) {
    IView1D ecnt = ws.i1("pet_ecnt", N), eoff = ws.i1("pet_eoff", N + 1);
    Kokkos::parallel_for(
        "pet_neigh_count", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
          int rng[3], n = 0;
          bs.ranges(sid(gi), cutoff, rng);
          bs.visit(gi, rng, cutoff2, [&](int, int, int, int, const double*, double) { ++n; });
          ecnt(gi) = n;
        });
    alloc(E = prefix_sum(ecnt, eoff, N));
    Kokkos::parallel_for(
        "pet_neigh_fill", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
          int rng[3], e = eoff(gi);
          bs.ranges(sid(gi), cutoff, rng);
          bs.visit(gi, rng, cutoff2, [&](int j, int sa, int sb, int sc, const double* v, double d2) {
            re_i(e) = gi, re_j(e) = j;
            re_shift(e, 0) = sa, re_shift(e, 1) = sb, re_shift(e, 2) = sc;
            re_vec(e, 0) = v[0], re_vec(e, 1) = v[1], re_vec(e, 2) = v[2];
            re_dist(e++) = Kokkos::sqrt(d2);
          });
        });
  } else {
    // The Verlet cache: the topology within cutoff + skin, searched only when an
    // atom has moved, or the cell strained, far enough to bring a new pair
    // inside the cutoff; each call recomputes the vectors and keeps those within
    // the cutoff.
    const double skin = 1.0, reach = cutoff + skin, reach2 = reach * reach;
    bool rebuild = !cache->valid || cache->built_Ntot != N;
    if (!rebuild) {
      // Displacement since the build, on unwrapped positions.
      double d2max = 0;
      auto pos0 = cache->pos0;
      Kokkos::parallel_reduce(
          "nef_disp", RangePolicy(0, N),
          KOKKOS_LAMBDA(int i, double& m) {
            const double dx = pos(i, 0) - pos0(i, 0), dy = pos(i, 1) - pos0(i, 1), dz = pos(i, 2) - pos0(i, 2);
            m = Kokkos::max(m, dx * dx + dy * dy + dz * dz);
          },
          Kokkos::Max<double>(d2max));
      // A strain of relative size eps moves a pair up to `reach` apart by at most
      // eps * reach, which the displacement test cannot see.
      double eps = 0;
      auto cell0 = cache->cell0;
      Kokkos::parallel_reduce(
          "nef_cellchg", RangePolicy(0, B),
          KOKKOS_LAMBDA(int b, double& m) {
            for (int o = 0; o < 9; o += 3) {
              const double bx = cell0(b, o), by = cell0(b, o + 1), bz = cell0(b, o + 2);
              const double dx = scell(b, o) - bx, dy = scell(b, o + 1) - by, dz = scell(b, o + 2) - bz;
              const double len = Kokkos::sqrt(bx * bx + by * by + bz * bz);
              if (len > 0) m = Kokkos::max(m, Kokkos::sqrt(dx * dx + dy * dy + dz * dz) / len);
            }
          },
          Kokkos::Max<double>(eps));
      rebuild = d2max > 0.25 * skin * skin || eps * reach > 0.5 * skin;
    }
    if (rebuild) {
      // Image ranges for cutoff + skin: the model cutoff's ranges miss the outer
      // shell the skin reaches into, whose edges would then never be found.
      IView1D ecnt = ws.i1("pet_ecnt", N), eoff = ws.i1("pet_eoff", N + 1);
      Kokkos::parallel_for(
          "pet_neigh_count_s", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
            int rng[3], n = 0;
            bs.ranges(sid(gi), reach, rng);
            bs.visit(gi, rng, reach2, [&](int, int, int, int, const double*, double) { ++n; });
            ecnt(gi) = n;
          });
      const int Es = prefix_sum(ecnt, eoff, N);
      cache->ci = IView1D("nef_ci", std::max(Es, 1)), cache->cj = IView1D("nef_cj", std::max(Es, 1));
      cache->cshift = IView2D("nef_cshift", std::max(Es, 1), 3);
      auto ci = cache->ci, cj = cache->cj;
      auto cshift = cache->cshift;
      Kokkos::parallel_for(
          "pet_neigh_fill_s", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
            int rng[3], e = eoff(gi);
            bs.ranges(sid(gi), reach, rng);
            bs.visit(gi, rng, reach2, [&](int j, int sa, int sb, int sc, const double*, double) {
              // The image in unwrapped terms: undo each end's wrap.
              ci(e) = gi, cj(e) = j;
              cshift(e, 0) = sa - wrapi(j, 0) + wrapi(gi, 0);
              cshift(e, 1) = sb - wrapi(j, 1) + wrapi(gi, 1);
              cshift(e++, 2) = sc - wrapi(j, 2) + wrapi(gi, 2);
            });
          });
      cache->Es = Es, cache->valid = true, cache->built_Ntot = N;
      if ((int) cache->pos0.extent(0) < N) cache->pos0 = RView2D("nef_pos0", N, 3);
      if ((int) cache->cell0.extent(0) < B) cache->cell0 = RView2D("nef_cell0", B, 9);
      Kokkos::deep_copy(Kokkos::subview(cache->pos0, Kokkos::make_pair(0, N), Kokkos::ALL),
                        Kokkos::subview(pos, Kokkos::make_pair(0, N), Kokkos::ALL));
      Kokkos::deep_copy(Kokkos::subview(cache->cell0, Kokkos::make_pair(0, B), Kokkos::ALL),
                        Kokkos::subview(scell, Kokkos::make_pair(0, B), Kokkos::ALL));
    }
    // The cached pairs within the cutoff now, with vectors from the current
    // (unwrapped) positions.
    const int Es = cache->Es;
    auto ci = cache->ci, cj = cache->cj;
    auto cshift = cache->cshift;
    auto vec_of = KOKKOS_LAMBDA(int e, double* v) {
      const int gi = ci(e), j = cj(e), b = sid(gi);
      for (int d = 0; d < 3; ++d)
        v[d] = pos(j, d) + (cshift(e, 0) * scell(b, d) + cshift(e, 1) * scell(b, 3 + d) + cshift(e, 2) * scell(b, 6 + d)) -
               pos(gi, d);
      return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    };
    IView1D foff = ws.i1("pet_foff", Es + 1);
    Kokkos::parallel_scan(
        "nef_filter_scan", RangePolicy(0, Es), KOKKOS_LAMBDA(int e, int& upd, bool final) {
          if (final) foff(e) = upd;
          double v[3];
          const double d2 = vec_of(e, v);
          upd += (d2 >= 1e-24 && d2 <= cutoff2) ? 1 : 0;
          if (final && e == Es - 1) foff(Es) = upd;
        });
    if (Es > 0) Kokkos::deep_copy(E, Kokkos::subview(foff, Es));
    alloc(std::max(E, 1));
    Kokkos::parallel_for(
        "nef_filter_fill", RangePolicy(0, Es), KOKKOS_LAMBDA(int e) {
          double v[3];
          const double d2 = vec_of(e, v);
          if (d2 < 1e-24 || d2 > cutoff2) return;
          const int o = foff(e);
          re_i(o) = ci(e), re_j(o) = cj(e);
          for (int d = 0; d < 3; ++d) re_shift(o, d) = cshift(e, d), re_vec(o, d) = v[d];
          re_dist(o) = Kokkos::sqrt(d2);
        });
  }

  DeviceEdgeData dev =
      build_device_edge_data(ws, edge_map, m_high, probes, P, N, g.spec, re_i, re_j, re_shift, re_vec, re_dist, E, h);
  dev.struct_id = sid;
  dev.n_struct = B;
  dev.charge = g.charge, dev.spin_multiplicity = g.spin;
  return dev;
}

DeviceGeom stage_systems(Workspace& ws, const std::vector<System>& systems,
                         const std::vector<int>& species_to_index) {
  const int B = systems.size();
  int N = 0;
  for (const auto& s : systems) N += s.n_atoms;
  DeviceGeom g = stage_geometry_views(ws, N, B);
  auto pos = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.pos);
  auto sid = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.sid);
  auto spec = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.spec);
  auto soff = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.soff);
  auto scnt = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.scnt);
  auto sper = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.sper);
  auto scell = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.scell);
  auto charge = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.charge);
  auto spin = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.spin);
  for (int b = 0, gi = 0; b < B; ++b) {
    const System& s = systems[b];
    soff(b) = gi, scnt(b) = s.n_atoms;
    sper(b) = s.pbc[0] | s.pbc[1] << 1 | s.pbc[2] << 2;
    charge(b) = s.charge, spin(b) = s.spin_multiplicity;
    for (int e = 0; e < 9; ++e) scell(b, e) = s.cell[e];
    for (int i = 0; i < s.n_atoms; ++i, ++gi) {
      for (int d = 0; d < 3; ++d) pos(gi, d) = s.positions[3 * i + d];
      const int Z = s.atomic_numbers[i];
      sid(gi) = b;
      spec(gi) = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
      if (spec(gi) < 0) throw std::runtime_error("pet: unsupported atomic number " + std::to_string(Z));
    }
  }
  Kokkos::deep_copy(g.pos, pos), Kokkos::deep_copy(g.sid, sid), Kokkos::deep_copy(g.spec, spec);
  Kokkos::deep_copy(g.soff, soff), Kokkos::deep_copy(g.scnt, scnt), Kokkos::deep_copy(g.sper, sper);
  Kokkos::deep_copy(g.scell, scell), Kokkos::deep_copy(g.charge, charge), Kokkos::deep_copy(g.spin, spin);
  return g;
}

DeviceEdgeData build_device_batch(const std::vector<System>& systems, const Hypers& h,
                                  const std::vector<int>& species_to_index, const RView1D& probes, int P,
                                  Workspace& ws, EdgeMap& edge_map, int& m_high, NefCache* cache) {
  return build_nef_device(stage_systems(ws, systems, species_to_index), h, probes, P, ws, edge_map, m_high,
                          cache);
}

void EdgeSession::set(const EdgeListView& v, const std::vector<int>& species_to_index) {
  const int N = v.n_atoms, L = N > 0 ? v.offsets[N] : 0;
  if (N > detail::PET_KEY_ATOM_MAX) throw std::runtime_error("pet: too many atoms for the edge key");
  std::vector<int> sp(N);
  for (int i = 0; i < N; ++i) {
    const int Z = v.atomic_numbers[i];
    sp[i] = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
    if (sp[i] < 0) throw std::runtime_error("pet: unsupported atomic number " + std::to_string(Z));
  }
  // Directed edges grouped by centre: each atom's listed pairs, then the mirrors
  // of the pairs that list it.
  std::vector<int> off(N + 1, 0);
  for (int i = 0; i < N; ++i) {
    off[i + 1] += v.offsets[i + 1] - v.offsets[i];
    if (v.half)
      for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e) ++off[v.neighbors[e] + 1];
  }
  for (int i = 0; i < N; ++i) off[i + 1] += off[i];
  const int E = off[N];
  std::vector<int> ri(E), rj(E), sh(3 * std::size_t(E), 0), cur(off.begin(), off.end() - 1);
  src.assign(E, 0), dir.assign(E, 1);
  auto put = [&](int c, int n, int e, int sign) {
    const int k = cur[c]++;
    ri[k] = c, rj[k] = n, src[k] = e, dir[k] = sign;
    if (!v.shifts) return;
    for (int d = 0; d < 3; ++d) {
      sh[3 * k + d] = sign * v.shifts[3 * e + d];
      if (std::abs(sh[3 * k + d]) > detail::PET_KEY_SHIFT_BIAS) throw std::runtime_error("pet: cell shift too large for the edge key");
    }
  };
  for (int i = 0; i < N; ++i)
    for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e) put(i, v.neighbors[e], e, 1);
  if (v.half)
    for (int i = 0; i < N; ++i)
      for (int e = v.offsets[i]; e < v.offsets[i + 1]; ++e) put(v.neighbors[e], i, e, -1);

  using HostI = Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  IView1D species = ws.i1("md:species", N), re_i = ws.i1("md:re_i", E), re_j = ws.i1("md:re_j", E);
  IView2D re_shift = ws.i2("md:re_shift", E, 3);
  Kokkos::deep_copy(species, HostI(sp.data(), N));
  if (E) {
    Kokkos::deep_copy(re_i, HostI(ri.data(), E)), Kokkos::deep_copy(re_j, HostI(rj.data(), E));
    Kokkos::deep_copy(re_shift, Kokkos::View<const int**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(sh.data(), E, 3));
  }
  dev = DeviceEdgeData();
  edge_topology(ws, map, dev, N, species, re_i, re_j, re_shift, ws.r2("md:re_vec", E, 3), ws.r1("md:re_dist", E), E);
  dev.n_struct = 1;
  dev.n_local = v.n_local < 0 ? N : v.n_local;
  dev.struct_id = ws.i1("md:sid", N);  // zero-filled
  shift = re_shift;
  dev.charge = ws.i1("md:charge", 1), dev.spin_multiplicity = ws.i1("md:spin", 1);
  Kokkos::deep_copy(dev.charge, v.charge), Kokkos::deep_copy(dev.spin_multiplicity, v.spin_multiplicity);
  // Over several ranks, the edges to ghosts, in the engine's order.
  if (v.exchange) {
    if (v.half) throw std::runtime_error("pet: an exchanged list must be full");
    std::vector<int> remote;
    for (int e = 0; e < E; ++e)
      if (rj[e] >= dev.n_local) remote.push_back(e);
    dev.exchange = v.exchange;
    dev.remote_raw = ws.i1("md:remote", remote.size());
    if (!remote.empty()) Kokkos::deep_copy(dev.remote_raw, HostI(remote.data(), remote.size()));
  }
  n_pairs = L, M = 0, shifted = v.shifts != nullptr, valid = true;
}

const DeviceEdgeData& EdgeSession::step(const double* positions, bool on_device, const double* cell,
                                        const Hypers& h, RView1D probes, int P, bool fixed) {
  if (!valid) throw std::runtime_error("pet: no neighbour list set");
  if (shifted && !cell) throw std::runtime_error("pet: an edge list with shifts needs the cell");
  const int N = dev.n_atoms, E = dev.n_raw;
  RView2D pos = ws.r2("md:pos", N, 3);
  if (N && on_device) {
    const Kokkos::View<const double**, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryUnmanaged> x(positions, N, 3);
    Kokkos::parallel_for(
        "md_positions", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
          for (int c = 0; c < 3; ++c) pos(i, c) = x(i, c);
        });
  } else if (N) {
    Kokkos::deep_copy(pos, Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(positions, N, 3));
  }
  double hc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  if (cell) std::copy(cell, cell + 9, hc);
  const double h0 = hc[0], h1 = hc[1], h2 = hc[2], h3 = hc[3], h4 = hc[4], h5 = hc[5], h6 = hc[6], h7 = hc[7],
               h8 = hc[8];
  auto re_i = dev.raw_center, re_j = dev.raw_neigh;
  auto vec = dev.raw_vec;
  auto dist = dev.raw_dist;
  auto sh = shift;
  Kokkos::parallel_for(
      "md_vectors", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        const int i = re_i(e), j = re_j(e), a = sh(e, 0), b = sh(e, 1), c = sh(e, 2);
        const double x = pos(j, 0) + (a * h0 + b * h3 + c * h6) - pos(i, 0);
        const double y = pos(j, 1) + (a * h1 + b * h4 + c * h7) - pos(i, 1);
        const double z = pos(j, 2) + (a * h2 + b * h5 + c * h8) - pos(i, 2);
        const double d2 = x * x + y * y + z * z;
        vec(e, 0) = x, vec(e, 1) = y, vec(e, 2) = z;
        dist(e) = d2 > 1e-24 ? Kokkos::sqrt(d2) : 1e30;  // a coincident pair weighs nothing
      });
  int m_high = 0;
  if (!fixed) {
    edge_geometry(ws, dev, h, probes, P, m_high, 0, IView1D());
    return dev;
  }
  // A capacity from this geometry's largest count, with room to move.
  auto size_to_fit = [&] {
    edge_geometry(ws, dev, h, probes, P, m_high, 0, IView1D());
    M = dev.max_neighbors + std::max(2, dev.max_neighbors / 8);
  };
  if (M == 0) size_to_fit();
  IView1D overflow = ws.i1("md:overflow", 1);
  edge_geometry(ws, dev, h, probes, P, m_high, M, overflow);
  int over = 0;
  Kokkos::deep_copy(over, Kokkos::subview(overflow, 0));
  if (over) {
    size_to_fit();
    edge_geometry(ws, dev, h, probes, P, m_high, M, overflow);
  }
  return dev;
}

}  // namespace pet
