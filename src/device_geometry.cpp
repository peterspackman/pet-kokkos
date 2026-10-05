// The device-resident raw neighbour search: see device_geometry.hpp.
#include "pet/device_cell_list.hpp"
#include "pet/device_geometry.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pet {

namespace {


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

// Positions wrapped into each cell for the search, in a separate buffer, with
// the lattice cell each atom was folded by. pos stays unwrapped, which is what
// lets the Verlet cache state its images in unwrapped terms, where they survive
// an atom crossing a cell face. Only periodic axes are wrapped.
struct Wrapped {
  RView2D cinv, posw;  // [B,9] inverse cells; [N,3]
  IView2D wrapi;       // [N,3]
};

Wrapped wrap_positions(Workspace& ws, const DeviceGeom& g) {
  const int N = g.Ntot, B = g.B;
  auto pos = g.pos, scell = g.scell;
  auto sid = g.sid, sper = g.sper;
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
  return {cinv, posw, wrapi};
}

// Every pair within `radius`, each atom's contiguously in a fixed order: from a
// cell list for large batches, a brute-force search for small ones. Images are
// relative to the wrapped positions.
RawEdges search(Workspace& ws, const DeviceGeom& g, const Wrapped& w, double radius) {
  const int N = g.Ntot;
  const BruteSearch bs{w.posw, g.scell, g.sid, g.soff, g.scnt, g.sper};
  bool cells = device_search() == DeviceSearch::CellList;
  if (device_search() == DeviceSearch::Auto) {
    cells = N >= kCellListMinAtoms;
    if (!cells) {  // the brute force's worst thread: atoms x images of its structure (see kCellListMinChecks)
      auto scnt = g.scnt;
      int worst = 0;
      Kokkos::parallel_reduce(
          "pet_search_work", RangePolicy(0, g.B),
          KOKKOS_LAMBDA(int b, int& m) {
            int rng[3];
            bs.ranges(b, radius, rng);
            m = Kokkos::max(m, scnt(b) * (2 * rng[0] + 1) * (2 * rng[1] + 1) * (2 * rng[2] + 1));
          },
          Kokkos::Max<int>(worst));
      cells = worst >= kCellListMinChecks;
    }
  }
  if (cells) return build_raw_edges_cells(ws, g, w.posw, w.cinv, radius);

  auto sid = g.sid;
  const double r2 = radius * radius;
  RawEdges re;
  IView1D ecnt = ws.i1("pet_ecnt", N), eoff = re.offsets = ws.i1("pet_eoff", N + 1);
  Kokkos::parallel_for(
      "pet_neigh_count", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
        int rng[3], n = 0;
        bs.ranges(sid(gi), radius, rng);
        bs.visit(gi, rng, r2, [&](int, int, int, int, const double*, double) { ++n; });
        ecnt(gi) = n;
      });
  const int E = re.count = prefix_sum(ecnt, eoff, N);
  auto re_i = re.i = ws.i1("pet_re_i", E), re_j = re.j = ws.i1("pet_re_j", E);
  auto re_shift = re.shift = ws.i2("pet_re_shift", E, 3);
  auto re_vec = re.vec = ws.r2("pet_re_vec", E, 3);
  auto re_dist = re.dist = ws.r1("pet_re_dist", E);
  Kokkos::parallel_for(
      "pet_neigh_fill", RangePolicy(0, N), KOKKOS_LAMBDA(int gi) {
        int rng[3], e = eoff(gi);
        bs.ranges(sid(gi), radius, rng);
        bs.visit(gi, rng, r2, [&](int j, int sa, int sb, int sc, const double* v, double d2) {
          re_i(e) = gi, re_j(e) = j;
          re_shift(e, 0) = sa, re_shift(e, 1) = sb, re_shift(e, 2) = sc;
          re_vec(e, 0) = v[0], re_vec(e, 1) = v[1], re_vec(e, 2) = v[2];
          re_dist(e++) = Kokkos::sqrt(d2);
        });
      });
  return re;
}

// Search again: the pairs within cutoff + skin, their images unwrapped, their
// topology, and the batch they were found in, for the checks that follow.
void cache_search(NefCache& c, Workspace& ws, const DeviceGeom& g, double reach) {
  const int N = g.Ntot, B = g.B;
  const Wrapped w = wrap_positions(ws, g);
  const RawEdges re = search(ws, g, w, reach);
  const int E = re.count;
  auto wrapi = w.wrapi;
  auto ri = re.i, rj = re.j;
  auto rs = re.shift;
  IView1D ci = c.ws.i1("ref:i", E), cj = c.ws.i1("ref:j", E);
  IView2D cs = c.ws.i2("ref:shift", E, 3);
  Kokkos::parallel_for(
      "nef_unwrap", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        const int i = ri(e), j = rj(e);
        ci(e) = i, cj(e) = j;
        for (int d = 0; d < 3; ++d) cs(e, d) = rs(e, d) - wrapi(j, d) + wrapi(i, d);
      });
  c.ref = DeviceEdgeData();
  edge_topology(c.ws, c.map, c.ref, N, g.spec, ci, cj, cs, c.ws.r2("ref:vec", E, 3), c.ws.r1("ref:dist", E), E);
  c.ref_shift = cs;

  c.pos0 = c.ws.r2("ref:pos0", N, 3), c.cinv0 = c.ws.r2("ref:cinv0", B, 9);
  Kokkos::deep_copy(c.pos0, Kokkos::subview(g.pos, Kokkos::make_pair(0, N), Kokkos::ALL));
  Kokkos::deep_copy(c.cinv0, Kokkos::subview(w.cinv, Kokkos::make_pair(0, B), Kokkos::ALL));
  auto soff = g.soff, scnt = g.scnt, staged = g.staged, roff = c.ref.raw_off;
  auto rsoff = c.ref_soff = c.ws.i1("ref:soff", B), rscnt = c.ref_scnt = c.ws.i1("ref:scnt", B);
  auto reoff = c.ref_eoff = c.ws.i1("ref:eoff", B + 1);
  auto cur_staged = c.cur_staged = c.ws.i1("cur:staged", B);
  int cap = 0;
  Kokkos::parallel_reduce(
      "nef_slots", RangePolicy(0, B),
      KOKKOS_LAMBDA(int b, int& m) {
        rsoff(b) = soff(b), rscnt(b) = scnt(b), reoff(b) = roff(soff(b)), cur_staged(b) = staged(b);
        if (b == B - 1) reoff(B) = roff(N);
        m = Kokkos::max(m, staged(b) + 1);
      },
      Kokkos::Max<int>(cap));
  auto slot_of = c.slot_of = c.ws.i1("ref:slot_of", std::max(cap, 1));
  Kokkos::deep_copy(ExecSpace(), slot_of, -1);
  Kokkos::parallel_for("nef_slot_of", RangePolicy(0, B), KOKKOS_LAMBDA(int b) { slot_of(staged(b)) = b; });

  c.cur = c.ref, c.cur_shift = cs, c.cur_B = B, c.cur_N = N, c.valid = true;
}

// The last search's pairs for a smaller batch: each slot's structure's block,
// renumbered for where the structure now sits. Edges never cross structures,
// so each block's partners stay inside it.
void cache_reassemble(NefCache& c, const DeviceGeom& g, IView1D slot_ref, IView1D slot_ecnt, int E) {
  const int N = g.Ntot, B = g.B;
  IView1D eo = c.ws.i1("cur:eo", B + 1);
  prefix_sum(slot_ecnt, eo, B);  // reads the total back; E is known, but B is small
  auto ri = c.ref.raw_center, rj = c.ref.raw_neigh, rrev = c.ref.raw_reverse, roff = c.ref.raw_off;
  auto rs = c.ref_shift;
  auto rsoff = c.ref_soff, reoff = c.ref_eoff;
  auto soff = g.soff, sid = g.sid;
  auto ci = c.ws.i1("cur:i", E), cj = c.ws.i1("cur:j", E), crev = c.ws.i1("cur:rev", E);
  auto coff = c.ws.i1("cur:off", N + 1);
  auto cs = c.ws.i2("cur:shift", E, 3);
  Kokkos::parallel_for(
      "nef_reassemble", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        int lo = 0, hi = B - 1;  // the slot whose range holds e
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (eo(mid) <= e) lo = mid;
          else hi = mid - 1;
        }
        const int kc = lo, r = slot_ref(kc), from = reoff(r) + e - eo(kc), shift = soff(kc) - rsoff(r);
        ci(e) = ri(from) + shift, cj(e) = rj(from) + shift;
        crev(e) = rrev(from) < 0 ? -1 : rrev(from) - reoff(r) + eo(kc);
        for (int d = 0; d < 3; ++d) cs(e, d) = rs(from, d);
      });
  Kokkos::parallel_for(
      "nef_reassemble_off", RangePolicy(0, N + 1), KOKKOS_LAMBDA(int i) {
        if (i == N) return (void) (coff(N) = E);
        const int kc = sid(i), r = slot_ref(kc);
        coff(i) = roff(rsoff(r) + i - soff(kc)) - reoff(r) + eo(kc);
      });
  DeviceEdgeData d;
  d.n_atoms = N, d.n_raw = E;
  d.raw_center = ci, d.raw_neigh = cj, d.raw_reverse = crev, d.raw_off = coff;
  d.raw_vec = c.ws.r2("cur:vec", E, 3), d.raw_dist = c.ws.r1("cur:dist", E);
  auto staged = g.staged;
  auto cur_staged = c.cur_staged = c.ws.i1("cur:staged", B);
  Kokkos::deep_copy(ExecSpace(), cur_staged, Kokkos::subview(staged, Kokkos::make_pair(0, B)));
  c.cur = d, c.cur_shift = cs, c.cur_B = B, c.cur_N = N;
}

// The Verlet cache's edges for this geometry. One readback decides between
// reusing the last batch's pairs, reassembling them for a smaller batch, and
// searching again.
//
// A pair outside the cache was more than reach = cutoff + skin apart at the
// search. Write its vector then as u H0 (u fractional, H0 the cell's rows); now
// it is u' H with u' = u + ds_j - ds_i, the atoms' fractional displacements. So
// |v'| >= |u H0 D| - |ds_j H| - |ds_i H| >= reach (1 - |D - I|) - 2 d, where
// D = H0^-1 H is the strain since and d the largest |ds H|. It cannot be inside
// the cutoff while 2 d + reach |D - I|_F < skin, taken per structure; a
// displacement test in Cartesian terms would count a strain as a move.
DeviceEdgeData cached_edges(NefCache& c, Workspace& ws, const DeviceGeom& g, const Hypers& h,
                            const RView1D& probes, int P, int& m_high) {
  const int N = g.Ntot, B = g.B;
  const double skin = NefCache::kSkin, reach = h.cutoff + skin, big = 1e300;
  auto pos = g.pos, scell = g.scell;
  auto sid = g.sid, soff = g.soff, scnt = g.scnt, staged = g.staged;

  bool again = !c.valid;
  if (c.valid) {
    const int cap = c.slot_of.extent(0), cur_B = c.cur_B;
    auto slot_of = c.slot_of, rscnt = c.ref_scnt, rsoff = c.ref_soff, reoff = c.ref_eoff;
    auto cur_staged = c.cur_staged;
    auto cinv0 = c.cinv0, pos0 = c.pos0;
    RView1D eps = c.ws.r1("chk:eps", B), out = c.ws.r1("chk:out", 3);  // worst, set change, edges
    IView1D slot_ref = c.ws.i1("chk:ref", B), slot_ecnt = c.ws.i1("chk:ecnt", B);
    Kokkos::parallel_for(
        "nef_check_cells", RangePolicy(0, B), KOKKOS_LAMBDA(int kc) {
          const int b0 = staged(kc), r = (b0 >= 0 && b0 < cap) ? slot_of(b0) : -1;
          const bool ok = r >= 0 && rscnt(r) == scnt(kc);
          double e2 = 0.0;
          for (int a = 0; ok && a < 3; ++a)
            for (int d = 0; d < 3; ++d) {
              double x = a == d ? -1.0 : 0.0;
              for (int k = 0; k < 3; ++k) x += cinv0(r, 3 * a + k) * scell(kc, 3 * k + d);
              e2 += x * x;
            }
          eps(kc) = ok ? Kokkos::sqrt(e2) : big;
          slot_ref(kc) = r;
          slot_ecnt(kc) = ok ? reoff(r + 1) - reoff(r) : 0;
        });
    using Max = Kokkos::Max<double, MemSpace>;
    Kokkos::parallel_reduce(
        "nef_check_moves", RangePolicy(0, N),
        KOKKOS_LAMBDA(int i, double& m) {
          const int kc = sid(i), r = slot_ref(kc);
          if (!(eps(kc) < big)) return (void) (m = big);
          const int a0 = rsoff(r) + i - soff(kc);
          double ds[3], x[3], d2 = 0.0;
          for (int d = 0; d < 3; ++d) {
            ds[d] = 0.0;
            for (int k = 0; k < 3; ++k) ds[d] += (pos(i, k) - pos0(a0, k)) * cinv0(r, 3 * k + d);
          }
          for (int d = 0; d < 3; ++d) {
            x[d] = ds[0] * scell(kc, d) + ds[1] * scell(kc, 3 + d) + ds[2] * scell(kc, 6 + d);
            d2 += x[d] * x[d];
          }
          m = Kokkos::max(m, 2.0 * Kokkos::sqrt(d2) + reach * eps(kc));
        },
        Max(Kokkos::subview(out, 0)));
    Kokkos::parallel_reduce(
        "nef_check_slots", RangePolicy(0, B),
        KOKKOS_LAMBDA(int kc, double& m) {
          const double lost = eps(kc) < big ? 0.0 : 2.0, moved = (kc >= cur_B || cur_staged(kc) != staged(kc)) ? 1.0 : 0.0;
          m = Kokkos::max(m, Kokkos::max(lost, moved));
        },
        Max(Kokkos::subview(out, 1)));
    Kokkos::parallel_reduce(
        "nef_check_edges", RangePolicy(0, B), KOKKOS_LAMBDA(int kc, double& s) { s += slot_ecnt(kc); },
        Kokkos::Sum<double, MemSpace>(Kokkos::subview(out, 2)));
    const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    const bool worst_ok = N == 0 || o(0) < skin;
    again = !worst_ok || o(1) >= 2.0 || c.ref.orphan_off.extent(0) > 0;
    if (!again && (o(1) >= 1.0 || B != c.cur_B || N != c.cur_N)) cache_reassemble(c, g, slot_ref, slot_ecnt, (int) o(2));
  }
  if (again) cache_search(c, ws, g, reach);

  // This geometry's vectors for the cached pairs; those past the cutoff weigh nothing.
  DeviceEdgeData dev = c.cur;
  const int E = dev.n_raw;
  auto ri = dev.raw_center, rj = dev.raw_neigh;
  auto cs = c.cur_shift;
  auto vec = dev.raw_vec;
  auto dist = dev.raw_dist;
  Kokkos::parallel_for(
      "nef_vectors", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        const int i = ri(e), j = rj(e), b = sid(i);
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
          const double x =
              pos(j, d) + (cs(e, 0) * scell(b, d) + cs(e, 1) * scell(b, 3 + d) + cs(e, 2) * scell(b, 6 + d)) - pos(i, d);
          vec(e, d) = x, d2 += x * x;
        }
        dist(e) = d2 > 1e-24 ? Kokkos::sqrt(d2) : 1e30;  // a coincident pair weighs nothing
      });
  dev.species = g.spec;
  dev.struct_id = sid;
  dev.n_struct = B;
  dev.charge = g.charge, dev.spin_multiplicity = g.spin;
  edge_geometry(ws, dev, h, probes, P, m_high, 0, IView1D());
  return dev;
}

}  // namespace

DeviceEdgeData build_nef_device(const DeviceGeom& g, const Hypers& h, const RView1D& probes, int P,
                                Workspace& ws, EdgeMap& edge_map, int& m_high, NefCache* cache) {
  if (cache) return cached_edges(*cache, ws, g, h, probes, P, m_high);
  const RawEdges re = search(ws, g, wrap_positions(ws, g), h.cutoff);
  DeviceEdgeData dev = build_device_edge_data(ws, edge_map, m_high, probes, P, g.Ntot, g.spec, re.i, re.j,
                                              re.shift, re.vec, re.dist, re.count, h);
  dev.struct_id = g.sid;
  dev.n_struct = g.B;
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
  auto staged = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.staged);
  for (int b = 0, gi = 0; b < B; ++b) {
    const System& s = systems[b];
    soff(b) = gi, scnt(b) = s.n_atoms;
    sper(b) = s.pbc[0] | s.pbc[1] << 1 | s.pbc[2] << 2;
    charge(b) = s.charge, spin(b) = s.spin_multiplicity, staged(b) = b;
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
  Kokkos::deep_copy(g.staged, staged);
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
  dev.n_centres = v.exchange ? dev.n_local : N;  // over several ranks, ghosts are only neighbours
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
