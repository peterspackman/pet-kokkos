#include "pet/device_cell_list.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace pet {

namespace {
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

// Floor division and the matching non-negative remainder. A bin index outside
// [0, n) folds into the box and the quotient is the lattice shift that took it
// there; C's / and % truncate toward zero instead, which gets both wrong for
// negative indices.
KOKKOS_INLINE_FUNCTION int floor_div(int a, int n) {
  const int q = a / n;
  return (a % n != 0 && ((a < 0) != (n < 0))) ? q - 1 : q;
}
KOKKOS_INLINE_FUNCTION int floor_mod(int a, int n) { return a - floor_div(a, n) * n; }

// The cell-list search for one atom, shared by the count and fill passes so the
// two cannot disagree about which candidates are edges.
struct CellSearch {
  IView1D sid, abin, goff, bstart, batom;
  IView2D gnc, gnr;
  RView2D posw, scell;
  double cutoff2;

  // hits(bin_start, n_members, shift) for each of atom gi's candidate bins,
  // where `shift` is that bin's lattice image.
  template <class F>
  KOKKOS_INLINE_FUNCTION void visit(int gi, F&& hits) const {
    const int b = sid(gi);
    const int nca = gnc(b, 0), ncb = gnc(b, 1), ncc = gnc(b, 2);
    const int nra = gnr(b, 0), nrb = gnr(b, 1), nrc = gnr(b, 2);
    const int loc = abin(gi) - goff(b);  // this atom's own bin, from the flat index
    const int ca = loc / (ncb * ncc), cb = (loc / ncc) % ncb, cc = loc % ncc;
    for (int da = -nra; da <= nra; ++da)
      for (int db = -nrb; db <= nrb; ++db)
        for (int dc = -nrc; dc <= nrc; ++dc) {
          // Fold the target bin back into the box; the quotient is the image.
          const int ta = ca + da, tb = cb + db, tc = cc + dc;
          const int sh[3] = {floor_div(ta, nca), floor_div(tb, ncb), floor_div(tc, ncc)};
          const int bin = goff(b) + (floor_mod(ta, nca) * ncb + floor_mod(tb, ncb)) * ncc + floor_mod(tc, ncc);
          hits(bstart(bin), bstart(bin + 1) - bstart(bin), sh);
        }
  }
  // v <- the vector from gi to candidate j's image, v[3] <- its square; true if an edge.
  KOKKOS_INLINE_FUNCTION bool edge(int gi, int j, const int* sh, double* v) const {
    const int b = sid(gi);
    for (int d = 0; d < 3; ++d)  // in this order: the image offset first, then the atom
      v[d] = posw(j, d) + (sh[0] * scell(b, d) + sh[1] * scell(b, 3 + d) + sh[2] * scell(b, 6 + d) - posw(gi, d));
    v[3] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    return v[3] >= 1e-24 && v[3] <= cutoff2;
  }
};

}  // namespace

DeviceSearch& device_search() {
  static DeviceSearch s = [] {
    const char* e = std::getenv("PET_DEVICE_SEARCH");
    if (e) {
      const std::string v(e);
      if (v == "cells") return DeviceSearch::CellList;
      if (v == "brute") return DeviceSearch::BruteForce;
    }
    return DeviceSearch::Auto;
  }();
  return s;
}

RawEdges build_raw_edges_cells(Workspace& ws, const DeviceGeom& g, const RView2D& posw,
                               const RView2D& cinv, double cutoff) {
  const int Ntot = g.Ntot, B = g.B;
  const double cutoff2 = cutoff * cutoff;
  const IView1D& sid = g.sid;
  const IView1D& sper = g.sper;
  const RView2D& scell = g.scell;

  RawEdges out;

  // ---- per-structure grid geometry ----------------------------------------
  // gnc: bins along each lattice direction. gnr: how many bins out to search.
  // gbins: this structure's bin block size, and goff its start in the global
  // bin numbering, so one flat array holds every structure's bins.
  IView2D gnc = ws.i2("cl_nc", B, 3), gnr = ws.i2("cl_nr", B, 3);
  IView1D gbins = ws.i1("cl_bins", B), goff = ws.i1("cl_off", B + 1);
  Kokkos::parallel_for(
      "cl_grid", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
        const double a0=scell(b,0),a1=scell(b,1),a2=scell(b,2),
                     b0=scell(b,3),b1=scell(b,4),b2=scell(b,5),
                     c0=scell(b,6),c1=scell(b,7),c2=scell(b,8);
        // Interplanar spacings: volume over the area of the opposite face.
        const double bc[3] = {b1*c2-b2*c1, b2*c0-b0*c2, b0*c1-b1*c0};
        const double ca[3] = {c1*a2-c2*a1, c2*a0-c0*a2, c0*a1-c1*a0};
        const double ab[3] = {a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0};
        const double vol = Kokkos::fabs(a0*bc[0] + a1*bc[1] + a2*bc[2]);
        const double area[3] = {Kokkos::sqrt(bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]),
                                Kokkos::sqrt(ca[0]*ca[0]+ca[1]*ca[1]+ca[2]*ca[2]),
                                Kokkos::sqrt(ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2])};
        int nb = 1;
        for (int d = 0; d < 3; ++d) {
          int nc = 1, nr = 0;
          if (sper(b) && vol > 1e-12 && area[d] > 0.0) {
            const double perp = vol / area[d];
            nc = (int) Kokkos::floor(perp / cutoff);
            if (nc < 1) nc = 1;
            // Bin width along d. At least `cutoff` whenever the box is, so the
            // search reaches one bin; narrower boxes need proportionally more.
            const double w = perp / (double) nc;
            nr = (int) Kokkos::ceil(cutoff / w);
            if (nr < 1) nr = 1;
          }
          gnc(b, d) = nc;
          gnr(b, d) = nr;
          nb *= nc;
        }
        gbins(b) = nb;
      });
  Kokkos::parallel_scan(
      "cl_binscan", RangePolicy(0, B), KOKKOS_LAMBDA(int b, int& upd, bool final) {
        if (final) goff(b) = upd;
        upd += gbins(b);
        if (final && b == B - 1) goff(B) = upd;
      });
  int nbins = 0;
  Kokkos::deep_copy(nbins, Kokkos::subview(goff, B));
  if (nbins <= 0) nbins = 1;

  // ---- bin every atom ------------------------------------------------------
  IView1D abin = ws.i1("cl_abin", Ntot);
  Kokkos::parallel_for(
      "cl_assign", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
        const int b = sid(gi);
        int ijk[3] = {0, 0, 0};
        if (sper(b)) {
          const double x = posw(gi, 0), y = posw(gi, 1), z = posw(gi, 2);
          // posw is already wrapped, so the fractional coordinate is in [0, 1);
          // the clamp only guards the boundary, where rounding can land on 1.0.
          for (int d = 0; d < 3; ++d) {
            const double f = x * cinv(b, d) + y * cinv(b, 3 + d) + z * cinv(b, 6 + d);
            int c = (int) Kokkos::floor(f * (double) gnc(b, d));
            if (c < 0) c = 0;
            if (c >= gnc(b, d)) c = gnc(b, d) - 1;
            ijk[d] = c;
          }
        }
        abin(gi) = goff(b) + (ijk[0] * gnc(b, 1) + ijk[1]) * gnc(b, 2) + ijk[2];
      });

  // Counting sort of atoms by bin.
  // Zeroed explicitly: atomic accumulators must not depend on the pool's
  // zeroing policy, which a caller elsewhere may have turned off. See the note
  // in PetModel::energy_forces_batch.
  IView1D bcnt = ws.i1("cl_bcnt", nbins);
  Kokkos::deep_copy(ExecSpace(), bcnt, 0);
  IView1D bstart = ws.i1("cl_bstart", nbins + 1);
  Kokkos::parallel_for(
      "cl_count", RangePolicy(0, Ntot),
      KOKKOS_LAMBDA(int gi) { Kokkos::atomic_inc(&bcnt(abin(gi))); });
  Kokkos::parallel_scan(
      "cl_scan", RangePolicy(0, nbins), KOKKOS_LAMBDA(int c, int& upd, bool final) {
        if (final) bstart(c) = upd;
        upd += bcnt(c);
        if (final && c == nbins - 1) bstart(nbins) = upd;
      });
  IView1D fill = ws.i1("cl_fill", nbins);
  Kokkos::deep_copy(ExecSpace(), fill, 0);  // atomic accumulator
  IView1D bunsorted = ws.i1("cl_batom_fill", Ntot), batom = ws.i1("cl_batom", Ntot);
  Kokkos::parallel_for(
      "cl_fill", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
        const int c = abin(gi);
        bunsorted(bstart(c) + Kokkos::atomic_fetch_add(&fill(c), 1)) = gi;
      });
  // The fill above used an atomic counter, so membership order is thread-arrival
  // order and differs run to run. Put each bin in atom-index order -- without it
  // the neighbour list, and therefore the energy's last bits, would not be
  // reproducible (see the header). One warp per bin, each lane placing its atoms
  // at their rank among the bin's: a big cell at a long cutoff has only a few
  // bins of hundreds of atoms, and a sequential sort per bin was a long chain.
  Kokkos::parallel_for(
      "cl_sortbins", Kokkos::TeamPolicy<ExecSpace>(nbins, 1, 32),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& t) {
        const int lo = bstart(t.league_rank()), n = bstart(t.league_rank() + 1) - lo;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, n), [&](int q) {
          const int a = bunsorted(lo + q);
          int rank = 0;
          for (int p = 0; p < n; ++p) rank += bunsorted(lo + p) < a;
          batom(lo + rank) = a;
        });
      });

  // ---- count, scan, fill ---------------------------------------------------
  // One warp per atom: its lanes take the members of each candidate bin. (A
  // thread per atom left a 1728-atom cell with ~54 warps on the whole GPU, each
  // walking ~27 bins.) The fill pass places each hit with an ordered warp scan,
  // so the edge list comes out in exactly the order a sequential walk would give
  // -- bins in fixed order, members sorted by atom index.
  using Teams = Kokkos::TeamPolicy<ExecSpace>;
  using Team = Teams::member_type;
  IView1D ecnt = ws.i1("cl_ecnt", Ntot);
  IView1D eoff = ws.i1("cl_eoff", Ntot + 1);
  const CellSearch cs{sid, abin, goff, bstart, batom, gnc, gnr, posw, scell, cutoff2};
  Kokkos::parallel_for(
      "cl_ecount", Teams(Ntot, 1, 32), KOKKOS_LAMBDA(const Team& t) {
        const int gi = t.league_rank();
        int e = 0;
        cs.visit(gi, [&](int k0, int nk, const int* sh) {
          int found = 0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nk), [&](int q, int& c) {
            double v[4];
            c += cs.edge(gi, batom(k0 + q), sh, v);
          }, found);
          e += found;
        });
        Kokkos::single(Kokkos::PerThread(t), [&] { ecnt(gi) = e; });
      });
  Kokkos::parallel_scan(
      "cl_escan", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi, int& upd, bool final) {
        if (final) eoff(gi) = upd;
        upd += ecnt(gi);
        if (final && gi == Ntot - 1) eoff(Ntot) = upd;
      });
  int E = 0;
  Kokkos::deep_copy(E, Kokkos::subview(eoff, Ntot));

  IView1D re_i = ws.i1("pet_re_i", E > 0 ? E : 1);
  IView1D re_j = ws.i1("pet_re_j", E > 0 ? E : 1);
  IView2D re_shift = ws.i2("pet_re_shift", E > 0 ? E : 1, 3);
  RView2D re_vec = ws.r2("pet_re_vec", E > 0 ? E : 1, 3);
  RView1D re_dist = ws.r1("pet_re_dist", E > 0 ? E : 1);
  // The fill records how many edges it actually wrote, and that must equal what
  // the count pass promised.
  //
  // This is not paranoia. The two passes are separate instantiations of the same
  // search, and nvcc is free to contract `vx*vx + vy*vy + vz*vz` into FMAs
  // differently in each -- the store-free count pass has different register
  // pressure. An edge sitting exactly on the cutoff can then be counted by one
  // pass and not the other, and the fill walks past the region the scan reserved
  // for that atom, into the next atom's edges. On a freshly allocated buffer
  // that overrun lands in slack and nothing happens; once the pool is warm it
  // corrupts a live buffer. Silent either way.
  IView1D wrote = ws.i1("cl_wrote", Ntot);
  Kokkos::parallel_for(
      "cl_efill", Teams(Ntot, 1, 32), KOKKOS_LAMBDA(const Team& t) {
        const int gi = t.league_rank();
        int e = eoff(gi);
        cs.visit(gi, [&](int k0, int nk, const int* sh) {
          int found = 0;
          Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, nk), [&](int q, int& pos, bool final) {
            const int j = batom(k0 + q);
            double v[4];
            const bool hit = cs.edge(gi, j, sh, v);
            if (final && hit) {
              const int w = e + pos;
              re_i(w) = gi;
              re_j(w) = j;
              for (int d = 0; d < 3; ++d) re_shift(w, d) = sh[d], re_vec(w, d) = v[d];
              re_dist(w) = Kokkos::sqrt(v[3]);
            }
            pos += hit;
          }, found);
          e += found;
        });
        Kokkos::single(Kokkos::PerThread(t), [&] { wrote(gi) = e - eoff(gi); });
      });
  {
    int mismatch = 0;
    Kokkos::parallel_reduce(
        "cl_check", RangePolicy(0, Ntot),
        KOKKOS_LAMBDA(int gi, int& acc) { acc += (wrote(gi) != ecnt(gi)) ? 1 : 0; },
        mismatch);
    if (mismatch != 0)
      throw std::runtime_error(
          "pet: the cell list's count and fill passes disagreed for " +
          std::to_string(mismatch) +
          " atoms. The two passes must find exactly the same edges; if they do not, "
          "the fill overruns the region the prefix sum reserved. See device_cell_list.cpp.");
  }

  out.i = re_i;
  out.j = re_j;
  out.shift = re_shift;
  out.vec = re_vec;
  out.dist = re_dist;
  out.offsets = eoff;
  out.count = E;
  return out;
}

}  // namespace pet
