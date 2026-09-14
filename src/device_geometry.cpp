// Definitions for the device geometry staging / raw periodic neighbour search
// declared in pet/device_geometry.hpp.
//
// These live in a translation unit rather than the header because they are
// large, because every consumer would otherwise recompile them (and under nvcc
// that is minutes, not seconds), and because an unused inline KOKKOS_LAMBDA is
// not reliably instantiated for the device.
#include "pet/device_cell_list.hpp"
#include "pet/device_geometry.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pet {

DeviceEdgeData build_nef_device(const DeviceGeom& g, const Hypers& h,
                                       const RView1D& probes, int P, Workspace& ws,
                                       EdgeMap& edge_map, int& m_high,
                                NefCache* cache) {
  using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

  const int Ntot = g.Ntot, B = g.B;
  const RView2D& pos = g.pos;
  const IView1D& sid = g.sid;
  const IView1D& spec = g.spec;
  const IView1D& soff = g.soff;
  const IView1D& scnt = g.scnt;
  const IView2D& srng = g.srng;
  const IView1D& sper = g.sper;
  const RView2D& scell = g.scell;

  const double cutoff = h.cutoff;
  const double cutoff2 = cutoff * cutoff;

  // Wrap positions into each structure's cell before the search: image_ranges
  // assumes wrapped coordinates, and a caller staging symmetry-expanded or
  // integrated geometry hands us UNWRAPPED coordinates, which would truncate
  // the image search and miss neighbours. (Idempotent: re-wrapping wrapped
  // coordinates is a no-op.)
  //
  // Wrapped coordinates go to a SEPARATE buffer, and the integer cell each atom
  // was folded by is recorded alongside. Applying the wrap to `pos` in place is
  // what made the Verlet cache below nearly useless: the cache stores
  // (i, j, image) triples stated relative to wrapped coordinates, so the moment
  // an atom drifts across a cell face its stored image is wrong. The old skin
  // test caught that only because comparing wrapped positions made a face
  // crossing look like a full lattice-vector jump -- an accident that forced a
  // rebuild on 76% of rounds. Keeping `pos` unwrapped and recording the wrap
  // lets the cached topology be stated in unwrapped terms, where it survives a
  // crossing.
  RView2D cinv = ws.r2("pet_cinv", B, 9);       // per-structure scell^{-1}
  RView2D posw = ws.r2("pet_posw", Ntot, 3);    // wrapped positions (search only)
  IView2D wrapi = ws.i2("pet_wrapi", Ntot, 3);  // cell each atom was folded by
  {
    Kokkos::parallel_for("pet_cellinv", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
      const double a0=scell(b,0),a1=scell(b,1),a2=scell(b,2),
                   b0=scell(b,3),b1=scell(b,4),b2=scell(b,5),
                   c0=scell(b,6),c1=scell(b,7),c2=scell(b,8);
      const double det = a0*(b1*c2-b2*c1) - a1*(b0*c2-b2*c0) + a2*(b0*c1-b1*c0);
      const double id = (det != 0.0) ? 1.0/det : 0.0;
      cinv(b,0)=(b1*c2-b2*c1)*id; cinv(b,1)=(a2*c1-a1*c2)*id; cinv(b,2)=(a1*b2-a2*b1)*id;
      cinv(b,3)=(b2*c0-b0*c2)*id; cinv(b,4)=(a0*c2-a2*c0)*id; cinv(b,5)=(a2*b0-a0*b2)*id;
      cinv(b,6)=(b0*c1-b1*c0)*id; cinv(b,7)=(a1*c0-a0*c1)*id; cinv(b,8)=(a0*b1-a1*b0)*id;
    });
    Kokkos::parallel_for("pet_wrap", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int i) {
      const int b = sid(i);
      // An isolated molecule has no box to fold into. Gating the wrap on sper is
      // not optional: a molecule reaching this path without it is torn across
      // the padded box built around it -- a C-N bond came out at 13.8 A.
      if (!sper(b)) {
        posw(i,0)=pos(i,0); posw(i,1)=pos(i,1); posw(i,2)=pos(i,2);
        wrapi(i,0)=0; wrapi(i,1)=0; wrapi(i,2)=0;
        return;
      }
      const double x=pos(i,0), y=pos(i,1), z=pos(i,2);
      double f0 = x*cinv(b,0)+y*cinv(b,3)+z*cinv(b,6);  // fractional = pos . scell^{-1}
      double f1 = x*cinv(b,1)+y*cinv(b,4)+z*cinv(b,7);
      double f2 = x*cinv(b,2)+y*cinv(b,5)+z*cinv(b,8);
      const double w0 = Kokkos::floor(f0), w1 = Kokkos::floor(f1), w2 = Kokkos::floor(f2);
      f0 -= w0; f1 -= w1; f2 -= w2;
      wrapi(i,0) = (int) w0; wrapi(i,1) = (int) w1; wrapi(i,2) = (int) w2;
      posw(i,0) = f0*scell(b,0)+f1*scell(b,3)+f2*scell(b,6);  // back to cartesian
      posw(i,1) = f0*scell(b,1)+f1*scell(b,4)+f2*scell(b,7);
      posw(i,2) = f0*scell(b,2)+f1*scell(b,5)+f2*scell(b,8);
    });
  }

  // Raw periodic neighbour search -> COO edge list (re_*). Uncached: the full
  // O(N^2 x images) search within cutoff every call. Cached: the topology is
  // searched within cutoff+skin only on rebuild and reused, then filtered to the
  // exact within-cutoff edges here (bit-identical input to the NEF packer).
  IView1D re_i, re_j;
  IView2D re_shift;
  RView2D re_vec;
  RView1D re_dist;
  int E = 0;

  // The cell list replaces the brute-force search when it is worth its setup
  // cost. Below a few hundred atoms it is not: building and sorting a grid is
  // several kernel launches, and a CSP batch of eight-atom cells finds its
  // neighbours faster by just looking at all of them. Auto picks per call from
  // the atom count; PET_DEVICE_SEARCH forces either, which is how the two get
  // checked against each other.
  const bool use_cells =
      !cache && (device_search() == DeviceSearch::CellList ||
                 (device_search() == DeviceSearch::Auto && Ntot >= kCellListMinAtoms));

  if (use_cells) {
    RawEdges re = build_raw_edges_cells(ws, g, posw, cinv, cutoff);
    re_i = re.i;
    re_j = re.j;
    re_shift = re.shift;
    re_vec = re.vec;
    re_dist = re.dist;
    E = re.count;
  } else if (!cache) {
    // ---------- uncached brute-force path ----------
    IView1D ecnt = ws.i1("pet_ecnt", Ntot);
    Kokkos::parallel_for(
        "pet_neigh_count", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
          const int b = sid(gi);
          const double xi = posw(gi, 0), yi = posw(gi, 1), zi = posw(gi, 2);
          const int ra = srng(b, 0), rb = srng(b, 1), rc = srng(b, 2);
          const int j0 = soff(b), j1 = soff(b) + scnt(b);
          int n = 0;
          for (int sa = -ra; sa <= ra; ++sa)
            for (int sb = -rb; sb <= rb; ++sb)
              for (int sc = -rc; sc <= rc; ++sc) {
                const double shx = sa*scell(b,0) + sb*scell(b,3) + sc*scell(b,6) - xi;
                const double shy = sa*scell(b,1) + sb*scell(b,4) + sc*scell(b,7) - yi;
                const double shz = sa*scell(b,2) + sb*scell(b,5) + sc*scell(b,8) - zi;
                for (int j = j0; j < j1; ++j) {
                  const double vx = posw(j,0)+shx, vy = posw(j,1)+shy, vz = posw(j,2)+shz;
                  const double d2 = vx*vx + vy*vy + vz*vz;
                  if (d2 < 1e-24 || d2 > cutoff2) continue;
                  ++n;
                }
              }
          ecnt(gi) = n;
        });
    IView1D eoff = ws.i1("pet_eoff", Ntot + 1);
    Kokkos::parallel_scan(
        "pet_neigh_scan", RangePolicy(0, Ntot),
        KOKKOS_LAMBDA(int gi, int& upd, bool final) {
          if (final) eoff(gi) = upd;
          upd += ecnt(gi);
          if (final && gi == Ntot - 1) eoff(Ntot) = upd;
        });
    Kokkos::deep_copy(E, Kokkos::subview(eoff, Ntot));
    re_i = ws.i1("pet_re_i", E);
    re_j = ws.i1("pet_re_j", E);
    re_shift = ws.i2("pet_re_shift", E, 3);
    re_vec = ws.r2("pet_re_vec", E, 3);
    re_dist = ws.r1("pet_re_dist", E);
    // Fills [eoff(gi), eoff(gi+1)) in a fixed per-atom order, which is what lets
    // device_neighbors.hpp recover the per-atom ranges with a count plus a scan
    // -- and, more importantly, what makes slot assignment deterministic.
    Kokkos::parallel_for(
        "pet_neigh_fill", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
          const int b = sid(gi);
          const double xi = posw(gi, 0), yi = posw(gi, 1), zi = posw(gi, 2);
          const int ra = srng(b, 0), rb = srng(b, 1), rc = srng(b, 2);
          const int j0 = soff(b), j1 = soff(b) + scnt(b);
          int e = eoff(gi);
          for (int sa = -ra; sa <= ra; ++sa)
            for (int sb = -rb; sb <= rb; ++sb)
              for (int sc = -rc; sc <= rc; ++sc) {
                const double shx = sa*scell(b,0) + sb*scell(b,3) + sc*scell(b,6) - xi;
                const double shy = sa*scell(b,1) + sb*scell(b,4) + sc*scell(b,7) - yi;
                const double shz = sa*scell(b,2) + sb*scell(b,5) + sc*scell(b,8) - zi;
                for (int j = j0; j < j1; ++j) {
                  const double vx = posw(j,0)+shx, vy = posw(j,1)+shy, vz = posw(j,2)+shz;
                  const double d2 = vx*vx + vy*vy + vz*vz;
                  if (d2 < 1e-24 || d2 > cutoff2) continue;
                  re_i(e)=gi; re_j(e)=j;
                  re_shift(e,0)=sa; re_shift(e,1)=sb; re_shift(e,2)=sc;
                  re_vec(e,0)=vx; re_vec(e,1)=vy; re_vec(e,2)=vz;
                  re_dist(e)=Kokkos::sqrt(d2);
                  ++e;
                }
              }
        });
  } else {
    // ---------- Verlet-cached path ----------
    const double skin = 1.0;  // Angstrom
    const double rcs2 = (cutoff + skin) * (cutoff + skin);
    bool rebuild = !cache->valid || cache->built_Ntot != Ntot;
    if (!rebuild) {  // rebuild if any atom moved > skin/2 since the last build
      // Plain displacement, on UNWRAPPED positions -- `pos` is never folded in
      // place, so an atom crossing a cell face moves continuously here and no
      // longer looks like a lattice-vector jump. (A minimum-image reduction
      // would be wrong now: it would hide a genuine motion of about one cell.)
      double d2max = 0;
      auto pos0 = cache->pos0;
      Kokkos::parallel_reduce("nef_disp", RangePolicy(0, Ntot),
          KOKKOS_LAMBDA(int i, double& m) {
            const double dx=pos(i,0)-pos0(i,0), dy=pos(i,1)-pos0(i,1), dz=pos(i,2)-pos0(i,2);
            m = Kokkos::max(m, dx*dx+dy*dy+dz*dz);
          }, Kokkos::Max<double>(d2max));
      rebuild = d2max > 0.25*skin*skin;
      // The CELL moves too, and a displacement test cannot see it: every atom
      // can sit still in fractional coordinates while a contracting cell pulls
      // periodic images inside the cutoff. Relaxing under strain does exactly
      // that, so without this the cache happily served a stale topology --
      // structures lost neighbours, stopped converging, and one came out
      // 17 kJ/mol below the true minimum.
      //
      // Bound the induced change in any pair separation: a strain of relative
      // size eps moves a pair separated by up to (cutoff+skin) by at most
      // eps*(cutoff+skin). Require that to stay inside the same skin/2 budget
      // the displacement test uses.
      if (!rebuild) {
        const double reach = cutoff + skin;
        auto cell0 = cache->cell0;
        double epsmax = 0;
        Kokkos::parallel_reduce("nef_cellchg", RangePolicy(0, B),
            KOKKOS_LAMBDA(int b, double& m) {
              for (int v = 0; v < 3; ++v) {
                const int o = v * 3;
                const double ax=scell(b,o), ay=scell(b,o+1), az=scell(b,o+2);
                const double bx=cell0(b,o), by=cell0(b,o+1), bz=cell0(b,o+2);
                const double dx=ax-bx, dy=ay-by, dz=az-bz;
                const double len = Kokkos::sqrt(bx*bx+by*by+bz*bz);
                if (len > 0) m = Kokkos::max(m, Kokkos::sqrt(dx*dx+dy*dy+dz*dz)/len);
              }
            }, Kokkos::Max<double>(epsmax));
        rebuild = epsmax * reach > 0.5 * skin;
      }
    }
    if (rebuild) {  // search the topology within cutoff+skin (the expensive part)
      // Image ranges for cutoff + skin, not for cutoff.
      //
      // srng is sized for the bare model cutoff, because that is what the
      // uncached search needs. Reusing it here caches edges out to cutoff+skin
      // in DISTANCE but never enumerates the outermost image shell that region
      // reaches, so an edge living there is absent from the cached topology --
      // and stays absent, silently, for every round the cache is reused after
      // the geometry brings it inside the cutoff.
      //
      // The lost edges sit within ~0.1 A of the cutoff, where the bump factor
      // has already fallen to a few percent, so the energy error is below the
      // fp32 force noise. The reason to fix it anyway is that the cached path
      // documents itself as giving bit-identical input to the uncached one, and
      // that invariant is what the validation test relies on.
      //
      // Computed per structure inside the kernel rather than widening srng: the
      // uncached path filters at cutoff2 and would only enumerate a larger box
      // for nothing, which for a cell near 2*cutoff means 125 shifts instead of 27.
      const double rcskin = cutoff + skin;
      IView1D ecnt = ws.i1("pet_ecnt", Ntot);
      Kokkos::parallel_for(
          "pet_neigh_count_s", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
            const int b = sid(gi);
            const double xi = posw(gi, 0), yi = posw(gi, 1), zi = posw(gi, 2);
            int rng_[3];
            {
              const double cb[9] = {scell(b,0),scell(b,1),scell(b,2),
                                    scell(b,3),scell(b,4),scell(b,5),
                                    scell(b,6),scell(b,7),scell(b,8)};
              image_ranges_rows(cb, rcskin, rng_);
            }
            const int ra = rng_[0], rb = rng_[1], rc = rng_[2];
            const int j0 = soff(b), j1 = soff(b) + scnt(b);
            int n = 0;
            for (int sa = -ra; sa <= ra; ++sa)
              for (int sb = -rb; sb <= rb; ++sb)
                for (int sc = -rc; sc <= rc; ++sc) {
                  const double shx = sa*scell(b,0)+sb*scell(b,3)+sc*scell(b,6)-xi;
                  const double shy = sa*scell(b,1)+sb*scell(b,4)+sc*scell(b,7)-yi;
                  const double shz = sa*scell(b,2)+sb*scell(b,5)+sc*scell(b,8)-zi;
                  for (int j = j0; j < j1; ++j) {
                    const double vx=posw(j,0)+shx, vy=posw(j,1)+shy, vz=posw(j,2)+shz;
                    const double d2 = vx*vx+vy*vy+vz*vz;
                    if (d2 >= 1e-24 && d2 <= rcs2) ++n;
                  }
                }
            ecnt(gi) = n;
          });
      IView1D eoff = ws.i1("pet_eoff", Ntot + 1);
      Kokkos::parallel_scan("pet_neigh_scan_s", RangePolicy(0, Ntot),
          KOKKOS_LAMBDA(int gi, int& upd, bool final) {
            if (final) eoff(gi) = upd;
            upd += ecnt(gi);
            if (final && gi == Ntot - 1) eoff(Ntot) = upd;
          });
      int Es = 0;
      Kokkos::deep_copy(Es, Kokkos::subview(eoff, Ntot));
      cache->ci = IView1D("nef_ci", std::max(Es, 1));
      cache->cj = IView1D("nef_cj", std::max(Es, 1));
      cache->cshift = IView2D("nef_cshift", std::max(Es, 1), 3);
      auto ci = cache->ci; auto cj = cache->cj; auto cshift = cache->cshift;
      Kokkos::parallel_for(
          "pet_neigh_fill_s", RangePolicy(0, Ntot), KOKKOS_LAMBDA(int gi) {
            const int b = sid(gi);
            const double xi = posw(gi, 0), yi = posw(gi, 1), zi = posw(gi, 2);
            int rng_[3];
            {
              const double cb[9] = {scell(b,0),scell(b,1),scell(b,2),
                                    scell(b,3),scell(b,4),scell(b,5),
                                    scell(b,6),scell(b,7),scell(b,8)};
              image_ranges_rows(cb, rcskin, rng_);
            }
            const int ra = rng_[0], rb = rng_[1], rc = rng_[2];
            const int j0 = soff(b), j1 = soff(b) + scnt(b);
            int e = eoff(gi);
            for (int sa = -ra; sa <= ra; ++sa)
              for (int sb = -rb; sb <= rb; ++sb)
                for (int sc = -rc; sc <= rc; ++sc) {
                  const double shx = sa*scell(b,0)+sb*scell(b,3)+sc*scell(b,6)-xi;
                  const double shy = sa*scell(b,1)+sb*scell(b,4)+sc*scell(b,7)-yi;
                  const double shz = sa*scell(b,2)+sb*scell(b,5)+sc*scell(b,8)-zi;
                  for (int j = j0; j < j1; ++j) {
                    const double vx=posw(j,0)+shx, vy=posw(j,1)+shy, vz=posw(j,2)+shz;
                    const double d2 = vx*vx+vy*vy+vz*vz;
                    if (d2 < 1e-24 || d2 > rcs2) continue;
                    // Store the image in UNWRAPPED terms: the search ran on
                    // wrapped coordinates, so subtract the wrap each end was
                    // folded by. The triple then stays valid as atoms cross cell
                    // faces, which is the whole point of the cache.
                    ci(e)=gi; cj(e)=j;
                    cshift(e,0) = sa - wrapi(j,0) + wrapi(gi,0);
                    cshift(e,1) = sb - wrapi(j,1) + wrapi(gi,1);
                    cshift(e,2) = sc - wrapi(j,2) + wrapi(gi,2);
                    ++e;
                  }
                }
          });
      cache->Es = Es;
      cache->valid = true;
      cache->built_Ntot = Ntot;
      if ((int) cache->pos0.extent(0) < Ntot) cache->pos0 = RView2D("nef_pos0", Ntot, 3);
      Kokkos::deep_copy(Kokkos::subview(cache->pos0, Kokkos::make_pair(0, Ntot), Kokkos::ALL),
                        Kokkos::subview(pos, Kokkos::make_pair(0, Ntot), Kokkos::ALL));
      if ((int) cache->cell0.extent(0) < B) cache->cell0 = RView2D("nef_cell0", B, 9);
      Kokkos::deep_copy(Kokkos::subview(cache->cell0, Kokkos::make_pair(0, B), Kokkos::ALL),
                        Kokkos::subview(scell, Kokkos::make_pair(0, B), Kokkos::ALL));
    }
    // Filter the cached topology -> exact within-cutoff compacted COO, with the
    // edge vectors recomputed from the CURRENT positions.
    const int Es = cache->Es;
    auto ci = cache->ci; auto cj = cache->cj; auto cshift = cache->cshift;
    IView1D foff = ws.i1("pet_foff", Es + 1);
    Kokkos::parallel_scan("nef_filter_scan", RangePolicy(0, Es),
        KOKKOS_LAMBDA(int e, int& upd, bool final) {
          if (final) foff(e) = upd;
          const int gi=ci(e), j=cj(e), b=sid(gi);
          const double ix=cshift(e,0)*scell(b,0)+cshift(e,1)*scell(b,3)+cshift(e,2)*scell(b,6);
          const double iy=cshift(e,0)*scell(b,1)+cshift(e,1)*scell(b,4)+cshift(e,2)*scell(b,7);
          const double iz=cshift(e,0)*scell(b,2)+cshift(e,1)*scell(b,5)+cshift(e,2)*scell(b,8);
          const double vx=pos(j,0)+ix-pos(gi,0), vy=pos(j,1)+iy-pos(gi,1), vz=pos(j,2)+iz-pos(gi,2);
          const double d2 = vx*vx+vy*vy+vz*vz;
          upd += (d2 >= 1e-24 && d2 <= cutoff2) ? 1 : 0;
          if (final && e == Es-1) foff(Es) = upd;
        });
    Kokkos::deep_copy(E, Kokkos::subview(foff, Es));
    re_i = ws.i1("pet_re_i", std::max(E, 1));
    re_j = ws.i1("pet_re_j", std::max(E, 1));
    re_shift = ws.i2("pet_re_shift", std::max(E, 1), 3);
    re_vec = ws.r2("pet_re_vec", std::max(E, 1), 3);
    re_dist = ws.r1("pet_re_dist", std::max(E, 1));
    Kokkos::parallel_for("nef_filter_fill", RangePolicy(0, Es), KOKKOS_LAMBDA(int e) {
      const int gi=ci(e), j=cj(e), b=sid(gi);
      const double ix=cshift(e,0)*scell(b,0)+cshift(e,1)*scell(b,3)+cshift(e,2)*scell(b,6);
      const double iy=cshift(e,0)*scell(b,1)+cshift(e,1)*scell(b,4)+cshift(e,2)*scell(b,7);
      const double iz=cshift(e,0)*scell(b,2)+cshift(e,1)*scell(b,5)+cshift(e,2)*scell(b,8);
      const double vx=pos(j,0)+ix-pos(gi,0), vy=pos(j,1)+iy-pos(gi,1), vz=pos(j,2)+iz-pos(gi,2);
      const double d2 = vx*vx+vy*vy+vz*vz;
      if (d2 < 1e-24 || d2 > cutoff2) return;
      const int o = foff(e);
      re_i(o)=gi; re_j(o)=j;
      re_shift(o,0)=cshift(e,0); re_shift(o,1)=cshift(e,1); re_shift(o,2)=cshift(e,2);
      re_vec(o,0)=vx; re_vec(o,1)=vy; re_vec(o,2)=vz;
      re_dist(o)=Kokkos::sqrt(d2);
    });
  }

  DeviceEdgeData dev = build_device_edge_data(
      ws, edge_map, m_high, probes, P, Ntot, spec, re_i, re_j, re_shift, re_vec, re_dist, E, h);
  dev.struct_id = sid;
  dev.n_struct = B;
  return dev;
}

// Host entry point: marshal a batch of host Systems into the device staging
// Views, then build the NEF. One host->device copy per staging array.
DeviceEdgeData build_device_batch(const std::vector<System>& systems, const Hypers& h,
                                  const std::vector<int>& species_to_index, const RView1D& probes,
                                  int P, Workspace& ws, EdgeMap& edge_map, int& m_high,
                                  NefCache* cache) {
  const int B = static_cast<int>(systems.size());
  int Ntot = 0;
  std::vector<int> off(B);
  for (int b = 0; b < B; ++b) {
    off[b] = Ntot;
    Ntot += systems[b].n_atoms;
  }

  DeviceGeom g = stage_geometry_views(ws, Ntot, B);
  auto h_pos = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.pos);
  auto h_sid = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.sid);
  auto h_spec = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.spec);
  auto h_soff = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.soff);
  auto h_scnt = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.scnt);
  auto h_srng = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.srng);
  auto h_sper = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.sper);
  auto h_scell = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, g.scell);
  for (int b = 0; b < B; ++b) {
    const System& s = systems[b];
    h_soff(b) = off[b];
    h_scnt(b) = s.n_atoms;
    auto rng = detail::image_ranges(s, h.cutoff);
    for (int d = 0; d < 3; ++d) h_srng(b, d) = rng[d];
    // System has no explicit periodicity flag here; image_ranges returns
    // {0,0,0} for a degenerate or absent cell, which is exactly the
    // non-periodic case.
    h_sper(b) = (rng[0] || rng[1] || rng[2]) ? 1 : 0;
    for (int e = 0; e < 9; ++e) h_scell(b, e) = s.cell[e];
    for (int i = 0; i < s.n_atoms; ++i) {
      const int gi = off[b] + i;
      for (int d = 0; d < 3; ++d) h_pos(gi, d) = s.positions[3 * i + d];
      h_sid(gi) = b;
      const int Z = s.atomic_numbers[i];
      const int sp = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
      if (sp < 0) throw std::runtime_error("pet: unsupported atomic number in batch");
      h_spec(gi) = sp;
    }
  }
  Kokkos::deep_copy(g.pos, h_pos);
  Kokkos::deep_copy(g.sid, h_sid);
  Kokkos::deep_copy(g.spec, h_spec);
  Kokkos::deep_copy(g.soff, h_soff);
  Kokkos::deep_copy(g.scnt, h_scnt);
  Kokkos::deep_copy(g.srng, h_srng);
  Kokkos::deep_copy(g.sper, h_sper);
  Kokkos::deep_copy(g.scell, h_scell);

  DeviceEdgeData dev = build_nef_device(g, h, probes, P, ws, edge_map, m_high, cache);

  // Per-structure electronic state, for a conditioned model. Uploaded here
  // rather than derived on the device: it comes from the caller's Systems and
  // has nothing to do with geometry.
  {
    IView1D q("charge", B), sm("spin", B);
    auto h_q = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, q);
    auto h_s = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, sm);
    for (int b = 0; b < B; ++b) {
      h_q(b) = systems[b].charge;
      h_s(b) = systems[b].spin_multiplicity;
    }
    Kokkos::deep_copy(q, h_q);
    Kokkos::deep_copy(sm, h_s);
    dev.charge = q;
    dev.spin_multiplicity = sm;
  }
  return dev;
}

}  // namespace pet
