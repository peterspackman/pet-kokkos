// The device-resident path's front end: geometry staged on the device, and the
// raw periodic neighbour search that turns it into the edge list
// device_neighbors.hpp packs. A caller that owns its geometry on the device (a
// relaxer, an MD driver) writes DeviceGeom::pos and scell in place between steps
// and calls build_nef_device again; nothing crosses the bus but edge counts.
//
//   DeviceGeom --build_nef_device--> raw edges --build_device_edge_data--> DeviceEdgeData
//
// Everything comes from the Workspace pool, so a stepping loop does not allocate.
#pragma once

#include <Kokkos_Core.hpp>

#include <vector>

#include "pet/config.hpp"
#include "pet/device_neighbors.hpp"
#include "pet/kokkos.hpp"
#include "pet/neighbors.hpp"

namespace pet {

// Images to search along each lattice vector (cell rows) for `cutoff`: the
// device twin of detail::image_ranges.
KOKKOS_INLINE_FUNCTION
void image_ranges_rows(const double cell[9], double cutoff, int n[3]) {
  n[0] = n[1] = n[2] = 0;
  const double a[3] = {cell[0], cell[1], cell[2]};
  const double b[3] = {cell[3], cell[4], cell[5]};
  const double c[3] = {cell[6], cell[7], cell[8]};
  const double bc[3] = {b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};
  const double ca[3] = {c[1]*a[2]-c[2]*a[1], c[2]*a[0]-c[0]*a[2], c[0]*a[1]-c[1]*a[0]};
  const double ab[3] = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
  const double vol = Kokkos::fabs(a[0]*bc[0] + a[1]*bc[1] + a[2]*bc[2]);
  if (vol < 1e-12) return;
  const double nbc = Kokkos::sqrt(bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]);
  const double nca = Kokkos::sqrt(ca[0]*ca[0]+ca[1]*ca[1]+ca[2]*ca[2]);
  const double nab = Kokkos::sqrt(ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]);
  n[0] = (int) Kokkos::ceil(cutoff / (vol / nbc));
  n[1] = (int) Kokkos::ceil(cutoff / (vol / nca));
  n[2] = (int) Kokkos::ceil(cutoff / (vol / nab));
}

// One batch of B structures, N atoms in all, staged on the device. sid, spec,
// soff and scnt are fixed through a relaxation; pos and scell move.
struct DeviceGeom {
  RView2D pos;    // [Ntot,3] cartesian, Angstrom, UNWRAPPED (see build_nef_device)
  IView1D sid;    // [Ntot]   owning structure
  IView1D spec;   // [Ntot]   species index
  IView1D soff;   // [B]      atom offset per structure
  IView1D scnt;   // [B]      atom count per structure
  IView2D srng;   // [B,3]    periodic image ranges at the model cutoff
  IView1D sper;   // [B]      1 if periodic, 0 for an isolated molecule
  RView2D scell;  // [B,9]    cell rows = lattice vectors, Angstrom
  int Ntot = 0, B = 0;
};

inline DeviceGeom stage_geometry_views(Workspace& ws, int Ntot, int B) {
  DeviceGeom g;
  g.Ntot = Ntot;
  g.B = B;
  g.pos = ws.r2("pet_pos", Ntot, 3);
  g.sid = ws.i1("pet_sid", Ntot);
  g.spec = ws.i1("pet_spec", Ntot);
  g.soff = ws.i1("pet_soff", B);
  g.scnt = ws.i1("pet_scnt", B);
  g.srng = ws.i2("pet_srng", B, 3);
  g.sper = ws.i1("pet_sper", B);
  g.scell = ws.r2("pet_scell", B, 9);
  return g;
}

// A Verlet cache for the search: the pairs within cutoff + skin, found only when
// an atom moves, or the cell strains, far enough to bring a new pair inside the
// cutoff; each call recomputes their vectors and keeps those within the cutoff.
// Images are stored in unwrapped terms, so a pair survives an atom crossing a
// cell face. The edge set and order match the uncached search exactly; on a
// reused round the vectors differ in the last bits (unwrapped rather than
// wrapped arithmetic). tests/test_device_vs_host.cpp checks both.
struct NefCache {
  IView1D ci, cj;  // [Es] cached topology: centre + neighbour (global indices)
  IView2D cshift;  // [Es,3] periodic image (sa,sb,sc), in UNWRAPPED terms
  RView2D pos0;    // [Ntot,3] positions at last rebuild (skin check)
  RView2D cell0;   // [B,9] cell at last rebuild (strain check)
  int Es = 0;      // cached edge count (within cutoff+skin)
  int built_Ntot = -1;
  bool valid = false;
};

// The device neighbour list from staged geometry: a cell list for large batches
// (device_cell_list.hpp), a brute-force search for small ones, or the Verlet cache
// when given one. probes/P are PetModel::probes(); edge_map and m_high are
// build_device_edge_data's.
DeviceEdgeData build_nef_device(const DeviceGeom& g, const Hypers& h, const RView1D& probes,
                                int P, Workspace& ws, EdgeMap& edge_map, int& m_high,
                                NefCache* cache = nullptr);

// The same from host Systems, staged first. A cache only helps when successive
// calls describe the same atoms in the same order.
DeviceEdgeData build_device_batch(const std::vector<System>& systems, const Hypers& h,
                                  const std::vector<int>& species_to_index,
                                  const RView1D& probes, int P, Workspace& ws,
                                  EdgeMap& edge_map, int& m_high, NefCache* cache = nullptr);

}  // namespace pet
