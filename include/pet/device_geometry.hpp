// Device-resident geometry staging and the raw periodic neighbour search that
// feeds device_neighbors.hpp.
//
// device_neighbors.hpp turns a raw COO edge list into a NEF; this is what
// PRODUCES that edge list, on the device, from positions and cells that never
// leave it. Together they are the zero-host-round-trip path:
//
//     DeviceGeom (pos, cells, species)      <- written by the caller, on device
//        |  build_nef_device
//        v
//     raw COO edge list (re_i, re_j, re_shift, re_vec, re_dist)
//        |  build_device_edge_data          (device_neighbors.hpp)
//        v
//     DeviceEdgeData                        -> PetModel::compute
//
// A caller that owns its geometry on the device (a relaxer, an MD driver) writes
// DeviceGeom::pos / scell in place between steps and calls build_nef_device
// again; nothing crosses the bus but an edge count. build_device_batch is the
// host convenience entry point that marshals a vector<System> into the same
// staging Views.
//
// Everything here is drawn from the persistent Workspace pool, so a stepping
// loop pays no per-round device (re)allocation.
#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "pet/config.hpp"
#include "pet/device_neighbors.hpp"
#include "pet/kokkos.hpp"
#include "pet/neighbors.hpp"

namespace pet {

// Periodic image ranges for a cell whose ROWS are lattice vectors; the device
// counterpart of detail::image_ranges (neighbors.hpp), which takes a System on
// the host. n[k] = ceil(cutoff / interplanar spacing perpendicular to lattice
// vector k).
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

// Device geometry staging Views for one batch of B structures totalling Ntot
// atoms. Positions are Angstrom; scell rows = lattice vectors (Angstrom).
// sid/spec/soff/scnt are constant across a relaxation; pos/scell/srng change as
// the geometry moves. Splitting the staging out of the NEF build is what lets a
// device-resident optimizer write pos/scell directly on-device and call
// build_nef_device() with no host round-trip.
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

// All staging Views drawn from the capacity-reusing pool, so a stepping loop
// pays no per-round device (re)allocation.
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

// Verlet cache for the raw periodic neighbour search. The TOPOLOGY (which atom
// pairs + periodic images are neighbours) is geometry-stable across relaxation
// or MD steps; only the edge vectors/distances change. We cache the topology
// within cutoff+skin and, each evaluation, recompute vectors and filter to
// exactly the within-cutoff edges. Rebuilt only when the layout changed (Ntot)
// or an atom moved past skin/2, or the cell strained enough to move a pair that
// far. Pass nullptr for no caching.
//
// How closely this matches the uncached path, precisely:
//
//   * the edge SET and its per-atom ORDER are identical, always;
//   * on the round that BUILDS the cache, the result agrees to double round-off
//     (measured ~4e-16 relative) -- the same edges from the same coordinates;
//   * on a round that REUSES it, the result agrees to a few fp32 ulps (measured
//     ~9e-9 relative on the energy). The uncached search states each image in
//     WRAPPED terms and forms its vector from wrapped coordinates; the cache
//     stores images in UNWRAPPED terms -- which is what lets a stored triple
//     survive an atom crossing a cell face, and the only reason the cache pays
//     for itself -- and forms its vector from unwrapped ones. Same vector in
//     exact arithmetic, different rounding in floating point.
//
// tests/test_device_vs_host.cpp checks all three, and the second is the one that
// matters most: a dropped or spurious edge cannot hide under a 1e-16 bound. That
// is how this path was caught dropping edges within 0.1 A of the cutoff.
struct NefCache {
  IView1D ci, cj;  // [Es] cached topology: centre + neighbour (global indices)
  IView2D cshift;  // [Es,3] periodic image (sa,sb,sc), in UNWRAPPED terms
  RView2D pos0;    // [Ntot,3] positions at last rebuild (skin check)
  RView2D cell0;   // [B,9] cell at last rebuild (strain check)
  int Es = 0;      // cached edge count (within cutoff+skin)
  int built_Ntot = -1;
  bool valid = false;
};

// Build the combined device NEF from already-populated geometry staging Views.
// This is the device-resident entry point: positions/cells are read straight
// from device memory, so a caller can feed updated geometry without a host copy.
//
// `probes`/`P` are the adaptive-cutoff probe grid (PetModel::probes()).
// `edge_map` and `m_high` are the reusable state device_neighbors.hpp wants:
// the reverse-match map (grown and cleared, not reallocated) and the grow-only
// neighbour-count high-water mark (0 disables it; see device_neighbors.hpp).
//
// `cache` (optional) is the Verlet topology cache above; nullptr rebuilds the
// full within-cutoff search every call.
DeviceEdgeData build_nef_device(const DeviceGeom& g, const Hypers& h, const RView1D& probes,
                                int P, Workspace& ws, EdgeMap& edge_map, int& m_high,
                                NefCache* cache = nullptr);

// Host entry point: marshal a batch of host Systems into the device staging
// Views, then build the NEF. One host->device copy per staging array.
//
// `cache` is forwarded to build_nef_device. Passing one is only useful when
// successive calls describe the SAME atoms in the same order -- a stepping
// caller -- since the topology is keyed on atom count and displacement.
DeviceEdgeData build_device_batch(const std::vector<System>& systems, const Hypers& h,
                                  const std::vector<int>& species_to_index,
                                  const RView1D& probes, int P, Workspace& ws,
                                  EdgeMap& edge_map, int& m_high, NefCache* cache = nullptr);

}  // namespace pet
