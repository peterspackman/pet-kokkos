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
#include "pet/exchange.hpp"
#include "pet/kokkos.hpp"
#include "pet/neighbors.hpp"

namespace pet {

// Images to search along each lattice vector (cell rows) for `cutoff`: the
// device twin of detail::image_ranges; `periodic` is a bitmask of the periodic
// axes.
KOKKOS_INLINE_FUNCTION
void image_ranges_rows(const double cell[9], double cutoff, int periodic, int n[3]) {
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
  if (periodic & 1) n[0] = (int) Kokkos::ceil(cutoff / (vol / nbc));
  if (periodic & 2) n[1] = (int) Kokkos::ceil(cutoff / (vol / nca));
  if (periodic & 4) n[2] = (int) Kokkos::ceil(cutoff / (vol / nab));
}

// One batch of B structures, Ntot atoms in all, staged on the device. Between
// evaluations a driver moves pos and scell (a relaxation, MD). Lowering B and
// Ntot evaluates the leading structures only; a driver that also packs the
// structures still running to the front rewrites every field, and names in
// `staged` which staged structure each slot now holds, so a Verlet cache can
// follow them.
struct DeviceGeom {
  RView2D pos;     // [Ntot,3] cartesian, Angstrom, unwrapped (see build_nef_device)
  IView1D sid;     // [Ntot]   owning structure
  IView1D spec;    // [Ntot]   species index (Calculator::stage maps atomic numbers)
  IView1D soff;    // [B]      atom offset per structure
  IView1D scnt;    // [B]      atom count per structure
  IView1D sper;    // [B]      periodic axes, bit d for lattice vector d; 0 = a molecule
  RView2D scell;   // [B,9]    cell rows = lattice vectors, Angstrom
  IView1D charge;  // [B]      total charge (conditioned models)
  IView1D spin;    // [B]      spin multiplicity 2S+1 (conditioned models)
  IView1D staged;  // [B]      the staged structure in each slot (stage: b)
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
  g.sper = ws.i1("pet_sper", B);
  g.scell = ws.r2("pet_scell", B, 9);
  g.charge = ws.i1("pet_charge", B);
  g.spin = ws.i1("pet_spin", B);
  g.staged = ws.i1("pet_staged", B);
  return g;
}

// Atoms and pairs an MD engine supplies from its own neighbour list. Atoms
// [0, n_local) are owned and make up the energy; the rest are ghosts (periodic
// images, other ranks' atoms) that only shape it. Atom i's pairs are
// [offsets[i], offsets[i+1]), i -> neighbors[e], in the engine's order; pairs
// past the model cutoff are carried but weigh nothing (an engine lists cutoff +
// skin). A full list has both directions wherever both atoms have
// neighbourhoods; a `half` list has each pair once, from either end.
//
// v_e = r_j - r_i from `positions` (ghosts sit where they are), plus
// shift . cell for an engine that tells periodic images apart by `shifts`.
struct EdgeListView {
  int n_atoms = 0, n_local = -1;       // n_local -1: every atom is owned
  const double* positions = nullptr;   // [n_atoms, 3] Angstrom
  const int* atomic_numbers = nullptr; // [n_atoms]
  const int* offsets = nullptr;        // [n_atoms + 1]
  const int* neighbors = nullptr;      // [offsets[n_atoms]]
  const int* shifts = nullptr;         // [offsets[n_atoms], 3], optional
  const double* cell = nullptr;        // [9] lattice vectors as rows, with shifts
  bool half = false;
  int charge = 0, spin_multiplicity = 1;
  // Over several ranks: only the owned atoms have pairs, ghosts are one cutoff
  // deep, and the engine carries what crosses (exchange.hpp). A full list.
  Exchange* exchange = nullptr;
};

// An engine's list on the device between its rebuilds: the topology once (set),
// the geometry per step. A half list is mirrored here, each atom's own pairs
// first. With `fixed`, M is a capacity chosen at the first step, so every step
// until the next set has the same shapes; it grows when an atom outgrows it.
struct EdgeSession {
  Workspace ws;
  EdgeMap map{16};
  DeviceEdgeData dev;
  IView2D shift;                  // [E, 3] each directed edge's image shift
  std::vector<int> src;           // [E] each directed edge's pair in the engine's list
  std::vector<signed char> dir;   // [E] +1 as listed, -1 its mirror
  int n_pairs = 0, M = 0;
  bool shifted = false, valid = false;

  void set(const EdgeListView& v, const std::vector<int>& species_to_index);
  // positions: host, or device when on_device.
  const DeviceEdgeData& step(const double* positions, bool on_device, const double* cell, const Hypers& h,
                             RView1D probes, int P, bool fixed);
};

// Stage host structures into views from `ws`: species through
// species_to_index (throws on an unsupported element), everything else as given.
DeviceGeom stage_systems(Workspace& ws, const std::vector<System>& systems,
                         const std::vector<int>& species_to_index);

// A Verlet cache for a batch being stepped: the pairs within cutoff + skin and
// their topology (partners, each atom's range), searched for only when some atom
// has moved, or some cell strained, far enough that a pair outside could now be
// inside the cutoff. Between searches a call recomputes the cached pairs'
// vectors, and those past the cutoff weigh nothing, as in an engine's list.
//
// It follows a shrinking batch: when DeviceGeom::staged shows structures gone or
// packed into other slots, the survivors' pairs are reassembled from the last
// search, not searched again. Images are stored in unwrapped terms, so a pair
// survives an atom crossing a cell face. The kept edges and their order match the
// uncached search; the vectors differ in the last bits (unwrapped rather than
// wrapped arithmetic). tests/test_device_vs_host.cpp checks both.
struct NefCache {
  static constexpr double kSkin = 1.0;  // Angstrom
  Workspace ws;                         // everything below lives here
  EdgeMap map{16};
  // The batch as last searched, in its own numbering: the topology, each
  // pair's image, positions and inverse cells then, and each slot's staged
  // structure and ranges. slot_of: staged structure -> slot, -1 if absent.
  DeviceEdgeData ref;
  IView2D ref_shift;
  RView2D pos0, cinv0;
  IView1D ref_soff, ref_scnt, ref_eoff, slot_of;
  // The batch the last call evaluated: `ref` itself, or ref reassembled for a
  // smaller batch.
  DeviceEdgeData cur;
  IView2D cur_shift;
  IView1D cur_staged;
  int cur_B = -1, cur_N = -1;
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
