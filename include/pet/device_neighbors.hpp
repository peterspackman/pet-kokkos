// The device neighbour builder: from a raw edge list on the device to the
// DeviceEdgeData the model reads -- adaptive cutoff, per-atom slots, reverse
// map -- without the host. This is the path an MD or relaxation driver takes
// (the raw edges come from build_nef_device, device_geometry.hpp).
//
// Every per-atom result is an ordered walk over that atom's own edges, never an
// atomic in thread-arrival order: the network is fp32, and a neighbour order
// that changed run to run changed every result in its last bits.
#pragma once

#include <Kokkos_UnorderedMap.hpp>

#include <cstdint>
#include <vector>

#include "pet/cutoff.hpp"
#include "pet/model.hpp"

namespace pet {

namespace detail {

// (centre, neighbour, shift) as one 64-bit key: 24 + 24 bits of atom index and 5
// bits per shift component, biased by 15. Enough for 16M atoms and shifts in
// [-15, 15].
constexpr int PET_KEY_SHIFT_BIAS = 15;
constexpr long long PET_KEY_ATOM_MAX = 1LL << 24;  // exclusive
KOKKOS_INLINE_FUNCTION uint64_t pet_pack_key(int i, int j, int sa, int sb, int sc) {
  return (uint64_t(i) << 39) | (uint64_t(j) << 15) | (uint64_t(sa + PET_KEY_SHIFT_BIAS) << 10) |
         (uint64_t(sb + PET_KEY_SHIFT_BIAS) << 5) | uint64_t(sc + PET_KEY_SHIFT_BIAS);
}

}  // namespace detail

using EdgeMap = Kokkos::UnorderedMap<uint64_t, int, ExecSpace>;

// From the raw edges -- every directed edge within the cutoff, grouped by centre
// in a fixed order (re_i non-decreasing): re_i/re_j local atoms, re_shift the
// neighbour's lattice shift, re_vec r_j + shift - r_i, re_dist its length. They
// are kept as the DeviceEdgeData's raw_* for the adaptive cutoff's backward.
// species is [N] species indices; probes/P the adaptive probe grid.
//
// Scratch comes from `ws` and `edge_map`, so a warm call allocates nothing.
// m_high > 0 makes M grow-only across calls (seed it with 1 for MD, where a
// fluctuating M would change the shapes every step); 0 leaves M the batch's own.
DeviceEdgeData build_device_edge_data(Workspace& ws, EdgeMap& edge_map, int& m_high, RView1D probes,
                                      int P, int N, IView1D species, IView1D re_i, IView1D re_j,
                                      IView2D re_shift, RView2D re_vec, RView1D re_dist, int E,
                                      const Hypers& h);

// The same in two parts, for a caller that keeps a topology across geometries
// (an MD engine's list between its rebuilds). edge_topology: each atom's range
// of the raw list, each edge's partner, the partnerless edges; the raw vectors
// and distances are only held. edge_geometry, from dev.raw_vec/raw_dist: the
// adaptive cutoffs, the kept edges and their slots. m_fixed > 0 fixes M, and
// with it every shape (DeviceEdgeData::padded): an atom with more kept edges
// sets overflow(0), and its extra edges are dropped.
void edge_topology(Workspace& ws, EdgeMap& edge_map, DeviceEdgeData& dev, int N, IView1D species, IView1D re_i,
                   IView1D re_j, IView2D re_shift, RView2D re_vec, RView1D re_dist, int E);
void edge_geometry(Workspace& ws, DeviceEdgeData& dev, const Hypers& h, RView1D probes, int P, int& m_high,
                   int m_fixed, IView1D overflow);  // dev.exchange: ghosts' cutoffs from their owners

// The same for one structure, from a host neighbour search: for tests.
DeviceEdgeData build_device_edge_data(const System& sys, const Hypers& h,
                                      const std::vector<int>& species_to_index);

}  // namespace pet
