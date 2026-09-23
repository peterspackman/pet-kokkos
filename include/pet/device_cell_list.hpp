// A device cell list for the raw periodic search: O(N) where the brute-force
// search is O(N^2 x images), and the one worth its setup above a few hundred
// atoms.
//
// Per structure, nc[d] = floor(width[d] / cutoff) bins along lattice vector d
// (width: the interplanar spacing), at least one; each atom's neighbours are in
// the bins within nr[d] = ceil(cutoff / bin width) of its own, which covers a box
// narrower than the cutoff too. A bin index outside [0, nc) folds back with a
// lattice shift, and that shift is the edge's image -- floor division gives both.
//
// Bins are filled with an atomic counter and then put in atom order, so the edge
// order, and so the result, is reproducible. It differs from the brute-force
// search's order (the results differ in the last bits); the edge set is the same,
// which tests/test_cell_list.cpp checks.
#pragma once

#include <Kokkos_Core.hpp>

#include "pet/device_geometry.hpp"
#include "pet/kokkos.hpp"

namespace pet {

// Which raw search build_nef_device should use.
enum class DeviceSearch {
  Auto,       // cell list above a size threshold, brute force below it
  CellList,   // always the grid
  BruteForce  // always the O(N^2) search -- the reference the grid is checked against
};

// Process-wide default, from PET_DEVICE_SEARCH=auto|cells|brute.
DeviceSearch& device_search();

// Atoms in a batch below which Auto keeps the brute-force search: building a
// grid costs more than it saves there.
constexpr int kCellListMinAtoms = 256;

// The raw COO edge list, as build_nef_device's own search produces it: full and
// directed, grouped by centre atom, self-pairs excluded.
struct RawEdges {
  IView1D i, j;     // [E] centre and neighbour, global atom indices
  IView2D shift;    // [E,3] periodic image applied to j
  RView2D vec;      // [E,3] r_j + shift - r_i
  RView1D dist;     // [E]
  IView1D offsets;  // [Ntot+1] start of each atom's contiguous run
  int count = 0;    // E
};

// Build the raw edge list with a cell list. `posw` is the wrapped positions and
// `cinv` the per-structure inverse cell that build_nef_device already computes.
RawEdges build_raw_edges_cells(Workspace& ws, const DeviceGeom& g, const RView2D& posw,
                               const RView2D& cinv, double cutoff);

}  // namespace pet
