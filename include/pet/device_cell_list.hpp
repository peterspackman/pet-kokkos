// Device cell list for the raw periodic neighbour search.
//
// The search build_nef_device does is O(N^2 x images): every atom against every
// atom of its structure, across every periodic image within the cutoff. For the
// small cells a relaxation batch is made of that is the right algorithm -- at
// eight atoms a grid is pure overhead -- and for anything large it is the
// dominant cost of the whole evaluation. On a 1728-atom diamond supercell at a
// 7.5 A cutoff it was taking longer than the network it feeds.
//
// This is the O(N) replacement: bin atoms into a grid whose cells are at least
// one cutoff wide, then look only in the neighbouring bins.
//
// THE GRID. Per structure, nc[d] = floor(perp_width[d] / cutoff), at least 1,
// where perp_width is the interplanar spacing perpendicular to lattice vector d.
// A bin is then at least `cutoff` across, so the neighbours of an atom lie in
// the 27 bins around its own -- except when the box itself is narrower than the
// cutoff, where nc collapses to 1 and the search has to reach further. The range
// nr[d] = ceil(cutoff / bin_width[d]) covers both cases with one formula, and
// degenerates to exactly the old image enumeration when the box is tiny.
//
// WRAPPING. A bin index outside [0, nc) folds back with a lattice shift, and
// that shift is the edge's periodic image -- floor-division and the true modulus
// give both at once. This is why nc = 1 and nc = 2 need no special case: the
// same bin reached with different shifts is a different image of it, which is
// exactly right.
//
// DETERMINISM. Bin membership is filled with an atomic counter, so the order
// within a bin is thread-arrival order -- and the NEF packer assigns neighbour
// slots in list order, which fixes the attention softmax's summation order. Left
// alone that would put run-to-run noise straight into the energy, the one thing
// this evaluator is careful never to do. Each bin is therefore sorted by atom
// index after the fill: bins hold a handful of atoms, the sort is one small
// kernel, and the result is reproducible by construction rather than by luck.
//
// The edge ORDER differs from the brute-force search (bins and shifts, rather
// than images and atoms), so results differ in the last bits -- the same way the
// host and device builders already do. The edge SET is identical, and that is a
// test (tests/test_cell_list.cpp), not an assumption.
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

// Atoms per structure below which Auto keeps the brute-force search. Building
// and sorting a grid costs several kernel launches; for the eight-atom cells a
// CSP batch is made of, that is more than the search it replaces.
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
