// The host neighbour search through vesin's cell list (PET_WITH_VESIN), in place
// of the built-in O(N^2 x images) one.
#pragma once

#ifdef PET_HAVE_VESIN

#include <vector>

#include "pet/neighbors.hpp"

namespace pet::detail {

// As build_raw_edges, with each atom's edges sorted by (j, shift): vesin leaves
// that order unspecified and is threaded, and the slot order sets the
// attention's summation order.
std::vector<RawEdge> build_raw_edges_vesin(const System& sys, double cutoff);

}  // namespace pet::detail

#endif
