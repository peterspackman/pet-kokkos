// Optional vesin backend for the host periodic neighbour search.
//
// `detail::build_raw_edges` is an O(N^2 x images) brute-force search. That is
// fine for the small cells the goldens use and badly wrong for anything larger:
// measured on a 1728-atom diamond supercell at a 7.5 A cutoff it accounts for
// ~0.45 s of a 0.47 s evaluation -- 95% of the runtime, with the entire network
// costing 0.02 s. vesin's cell list does the same search in 14 ms.
//
// vesin (https://github.com/Luthaf/vesin) is the neighbour-list library from the
// same ecosystem as metatrain and metatomic, so a PET model's neighbour list is
// being built by the library its reference implementation uses.
//
// Enabled with -DPET_WITH_VESIN=ON; without it nothing here is compiled and the
// built-in search is used. Even when compiled in it is selectable at run time
// (Options::neighbors, PET_NEIGHBORS=vesin|builtin), because the two are meant
// to agree and a switch is how that gets checked.
//
// ORDERING. vesin documents the order of `j` within each `i` block as
// unspecified, and it is multithreaded, so that order is not something to build
// on: the NEF packer assigns neighbour slots in list order, and the slot order
// sets the attention softmax's summation order. This wrapper therefore sorts
// each atom's block into a canonical (j, shift) order before returning. That
// makes the vesin path reproducible run to run -- the property the whole
// determinism discipline rests on -- at the cost of not being bit-identical to
// the built-in path, whose order is the image-loop order instead. The two agree
// to fp32 noise, exactly as the host and device builders already do.
#pragma once

#ifdef PET_HAVE_VESIN

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "pet/neighbors.hpp"

namespace pet {
namespace detail {

// Same contract as build_raw_edges: the full directed list of edges within
// `cutoff`, self-pairs excluded, grouped by centre atom.
std::vector<RawEdge> build_raw_edges_vesin(const System& sys, double cutoff);

}  // namespace detail
}  // namespace pet

#endif  // PET_HAVE_VESIN
