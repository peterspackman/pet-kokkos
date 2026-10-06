// What crosses between ranks when an engine runs PET domain-decomposed (LAMMPS
// over MPI) with ghosts only one cutoff deep, rather than a shell as deep as the
// message passing.
//
// Each rank evaluates its owned atoms; ghosts have no neighbour lists. PET's
// atoms interact across a layer in one place: an edge i -> j reads its partner
// j -> i's output row. For an edge to a ghost g, the partner g -> i lives on the
// rank that owns g, so at every layer the ranks swap those rows (and, in the
// backward, their adjoints). The adaptive cutoff also needs each ghost's cutoff
// from its owner, and sends each ghost's cutoff adjoint home.
//
// The engine implements the three moves. Views are on the device.
#pragma once

#include "pet/kokkos.hpp"

namespace pet {

class Exchange {
 public:
  virtual ~Exchange() = default;

  // `a` [n_atoms] holds the owned atoms' values: fill each ghost's from its owner.
  virtual void atoms_forward(RView1D a) = 0;

  // Add each ghost's entry of `a` into its owner's, and zero the ghost's.
  virtual void atoms_reverse(RView1D a) = 0;

  // Once per evaluation, before any edges(): which edges to ghosts PET keeps
  // this time (the rest are past the cutoff, on both ends of the pair). Each is
  // an index into the edges to ghosts in the order of the engine's list (atom
  // i's pairs in turn, those whose neighbour is a ghost), ascending; with fixed
  // shapes, entries past the kept ones are -1 (rows to leave alone).
  virtual void set_live(IView1D live) = 0;

  // One row per kept edge to a ghost, in set_live's order. Send each row of
  // `out` to the rank owning that ghost, where it belongs to the partner edge;
  // fill `in` with the partner edges' rows in the same order. Rows are
  // `out.extent(1)` wide: one layer's edge outputs, or their adjoints.
  virtual void edges(View2D out, View2D in) = 0;
};

}  // namespace pet
