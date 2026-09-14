// Reading and writing structures in extended XYZ.
//
// Enough of the format to drive an evaluation, not a general parser: the atom
// count, a comment line carrying `Lattice="..."` and `Properties=...`, then one
// line per atom. That covers what ASE writes and what every MLIP benchmark
// ships, and it keeps the library free of a structure-file dependency.
#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "pet/neighbors.hpp"

namespace pet {

// Read every frame in an extended-XYZ file. A frame with a Lattice= entry is
// periodic in whichever directions its pbc= entry says (default: all three when
// a lattice is present, none when it is absent).
std::vector<System> read_extxyz(const std::string& path);
std::vector<System> read_extxyz(std::istream& in);

// Write one frame, optionally with an energy and per-atom forces, in the layout
// ASE reads back: Properties=species:S:1:pos:R:3[:forces:R:3].
void write_extxyz(std::ostream& out, const System& system, const double* forces = nullptr,
                  const double* energy = nullptr);

// Chemical symbol for an atomic number, and the reverse. Unknown symbols give 0.
const char* element_symbol(int atomic_number);
int element_number(const std::string& symbol);

}  // namespace pet
