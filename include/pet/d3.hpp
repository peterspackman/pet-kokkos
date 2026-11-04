// Grimme's DFT-D3 dispersion correction, on the device: on its own, or added to
// a PET model's energy, forces and virial (Options::d3).
//
//   E = -sum_{i<j} sum_{n=6,8} s_n C_n^ij / r^n f_n(r)  [ - sum_{i<j<k} E_ATM ]
//
// with C6 interpolated between reference systems by each atom's coordination
// number (CN), C8 = 3 C6 sqrt(Q_i Q_j), and f_n the damping: rational
// (Becke-Johnson, the default) or the original zero damping. The optional
// Axilrod-Teller-Muto three-body term (s9 != 0) is zero-damped, as in s-dftd3.
// Reference data and functional parameters: src/d3_data.cpp, from tad-dftd3
// (see NOTICE). With D3Cutoffs::reference() it reproduces s-dftd3 to ~1e-12.
//
// The sums are truncated with a smooth taper (D3Cutoffs), so energy and forces
// stay continuous (C3) as atoms cross a cutoff. No pair list is stored: each
// atom walks the cells of a binned grid out to each cutoff, three times -- the
// CN, then the pair energy, forces and dE/dCN, then the CN's chain rule. Every
// atom sums its own full neighbourhood in a fixed order, with no atomics, so a
// result is the same to the bit run to run and whichever batch a structure is
// in. Everything is in double.
//
// Units are pet-kokkos's: Angstrom, eV, eV/A; the virial is W = sum v (x) dE/dv,
// as Results::virial. The tables are converted from bohr/hartree once
// (CODATA 2018), when a D3 is made.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pet/device_geometry.hpp"
#include "pet/exchange.hpp"
#include "pet/kokkos.hpp"
#include "pet/model.hpp"
#include "pet/neighbors.hpp"

namespace pet {

struct Results;

enum class D3Damping { BJ, Zero };

// A parametrisation, in pet-kokkos units (a2 in Angstrom).
struct D3Params {
  D3Damping damping = D3Damping::BJ;
  double s6 = 1.0, s8 = 0.0;
  double a1 = 0.0, a2 = 0.0;    // rational (BJ): R0 = a1 sqrt(C8/C6) + a2
  double rs6 = 1.0, rs8 = 1.0;  // zero damping: f_n = 1 / (1 + 6 (r / (rs_n R0))^-alpha_n)
  double alpha = 14.0;          // zero damping's alpha_6 (alpha_8 = alpha + 2); the ATM's is alpha + 2
  double s9 = 0.0;              // the ATM three-body term's weight; 0 = off
};

// A functional's published parameters (s-dftd3's names and aliases, any case:
// "pbe", "pbesol", "b3lyp", "r2scan", "pbe0"/"pbeh", ...), with s9 = 1 when
// `atm`, else 0. Throws for a name without parameters for that damping.
D3Params d3_parameters(const std::string& functional, D3Damping damping = D3Damping::BJ, bool atm = false);
// The functionals with parameters for a damping.
std::vector<std::string> d3_functionals(D3Damping damping = D3Damping::BJ);
D3Damping d3_damping(const std::string& name);  // "bj" | "rational" | "zero"

// Where each sum stops, in Angstrom: a term is multiplied by a smooth step from 1
// at cutoff - taper to 0 at cutoff (taper 0: a hard cutoff, as s-dftd3's).
//
// The pair sum's tail falls off only as 1/r^3 in the energy: past 12 A it is
// still ~0.5 meV/atom in water and ~15-30 meV/atom in a transition metal,
// with a matching error in the pressure. With `tail`, a 3D-periodic structure
// gets it back as a uniform continuum past the taper (g(r) = 1, as an MD
// code's tail correction): its energy, its pressure, and its forces through
// the CNs. Not for molecules, slabs, or an engine's ghost atoms (see README
// "D3 dispersion" for the measured errors either way).
struct D3Cutoffs {
  double pair = 12.0, pair_taper = 2.0;
  double cn = 8.0, cn_taper = 2.0;
  double atm = 8.0, atm_taper = 2.0;
  bool tail = true;
  // s-dftd3's and Grimme's dftd3's: pairs to 95 bohr, CN and ATM to 40, hard,
  // no tail.
  static D3Cutoffs reference();
};

struct D3Settings {
  D3Params params;
  D3Cutoffs cutoffs;
  // Each pair's and triangle's arithmetic in float, its sums still in double:
  // several times faster on a GPU with slow double precision (consumer cards),
  // and within ~1e-7 relative of the double results. Off by default.
  bool mixed_precision = false;
  // A functional's parameters with the default cutoffs.
  static D3Settings functional(const std::string& name, D3Damping damping = D3Damping::BJ, bool atm = false);
};

// The correction itself. Reuse one across calls: it keeps its tables on the
// device and its scratch pool.
class D3 {
 public:
  explicit D3(D3Settings settings);
  ~D3();
  D3(D3&&) noexcept;
  D3& operator=(D3&&) noexcept;
  D3(const D3&) = delete;
  D3& operator=(const D3&) = delete;

  const D3Settings& settings() const;

  // Host structures: energy, per-atom energy, forces and virial (zeros for a
  // structure periodic along no axis), as a Calculator's.
  Results compute(const System& system, bool compute_forces = true) const;
  Results compute(const std::vector<System>& systems, bool compute_forces = true) const;

  // Structures staged on the device, their species the atomic numbers; results
  // stay there, in this D3's pool, valid until its next evaluation.
  DeviceGeom stage(const std::vector<System>& systems) const;
  BatchResult compute_device(const DeviceGeom& geom, bool compute_forces = true) const;

  // How far past its owned atoms an engine must supply ghosts: with an Exchange,
  // the pair (or ATM) cutoff -- ghosts' CNs come from their owners and their
  // dE/dCN goes home -- and without one, that plus the CN cutoff.
  double ghost_cutoff(bool exchange) const;

  // ---- the device interface the Calculator (and an engine) drives ----------

  // N atoms of B structures, as DeviceGeom has them. Atoms [0, n_local) are
  // owned (B = 1 only; -1: all): only their energy counts, and every atom gets
  // the force of that energy, ghosts included (the engine sums those home).
  struct Atoms {
    RView2D pos;                     // [N, 3], unwrapped
    IView1D type;                    // [N] each atom's type: its atomic number, or with ztab an index
    IView1D ztab;                    // [types] each type's atomic number; empty: types are atomic numbers
    IView1D sid, soff, scnt, sper;   // [N], [B], [B], [B]: as DeviceGeom
    RView2D scell;                   // [B, 9]
    int N = 0, B = 0;
    int n_local = -1;
    Exchange* exchange = nullptr;    // ghosts' CN from their owners (B = 1)
  };
  struct Out {
    Kokkos::View<double*, LR, MemSpace> energy;       // [B]
    Kokkos::View<double*, LR, MemSpace> per_atom;     // [N], 0 for ghosts
    Kokkos::View<double**, LR, MemSpace> forces;      // [N, 3]; with forces
    Kokkos::View<double**, LR, MemSpace> virial;      // [B, 6] Voigt; with forces
    Kokkos::View<double**, LR, MemSpace> atom_virial; // [N, 6] each atom's share; with forces
  };
  // keep_grid: an engine stepping one list keeps the grid's shape (its bins,
  // a molecule's box with room to move) from one call to the next, so a step
  // records as a graph; overflowed() says when an atom left the box, and the
  // step must be redone after reset_grid().
  Out evaluate(const Atoms& atoms, bool compute_forces, bool keep_grid = false) const;
  bool overflowed() const;
  IView1D overflow_flag() const;  // [1] on the device, nonzero once an atom left the box
  void reset_grid() const;
  // Bumped whenever a buffer moves or the grid is sized afresh: for a recorded
  // graph's key.
  std::size_t generation() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pet
