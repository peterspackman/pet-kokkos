// The high-level entry point: load a PET model once, evaluate structures
// against it many times.
//
// Everything here speaks the model's own units -- positions and cells in
// Angstrom, energies in eV, forces in eV/Angstrom -- and returns the virial as
// the SYMMETRIC tensor W = V*sigma in Voigt order [xx, yy, zz, xy, xz, yz].
// See the note on `Results::virial` before consuming it; a strain-DOF optimizer
// usually wants the off-diagonals doubled, and that conversion belongs at the
// consumer's seam, not here.
//
// Kokkos must be initialized before a Calculator is constructed and must stay
// initialized for its lifetime. The library never calls Kokkos::initialize or
// Kokkos::finalize: an application owns that, and a library that guesses gets it
// wrong exactly once, in someone else's main().
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "pet/config.hpp"
#include "pet/device_geometry.hpp"
#include "pet/kokkos.hpp"
#include "pet/model.hpp"
#include "pet/neighbors.hpp"

namespace pet {

// Knobs that select a code path or describe the machine. Anything that changes
// the ANSWER is off by default and has to be asked for.
struct Options {
  // Build the neighbour list on the device (the default, and the fast path) or
  // on the host. The two are meant to agree to the bit; a difference between
  // them is a bug, not a setting, which is why this exists at all.
  bool device_neighbors = true;

  // Reuse the Verlet topology cache across calls. Worth it only for a caller
  // stepping the same atoms (a relaxer, an MD driver); a one-shot evaluation
  // pays the rebuild either way. Also meant to be bit-identical to the uncached
  // path -- see NefCache in device_geometry.hpp.
  bool cache_neighbors = false;

  // Allow TF32 tensor-core GEMMs on NVIDIA. Faster, and it CHANGES THE ANSWER,
  // so it stays off unless asked for and must be off for anything compared
  // against a reference.
  bool allow_tf32 = false;

  // Device memory a batch may occupy, in bytes. 0 queries the device.
  std::size_t memory_budget_bytes = 0;

  // Hard cap on atoms per batch, ahead of the memory estimate. 0 = derive it.
  int max_batch_atoms = 0;
};

// Energy, forces and virial for one or more structures, in model units.
struct Results {
  std::vector<double> energy;           // [B] total energy, eV
  std::vector<double> per_atom_energy;  // [Ntot] eV (single-structure path only)
  std::vector<double> forces;           // [Ntot*3] eV/Angstrom, F = -dE/dx
  // [B*6] Voigt [xx, yy, zz, xy, xz, yz], eV. The SYMMETRIC virial
  // W = V*sigma -- the physical stress times the volume. A strain-DOF optimizer
  // typically wants dE/d(eps), whose off-diagonals are W_xy + W_yx = 2*W_xy;
  // apply that factor at your own seam.
  std::vector<double> virial;
  std::vector<int> struct_id;  // [Ntot] owning structure per atom
  std::vector<int> n_atoms;    // [B] atom count per structure (force-array offsets)
};

// Directories searched for a NAMED model, highest priority first:
//   1. $PET_MODEL_DIR (':'-separated) -- the knob for batch jobs
//   2. ./models and .                 -- a self-contained run directory
//   3. $XDG_DATA_HOME/pet/models, else ~/.local/share/pet/models
//   4. the build-time source tree's models/ (developer convenience)
// A directory matches only if it holds BOTH halves of the pair.
std::vector<std::string> model_search_dirs();

// Resolve a model spec to a metadata-JSON + weights path pair. `spec` may be a
// canonical model name ("pet-mad-xs"), or a filesystem path prefix
// (<spec>.json + <spec>.safetensors). Empty resolves the default model.
// Throws, listing every directory tried, when a named model is not found.
void resolve_model(const std::string& spec, std::string& json_out, std::string& weights_out);

// Loads a PET model once (the safetensors parse and the device upload are the
// expensive part) and evaluates structures against it. Reuse one Calculator
// across many calls: it carries the persistent scratch pools that make a
// stepping loop free of per-round device allocation.
class Calculator {
 public:
  // Resolve `spec` through the search path above; empty = the default model.
  explicit Calculator(const std::string& spec = "", Options opts = {});
  // Explicit file pair, no searching.
  Calculator(const std::string& model_json, const std::string& model_weights, Options opts = {});
  ~Calculator();

  Calculator(Calculator&&) noexcept;
  Calculator& operator=(Calculator&&) noexcept;
  Calculator(const Calculator&) = delete;
  Calculator& operator=(const Calculator&) = delete;

  // Evaluate one structure. Takes the host neighbour-list path, which is what
  // fills per_atom_energy.
  Results compute(const System& system, bool compute_forces = true) const;

  // Evaluate many (typically small) structures in ONE pass: each structure's
  // NEF is built and the lot concatenated, so B small cells use a GPU as well
  // as one big cell does. PET's network is per-atom/per-edge, so the per-atom
  // energies and forces equal evaluating each structure alone.
  Results compute(const std::vector<System>& systems, bool compute_forces = true) const;

  // Fully device-resident evaluation: `geom` is already-populated staging (see
  // device_geometry.hpp) and the result stays in device Views. This is the path
  // for a caller that owns its geometry on the device and never wants a host
  // round-trip. Honours Options::cache_neighbors.
  BatchResult compute_device(const DeviceGeom& geom, bool compute_forces = true) const;

  // Staging Views sized for Ntot atoms in B structures, drawn from this
  // Calculator's persistent pool. Fill them, then call compute_device.
  DeviceGeom stage(int n_atoms_total, int n_struct) const;

  // --- model properties ---
  const Hypers& hypers() const;
  double cutoff() const;
  // Atomic numbers the model supports, ascending.
  const std::vector<int>& atomic_types() const;
  bool supports(int atomic_number) const;
  const std::string& length_unit() const;
  const std::string& energy_unit() const;

  // Atoms per batch this model can afford right now. Two ceilings bind and the
  // smaller wins: device MEMORY (the backward's saved activations are linear in
  // EDGE SLOTS, atoms x neighbours, not in atoms) and THROUGHPUT (cost per round
  // stops being flat well before memory runs out). Both need the neighbour count
  // M, which is a property of the structures rather than the model, so the first
  // call returns a measured per-featurizer floor and every call after it adapts.
  // Ask per chunk, not once.
  int recommended_batch_atoms() const;

  Options& options();
  const Options& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pet
