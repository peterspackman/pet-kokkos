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
#include <utility>
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
  // on the host. Both produce the same edge set in the same per-atom order; the
  // arithmetic is issued differently, so results differ by a few fp32 ulps
  // (measured ~3e-8 relative on the energy). Anything larger than that is a
  // bug, not a setting -- which is why this switch exists at all, and what
  // tests/test_device_vs_host.cpp holds it to.
  bool device_neighbors = true;

  // Reuse the Verlet topology cache across calls. Worth it only for a caller
  // stepping the same atoms (a relaxer, an MD driver); a one-shot evaluation
  // pays the rebuild either way. On the round that builds the cache the result
  // matches the uncached path to double round-off; once reused after the
  // geometry moves it matches to a few fp32 ulps, because the cache states
  // periodic images in unwrapped coordinates and the uncached search in wrapped
  // ones. See NefCache in device_geometry.hpp for why that trade is the right
  // one.
  bool cache_neighbors = false;

  // Allow TF32 tensor-core GEMMs on NVIDIA. Faster, and it CHANGES THE ANSWER,
  // so it stays off unless asked for and must be off for anything compared
  // against a reference or checked for determinism.
  //
  // PROCESS-GLOBAL despite living here: cuBLAS fixes a handle's math mode when
  // the handle is created, and there is one handle per process. A Calculator
  // applies this at construction (early enough), but a second Calculator asking
  // for something different will not change it. pet::set_tf32 in gemm.hpp is
  // the direct control, and PET_TF32=1 sets the initial value.
  bool allow_tf32 = false;

  // Which host neighbour search to use. The built-in one is an O(N^2 x images)
  // brute force; vesin's is an O(N) cell list, and on a 1728-atom supercell the
  // difference is ~450 ms against ~14 ms -- 95% of that evaluation's runtime.
  //
  // Only meaningful when the library was built with vesin (PET_WITH_VESIN);
  // without it this falls back to the built-in search rather than failing, and
  // `vesin_available()` says which you will get.
  //
  // The two do NOT produce the same edge ORDER -- the built-in search walks
  // images in a fixed nested loop, vesin's list is sorted canonically by
  // (i, j, shift) -- so results differ in the last bits, the same way the host
  // and device builders already do. Both are individually reproducible.
  enum class Neighbors { Builtin, Vesin };
  Neighbors neighbors = Neighbors::Vesin;

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

// True when this build can use vesin for the host neighbour search. When false,
// Options::neighbors is ignored and the built-in search is always used.
bool vesin_available();

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

  // Evaluate one structure. Always takes the host neighbour-list path -- it is
  // what the goldens validate, it returns the virial directly, and it is the
  // only path that fills per_atom_energy.
  Results compute(const System& system, bool compute_forces = true) const;

  // Evaluate many (typically small) structures in ONE pass: each structure's
  // NEF is built and the lot concatenated, so B small cells use a GPU as well
  // as one big cell does. PET's network is per-atom/per-edge and edges never
  // cross a structure boundary, so the per-atom energies and forces are
  // identical to evaluating each structure alone -- bit-identical, in fact, on
  // the host neighbour path (tests/test_determinism.cpp).
  //
  // A single-element vector delegates to the overload above, which means it
  // takes the HOST neighbour path whatever Options::device_neighbors says, and
  // fills per_atom_energy. That is deliberate -- one structure is not worth a
  // device NEF build -- but it does mean the two entry points differ in the last
  // bits for B == 1 (see the host/device comparison in
  // tests/test_device_vs_host.cpp for the size of that: ~3e-8 relative).
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
  // The output scaler fitted with this model. Not needed to evaluate anything --
  // it is applied internally -- but it is the sharpest single fingerprint of
  // WHICH checkpoint is loaded, which a golden uses to refuse a model it was not
  // generated from.
  double energy_scale() const;

  // Atoms per batch this model can afford right now. Two ceilings bind and the
  // smaller wins: device MEMORY (the backward's saved activations are linear in
  // EDGE SLOTS, atoms x neighbours, not in atoms) and THROUGHPUT (cost per round
  // stops being flat well before memory runs out). Both need the neighbour count
  // M, which is a property of the structures rather than the model, so the first
  // call returns a measured per-featurizer floor and every call after it adapts.
  // Ask per chunk, not once.
  int recommended_batch_atoms() const;

  // Device scratch the evaluator is holding, and where it went. The pool is
  // grow-only and keyed by label, so the breakdown attributes every byte to the
  // buffer that asked for it -- which is the only practical way to find out why
  // a large model in fp64 does not fit. Both are zero until something has been
  // evaluated.
  std::size_t workspace_bytes() const;
  std::vector<std::pair<std::string, std::size_t>> workspace_breakdown() const;

  Options& options();
  const Options& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pet
