// The high-level entry point: load a PET model once, evaluate structures with it
// many times.
//
// Units are the model's: Angstrom, eV, eV/A. The virial is the symmetric
// W = V * stress, in Voigt order [xx, yy, zz, xy, xz, yz]; a strain-DOF optimizer
// usually wants the off-diagonals doubled (dE/d(eps)), which is the consumer's
// to do.
//
// Kokkos must be initialized before a Calculator is made and stay so while it
// lives. The library never initializes or finalizes Kokkos: that belongs to the
// application.
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

// Anything that changes the answer beyond rounding is off by default.
struct Options {
  // Build neighbour lists on the device. Host and device lists hold the same
  // edges in the same order, so they agree to a few fp32 ulps (~3e-8 relative);
  // tests/test_device_vs_host.cpp holds them to it.
  bool device_neighbors = true;

  // Keep a Verlet cache of the neighbour topology across calls, for a caller
  // stepping the same atoms (MD, a relaxation). See NefCache.
  bool cache_neighbors = false;

  // TF32 tensor-core GEMMs: faster (~15% on an RTX 4080, more on A100/H100),
  // and ~0.1 meV/atom and a few meV/A off -- so never for a reference
  // comparison or a determinism check. Process-global (see gemm.hpp).
  bool allow_tf32 = false;

  // The host neighbour search: vesin's cell list, or the built-in O(N^2) one.
  // Without vesin in the build, the built-in one either way. They agree on the
  // edges, not on their order, so results differ in the last bits.
  enum class Neighbors { Builtin, Vesin };
  Neighbors neighbors = Neighbors::Vesin;

  // Device memory an evaluation may use, in bytes; 0 asks the device.
  std::size_t memory_budget_bytes = 0;

  // How much of the forward the backward recomputes instead of keeping (see
  // pet::Recompute). Auto keeps everything that fits memory_budget_bytes.
  Recompute recompute = Recompute::Auto;

  // Replay a repeated evaluation (same shapes and buffers, as in a stepping
  // loop) as one CUDA graph: bit-identical, much faster for small structures.
  // CUDA only; PET_GRAPHS=0 turns it off.
  bool graphs = true;

  // A cap on atoms per batch ahead of the memory estimate; 0 = none.
  int max_batch_atoms = 0;
};

// Energy, forces and virial for B structures of Ntot atoms in all.
struct Results {
  std::vector<double> energy;           // [B] eV
  std::vector<double> per_atom_energy;  // [Ntot] eV
  std::vector<double> forces;           // [Ntot*3] eV/A, F = -dE/dx
  std::vector<double> virial;           // [B*6] eV, the symmetric W, Voigt order
  std::vector<int> struct_id;           // [Ntot] each atom's structure
  std::vector<int> n_atoms;             // [B]
};

// Whether the build can use vesin for the host neighbour search.
bool vesin_available();

// Where a named model is looked for, in order: $PET_MODEL_DIR (':'-separated),
// ./models and ., $XDG_DATA_HOME/pet/models (else ~/.local/share/pet/models),
// and this source tree's models/. A directory counts only if it holds both
// <name>.json and <name>.safetensors.
std::vector<std::string> model_search_dirs();

// A model name ("pet-mad-xs") or a path prefix to <spec>.json and
// <spec>.safetensors, to the two paths; empty is the default model. Throws,
// listing where it looked, if a named model is not found.
void resolve_model(const std::string& spec, std::string& json_out, std::string& weights_out);

// A single structure goes to the device neighbour path when its raw edge count
// (every pair within the cutoff) is estimated above this. The host path's cost
// is its single-threaded work over that list: a 216-atom diamond cell at 8 A
// (~80k raw edges) is 3x faster on the device, a 64-atom one (~24k) slightly
// faster on the host.
constexpr double kDeviceSingleMinRawEdges = 5e4;

// A loaded model. Reuse one across calls: it holds the scratch pools, caches and
// recorded graphs that make a stepping loop cheap.
class Calculator {
 public:
  // Resolve `spec` as above; empty = the default model.
  explicit Calculator(const std::string& spec = "", Options opts = {});
  Calculator(const std::string& model_json, const std::string& model_weights, Options opts = {});
  ~Calculator();
  Calculator(Calculator&&) noexcept;
  Calculator& operator=(Calculator&&) noexcept;
  Calculator(const Calculator&) = delete;
  Calculator& operator=(const Calculator&) = delete;

  // One structure, on the host or device neighbour path by size
  // (kDeviceSingleMinRawEdges).
  Results compute(const System& system, bool compute_forces = true) const;
  // Several structures as one evaluation. Edges never cross structures, so each
  // gets the answer it would alone (to the bit on the host path). One structure
  // goes to the overload above.
  Results compute(const std::vector<System>& systems, bool compute_forces = true) const;
  // From geometry already on the device (see stage), results left there.
  BatchResult compute_device(const DeviceGeom& geom, bool compute_forces = true) const;
  // Device staging for n_atoms_total atoms in n_struct structures, from this
  // Calculator's pool: fill it, then compute_device.
  DeviceGeom stage(int n_atoms_total, int n_struct) const;

  const Hypers& hypers() const;
  double cutoff() const;
  const std::vector<int>& atomic_types() const;  // supported atomic numbers, ascending
  bool supports(int atomic_number) const;
  const std::string& length_unit() const;
  const std::string& energy_unit() const;
  // Applied internally; exposed as a fingerprint of which checkpoint is loaded.
  double energy_scale() const;

  // Atoms per batch the device can afford: the smaller of what memory allows
  // (the backward's saves scale with atoms x neighbours) and where throughput
  // stops improving. The neighbour count is known only once something has been
  // evaluated, so ask per chunk.
  int recommended_batch_atoms() const;

  // Device scratch held, and what holds it.
  std::size_t workspace_bytes() const;
  std::vector<std::pair<std::string, std::size_t>> workspace_breakdown() const;

  Options& options();
  const Options& options() const;

 private:
  Results compute_batch(const std::vector<System>& systems, bool compute_forces) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pet
