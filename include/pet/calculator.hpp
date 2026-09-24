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
  // Processes sharing the device (MPI ranks on one GPU): the budget is split.
  int device_share = 1;

  // How much of the forward the backward recomputes instead of keeping (see
  // pet::Recompute). Auto keeps everything that fits memory_budget_bytes.
  Recompute recompute = Recompute::Auto;

  // Replay a repeated evaluation (same shapes and buffers, as in a stepping
  // loop) as one CUDA graph: bit-identical, much faster for small structures.
  // CUDA only; PET_GRAPHS=0 turns it off.
  bool graphs = true;

  // A cap on atoms per batch ahead of the memory estimate; 0 = none.
  int max_batch_atoms = 0;

  // compute_step: size every step between neighbour-list rebuilds to one fixed
  // capacity, so each replays one CUDA graph. It pads every atom to the largest
  // neighbour count plus a margin, which costs more than replay saves wherever
  // measured (64-atom crystal: equal; 648-atom water: 32.6 vs 27.5 ms/step).
  bool md_fixed_shapes = false;
};

// Energy, forces and virial for B structures of Ntot atoms in all.
struct Results {
  std::vector<double> energy;           // [B] eV
  std::vector<double> per_atom_energy;  // [Ntot] eV
  std::vector<double> forces;           // [Ntot*3] eV/A, F = -dE/dx
  std::vector<double> virial;           // [B*6] eV, the symmetric W, Voigt order
  std::vector<int> struct_id;           // [Ntot] each atom's structure
  std::vector<int> n_atoms;             // [B]
  // compute_edges only: dE/dv for each of the engine's edges v = r_j - r_i, in
  // its order (0 past the cutoff). Forces, the virial, per-atom stress, heat
  // flux... are all folds of it: F_i = sum over i's edges of g - sum over edges
  // into i of g, W = sum over edges of v (x) g.
  std::vector<double> edge_gradient;    // [n_edges*3] eV/A
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
  // Structures staged on the device, from this Calculator's pool, for a caller
  // that steps them there (a relaxation, MD): move geom.pos and geom.scell in
  // place, then compute_device. Staging resets the neighbour cache.
  DeviceGeom stage(const std::vector<System>& systems) const;
  // From staged geometry, results left on the device. With a smaller geom.B and
  // geom.Ntot, the leading structures only (a batch whose finished structures
  // were moved to the back).
  BatchResult compute_device(const DeviceGeom& geom, bool compute_forces = true) const;

  // Atoms and edges from an MD engine's own neighbour list (see EdgeListView):
  // the energy of the owned atoms, forces on every atom, ghosts included (the
  // engine sums those back to their owners), the virial and the edge gradients.
  // Ghosts must reach ghost_cutoff() past the owned atoms, and every atom within
  // ghost_cutoff() - cutoff() must have its full neighbour list.
  // Edge gradients cost a copy of every edge back to the host: off when unused.
  // It replaces any list set_neighbors holds.
  Results compute_edges(const EdgeListView& edges, bool compute_forces = true, bool edge_gradients = true) const;

  // The same for an MD engine stepping one list: set_neighbors when the engine
  // rebuilds it, then compute_step every step with the atoms' positions in the
  // same order (and the cell, if the list has shifts). The list crosses to the
  // device once per rebuild; a step moves positions and results. Pairs that
  // drift past the cutoff weigh nothing; pairs past the list's own reach are the
  // engine's to rebuild for, as with any Verlet list. See md_fixed_shapes.
  void set_neighbors(const EdgeListView& list);
  Results compute_step(const double* positions, const double* cell = nullptr, bool compute_forces = true,
                       bool edge_gradients = false) const;

  // compute_step for an engine whose arrays live on the device (LAMMPS's KOKKOS
  // package): device pointers, [n, 3] row-major doubles, forces added into. Only
  // the totals cross to the host. Optional per-atom outputs, added into too:
  // energy [n], and the virial [n, 6] (xx yy zz xy xz yz) -- each pair's
  // symmetrised v (x) dE/dv, half to each end, times virial_scale (LAMMPS's sign
  // is -1). They sum to the returned virial (times virial_scale).
  struct DeviceArrays {
    const double* positions = nullptr;
    double* forces = nullptr;
    double* per_atom_energy = nullptr;
    double* per_atom_virial = nullptr;
    double virial_scale = 1.0;
  };
  struct Totals {
    double energy = 0.0;
    double virial[6] = {0, 0, 0, 0, 0, 0};  // as Results::virial
  };
  Totals compute_step(const DeviceArrays& arrays, const double* cell = nullptr) const;
  // How far past its owned atoms an engine must supply ghosts: one cutoff per
  // message-passing layer, and one more for an adaptive cutoff, which needs each
  // of those atoms' complete neighbourhoods.
  double ghost_cutoff() const;

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
  Results session_results(const DeviceEdgeData& dev, bool compute_forces, bool edge_gradients) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pet
