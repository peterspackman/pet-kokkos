// PET model: holds weights as Kokkos Views and evaluates energy (forces later).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "pet/checkpoint.hpp"
#include "pet/config.hpp"
#include "pet/kokkos.hpp"
#include "pet/neighbors.hpp"
#include "pet/ozaki.hpp"

namespace pet {

struct EnergyResult {
  double total = 0.0;
  std::vector<double> per_atom;  // [N] eV, includes scale + composition
  // Symmetric virial W_ab = dE/d(strain_ab) = V * stress, Voigt order
  // [xx, yy, zz, xy, xz, yz], eV. Only filled when forces are requested.
  double virial[6] = {0, 0, 0, 0, 0, 0};
};

// Device-resident result of a batched evaluation of B structures concatenated into
// one NEF. All arrays stay on the device so a Kokkos consumer reads them without
// a host round-trip; structure b owns force rows where struct_id==b.
struct BatchResult {
  RView1D energy;      // [B] total energy per structure (eV), device
  RView2D forces;      // [N_total,3] per-atom forces (eV/A), device; empty if not requested
  RView2D virial;      // [B,6] per-structure virial (Voigt, eV), device; empty if no forces
  IView1D struct_id;   // [N_total] owning structure per atom, device
  int n_struct = 0;
  int n_atoms = 0;
};

// Device-resident NEF edge data: exactly the inputs PetModel::compute consumes,
// already in device memory. Produced either by uploading a host EdgeData
// (upload_edge_data, the host neighbor-list path) or by building it on-device
// from a raw neighbor list (build_device_edge_data, the device path). Keeping the
// host->device marshaling in one place lets compute() be fully device-native and
// lets the on-device builder feed the same kernels without a host round-trip.
struct DeviceEdgeData {
  int n_atoms = 0;        // N
  int max_neighbors = 0;  // M
  int n_raw = 0;          // E (raw edges retained for the adaptive-cutoff backward)

  IView1D species;        // [N]   per-atom species index
  IView1D neigh_species;  // [N*M] neighbor species index (padding -> 0)
  IView1D reverse_index;  // [N*M] flat slot of reverse edge (j->i), -1 if none
  RView2D edge_vec;       // [N*M,3] edge vector r_j(+image) - r_i (double)
  RView1D dist;           // [N*M] edge distance (double)
  RView1D mask;           // [N*M] 1.0 real / 0.0 padding (double)
  RView1D pair_cutoff;    // [N*M] per-edge cutoff radius (double)
  View1D cutoff_factor;   // [N*M] smooth cutoff factor (Net)
  View2D cf_seq;          // [N,S=M+1] attention-bias source: col0=1, cols 1..M = factor

  IView1D raw_center;     // [E] center local index
  IView1D raw_neigh;      // [E] neighbor owner local index
  RView1D raw_dist;       // [E] distance (double)
  RView2D raw_vec;        // [E,3] edge vector (double)
  // [N+1] start of each atom's contiguous run in the raw list, and [E] the raw
  // index of each edge's (j,i,-shift) partner (-1 if absent). Both exist so the
  // adaptive-cutoff backward can GATHER per atom instead of scattering with float
  // atomics, whose accumulation order is not reproducible between runs.
  IView1D raw_off;
  IView1D raw_reverse;

  // [N,P] effective neighbor counts over the probe grid, computed by the on-device
  // neighbor build to set the adaptive cutoff. Referenced (not copied) so the
  // adaptive backward can reuse it instead of recomputing K1 (ad_eff). Empty on
  // the host-upload path (the backward then recomputes). GRID method only.
  RView2D adapt_eff;

  // SOLVER method only: [N] the root r where the smoothed neighbour count reaches
  // the target, and [N] dn/dr there. The backward's implicit-function step needs
  // both, and recomputing them would mean redoing the whole Newton solve. A zero
  // in adapt_dn means that atom's cutoff hit a clamp bound and has no gradient.
  // Empty on the host-upload path, which recomputes.
  RView1D adapt_r;
  RView1D adapt_dn;

  // Batched evaluation: when several structures are concatenated into one NEF,
  // struct_id[atom] is the owning structure and n_struct the count. Default
  // (n_struct=1, empty struct_id) is a single structure.
  IView1D struct_id;
  int n_struct = 1;

  // [n_struct] electronic state, for a model trained with system_conditioning.
  // Empty for every other model. Per structure, not per atom -- the embedding is
  // a system property broadcast to the atoms that belong to it.
  IView1D charge;
  IView1D spin_multiplicity;
};

// A weight matrix, together with its Ozaki decomposition when there is one.
//
// Carried as a pair rather than threading the split through linear() /
// linear_silu() / linear_bwd() separately, because those have some eighty call
// sites, all spelled `linear(out, in, mat("..."), vec("..."))`. The implicit
// conversion both ways means every other use of mat() -- the embedding gathers,
// a plain View2D held in a local -- keeps working unchanged, just without the
// pre-split (the GEMM then splits that operand in place).
//
// `v` is held by value: a Kokkos::View is a reference-counted handle, so copying
// it is cheap, and a reference member would dangle the moment one of these was
// built from a temporary.
struct WeightRef {
  View2D v;
  // Two decompositions, because int8 tensor-core GEMM needs the contraction
  // index contiguous and a weight is contracted along opposite axes by
  // linear() (transposed) and linear_bwd() (not). `split` is the natural
  // layout, `split_t` the transposed one; both are null unless the Ozaki path
  // is on.
  const OzakiSplit* split = nullptr;
  const OzakiSplit* split_t = nullptr;
  WeightRef() = default;
  WeightRef(const View2D& m, const OzakiSplit* s = nullptr, const OzakiSplit* st = nullptr)
      : v(m), split(s), split_t(st) {}
  // The decomposition to use when the GEMM wants B transposed (`tb`) or not.
  const OzakiSplit* for_orientation(bool tb) const { return tb ? split : split_t; }
  operator const View2D&() const { return v; }
  std::size_t extent(int d) const { return v.extent(d); }
};

class PetModel {
 public:
  explicit PetModel(const Checkpoint& ckpt);

  // Evaluate total + per-atom energy for one preprocessed system.
  EnergyResult energy(const EdgeData& ed) {
    // No forces, so no backward, so nothing will read the reverse map.
    return compute(upload_edge_data(ed, /*need_reverse=*/false), nullptr, nullptr);
  }

  // Evaluate energy and conservative forces (F[i] = -dE/dx_i, eV/Angstrom).
  // forces is filled with N*3 values (row-major). Returns the same energy as
  // energy() (identical forward pass).
  EnergyResult energy_forces(const EdgeData& ed, std::vector<double>& forces) {
    return compute(upload_edge_data(ed), &forces, nullptr);
  }

  // Batched evaluation of many (typically small) structures in one GPU pass: builds
  // each structure's NEF, concatenates into one combined NEF, and runs ONE compute()
  // over all atoms, so B small cells use the GPU as efficiently as one big cell. The
  // result is fully device-resident (per-structure energy[B], per-atom forces[N,3],
  // struct_id[N]) -- no host copy. PET's network is per-atom/per-edge, so per-atom
  // energies/forces equal evaluating each structure alone. Intended for re-ranking /
  // re-relaxing CSP candidates. (Per-structure virial: forthcoming.)
  BatchResult energy_forces_batch(const std::vector<System>& systems, bool compute_forces = true);

  // Device-resident batched entry point (the efficient path for a Kokkos caller):
  // `dev` is an already-combined NEF over all structures with
  // dev.struct_id[N] and dev.n_struct=B set -- e.g. built by build_device_edge_data
  // from a concatenated raw COO edge list (global atom indices, edges within a
  // structure). Runs ONE compute() + on-device per-structure energy segmentation;
  // no host round-trip.
  BatchResult energy_forces_batch(const DeviceEdgeData& dev, bool compute_forces = true);

  // Device-resident overloads: the edge data is already on-device.
  EnergyResult energy(const DeviceEdgeData& dev) { return compute(dev, nullptr, nullptr); }
  EnergyResult energy_forces(const DeviceEdgeData& dev, std::vector<double>& forces) {
    return compute(dev, &forces, nullptr);
  }
  // Fully device-resident force path: leaves per-atom forces in `out_forces`
  // (allocated/resized to [N,3], F[i] = -dE/dx_i) without any host copy. This is
  // what a batched relaxer wants: fold forces onto DOFs without leaving the device.
  EnergyResult energy_forces(const DeviceEdgeData& dev, RView2D& out_forces) {
    return compute(dev, nullptr, &out_forces);
  }

  // Marshal a host EdgeData into device Views (one host->device copy per array).
  // `need_reverse` builds raw_off / raw_reverse, the per-atom segmentation and
  // partner map the adaptive-cutoff BACKWARD needs in order to gather rather
  // than scatter. They cost real time to construct -- on a 1728-atom supercell
  // the reverse map was the single largest item on the host path -- and nothing
  // else reads them, so an energy-only evaluation should not pay for them.
  //
  // Be careful turning it off: without them adaptive_backward falls back to an
  // atomic scatter, which is correct but NOT reproducible run to run. Pass false
  // only where no backward will run at all.
  DeviceEdgeData upload_edge_data(const EdgeData& ed, bool need_reverse = true) const;

  const Hypers& hypers() const { return h_; }

  // Adaptive-cutoff probe grid: constant given the hypers, so it is precomputed
  // once in the constructor and reused (the NEF builder and the backward both
  // consume it, avoiding a per-step allocation + host->device copy).
  const RView1D& probes() const { return probes_; }
  int n_probes() const { return n_probes_; }

  // Forward (+ optional analytic backward for forces) on device-resident data.
  // host_forces (if non-null) receives N*3 forces copied to the host; dev_forces
  // (if non-null) receives the device [N,3] force view with no host copy. Passing
  // either enables the backward; passing both is allowed.
  //
  // Public because nvcc forbids extended __host__ __device__ lambdas (KOKKOS_LAMBDA)
  // inside private/protected member functions; the energy*/energy_forces* wrappers
  // are the intended entry points.
  // dev_per_atom (if non-null) receives the device [N] per-atom energy view (no
  // copy) so a batched caller can segment it by structure on-device. dev_virial
  // (if non-null, batched only) receives the per-structure virial [n_struct,6]
  // (Voigt) on the device.
  EnergyResult compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                       RView2D* dev_forces, RView1D* dev_per_atom = nullptr,
                       RView2D* dev_virial = nullptr);

  // Per-atom charge/spin conditioning features, or an empty View when the model
  // is not conditioned. Computed once per evaluation and added to the node
  // features after every GNN layer. Forward only -- it is a per-system constant
  // with no dependence on any position, so nothing flows back into it.
  //
  // Public for the same reason compute() is: nvcc forbids extended
  // __host__ __device__ lambdas (KOKKOS_LAMBDA) inside a private or protected
  // member function. It is not part of the intended API.
  View2D conditioning(const DeviceEdgeData& dev, int N, int n_struct);

  // Residual-featurizer forward/backward (PostLN, LayerNorm, SiLU, non-expanded
  // central token, num_attention_layers>=1). compute() dispatches here when
  // featurizer_type == Residual. Same signature/contract as compute().
  EnergyResult compute_residual(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                                RView2D* dev_forces, RView1D* dev_per_atom = nullptr,
                                RView2D* dev_virial = nullptr);

 private:
  Hypers h_;
  double energy_scale_;
  std::vector<double> composition_;  // host, [n_species]
  std::vector<int> species_to_index_;  // Z -> species index (for building NEF from a System)
  RView1D comp_view_;                // device mirror of composition_ (double)
  RView1D probes_;                   // adaptive-cutoff probe grid (constant, precomputed)
  int n_probes_ = 0;
  Workspace ws_;                     // persistent scratch buffers reused per compute()

  std::unordered_map<std::string, View2D> mat_;  // matrices / embedding tables
  std::unordered_map<std::string, View1D> vec_;  // biases / norm weights
  // Ozaki slice decompositions of the weight matrices, when that path is in
  // use. Empty otherwise.
  //
  // Computed once here, at load, and reused by every evaluation for the life of
  // the model. That is the whole reason the Ozaki scheme is affordable for this
  // network: every GEMM in PET is `activation x weight`, and a slice
  // decomposition depends only on its own operand, so half of each product's
  // splitting work is done before the first evaluation runs.
  std::unordered_map<std::string, OzakiSplit> wsplit_;
  std::unordered_map<std::string, OzakiSplit> wsplit_t_;  // transposed layout

  // What the last evaluation actually cost: pool bytes and the atom count they
  // were for. A caller sizing a batch divides the two -- see
  // PetEvaluator::recommended_batch_atoms. Zero until something has been
  // evaluated.
 public:
  std::size_t workspace_bytes() const { return ws_.capacity_bytes(); }
  int last_n_atoms() const { return last_n_atoms_; }
  int last_max_neighbors() const { return last_max_neighbors_; }
  // The LARGEST edge-slot count evaluated so far. The pool is grow-only, so its
  // size corresponds to this, not to the most recent evaluation -- and the most
  // recent one is typically the smallest, because a batched relaxation harvests
  // converged structures out as it goes. Dividing pool bytes by the last count
  // instead of the peak understates the per-slot cost several-fold.
  long peak_edge_slots() const { return peak_edge_slots_; }
  // PEAK neighbour count, not the most recent. A batched relaxation harvests
  // converged structures as it goes, so the last evaluation of a chunk is a
  // handful of small structures whose M is well below the M the next chunk opens
  // with -- measured 38-43 against 56-57. M divides a batch-width budget, so
  // taking the tail value makes that budget too generous exactly when it matters.
  int peak_max_neighbors() const { return peak_max_neighbors_; }

 private:
  int last_n_atoms_ = 0;
  int last_max_neighbors_ = 0;
  long peak_edge_slots_ = 0;
  int peak_max_neighbors_ = 0;

  // A weight matrix, together with its Ozaki decomposition when there is one.
  //
  // Returned as a pair rather than threaded through linear()/linear_silu()/
  // linear_bwd() separately because those have some eighty call sites, all
  // spelled `linear(out, in, mat("..."), vec("..."))`. The implicit conversion
  // means every other use of mat() -- attention's projections, the embedding
  // gathers -- keeps compiling unchanged.
  WeightRef mat(const std::string& name) const;
  const View1D& vec(const std::string& name) const;

  // --- architecture-varying components, resolved from the loaded hypers -------
  //
  // The PET architecture is one pipeline whose pieces vary per model: the two
  // featurizers differ in which normalization and which activation they use, not
  // in what those things are FOR. Routing every call through these means a new
  // checkpoint that mixes them differently (RMSNorm with SiLU, say) works with no
  // new code, and it stops the two paths from drifting apart in the pieces they
  // genuinely share.
  //
  // The dispatch has to happen before the weight lookup, not after: RMSNorm
  // checkpoints carry no `.bias` at all, so asking for one would throw.
  void norm(View2D out, View2D in, const std::string& key) const;
  void norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key) const;

  // out = w_out(activation(w_in(in))). `pre` receives w_in(in) and is what the
  // backward reads: [R, 2*dff] for SwiGLU, whose w_in emits an interleaved
  // value/gate pair, or [R, dff] for a plain SiLU. Its width comes from the loaded
  // w_in, so the caller does not need to know which activation is in play.
  void feedforward(const std::string& key, View2D out, View2D in, const std::string& wkey,
                   View2D pre, bool save);
  void feedforward_bwd(const std::string& key, View2D in_adj, View2D out_adj,
                       const std::string& wkey, View2D pre);
  // Width of `pre` for a given w_in — 2*dff or dff, per the activation.
  int ffn_pre_width(const std::string& wkey) const;

  // Energy readout. Per readout layer: a node head and an edge head, each
  // Linear-SiLU-Linear-SiLU-Linear, summed into per_atom_net with the edge terms
  // weighted by mask*cutoff.
  //
  // The two featurizers differ only in HOW MANY sets of features they hand it and
  // where those come from -- the feedforward one reads out once from its final
  // node/edge features, the residual one reads out from every GNN layer and sums
  // -- which is num_readout_layers, a hyper. The heads themselves are identical,
  // so they are written once.
  //
  // sav_* receive the pre-activations the backward needs; pass empty vectors when
  // not computing gradients.
  void readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
               View1D per_atom_net, RView1D d_mask, View1D d_cutoff, int N, int M,
               std::vector<View2D>& sav_nh0, std::vector<View2D>& sav_nh1,
               std::vector<View2D>& sav_eh0, std::vector<View2D>& sav_eh1,
               std::vector<View2D>& sav_epred, const std::string& key, bool grad);

  void load_all(const Checkpoint& ckpt);
};

}  // namespace pet
