// The PET network: weights on the device, and energy, forces and virial from a
// neighbour list, by a forward pass and a hand-written analytic backward.
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pet/checkpoint.hpp"
#include "pet/config.hpp"
#include "pet/exchange.hpp"
#include "pet/graph.hpp"
#include "pet/kokkos.hpp"
#include "pet/neighbors.hpp"
#include "pet/ozaki.hpp"

namespace pet {

struct EnergyResult {
  double total = 0.0;
  std::vector<double> per_atom;  // [N] eV
  // Symmetric virial W = V * stress, Voigt order [xx, yy, zz, xy, xz, yz], eV.
  // Filled only with forces.
  double virial[6] = {0, 0, 0, 0, 0, 0};
};

// A batched evaluation's results, left on the device. Atom a belongs to
// structure struct_id(a).
struct BatchResult {
  RView1D energy;     // [B] eV
  RView2D forces;     // [N, 3] eV/A; empty without forces
  RView2D virial;     // [B, 6] Voigt, eV; empty without forces
  IView1D struct_id;  // [N]
  RView1D per_atom;   // [N] eV; 0 for atoms past n_local
  // dE/dv for every raw edge v = r_j + shift - r_i, in the raw list's order: what
  // forces, the virial, per-atom stress etc. are all folded from. With forces,
  // on device-built lists; empty otherwise.
  RView2D edge_grad;  // [n_raw, 3] eV/A
  int n_struct = 0;
  int n_atoms = 0;
};

// A neighbour list on the device, in the layout the model reads: atom n's kept
// edges in slots [n*M, n*M + count), then padding up to M, the largest count.
// Built by upload_edge_data (from a host EdgeData) or build_device_edge_data.
struct DeviceEdgeData {
  int n_atoms = 0;        // N
  int max_neighbors = 0;  // M
  int n_raw = 0;          // raw (pre-adaptive-cutoff) edges
  int n_edges = -1;       // kept edges; -1 = unknown, counted when needed
  bool padded = false;    // every slot is an edge row, padding zero-weight: fixed shapes
  // n_edges is a capacity: the kept edges are packed into its first rows (their
  // count is the packed offsets' last), the rest are dead, and more than fit
  // set overflow(0). Fixed shapes without padding every atom to M.
  bool edge_capacity = false;
  IView1D overflow;
  // Over several ranks with a capacity: room for this many live edges to ghosts
  // (Exchange::set_live), the rest -1; more set overflow(0).
  int live_capacity = 0;
  int n_centres = -1;     // atoms [0, n_centres) have neighbourhoods and are the
                          // model's rows; the rest are only neighbours (over
                          // several ranks, the ghosts). -1 = all
  int centres() const { return n_centres < 0 ? n_atoms : n_centres; }
  int n_local = -1;       // atoms [0, n_local) count toward the energy, the rest
                          // (an MD engine's ghosts) only shape it; -1 = all

  IView1D species;        // [N]
  IView1D neigh_species;  // [N*M] the neighbour's species (padding: 0)
  IView1D reverse_index;  // [N*M] slot of the (j, i, -shift) edge, -1 if none
  RView2D edge_vec;       // [N*M, 3] r_j + shift - r_i
  RView1D dist;           // [N*M]
  RView1D mask;           // [N*M] 1 kept, 0 padding
  RView1D pair_cutoff;    // [N*M]
  View1D cutoff_factor;   // [N*M] the smooth cutoff factor
  View2D cf_seq;          // [N, M+1] attention bias source: 1, then the factors

  // The raw list, for the adaptive cutoff's backward: edges grouped by centre,
  // raw_off(a) where atom a's begin, raw_reverse each edge's partner. Without
  // raw_off/raw_reverse that backward falls back to atomics.
  IView1D raw_center, raw_neigh;  // [E]
  RView1D raw_dist;               // [E]
  RView2D raw_vec;                // [E, 3]
  IView1D raw_off;                // [N+1]
  IView1D raw_reverse;            // [E]
  IView1D raw_slot;               // [E] the kept edge's slot n*M + m, -1 if dropped
  // Raw edges with no partner -- into a ghost the engine listed no neighbours
  // for -- by target atom: orphan_edge[orphan_off(a), orphan_off(a+1)) point at a.
  IView1D orphan_off, orphan_edge;  // [N+1], [n_orphans]
  // Over several ranks (exchange.hpp): the engine's side of the swap, and each
  // edge to a ghost, in the engine's order, as a raw index.
  Exchange* exchange = nullptr;
  IView1D remote_raw;  // [n_remote]

  // What the adaptive cutoff's forward computed, kept for its backward. Grid:
  // adapt_eff, the smoothed neighbour counts [N, P] (optional; recomputed if
  // absent). Solver: adapt_r, the root, and adapt_dn, the slope there (0 when the
  // cutoff was clamped) -- required.
  RView2D adapt_eff;
  RView1D adapt_r, adapt_dn;

  // A batch: structure of each atom (atoms of a structure are contiguous), and
  // the number of structures. Empty struct_id means one structure.
  IView1D struct_id;
  int n_struct = 1;

  // [n_struct] total charge and spin multiplicity, for a conditioned model.
  IView1D charge, spin_multiplicity;
};

// A weight matrix and, when the Ozaki GEMM is on, its slice decompositions --
// one for each orientation a weight is used in (transposed in linear(), not in
// linear_bwd()). The conversion to View2D is explicit because an implicit one
// would silently drop the decompositions.
struct WeightRef {
  View2D v;
  const OzakiSplit* split = nullptr;
  const OzakiSplit* split_t = nullptr;
  WeightRef() = default;
  WeightRef(const View2D& m, const OzakiSplit* s = nullptr, const OzakiSplit* st = nullptr)
      : v(m), split(s), split_t(st) {}
  const OzakiSplit* for_orientation(bool transposed) const { return transposed ? split : split_t; }
  explicit operator const View2D&() const { return v; }
  std::size_t extent(int d) const { return v.extent(d); }
};

// compress.0 of one GNN layer with its constant inputs folded in at load time.
// Its input is [edge_emb | nb_emb | input_edge]: edge_emb = edge_embedder(v, |v|)
// is linear in the geometry, and nb_emb (at layer 0, input_edge) is a species
// lookup, so compress.0 applied to them is a [D, 4] matrix and a species table.
// Only input_edge past layer 0 is an activation and needs a GEMM.
struct CompressFold {
  View2D wx;     // [D, 4]
  View1D b;      // [D]
  View2D tab;    // [n_species, D]
  WeightRef wi;  // [D, D]; empty at layer 0
};

// The kept edges, packed: row e of every edge tensor in the model is a real edge,
// where the builders' slot layout pads every atom up to M. Atom n's edges are
// [off(n), off(n+1)), in slot order. Attention keeps the padded [N, M+1] token
// layout: token (n, 1 + m) is edge off(n) + m when that is below off(n+1).
struct PackedEdges {
  int E = 0;
  IView1D off;         // [N+1]
  IView1D center;      // [E] the edge's own atom
  IView1D species;     // [E] the neighbour's species
  IView1D reverse;     // [E] the (j, i, -shift) edge, -1 if none
  RView2D vec;         // [E, 3]
  RView1D dist, pcut;  // [E]
  View1D cut;          // [E] smooth cutoff factor
  IView1D slot_edge;   // [N*M] each slot's packed edge, -1 for padding
  int n_local = 0;     // see DeviceEdgeData::n_local
  int n_remote = 0;     // edges to ghosts (DeviceEdgeData::remote_raw)
  int n_live = 0;       // of those, the ones kept this evaluation (Exchange::set_live)
  IView1D remote_of;    // [E] each edge's row among the live ones, -1 if its partner is here
  IView1D live_packed;  // [n_live] each live remote edge's packed index
};

// One evaluation's outputs, on the device: per-atom energy [N] and, with forces,
// forces [N, 3] and the per-structure virial [n_struct, 9] (row-major 3x3).
struct DeviceOut {
  RView1D per_atom;
  RView2D forces, vir9, edge_grad;
};

// What one GNN layer of the feedforward featurizer saves for its backward, one
// entry per attention block where it is a vector. Workspace keys carry `tag`.
struct LayerSaves {
  std::string tag;
  View2D cpre, concat, cph;
  std::vector<View2D> tokens, qkv, node_new, tmp_center, eps, tmp_edge;
};

// What the readout saves for its backward, one entry per readout layer.
struct ReadoutSaves {
  std::vector<View2D> nh0, nh1, eh0, eh1, epred;
};

class PetModel {
 public:
  explicit PetModel(const Checkpoint& ckpt);

  // Energy, or energy and forces (F = -dE/dx, eV/A, [N*3]), from a host list.
  EnergyResult energy(const EdgeData& ed) {
    return compute(upload_edge_data(ed, /*need_reverse=*/false), nullptr, nullptr);
  }
  EnergyResult energy_forces(const EdgeData& ed, std::vector<double>& forces) {
    return compute(upload_edge_data(ed), &forces, nullptr);
  }
  // The same from a device list; the second leaves the forces on the device.
  EnergyResult energy(const DeviceEdgeData& dev) { return compute(dev, nullptr, nullptr); }
  EnergyResult energy_forces(const DeviceEdgeData& dev, std::vector<double>& forces) {
    return compute(dev, &forces, nullptr);
  }
  EnergyResult energy_forces(const DeviceEdgeData& dev, RView2D& forces) {
    return compute(dev, nullptr, &forces);
  }

  // Many structures as one evaluation, results left on the device. Each gets
  // the answer it would get alone. The second form takes a list already built
  // over the whole batch, with struct_id and n_struct set.
  BatchResult energy_forces_batch(const std::vector<System>& systems, bool compute_forces = true);
  BatchResult energy_forces_batch(const DeviceEdgeData& dev, bool compute_forces = true);

  // A host list to the device, as one copy. The views alias the model's upload
  // buffer and are valid until the next upload. `need_reverse` builds raw_off
  // and raw_reverse, which only the adaptive cutoff's backward reads.
  DeviceEdgeData upload_edge_data(const EdgeData& ed, bool need_reverse = true) const;

  // One evaluation. host_forces or dev_forces (or both) ask for forces; with
  // dev_per_atom and dev_virial ([n_struct, 6], Voigt) the energy and virial stay
  // on the device, and the EnergyResult is left empty.
  EnergyResult compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                       RView2D* dev_forces, RView1D* dev_per_atom = nullptr,
                       RView2D* dev_virial = nullptr, RView2D* dev_edge_grad = nullptr);

  const Hypers& hypers() const { return h_; }
  // The adaptive cutoff's probe grid, for the neighbour builders.
  const RView1D& probes() const { return probes_; }
  int n_probes() const { return n_probes_; }

  // Memory the saved activations may take before Recompute::Auto trades them for
  // recomputation (0 = unlimited), and the policy.
  void set_memory_policy(std::size_t budget_bytes, Recompute r) { mem_budget_ = budget_bytes, recompute_ = r; }
  // Replay repeated evaluations as one CUDA graph (graph.hpp). On by default.
  void set_graphs(bool on) { graphs_ = on; }
  bool graphs() const { return graphs_; }
  std::size_t ws_generation() const { return ws_.generation(); }

  // What evaluations have cost, for sizing batches: the pool, largest buffers
  // first, and the largest shapes seen so far (the pool is grow-only, so it
  // reflects those, not the latest).
  std::size_t workspace_bytes() const { return ws_.capacity_bytes(); }
  std::vector<std::pair<std::string, std::size_t>> workspace_breakdown() const {
    return ws_.capacity_breakdown();
  }
  long peak_edge_slots() const { return peak_edge_slots_; }
  // Device bytes per edge slot with every save kept, from the shapes (as
  // recompute_tier estimates them), for sizing a batch before any is measured.
  std::size_t bytes_per_slot() const;
  int peak_max_neighbors() const { return peak_max_neighbors_; }

  // Internal, and public only because nvcc refuses a KOKKOS_LAMBDA inside a
  // private member function.
  DeviceOut ff_pass(const DeviceEdgeData& dev, bool grad);        // feedforward.cpp
  DeviceOut residual_pass(const DeviceEdgeData& dev, bool grad);  // residual.cpp
  // remote_in: the partners' rows for edges to ghosts, fetched when `share`,
  // else (a recompute) already there.
  void ff_layer(const DeviceEdgeData& dev, const PackedEdges& pk, int L, View2D& node,
                View2D input_edge, View2D cond, LayerSaves* sav, bool save_wide, View2D remote_in = {},
                bool share = false);
  void ff_layer_bwd(const DeviceEdgeData& dev, const PackedEdges& pk, int L, const LayerSaves& sav,
                    bool kept_wide, View2D node_adj, View2D input_edge_adj, View2D x4_adj,
                    View2D cf_seq_adj);
  PackedEdges pack_edges(const DeviceEdgeData& dev);
  View2D conditioning(const DeviceEdgeData& dev);
  void readout_bwd(const ReadoutSaves& sav, int i, const PackedEdges& pk, int N, View2D node_adj,
                   View2D edge_adj, View1D cutoff_adj, bool acc_cutoff);

 private:
  WeightRef mat(const std::string& name) const;
  const View1D& vec(const std::string& name) const;
  CompressFold compress_fold(int layer) const;

  // The pieces the two featurizers share but configure differently (which norm,
  // which activation), dispatched on the hypers.
  void norm(View2D out, View2D in, const std::string& key) const;
  void norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key, bool acc = true) const;
  // out = w_out(act(w_in(in))) onto beta * out; `pre` receives w_in(in), [R,
  // ffn_pre_width] (2*dff for SwiGLU's value|gate pair, dff for SiLU).
  void feedforward(View2D out, View2D in, const std::string& key, View2D pre, bool save, Net beta = 0);
  void feedforward_bwd(View2D in_adj, View2D out_adj, const std::string& key, View2D pre, Net beta = 1);
  int ffn_pre_width(const std::string& key) const;
  // Per readout layer i: node and edge heads (Linear-SiLU-Linear-SiLU-Linear),
  // summed into per_atom_net, the edge terms weighted by the cutoff factor. `sav`
  // null saves nothing.
  void readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
               View1D per_atom_net, const PackedEdges& pk, ReadoutSaves* sav, const std::string& key);
  // How much of the forward the backward keeps (see Recompute), from the shapes.
  Recompute recompute_tier(int N, int S, int E) const;
  std::vector<std::uintptr_t> graph_key(const DeviceEdgeData& dev, bool grad) const;
  void load_all(const Checkpoint& ckpt);

  Hypers h_;
  double energy_scale_;
  std::vector<double> composition_;    // [n_species]
  std::vector<int> species_to_index_;  // Z -> species index
  RView1D comp_view_;                  // composition_ on the device
  RView1D probes_;
  int n_probes_ = 0;

  std::unordered_map<std::string, View2D> mat_;
  std::unordered_map<std::string, View1D> vec_;
  std::unordered_map<std::string, OzakiSplit> wsplit_, wsplit_t_;  // Ozaki only

  Workspace ws_;
  mutable Upload upload_;
  std::size_t mem_budget_ = 0;
  Recompute recompute_ = Recompute::Auto;
  bool graphs_ = true;
  GraphCache<DeviceOut> graph_;

  long peak_edge_slots_ = 0;
  int peak_max_neighbors_ = 0;
};

}  // namespace pet
