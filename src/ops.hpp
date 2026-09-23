// The building blocks of the PET network and of its analytic backward. Internal
// to the library.
//
// Conventions. Activations are row-major [rows, features] views. A backward
// takes the adjoint of its op's output and produces the adjoint of its input,
// and its `beta` (or `acc`) says whether that result overwrites its buffer (0)
// or accumulates onto it (1): a first writer needs no zeroed buffer.
#pragma once

#include "pet/model.hpp"

#include <string>

namespace pet {

using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using Team = TeamPolicy::member_type;
constexpr int kLanes = 32;        // vector lanes per team: one warp per row
constexpr int kMaxHeadDim = 128;  // attention's per-thread arrays

// ---- ops.cpp ----------------------------------------------------------------

// out = in W^T + b, onto beta * out.
void linear(View2D out, View2D in, const WeightRef& W, View1D b, Net beta = 0);
// in_adj = out_adj W, onto beta * in_adj.
void linear_bwd(View2D in_adj, View2D out_adj, const WeightRef& W, Net beta = 1);
// out = silu(in W^T + b); pre, unless empty, receives the pre-activation.
void linear_silu(View2D out, View2D pre, View2D in, const WeightRef& W, View1D b);
// grad *= silu'(pre), in place.
void silu_bwd(View2D grad, View2D pre);

// out(r, :) = table(idx(r), :).
void gather(View2D out, View2D table, IView1D idx);
// a += b, and a = b -- the latter a plain kernel, so it is always safe inside a
// recorded CUDA graph.
void add_inplace(View2D a, View2D b);
void copy(View2D a, View2D b);

// RMSNorm, or LayerNorm when `bias` is non-empty (see ops.cpp).
void norm_fwd(View2D out, View2D in, View1D weight, View1D bias);
void norm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight, bool layernorm, bool acc);

// out = w_out(v * sigmoid(g)) onto beta * out, where pre = w_in(in) holds v and
// g interleaved (v_j, g_j at 2j, 2j+1: w_in's rows are reordered at load so the
// pair meets in one GEMM thread). The backward reads pre.
void swiglu(Workspace& ws, View2D out, View2D in, const WeightRef& w_in, View1D b_in,
            const WeightRef& w_out, View1D b_out, View2D pre, Net beta);
// Its first half: pre = w_in(in), and h = v * sigmoid(g) unless h is empty. A
// recompute of pre must come through here, to run the forward's own GEMM.
void swiglu_in(View2D pre, View2D h, View2D in, const WeightRef& w_in, View1D b_in);
void swiglu_bwd(Workspace& ws, View2D in_adj, View2D out_adj, View2D pre, const WeightRef& w_in,
                const WeightRef& w_out, Net beta);

// The same with the GEMM and the elementwise step fused (fused_gemm.cpp), h
// optional; false when that cannot run here.
bool swiglu_in_fused(View2D pre, View2D h, View2D in, const View2D& w, View1D b);
bool swiglu_bwd_fused(View2D pre_adj, View2D out_adj, const View2D& w_out, View2D pre);

// out = silu(pre), pre = compress.0 of one GNN layer through its CompressFold.
// `sav`, unless empty, receives pre.
void compress_fwd(View2D out, View2D sav, const CompressFold& f, View2D input_edge,
                  const PackedEdges& pk);
// g: the adjoint of compress_fwd's output, turned in place into pre's. The
// geometry adjoint accumulates into x4_adj [E, 4] and, past layer 0, input_edge's
// into ie_adj.
void compress_bwd(View2D g, View2D pre, const CompressFold& f, View2D x4_adj, View2D ie_adj);

// e(n) (+)= node_pred(n) + sum over n's edges of cutoff * edge_pred.
void readout_accumulate(View1D e, View2D node_pred, View2D edge_pred, const PackedEdges& pk, bool acc);
// per_atom(n) = scale * net(n) + composition(species(n)).
void assemble_energy(RView1D per_atom, View1D net, IView1D species, RView1D composition,
                     double scale);

// ---- attention.cpp ----------------------------------------------------------

// Multi-head attention over each atom's S = M+1 tokens, with log(cf) as a
// per-key bias, followed by the output linear. `save` keeps what the backward
// needs under `key`.
void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, int N, int S, int heads, int head_dim,
               double temperature, bool save);
// in_adj (onto beta * in_adj) is the adjoint of the input linear's input;
// cf_seq_adj accumulates. `key` names the forward's saves.
void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, int N, int S, int heads, int head_dim,
                   double temperature, Net beta);

// ---- forces.cpp -------------------------------------------------------------

// soff(b) = the first atom of structure b, soff(NS) = N. A structure's atoms are
// contiguous; sid may be empty when NS == 1.
IView1D structure_offsets(Workspace& ws, const std::string& key, IView1D sid, int N, int NS);
// out(b, :) (+)= the sum of x over structure b's atoms, in atom order.
void sum_by_structure(RView2D x, IView1D soff, RView2D out, bool acc);

// Forces and virial from the adjoints the network leaves on each edge: x4_adj
// ([E, 4], of (v, |v|)), cutoff_adj ([E], of the readout's cutoff factor) and
// cf_seq_adj ([N, S], of the attention bias). Adds the adaptive cutoff's share
// when the model has one. Returns forces [N, 3] and the per-structure virial
// [n_struct, 9], both scaled by `scale`.
void forces_and_virial(Workspace& ws, const DeviceEdgeData& dev, const PackedEdges& pk,
                       const Hypers& h, RView1D probes, int n_probes, double scale, View2D x4_adj,
                       View1D cutoff_adj, View2D cf_seq_adj, RView2D& forces, RView2D& vir9);

}  // namespace pet
