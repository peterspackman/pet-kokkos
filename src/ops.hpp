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

using Team = TeamPolicy::member_type;
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
// grad = linear_bwd(out_adj through W), then silu_bwd by pre: the backward of
// linear_silu's successor and its activation, in one pass where it can.
void linear_silu_bwd(View2D grad, View2D out_adj, const WeightRef& W, View2D pre);

// out(r, :) = table(idx(r), :).
void gather(View2D out, View2D table, IView1D idx);
// a = b, a plain kernel, so it is always safe inside a recorded CUDA graph.
void copy(View2D a, View2D b);

// RMSNorm, or LayerNorm when `bias` is non-empty (norm_rows below).
// add, unless empty, is first added into in (a residual sum the norm reads);
// also, unless empty, receives a second copy of in_adj.
void norm_fwd(View2D out, View2D in, View1D weight, View1D bias, View2D add = {});
void norm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight, bool layernorm, bool acc, View2D also = {});

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
// out = in . W^T (w_transposed, W [N, K]) or in . W (W [K, N]), + b, + beta out.
bool linear_fused(View2D out, View2D in, const View2D& w, bool w_transposed, View1D b, Net beta);
bool silu_in_fused(View2D out, View2D pre, View2D in, const View2D& w, View1D b);
bool silu_bwd_fused(View2D grad, View2D out_adj, const View2D& w, View2D pre);
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
// per_atom(n) = scale * net(n) + composition(species(n)), and 0 past n_local.
void assemble_energy(RView1D per_atom, View1D net, IView1D species, RView1D composition,
                     double scale, int n_local);

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
// out(b, :) (+)= the sum of x over structure b's atoms, in a fixed order.
void sum_by_structure(RView2D x, IView1D soff, RView2D out, bool acc);

// Forces and virial from the adjoints the network leaves on each edge: x4_adj
// ([E, 4], of (v, |v|)), cutoff_adj ([E], of the readout's cutoff factor) and
// cf_seq_adj ([N, S], of the attention bias). Adds the adaptive cutoff's share
// when the model has one. Returns forces [N, 3] and the per-structure virial
// [n_struct, 9], both scaled by `scale`, and -- when dev has raw_slot -- dE/dv for
// every raw edge [n_raw, 3], which the forces and virial are a fold of.
void forces_and_virial(Workspace& ws, const DeviceEdgeData& dev, const PackedEdges& pk,
                       const Hypers& h, RView1D probes, int n_probes, double scale, View2D x4_adj,
                       View1D cutoff_adj, View2D cf_seq_adj, RView2D& forces, RView2D& vir9, RView2D& edge_grad);

// ---- the norm kernels -------------------------------------------------------
//
// RMSNorm and LayerNorm are one operation: out = (x - mu) * inv * weight (+ bias),
// where LayerNorm centres (mu = mean) and RMSNorm does not (mu = 0). One warp per
// row, in the network type (as torch does), and in two passes so LayerNorm's
// variance never subtracts two large squares; the second pass reads from L1.
// A block holds several rows: one 32-thread block per row would leave the GPU at
// its limit of resident blocks with half its warp slots empty.
//
// A norm reads its rows just after another pass writes them, or its adjoint is
// just scattered somewhere: those passes ride along, a row at a time, as the
// norm's prologue and sink, saving a trip through memory and a launch.

constexpr double kRmsNormEps = 1.1920928955078125e-07;  // float32 epsilon
constexpr double kLayerNormEps = 1e-5;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
constexpr int kNormRows = 256 / kLanes;
#else
constexpr int kNormRows = 1;
#endif

inline TeamPolicy norm_policy(int rows) { return TeamPolicy((rows + kNormRows - 1) / kNormRows, kNormRows, kLanes); }

KOKKOS_INLINE_FUNCTION Net norm_inv(Net sq, int D, bool ln) {
  return Net(1) / Kokkos::sqrt(sq / D + Net(ln ? kLayerNormEps : kRmsNormEps));
}

KOKKOS_INLINE_FUNCTION Net row_mean(const Team& t, const View2D& x, int r) {
  Net s = Net(0);
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, int(x.extent(1))), [&](int d, Net& a) { a += x(r, d); }, s);
  return s / x.extent(1);
}

// out = norm(in), row by row, after pro(r, d) has written in(r, d) for every d
// of row r. Each lane rereads only what it (or pro) wrote, so the passes need no
// barrier, and out may be in itself.
template <class Pro>
void norm_rows(const char* name, View2D out, View2D in, View1D weight, View1D bias, Pro pro) {
  const bool ln = bias.extent(0) > 0;
  const int N = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      name, norm_policy(N), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank() * kNormRows + t.team_rank();
        if (r >= N) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) { pro(r, d); });
        const Net mu = ln ? row_mean(t, in, r) : Net(0);
        Net v = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D), [&](int d, Net& s) { s += (in(r, d) - mu) * (in(r, d) - mu); }, v);
        const Net inv = norm_inv(v, D, ln);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          out(r, d) = (in(r, d) - mu) * inv * weight(d) + (ln ? bias(d) : Net(0));
        });
      });
}

// The backward: sink(r, d, g) receives in's adjoint g. With xhat = (x - mu) * inv
// and gw = out_adj * weight:
//   g = inv * (gw - [mean(gw) for LayerNorm] - xhat * mean(gw * xhat)).
// The bias is a shift, so it does not appear. The variance and both gradient
// sums come from one pass over the row (after the mean, for LayerNorm).
template <class Sink>
void norm_rows_bwd(const char* name, View2D out_adj, View2D in, View1D weight, bool ln, Sink sink) {
  const int N = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      name, norm_policy(N), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank() * kNormRows + t.team_rank();
        if (r >= N) return;
        const Net mu = ln ? row_mean(t, in, r) : Net(0);
        Net sq = Net(0), sg = Net(0), sgx = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& a, Net& b, Net& c) {
              const Net x = in(r, d) - mu, gw = out_adj(r, d) * weight(d);
              a += x * x;
              b += gw;
              c += gw * x;
            },
            sq, sg, sgx);
        const Net inv = norm_inv(sq, D, ln);
        const Net mg = ln ? sg / D : Net(0), mgx = sgx * inv / D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          sink(r, d, inv * (out_adj(r, d) * weight(d) - mg - (in(r, d) - mu) * inv * mgx));
        });
      });
}

}  // namespace pet
