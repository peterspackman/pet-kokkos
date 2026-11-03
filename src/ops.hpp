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
#include <type_traits>

namespace pet {

using Team = TeamPolicy::member_type;
constexpr int kMaxHeadDim = 128;  // attention's per-thread arrays

// ---- token rows ---------------------------------------------------------------
//
// The attention runs over each atom's tokens: its own, then one per kept edge.
// They are rows of one [N + E, width] matrix: atom n's own token is row n and
// its edge tokens are rows N + off(n) .. N + off(n+1) - 1, so token N + k is
// edge k's, aligned with the edge rows. Atom n has count(n) tokens; slot s of
// it (0 its own, s its s-th kept edge) is row(n, s). Per-slot values -- cf_seq,
// the softmax's max and sum -- stay [N, S], S the most tokens an atom has.
// With a capacity, the edges past off(N) are dead: their rows are no atom's,
// and attention zeroes them (`dead`).
struct Tokens {
  IView1D off;
  int N = 0;
  bool dead = false;  // edge rows past off(N) may exist
  KOKKOS_INLINE_FUNCTION int count(int n) const { return off(n + 1) - off(n) + 1; }
  KOKKOS_INLINE_FUNCTION int row(int n, int s) const { return s == 0 ? n : N + off(n) + s - 1; }
};

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
// The combination norm's backward and the gather after it, in one pass. The
// norm's input row k is [oe(k) | oe(rev(k))] (0 where an edge has no partner),
// so out_edge's adjoint takes row k's first half and its partner's second:
//   edge_adj(k) = ie_adj(k) + g(k)[:D] + g(rev(k))[D:],
// g the norm's input adjoint from out_adj [E, 2D]. One team takes an edge and
// its partner together, reading each oe and out_adj row once, and neither the
// concat nor its adjoint is ever written. Requires rev to pair edges up
// (rev(rev(k)) = k); an edge whose partner does not point back is done alone.
void concat_norm_bwd(View2D edge_adj, View2D out_adj, View2D oe, View1D weight, bool layernorm, View2D ie_adj,
                     IView1D rev);

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

// On AMD's CDNA GPUs the fused GEMMs are fused_gemm_hip.cpp's (MFMA), not
// fused_gemm.cpp's.
#if defined(KOKKOS_ENABLE_HIP) && !defined(PET_KOKKOS_FP64) &&                                     \
    (defined(KOKKOS_ARCH_AMD_GFX908) || defined(KOKKOS_ARCH_AMD_GFX90A) || defined(KOKKOS_ARCH_AMD_GFX940) || \
     defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU))
#define PET_FUSED_HIP 1
#endif
// The same with the GEMM and the elementwise step fused (fused_gemm.cpp), h
// optional; false when that cannot run here.
bool swiglu_in_fused(View2D pre, View2D h, View2D in, const View2D& w, View1D b);
// out = in . W^T (w_transposed, W [N, K]) or in . W (W [K, N]), + b, + beta out.
bool linear_fused(View2D out, View2D in, const View2D& w, bool w_transposed, View1D b, Net beta);
bool silu_in_fused(View2D out, View2D pre, View2D in, const View2D& w, View1D b);
bool silu_bwd_fused(View2D grad, View2D out_adj, const View2D& w, View2D pre);
bool swiglu_bwd_fused(View2D pre_adj, View2D out_adj, const View2D& w_out, View2D pre);
// All of swiglu_bwd with pre_adj never written out: made as the second GEMM
// loads it (fused_gemm_hip.cpp, AMD CDNA); false elsewhere.
bool swiglu_bwd_deferred(Workspace& ws, View2D in_adj, View2D out_adj, View2D pre, const WeightRef& w_in,
                         const WeightRef& w_out, Net beta);

// out = silu(pre), pre = compress.0 of one GNN layer through its CompressFold.
// `sav`, unless empty, receives pre.
void compress_fwd(View2D out, View2D sav, const CompressFold& f, View2D input_edge,
                  const PackedEdges& pk);
// g receives pre's adjoint, from et_adj (the adjoint of compress.2's output, w2
// its weight) back through compress.2 and the silu. The geometry adjoint
// accumulates into x4_adj [E, 4] and, past layer 0, input_edge's into ie_adj.
void compress_bwd(View2D g, View2D et_adj, const WeightRef& w2, View2D pre, const CompressFold& f, View2D x4_adj,
                  View2D ie_adj);

// e(n) (+)= node_pred(n) + sum over n's edges of cutoff * edge_pred.
void readout_accumulate(View2D out, View2D node_pred, View2D edge_pred, const PackedEdges& pk, bool acc);
// per_atom(n) = scale * net(n) + composition(species(n)), and 0 past n_local.
void assemble_energy(RView1D per_atom, View1D net, IView1D species, RView1D composition,
                     double scale, int n_local);

// ---- attention.cpp ----------------------------------------------------------

// Multi-head attention over each atom's tokens (Tokens above), with log(cf) as a
// per-key bias, followed by the output linear. `save` keeps what the backward
// needs under `key`.
void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, const Tokens& tk, int S, int heads, int head_dim,
               double temperature, bool save);
// in_adj (onto beta * in_adj) is the adjoint of the input linear's input;
// cf_seq_adj accumulates. `key` names the forward's saves.
void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, const Tokens& tk, int S, int heads, int head_dim,
                   double temperature, Net beta);

// The tangents of the heads' attention for the Hessian-vector product (hvp.cpp)
// on AMD's matrix cores (fp32), or on NVIDIA's tensor cores in the TF32
// attention mode, K tangents stacked in rows; false when they cannot run here
// (neither, or a head or token count the kernels do not hold), and then the
// generic kernels must. merged and ml are the forward's saves; the
// forward's tangent leaves merged_t and each query's pbar ([K N H S]) for the
// backward's, which writes qkv_adj_t and adds onto cf_adj_t.
bool attention_tan_fast(View2D merged_t, View2D pbar, View2D qkv, View2D qkv_t, View2D merged, View2D ml,
                        View2D cf_seq, View2D cf_t, const Tokens& tk, int S, int heads, int head_dim,
                        double temperature, int K);
bool attention_bwd_tan_fast(Workspace& ws, View2D qkv_adj_t, View2D cf_adj_t, View2D dm, View2D dm_t, View2D qkv,
                            View2D qkv_t, View2D merged, View2D merged_t, View2D ml, View2D pbar, View2D cf_seq,
                            View2D cf_t, const Tokens& tk, int S, int heads, int head_dim, double temperature, int K);

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

// a * b rounded on its own and never contracted into an add that follows: the
// value a kernel that stores the product and another that reads it back would
// see. A fused pass keeps the unfused passes' bits with it.
KOKKOS_INLINE_FUNCTION Net mul_rn(Net a, Net b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if constexpr (sizeof(Net) == 4) return __fmul_rn(a, b);
  else return __dmul_rn(a, b);
#else
  volatile Net p = a * b;
  return p;
#endif
}

KOKKOS_INLINE_FUNCTION Net norm_inv(Net sq, int D, bool ln) {
  return Net(1) / Kokkos::sqrt(sq / D + Net(ln ? kLayerNormEps : kRmsNormEps));
}

// On AMD a norm's row lives in a wavefront's registers: lane l holds columns
// l + 64 j, read from memory once (coalesced), with the sums a butterfly over the
// lanes, which leaves every lane the same bits. The Kokkos passes below read a
// row three times, the later two from cache, which on an MI250X's 16 KiB L1 means
// from L2: they ran at about half the HBM bandwidth (pet-mad-s, 5184 water, 18 ms
// of norms an evaluation). Widths a multiple of 64 up to 2048 (every published
// model's); others take the Kokkos passes.
#if defined(KOKKOS_ENABLE_HIP) && defined(__HIPCC__)
#define PET_HIP_NORMS 1
namespace hip_rows {

template <class T>
__device__ inline T wave_sum(T x) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

// Calls f.template row<V>(r, lane) for every row, a wavefront a row, four a block.
template <int V, class F>
__global__ void __launch_bounds__(256) rows_kernel(int N, F f) {
  const int r = blockIdx.x * 4 + int(threadIdx.x) / 64;
  if (r < N) f.template row<V>(r, int(threadIdx.x) % 64);
}

// Launch with V = D / 64 a compile-time constant; false for a width not taken.
template <class F>
bool launch(const char* name, int N, int D, const F& f) {
  if (D % 64 != 0) return false;
  const int V = D / 64;
  if (V != 2 && V != 4 && V != 6 && V != 8 && V != 12 && V != 16 && V != 24 && V != 32) return false;
  if (N == 0) return true;
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  const hipStream_t st = ExecSpace().hip_stream();
  const dim3 grid((N + 3) / 4), block(256);
  switch (V) {
    case 2: hipLaunchKernelGGL((rows_kernel<2, F>), grid, block, 0, st, N, f); break;
    case 4: hipLaunchKernelGGL((rows_kernel<4, F>), grid, block, 0, st, N, f); break;
    case 6: hipLaunchKernelGGL((rows_kernel<6, F>), grid, block, 0, st, N, f); break;
    case 8: hipLaunchKernelGGL((rows_kernel<8, F>), grid, block, 0, st, N, f); break;
    case 12: hipLaunchKernelGGL((rows_kernel<12, F>), grid, block, 0, st, N, f); break;
    case 16: hipLaunchKernelGGL((rows_kernel<16, F>), grid, block, 0, st, N, f); break;
    case 24: hipLaunchKernelGGL((rows_kernel<24, F>), grid, block, 0, st, N, f); break;
    default: hipLaunchKernelGGL((rows_kernel<32, F>), grid, block, 0, st, N, f); break;
  }
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
  return true;
}

// A row's statistics from the lane's V values x (and, for the backward, the
// output adjoint ga with the weight): the mean (LayerNorm), 1/rms, and the
// backward's two gradient means.
struct Stats {
  Net mu, inv, mg, mgx;
};
template <int V>
__device__ inline Net mean(const Net* x, int D, bool ln) {
  if (!ln) return Net(0);
  Net s = Net(0);
#pragma unroll
  for (int j = 0; j < V; ++j) s += x[j];
  return wave_sum(s) / D;
}
template <int V, class W>
__device__ inline Stats bwd_stats(const Net* x, const Net* ga, const W& weight, int l, int D, bool ln) {
  const Net mu = mean<V>(x, D, ln);
  Net sq = Net(0), sg = Net(0), sgx = Net(0);
#pragma unroll
  for (int j = 0; j < V; ++j) {
    const Net xv = x[j] - mu, gw = ga[j] * weight(l + 64 * j);
    sq += xv * xv;
    sg += gw;
    sgx += gw * xv;
  }
  sq = wave_sum(sq), sg = wave_sum(sg), sgx = wave_sum(sgx);
  const Net inv = Net(1) / Kokkos::sqrt(sq / D + Net(ln ? 1e-5 : 1.1920928955078125e-07));
  return {mu, inv, ln ? sg / D : Net(0), sgx * inv / D};
}
// The input adjoint of one value: inv (ga w - mg - xhat mgx), xhat = (x - mu) inv.
__device__ inline Net grad(const Stats& s, Net x, Net ga, Net w) {
  const Net a = Kokkos::fma(ga, w, -s.mg), b = mul_rn(x - s.mu, s.inv);
  return mul_rn(s.inv, Kokkos::fma(-b, s.mgx, a));
}

template <class Pro>
struct Fwd {
  View2D out, in;
  View1D weight, bias;
  bool ln;
  Pro pro;
  template <int V>
  __device__ void row(int r, int l) const {
    const int D = 64 * V;
    Net x[V];
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const int d = l + 64 * j;
      if constexpr (std::is_void_v<decltype(pro(r, d))>) {
        pro(r, d);
        x[j] = in(r, d);
      } else {  // the value itself: stored only where in is not the output too
        x[j] = pro(r, d);
        if (in.data() != out.data()) in(r, d) = x[j];
      }
    }
    const Net mu = mean<V>(x, D, ln);
    Net v = Net(0);
#pragma unroll
    for (int j = 0; j < V; ++j) v += (x[j] - mu) * (x[j] - mu);
    v = wave_sum(v);
    const Net inv = Net(1) / Kokkos::sqrt(v / D + Net(ln ? 1e-5 : 1.1920928955078125e-07));
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const int d = l + 64 * j;
      out(r, d) = (x[j] - mu) * inv * weight(d) + (ln ? bias(d) : Net(0));
    }
  }
};

template <class Sink>
struct Bwd {
  View2D out_adj, in;
  View1D weight;
  bool ln;
  Sink sink;
  template <int V>
  __device__ void row(int r, int l) const {
    Net x[V], ga[V];
#pragma unroll
    for (int j = 0; j < V; ++j) x[j] = in(r, l + 64 * j), ga[j] = out_adj(r, l + 64 * j);
    const Stats s = bwd_stats<V>(x, ga, weight, l, 64 * V, ln);
#pragma unroll
    for (int j = 0; j < V; ++j) sink(r, l + 64 * j, grad(s, x[j], ga[j], weight(l + 64 * j)));
  }
};

}  // namespace hip_rows
#endif

KOKKOS_INLINE_FUNCTION Net row_mean(const Team& t, const View2D& x, int r) {
  Net s = Net(0);
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, int(x.extent(1))), [&](int d, Net& a) { a += x(r, d); }, s);
  return s / x.extent(1);
}

// A norm's prologue at (r, d): it writes in(r, d), or returns what goes there.
template <class Pro>
KOKKOS_INLINE_FUNCTION void run_pro(const Pro& pro, const View2D& in, int r, int d) {
  if constexpr (std::is_void_v<decltype(pro(r, d))>) pro(r, d);
  else in(r, d) = pro(r, d);
}

// out = norm(in), row by row, after pro(r, d) has written in(r, d) for every d
// of row r -- or, a pro that returns a Net, after in(r, d) = pro(r, d), which on
// HIP is not stored where in is out too (the row is in registers). Each lane
// rereads only what it (or pro) wrote, so the passes need no barrier, and out
// may be in itself.
template <class Pro>
void norm_rows(const char* name, View2D out, View2D in, View1D weight, View1D bias, Pro pro) {
  const bool ln = bias.extent(0) > 0;
  const int N = in.extent(0), D = in.extent(1);
#if defined(PET_HIP_NORMS)
  if (hip_rows::launch(name, N, D, hip_rows::Fwd<Pro>{out, in, weight, bias, ln, pro})) return;
#endif
  Kokkos::parallel_for(
      name, norm_policy(N), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank() * kNormRows + t.team_rank();
        if (r >= N) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) { run_pro(pro, in, r, d); });
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
#if defined(PET_HIP_NORMS)
  if (hip_rows::launch(name, N, D, hip_rows::Bwd<Sink>{out_adj, in, weight, ln, sink})) return;
#endif
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
