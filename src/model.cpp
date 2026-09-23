#include "pet/model.hpp"
#include "pet/ozaki.hpp"

#include "pet/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace pet {

namespace {

using Kokkos::MDRangePolicy;
using Kokkos::Rank;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPol = Kokkos::TeamPolicy<ExecSpace>;
using TeamMem = TeamPol::member_type;

constexpr double RMSNORM_EPS = 1.1920928955078125e-07;  // float32 finfo eps
constexpr double LAYERNORM_EPS = 1e-5;
constexpr int VBINS = 256;  // virial-accumulator bins (cut atomic contention)


// ----------------------------------------------------------------------------
// forward helpers
// ----------------------------------------------------------------------------

// out(R,Dout) = in(R,Din) @ W(Dout,Din)^T + b(Dout). gemm_ozaki falls through
// to the vendor GEMM unless the Ozaki path is both asked for and available, and
// either way the bias rides in the GEMM (its epilogue, where the library has one).
// beta = 1 accumulates onto `out` instead.
void linear(View2D out, View2D in, WeightRef W, View1D b, Net beta = 0) {
  gemm_ozaki('N', 'T', (Net)1.0, in, W.v, beta, out, W.for_orientation(true), b);
}

// linear + SiLU, saving the pre-activation into `sav` when `save` (the analytic
// backward needs it); pass an empty View when !save.
void linear_silu(View2D out, View2D sav, View2D in, WeightRef W, View1D b, bool save) {
  linear(out, in, W, b);
  const int n = out.extent(0) * out.extent(1);
  View1D o1(out.data(), n), s1(save ? sav.data() : nullptr, save ? n : 0);
  Kokkos::parallel_for(
      "silu", RangePolicy(0, n), KOKKOS_LAMBDA(int i) {
        const Net z = o1(i);
        if (save) s1(i) = z;
        o1(i) = silud(z);
      });
}

void gather(View2D out, View2D table, IView1D idx) {
  const int R = out.extent(0), D = out.extent(1);
  Kokkos::parallel_for(
      "gather", RangePolicy(0, R * D), KOKKOS_LAMBDA(int i) {
        const int r = i / D, d = i % D;
        out(r, d) = table(idx(r), d);
      });
}

// Pure elementwise helpers run as flat 1D ranges over the (contiguous LayoutRight)
// data: consecutive index = consecutive memory = coalesced, full occupancy. This
// is notably faster than MDRange<Rank<2>> tiling for these memory-bound passes.
void add_inplace(View2D a, View2D b) {
  const int n = a.extent(0) * a.extent(1);
  View1D a1(a.data(), n), b1(b.data(), n);
  Kokkos::parallel_for("add", RangePolicy(0, n), KOKKOS_LAMBDA(int i) { a1(i) += b1(i); });
}

// RMSNorm and LayerNorm are one operation: LayerNorm centres (mu = mean) and
// adds a bias, RMSNorm has mu = 0 and none. One warp per row (a team of 1 x 32
// vector lanes): coalesced reads and a shuffle reduction. Computed in the network
// type, as torch does -- fp64 here ran the norms at a fraction of bandwidth on a
// consumer GPU -- and in two passes, so LayerNorm's variance never subtracts two
// large squares. The row is re-read from L1, not from memory.
constexpr int NORM_VEC = 32;
KOKKOS_INLINE_FUNCTION void row_stats(const TeamMem& team, const View2D& in, int r, bool ln,
                                      Net& mu, Net& inv) {
  const int D = in.extent(1);
  mu = Net(0);
  if (ln) {
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, D), [&](int d, Net& s) { s += in(r, d); }, mu);
    mu /= D;
  }
  Net v = Net(0);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(team, D), [&](int d, Net& s) { s += (in(r, d) - mu) * (in(r, d) - mu); }, v);
  inv = Net(1) / Kokkos::sqrt(v / D + Net(ln ? LAYERNORM_EPS : RMSNORM_EPS));
}

// out = (in - mu) * inv * weight (+ bias). An empty `bias` means RMSNorm.
void norm_fwd(View2D out, View2D in, View1D weight, View1D bias) {
  const bool ln = bias.extent(0) > 0;
  Kokkos::parallel_for(
      "norm", TeamPol(in.extent(0), 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        Net mu, inv;
        row_stats(team, in, r, ln, mu, inv);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, in.extent(1)), [&](int d) {
          out(r, d) = (in(r, d) - mu) * inv * weight(d) + (ln ? bias(d) : Net(0));
        });
      });
}

// SwiGLU FF: out = w_out( v * sigmoid(g) ), [v,g]=w_in(in).chunk(2). tmp holds w_in(in).
void feedforward_swiglu(Workspace& ws, const std::string& key, View2D out, View2D in,
                        WeightRef w_in, View1D b_in, WeightRef w_out, View1D b_out, View2D tmp,
                        Net beta) {
  const int R = in.extent(0);
  const int dff = tmp.extent(1) / 2;
  linear(tmp, in, w_in, b_in);
  Workspace::Scope scope(ws);
  View2D h = ws.tmp(R, dff);
  {
    Kokkos::parallel_for(
        "swiglu", RangePolicy(0, (R) * (dff)),
        KOKKOS_LAMBDA(int _i) { const int r = _i / (dff), c = _i % (dff); h(r, c) = tmp(r, c) * sigmoidd(tmp(r, dff + c)); });
  }
  linear(out, h, w_out, b_out, beta);
}

// Multi-head self-attention (online softmax; bias = log(cf_seq) per key). When
// `save` is set, the per-query (m, 1/l) are written to key:ml and `merged` (the
// per-query output = out_sq) is left in key:merged so the backward can reuse them
// instead of recomputing the forward softmax.
//
// Templated on the head dimension HD so the per-thread accumulator/query arrays
// are exactly head_dim wide and the inner loops unroll. With HD=0 (dynamic
// fallback) the arrays are a fixed 64 and the loop bound is runtime. This matters
// a LOT on CDNA2/MI250x: oversized [64] arrays (head_dim is only 16) blow the VGPR
// budget on the 8-waveslot SIMDs and SPILL to scratch (= HBM), so every acc[d] in
// the inner loop becomes a global round-trip -- the dominant cost there. The query
// is also hoisted into registers (was reread from qkv every key).
//
// THREAD ORDER. The (n, head, query) space is walked as a FLAT range with the
// query index fastest, not as an MDRangePolicy<Rank<3>>. MDRange takes its
// iteration pattern from the execution space's default array layout, which is
// LayoutLeft on CUDA/HIP -- so with bounds {N, heads, S} the LEFTMOST index (the
// structure/atom n) became the fastest-varying one, i.e. the 32 threads of a warp
// were 32 DIFFERENT atoms and shared nothing. Every thread then pulled its own
// key/value rows for the whole S-loop and the kernel ran at S x its necessary
// global traffic.
//
// With the query index fastest, a warp is 32 consecutive queries of the SAME
// (atom, head): all 32 lanes read the same qkv key/value element at the same
// instruction, which is a broadcast served once from L1. Nothing about the
// arithmetic or the key order changes, so the output is bit-identical -- this is
// purely where the threads sit. Measured on an RTX 4080 SUPER (microbenchmark at
// the production shapes): forward 2.5-3.2x at S=13/head_dim=16 (pet-mad-xs) and
// 5.5x at S=46/head_dim=32 (pbe0-pet).
//
// Padding each (atom, head) group up to a warp boundary so a warp can never
// straddle two groups was also tried; it is consistently SLOWER (3.9x vs 5.6x at
// S=46) -- the idle lanes cost more than the removed divergence saves.
// Largest head_dim the attention kernels accept.
//
// head_dim is d_pet / num_heads, so it tracks model width: 16 for pet-mad-xs,
// 32 for the -s models, 64 for -l, and 80 for pet-oam-xl / pet-omat-xl
// (d_pet 640 over 8 heads). It was capped at 64, which refused those two
// outright.
//
// The cost of raising it is register pressure, and it is worth writing down
// what that actually is -- measured with `cuobjdump -res-usage` on sm_89,
// registers and spilled bytes per thread:
//
//   head_dim  | forward         | backward dQ      | backward dK/dV
//   ----------+-----------------+------------------+------------------
//   16        | 64 reg,    0 B  | 95 reg,     0 B  | 122 reg,    0 B
//   32        | 95 reg,    0 B  | 168 reg,    0 B  | 218 reg,    0 B
//   48        | 128 reg,   0 B  | 255 reg,    0 B  | 255 reg,   56 B
//   64        | 168 reg,   0 B  | 255 reg,  160 B  | 255 reg,  408 B
//   80        | 254 reg,   0 B  | 255 reg,  392 B  | 255 reg,  728 B
//   generic   | 40 reg, 1024 B  | 40 reg,  1536 B  | 46 reg,  2048 B
//
// Three things follow.
//
// The FORWARD is fine everywhere -- even at 80 it lands on 254 of the 255
// available registers without spilling a byte. It is the backward that hurts:
// it holds four CAP-sized arrays (k, v, dk, dv) against the forward's two, and
// it saturates the register file from head_dim 48 upward, spilling to local
// memory beyond that.
//
// The GENERIC path is much the worst case. It sizes its arrays from kMaxHeadDim
// regardless of the actual head_dim, so it spills 1-2 KB per thread whatever it
// is run on. That is the whole reason the sizes below get their own
// instantiations: a specialization sets CAP == head_dim exactly, and the arrays
// are then no larger than the data they hold.
//
// And 255 registers per thread caps occupancy at ~8 warps of the 48 an SM can
// hold, so these kernels run near 17% occupancy at the large head_dims and are
// latency-bound rather than throughput-bound. Anything that adds per-thread
// state here -- compensated/double-double accumulation for accuracy, say -- has
// no room to do it in registers and would need a different structure
// (Kokkos scratch, or splitting the head dimension across a team) rather than
// another CAP-sized array.
constexpr int kMaxHeadDim = 128;

template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D attn_out, View2D qkv,
                    View2D cf_seq, WeightRef w_out, View1D b_out, int N, int S, int num_heads,
                    int head_dim, double temperature, bool save) {
  const int D = num_heads * head_dim, H = num_heads;
  // Saved for the backward only when `save`; otherwise one-layer scratch.
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", N * S, D) : ws.tmp(N * S, D);
  View2D sml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  const double scale = 1.0 / (Kokkos::sqrt((double) head_dim) * temperature);
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
  {
    Kokkos::parallel_for(
        "attention", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
          const int sq = i % S, hh = (i / S) % H, n = i / (S * H);
          const int nd = HD > 0 ? HD : head_dim;  // compile-time when HD>0
          const int row_q = n * S + sq;
          // Padding-query skip: token sq>0 whose cutoff factor is 0. For a real
          // padding slot this is exact -- the edge readout multiplies by
          // mask*cutoff, and no real edge's reverse index points at a padding slot,
          // so nothing downstream can observe the value. It is a large win whenever
          // the batch-max neighbour count exceeds the typical one.
          //
          // A *kept* edge can also reach factor 0, at d == pair_cutoff or within a
          // float ulp of it (dev_cutoff_value casts the scaled distance to float).
          // Such an edge is masked out of the readout as well, but its reverse index
          // is valid, so its partner reads this row through the featurizer's
          // reverse-edge concat. Skipping it therefore leaves the partner seeing
          // output_linear's bias instead of a true attention output. The forward and
          // the backward make the same choice, so forces stay consistent with the
          // energy; the difference is against a reference implementation, on a
          // measure-zero set of distances.
          if (sq > 0 && cf_seq(n, sq) <= Net(0)) {
            for (int d = 0; d < nd; ++d) merged(row_q, hh * head_dim + d) = Net(0);
            if (save) {
              sml((n * H + hh) * S + sq, 0) = -Net(1e30);
              sml((n * H + hh) * S + sq, 1) = Net(0);
            }
            return;
          }
          const int qoff = hh * head_dim, koff = D + hh * head_dim, voff = 2 * D + hh * head_dim;
          const Net sc = (Net) scale;
          Net q[CAP], acc[CAP];
          for (int d = 0; d < nd; ++d) { q[d] = qkv(row_q, qoff + d); acc[d] = Net(0); }
          Net m = -Net(1e30), l = Net(0);
          for (int sk = 0; sk < S; ++sk) {
            const Net cf = cf_seq(n, sk);
            if (cf <= Net(0)) continue;
            const int row_k = n * S + sk;
            Net dot = Net(0);
            for (int d = 0; d < nd; ++d) dot += q[d] * qkv(row_k, koff + d);
            const Net s = dot * sc + fast_log(cf);
            const Net new_m = (s > m) ? s : m;
            const Net c = fast_exp(m - new_m), p = fast_exp(s - new_m);
            l = l * c + p;
            for (int d = 0; d < nd; ++d) acc[d] = acc[d] * c + p * qkv(row_k, voff + d);
            m = new_m;
          }
          const Net invl = (l > Net(0)) ? Net(1) / l : Net(0);
          for (int d = 0; d < nd; ++d) merged(row_q, hh * head_dim + d) = acc[d] * invl;
          if (save) {
            sml((n * H + hh) * S + sq, 0) = m;
            sml((n * H + hh) * S + sq, 1) = invl;
          }
        });
  }
  linear(attn_out, merged, w_out, b_out);
}
// Dispatch on the runtime head_dim to a compile-time-sized instantiation. The
// listed sizes are the ones the published upet catalogue uses -- 16 (xs), 32
// (s), 48 (m, d_pet 384 over 8 heads), 64 (l), 80 (xl) -- and each avoids the
// oversized per-thread arrays the generic HD=0 path has to allocate. Anything
// else still runs, at kMaxHeadDim registers.
#define PET_ATTN_DISPATCH(FN, ...)                                  \
  switch (head_dim) {                                               \
    case 16: FN<16>(__VA_ARGS__); break;                            \
    case 32: FN<32>(__VA_ARGS__); break;                            \
    case 48: FN<48>(__VA_ARGS__); break;                            \
    case 64: FN<64>(__VA_ARGS__); break;                            \
    case 80: FN<80>(__VA_ARGS__); break;                            \
    default: FN<0>(__VA_ARGS__); break;                             \
  }
void attention(Workspace& ws, const std::string& key, View2D attn_out, View2D qkv, View2D cf_seq,
               WeightRef w_out, View1D b_out, int N, int S, int num_heads, int head_dim,
               double temperature, bool save) {
  PET_ATTN_DISPATCH(attention_impl, ws, key, attn_out, qkv, cf_seq, w_out, b_out, N, S, num_heads,
                    head_dim, temperature, save);
}

// ----------------------------------------------------------------------------
// backward helpers. Each takes `beta` (or `acc`): 1 accumulates into its output,
// 0 overwrites it -- so a first writer needs no zeroed buffer, and nothing
// depends on the workspace's zeroing policy.
// ----------------------------------------------------------------------------

// in_adj(R,Din) = beta * in_adj + out_adj(R,Dout) @ W(Dout,Din)
void linear_bwd(View2D in_adj, View2D out_adj, WeightRef W, Net beta = 1) {
  // The BACKWARD goes through the same path as the forward, which is the point
  // for anything that differentiates the forces again -- a Hessian, a phonon
  // calculation, a nested finite difference. An accurate forward with an fp32
  // backward would leave the second derivative limited by the backward.
  gemm_ozaki('N', 'N', (Net)1.0, out_adj, W.v, beta, in_adj, W.for_orientation(false));
}

// grad *= dsilu(pre)  (in place)
void silu_bwd(View2D grad, View2D pre) {
  const int n = grad.extent(0) * grad.extent(1);
  View1D g1(grad.data(), n), p1(pre.data(), n);
  Kokkos::parallel_for(
      "silu_bwd", RangePolicy(0, n), KOKKOS_LAMBDA(int i) {
        const Net x = p1(i);
        const Net sg = sigmoidd(x);  // expf, not double exp
        g1(i) *= sg * (Net(1) + x * (Net(1) - sg));
      });
}

// out = silu(pre), pre = [input_edge wi^T] + wx [v, |v|] + b + tab[species_j],
// and sav <- pre when `save`: compress.0 through its load-time fold (CompressFold).
void compress_fwd(View2D out, View2D sav, bool save, const CompressFold& f, View2D input_edge,
                  IView1D nsp, RView2D ev, RView1D dist) {
  const bool acc = f.wi.extent(0) > 0;
  if (acc) gemm_ozaki('N', 'T', (Net)1.0, input_edge, f.wi.v, (Net)0.0, out, f.wi.for_orientation(true));
  const int D = out.extent(1);
  auto wx = f.wx; auto b = f.b; auto tab = f.tab;
  Kokkos::parallel_for(
      "compress", RangePolicy(0, out.extent(0) * D), KOKKOS_LAMBDA(int i) {
        const int k = i / D, d = i % D;
        const Net z = (acc ? out(k, d) : Net(0)) + b(d) + tab(nsp(k), d) + wx(d, 0) * (Net) ev(k, 0) +
                      wx(d, 1) * (Net) ev(k, 1) + wx(d, 2) * (Net) ev(k, 2) + wx(d, 3) * (Net) dist(k);
        if (save) sav(k, d) = z;
        out(k, d) = silud(z);
      });
}

// Backward of compress_fwd. g arrives as the adjoint of its output and is turned,
// in place, into the adjoint of pre; the geometry adjoint accumulates into
// x4_adj[k, 0..3] (one warp per row) and, past layer 0, input_edge's into ie_adj.
// Nothing is computed for the species blocks: they are constants.
void compress_bwd(View2D g, View2D pre, const CompressFold& f, View2D x4_adj, View2D ie_adj) {
  const int D = g.extent(1);
  auto wx = f.wx;
  Kokkos::parallel_for(
      "compress_bwd", TeamPol(g.extent(0), 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        Net a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, Net& s0, Net& s1, Net& s2, Net& s3) {
              const Net x = pre(r, d), sg = sigmoidd(x);
              const Net gd = g(r, d) * sg * (Net(1) + x * (Net(1) - sg));
              g(r, d) = gd;
              s0 += gd * wx(d, 0), s1 += gd * wx(d, 1), s2 += gd * wx(d, 2), s3 += gd * wx(d, 3);
            },
            a0, a1, a2, a3);
        Kokkos::single(Kokkos::PerThread(team), [&] {
          x4_adj(r, 0) += a0, x4_adj(r, 1) += a1, x4_adj(r, 2) += a2, x4_adj(r, 3) += a3;
        });
      });
  if (f.wi.extent(0) > 0) linear_bwd(ie_adj, g, f.wi);  // accumulates
}

// in_adj += d(norm)/d(in) . out_adj. With xhat = (in - mu) * inv and gw = out_adj * weight:
//   in_adj += inv * (gw - [mean(gw) if LayerNorm] - xhat * mean(gw * xhat)).
void norm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight, bool ln, bool acc = true) {
  const int D = in.extent(1);
  Kokkos::parallel_for(
      "norm_bwd", TeamPol(in.extent(0), 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        Net mu, inv, sg = Net(0), sgx = Net(0);
        row_stats(team, in, r, ln, mu, inv);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, Net& a, Net& b) {
              const Net gw = out_adj(r, d) * weight(d);
              a += gw;
              b += gw * (in(r, d) - mu) * inv;
            },
            sg, sgx);
        const Net mg = ln ? sg / D : Net(0), mgx = sgx / D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, D), [&](int d) {
          in_adj(r, d) = (acc ? in_adj(r, d) : Net(0)) +
                         inv * (out_adj(r, d) * weight(d) - mg - (in(r, d) - mu) * inv * mgx);
        });
      });
}

// in_adj(R,Dmodel) += backward of SwiGLU FF. tmp holds w_in(in) from forward.
void feedforward_swiglu_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D out_adj,
                            View2D tmp, WeightRef w_in, WeightRef w_out, Net beta) {
  const int R = out_adj.extent(0);
  const int dff = tmp.extent(1) / 2;
  Workspace::Scope scope(ws);
  View2D h_adj = ws.tmp(R, dff);
  linear_bwd(h_adj, out_adj, w_out, 0);  // h_adj = out_adj @ w_out
  View2D tmp_adj = ws.tmp(R, 2 * dff);
  {
    Kokkos::parallel_for(
        "swiglu_bwd", RangePolicy(0, (R) * (dff)),
        KOKKOS_LAMBDA(int _i) { const int r = _i / (dff), c = _i % (dff);
          const Net v = tmp(r, c), sg = sigmoidd(tmp(r, dff + c));
          tmp_adj(r, c) = h_adj(r, c) * sg;
          tmp_adj(r, dff + c) = h_adj(r, c) * v * sg * (Net(1) - sg);
        });
  }
  linear_bwd(in_adj, tmp_adj, w_in, beta);
}

// Backward of attention. an_adj(N*S,D) += ; cf_seq_adj(N,S) += (atomic, heads-way).
//
// FlashAttention-style atomic-free design: instead of one query-parallel kernel
// that atomic-scatters dK/dV to key rows (S-way contended) and cf_seq_adj (heads*S-
// way), split into kernels that each OWN their output rows:
//   A (per query sq): recompute the online softmax, save per-query stats
//      (m, 1/l, dot_do_out), and write dQ (own row, no atomic).
//   B (per key sk):   loop queries using the saved stats, write dK/dV (own row,
//      no atomic), and leave its per-head cutoff-adjoint partial in cfh.
//   C (per key sk):   sum cfh over the heads IN INDEX ORDER into cf_seq_adj.
// dQ and dK/dV occupy disjoint columns of qkv_adj, so the kernels never race.
//
// Both A and B are flat ranges with the SEQUENCE index fastest, for the reason
// spelled out over attention_impl: MDRangePolicy on CUDA/HIP makes the leftmost
// bound the fastest-varying one, which put 32 unrelated atoms in a warp.
//
// The head index is a parallel dimension of B, not a loop inside it. It used to be
// a loop because the cutoff-factor adjoint has to be summed over heads in a fixed
// order (a float atomic there made two identical structures in one batch disagree,
// see below) -- but that serialised 8 heads into one thread and left kernel B with
// only N*S threads, an 8x parallelism deficit on top of ~220 registers/thread of
// live dk/dv/k/v arrays. Splitting the heads out and giving each its own cfh slot
// keeps the ordered sum -- kernel C walks hh = 0..H-1 in one thread, exactly the
// order the loop used -- while giving B its 8x. Measured 6.5-9x on the backward at
// S=46/head_dim=32 and 2.4-2.9x at S=13/head_dim=16, bit-identical either way.
//
// A team-per-(atom,head) variant staging K/V/Q/dO in shared memory was also
// measured: ~20% on top of this for pbe0-pet, nothing for pet-mad-xs (at S=13 a
// team wastes most of its threads), and it needs a scratch-size fallback for large
// S. Not worth the second code path.
template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, const std::string& fwd_key,
                        View2D an_adj, View2D cf_seq_adj, View2D ao_adj, View2D qkv, View2D cf_seq,
                        WeightRef w_in, WeightRef w_out, int N, int S, int num_heads, int head_dim,
                        double temperature, Net beta) {
  const int D = num_heads * head_dim;
  const int R = N * S;
  const int H = num_heads;
  Workspace::Scope scope(ws);
  View2D merged_adj = ws.tmp(R, D);
  linear_bwd(merged_adj, ao_adj, w_out, 0);  // d(merged) = ao_adj @ w_out
  View2D qkv_adj = ws.tmp(R, 3 * D);  // each element written by exactly one thread
  // softmax math in Net precision (expf/logf): much faster than fp64 on consumer GPUs.
  View2D stats = ws.tmp(N * H * S, 3);  // per-query (m, 1/l, dot_do_out)
  View2D cfh = ws.tmp(N * S, H);          // per-(key, head) cutoff-adjoint partial
  View2D merged_saved = ws.peek2(fwd_key + ":merged");  // forward out_sq (no recompute)
  View2D sml = ws.peek2(fwd_key + ":ml");               // forward per-query (m, 1/l)
  const Net sc = (Net) (1.0 / (Kokkos::sqrt((double) head_dim) * temperature));
  // Exact size when the dispatch matched, so the arrays are no bigger than the
  // data; the generic fallback pays kMaxHeadDim registers either way.
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;

  {
    // Kernel A: per query -> stats + dQ (atomic-free; own row). Uses the saved
    // forward (m, 1/l, out_sq) instead of recomputing the online softmax.
    Kokkos::parallel_for(
        "attn_bwd_dQ", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
          const int sq = i % S, hh = (i / S) % H, n = i / (S * H);
          const int nd = HD > 0 ? HD : head_dim;
          const int row_q = n * S + sq;
          const int qoff = hh * head_dim, koff = D + hh * head_dim, voff = 2 * D + hh * head_dim;
          const int moff = hh * head_dim;
          const int si = (n * H + hh) * S + sq;
          const Net m = sml(si, 0), invl = sml(si, 1);
          // Padding query (invl==0 from the forward): zero dQ and its stats, and
          // kernel B reads invl==0 and skips it too.
          if (invl <= Net(0)) {
            for (int d = 0; d < nd; ++d) qkv_adj(row_q, qoff + d) = Net(0);
            stats(si, 0) = stats(si, 1) = stats(si, 2) = Net(0);
            return;
          }
          Net q[CAP], dout[CAP], dq[CAP];
          Net dot_do_out = Net(0);  // sum dout * out_sq (out_sq = saved merged)
          for (int d = 0; d < nd; ++d) {
            q[d] = qkv(row_q, qoff + d);
            dout[d] = merged_adj(row_q, moff + d);
            dot_do_out += dout[d] * merged_saved(row_q, moff + d);
            dq[d] = Net(0);
          }
          stats(si, 0) = m;
          stats(si, 1) = invl;
          stats(si, 2) = dot_do_out;
          for (int sk = 0; sk < S; ++sk) {
            const Net cf = cf_seq(n, sk);
            if (cf <= Net(0)) continue;
            const int row_k = n * S + sk;
            Net dot = Net(0), dA = Net(0);
            for (int d = 0; d < nd; ++d) {
              dot += q[d] * qkv(row_k, koff + d);
              dA += dout[d] * qkv(row_k, voff + d);
            }
            const Net A = fast_exp(dot * sc + fast_log(cf) - m) * invl;
            const Net dscore = A * (dA - dot_do_out);
            for (int d = 0; d < nd; ++d) dq[d] += dscore * sc * qkv(row_k, koff + d);
          }
          for (int d = 0; d < nd; ++d) qkv_adj(row_q, qoff + d) = dq[d];
        });
    // Kernel B: per (key, head) -> dK, dV (own row, no sharing) + this head's
    // cutoff-adjoint partial. Every element of cfh is written here (0 for padding
    // keys), so kernel C never reads a stale slot.
    Kokkos::parallel_for(
        "attn_bwd_dKV", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
          const int sk = i % S, hh = (i / S) % H, n = i / (S * H);
          const int nd = HD > 0 ? HD : head_dim;
          const int row_k = n * S + sk;
          const Net cf = cf_seq(n, sk);
          const int qoff = hh * head_dim, koff = D + hh * head_dim, voff = 2 * D + hh * head_dim;
          // padding key: no query attends -> dK = dV = 0
          if (cf <= Net(0)) {
            for (int d = 0; d < nd; ++d) qkv_adj(row_k, koff + d) = qkv_adj(row_k, voff + d) = Net(0);
            cfh(row_k, hh) = Net(0);
            return;
          }
          const Net logcf = fast_log(cf);
          const int moff = hh * head_dim;
          // k/v are loop-invariant over the queries: hoist them out of the sq loop.
          Net dk[CAP], dv[CAP], k[CAP], v[CAP];
          for (int d = 0; d < nd; ++d) {
            dk[d] = Net(0);
            dv[d] = Net(0);
            k[d] = qkv(row_k, koff + d);
            v[d] = qkv(row_k, voff + d);
          }
          Net cf_acc = Net(0);
          for (int sq = 0; sq < S; ++sq) {
            const int row_q = n * S + sq;
            const int si = (n * H + hh) * S + sq;
            const Net m = stats(si, 0), invl = stats(si, 1), dot_do_out = stats(si, 2);
            if (invl <= Net(0)) continue;  // padding query contributes nothing to dK/dV
            Net dot = Net(0), dA = Net(0);
            for (int d = 0; d < nd; ++d) {
              dot += qkv(row_q, qoff + d) * k[d];
              dA += merged_adj(row_q, moff + d) * v[d];
            }
            const Net A = fast_exp(dot * sc + logcf - m) * invl;
            const Net dscore = A * (dA - dot_do_out);
            for (int d = 0; d < nd; ++d) {
              dk[d] += dscore * sc * qkv(row_q, qoff + d);
              dv[d] += A * merged_adj(row_q, moff + d);
            }
            cf_acc += dscore;
          }
          for (int d = 0; d < nd; ++d) {
            qkv_adj(row_k, koff + d) = dk[d];
            qkv_adj(row_k, voff + d) = dv[d];
          }
          cfh(row_k, hh) = cf_acc;
        });
    // Kernel C: the heads-way sum that used to be the outer loop of B, in the same
    // index order and inside one thread -- no atomic, no arrival-order dependence.
    Kokkos::parallel_for(
        "attn_bwd_cf", RangePolicy(0, N * S), KOKKOS_LAMBDA(int i) {
          const int sk = i % S, n = i / S;
          const Net cf = cf_seq(n, sk);
          if (cf <= Net(0)) return;
          Net cf_total = Net(0);
          for (int hh = 0; hh < H; ++hh) cf_total += cfh(n * S + sk, hh);
          cf_seq_adj(n, sk) += cf_total / cf;
        });
  }
  linear_bwd(an_adj, qkv_adj, w_in, beta);  // d(attn_in) = qkv_adj @ w_in
}
void attention_bwd(Workspace& ws, const std::string& key, const std::string& fwd_key, View2D an_adj,
                   View2D cf_seq_adj, View2D ao_adj, View2D qkv, View2D cf_seq, WeightRef w_in,
                   WeightRef w_out, int N, int S, int num_heads, int head_dim, double temperature,
                   Net beta = 1) {
  PET_ATTN_DISPATCH(attention_bwd_impl, ws, key, fwd_key, an_adj, cf_seq_adj, ao_adj, qkv, cf_seq,
                    w_in, w_out, N, S, num_heads, head_dim, temperature, beta);
}

// Bump cutoff derivative d f / d distance. d f / d rc = -(this). Transcendentals
// in float to match cutoff_val / dev_bump_cutoff (fast on consumer GPUs).
KOKKOS_INLINE_FUNCTION double bump_ddist(double d, double rc, double w) {
  const float scaled = (float) ((d - (rc - w)) / w);
  if (scaled <= 0.0f || scaled >= 1.0f) return 0.0;
  const float ps = (float) M_PI * scaled;
  const float si = Kokkos::sin(ps);
  const float u = Kokkos::cos(ps) / si;  // cot
  const float t = Kokkos::tanh(u);
  const float dudd = (-(float) M_PI / (si * si)) / (float) w;
  return 0.5 * (1.0 - (double) (t * t)) * (double) dudd;
}
KOKKOS_INLINE_FUNCTION double cosine_ddist(double d, double rc, double w) {
  const float scaled = (float) ((d - (rc - w)) / w);
  if (scaled <= 0.0f || scaled >= 1.0f) return 0.0;
  return 0.5 * (double) (-(float) M_PI * Kokkos::sin((float) M_PI * scaled)) / w;
}

// Cutoff function value (device). Same float formula as dev_bump_cutoff so the
// adaptive forward (nef builder) and this backward recompute stay consistent.
KOKKOS_INLINE_FUNCTION double cutoff_val(double d, double rc, double w, bool bump) {
  const float scaled = (float) ((d - (rc - w)) / w);
  if (scaled <= 0.0f) return 1.0;
  if (scaled >= 1.0f) return 0.0;
  if (bump)
    return 0.5 * (1.0 + (double) Kokkos::tanh(1.0f / Kokkos::tan((float) M_PI * scaled)));
  return 0.5 * (1.0 + (double) Kokkos::cos((float) M_PI * scaled));
}
KOKKOS_INLINE_FUNCTION double cutoff_ddist(double d, double rc, double w, bool bump) {
  return bump ? bump_ddist(d, rc, w) : cosine_ddist(d, rc, w);
}

// Backward of the adaptive cutoff: given adapted_adj[a] = dE/d(adapted_cutoff[a]),
// recompute the adaptive forward and propagate to every raw edge distance, then
// scatter per-edge gradients into d_forces (scaled by `scale`). See
// metatrain adaptive_cutoff.py for the forward this differentiates.
// Fold per-edge gradients into per-atom forces and a per-structure virial.
//
// Shared by both featurizers: once you have edge_grad the geometry backward is the
// same operation, and this is where the determinism work lives, so having two
// copies meant one of them was always the nondeterministic one.
//
// A GATHER, one thread per atom, not a scatter. The scatter had each edge
// k = (i, j) do atomic_add(+g) on i and atomic_add(-g) on j, and float atomics
// accumulate in thread-arrival order -- so forces differed in their last bits
// between two identical runs, which a relaxer amplifies into multi-kJ/mol
// differences in the relaxed energy and a reshuffled ranking (a relaxation is a
// chaotic map; see README.md on determinism).
//
// The edge list is FULL DIRECTED -- every (i, j, shift) has its (j, i, -shift)
// partner -- so the edges for which atom a is the NEIGHBOUR are exactly the
// reverses of a's own edges. That turns the scatter into
//
//     F(a) = sum over a's own slots k of [ g(k) - g(reverse(k)) ]
//
// which each thread evaluates alone, in slot order, with no atomics. The reverse
// map already exists; it is built for message passing.
//
// The virial accumulates per atom for the same reason, then sums over each
// structure's contiguous atom range in index order.
void fold_edge_gradients(Workspace& ws, const std::string& key, RView2D edge_grad,
                         const PackedEdges& pk, RView2D d_forces, RView2D dvir, IView1D sid,
                         int N, int NS, double scale) {
  RView2D vir_atom = ws.r2(key + ":vir_atom", N, 9);
  auto off = pk.off, rev = pk.reverse;
  auto vec = pk.vec;
  Kokkos::parallel_for(
      key + ":gather_force", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
        double f[3] = {0.0, 0.0, 0.0};
        double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = off(i); k < off(i + 1); ++k) {
          for (int c = 0; c < 3; ++c) f[c] += edge_grad(k, c);
          for (int a = 0; a < 3; ++a)  // this edge's own virial share v_a * g_b
            for (int b = 0; b < 3; ++b) w[a * 3 + b] += vec(k, a) * edge_grad(k, b);
          const int r = rev(k);
          if (r >= 0)
            for (int c = 0; c < 3; ++c) f[c] -= edge_grad(r, c);
        }
        for (int c = 0; c < 3; ++c) d_forces(i, c) = scale * f[c];  // the first writer
        for (int t = 0; t < 9; ++t) vir_atom(i, t) = scale * w[t];
      });
  // Atoms of a structure are contiguous in the batch, so a count plus a prefix sum
  // gives each structure its range and the sum runs in atom-index order. Integer
  // counts are order-independent in value, so the count itself is safe.
  // Explicitly zeroed rather than trusting the pool's policy -- see the note in
  // energy_forces_batch. This one is only reached from the backward, where the
  // policy is on, but that is a property of the caller and not of this code.
  IView1D scnt = ws.i1(key + ":vir_scnt", NS);
  Kokkos::deep_copy(ExecSpace(), scnt, 0);
  IView1D soff = ws.i1(key + ":vir_soff", NS + 1);
  if (NS > 1) {
    Kokkos::parallel_for(
        key + ":vir_count", RangePolicy(0, N),
        KOKKOS_LAMBDA(int i) { Kokkos::atomic_inc(&scnt(sid(i))); });
    Kokkos::parallel_scan(
        key + ":vir_scan", RangePolicy(0, NS), KOKKOS_LAMBDA(int b, int& upd, bool final) {
          if (final) soff(b) = upd;
          upd += scnt(b);
          if (final && b == NS - 1) soff(NS) = upd;
        });
  } else {
    Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 0), 0);
    Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 1), N);
  }
  Kokkos::parallel_for(
      key + ":vir_reduce", RangePolicy(0, NS * 9), KOKKOS_LAMBDA(int _i) {
        const int b = _i / 9, t = _i % 9;
        double s = 0.0;
        for (int i = soff(b); i < soff(b + 1); ++i) s += vir_atom(i, t);
        dvir(b, t) = s;  // the first writer
      });
}

// per_atom_net(n) += node_pred(n) + sum over n's edges of cutoff*edge_pred.
// Free-standing because nvcc will not take an extended lambda inside a private
// member, and PetModel::readout is one.
void readout_accumulate(View1D per_atom_net, View2D node_pred, View2D edge_pred, IView1D off,
                        View1D cutoff, int N, bool zero_first) {
  Kokkos::parallel_for(
      "readout", RangePolicy(0, N), KOKKOS_LAMBDA(int n) {
        // Accumulated in double, not in the network type. Summing M edge terms in
        // fp32 costs about 1e-8 relative on the per-atom energy -- small, but the
        // two paths disagreed on it (the feedforward one used double, the residual
        // one did not), and this is the more accurate of the two.
        double e = node_pred(n, 0);
        for (int k = off(n); k < off(n + 1); ++k) e += (double) cutoff(k) * (double) edge_pred(k, 0);
        per_atom_net(n) = zero_first ? Net(e) : Net(per_atom_net(n) + e);
      });
}

// Backward of the adaptive cutoff. Both schemes reduce to ONE per-edge scalar --
// d(adapted_cutoff[centre]) / d(dist_e), times the incoming adjoint -- after
// which the gather into forces and the per-structure virial are identical. So
// the methods differ only in how `gmag` is filled, and everything below that is
// written once. (Having two copies of the gather is how one of them ends up
// being the non-deterministic one.)
//
// `solver_r` / `solver_dn` are the saved root and slope from the solver forward;
// they are empty for the grid method. A zero in solver_dn marks an atom whose
// cutoff hit a clamp bound, and therefore has no gradient.
void adaptive_backward(Workspace& ws, RView2D d_forces, RView2D dvir, RView1D adapted_adj,
                       IView1D raw_center, IView1D raw_neigh, RView1D raw_dist, RView2D raw_vec,
                       int E, RView1D probes, int P, double width, double target, double scale,
                       int N, bool bump, IView1D struct_id, int n_struct,
                       IView1D raw_off, IView1D raw_reverse,
                       bool solver, RView1D solver_r, RView1D solver_dn,
                       RView2D eff_saved = RView2D()) {
  if (E == 0) return;
  if (!solver && P < 2) return;
  if (solver && ((int) solver_r.extent(0) != N || (int) solver_dn.extent(0) != N))
    throw std::runtime_error(
        "pet: the solver adaptive backward needs the forward's saved root and slope "
        "(DeviceEdgeData::adapt_r / adapt_dn). A caller supplying its own edge data "
        "must fill them -- they cannot be recovered here, because this path has no "
        "per-atom segmentation of the raw edge list to recompute them with.");
  // The adaptive scheme always tapers with the BUMP function, whatever the
  // model's own cutoff_function is -- metatrain's adaptive_cutoff.py imports
  // cutoff_func_bump unconditionally. The forwards (host and device) hardcode
  // bump accordingly; passing the model's flag down here instead would
  // differentiate a cosine while the forward summed bumps, for any Cosine model
  // that also uses an adaptive cutoff.
  (void) bump;
  constexpr bool kAdaptiveIsBump = true;
  // --- per-edge d(adapted_cutoff[centre]) / d(dist), by method -----------------
  RView1D gmag = ws.r1("ad:gmag", E);

  if (solver) {
    // Implicit-function theorem. The forward's last act was
    //     adapted = r - (n_total(r) - target) / dn_root,
    // with r and dn_root constants, so only the residual carries a dependence on
    // the distances:
    //     d(adapted)/d(d_j) = -(1/dn_root) * d(n_total)/d(d_j)
    //                       = -(1/dn_root) * df/dd_j
    //                       = +(1/dn_root) * df/dr_j        (since f depends on d - r)
    // and df/dr is exactly -bump_ddist, which is why no new derivative is needed
    // here. dn_root arrives zeroed for any atom whose cutoff was clamped, which
    // makes that atom's gradient vanish without a second branch.
    //
    // This is the whole reason the solver forward takes a trailing IFT step it
    // does not numerically need: it puts a differentiable expression at the root
    // so the backward never has to unroll ten Newton iterations.
    Kokkos::parallel_for(
        "ad_gmag_solver", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const int c = raw_center(e);
          const double d = raw_dist(e);
          const double dn = solver_dn(c);
          if (d <= 0.0 || dn <= 0.0) { gmag(e) = 0.0; return; }
          const double df_dr = -bump_ddist(d, solver_r(c), width);
          gmag(e) = scale * adapted_adj(c) * (df_dr / dn) / d;
        });
  } else {
    // Reuse the effective-neighbor-count grid from the on-device forward build when
    // it was provided (the same quantity the backward would recompute in K1).
    const bool have_eff = (eff_saved.extent(0) == (size_t) N && eff_saved.extent(1) == (size_t) P);
    RView2D eff = have_eff ? eff_saved : ws.r2("ad:eff", N, P);
    if (!have_eff) Kokkos::deep_copy(ExecSpace(), eff, 0.0);  // atomic accumulator
    RView2D diff = ws.r2("ad:diff", N, P), grad = ws.r2("ad:grad", N, P),
            gsign = ws.r2("ad:gsign", N, P), w = ws.r2("ad:w", N, P), Cq = ws.r2("ad:Cq", N, P),
            diffadj = ws.r2("ad:diffadj", N, P);
    RView1D adapted = ws.r1("ad:adapted", N);

    // K1: effective neighbor counts eff[center,p] = sum_edges bump(dist, probes[p], width).
    // Skipped when reusing the forward build's eff -- an O(E*P) transcendental+atomic pass.
    if (!have_eff)
      Kokkos::parallel_for(
          "ad_eff", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
            const int c = raw_center(e);
            const double d = raw_dist(e);
            for (int p = 0; p < P; ++p)
              Kokkos::atomic_add(&eff(c, p), cutoff_val(d, probes(p), width, bump));
          });
    // K2: diff = eff - target + target*x^3
    Kokkos::parallel_for(
        "ad_diff", RangePolicy(0, (N) * (P)),
        KOKKOS_LAMBDA(int _i) { const int a = _i / (P), p = _i % (P);
          const double x = (double)p / (P - 1);
          diff(a, p) = eff(a, p) - target + target * x * x * x;
        });
    // K3: grad = |torch.gradient(diff)| clamped ; gsign = d|.|/dG (0 if clamped)
    Kokkos::parallel_for(
        "ad_grad", RangePolicy(0, (N) * (P)),
        KOKKOS_LAMBDA(int _i) { const int a = _i / (P), p = _i % (P);
          double g;
          if (p == 0) g = diff(a, 1) - diff(a, 0);
          else if (p == P - 1) g = diff(a, P - 1) - diff(a, P - 2);
          else g = 0.5 * (diff(a, p + 1) - diff(a, p - 1));
          const double ag = Kokkos::fabs(g);
          grad(a, p) = (ag > 1e-12) ? ag : 1e-12;
          gsign(a, p) = (ag >= 1e-12) ? ((g > 0.0) ? 1.0 : (g < 0.0 ? -1.0 : 0.0)) : 0.0;
        });
    // K4: w = softmax_p(logw), logw=-0.5*(diff/grad)^2 ; adapted = sum probes*w
    Kokkos::parallel_for(
        "ad_w", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
          double mx = -1e300;
          for (int p = 0; p < P; ++p) {
            const double lw = -0.5 * (diff(a, p) / grad(a, p)) * (diff(a, p) / grad(a, p));
            w(a, p) = lw;
            if (lw > mx) mx = lw;
          }
          double s = 0.0;
          for (int p = 0; p < P; ++p) {
            const double e = Kokkos::exp(w(a, p) - mx);
            w(a, p) = e;
            s += e;
          }
          double ad = 0.0;
          for (int p = 0; p < P; ++p) {
            w(a, p) /= s;
            ad += probes(p) * w(a, p);
          }
          adapted(a) = ad;
        });
    // K5: logw adjoint -> direct diff term + Cq (for the grad-stencil chain)
    Kokkos::parallel_for(
        "ad_C", RangePolicy(0, (N) * (P)),
        KOKKOS_LAMBDA(int _i) { const int a = _i / (P), p = _i % (P);
          const double A = adapted_adj(a) * w(a, p) * (probes(p) - adapted(a));
          const double gd = grad(a, p);
          diffadj(a, p) = A * (-(diff(a, p) / (gd * gd)));      // direct d logw/d diff
          Cq(a, p) = A * (diff(a, p) * diff(a, p) / (gd * gd * gd)) * gsign(a, p);
        });
    // K6: add adjoint of torch.gradient operator (B_r = sum_q C_q dG_q/ddiff_r)
    Kokkos::parallel_for(
        "ad_B", RangePolicy(0, (N) * (P)),
        KOKKOS_LAMBDA(int _i) { const int a = _i / (P), r = _i % (P);
          double B = 0.0;
          if (r - 1 >= 1 && r - 1 <= P - 2) B += 0.5 * Cq(a, r - 1);
          if (r + 1 >= 1 && r + 1 <= P - 2) B += -0.5 * Cq(a, r + 1);
          if (r == 1) B += Cq(a, 0);
          if (r == 0) B += -Cq(a, 0);
          if (r == P - 1) B += Cq(a, P - 1);
          if (r == P - 2) B += -Cq(a, P - 1);
          diffadj(a, r) += B;
        });
    // Grid: the adjoint of the probe-weighted average, accumulated over probes.
    Kokkos::parallel_for(
        "ad_gmag_grid", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const double d = raw_dist(e);
          if (d <= 0.0) { gmag(e) = 0.0; return; }
          double da = 0.0;
          for (int p = 0; p < P; ++p)
            da += diffadj(raw_center(e), p) * cutoff_ddist(d, probes(p), width, kAdaptiveIsBump);
          gmag(e) = scale * da / d;
        });
  }
  // K7: scatter per-raw-edge distance gradient to atoms. Virial goes to one of
  // VBINS bins (~VBINS-fold less contention than every edge hitting 9 globals,
  // which is E-way contended here), reduced below.
  // The per-atom gather needs the raw list segmented by centre and each edge's
  // partner. The device neighbour build supplies both; a caller that does not (an
  // edge list marshalled from elsewhere, where the ordering is not guaranteed)
  // falls back to the original atomic scatter, which is correct but not
  // reproducible run to run.
  const bool can_gather = (raw_off.extent(0) == (size_t)(N + 1) &&
                           raw_reverse.extent(0) == (size_t) E);
  if (!can_gather) {
    RView2D dvir_bins = ws.r2("ad:virbins", VBINS, 9);
    Kokkos::deep_copy(ExecSpace(), dvir_bins, 0.0);  // atomic accumulator
    Kokkos::parallel_for(
        "ad_scatter", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const int c = raw_center(e), j = raw_neigh(e);
          const double ge = gmag(e);
          double gg[3];
          for (int cc = 0; cc < 3; ++cc) {
            gg[cc] = ge * raw_vec(e, cc);
            Kokkos::atomic_add(&d_forces(c, cc), gg[cc]);
            Kokkos::atomic_add(&d_forces(j, cc), -gg[cc]);
          }
          if (n_struct > 1) {
            const int vs = struct_id(c);
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b)
                Kokkos::atomic_add(&dvir(vs, a * 3 + b), raw_vec(e, a) * gg[b]);
          } else {
            const int bin = e & (VBINS - 1);
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b)
                Kokkos::atomic_add(&dvir_bins(bin, a * 3 + b), raw_vec(e, a) * gg[b]);
          }
        });
    if (n_struct <= 1)
      Kokkos::parallel_for(
          "ad_virreduce_bins", RangePolicy(0, 9), KOKKOS_LAMBDA(int t) {
            double s = 0.0;
            for (int bn = 0; bn < VBINS; ++bn) s += dvir_bins(bn, t);
            dvir(0, t) += s;
          });
    return;
  }

  // gmag was computed above, once, for whichever method is in play -- the gather
  // needs to read any edge's contribution including its partner's, so it has to
  // exist per edge rather than be recomputed inside the loop.
  // Force gather, one thread per atom: atom a takes +g from each of its own edges
  // and -g from each edge pointing at it, and those are exactly the partners of its
  // own edges (the list is full directed). Was two float atomic_adds per edge.
  RView2D vir_atom = ws.r2("ad:viratom", N, 9);
  Kokkos::parallel_for(
      "ad_gather", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        double f[3] = {0.0, 0.0, 0.0};
        double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int e = raw_off(a); e < raw_off(a + 1); ++e) {
          const double ge = gmag(e);
          for (int cc = 0; cc < 3; ++cc) f[cc] += ge * raw_vec(e, cc);
          // this edge's own virial share v_x * g_y
          for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y) w[x * 3 + y] += raw_vec(e, x) * ge * raw_vec(e, y);
          const int r = raw_reverse(e);
          if (r >= 0) {
            const double gr = gmag(r);
            // partner's vector is the negative of this one, and its contribution
            // arrives with a minus sign, so the two negations cancel
            for (int cc = 0; cc < 3; ++cc) f[cc] += gr * raw_vec(e, cc);
          }
        }
        for (int cc = 0; cc < 3; ++cc) d_forces(a, cc) += f[cc];
        for (int t = 0; t < 9; ++t) vir_atom(a, t) = w[t];
      });
  // Per-structure virial, summed over each structure's contiguous atom range.
  {
    IView1D scnt = ws.i1("ad:vscnt", n_struct);
    Kokkos::deep_copy(ExecSpace(), scnt, 0);  // atomic accumulator; see above
    IView1D soff = ws.i1("ad:vsoff", n_struct + 1);
    if (n_struct > 1) {
      Kokkos::parallel_for(
          "ad_vir_count", RangePolicy(0, N),
          KOKKOS_LAMBDA(int a) { Kokkos::atomic_inc(&scnt(struct_id(a))); });
      Kokkos::parallel_scan(
          "ad_vir_scan", RangePolicy(0, n_struct), KOKKOS_LAMBDA(int b, int& upd, bool final) {
            if (final) soff(b) = upd;
            upd += scnt(b);
            if (final && b == n_struct - 1) soff(n_struct) = upd;
          });
    } else {
      Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 0), 0);
      Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 1), N);
    }
    Kokkos::parallel_for(
        "ad_virreduce", RangePolicy(0, n_struct * 9), KOKKOS_LAMBDA(int _i) {
          const int b = _i / 9, t = _i % 9;
          double s = 0.0;
          for (int a = soff(b); a < soff(b + 1); ++a) s += vir_atom(a, t);
          dvir(b, t) += s;
        });
  }
}

}  // namespace

// Charge / spin conditioning: per-system charge and spin multiplicity embedded,
// projected, and broadcast to the atoms of that system. The result is ADDED to
// the node features after every GNN layer (metatrain does this in both
// featurizers, identically).
//
// Forward only. It is a per-system constant with no dependence on any position,
// so no gradient flows into it and the geometry backward never sees it. That is
// why there is no conditioning_bwd.
//
// Returns an empty View when the model is not conditioned, which the callers
// test rather than branching on the hyper.
View2D PetModel::conditioning(const DeviceEdgeData& dev, int N, int NS) {
  if (!h_.system_conditioning) return View2D();
  const int Dn = h_.d_node;
  const int max_q = h_.max_charge;

  // Defaults when the caller supplied no electronic state: neutral singlet,
  // which is metatrain's own fallback for a system that carries none.
  IView1D q = dev.charge, sm = dev.spin_multiplicity;
  if ((int) q.extent(0) != NS || (int) sm.extent(0) != NS) {
    q = ws_.i1("cond_q", NS);
    sm = ws_.i1("cond_sm", NS);
    Kokkos::deep_copy(ExecSpace(), q, 0);
    Kokkos::deep_copy(ExecSpace(), sm, 1);
  }

  // [NS, 2*Dn] = [charge_embedding(q + max_charge) | spin_embedding(2S+1 - 1)]
  View2D cat = ws_.n2("cond_cat", NS, 2 * Dn);
  {
    // The Views themselves: these are gathered from element-wise in a kernel,
    // not multiplied, so the Ozaki decomposition mat() also carries is of no use
    // here and a WeightRef is not indexable.
    const View2D qe = mat("system_conditioning.charge_embedding.weight").v;
    const View2D se = mat("system_conditioning.spin_multiplicity_embedding.weight").v;
    const int nq = (int) qe.extent(0), ns = (int) se.extent(0);
    Kokkos::parallel_for(
        "cond_gather", RangePolicy(0, NS * Dn), KOKKOS_LAMBDA(int _i) {
          const int b = _i / Dn, d = _i % Dn;
          // Clamp rather than index out of the table. An out-of-range charge is
          // a caller error, but reading past the embedding is undefined
          // behaviour on the device, where it would surface as a wrong number
          // somewhere else entirely.
          int iq = q(b) + max_q;
          iq = iq < 0 ? 0 : (iq >= nq ? nq - 1 : iq);
          int is = sm(b) - 1;
          is = is < 0 ? 0 : (is >= ns ? ns - 1 : is);
          cat(b, d) = qe(iq, d);
          cat(b, Dn + d) = se(is, d);
        });
  }
  // project = Linear(2*Dn -> Dn) -> SiLU -> Linear(Dn -> Dn)
  View2D h0 = ws_.n2("cond_h0", NS, Dn);
  linear_silu(h0, View2D(), cat, mat("system_conditioning.project.0.weight"),
              vec("system_conditioning.project.0.bias"), false);
  View2D per_struct = ws_.n2("cond_ps", NS, Dn);
  linear(per_struct, h0, mat("system_conditioning.project.2.weight"),
         vec("system_conditioning.project.2.bias"));

  // Broadcast to atoms. Done once here rather than indexing through struct_id at
  // every use, because it is read num_gnn_layers times.
  View2D out = ws_.n2("cond_atom", N, Dn);
  IView1D sid = dev.struct_id;
  const bool have_sid = ((int) sid.extent(0) == N);
  Kokkos::parallel_for(
      "cond_bcast", RangePolicy(0, N * Dn), KOKKOS_LAMBDA(int _i) {
        const int n = _i / Dn, d = _i % Dn;
        out(n, d) = per_struct(have_sid ? sid(n) : 0, d);
      });
  return out;
}

// --- architecture-varying components (see pet/model.hpp) ---------------------

void PetModel::norm(View2D out, View2D in, const std::string& key) const {
  const bool ln = h_.normalization == Normalization::LayerNorm;  // RMSNorm has no bias
  norm_fwd(out, in, vec(key + ".weight"), ln ? vec(key + ".bias") : View1D());
}

void PetModel::norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key, bool acc) const {
  // The bias is a pure shift, so it does not enter the input adjoint.
  pet::norm_bwd(in_adj, out_adj, in, vec(key + ".weight"),
                h_.normalization == Normalization::LayerNorm, acc);
}

int PetModel::ffn_pre_width(const std::string& wkey) const {
  // w_in emits 2*dff for SwiGLU (value|gate) and dff otherwise; the loaded weight
  // already has the right shape, so read it rather than deriving it from hypers.
  return mat(wkey + ".w_in.weight").extent(0);
}

void PetModel::feedforward(const std::string& key, View2D out, View2D in,
                           const std::string& wkey, View2D pre, bool save, Net beta) {
  const int R = in.extent(0);
  WeightRef w_in = mat(wkey + ".w_in.weight"), w_out = mat(wkey + ".w_out.weight");
  View1D b_in = vec(wkey + ".w_in.bias"), b_out = vec(wkey + ".w_out.bias");
  // Delegated rather than open-coded: nvcc refuses an extended __host__ __device__
  // lambda inside a private member function, and these dispatchers are private.
  if (h_.activation == Activation::SwiGLU) {
    feedforward_swiglu(ws_, key, out, in, w_in, b_in, w_out, b_out, pre, beta);
  } else {
    const int dff = pre.extent(1);
    Workspace::Scope scope(ws_);
    View2D h = ws_.tmp(R, dff);
    linear_silu(h, save ? pre : View2D(), in, w_in, b_in, save);  // h = silu(pre)
    linear(out, h, w_out, b_out, beta);
  }
}

void PetModel::feedforward_bwd(const std::string& key, View2D in_adj, View2D out_adj,
                               const std::string& wkey, View2D pre, Net beta) {
  WeightRef w_in = mat(wkey + ".w_in.weight"), w_out = mat(wkey + ".w_out.weight");
  const int R = out_adj.extent(0);
  if (h_.activation == Activation::SwiGLU) {
    feedforward_swiglu_bwd(ws_, key, in_adj, out_adj, pre, w_in, w_out, beta);
  } else {
    const int dff = pre.extent(1);
    Workspace::Scope scope(ws_);
    View2D h_adj = ws_.tmp(R, dff);
    linear_bwd(h_adj, out_adj, w_out, 0);
    silu_bwd(h_adj, pre);
    linear_bwd(in_adj, h_adj, w_in, beta);
  }
}

void PetModel::readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
                       View1D per_atom_net, const PackedEdges& pk, int N,
                       std::vector<View2D>& sav_nh0, std::vector<View2D>& sav_nh1,
                       std::vector<View2D>& sav_eh0, std::vector<View2D>& sav_eh1,
                       std::vector<View2D>& sav_epred, const std::string& key, bool grad) {
  const int R = static_cast<int>(node_feat.size());
  const int Dh = h_.d_head;
  const int NM = pk.E;
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i);
    // node head: Linear(D->Dh) SiLU Linear(Dh->Dh) SiLU -> Linear(Dh->1)
    View2D nh0 = ws_.n2(key + "nh0", N, Dh);
    if (grad) sav_nh0[i] = ws_.n2(key + "s_nh0_" + si, N, Dh);
    linear_silu(nh0, grad ? sav_nh0[i] : View2D(), node_feat[i],
                mat("node_heads.energy." + si + ".0.weight"),
                vec("node_heads.energy." + si + ".0.bias"), grad);
    View2D nh1 = ws_.n2(key + "nh1", N, Dh);
    if (grad) sav_nh1[i] = ws_.n2(key + "s_nh1_" + si, N, Dh);
    linear_silu(nh1, grad ? sav_nh1[i] : View2D(), nh0,
                mat("node_heads.energy." + si + ".2.weight"),
                vec("node_heads.energy." + si + ".2.bias"), grad);
    View2D node_pred = ws_.n2(key + "node_pred", N, 1);
    linear(node_pred, nh1, mat("node_last_layers.energy." + si + ".energy___0.weight"),
           vec("node_last_layers.energy." + si + ".energy___0.bias"));

    // edge head: same shape, over edges
    View2D eh0 = ws_.n2(key + "eh0", NM, Dh);
    if (grad) sav_eh0[i] = ws_.n2(key + "s_eh0_" + si, NM, Dh);
    linear_silu(eh0, grad ? sav_eh0[i] : View2D(), edge_feat[i],
                mat("edge_heads.energy." + si + ".0.weight"),
                vec("edge_heads.energy." + si + ".0.bias"), grad);
    View2D eh1 = ws_.n2(key + "eh1", NM, Dh);
    if (grad) sav_eh1[i] = ws_.n2(key + "s_eh1_" + si, NM, Dh);
    linear_silu(eh1, grad ? sav_eh1[i] : View2D(), eh0,
                mat("edge_heads.energy." + si + ".2.weight"),
                vec("edge_heads.energy." + si + ".2.bias"), grad);
    View2D edge_pred = grad ? (sav_epred[i] = ws_.n2(key + "s_epred_" + si, NM, 1))
                            : ws_.n2(key + "edge_pred", NM, 1);
    linear(edge_pred, eh1, mat("edge_last_layers.energy." + si + ".energy___0.weight"),
           vec("edge_last_layers.energy." + si + ".energy___0.bias"));

    readout_accumulate(per_atom_net, node_pred, edge_pred, pk.off, pk.cut, N, /*zero_first=*/i == 0);
  }
}

WeightRef PetModel::mat(const std::string& name) const {
  auto it = mat_.find(name);
  if (it == mat_.end()) throw std::runtime_error("PetModel: missing matrix '" + name + "'");
  const auto sit = wsplit_.find(name);
  const auto tit = wsplit_t_.find(name);
  return WeightRef{it->second, sit == wsplit_.end() ? nullptr : &sit->second,
                   tit == wsplit_t_.end() ? nullptr : &tit->second};
}
PackedEdges PetModel::pack_edges(const DeviceEdgeData& d) {
  const int N = d.n_atoms, M = d.max_neighbors;
  PackedEdges p;
  p.E = d.n_edges;
  p.off = ws_.i1("pk:off", N + 1);
  p.center = ws_.i1("pk:center", p.E), p.species = ws_.i1("pk:species", p.E);
  p.reverse = ws_.i1("pk:reverse", p.E);
  p.vec = ws_.r2("pk:vec", p.E, 3);
  p.dist = ws_.r1("pk:dist", p.E), p.pcut = ws_.r1("pk:pcut", p.E);
  p.cut = ws_.n1("pk:cut", p.E);
  IView1D pos = ws_.i1("pk:pos", N * M);  // slot -> packed index, -1 for padding
  auto mask = d.mask;
  auto kept = KOKKOS_LAMBDA(int n, int end) {  // kept slots of atom n before `end`
    int c = 0;
    for (int m = 0; m < end; ++m) c += mask(n * M + m) > 0.0;
    return c;
  };
  auto off = p.off;
  Kokkos::parallel_scan(
      "pk_off", RangePolicy(0, N), KOKKOS_LAMBDA(int n, int& upd, bool final) {
        if (final) off(n) = upd;
        upd += kept(n, M);
        if (final && n == N - 1) off(N) = upd;
      });
  if (N == 0) Kokkos::deep_copy(ExecSpace(), off, 0);
  // An edge keeps its slot order: its packed index is its rank among the atom's
  // kept slots, whatever the builder put between them.
  auto center = p.center, species = p.species, reverse = p.reverse;
  auto vec = p.vec;
  auto dist = p.dist, pcut = p.pcut;
  auto cut = p.cut;
  auto nsp = d.neigh_species, rev = d.reverse_index;
  auto ev = d.edge_vec;
  auto dd = d.dist, dpc = d.pair_cutoff;
  auto dcut = d.cutoff_factor;
  Kokkos::parallel_for(
      "pk_edges", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        const int n = k / M;
        if (mask(k) <= 0.0) return (void) (pos(k) = -1);
        const int e = off(n) + kept(n, k % M);
        pos(k) = e, center(e) = n, species(e) = nsp(k);
        for (int c = 0; c < 3; ++c) vec(e, c) = ev(k, c);
        dist(e) = dd(k), pcut(e) = dpc(k), cut(e) = dcut(k);
      });
  Kokkos::parallel_for(
      "pk_reverse", RangePolicy(0, N * M), KOKKOS_LAMBDA(int k) {
        if (pos(k) >= 0) reverse(pos(k)) = rev(k) >= 0 ? pos(rev(k)) : -1;
      });
  return p;
}

CompressFold PetModel::compress_fold(int L) const {
  const std::string c0 = "gnn_layers." + std::to_string(L) + ".compress.0";
  return {mat(c0 + "@x4").v, vec(c0 + "@b"), mat(c0 + "@tab").v, L ? mat(c0 + "@in") : WeightRef{}};
}
const View1D& PetModel::vec(const std::string& name) const {
  auto it = vec_.find(name);
  if (it == vec_.end()) throw std::runtime_error("PetModel: missing vector '" + name + "'");
  return it->second;
}

PetModel::PetModel(const Checkpoint& ckpt)
    : h_(ckpt.hypers),
      energy_scale_(ckpt.energy_scale),
      composition_(ckpt.composition_energies),
      species_to_index_(ckpt.species_to_index) {
  load_all(ckpt);

  // Precompute the constant adaptive-cutoff probe grid once (matches the grid in
  // neighbors.hpp / device_neighbors.hpp). The spacing is set by the ADAPTIVE
  // taper width, not the edge cutoff factor's -- see Hypers::cutoff_width_adaptive.
  std::vector<double> probes_h;
  const double min_cutoff = 0.5, spacing = h_.cutoff_width_adaptive / 4.0;
  for (double p = min_cutoff; p < h_.cutoff - 1e-12; p += spacing) probes_h.push_back(p);
  n_probes_ = static_cast<int>(probes_h.size());
  probes_ = RView1D("probes", n_probes_ > 0 ? n_probes_ : 1);
  auto h_probes = Kokkos::create_mirror_view(probes_);
  for (int p = 0; p < n_probes_; ++p) h_probes(p) = probes_h[p];
  Kokkos::deep_copy(probes_, h_probes);
}

void PetModel::load_all(const Checkpoint& ckpt) {
  const auto& st = ckpt.weights;
  const int G = h_.num_gnn_layers;

  // Row-major host data -> a device matrix / vector under `n`.
  auto put_mat = [&](const std::string& n, int rows, int cols, const double* d) {
    View2D v(n, rows, cols);
    auto hv = Kokkos::create_mirror_view(v);
    for (int a = 0; a < rows; ++a)
      for (int b = 0; b < cols; ++b) hv(a, b) = d[a * cols + b];
    Kokkos::deep_copy(v, hv);
    mat_.emplace(n, v);
  };
  auto put_vec = [&](const std::string& n, int len, const double* d) {
    View1D v(n, len);
    auto hv = Kokkos::create_mirror_view(v);
    for (int a = 0; a < len; ++a) hv(a) = d[a];
    Kokkos::deep_copy(v, hv);
    vec_.emplace(n, v);
  };
  auto load_mat = [&](const std::string& n) {
    const Tensor& t = st.at(n);
    if (t.ndim() != 2) throw std::runtime_error("expected 2D tensor '" + n + "'");
    put_mat(n, t.dim(0), t.dim(1), t.data.data());
  };
  auto load_vec = [&](const std::string& n) {
    const Tensor& t = st.at(n);
    put_vec(n, t.numel(), t.data.data());
  };
  // The CompressFold of layer L (see model.hpp), computed in double. compress.0's
  // input columns are [edge_emb | species block | input_edge], where the species
  // block is the neighbour embedding past layer 0, and at layer 0 is input_edge
  // itself (the edge_embedder lookup) with nothing after it.
  auto fold_compress = [&](int L) {
    const std::string g = "gnn_layers." + std::to_string(L), c0 = g + ".compress.0";
    const Tensor &W = st.at(c0 + ".weight"), &b0 = st.at(c0 + ".bias");
    const Tensor &We = st.at(g + ".edge_embedder.weight"), &be = st.at(g + ".edge_embedder.bias");
    const Tensor& T = st.at(L == 0 ? "edge_embedder.weight" : g + ".neighbor_embedder.weight");
    const int D = W.dim(0), K = W.dim(1), ns = T.dim(0);
    auto w = [&](int d, int j) { return W.data[(std::size_t) d * K + j]; };
    std::vector<double> wx(D * 4, 0.0), b(b0.data), tab((std::size_t) ns * D, 0.0), wi;
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < D; ++j) {
        for (int c = 0; c < 4; ++c) wx[d * 4 + c] += w(d, j) * We.data[j * 4 + c];
        b[d] += w(d, j) * be.data[j];
        for (int sp = 0; sp < ns; ++sp) tab[(std::size_t) sp * D + d] += T.data[(std::size_t) sp * D + j] * w(d, D + j);
      }
    put_mat(c0 + "@x4", D, 4, wx.data());
    put_vec(c0 + "@b", D, b.data());
    put_mat(c0 + "@tab", ns, D, tab.data());
    if (L == 0) return;
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < D; ++j) wi.push_back(w(d, 2 * D + j));
    put_mat(c0 + "@in", D, D, wi.data());
  };
  auto load_linear = [&](const std::string& prefix) {
    load_mat(prefix + ".weight");
    load_vec(prefix + ".bias");
  };

  // A norm carries a bias only for LayerNorm (RMSNorm is scale-only).
  const bool ln = (h_.normalization == Normalization::LayerNorm);
  auto load_norm = [&](const std::string& prefix) {
    load_vec(prefix + ".weight");
    if (ln) load_vec(prefix + ".bias");
  };
  const bool expanded = h_.expanded_node();  // central-token contraction/expansion active
  const int R = h_.num_readout_layers;

  for (int L = 0; L < G; ++L) {
    const std::string g = "gnn_layers." + std::to_string(L);
    fold_compress(L);
    load_linear(g + ".compress.2");
    // One weight set per attention sub-layer (trans.layers.{a}); xs has a single
    // layer, larger PET-MAD models stack several.
    for (int a = 0; a < h_.num_attention_layers; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      load_linear(tl + ".attention.input_linear");
      load_linear(tl + ".attention.output_linear");
      load_norm(tl + ".norm_attention");
      load_norm(tl + ".norm_mlp");
      load_linear(tl + ".mlp.w_in");
      load_linear(tl + ".mlp.w_out");
      // Central-token expansion path exists only when d_node != d_pet; otherwise
      // metatrain makes these Identity and the checkpoint has no center_* weights.
      if (expanded) {
        load_linear(tl + ".center_contraction");
        load_linear(tl + ".center_expansion");
        load_norm(tl + ".norm_center_features");
        load_linear(tl + ".center_mlp.w_in");
        load_linear(tl + ".center_mlp.w_out");
      }
    }
    // combination_mlps/norms exist only for the feedforward featurizer.
    if (h_.featurizer_type == FeaturizerType::FeedForward) {
      load_vec("combination_norms." + std::to_string(L) + ".weight");
      load_vec("combination_norms." + std::to_string(L) + ".bias");
      load_linear("combination_mlps." + std::to_string(L) + ".0");
      load_linear("combination_mlps." + std::to_string(L) + ".2");
    }
  }
  load_mat("edge_embedder.weight");

  // Charge / spin conditioning: two embedding tables plus a Linear-SiLU-Linear
  // projection, all per system rather than per atom.
  if (h_.system_conditioning) {
    load_mat("system_conditioning.charge_embedding.weight");
    load_mat("system_conditioning.spin_multiplicity_embedding.weight");
    load_linear("system_conditioning.project.0");
    load_linear("system_conditioning.project.2");
  }
  // One node embedder + one node/edge readout head per readout layer (feedforward
  // reads out once from the final layer; residual reads out from every GNN layer).
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i);
    load_mat("node_embedders." + si + ".weight");
    load_linear("node_heads.energy." + si + ".0");
    load_linear("node_heads.energy." + si + ".2");
    load_linear("edge_heads.energy." + si + ".0");
    load_linear("edge_heads.energy." + si + ".2");
    load_linear("node_last_layers.energy." + si + ".energy___0");
    load_linear("edge_last_layers.energy." + si + ".energy___0");
  }

  // Two validated architecture families; each compute path assumes its sub-config.
  if (h_.featurizer_type == FeaturizerType::FeedForward) {
    if (h_.transformer_type != TransformerType::PreLN ||
        h_.normalization != Normalization::RMSNorm || h_.activation != Activation::SwiGLU ||
        !expanded)
      throw std::runtime_error(
          "PetModel: feedforward featurizer is only implemented for "
          "PreLN + RMSNorm + SwiGLU with expanded central token (d_node != d_pet)");
  } else {  // Residual
    if (h_.transformer_type != TransformerType::PostLN ||
        h_.normalization != Normalization::LayerNorm || h_.activation != Activation::SiLU ||
        expanded)
      throw std::runtime_error(
          "PetModel: residual featurizer is only implemented for "
          "PostLN + LayerNorm + SiLU with non-expanded central token (d_node == d_pet)");
    // compute_residual()'s geometry backward has no adaptive-cutoff chain-rule
    // term (the feedforward path's adaptive_backward is absent). A residual model
    // with an adaptive cutoff would produce a correct forward energy but forces
    // and virial inconsistent with it, silently. Reject rather than mislead.
    if (h_.adaptive())
      throw std::runtime_error(
          "PetModel: residual featurizer with an adaptive cutoff "
          "(num_neighbors_adaptive > 0) is not implemented (forces would be "
          "inconsistent with the energy); use a fixed cutoff");
  }
  if (h_.num_attention_layers < 1)
    throw std::runtime_error("PetModel: num_attention_layers must be >= 1");
  // Ozaki: decompose every weight once, here, and never again. This is what
  // makes the scheme affordable for this network -- each GEMM is
  // `activation x weight`, and a decomposition depends only on its own operand,
  // so half of every product's splitting work is done before the first
  // evaluation runs.
  //
  // Weights use a matrix-wide scale (ozaki_split_weight), not per-row: a weight
  // is used transposed by linear() and untransposed by linear_bwd(), and those
  // want normalisation along opposite axes. A scalar is constant along both, so
  // one decomposition serves both directions instead of two.
  if (ozaki_active()) {
    const int slices = ozaki_config().slices;
    for (const auto& kv : mat_) {
      wsplit_.emplace(kv.first, ozaki_split_weight(kv.second, slices, false));
      wsplit_t_.emplace(kv.first, ozaki_split_weight(kv.second, slices, true));
    }
  }

  if (h_.head_dim > kMaxHeadDim)
    throw std::runtime_error("PetModel: head_dim " + std::to_string(h_.head_dim) +
                             " exceeds the attention kernels' ceiling of " +
                             std::to_string(kMaxHeadDim) + " (see kMaxHeadDim in model.cpp)");

  // device mirror of composition energies (double)
  comp_view_ = RView1D("composition", composition_.size());
  auto hcomp = Kokkos::create_mirror_view(comp_view_);
  for (std::size_t i = 0; i < composition_.size(); ++i) hcomp(i) = composition_[i];
  Kokkos::deep_copy(comp_view_, hcomp);
}

// Marshal a host EdgeData into device Views: the single host->device boundary of
// the host neighbour path, as one copy (see Upload). The Views alias pooled
// memory, so they are valid until the next upload -- every caller consumes them
// straight away.
DeviceEdgeData PetModel::upload_edge_data(const EdgeData& ed, bool need_reverse) const {
  const int N = ed.n_atoms, M = ed.max_neighbors, S = M + 1, NM = N * M, E = ed.n_raw;
  DeviceEdgeData dev;
  dev.n_atoms = N;
  dev.max_neighbors = M;
  dev.n_raw = E;
  dev.n_edges = (int) std::count_if(ed.mask.begin(), ed.mask.end(), [](char m) { return m != 0; });

  // cf_seq: the attention bias source, col 0 = 1 (the central token), then the
  // per-edge cutoff factors.
  std::vector<double> cf((std::size_t) N * S, 1.0);
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) cf[(std::size_t) n * S + 1 + m] = ed.cutoff_factor[(std::size_t) n * M + m];

  // Per-centre ranges and edge partners of the raw list, so the adaptive
  // backward can gather instead of scattering with atomics (see
  // adaptive_backward). Built only when the list really is grouped by centre and
  // every edge finds its partner, and only when a backward of an adaptive model
  // will read them; otherwise left empty and the gather falls back.
  std::vector<int> off(N + 1, 0), rev;
  bool grouped = E > 0 && need_reverse && h_.adaptive();
  for (int e = 0; e < E && grouped; ++e) {
    if (e > 0 && ed.raw_center[e] < ed.raw_center[e - 1]) grouped = false;
    ++off[ed.raw_center[e] + 1];
  }
  if (grouped) {
    for (int a = 0; a < N; ++a) off[a + 1] += off[a];
    // Partner of (i, j, v) is (j, i, -v). Matched on the neighbour index and the
    // closest opposing vector, which distinguishes periodic images of the same
    // pair; an exact float compare would be at the mercy of how each vector was
    // rounded.
    //
    // Is raw_neigh non-decreasing within each centre's run? Both host searches
    // emit the list sorted by (i, j, shift) -- the vesin wrapper sorts
    // explicitly, the built-in search's nested loops produce it naturally -- but
    // neither is contractually required to, and a caller supplying its own edge
    // list certainly is not. Checked in one pass rather than assumed.
    bool sorted_runs = true;
    for (int a = 0; a < N && sorted_runs; ++a)
      for (int f = off[a] + 1; f < off[a + 1]; ++f)
        if (ed.raw_neigh[f] < ed.raw_neigh[f - 1]) { sorted_runs = false; break; }

    rev.assign(E, -1);
    for (int e = 0; e < E && grouped; ++e) {
      const int i = ed.raw_center[e], j = ed.raw_neigh[e];
      const double vx = -ed.raw_vec[3 * e + 0], vy = -ed.raw_vec[3 * e + 1],
                   vz = -ed.raw_vec[3 * e + 2];
      // Narrow the scan to the block of j's edges that point back at i.
      //
      // Without this the loop walks all of atom j's neighbours for every edge:
      // O(E x neighbours), 147 million iterations on a 1728-atom supercell and
      // ~72 ms -- more than the adaptive-cutoff solver and the whole NEF packer
      // combined. When the run is sorted, the edges with raw_neigh == i are
      // contiguous (one to three periodic images of the same pair), so a pair of
      // binary searches finds them directly. It only skips entries the scan
      // would have rejected on the `raw_neigh != i` test anyway.
      int lo = off[j], hi = off[j + 1];
      if (sorted_runs) {
        const auto begin = ed.raw_neigh.begin();
        lo = (int) (std::lower_bound(begin + off[j], begin + off[j + 1], i) - begin);
        hi = (int) (std::upper_bound(begin + lo, begin + off[j + 1], i) - begin);
      }
      double best = 1e300;
      int found = -1;
      for (int f = lo; f < hi; ++f) {
        if (ed.raw_neigh[f] != i) continue;
        const double dx = ed.raw_vec[3 * f + 0] - vx, dy = ed.raw_vec[3 * f + 1] - vy,
                     dz = ed.raw_vec[3 * f + 2] - vz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < best) { best = r2; found = f; }
      }
      if (found < 0 || best > 1e-12) grouped = false;  // unmatched -> use the fallback
      rev[e] = found;
    }
  }

  Upload& up = upload_;
  up.clear();
  const auto o_sp = up.add<int>(ed.species), o_ns = up.add<int>(ed.neigh_species),
             o_rev = up.add<int>(ed.reverse_index), o_ev = up.add<Real>(ed.edge_vec),
             o_d = up.add<Real>(ed.edge_dist), o_mask = up.add<Real>(ed.mask),
             o_pc = up.add<Real>(ed.pair_cutoff), o_cut = up.add<Net>(ed.cutoff_factor),
             o_cf = up.add<Net>(cf), o_rc = up.add<int>(ed.raw_center),
             o_rj = up.add<int>(ed.raw_neigh), o_rd = up.add<Real>(ed.raw_dist),
             o_rv = up.add<Real>(ed.raw_vec);
  // Per-structure electronic state, for a conditioned model.
  const int B = (int) ed.charge.size();
  std::vector<int> spin(B, 1);
  for (int b = 0; b < B && b < (int) ed.spin_multiplicity.size(); ++b) spin[b] = ed.spin_multiplicity[b];
  const auto o_q = up.add<int>(ed.charge), o_s = up.add<int>(spin);
  // Solver adaptive cutoff: the forward's root and slope. Without these the
  // backward has nothing to differentiate through -- it cannot recompute them,
  // having no per-atom segmentation of the raw edge list on this path.
  const bool solver = (int) ed.adapt_r.size() == N && N > 0;
  const auto o_ar = up.add<Real>(ed.adapt_r), o_adn = up.add<Real>(ed.adapt_dn);
  const auto o_off = up.add<int>(off), o_rrev = up.add<int>(rev);
  up.send();

  dev.species = up.get<IView1D>(o_sp, N);
  dev.neigh_species = up.get<IView1D>(o_ns, NM);
  dev.reverse_index = up.get<IView1D>(o_rev, NM);
  dev.edge_vec = up.get<RView2D>(o_ev, NM, 3);
  dev.dist = up.get<RView1D>(o_d, NM);
  dev.mask = up.get<RView1D>(o_mask, NM);
  dev.pair_cutoff = up.get<RView1D>(o_pc, NM);
  dev.cutoff_factor = up.get<View1D>(o_cut, NM);
  dev.cf_seq = up.get<View2D>(o_cf, N, S);
  dev.raw_center = up.get<IView1D>(o_rc, E);
  dev.raw_neigh = up.get<IView1D>(o_rj, E);
  dev.raw_dist = up.get<RView1D>(o_rd, E);
  dev.raw_vec = up.get<RView2D>(o_rv, E, 3);
  if (B > 0) dev.charge = up.get<IView1D>(o_q, B), dev.spin_multiplicity = up.get<IView1D>(o_s, B);
  if (solver) dev.adapt_r = up.get<RView1D>(o_ar, N), dev.adapt_dn = up.get<RView1D>(o_adn, N);
  if (grouped) dev.raw_off = up.get<IView1D>(o_off, N + 1), dev.raw_reverse = up.get<IView1D>(o_rrev, E);
  return dev;
}

BatchResult PetModel::energy_forces_batch(const std::vector<System>& systems, bool compute_forces) {
  BatchResult out;
  const int B = static_cast<int>(systems.size());
  out.n_struct = B;
  if (B == 0) return out;

  // build each structure's NEF (host), concatenate into one combined NEF
  std::vector<EdgeData> parts;
  parts.reserve(B);
  for (const auto& sys : systems) parts.push_back(build_edge_data(sys, h_, species_to_index_));
  std::vector<int> struct_id_h;
  EdgeData combined = concat_edge_data(parts, struct_id_h);
  const int N = combined.n_atoms;

  // single upload of the combined NEF + the per-atom structure id, then dispatch
  // to the device-resident core
  DeviceEdgeData dev = upload_edge_data(combined);
  IView1D struct_id("struct_id", N);
  auto h_sid = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, struct_id);
  for (int a = 0; a < N; ++a) h_sid(a) = struct_id_h[a];
  Kokkos::deep_copy(struct_id, h_sid);
  dev.struct_id = struct_id;
  dev.n_struct = B;
  return energy_forces_batch(dev, compute_forces);
}

BatchResult PetModel::energy_forces_batch(const DeviceEdgeData& dev, bool compute_forces) {
  BatchResult out;
  const int N = dev.n_atoms, B = dev.n_struct;
  out.n_struct = B;
  out.n_atoms = N;
  out.struct_id = dev.struct_id;
  if (N == 0 || B == 0) return out;

  // one compute() over all atoms; device per-atom energy view + per-structure
  // virial come back without a host copy
  RView2D forces, virial;
  RView1D per_atom;
  compute(dev, nullptr, compute_forces ? &forces : nullptr, &per_atom,
          compute_forces ? &virial : nullptr);
  out.per_atom = per_atom;

  // per-structure energy: on-device segmented sum of the per-atom energy.
  // Pooled rather than freshly allocated: this ran once per optimizer round, and a
  // fresh View is a cudaMalloc plus a synchronizing cudaFree. Zeroed explicitly
  // because it is an atomic accumulator and the pool's zeroing policy is whatever
  // the forward/backward passes left it as. The caller consumes out.energy within
  // the round, before the next compute() hands the same buffer back.
  RView1D energy = ws_.r1("batch_energy", B);
  Kokkos::deep_copy(ExecSpace(), energy, Real(0));
  auto sid = dev.struct_id;
  // Ordered segmented sum, not an atomic scatter: atoms of a structure are
  // contiguous in the batch, so counting them and prefix-summing gives each
  // structure its atom range and the sum runs in atom-index order. A float
  // atomic_add here made the reported ENERGY itself depend on thread arrival
  // order, which is the quantity the whole re-ranking stage is built on.
  {
    // Zeroed EXPLICITLY, for the same reason `energy` above is: this is an
    // atomic accumulator, and the pool only zeroes on reuse when its zeroing
    // policy happens to be on. compute() turns that policy off for the forward
    // and back on for the backward -- so on an energy-only evaluation, which
    // never runs a backward, it is left off and this counter accumulated across
    // calls. The symptom was a second batched energy-only evaluation putting
    // every atom in structure 0: correct on the first call, silently wrong on
    // the second. Relying on a policy set by a different function is not
    // something to repeat.
    IView1D ecnt = ws_.i1("batch_energy_cnt", B);
    Kokkos::deep_copy(ExecSpace(), ecnt, 0);
    IView1D eoff = ws_.i1("batch_energy_off", B + 1);
    Kokkos::parallel_for(
        "batch_energy_count", Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(int a) { Kokkos::atomic_inc(&ecnt(sid(a))); });
    Kokkos::parallel_scan(
        "batch_energy_scan", Kokkos::RangePolicy<ExecSpace>(0, B),
        KOKKOS_LAMBDA(int b, int& upd, bool final) {
          if (final) eoff(b) = upd;
          upd += ecnt(b);
          if (final && b == B - 1) eoff(B) = upd;
        });
    Kokkos::parallel_for(
        "batch_energy_seg", Kokkos::RangePolicy<ExecSpace>(0, B), KOKKOS_LAMBDA(int b) {
          Real s = Real(0);
          for (int a = eoff(b); a < eoff(b + 1); ++a) s += per_atom(a);
          energy(b) = s;
        });
  }
  out.energy = energy;
  if (compute_forces) {
    out.forces = forces;
    out.virial = virial;
  }
  return out;
}

// ============================================================================
// Residual-featurizer forward/backward (PostLN + LayerNorm + SiLU, non-expanded
// central token, num_attention_layers>=1). Structurally distinct from the
// feedforward path in compute(): fresh per-layer node embedding, per-layer
// node+edge energy readout, and a 0.5*(input + reversed) edge message update.
// Kept as a separate method so the tightly-optimized feedforward path is
// untouched; the geometry-gradient scatter tail mirrors compute().
// ============================================================================
DeviceOut PetModel::residual_pass(const DeviceEdgeData& dev, bool grad) {
  Workspace::Scope scope(ws_);  // every Workspace::tmp() is inside some Scope
  const int N = dev.n_atoms;
  const int M = dev.max_neighbors;
  const int S = M + 1;
  const int D = h_.d_pet;   // == d_node (non-expanded central token)
  const int Dh = h_.d_head;
  const int G = h_.num_gnn_layers;
  const int A = h_.num_attention_layers;
  const int R = h_.num_readout_layers;  // == G for the residual featurizer
  // Charge / spin conditioning, computed once and reused by every GNN layer.
  // Empty (and free) for a model without it.
  View2D cond = conditioning(dev, dev.n_atoms, dev.n_struct);

  // Forward activations are fully overwritten; skip per-reuse zeroing.
  ws_.set_zero(false);

  auto d_species = dev.species;
  auto d_cf_seq = dev.cf_seq;
  // Edge tensors are [NM, .] over the kept edges only (see PackedEdges).
  const PackedEdges pk = pack_edges(dev);
  const int NM = pk.E;
  auto off = pk.off, d_reverse = pk.reverse;

  // initial edge messages = species embedding of the neighbor; geometric (v,|v|)
  View2D input_edge = ws_.n2("re_input_edge", NM, D);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);

  // per-GNN-layer node/edge features (all live simultaneously at readout).
  std::vector<View2D> node_feat(G), edge_feat(G);

  // Saved forward activations the analytic backward needs (only when grad).
  std::vector<View2D> sav_scpre(G);  // compress.0 pre-SiLU, per layer
  std::vector<std::vector<View2D>> sav_qkv(G, std::vector<View2D>(A)),
      sav_s1(G, std::vector<View2D>(A)),      // LayerNorm-attention input (tokens+attn)
      sav_s2(G, std::vector<View2D>(A)),      // LayerNorm-mlp input (tok_a+ff)
      sav_mlppre(G, std::vector<View2D>(A));  // mlp.w_in pre-SiLU
  std::vector<View2D> sav_nh0(R), sav_nh1(R), sav_eh0(R), sav_eh1(R), sav_epred(R);

  for (int L = 0; L < G; ++L) {
    const std::string g = "gnn_layers." + std::to_string(L);
    const std::string ls = std::to_string(L);

    // fresh per-layer central-node embedding (residual featurizer: node input is
    // re-embedded from species every layer; only edge messages carry over).
    View2D node_L = ws_.n2("re_node_L", N, D);
    gather(node_L, mat("node_embedders." + ls + ".weight").v, d_species);

    // et = compress.2(silu(compress.0([edge_emb | nb_emb | input_edge]))), through
    // the load-time fold (CompressFold), exactly as in the feedforward path.
    View2D cpre = ws_.n2("re_cpre", NM, D);
    if (grad) sav_scpre[L] = ws_.n2("rs_cpre_" + ls, NM, D);
    compress_fwd(cpre, sav_scpre[L], grad, compress_fold(L), input_edge, pk.species, pk.vec, pk.dist);
    View2D et = ws_.n2("re_et", NM, D);
    linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

    // tokens[N,S,D]: s==0 central node token, s>=1 edge tokens
    View2D tokens = ws_.n2("re_tokens", N * S, D);
    Kokkos::parallel_for(
        "re_tok", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
          const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
          tokens(row, d) = row % S == 0 ? node_L(n, d) : k < off(n + 1) ? et(k, d) : Net(0);
        });

    // PostLN transformer stack
    for (int a = 0; a < A; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string as = ls + "_" + std::to_string(a);
      Workspace::Scope block_scope(ws_);

      // attention on the RAW tokens (PostLN)
      View2D qkv = grad ? (sav_qkv[L][a] = ws_.n2("rs_qkv_" + as, N * S, 3 * D))
                        : ws_.n2("re_qkv", N * S, 3 * D);
      linear(qkv, tokens, mat(tl + ".attention.input_linear.weight"),
             vec(tl + ".attention.input_linear.bias"));
      // s1 buffer receives attn, then += tokens in place -> LayerNorm-attention input.
      View2D s1 = grad ? (sav_s1[L][a] = ws_.n2("rs_s1_" + as, N * S, D))
                       : ws_.n2("re_attn_out", N * S, D);
      attention(ws_, "re_attn_" + as, s1, qkv, d_cf_seq,
                mat(tl + ".attention.output_linear.weight"),
                vec(tl + ".attention.output_linear.bias"), N, S, h_.num_heads, h_.head_dim,
                h_.attention_temperature, grad);

      // tokens := norm_attention(tokens + attn_out)
      add_inplace(s1, tokens);  // s1 += tokens
      View2D tok_a = ws_.n2("re_tok_a", N * S, D);
      norm(tok_a, s1, tl + ".norm_attention");

      // feedforward: w_out(silu(w_in(tok_a)))
      const int pw = ffn_pre_width(tl + ".mlp");
      sav_mlppre[L][a] = grad ? ws_.n2("rs_mlppre_" + as, N * S, pw) : ws_.tmp(N * S, pw);
      // s2 receives ff, then += tok_a in place -> the norm-mlp input.
      View2D s2 = grad ? (sav_s2[L][a] = ws_.n2("rs_s2_" + as, N * S, D))
                       : ws_.n2("re_ff", N * S, D);
      feedforward("re_mlp_" + as, s2, tok_a, tl + ".mlp", sav_mlppre[L][a], grad);

      // tokens := norm_mlp(tok_a + ff)
      add_inplace(s2, tok_a);  // s2 += tok_a
      View2D tok_next = ws_.n2("re_tokens", N * S, D);
      norm(tok_next, s2, tl + ".norm_mlp");
      tokens = tok_next;
    }

    // split tokens -> per-layer node/edge features (kept for readout)
    View2D onode = ws_.n2("re_nfeat_" + ls, N, D);
    View2D oedge = ws_.n2("re_efeat_" + ls, NM, D);
    node_feat[L] = onode;
    edge_feat[L] = oedge;
    Kokkos::parallel_for(
        "re_split", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
          const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
          if (row % S == 0) onode(n, d) = tokens(row, d);
          else if (k < off(n + 1)) oedge(k, d) = tokens(row, d);
        });

    // Charge / spin conditioning, added to this layer's node output before it is
    // read out -- the same point metatrain adds it. The residual featurizer is
    // non-expanded, so d_node == d_pet and the projection's width matches D.
    if (cond.extent(0) == (std::size_t) N) {
      Kokkos::parallel_for(
          "re_cond_add", RangePolicy(0, N * D), KOKKOS_LAMBDA(int _i) {
            const int n = _i / D, d = _i % D;
            onode(n, d) += cond(n, d);
          });
    }

    // message passing: input_edge := 0.5*(input_edge + output_edge[reverse])
    if (L + 1 < G) {
      Kokkos::parallel_for(
          "re_msg", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            const int r = d_reverse(k);
            const Net rev = (r >= 0) ? oedge(r, d) : Net(0);
            input_edge(k, d) = Net(0.5) * (input_edge(k, d) + rev);
          });
    }
  }

  // ---- readout: sum over readout layers of (node_pred + sum_m cf*edge_pred) ----
  // Readout: the residual featurizer reads out from every GNN layer (R == G) and
  // sums; see PetModel::readout, shared with the feedforward path.
  View1D per_atom_net = ws_.n1("re_per_atom_net", N);
  readout(node_feat, edge_feat, per_atom_net, pk, N,
          sav_nh0, sav_nh1, sav_eh0, sav_eh1, sav_epred, "re_", grad);

  // per-atom energy assembly: e = scale*net + composition[species]
  RView1D per_atom = ws_.r1("re_per_atom", N);
  {
    const double scale = energy_scale_;
    auto comp = comp_view_;
    auto species = d_species;
    Kokkos::parallel_for(
        "re_assemble", RangePolicy(0, N), KOKKOS_LAMBDA(int n) {
          per_atom(n) = scale * per_atom_net(n) + comp(species(n));
        });
  }
  if (!grad) return {per_atom, {}, {}};

  // ==========================================================================
  // backward: adjoint of sum(per_atom_net) wrt edge vectors (energy_scale is
  // applied at the force/virial scatter, matching the feedforward path).
  // ==========================================================================
  ws_.set_zero(true);  // adjoint accumulators (linear_bwd/norm_bwd/atomics) need zeroing
  View2D edge_in4_adj = ws_.n2("re_edge_in4_adj", NM, 4);  // geometry adjoint (all layers)
  View2D cf_seq_adj = ws_.n2("re_cf_seq_adj", N, S);       // attention-bias adjoint (all attn)
  View1D cutoff_adj = ws_.n1("re_cutoff_adj", NM);         // readout cutoff-factor adjoint

  // per-GNN-layer node/edge feature adjoints (readout term; edges also get the
  // message-passing term inside the reverse layer loop).
  std::vector<View2D> nfeat_adj(G), efeat_adj(G);
  for (int L = 0; L < G; ++L) {
    nfeat_adj[L] = ws_.n2("re_nfa_" + std::to_string(L), N, D);
    efeat_adj[L] = ws_.n2("re_efa_" + std::to_string(L), NM, D);
  }

  // ---- readout backward ----
  for (int i = 0; i < R; ++i) {
    const std::string si = std::to_string(i);
    View2D node_pred_adj = ws_.n2("re_npa", N, 1);
    View2D edge_pred_adj = ws_.n2("re_epa", NM, 1);
    auto epred = sav_epred[i];
    auto cut = pk.cut;
    Kokkos::parallel_for(
        "re_ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { node_pred_adj(n, 0) = Net(1); });
    Kokkos::parallel_for(
        "re_ro_seed_e", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          edge_pred_adj(k, 0) = cut(k);
          cutoff_adj(k) += epred(k, 0);  // sequential over i: no race
        });
    // node head backward
    View2D nh1_adj = ws_.n2("re_nh1_adj", N, Dh);
    linear_bwd(nh1_adj, node_pred_adj, mat("node_last_layers.energy." + si + ".energy___0.weight"));
    silu_bwd(nh1_adj, sav_nh1[i]);
    View2D nh0_adj = ws_.n2("re_nh0_adj", N, Dh);
    linear_bwd(nh0_adj, nh1_adj, mat("node_heads.energy." + si + ".2.weight"));
    silu_bwd(nh0_adj, sav_nh0[i]);
    linear_bwd(nfeat_adj[i], nh0_adj, mat("node_heads.energy." + si + ".0.weight"));
    // edge head backward
    View2D eh1_adj = ws_.n2("re_eh1_adj", NM, Dh);
    linear_bwd(eh1_adj, edge_pred_adj, mat("edge_last_layers.energy." + si + ".energy___0.weight"));
    silu_bwd(eh1_adj, sav_eh1[i]);
    View2D eh0_adj = ws_.n2("re_eh0_adj", NM, Dh);
    linear_bwd(eh0_adj, eh1_adj, mat("edge_heads.energy." + si + ".2.weight"));
    silu_bwd(eh0_adj, sav_eh0[i]);
    linear_bwd(efeat_adj[i], eh0_adj, mat("edge_heads.energy." + si + ".0.weight"));
  }

  // ---- layers backward (reverse order) ----
  // ie_adj carries the adjoint of input_edge_{L+1} into layer L's message term.
  View2D ie_adj = ws_.n2("re_ie_adj", NM, D);  // zero for the (nonexistent) input_edge_G
  for (int L = G - 1; L >= 0; --L) {
    const std::string g = "gnn_layers." + std::to_string(L);
    const std::string ls = std::to_string(L);

    // message passing L->L+1: input_edge_{L+1}(k) = 0.5*(input_edge_L(k) + oedge(reverse(k))).
    // Scatter 0.5*ie_adj into edge_feat[L] (reverse is a bijection on real edges -> no race).
    if (L + 1 < G) {
      auto efa = efeat_adj[L];
      Kokkos::parallel_for(
          "reb_msg_edge", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            const int r = d_reverse(k);
            if (r >= 0) efa(r, d) += Net(0.5) * ie_adj(k, d);
          });
    }

    // assemble tokens_adj from node/edge feature adjoints
    View2D tokens_adj = ws_.n2("re_tokens_adj", N * S, D);
    {
      auto nfa = nfeat_adj[L];
      auto efa = efeat_adj[L];
      Kokkos::parallel_for(
          "reb_asm_tok", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
            const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
            tokens_adj(row, d) = row % S == 0 ? nfa(n, d) : k < off(n + 1) ? efa(k, d) : Net(0);
          });
    }

    // reverse the PostLN transformer stack
    for (int a = A - 1; a >= 0; --a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string as = ls + "_" + std::to_string(a);
      // tokens_out = norm_mlp(s2); s2 = tok_a + ff
      View2D s2_adj = ws_.n2("re_s2_adj", N * S, D);
      norm_bwd(s2_adj, tokens_adj, sav_s2[L][a], tl + ".norm_mlp");
      // tok_a_adj = s2_adj (the residual) + the MLP's own contribution, which
      // feedforward_bwd accumulates on top -- hence the seed first.
      View2D tok_a_adj = ws_.n2("re_tok_a_adj", N * S, D);
      Kokkos::parallel_for(
          "reb_tokacopy", RangePolicy(0, (N * S) * D),
          KOKKOS_LAMBDA(int _i) { const int r = _i / D, d = _i % D; tok_a_adj(r, d) = s2_adj(r, d); });
      feedforward_bwd("re_mlpb_" + as, tok_a_adj, s2_adj, tl + ".mlp", sav_mlppre[L][a]);
      // tok_a = norm_attention(s1); s1 = tokens_in + attn
      View2D s1_adj = ws_.n2("re_s1_adj", N * S, D);
      norm_bwd(s1_adj, tok_a_adj, sav_s1[L][a], tl + ".norm_attention");
      // attn = output_linear(attention(input_linear(tokens_in)))
      View2D an_adj = ws_.n2("re_an_adj", N * S, D);
      attention_bwd(ws_, "reb_attn_" + as, "re_attn_" + as, an_adj, cf_seq_adj, s1_adj, sav_qkv[L][a],
                    d_cf_seq, mat(tl + ".attention.input_linear.weight"),
                    mat(tl + ".attention.output_linear.weight"), N, S, h_.num_heads, h_.head_dim,
                    h_.attention_temperature);
      // tokens_in_adj = s1_adj (residual) + an_adj (through attention)
      View2D tin_adj = ws_.n2("re_tin_adj", N * S, D);
      Kokkos::parallel_for(
          "reb_tin", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
            const int r = _i / D, d = _i % D;
            tin_adj(r, d) = s1_adj(r, d) + an_adj(r, d);
          });
      tokens_adj = tin_adj;  // adjoint of this sub-layer's input tokens
    }

    // tokens_L = [node_L ; et]. Central token -> species embedding (no geometry);
    // edge tokens -> et.
    View2D et_adj = ws_.n2("re_et_adj", NM, D);
    auto center = pk.center;
    Kokkos::parallel_for(
        "reb_split_tok", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
          const int k = _i / D, d = _i % D, n = center(k);
          et_adj(k, d) = tokens_adj(n * S + 1 + k - off(n), d);
        });
    // input_edge_L feeds compress.0 AND, through message passing, carries 0.5 of
    // input_edge_{L+1}'s adjoint. Two buffers alternating by layer parity: the
    // pool zeroes whatever it hands out, so asking for the one ie_adj lives in
    // would wipe it before it is read.
    View2D ie_next = ws_.n2("re_ie_" + std::to_string(L % 2), NM, D);
    {
      const bool has_msg = (L + 1 < G);
      auto ie_prev = ie_adj;
      Kokkos::parallel_for(
          "reb_ie", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            ie_next(k, d) = has_msg ? Net(0.5) * ie_prev(k, d) : Net(0);
          });
    }
    View2D cpre_adj = ws_.n2("re_cpre_adj", NM, D);
    linear_bwd(cpre_adj, et_adj, mat(g + ".compress.2.weight"));
    compress_bwd(cpre_adj, sav_scpre[L], compress_fold(L), edge_in4_adj, ie_next);
    ie_adj = ie_next;  // carry to the previous (earlier) layer
  }

  // ---- geometry backward + force/virial scatter (non-adaptive; mirrors compute()) ----
  RView2D edge_grad = ws_.r2("re_edge_grad", NM, 3);  // dE/dv per kept edge (double)
  {
    const bool is_bump = (h_.cutoff_function == CutoffFunction::Bump);
    const double width = h_.cutoff_width;
    auto center = pk.center;
    auto vec = pk.vec;
    auto dist = pk.dist, pcut = pk.pcut;
    Kokkos::parallel_for(
        "re_geom_bwd", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          const int n = center(k), m = k - off(n);
          const double cutoff_total = cutoff_adj(k) + cf_seq_adj(n, 1 + m);
          const double dcut_dd = cutoff_ddist(dist(k), pcut(k), width, is_bump);
          const double dist_adj = edge_in4_adj(k, 3) + cutoff_total * dcut_dd;
          const double invd = (dist(k) > 0.0) ? 1.0 / dist(k) : 0.0;
          for (int c = 0; c < 3; ++c)
            edge_grad(k, c) = edge_in4_adj(k, c) + dist_adj * vec(k, c) * invd;
          // non-adaptive: pair_cutoff is fixed, so no cutoff-radius backward term.
        });
  }

  RView2D d_forces = ws_.r2("re_forces", N, 3);
  const int NS = dev.n_struct;
  auto sid = dev.struct_id;
  RView2D dvir = ws_.r2("re_virial9", NS, 9);
  fold_edge_gradients(ws_, "re", edge_grad, pk, d_forces, dvir, sid, N, NS, energy_scale_);

  return {per_atom, d_forces, dvir};
}

// ============================================================================
// Feedforward featurizer: one GNN layer, forward and backward (see model.hpp).
// ============================================================================
void PetModel::ff_layer(const DeviceEdgeData& dev, const PackedEdges& pk, int L, View2D& node,
                        View2D input_edge, View2D cond, LayerSaves* sav, bool save_wide) {
  const int N = dev.n_atoms, S = dev.max_neighbors + 1, NM = pk.E;  // NM: edge rows
  const int D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  const bool save = sav != nullptr;
  auto rev = pk.reverse;
  auto off = pk.off;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  if (save)
    for (auto* v : {&sav->tokens, &sav->qkv, &sav->node_new, &sav->tmp_center, &sav->eps, &sav->tmp_edge})
      v->assign(A, View2D());
  auto keep = [&](View2D& slot, const std::string& key, int r, int c) { return slot = ws_.n2(key + sav->tag, r, c); };
  Workspace::Scope layer_scope(ws_);  // temporaries below die with the layer

  // et = compress.2(silu(compress.0([edge_emb | nb_emb | input_edge]))), with
  // compress.0 through its fold (CompressFold) -- no edge_emb, no concat.
  View2D cpre = ws_.tmp(NM, D);
  compress_fwd(cpre, save ? keep(sav->cpre, "s_cpre_", NM, D) : View2D(), save, compress_fold(L),
               input_edge, pk.species, pk.vec, pk.dist);
  View2D et = ws_.tmp(NM, D);
  linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

  // ---- transformer stack: A TransformerLayer blocks --------------------------
  //
  // metatrain's TransformerLayer is the WHOLE unit -- centre contraction,
  // attention, centre expansion and MLP, edge residual and MLP -- and
  // Transformer.forward runs A of them in sequence, threading (node, edge)
  // through. So A > 1 is a loop over this entire body, not just over the
  // attention itself; `et` seeds the first block and each block's outputs feed
  // the next.
  View2D node_cur = node, edge_cur = et;
  for (int a = 0; a < A; ++a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    const std::string as = (save ? sav->tag : ls) + "_" + std::to_string(a);
    // The node output leaves the layer, so it has a name -- two, alternating
    // across the whole (GNN layer, attention layer) sequence, since a block's
    // output only has to outlive the next block. The edge output lives in the
    // layer's scope; everything else here dies with the block.
    const std::string pp = std::to_string((L * A + a) % 2);
    View2D edge_next = ws_.tmp(NM, D);
    Workspace::Scope block_scope(ws_);

    // central token contraction
    View2D input_node = ws_.tmp(N, D);
    linear(input_node, node_cur, mat(tl + ".center_contraction.weight"),
           vec(tl + ".center_contraction.bias"));

    View2D tokens = save ? keep(sav->tokens[a], "s_tok_" + std::to_string(a) + "_", N * S, D)
                         : ws_.tmp(N * S, D);
    Kokkos::parallel_for(
        "tok", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
          // row % S == 0 is the central token; edge rows past the atom's count are padding
          const int row = _i / D, d = _i % D, n = row / S, e = off(n) + row % S - 1;
          tokens(row, d) = row % S == 0 ? input_node(n, d) : e < off(n + 1) ? edge_cur(e, d) : Net(0);
        });

    // Split the attention output and take both residuals, one pass per side.
    // Edge: eps = attn + edge_in is the norm-mlp input (saved); the block's output
    // is eps + mlp(norm(eps)), accumulated onto a copy of eps in edge_next --
    // which, when nothing is saved, is eps itself. Node: the same, with
    // node_res = node_in + expansion(central row).
    View2D out_node128 = ws_.tmp(N, D);
    View2D eps = save ? keep(sav->eps[a], "s_eps_" + std::to_string(a) + "_", NM, D) : edge_next;
    {
      Workspace::Scope attn_scope(ws_);
      View2D attn_out = ws_.tmp(N * S, D);
      {
        Workspace::Scope qkv_scope(ws_);
        View2D attn_in = ws_.tmp(N * S, D);
        norm(attn_in, tokens, tl + ".norm_attention");
        View2D qkv = save && save_wide ? keep(sav->qkv[a], "s_qkv_" + std::to_string(a) + "_", N * S, 3 * D)
                                       : ws_.tmp(N * S, 3 * D);
        linear(qkv, attn_in, mat(tl + ".attention.input_linear.weight"),
               vec(tl + ".attention.input_linear.bias"));
        attention(ws_, "attn_" + as, attn_out, qkv, dev.cf_seq,
                  mat(tl + ".attention.output_linear.weight"),
                  vec(tl + ".attention.output_linear.bias"), N, S, h_.num_heads, h_.head_dim,
                  h_.attention_temperature, save);
      }
      Kokkos::parallel_for(
          "split_res", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
            const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
            if (row % S == 0) return (void) (out_node128(n, d) = attn_out(row, d));
            if (k >= off(n + 1)) return;  // padding
            eps(k, d) = attn_out(row, d) + edge_cur(k, d);
            if (save) edge_next(k, d) = eps(k, d);
          });
    }
    View2D node_exp = ws_.tmp(N, Dn);
    linear(node_exp, out_node128, mat(tl + ".center_expansion.weight"),
           vec(tl + ".center_expansion.bias"));
    View2D node_next = ws_.n2("node_new_" + pp, N, Dn);
    View2D node_res = save ? keep(sav->node_new[a], "s_nn_" + std::to_string(a) + "_", N, Dn) : node_next;
    Kokkos::parallel_for(
        "nres", RangePolicy(0, N * Dn), KOKKOS_LAMBDA(int _i) {
          const int n = _i / Dn, d = _i % Dn;
          node_res(n, d) = node_cur(n, d) + node_exp(n, d);
          if (save) node_next(n, d) = node_res(n, d);
        });

    const int cw = ffn_pre_width(tl + ".center_mlp"), ew = ffn_pre_width(tl + ".mlp");
    View2D node_norm = ws_.tmp(N, Dn);
    norm(node_norm, node_res, tl + ".norm_center_features");
    View2D tmp_center = save ? keep(sav->tmp_center[a], "s_tc_" + std::to_string(a) + "_", N, cw)
                             : ws_.tmp(N, cw);
    feedforward("cmlp_" + as, node_next, node_norm, tl + ".center_mlp", tmp_center, save, 1);

    {
      Workspace::Scope mlp_scope(ws_);
      View2D edge_norm = ws_.tmp(NM, D);
      norm(edge_norm, eps, tl + ".norm_mlp");
      View2D tmp_edge = save && save_wide ? keep(sav->tmp_edge[a], "s_te_" + std::to_string(a) + "_", NM, ew)
                                          : ws_.tmp(NM, ew);
      feedforward("emlp_" + as, edge_next, edge_norm, tl + ".mlp", tmp_edge, save, 1);
    }

    node_cur = node_next;
    edge_cur = edge_next;
  }

  // featurizer (feedforward) update
  if (cond.extent(0) == (std::size_t) N) {
    Kokkos::parallel_for(
        "cond_add", RangePolicy(0, N * Dn), KOKKOS_LAMBDA(int _i) {
          node_cur(_i / Dn, _i % Dn) += cond(_i / Dn, _i % Dn);
        });
  }
  node = node_cur;
  // input_edge += out_edge + comb_mlp(layernorm([out_edge | out_edge of the
  // reverse edge])). The concat is the norm's input (saved); input_edge takes
  // out_edge in the same pass and the MLP's last linear accumulates onto it.
  const View2D out_edge = edge_cur;
  View2D concat = save ? keep(sav->concat, "s_cc_", NM, 2 * D) : ws_.tmp(NM, 2 * D);
  Kokkos::parallel_for(
      "cat_rev", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
        const int k = _i / D, d = _i % D, r = rev(k);
        concat(k, d) = out_edge(k, d);
        concat(k, D + d) = (r >= 0) ? out_edge(r, d) : Net(0);
        input_edge(k, d) += out_edge(k, d);
      });
  View2D comb_norm = ws_.tmp(NM, 2 * D);
  norm_fwd(comb_norm, concat, vec("combination_norms." + ls + ".weight"),
           vec("combination_norms." + ls + ".bias"));
  View2D cph = ws_.tmp(NM, 2 * D);
  linear_silu(cph, save ? keep(sav->cph, "s_cph_", NM, 2 * D) : View2D(), comb_norm,
              mat("combination_mlps." + ls + ".0.weight"), vec("combination_mlps." + ls + ".0.bias"),
              save);
  linear(input_edge, cph, mat("combination_mlps." + ls + ".2.weight"),
         vec("combination_mlps." + ls + ".2.bias"), 1);
}

void PetModel::ff_layer_bwd(const DeviceEdgeData& dev, const PackedEdges& pk, int L,
                            const LayerSaves& sav, bool kept_wide, View2D node_adj,
                            View2D input_edge_adj, View2D edge_in4_adj, View2D cf_seq_adj) {
  const int N = dev.n_atoms, S = dev.max_neighbors + 1, NM = pk.E;  // NM: edge rows
  const int D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  auto rev = pk.reverse;
  auto off = pk.off;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  // Rebuild `lin(norm(in))` into scratch: a wide activation the forward did not keep.
  auto rebuild = [&](View2D in, const std::string& nkey, const std::string& lin) {
    View2D x = ws_.tmp(in.extent(0), in.extent(1));
    norm(x, in, nkey);
    View2D out = ws_.tmp(in.extent(0), mat(lin + ".weight").extent(0));
    linear(out, x, mat(lin + ".weight"), vec(lin + ".bias"));
    return out;
  };

  // node_adj and edge_adj are the adjoints of the current block's OUTPUTS, and
  // each step below turns them, in place, into the adjoints of its inputs --
  // every residual connection is then just "leave the value where it is".
  Workspace::Scope layer_scope(ws_);
  View2D edge_adj = ws_.tmp(NM, D);

  // input_edge_out = input_edge_in + out_edge + comb_mlp(layernorm(concat)):
  // input_edge_adj passes to input_edge_in unchanged, and is the MLP's out adj.
  // (Each brace below is a Workspace scope: its temporaries die at the brace.)
  {
    Workspace::Scope comb_scope(ws_);
    View2D cph_adj = ws_.tmp(NM, 2 * D);
    linear_bwd(cph_adj, input_edge_adj, mat("combination_mlps." + ls + ".2.weight"), 0);
    silu_bwd(cph_adj, sav.cph);
    View2D cnrm_adj = ws_.tmp(NM, 2 * D);
    linear_bwd(cnrm_adj, cph_adj, mat("combination_mlps." + ls + ".0.weight"), 0);
    View2D concat_adj = ws_.tmp(NM, 2 * D);
    pet::norm_bwd(concat_adj, cnrm_adj, sav.concat, vec("combination_norms." + ls + ".weight"), true, false);
    // out_edge: its direct term, its own concat slot, and the reversed slot of the
    // edge pointing back at it. The reverse map is an involution, so that edge is
    // reverse(k): a gather in a fixed order, never a (float-atomic) scatter.
    Kokkos::parallel_for(
        "bw_concat", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
          const int k = _i / D, d = _i % D, r = rev(k);
          edge_adj(k, d) = input_edge_adj(k, d) + concat_adj(k, d) + (r >= 0 ? concat_adj(r, D + d) : Net(0));
        });
  }

  for (int a = A - 1; a >= 0; --a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    const std::string as = sav.tag + "_" + std::to_string(a);

    // edge_out = eps + mlp(norm(eps)) and node_out = node_res + mlp(norm(node_res)):
    // afterwards edge_adj / node_adj hold the adjoints of eps / node_res.
    {
      Workspace::Scope mlp_scope(ws_);
      View2D enrm_adj = ws_.tmp(NM, D);
      feedforward_bwd("emlpb_" + as, enrm_adj, edge_adj, tl + ".mlp",
                      kept_wide ? sav.tmp_edge[a]
                                : rebuild(sav.eps[a], tl + ".norm_mlp", tl + ".mlp.w_in"),
                      0);
      norm_bwd(edge_adj, enrm_adj, sav.eps[a], tl + ".norm_mlp");
      View2D ncn_adj = ws_.tmp(N, Dn);
      feedforward_bwd("cmlpb_" + as, ncn_adj, node_adj, tl + ".center_mlp", sav.tmp_center[a], 0);
      norm_bwd(node_adj, ncn_adj, sav.node_new[a], tl + ".norm_center_features");
    }
    Workspace::Scope block_scope(ws_);

    // node_res = node_in + expansion(central row); eps = edge_in + edge rows. So
    // attention's output adjoint is the expansion's input adjoint on the central
    // row and eps's on the edge rows, and node_adj / edge_adj already hold the
    // residual shares of node_in / edge_in.
    View2D tokens_adj = ws_.tmp(N * S, D);
    {
      Workspace::Scope attn_scope(ws_);
      View2D out_node128_adj = ws_.tmp(N, D);
      linear_bwd(out_node128_adj, node_adj, mat(tl + ".center_expansion.weight"), 0);
      View2D ao_adj = ws_.tmp(N * S, D);
      Kokkos::parallel_for(
          "bw_ao", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
            const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
            ao_adj(row, d) = row % S == 0 ? out_node128_adj(n, d) : k < off(n + 1) ? edge_adj(k, d) : Net(0);
          });

      View2D attn_in_adj = ws_.tmp(N * S, D);
      attention_bwd(ws_, "attnb_" + as, "attn_" + as, attn_in_adj, cf_seq_adj, ao_adj,
                    kept_wide ? sav.qkv[a]
                              : rebuild(sav.tokens[a], tl + ".norm_attention",
                                        tl + ".attention.input_linear"),
                    dev.cf_seq, mat(tl + ".attention.input_linear.weight"),
                    mat(tl + ".attention.output_linear.weight"), N, S, h_.num_heads, h_.head_dim,
                    h_.attention_temperature, 0);
      norm_bwd(tokens_adj, attn_in_adj, sav.tokens[a], tl + ".norm_attention", false);
    }

    // tokens = [input_node ; edge_in]: edge rows add onto edge_in's residual
    // share, the central row feeds the contraction back onto node_in's.
    View2D input_node_adj = ws_.tmp(N, D);
    Kokkos::parallel_for(
        "bw_tok", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
          const int row = _i / D, d = _i % D, n = row / S, k = off(n) + row % S - 1;
          if (row % S == 0) input_node_adj(n, d) = tokens_adj(row, d);
          else if (k < off(n + 1)) edge_adj(k, d) += tokens_adj(row, d);
        });
    linear_bwd(node_adj, input_node_adj, mat(tl + ".center_contraction.weight"));
  }

  // et = compress.2(silu(compress.0(...))): edge_adj is now et's adjoint, and
  // node_adj that of the node features entering this layer.
  View2D cpre_adj = ws_.tmp(NM, D);
  linear_bwd(cpre_adj, edge_adj, mat(g + ".compress.2.weight"), 0);
  compress_bwd(cpre_adj, sav.cpre, compress_fold(L), edge_in4_adj, input_edge_adj);
}

DeviceOut PetModel::ff_pass(const DeviceEdgeData& dev, bool grad) {
  Workspace::Scope scope(ws_);  // every Workspace::tmp() is inside some Scope
  const int N = dev.n_atoms;
  const int M = dev.max_neighbors;
  const int S = M + 1;
  const int D = h_.d_pet;
  const int Dn = h_.d_node;
  const int Dh = h_.d_head;  // readout-head hidden width (not necessarily d_pet)
  const int G = h_.num_gnn_layers;
  const int A = h_.num_attention_layers;

  // Nothing here relies on the workspace zeroing what it hands out: every buffer's
  // first writer overwrites it (gemm beta=0, a norm, a full-coverage kernel), and
  // the few true accumulators are zeroed explicitly where they are taken. That
  // was ~90 memsets per evaluation, most of them full passes over an edge tensor.
  ws_.set_zero(false);

  // ---- device-resident NEF data (already uploaded / built on device) ----
  auto d_species = dev.species;
  auto d_cf_seq = dev.cf_seq;
  // Edge tensors are [NM, .] over the kept edges only (see PackedEdges).
  const PackedEdges pk = pack_edges(dev);
  const int NM = pk.E;

  // ---- initial embeddings ----
  View2D node = ws_.n2("node", N, Dn);
  View2D input_edge = ws_.n2("input_edge", NM, D);
  gather(node, mat("node_embedders.0.weight").v, d_species);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);

  // Charge / spin conditioning, computed once and reused by every GNN layer.
  // Empty (and free) for a model without it.
  View2D cond = conditioning(dev, N, dev.n_struct);

  // Readout saves. One set: this featurizer reads out once. Sized as vectors so
  // PetModel::readout is the same call for both featurizers.
  std::vector<View2D> sav_nh0(1), sav_nh1(1), sav_eh0(1), sav_eh1(1), sav_epred(1);

  // How much the backward keeps (see Recompute), estimated from the shapes:
  // what one layer saves, how much of that is the two widest activations, and
  // the working set every tier needs
  // (the peak of live temporaries plus the readout's saves and adjoints).
  const std::size_t NSz = std::size_t(N) * S, NMz = NM;
  const std::size_t ew = ffn_pre_width("gnn_layers.0.trans.layers.0.mlp");
  const std::size_t layer = sizeof(Net) * (A * (NSz * 5 * D + NMz * (ew + D)) + NMz * 5 * D);
  const std::size_t wide = sizeof(Net) * A * (NSz * 3 * D + NMz * ew);
  const std::size_t work = sizeof(Net) * NMz * (2 * ew + 6 * D + 4 * Dh);
  Recompute rc = recompute_;
  if (rc == Recompute::Auto) {
    auto fits = [&](std::size_t b) { return mem_budget_ == 0 || work + b <= mem_budget_; };
    rc = fits(G * layer)          ? Recompute::Never
         : fits(G * (layer - wide)) ? Recompute::Wide
                                    : Recompute::Layers;
  }
  const bool keep_wide = rc == Recompute::Never, ckpt = grad && rc == Recompute::Layers;

  // Per-layer saves, or under checkpointing only each layer's inputs.
  std::vector<LayerSaves> sav(grad && !ckpt ? G : 0);
  std::vector<std::pair<View2D, View2D>> inputs(ckpt ? G : 0);
  for (int L = 0; L < G; ++L) {
    if (ckpt) {
      inputs[L] = {ws_.n2("ck_node_" + std::to_string(L), N, Dn),
                   ws_.n2("ck_ie_" + std::to_string(L), NM, D)};
      Kokkos::deep_copy(ExecSpace(), inputs[L].first, node);
      Kokkos::deep_copy(ExecSpace(), inputs[L].second, input_edge);
    }
    if (!sav.empty()) sav[L].tag = std::to_string(L);
    ff_layer(dev, pk, L, node, input_edge, cond, sav.empty() ? nullptr : &sav[L], grad && keep_wide);
  }

  // ---- readout ----
  // Readout: the feedforward featurizer reads out once, from the final node and
  // edge features. Same heads as the residual path, one set instead of G.
  View1D per_atom_net = ws_.n1("per_atom_net", N);
  {
    std::vector<View2D> nfeat{node}, efeat{input_edge};
    readout(nfeat, efeat, per_atom_net, pk, N,
            sav_nh0, sav_nh1, sav_eh0, sav_eh1, sav_epred, "", grad);
  }

  // per-atom energy assembly (device): e = scale*net + composition[species]; total via reduce
  RView1D per_atom = ws_.r1("per_atom", N);  // double
  {
    const double scale = energy_scale_;
    auto comp = comp_view_;
    auto species = d_species;
    Kokkos::parallel_for(
        "assemble", RangePolicy(0, N), KOKKOS_LAMBDA(int n) {
          per_atom(n) = scale * per_atom_net(n) + comp(species(n));
        });
  }
  if (!grad) return {per_atom, {}, {}};

  // ==========================================================================
  // backward pass: adjoint of sum(per_atom_net) wrt edge vectors
  // ==========================================================================
  // shared, cross-layer accumulators
  View2D edge_in4_adj = ws_.n2("edge_in4_adj", NM, 4);
  View2D cf_seq_adj = ws_.n2("cf_seq_adj", N, S);
  Kokkos::deep_copy(ExecSpace(), edge_in4_adj, Net(0));
  Kokkos::deep_copy(ExecSpace(), cf_seq_adj, Net(0));

  // ---- readout backward ----
  // per_atom_net_adj = 1 ; node_pred_adj = 1 ; edge_pred_adj = cutoff
  View2D node_adj = ws_.n2("node_adj", N, Dn);      // adjoint of final node features
  View2D input_edge_adj = ws_.n2("ie_adj", NM, D);  // adjoint of final edge features (accumulator)
  View1D cutoff_adj = ws_.n1("cutoff_adj", NM);     // adjoint of cutoff_factor[k]
  {
    View2D node_pred_adj = ws_.n2("npa", N, 1);
    View2D edge_pred_adj = ws_.n2("epa", NM, 1);
    View2D epred = sav_epred[0];  // the readout's saved edge predictions
    auto cut = pk.cut;
    Kokkos::parallel_for(
        "ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { node_pred_adj(n, 0) = 1.0; });
    Kokkos::parallel_for(
        "ro_seed_e", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          edge_pred_adj(k, 0) = cut(k);
          cutoff_adj(k) = epred(k, 0);
        });
    // node head backward. The head's hidden width is d_head, NOT d_pet: these are
    // the adjoints of PetModel::readout's nh0/nh1, which are [N, d_head]. They were
    // sized [N, d_pet], which is the same number only because every model validated
    // so far has d_head == d_pet -- with an unequal pair the gemm would read the
    // weight past its own row and silu_bwd would walk off the saved activation.
    // (compute_residual's copy of this block already uses Dh.)
    View2D nh1_adj = ws_.n2("nh1_adj", N, Dh);
    linear_bwd(nh1_adj, node_pred_adj, mat("node_last_layers.energy.0.energy___0.weight"), 0);
    silu_bwd(nh1_adj, sav_nh1[0]);
    View2D nh0_adj = ws_.n2("nh0_adj", N, Dh);
    linear_bwd(nh0_adj, nh1_adj, mat("node_heads.energy.0.2.weight"), 0);
    silu_bwd(nh0_adj, sav_nh0[0]);
    linear_bwd(node_adj, nh0_adj, mat("node_heads.energy.0.0.weight"), 0);
    // edge head backward
    View2D eh1_adj = ws_.n2("eh1_adj", NM, Dh);
    linear_bwd(eh1_adj, edge_pred_adj, mat("edge_last_layers.energy.0.energy___0.weight"), 0);
    silu_bwd(eh1_adj, sav_eh1[0]);
    View2D eh0_adj = ws_.n2("eh0_adj", NM, Dh);
    linear_bwd(eh0_adj, eh1_adj, mat("edge_heads.energy.0.2.weight"), 0);
    silu_bwd(eh0_adj, sav_eh0[0]);
    linear_bwd(input_edge_adj, eh0_adj, mat("edge_heads.energy.0.0.weight"), 0);
  }

  // ---- layers backward (reverse order) ----
  for (int L = G - 1; L >= 0; --L) {
    LayerSaves one;
    Workspace::Scope layer_scope(ws_);
    if (ckpt) {
      // Re-run this layer's forward from its checkpointed inputs, saving into one
      // shared set of buffers.
      View2D n = ws_.tmp(N, Dn), ie = ws_.tmp(NM, D);
      Kokkos::deep_copy(ExecSpace(), n, inputs[L].first);
      Kokkos::deep_copy(ExecSpace(), ie, inputs[L].second);
      one.tag = "c";
      ff_layer(dev, pk, L, n, ie, cond, &one, false);
    }
    ff_layer_bwd(dev, pk, L, ckpt ? one : sav[L], keep_wide, node_adj, input_edge_adj, edge_in4_adj,
                 cf_seq_adj);
  }

  // ---- geometry backward: edge_in4_adj + cutoff path -> edge vectors ----
  // total cutoff_factor adjoint = readout term + attention bias term (cols 1..M)
  // (cf_seq_adj col 0 is the constant central token -> ignored)
  const bool do_adapt = h_.adaptive();
  RView1D adapted_adj = ws_.r1("adapted_adj", N);  // dE/d(adapted_cutoff[a]) (double)
  RView2D edge_grad = ws_.r2("edge_grad", NM, 3);  // dE/dv per kept edge (double)
  RView1D pc_adj_e = ws_.r1("pc_adj_e", NM);       // per-edge pair-cutoff adjoint (adaptive)
  {
    const bool is_bump = (h_.cutoff_function == CutoffFunction::Bump);
    const double width = h_.cutoff_width;
    auto off = pk.off, center = pk.center, rev = pk.reverse;
    auto vec = pk.vec;
    auto dist = pk.dist, pcut = pk.pcut;
    Kokkos::parallel_for(
        "geom_bwd", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          const int n = center(k), m = k - off(n);
          const double cutoff_total = cutoff_adj(k) + cf_seq_adj(n, 1 + m);
          const double dcut_dd = cutoff_ddist(dist(k), pcut(k), width, is_bump);
          // direct dependence of cutoff_factor on its own edge distance (rc fixed)
          const double dist_adj = edge_in4_adj(k, 3) + cutoff_total * dcut_dd;
          const double invd = (dist(k) > 0.0) ? 1.0 / dist(k) : 0.0;
          for (int c = 0; c < 3; ++c)
            edge_grad(k, c) = edge_in4_adj(k, c) + dist_adj * vec(k, c) * invd;
          // adaptive path: d cutoff_factor / d pair_cutoff = -d cutoff_factor / d dist.
          // Parked per edge here and gathered per atom below, rather than scattered
          // with atomics -- adapted_adj feeds the adaptive cutoff, whose value
          // decides a discrete keep/drop, so its rounding is worth pinning down.
          if (do_adapt) pc_adj_e(k) = -cutoff_total * dcut_dd;
        });
    if (do_adapt) {
      // atom a's share is half of each of its own edges plus half of each edge
      // pointing at it -- and the latter are exactly the reverses of the former.
      Kokkos::parallel_for(
          "adapted_adj_gather", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            double s = 0.0;
            for (int k = off(a); k < off(a + 1); ++k) {
              s += 0.5 * pc_adj_e(k);
              if (rev(k) >= 0) s += 0.5 * pc_adj_e(rev(k));
            }
            adapted_adj(a) = s;
          });
    }
  }

  // scatter per-edge gradient to atoms (device): F[i]+=g, F[j]-=g, scaled by energy_scale
  // and accumulate the virial W_ab = sum_edges v_a * (scale*dE/dv)_b.
  RView2D d_forces = ws_.r2("forces", N, 3);  // double; fold_edge_gradients writes it first
  const int NS = dev.n_struct;             // 1 = single structure; >1 = batched
  auto sid = dev.struct_id;                // [N] owning structure (valid when NS>1)
  RView2D dvir = ws_.r2("virial9", NS, 9);  // per-structure virial (scatter + adaptive)
  {
    fold_edge_gradients(ws_, "ff", edge_grad, pk, d_forces, dvir, sid, N, NS, energy_scale_);
  }

  // adaptive-cutoff chain rule: propagate adapted_adj to all raw edges
  if (do_adapt) {
    const int E = dev.n_raw;
    auto d_rc = dev.raw_center;
    auto d_rj = dev.raw_neigh;
    auto d_rd = dev.raw_dist;
    auto d_rv = dev.raw_vec;

    // probe grid: precomputed once (constant given the hypers)
    // cutoff_width_adaptive, not cutoff_width: this differentiates the SMOOTHED
    // NEIGHBOUR COUNT, whose taper is the adaptive one. (The edge cutoff
    // factor's own width is used in geom_bwd above, and stays cutoff_width.)
    const bool solver = (h_.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver);
    adaptive_backward(ws_, d_forces, dvir, adapted_adj, d_rc, d_rj, d_rd, d_rv, E, probes_, n_probes_,
                      h_.cutoff_width_adaptive, h_.num_neighbors_adaptive, energy_scale_, N,
                      h_.cutoff_function == CutoffFunction::Bump, sid, NS,
                      dev.raw_off, dev.raw_reverse, solver, dev.adapt_r, dev.adapt_dn,
                      dev.adapt_eff);
  }

  return {per_atom, d_forces, dvir};
}

// One evaluation: the device pass for this featurizer -- through the graph cache,
// which replays it as one launch when this exact evaluation has been seen before
// -- then the results off the device.
EnergyResult PetModel::compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                               RView2D* dev_forces, RView1D* dev_per_atom, RView2D* dev_virial) {
  last_n_atoms_ = dev.n_atoms;
  last_max_neighbors_ = dev.max_neighbors;
  peak_max_neighbors_ = std::max(peak_max_neighbors_, dev.max_neighbors);
  peak_edge_slots_ = std::max(peak_edge_slots_, (long) dev.n_atoms * std::max(1, dev.max_neighbors));
  const bool grad = host_forces || dev_forces;
  // The kept-edge count sizes the packed edge tensors, so the pass needs it on the
  // host; the builders supply it, and anything else gets it counted here, once,
  // outside anything a graph records.
  DeviceEdgeData d = dev;
  if (d.n_edges < 0) {
    auto mask = d.mask;
    Kokkos::parallel_reduce("pk_count", RangePolicy(0, d.n_atoms * d.max_neighbors),
                            KOKKOS_LAMBDA(int k, int& c) { c += mask(k) > 0.0; }, d.n_edges);
  }
  auto pass = [&] {
    return h_.featurizer_type == FeaturizerType::Residual ? residual_pass(d, grad) : ff_pass(d, grad);
  };
  const DeviceOut out = graphs_ && !ozaki_active() ? graph_.run(graph_key(d, grad), pass) : pass();
  const int N = dev.n_atoms, NS = dev.n_struct;

  EnergyResult res;
  // The batched device path segments per_atom on the device and never reads
  // this EnergyResult, so it gets no host copy (which would be a device sync).
  if (dev_per_atom) *dev_per_atom = out.per_atom;
  else {
    auto h_pa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.per_atom);
    res.per_atom.assign(h_pa.data(), h_pa.data() + N);
    for (double e : res.per_atom) res.total += e;
  }
  if (!grad) return res;

  // The symmetric virial in Voigt order [xx, yy, zz, xy, xz, yz]: per structure
  // on the device for a batched caller, otherwise the one structure's on the host.
  const RView2D w9 = out.vir9;
  if (dev_virial) {
    RView2D bvir = ws_.r2("batch_virial", NS, 6);  // pooled; fully overwritten
    Kokkos::parallel_for(
        "virial_voigt", RangePolicy(0, NS), KOKKOS_LAMBDA(int b) {
          bvir(b, 0) = w9(b, 0), bvir(b, 1) = w9(b, 4), bvir(b, 2) = w9(b, 8);
          bvir(b, 3) = 0.5 * (w9(b, 1) + w9(b, 3));
          bvir(b, 4) = 0.5 * (w9(b, 2) + w9(b, 6));
          bvir(b, 5) = 0.5 * (w9(b, 5) + w9(b, 7));
        });
    *dev_virial = bvir;
  } else {
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w9);
    const double v[6] = {w(0, 0), w(0, 4), w(0, 8), 0.5 * (w(0, 1) + w(0, 3)),
                         0.5 * (w(0, 2) + w(0, 6)), 0.5 * (w(0, 5) + w(0, 7))};
    std::copy(v, v + 6, res.virial);
  }
  if (dev_forces) *dev_forces = out.forces;
  if (host_forces) {
    auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.forces);
    host_forces->assign(h_f.data(), h_f.data() + std::size_t(N) * 3);
  }
  return res;
}

// Everything the recorded device work depends on that is not a model constant:
// the shapes, where each input lives, and the workspace's allocation generation
// (a replay must never touch a buffer the pool has since reallocated).
std::vector<std::uintptr_t> PetModel::graph_key(const DeviceEdgeData& d, bool grad) const {
  std::vector<std::uintptr_t> k{(std::uintptr_t) grad, (std::uintptr_t) d.n_atoms,
                                (std::uintptr_t) d.max_neighbors, (std::uintptr_t) d.n_raw,
                                (std::uintptr_t) d.n_struct, (std::uintptr_t) d.n_edges,
                                ws_.generation()};
  auto add = [&k](const auto& v) { k.push_back((std::uintptr_t) v.data()), k.push_back(v.size()); };
  add(d.species), add(d.neigh_species), add(d.reverse_index), add(d.edge_vec), add(d.dist);
  add(d.mask), add(d.pair_cutoff), add(d.cutoff_factor), add(d.cf_seq), add(d.raw_center);
  add(d.raw_neigh), add(d.raw_dist), add(d.raw_vec), add(d.raw_off), add(d.raw_reverse);
  add(d.adapt_eff), add(d.adapt_r), add(d.adapt_dn), add(d.struct_id), add(d.charge);
  add(d.spin_multiplicity);
  return k;
}

}  // namespace pet
