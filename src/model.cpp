#include "pet/model.hpp"

#include "pet/gemm.hpp"

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

// out(R,Dout) = in(R,Din) @ W(Dout,Din)^T + b(Dout)
void linear(View2D out, View2D in, View2D W, View1D b) {
  gemm('N', 'T', (Net)1.0, in, W, (Net)0.0, out);  // out = in @ W^T
  const int R = out.extent(0), Dout = out.extent(1);
  Kokkos::parallel_for(
      "bias", RangePolicy(0, (R) * (Dout)),
      KOKKOS_LAMBDA(int _i) { const int r = _i / (Dout), o = _i % (Dout); out(r, o) += b(o); });
}

// Fused linear + bias + (optional save of pre-activation) + SiLU. Replaces the
// linear / copy_into / silu_inplace triple (3 elementwise passes) with the gemm
// plus a single fused epilogue kernel. `sav` (when save) receives the pre-SiLU
// value the analytic backward needs; pass an empty View when !save.
void linear_silu(View2D out, View2D sav, View2D in, View2D W, View1D b, bool save) {
  gemm('N', 'T', (Net)1.0, in, W, (Net)0.0, out);  // out = in @ W^T
  const int R = out.extent(0), Dout = out.extent(1);
  Kokkos::parallel_for(
      "linear_silu", RangePolicy(0, (R) * (Dout)),
      KOKKOS_LAMBDA(int _i) { const int r = _i / (Dout), o = _i % (Dout);
        const Net z = out(r, o) + b(o);
        if (save) sav(r, o) = z;
        out(r, o) = silud(z);
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

// Warp-per-row norms: one team (1 thread x 32 vector lanes = a warp) per row,
// ThreadVectorRange over D -> coalesced reads (consecutive lanes = consecutive
// memory in LayoutRight) + a warp-shuffle reduction, vs the strided one-thread-
// per-row pattern.
constexpr int NORM_VEC = 32;
void rmsnorm(View2D out, View2D in, View1D weight) {
  const int R = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      "rmsnorm", TeamPol(R, 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        double ms = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, double& s) { s += (double) in(r, d) * (double) in(r, d); }, ms);
        const double inv = 1.0 / Kokkos::sqrt(ms / D + RMSNORM_EPS);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, D),
                             [&](int d) { out(r, d) = in(r, d) * inv * weight(d); });
      });
}

void layernorm(View2D out, View2D in, View1D weight, View1D bias) {
  const int R = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      "layernorm", TeamPol(R, 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        // mean and variance in a single warp reduction: E[x] and E[x^2] together,
        // var = E[x^2] - mu^2 (one reduction instead of two sequential ones).
        double sx = 0.0, sx2 = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, double& a, double& b) {
              const double v = (double) in(r, d);
              a += v;
              b += v * v;
            },
            sx, sx2);
        const double mu = sx / D;
        const double inv = 1.0 / Kokkos::sqrt(sx2 / D - mu * mu + LAYERNORM_EPS);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, D), [&](int d) {
          out(r, d) = (in(r, d) - mu) * inv * weight(d) + bias(d);
        });
      });
}

// SwiGLU FF: out = w_out( v * sigmoid(g) ), [v,g]=w_in(in).chunk(2). tmp holds w_in(in).
void feedforward_swiglu(Workspace& ws, const std::string& key, View2D out, View2D in, View2D w_in,
                        View1D b_in, View2D w_out, View1D b_out, View2D tmp) {
  const int R = in.extent(0);
  const int dff = tmp.extent(1) / 2;
  linear(tmp, in, w_in, b_in);
  View2D h = ws.n2(key + ":ffh", R, dff);
  {
    Kokkos::parallel_for(
        "swiglu", RangePolicy(0, (R) * (dff)),
        KOKKOS_LAMBDA(int _i) { const int r = _i / (dff), c = _i % (dff); h(r, c) = tmp(r, c) * sigmoidd(tmp(r, dff + c)); });
  }
  linear(out, h, w_out, b_out);
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
template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D attn_out, View2D qkv,
                    View2D cf_seq, View2D w_out, View1D b_out, int N, int S, int num_heads,
                    int head_dim, double temperature, bool save) {
  const int D = num_heads * head_dim, H = num_heads;
  View2D merged = ws.n2(key + ":merged", N * S, D);
  View2D sml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  const double scale = 1.0 / (Kokkos::sqrt((double) head_dim) * temperature);
  constexpr int CAP = HD > 0 ? HD : 64;
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
// Dispatch on the runtime head_dim to a compile-time-sized instantiation.
#define PET_ATTN_DISPATCH(FN, ...)                                  \
  switch (head_dim) {                                               \
    case 16: FN<16>(__VA_ARGS__); break;                            \
    case 32: FN<32>(__VA_ARGS__); break;                            \
    case 64: FN<64>(__VA_ARGS__); break;                            \
    default: FN<0>(__VA_ARGS__); break;                             \
  }
void attention(Workspace& ws, const std::string& key, View2D attn_out, View2D qkv, View2D cf_seq,
               View2D w_out, View1D b_out, int N, int S, int num_heads, int head_dim,
               double temperature, bool save) {
  PET_ATTN_DISPATCH(attention_impl, ws, key, attn_out, qkv, cf_seq, w_out, b_out, N, S, num_heads,
                    head_dim, temperature, save);
}

// ----------------------------------------------------------------------------
// backward helpers (adjoints accumulate with +=; fresh Views are zero-init)
// ----------------------------------------------------------------------------

// in_adj(R,Din) += out_adj(R,Dout) @ W(Dout,Din)
void linear_bwd(View2D in_adj, View2D out_adj, View2D W) {
  gemm('N', 'N', (Net)1.0, out_adj, W, (Net)1.0, in_adj);  // accumulate
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

void rmsnorm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight) {
  const int R = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      "rmsnorm_bwd", TeamPol(R, 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        // sum(in^2) and sum(out_adj*weight*in) are independent -> one warp reduction.
        double ms = 0.0, s = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, double& a, double& b) {
              const double x = (double) in(r, d);
              a += x * x;
              b += (double) out_adj(r, d) * weight(d) * x;
            },
            ms, s);
        const double inv = 1.0 / Kokkos::sqrt(ms / D + RMSNORM_EPS);
        const double coef = inv * inv * inv / D * s;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, D), [&](int d) {
          in_adj(r, d) += inv * out_adj(r, d) * weight(d) - coef * in(r, d);
        });
      });
}

void layernorm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight) {
  const int R = in.extent(0), D = in.extent(1);
  Kokkos::parallel_for(
      "layernorm_bwd", TeamPol(R, 1, NORM_VEC), KOKKOS_LAMBDA(const TeamMem& team) {
        const int r = team.league_rank();
        // mean + variance in one reduction (E[x], E[x^2]); then the two gradient
        // sums sg, sgx in one more -> two warp reductions instead of four.
        double sx = 0.0, sx2 = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, double& a, double& b) {
              const double v = (double) in(r, d);
              a += v;
              b += v * v;
            },
            sx, sx2);
        const double mu = sx / D;
        const double inv = 1.0 / Kokkos::sqrt(sx2 / D - mu * mu + LAYERNORM_EPS);
        double sg = 0.0, sgx = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, D),
            [&](int d, double& a, double& b) {
              const double gw = (double) out_adj(r, d) * weight(d);
              a += gw;
              b += gw * ((double) in(r, d) - mu) * inv;
            },
            sg, sgx);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, D), [&](int d) {
          const double gd = out_adj(r, d) * weight(d);
          const double xhat = (in(r, d) - mu) * inv;
          in_adj(r, d) += inv * (gd - sg / D - xhat * sgx / D);
        });
      });
}

// in_adj(R,Dmodel) += backward of SwiGLU FF. tmp holds w_in(in) from forward.
void feedforward_swiglu_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D out_adj,
                            View2D tmp, View2D w_in, View2D w_out) {
  const int R = out_adj.extent(0);
  const int dff = tmp.extent(1) / 2;
  View2D h_adj = ws.n2(key + ":hadj", R, dff);
  linear_bwd(h_adj, out_adj, w_out);  // h_adj = out_adj @ w_out
  View2D tmp_adj = ws.n2(key + ":tmpadj", R, 2 * dff);
  {
    Kokkos::parallel_for(
        "swiglu_bwd", RangePolicy(0, (R) * (dff)),
        KOKKOS_LAMBDA(int _i) { const int r = _i / (dff), c = _i % (dff);
          const double v = tmp(r, c), g = tmp(r, dff + c);
          const double sg = sigmoidd(g);
          tmp_adj(r, c) = h_adj(r, c) * sg;
          tmp_adj(r, dff + c) = h_adj(r, c) * v * sg * (1.0 - sg);
        });
  }
  linear_bwd(in_adj, tmp_adj, w_in);
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
                        View2D w_in, View2D w_out, int N, int S, int num_heads, int head_dim,
                        double temperature) {
  const int D = num_heads * head_dim;
  const int R = N * S;
  const int H = num_heads;
  View2D merged_adj = ws.n2(key + ":madj", R, D);
  linear_bwd(merged_adj, ao_adj, w_out);  // d(merged) = ao_adj @ w_out
  View2D qkv_adj = ws.n2(key + ":qkvadj", R, 3 * D);  // each element written by exactly one thread
  // softmax math in Net precision (expf/logf): much faster than fp64 on consumer GPUs.
  View2D stats = ws.n2(key + ":stats", N * H * S, 3);  // per-query (m, 1/l, dot_do_out)
  View2D cfh = ws.n2(key + ":cfh", N * S, H);          // per-(key, head) cutoff-adjoint partial
  View2D merged_saved = ws.peek2(fwd_key + ":merged");  // forward out_sq (no recompute)
  View2D sml = ws.peek2(fwd_key + ":ml");               // forward per-query (m, 1/l)
  const Net sc = (Net) (1.0 / (Kokkos::sqrt((double) head_dim) * temperature));
  constexpr int CAP = HD > 0 ? HD : 64;  // right-size the per-thread arrays (no scratch spill)

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
          // Padding query (invl==0 from the forward): its qkv_adj row and stats
          // stay zero-initialized, so kernel B reads invl==0 and skips it too.
          if (invl <= Net(0)) return;
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
          // padding key: no query attends -> dK=dV=0 (zero-init)
          if (cf <= Net(0)) { cfh(row_k, hh) = Net(0); return; }
          const Net logcf = fast_log(cf);
          const int qoff = hh * head_dim, koff = D + hh * head_dim, voff = 2 * D + hh * head_dim;
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
  linear_bwd(an_adj, qkv_adj, w_in);  // d(attn_in) = qkv_adj @ w_in
}
void attention_bwd(Workspace& ws, const std::string& key, const std::string& fwd_key, View2D an_adj,
                   View2D cf_seq_adj, View2D ao_adj, View2D qkv, View2D cf_seq, View2D w_in,
                   View2D w_out, int N, int S, int num_heads, int head_dim, double temperature) {
  PET_ATTN_DISPATCH(attention_bwd_impl, ws, key, fwd_key, an_adj, cf_seq_adj, ao_adj, qkv, cf_seq,
                    w_in, w_out, N, S, num_heads, head_dim, temperature);
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
                         RView2D d_edge_vec, RView1D mask, IView1D d_reverse,
                         RView2D d_forces, RView2D dvir, IView1D sid,
                         int N, int M, int NS, double scale) {
  RView2D vir_atom = ws.r2(key + ":vir_atom", N, 9);
  Kokkos::parallel_for(
      key + ":gather_force", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
        double f[3] = {0.0, 0.0, 0.0};
        double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int s = 0; s < M; ++s) {
          const int k = i * M + s;
          if (mask(k) <= 0.0) continue;
          for (int c = 0; c < 3; ++c) f[c] += edge_grad(k, c);
          for (int a = 0; a < 3; ++a)  // this edge's own virial share v_a * g_b
            for (int b = 0; b < 3; ++b) w[a * 3 + b] += d_edge_vec(k, a) * edge_grad(k, b);
          const int r = d_reverse(k);
          if (r >= 0)
            for (int c = 0; c < 3; ++c) f[c] -= edge_grad(r, c);
        }
        for (int c = 0; c < 3; ++c) d_forces(i, c) += scale * f[c];
        for (int t = 0; t < 9; ++t) vir_atom(i, t) = scale * w[t];
      });
  // Atoms of a structure are contiguous in the batch, so a count plus a prefix sum
  // gives each structure its range and the sum runs in atom-index order. Integer
  // counts are order-independent in value, so the count itself is safe.
  IView1D scnt = ws.i1(key + ":vir_scnt", NS);  // zeroed on reuse
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
        dvir(b, t) += s;
      });
}

// per_atom_net(n) += node_pred(n) + sum over n's edges of mask*cutoff*edge_pred.
// Free-standing because nvcc will not take an extended lambda inside a private
// member, and PetModel::readout is one.
void readout_accumulate(View1D per_atom_net, View2D node_pred, View2D edge_pred, RView1D mask,
                        View1D cutoff, int N, int M, bool zero_first) {
  Kokkos::parallel_for(
      "readout", RangePolicy(0, N), KOKKOS_LAMBDA(int n) {
        // Accumulated in double, not in the network type. Summing M edge terms in
        // fp32 costs about 1e-8 relative on the per-atom energy -- small, but the
        // two paths disagreed on it (the feedforward one used double, the residual
        // one did not), and this is the more accurate of the two.
        double e = node_pred(n, 0);
        for (int m = 0; m < M; ++m) {
          const int k = n * M + m;
          e += (double) mask(k) * (double) cutoff(k) * (double) edge_pred(k, 0);
        }
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
    auto qe = mat("system_conditioning.charge_embedding.weight");
    auto se = mat("system_conditioning.spin_multiplicity_embedding.weight");
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
  if (h_.normalization == Normalization::LayerNorm)
    layernorm(out, in, vec(key + ".weight"), vec(key + ".bias"));
  else
    rmsnorm(out, in, vec(key + ".weight"));
}

void PetModel::norm_bwd(View2D in_adj, View2D out_adj, View2D in, const std::string& key) const {
  // Both backwards take the same arguments: the bias is a pure shift, so it does
  // not enter the input adjoint.
  if (h_.normalization == Normalization::LayerNorm)
    layernorm_bwd(in_adj, out_adj, in, vec(key + ".weight"));
  else
    rmsnorm_bwd(in_adj, out_adj, in, vec(key + ".weight"));
}

int PetModel::ffn_pre_width(const std::string& wkey) const {
  // w_in emits 2*dff for SwiGLU (value|gate) and dff otherwise; the loaded weight
  // already has the right shape, so read it rather than deriving it from hypers.
  return mat(wkey + ".w_in.weight").extent(0);
}

void PetModel::feedforward(const std::string& key, View2D out, View2D in,
                           const std::string& wkey, View2D pre, bool save) {
  const int R = in.extent(0);
  View2D w_in = mat(wkey + ".w_in.weight"), w_out = mat(wkey + ".w_out.weight");
  View1D b_in = vec(wkey + ".w_in.bias"), b_out = vec(wkey + ".w_out.bias");
  // Delegated rather than open-coded: nvcc refuses an extended __host__ __device__
  // lambda inside a private member function, and these dispatchers are private.
  if (h_.activation == Activation::SwiGLU) {
    feedforward_swiglu(ws_, key, out, in, w_in, b_in, w_out, b_out, pre);
  } else {
    const int dff = pre.extent(1);
    View2D h = ws_.n2(key + ":ffh", R, dff);
    linear_silu(h, save ? pre : View2D(), in, w_in, b_in, save);  // h = silu(pre)
    linear(out, h, w_out, b_out);
  }
}

void PetModel::feedforward_bwd(const std::string& key, View2D in_adj, View2D out_adj,
                               const std::string& wkey, View2D pre) {
  View2D w_in = mat(wkey + ".w_in.weight"), w_out = mat(wkey + ".w_out.weight");
  const int R = out_adj.extent(0);
  if (h_.activation == Activation::SwiGLU) {
    feedforward_swiglu_bwd(ws_, key, in_adj, out_adj, pre, w_in, w_out);
  } else {
    const int dff = pre.extent(1);
    View2D h_adj = ws_.n2(key + ":hadj", R, dff);
    linear_bwd(h_adj, out_adj, w_out);
    silu_bwd(h_adj, pre);
    linear_bwd(in_adj, h_adj, w_in);
  }
}

void PetModel::readout(const std::vector<View2D>& node_feat, const std::vector<View2D>& edge_feat,
                       View1D per_atom_net, RView1D d_mask, View1D d_cutoff, int N, int M,
                       std::vector<View2D>& sav_nh0, std::vector<View2D>& sav_nh1,
                       std::vector<View2D>& sav_eh0, std::vector<View2D>& sav_eh1,
                       std::vector<View2D>& sav_epred, const std::string& key, bool grad) {
  const int R = static_cast<int>(node_feat.size());
  const int Dh = h_.d_head;
  const int NM = N * M;
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

    readout_accumulate(per_atom_net, node_pred, edge_pred, d_mask, d_cutoff, N, M,
                       /*zero_first=*/i == 0);
  }
}

const View2D& PetModel::mat(const std::string& name) const {
  auto it = mat_.find(name);
  if (it == mat_.end()) throw std::runtime_error("PetModel: missing matrix '" + name + "'");
  return it->second;
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

  auto load_mat = [&](const std::string& n) {
    const Tensor& t = st.at(n);
    if (t.ndim() != 2) throw std::runtime_error("expected 2D tensor '" + n + "'");
    View2D v(n, t.dim(0), t.dim(1));
    auto hv = Kokkos::create_mirror_view(v);
    for (int64_t a = 0; a < t.dim(0); ++a)
      for (int64_t b = 0; b < t.dim(1); ++b) hv(a, b) = t.data[a * t.dim(1) + b];
    Kokkos::deep_copy(v, hv);
    mat_.emplace(n, v);
  };
  auto load_vec = [&](const std::string& n) {
    const Tensor& t = st.at(n);
    View1D v(n, t.numel());
    auto hv = Kokkos::create_mirror_view(v);
    for (int64_t a = 0; a < t.numel(); ++a) hv(a) = t.data[a];
    Kokkos::deep_copy(v, hv);
    vec_.emplace(n, v);
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
    load_linear(g + ".edge_embedder");
    load_linear(g + ".compress.0");
    load_linear(g + ".compress.2");
    if (st.has(g + ".neighbor_embedder.weight")) load_mat(g + ".neighbor_embedder.weight");
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
  if (h_.head_dim > 64) throw std::runtime_error("PetModel: head_dim>64 not supported");

  // device mirror of composition energies (double)
  comp_view_ = RView1D("composition", composition_.size());
  auto hcomp = Kokkos::create_mirror_view(comp_view_);
  for (std::size_t i = 0; i < composition_.size(); ++i) hcomp(i) = composition_[i];
  Kokkos::deep_copy(comp_view_, hcomp);
}

// Marshal a host EdgeData into device Views. This is the single host->device
// boundary for the host-neighbor-list path; compute() itself is device-native
// and never touches the host EdgeData.
DeviceEdgeData PetModel::upload_edge_data(const EdgeData& ed) const {
  const int N = ed.n_atoms;
  const int M = ed.max_neighbors;
  const int S = M + 1;
  const int NM = N * M;
  const int E = ed.n_raw;

  DeviceEdgeData dev;
  dev.n_atoms = N;
  dev.max_neighbors = M;
  dev.n_raw = E;
  dev.species = IView1D("species", N);
  dev.neigh_species = IView1D("neigh_species", NM);
  dev.reverse_index = IView1D("reverse", NM);
  dev.edge_vec = RView2D("edge_vec", NM, 3);
  dev.dist = RView1D("dist", NM);
  dev.mask = RView1D("mask", NM);
  dev.pair_cutoff = RView1D("pcut", NM);
  dev.cutoff_factor = View1D("cutoff", NM);
  dev.cf_seq = View2D("cf_seq", N, S);

  // Per-structure electronic state, for a conditioned model.
  if (!ed.charge.empty()) {
    const int B = (int) ed.charge.size();
    dev.charge = IView1D("charge", B);
    dev.spin_multiplicity = IView1D("spin", B);
    auto h_q = Kokkos::create_mirror_view(dev.charge);
    auto h_s = Kokkos::create_mirror_view(dev.spin_multiplicity);
    for (int b = 0; b < B; ++b) {
      h_q(b) = ed.charge[b];
      h_s(b) = (b < (int) ed.spin_multiplicity.size()) ? ed.spin_multiplicity[b] : 1;
    }
    Kokkos::deep_copy(dev.charge, h_q);
    Kokkos::deep_copy(dev.spin_multiplicity, h_s);
  }

  // Solver adaptive cutoff: the forward's root and slope, uploaded alongside
  // everything else. Without these the backward has nothing to differentiate
  // through -- it cannot recompute them, because it has no per-atom segmentation
  // of the raw edge list on this path.
  if (!ed.adapt_r.empty() && (int) ed.adapt_r.size() == N) {
    dev.adapt_r = RView1D("adapt_r", N);
    dev.adapt_dn = RView1D("adapt_dn", N);
    auto h_ar = Kokkos::create_mirror_view(dev.adapt_r);
    auto h_adn = Kokkos::create_mirror_view(dev.adapt_dn);
    for (int i = 0; i < N; ++i) {
      h_ar(i) = ed.adapt_r[i];
      h_adn(i) = ed.adapt_dn[i];
    }
    Kokkos::deep_copy(dev.adapt_r, h_ar);
    Kokkos::deep_copy(dev.adapt_dn, h_adn);
  }

  auto h_species = Kokkos::create_mirror_view(dev.species);
  for (int i = 0; i < N; ++i) h_species(i) = ed.species[i];
  Kokkos::deep_copy(dev.species, h_species);

  auto h_ns = Kokkos::create_mirror_view(dev.neigh_species);
  auto h_rev = Kokkos::create_mirror_view(dev.reverse_index);
  auto h_ev = Kokkos::create_mirror_view(dev.edge_vec);
  auto h_dist = Kokkos::create_mirror_view(dev.dist);
  auto h_cut = Kokkos::create_mirror_view(dev.cutoff_factor);
  auto h_mask = Kokkos::create_mirror_view(dev.mask);
  auto h_pc = Kokkos::create_mirror_view(dev.pair_cutoff);
  for (int k = 0; k < NM; ++k) {
    h_ns(k) = ed.neigh_species[k];
    h_rev(k) = ed.reverse_index[k];
    h_ev(k, 0) = ed.edge_vec[3 * k + 0];
    h_ev(k, 1) = ed.edge_vec[3 * k + 1];
    h_ev(k, 2) = ed.edge_vec[3 * k + 2];
    h_dist(k) = ed.edge_dist[k];
    h_cut(k) = ed.cutoff_factor[k];
    h_mask(k) = ed.mask[k] ? 1.0 : 0.0;
    h_pc(k) = ed.pair_cutoff[k];
  }
  Kokkos::deep_copy(dev.neigh_species, h_ns);
  Kokkos::deep_copy(dev.reverse_index, h_rev);
  Kokkos::deep_copy(dev.edge_vec, h_ev);
  Kokkos::deep_copy(dev.dist, h_dist);
  Kokkos::deep_copy(dev.cutoff_factor, h_cut);
  Kokkos::deep_copy(dev.mask, h_mask);
  Kokkos::deep_copy(dev.pair_cutoff, h_pc);

  auto h_cf = Kokkos::create_mirror_view(dev.cf_seq);
  for (int n = 0; n < N; ++n) {
    h_cf(n, 0) = 1.0;
    for (int m = 0; m < M; ++m) h_cf(n, 1 + m) = ed.cutoff_factor[n * M + m];
  }
  Kokkos::deep_copy(dev.cf_seq, h_cf);

  // raw edges (for the adaptive-cutoff backward)
  dev.raw_center = IView1D("raw_center", E);
  dev.raw_neigh = IView1D("raw_neigh", E);
  dev.raw_dist = RView1D("raw_dist", E);
  dev.raw_vec = RView2D("raw_vec", E, 3);
  if (E > 0) {
    auto h_rc = Kokkos::create_mirror_view(dev.raw_center);
    auto h_rj = Kokkos::create_mirror_view(dev.raw_neigh);
    auto h_rd = Kokkos::create_mirror_view(dev.raw_dist);
    auto h_rv = Kokkos::create_mirror_view(dev.raw_vec);
    for (int e = 0; e < E; ++e) {
      h_rc(e) = ed.raw_center[e];
      h_rj(e) = ed.raw_neigh[e];
      h_rd(e) = ed.raw_dist[e];
      h_rv(e, 0) = ed.raw_vec[3 * e + 0];
      h_rv(e, 1) = ed.raw_vec[3 * e + 1];
      h_rv(e, 2) = ed.raw_vec[3 * e + 2];
    }
    Kokkos::deep_copy(dev.raw_center, h_rc);
    Kokkos::deep_copy(dev.raw_neigh, h_rj);
    Kokkos::deep_copy(dev.raw_dist, h_rd);
    Kokkos::deep_copy(dev.raw_vec, h_rv);

    // Per-centre ranges and edge partners, so the adaptive backward can gather
    // instead of scattering with atomics here too (see adaptive_backward). Built
    // only when the list really is grouped by centre and every edge finds its
    // partner; otherwise both are left empty and the gather falls back.
    std::vector<int> off(N + 1, 0);
    bool grouped = true;
    for (int e = 0; e < E; ++e) {
      if (e > 0 && ed.raw_center[e] < ed.raw_center[e - 1]) { grouped = false; break; }
      ++off[ed.raw_center[e] + 1];
    }
    if (grouped) {
      for (int a = 0; a < N; ++a) off[a + 1] += off[a];
      // Partner of (i, j, v) is (j, i, -v). Matched on the neighbour index and the
      // closest opposing vector, which distinguishes periodic images of the same
      // pair; an exact float compare would be at the mercy of how each vector was
      // rounded.
      std::vector<int> rev(E, -1);
      for (int e = 0; e < E && grouped; ++e) {
        const int i = ed.raw_center[e], j = ed.raw_neigh[e];
        const double vx = -ed.raw_vec[3 * e + 0], vy = -ed.raw_vec[3 * e + 1],
                     vz = -ed.raw_vec[3 * e + 2];
        double best = 1e300;
        int found = -1;
        for (int f = off[j]; f < off[j + 1]; ++f) {
          if (ed.raw_neigh[f] != i) continue;
          const double dx = ed.raw_vec[3 * f + 0] - vx, dy = ed.raw_vec[3 * f + 1] - vy,
                       dz = ed.raw_vec[3 * f + 2] - vz;
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < best) { best = r2; found = f; }
        }
        if (found < 0 || best > 1e-12) grouped = false;  // unmatched -> use the fallback
        rev[e] = found;
      }
      if (grouped) {
        dev.raw_off = IView1D("raw_off", N + 1);
        dev.raw_reverse = IView1D("raw_reverse", E);
        auto h_off = Kokkos::create_mirror_view(dev.raw_off);
        auto h_rev2 = Kokkos::create_mirror_view(dev.raw_reverse);
        for (int a = 0; a <= N; ++a) h_off(a) = off[a];
        for (int e = 0; e < E; ++e) h_rev2(e) = rev[e];
        Kokkos::deep_copy(dev.raw_off, h_off);
        Kokkos::deep_copy(dev.raw_reverse, h_rev2);
      }
    }
  }
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
  auto h_sid = Kokkos::create_mirror_view(struct_id);
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
    IView1D ecnt = ws_.i1("batch_energy_cnt", B);  // zeroed on reuse
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
EnergyResult PetModel::compute_residual(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                                        RView2D* dev_forces, RView1D* dev_per_atom,
                                        RView2D* dev_virial) {
  last_n_atoms_ = dev.n_atoms;
  last_max_neighbors_ = dev.max_neighbors;
  peak_max_neighbors_ = std::max(peak_max_neighbors_, dev.max_neighbors);
  peak_edge_slots_ = std::max(peak_edge_slots_,
                              (long) dev.n_atoms * std::max(1, dev.max_neighbors));
  const int N = dev.n_atoms;
  const int M = dev.max_neighbors;
  const int S = M + 1;
  const int D = h_.d_pet;   // == d_node (non-expanded central token)
  const int Dh = h_.d_head;
  const int NM = N * M;
  const int G = h_.num_gnn_layers;
  const int A = h_.num_attention_layers;
  const int R = h_.num_readout_layers;  // == G for the residual featurizer
  // Charge / spin conditioning, computed once and reused by every GNN layer.
  // Empty (and free) for a model without it.
  View2D cond = conditioning(dev, dev.n_atoms, dev.n_struct);
  const bool grad = (host_forces != nullptr || dev_forces != nullptr);

  // Forward activations are fully overwritten; skip per-reuse zeroing.
  ws_.set_zero(false);

  auto d_species = dev.species;
  auto d_neigh_species = dev.neigh_species;
  auto d_reverse = dev.reverse_index;
  auto d_edge_vec = dev.edge_vec;
  auto d_dist = dev.dist;
  auto d_mask = dev.mask;
  auto d_cutoff = dev.cutoff_factor;
  auto d_cf_seq = dev.cf_seq;

  // initial edge messages = species embedding of the neighbor; geometric (v,|v|)
  View2D input_edge = ws_.n2("re_input_edge", NM, D);
  gather(input_edge, mat("edge_embedder.weight"), d_neigh_species);
  View2D edge_in4 = ws_.n2("re_edge_in4", NM, 4);
  Kokkos::parallel_for(
      "re_edge_in4", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
        edge_in4(k, 0) = d_edge_vec(k, 0);
        edge_in4(k, 1) = d_edge_vec(k, 1);
        edge_in4(k, 2) = d_edge_vec(k, 2);
        edge_in4(k, 3) = d_dist(k);
      });

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
    const bool is_first = (L == 0);

    // fresh per-layer central-node embedding (residual featurizer: node input is
    // re-embedded from species every layer; only edge messages carry over).
    View2D node_L = ws_.n2("re_node_L", N, D);
    gather(node_L, mat("node_embedders." + ls + ".weight"), d_species);

    // geometric edge embedding: Linear(4 -> D) on [edge_vec, dist]
    View2D edge_emb = ws_.n2("re_edge_emb", NM, D);
    linear(edge_emb, edge_in4, mat(g + ".edge_embedder.weight"), vec(g + ".edge_embedder.bias"));

    // compress( cat[edge_emb, (neighbor_embedder), input_edge] ) -> edge token
    const int n_merge = is_first ? 2 : 3;
    View2D comp_in = ws_.n2("re_comp_in" + std::to_string(n_merge), NM, n_merge * D);
    if (is_first) {
      Kokkos::parallel_for(
          "re_cat2", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            comp_in(k, d) = edge_emb(k, d);
            comp_in(k, D + d) = input_edge(k, d);
          });
    } else {
      View2D nb_emb = ws_.n2("re_nb_emb", NM, D);
      gather(nb_emb, mat(g + ".neighbor_embedder.weight"), d_neigh_species);
      Kokkos::parallel_for(
          "re_cat3", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            comp_in(k, d) = edge_emb(k, d);
            comp_in(k, D + d) = nb_emb(k, d);
            comp_in(k, 2 * D + d) = input_edge(k, d);
          });
    }
    View2D cpre = ws_.n2("re_cpre", NM, D);
    if (grad) sav_scpre[L] = ws_.n2("rs_cpre_" + ls, NM, D);
    linear_silu(cpre, grad ? sav_scpre[L] : View2D(), comp_in, mat(g + ".compress.0.weight"),
                vec(g + ".compress.0.bias"), grad);
    View2D et = ws_.n2("re_et", NM, D);
    linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

    // tokens[N,S,D]: s==0 central node token, s>=1 edge tokens
    View2D tokens = ws_.n2("re_tokens", N * S, D);
    Kokkos::parallel_for(
        "re_tok", RangePolicy(0, (N * S) * D), KOKKOS_LAMBDA(int _i) {
          const int row = _i / D, d = _i % D;
          const int n = row / S, s = row % S;
          tokens(row, d) = (s == 0) ? node_L(n, d) : et(n * M + (s - 1), d);
        });

    // PostLN transformer stack
    for (int a = 0; a < A; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string as = ls + "_" + std::to_string(a);

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
      sav_mlppre[L][a] = ws_.n2("rs_mlppre_" + as, N * S, ffn_pre_width(tl + ".mlp"));
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
          const int row = _i / D, d = _i % D;
          const int n = row / S, s = row % S;
          if (s == 0) onode(n, d) = tokens(row, d);
          else oedge(n * M + (s - 1), d) = tokens(row, d);
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
  readout(node_feat, edge_feat, per_atom_net, d_mask, d_cutoff, N, M,
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
  if (dev_per_atom) *dev_per_atom = per_atom;

  EnergyResult res;
  if (!dev_per_atom) {
    res.per_atom.resize(N);
    auto h_pa = Kokkos::create_mirror_view(per_atom);
    Kokkos::deep_copy(h_pa, per_atom);
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      res.per_atom[i] = h_pa(i);
      total += h_pa(i);
    }
    res.total = total;
  }
  if (!grad) return res;

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
    Kokkos::parallel_for(
        "re_ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { node_pred_adj(n, 0) = Net(1); });
    Kokkos::parallel_for(
        "re_ro_seed_e", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          edge_pred_adj(k, 0) = d_mask(k) * d_cutoff(k);
          cutoff_adj(k) += d_mask(k) * epred(k, 0);  // sequential over i: no race
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
    const bool is_first = (L == 0);
    const int n_merge = is_first ? 2 : 3;

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
            const int row = _i / D, d = _i % D;
            const int n = row / S, s = row % S;
            tokens_adj(row, d) = (s == 0) ? nfa(n, d) : efa(n * M + (s - 1), d);
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
    Kokkos::parallel_for(
        "reb_split_tok", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
          const int k = _i / D, d = _i % D;
          const int n = k / M, m = k % M;
          et_adj(k, d) = tokens_adj((n * S) + (1 + m), d);
        });
    // et = compress.2(silu(compress.0(comp_in)))
    View2D cpre_adj = ws_.n2("re_cpre_adj", NM, D);
    linear_bwd(cpre_adj, et_adj, mat(g + ".compress.2.weight"));
    silu_bwd(cpre_adj, sav_scpre[L]);
    View2D comp_in_adj = ws_.n2("re_comp_in_adj" + std::to_string(n_merge), NM, n_merge * D);
    linear_bwd(comp_in_adj, cpre_adj, mat(g + ".compress.0.weight"));
    // comp_in = [edge_emb | (nb_emb) | input_edge]; edge_emb -> geometry, input_edge -> ie_adj
    View2D edge_emb_adj = ws_.n2("re_edge_emb_adj", NM, D);
    View2D ie_next = ws_.n2("re_ie_next", NM, D);  // adjoint of input_edge_L
    {
      const bool has_msg = (L + 1 < G);
      auto ie_prev = ie_adj;  // adjoint of input_edge_{L+1}
      Kokkos::parallel_for(
          "reb_comp_split", RangePolicy(0, NM * D), KOKKOS_LAMBDA(int _i) {
            const int k = _i / D, d = _i % D;
            edge_emb_adj(k, d) = comp_in_adj(k, d);
            // input_edge_L feeds the compress cat (last block) AND, via message
            // passing, 0.5 of input_edge_{L+1}'s adjoint.
            Net v = comp_in_adj(k, (n_merge - 1) * D + d);
            if (has_msg) v += Net(0.5) * ie_prev(k, d);
            ie_next(k, d) = v;
          });
    }
    // edge_emb = edge_embedder(edge_in4) -> accumulate geometry adjoint
    linear_bwd(edge_in4_adj, edge_emb_adj, mat(g + ".edge_embedder.weight"));
    ie_adj = ie_next;  // carry to the previous (earlier) layer
  }

  // ---- geometry backward + force/virial scatter (non-adaptive; mirrors compute()) ----
  RView2D edge_grad = ws_.r2("re_edge_grad", NM, 3);  // dE/dv per kept edge (double)
  {
    const bool is_bump = (h_.cutoff_function == CutoffFunction::Bump);
    const double width = h_.cutoff_width;
    auto d_pcut = dev.pair_cutoff;
    Kokkos::parallel_for(
        "re_geom_bwd", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          const double mask = d_mask(k);
          if (mask <= 0.0) {
            edge_grad(k, 0) = edge_grad(k, 1) = edge_grad(k, 2) = 0.0;
            return;
          }
          const int n = k / M, m = k % M;
          const double cutoff_total = cutoff_adj(k) + cf_seq_adj(n, 1 + m);
          const double dist = d_dist(k);
          const double rc = d_pcut(k);
          const double dcut_dd = cutoff_ddist(dist, rc, width, is_bump);
          const double dist_adj = edge_in4_adj(k, 3) + cutoff_total * dcut_dd;
          const double invd = (dist > 0.0) ? 1.0 / dist : 0.0;
          for (int c = 0; c < 3; ++c)
            edge_grad(k, c) = edge_in4_adj(k, c) + dist_adj * d_edge_vec(k, c) * invd;
          // non-adaptive: pair_cutoff is fixed, so no cutoff-radius backward term.
        });
  }

  RView2D d_forces = ws_.r2("re_forces", N, 3);
  const int NS = dev.n_struct;
  auto sid = dev.struct_id;
  RView2D dvir = ws_.r2("re_virial9", NS, 9);
  fold_edge_gradients(ws_, "re", edge_grad, d_edge_vec, d_mask, d_reverse, d_forces, dvir,
                      sid, N, M, NS, energy_scale_);

  if (dev_virial) {
    RView2D bvir = ws_.r2("re_batch_virial", NS, 6);  // pooled; see batch_virial
    Kokkos::parallel_for(
        "re_virial_voigt", RangePolicy(0, NS), KOKKOS_LAMBDA(int b) {
          bvir(b, 0) = dvir(b, 0);
          bvir(b, 1) = dvir(b, 4);
          bvir(b, 2) = dvir(b, 8);
          bvir(b, 3) = 0.5 * (dvir(b, 1) + dvir(b, 3));
          bvir(b, 4) = 0.5 * (dvir(b, 2) + dvir(b, 6));
          bvir(b, 5) = 0.5 * (dvir(b, 5) + dvir(b, 7));
        });
    *dev_virial = bvir;
  } else {
    auto h_vir = Kokkos::create_mirror_view(dvir);
    Kokkos::deep_copy(h_vir, dvir);
    res.virial[0] = h_vir(0, 0);
    res.virial[1] = h_vir(0, 4);
    res.virial[2] = h_vir(0, 8);
    res.virial[3] = 0.5 * (h_vir(0, 1) + h_vir(0, 3));
    res.virial[4] = 0.5 * (h_vir(0, 2) + h_vir(0, 6));
    res.virial[5] = 0.5 * (h_vir(0, 5) + h_vir(0, 7));
  }
  if (dev_forces) *dev_forces = d_forces;
  if (host_forces) {
    host_forces->resize(static_cast<std::size_t>(N) * 3);
    auto h_f = Kokkos::create_mirror_view(d_forces);
    Kokkos::deep_copy(h_f, d_forces);
    for (int i = 0; i < N; ++i)
      for (int c = 0; c < 3; ++c) (*host_forces)[3 * i + c] = h_f(i, c);
  }

  return res;
}

EnergyResult PetModel::compute(const DeviceEdgeData& dev, std::vector<double>* host_forces,
                               RView2D* dev_forces, RView1D* dev_per_atom, RView2D* dev_virial) {
  if (h_.featurizer_type == FeaturizerType::Residual)
    return compute_residual(dev, host_forces, dev_forces, dev_per_atom, dev_virial);

  last_n_atoms_ = dev.n_atoms;
  last_max_neighbors_ = dev.max_neighbors;
  peak_max_neighbors_ = std::max(peak_max_neighbors_, dev.max_neighbors);
  peak_edge_slots_ = std::max(peak_edge_slots_,
                              (long) dev.n_atoms * std::max(1, dev.max_neighbors));
  const int N = dev.n_atoms;
  const int M = dev.max_neighbors;
  const int S = M + 1;
  const int D = h_.d_pet;
  const int Dn = h_.d_node;
  const int Dh = h_.d_head;  // readout-head hidden width (not necessarily d_pet)
  const int NM = N * M;
  const bool grad = (host_forces != nullptr || dev_forces != nullptr);
  const int G = h_.num_gnn_layers;
  const int A = h_.num_attention_layers;

  // Forward activations are all fully overwritten (gemm beta=0 / gather / norm /
  // full-coverage writes), so skip the per-reuse zeroing here; the backward pass
  // re-enables it for its accumulators (see Workspace::set_zero).
  ws_.set_zero(false);

  // ---- device-resident NEF data (already uploaded / built on device) ----
  auto d_species = dev.species;
  auto d_neigh_species = dev.neigh_species;
  auto d_reverse = dev.reverse_index;
  auto d_edge_vec = dev.edge_vec;
  auto d_dist = dev.dist;
  auto d_mask = dev.mask;
  auto d_pcut = dev.pair_cutoff;
  auto d_cutoff = dev.cutoff_factor;
  auto d_cf_seq = dev.cf_seq;

  // ---- initial embeddings ----
  View2D node = ws_.n2("node", N, Dn);
  View2D input_edge = ws_.n2("input_edge", NM, D);
  View2D edge_in4 = ws_.n2("edge_in4", NM, 4);
  {
    gather(node, mat("node_embedders.0.weight"), d_species);
    gather(input_edge, mat("edge_embedder.weight"), d_neigh_species);
    Kokkos::parallel_for(
        "edge_in4", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          edge_in4(k, 0) = d_edge_vec(k, 0);
          edge_in4(k, 1) = d_edge_vec(k, 1);
          edge_in4(k, 2) = d_edge_vec(k, 2);
          edge_in4(k, 3) = d_dist(k);
        });
  }

  // saved activations (only when computing gradients)
  // Per GNN layer. sav_cpre / sav_concat / sav_cph belong to the featurizer
  // machinery around the transformer stack, so they stay one-per-layer.
  std::vector<View2D> sav_cpre(G), sav_concat(G), sav_cph(G);
  // Per GNN layer AND per attention layer: the transformer stack is A blocks
  // deep, and the backward walks them in reverse, so each needs its own saved
  // activations. (The residual featurizer has had this shape all along; the
  // feedforward one used to be hardcoded to A == 1.)
  std::vector<std::vector<View2D>> sav_tokens(G, std::vector<View2D>(A)),
      sav_qkv(G, std::vector<View2D>(A)), sav_node_new(G, std::vector<View2D>(A)),
      sav_tmp_center(G, std::vector<View2D>(A)), sav_eps(G, std::vector<View2D>(A)),
      sav_tmp_edge(G, std::vector<View2D>(A));
  // Charge / spin conditioning, computed once and reused by every GNN layer.
  // Empty (and free) for a model without it.
  View2D cond = conditioning(dev, N, dev.n_struct);

  // Readout saves. One set: this featurizer reads out once. Sized as vectors so
  // PetModel::readout is the same call for both featurizers.
  std::vector<View2D> sav_nh0(1), sav_nh1(1), sav_eh0(1), sav_eh1(1), sav_epred(1);

  for (int L = 0; L < G; ++L) {
    const std::string g = "gnn_layers." + std::to_string(L);
    const std::string ls = std::to_string(L);  // per-layer workspace-key suffix
    const bool is_first = (L == 0);

    View2D edge_emb = ws_.n2("edge_emb", NM, D);
    linear(edge_emb, edge_in4, mat(g + ".edge_embedder.weight"), vec(g + ".edge_embedder.bias"));

    const int n_merge = is_first ? 2 : 3;
    // Key by n_merge, not just "comp_in": the first GNN layer concatenates 2*D
    // columns and later layers 3*D, and Workspace::get2 only reuses a slot when the
    // column count matches exactly -- so a single key made every layer switch
    // destroy and reallocate (and zero-initialize) this ~[NM, 3*D] buffer. Keying
    // by the column count instead of by the layer index keeps it at two buffers
    // however many layers the model has.
    View2D comp_in = ws_.n2("comp_in" + std::to_string(n_merge), NM, n_merge * D);
    if (is_first) {
      Kokkos::parallel_for(
          "cat2", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            comp_in(k, d) = edge_emb(k, d);
            comp_in(k, D + d) = input_edge(k, d);
          });
    } else {
      View2D nb_emb = ws_.n2("nb_emb", NM, D);
      gather(nb_emb, mat(g + ".neighbor_embedder.weight"), d_neigh_species);
      Kokkos::parallel_for(
          "cat3", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            comp_in(k, d) = edge_emb(k, d);
            comp_in(k, D + d) = nb_emb(k, d);
            comp_in(k, 2 * D + d) = input_edge(k, d);
          });
    }
    View2D cpre = ws_.n2("cpre", NM, D);
    if (grad) sav_cpre[L] = ws_.n2("s_cpre_" + ls, NM, D);
    linear_silu(cpre, grad ? sav_cpre[L] : View2D(), comp_in, mat(g + ".compress.0.weight"),
                vec(g + ".compress.0.bias"), grad);
    View2D et = ws_.n2("et", NM, D);
    linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

    // ---- transformer stack: A TransformerLayer blocks ------------------------
    //
    // metatrain's TransformerLayer is the WHOLE unit -- centre contraction,
    // attention, centre expansion and MLP, edge residual and MLP -- and
    // Transformer.forward runs A of them in sequence, threading (node, edge)
    // through. So A > 1 is a loop over this entire body, not just over the
    // attention itself; the edge embedding `et` seeds the first block and each
    // block's outputs feed the next.
    View2D node_cur = node;   // node features entering this block  [N, Dn]
    View2D edge_cur = et;     // edge features entering this block  [NM, D]
    View2D out_edge;          // this block's edge output, carried out of the loop

    for (int a = 0; a < A; ++a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      // Distinct workspace keys per (GNN layer, attention layer): two buffers
      // that are simultaneously live must never share a key.
      const std::string as = ls + "_" + std::to_string(a);

      // central token contraction
      View2D input_node = ws_.n2("input_node", N, D);
      linear(input_node, node_cur, mat(tl + ".center_contraction.weight"),
             vec(tl + ".center_contraction.bias"));

      // When grad, the producer writes straight into the per-layer save buffer the
      // backward reads (tokens is read-only afterwards), so no copy_into snapshot
      // and no separate transient buffer are needed.
      View2D tokens = grad ? (sav_tokens[L][a] = ws_.n2("s_tok_" + as, N * S, D))
                           : ws_.n2("tokens", N * S, D);
      Kokkos::parallel_for(
          "tok", RangePolicy(0, (N * S) * (D)), KOKKOS_LAMBDA(int _i) { const int row = _i / (D), d = _i % (D);
            const int n = row / S, s = row % S;  // s==0 -> central token, else neighbor edge
            tokens(row, d) = (s == 0) ? input_node(n, d) : edge_cur(n * M + (s - 1), d);
          });

      View2D attn_in = ws_.n2("attn_in", N * S, D);
      norm(attn_in, tokens, tl + ".norm_attention");
      View2D qkv = grad ? (sav_qkv[L][a] = ws_.n2("s_qkv_" + as, N * S, 3 * D))
                        : ws_.n2("qkv", N * S, 3 * D);  // gemm overwrites it fully (no copy needed)
      linear(qkv, attn_in, mat(tl + ".attention.input_linear.weight"),
             vec(tl + ".attention.input_linear.bias"));
      View2D attn_out = ws_.n2("attn_out", N * S, D);
      attention(ws_, "attn_" + as, attn_out, qkv, d_cf_seq,
                mat(tl + ".attention.output_linear.weight"),
                vec(tl + ".attention.output_linear.bias"), N, S, h_.num_heads, h_.head_dim,
                h_.attention_temperature, grad);

      View2D out_node128 = ws_.n2("out_node128", N, D);
      View2D edge_attn = ws_.n2("out_edge_" + as, NM, D);
      Kokkos::parallel_for(
          "split_no", RangePolicy(0, (N * S) * (D)),
          KOKKOS_LAMBDA(int _i) { const int row = _i / (D), d = _i % (D);
            const int n = row / S, s = row % S;
            if (s == 0) out_node128(n, d) = attn_out(row, d);
            else edge_attn(n * M + (s - 1), d) = attn_out(row, d);
          });

      // node residual + center MLP
      View2D node_exp = ws_.n2("node_exp", N, Dn);
      linear(node_exp, out_node128, mat(tl + ".center_expansion.weight"),
             vec(tl + ".center_expansion.bias"));
      View2D node_next = ws_.n2("node_new_" + as, N, Dn);  // carried to the next block
      // residual = node_cur + node_exp is the norm input the backward needs. When
      // grad, write it straight into the save buffer; the final
      // node_next = residual + node_ff is a separate add, so no copy_into snapshot
      // is required. When !grad, node_res aliases node_next (in-place residual then
      // in-place += node_ff).
      View2D node_res = grad ? (sav_node_new[L][a] = ws_.n2("s_nn_" + as, N, Dn)) : node_next;
      Kokkos::parallel_for(
          "nres", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn); node_res(n, d) = node_cur(n, d) + node_exp(n, d); });
      View2D node_norm = ws_.n2("node_norm", N, Dn);
      norm(node_norm, node_res, tl + ".norm_center_features");
      View2D tmp_center = ws_.n2("tmp_center_" + as, N, ffn_pre_width(tl + ".center_mlp"));
      View2D node_ff = ws_.n2("node_ff", N, Dn);
      feedforward("cmlp_" + as, node_ff, node_norm, tl + ".center_mlp", tmp_center, grad);
      if (grad) sav_tmp_center[L][a] = tmp_center;
      Kokkos::parallel_for(
          "nres2", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn); node_next(n, d) = node_res(n, d) + node_ff(n, d); });

      // edge residual + edge MLP. eps = attn + edge_in is the norm input the
      // backward needs; when grad, write it straight into the save buffer (no
      // copy). edge_attn currently holds attn; the final output is eps + edge_ff.
      // When !grad, eps aliases edge_attn (in-place add then in-place add).
      View2D eps = grad ? (sav_eps[L][a] = ws_.n2("s_eps_" + as, NM, D)) : edge_attn;
      Kokkos::parallel_for(
          "epsadd", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D); eps(k, d) = edge_attn(k, d) + edge_cur(k, d); });
      View2D edge_norm = ws_.n2("edge_norm", NM, D);
      norm(edge_norm, eps, tl + ".norm_mlp");
      View2D tmp_edge = ws_.n2("tmp_edge_" + as, NM, ffn_pre_width(tl + ".mlp"));
      View2D edge_ff = ws_.n2("edge_ff", NM, D);
      feedforward("emlp_" + as, edge_ff, edge_norm, tl + ".mlp", tmp_edge, grad);
      if (grad) sav_tmp_edge[L][a] = tmp_edge;
      View2D edge_next = ws_.n2("edge_out_" + as, NM, D);
      Kokkos::parallel_for(
          "eres2", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D); edge_next(k, d) = eps(k, d) + edge_ff(k, d); });

      node_cur = node_next;
      edge_cur = edge_next;
      out_edge = edge_next;
    }

    // featurizer (feedforward) update
    if (cond.extent(0) == (std::size_t) N) {
      Kokkos::parallel_for(
          "cond_add", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn);
            node_cur(n, d) += cond(n, d); });
    }
    node = node_cur;
    View2D rev_edge = ws_.n2("rev_edge", NM, D);
    // cat_rev writes concat fully; when grad it is the layernorm input the backward
    // needs, so write straight into the save buffer (no copy).
    View2D concat = grad ? (sav_concat[L] = ws_.n2("s_cc_" + ls, NM, 2 * D))
                         : ws_.n2("concat", NM, 2 * D);
    {
      Kokkos::parallel_for(
          "rev", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            const int r = d_reverse(k);
            rev_edge(k, d) = (r >= 0) ? out_edge(r, d) : 0.0;
          });
      Kokkos::parallel_for(
          "cat_rev", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            concat(k, d) = out_edge(k, d);
            concat(k, D + d) = rev_edge(k, d);
          });
    }
    View2D comb_norm = ws_.n2("comb_norm", NM, 2 * D);
    layernorm(comb_norm, concat, vec("combination_norms." + std::to_string(L) + ".weight"),
              vec("combination_norms." + std::to_string(L) + ".bias"));
    View2D cph = ws_.n2("cph", NM, 2 * D);
    if (grad) sav_cph[L] = ws_.n2("s_cph_" + ls, NM, 2 * D);
    linear_silu(cph, grad ? sav_cph[L] : View2D(), comb_norm,
                mat("combination_mlps." + std::to_string(L) + ".0.weight"),
                vec("combination_mlps." + std::to_string(L) + ".0.bias"), grad);
    View2D comb_out = ws_.n2("comb_out", NM, D);
    linear(comb_out, cph, mat("combination_mlps." + std::to_string(L) + ".2.weight"),
           vec("combination_mlps." + std::to_string(L) + ".2.bias"));
    Kokkos::parallel_for(
        "eupd", RangePolicy(0, (NM) * (D)),
        KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
          input_edge(k, d) = input_edge(k, d) + out_edge(k, d) + comb_out(k, d);
        });
  }

  // ---- readout ----
  // Readout: the feedforward featurizer reads out once, from the final node and
  // edge features. Same heads as the residual path, one set instead of G.
  View1D per_atom_net = ws_.n1("per_atom_net", N);
  {
    std::vector<View2D> nfeat{node}, efeat{input_edge};
    readout(nfeat, efeat, per_atom_net, d_mask, d_cutoff, N, M,
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
  if (dev_per_atom) *dev_per_atom = per_atom;  // device view for batched segmentation
  EnergyResult res;
  // Host per-atom/total are only consumed by the single-structure host callers
  // (energy / energy_forces). The batched device path sets dev_per_atom, segments
  // the energy on-device, and discards this EnergyResult -- so skip the D2H copy +
  // host reduction there, which otherwise forces a device sync on every step.
  if (!dev_per_atom) {
    res.per_atom.resize(N);
    auto h_pa = Kokkos::create_mirror_view(per_atom);
    Kokkos::deep_copy(h_pa, per_atom);
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      res.per_atom[i] = h_pa(i);
      total += h_pa(i);
    }
    res.total = total;
  }
  if (!grad) return res;

  // ==========================================================================
  // backward pass: adjoint of sum(per_atom_net) wrt edge vectors
  // ==========================================================================
  // Adjoints accumulate (linear_bwd beta=1 / norm_bwd += / atomic scatters), so the
  // backward needs reused buffers zeroed again.
  ws_.set_zero(true);
  // shared, cross-layer adjoints (zero-init)
  View2D edge_in4_adj = ws_.n2("edge_in4_adj", NM, 4);
  View2D cf_seq_adj = ws_.n2("cf_seq_adj", N, S);

  // ---- readout backward ----
  // per_atom_net_adj = 1 ; node_pred_adj = 1 ; edge_pred_adj = mask*cutoff
  View2D node_adj = ws_.n2("node_adj", N, Dn);      // adjoint of final node features
  View2D input_edge_adj = ws_.n2("ie_adj", NM, D);  // adjoint of final edge features (accumulator)
  View1D cutoff_adj = ws_.n1("cutoff_adj", NM);     // adjoint of cutoff_factor[k]
  {
    View2D node_pred_adj = ws_.n2("npa", N, 1);
    View2D edge_pred_adj = ws_.n2("epa", NM, 1);
    View2D epred = sav_epred[0];  // the readout's saved edge predictions
    Kokkos::parallel_for(
        "ro_seed_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { node_pred_adj(n, 0) = 1.0; });
    Kokkos::parallel_for(
        "ro_seed_e", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          edge_pred_adj(k, 0) = d_mask(k) * d_cutoff(k);
          cutoff_adj(k) = d_mask(k) * epred(k, 0);
        });
    // node head backward. The head's hidden width is d_head, NOT d_pet: these are
    // the adjoints of PetModel::readout's nh0/nh1, which are [N, d_head]. They were
    // sized [N, d_pet], which is the same number only because every model validated
    // so far has d_head == d_pet -- with an unequal pair the gemm would read the
    // weight past its own row and silu_bwd would walk off the saved activation.
    // (compute_residual's copy of this block already uses Dh.)
    View2D nh1_adj = ws_.n2("nh1_adj", N, Dh);
    linear_bwd(nh1_adj, node_pred_adj, mat("node_last_layers.energy.0.energy___0.weight"));
    silu_bwd(nh1_adj, sav_nh1[0]);
    View2D nh0_adj = ws_.n2("nh0_adj", N, Dh);
    linear_bwd(nh0_adj, nh1_adj, mat("node_heads.energy.0.2.weight"));
    silu_bwd(nh0_adj, sav_nh0[0]);
    linear_bwd(node_adj, nh0_adj, mat("node_heads.energy.0.0.weight"));
    // edge head backward
    View2D eh1_adj = ws_.n2("eh1_adj", NM, Dh);
    linear_bwd(eh1_adj, edge_pred_adj, mat("edge_last_layers.energy.0.energy___0.weight"));
    silu_bwd(eh1_adj, sav_eh1[0]);
    View2D eh0_adj = ws_.n2("eh0_adj", NM, Dh);
    linear_bwd(eh0_adj, eh1_adj, mat("edge_heads.energy.0.2.weight"));
    silu_bwd(eh0_adj, sav_eh0[0]);
    linear_bwd(input_edge_adj, eh0_adj, mat("edge_heads.energy.0.0.weight"));
  }

  // ---- layers backward (reverse order) ----
  for (int L = G - 1; L >= 0; --L) {
    const std::string g = "gnn_layers." + std::to_string(L);
    const std::string tl = g + ".trans.layers.0";
    const std::string ls = std::to_string(L);  // per-layer workspace-key suffix
    const bool is_first = (L == 0);
    const int n_merge = is_first ? 2 : 3;

    // featurizer update: input_edge_out = input_edge_in + out_edge + comb_out
    View2D out_edge_adj = ws_.n2("out_edge_adj", NM, D);
    View2D comb_out_adj = ws_.n2("comb_out_adj", NM, D);
    Kokkos::parallel_for(
        "bw_eupd", RangePolicy(0, (NM) * (D)),
        KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
          out_edge_adj(k, d) += input_edge_adj(k, d);  // out_edge gets input_edge_out_adj
          comb_out_adj(k, d) = input_edge_adj(k, d);
        });
    // (input_edge_in_adj == input_edge_adj is carried as-is to previous layer; the
    //  += out_edge & comb paths are separate, so input_edge_adj already equals
    //  d(input_edge_in) from this term.)

    // comb_out = Lin_cm2(silu(Lin_cm0(layernorm(concat))))
    View2D cph_adj = ws_.n2("cph_adj", NM, 2 * D);
    linear_bwd(cph_adj, comb_out_adj, mat("combination_mlps." + std::to_string(L) + ".2.weight"));
    silu_bwd(cph_adj, sav_cph[L]);
    View2D cnrm_adj = ws_.n2("cnrm_adj", NM, 2 * D);
    linear_bwd(cnrm_adj, cph_adj, mat("combination_mlps." + std::to_string(L) + ".0.weight"));
    View2D concat_adj = ws_.n2("concat_adj", NM, 2 * D);
    layernorm_bwd(concat_adj, cnrm_adj, sav_concat[L],
                  vec("combination_norms." + std::to_string(L) + ".weight"));
    // concat = [out_edge | rev_edge] ; rev_edge[k] = out_edge[reverse[k]]
    {
      Kokkos::parallel_for(
          "bw_concat", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            // Gather, not scatter. Slot k receives its own forward branch plus the
            // reverse branch of whichever slot points back at it -- and since the
            // reverse map is an involution, that slot is exactly reverse(k). So each
            // thread can compute its own value alone, in a fixed order.
            //
            // This was two atomic_adds (the forward one had to be atomic because the
            // reverse index of another thread could target this same slot), which
            // made the accumulated order, and therefore the fp32 rounding, vary run
            // to run.
            const int r = d_reverse(k);
            out_edge_adj(k, d) += concat_adj(k, d) + (r >= 0 ? concat_adj(r, D + d) : Net(0));
          });
    }

    // ---- transformer stack backward: A blocks, reverse order ------------------
    //
    // The mirror of the forward loop above. Each iteration turns the adjoints of
    // one block's OUTPUTS (out_edge_adj, node_adj) into the adjoints of its
    // INPUTS, which are the previous block's outputs. After the last iteration
    // they are the adjoints of `et` and of the node features entering the GNN
    // layer, which is what the featurizer backward below consumes.
    View2D et_adj = ws_.n2("et_adj", NM, D);
    View2D node_in_adj = ws_.n2("node_in_adj_" + ls, N, Dn);  // carried to prev GNN layer

    for (int a = A - 1; a >= 0; --a) {
      const std::string tl = g + ".trans.layers." + std::to_string(a);
      const std::string as = ls + "_" + std::to_string(a);

      // out_edge (final) = eps + edge_ff ; edge_ff = ffn(norm(eps))
      View2D eps_adj = ws_.n2("eps_adj", NM, D);
      Kokkos::parallel_for(
          "bw_oe", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D); eps_adj(k, d) = out_edge_adj(k, d); });
      {
        View2D enrm_adj = ws_.n2("enrm_adj", NM, D);
        feedforward_bwd("emlpb_" + as, enrm_adj, out_edge_adj, tl + ".mlp", sav_tmp_edge[L][a]);
        norm_bwd(eps_adj, enrm_adj, sav_eps[L][a], tl + ".norm_mlp");
      }

      // node_out = node_res + node_ff ; node_ff = ffn(norm(node_res))
      View2D node_new_adj = ws_.n2("node_new_adj", N, Dn);
      Kokkos::parallel_for(
          "bw_no", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn); node_new_adj(n, d) = node_adj(n, d); });
      {
        View2D ncn_adj = ws_.n2("ncn_adj", N, Dn);
        feedforward_bwd("cmlpb_" + as, ncn_adj, node_adj, tl + ".center_mlp", sav_tmp_center[L][a]);
        norm_bwd(node_new_adj, ncn_adj, sav_node_new[L][a], tl + ".norm_center_features");
      }
      // node_res = node_in + node_exp ; node_exp = Lin_ce(out_node128)
      View2D blk_node_in_adj = ws_.n2("blk_node_in_adj", N, Dn);
      Kokkos::parallel_for(
          "bw_nn", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn); blk_node_in_adj(n, d) = node_new_adj(n, d); });
      View2D out_node128_adj = ws_.n2("on128_adj", N, D);
      linear_bwd(out_node128_adj, node_new_adj, mat(tl + ".center_expansion.weight"));

      // assemble ao_adj from out_node128_adj (central) and eps_adj (edges)
      View2D ao_adj = ws_.n2("ao_adj", N * S, D);
      Kokkos::parallel_for(
          "bw_ao", RangePolicy(0, (N * S) * (D)),
          KOKKOS_LAMBDA(int _i) { const int row = _i / (D), d = _i % (D);
            const int n = row / S, s = row % S;
            ao_adj(row, d) = (s == 0) ? out_node128_adj(n, d) : eps_adj(n * M + (s - 1), d);
          });

      // attention backward -> attn_in_adj + cf_seq_adj
      View2D attn_in_adj = ws_.n2("attn_in_adj", N * S, D);
      attention_bwd(ws_, "attnb_" + as, "attn_" + as, attn_in_adj, cf_seq_adj, ao_adj,
                    sav_qkv[L][a], d_cf_seq, mat(tl + ".attention.input_linear.weight"),
                    mat(tl + ".attention.output_linear.weight"), N, S, h_.num_heads, h_.head_dim,
                    h_.attention_temperature);
      // attn_in = norm(tokens)
      View2D tokens_adj = ws_.n2("tokens_adj", N * S, D);
      norm_bwd(tokens_adj, attn_in_adj, sav_tokens[L][a], tl + ".norm_attention");

      // tokens: central = input_node, edges = the block's input edge features.
      // The edge input also reaches the output through the residual (eps), so its
      // adjoint is the sum of the two paths.
      View2D input_node_adj = ws_.n2("in_node_adj", N, D);
      View2D blk_edge_in_adj = ws_.n2("blk_edge_in_adj", NM, D);
      Kokkos::parallel_for(
          "bw_tok", RangePolicy(0, (N * S) * (D)),
          KOKKOS_LAMBDA(int _i) { const int row = _i / (D), d = _i % (D);
            const int n = row / S, s = row % S;
            if (s == 0) {
              input_node_adj(n, d) = tokens_adj(row, d);
            } else {
              const int k = n * M + (s - 1);
              blk_edge_in_adj(k, d) = eps_adj(k, d) + tokens_adj(row, d);
            }
          });
      // input_node = Lin_cc(node_in): adds the contraction path onto the residual
      // path already in blk_node_in_adj.
      linear_bwd(blk_node_in_adj, input_node_adj, mat(tl + ".center_contraction.weight"));

      // Hand this block's input adjoints to the previous block (or, at a == 0,
      // out to the featurizer backward). Copied rather than aliased because the
      // next iteration writes out_edge_adj/node_adj through the same keys.
      Kokkos::parallel_for(
          "bw_blk_e", RangePolicy(0, (NM) * (D)),
          KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
            const Net v = blk_edge_in_adj(k, d);
            out_edge_adj(k, d) = v;
            et_adj(k, d) = v;
          });
      Kokkos::parallel_for(
          "bw_blk_n", RangePolicy(0, (N) * (Dn)),
          KOKKOS_LAMBDA(int _i) { const int n = _i / (Dn), d = _i % (Dn); 
            const Net v = blk_node_in_adj(n, d);
            node_adj(n, d) = v;
            node_in_adj(n, d) = v;
          });
    }

    // et = Lin_c2(silu(Lin_c0(comp_in)))
    View2D cpre_adj = ws_.n2("cpre_adj", NM, D);
    linear_bwd(cpre_adj, et_adj, mat(g + ".compress.2.weight"));
    silu_bwd(cpre_adj, sav_cpre[L]);
    View2D comp_in_adj = ws_.n2("comp_in_adj" + std::to_string(n_merge), NM, n_merge * D);
    linear_bwd(comp_in_adj, cpre_adj, mat(g + ".compress.0.weight"));

    // comp_in = [edge_emb | (nb_emb) | input_edge_in]
    View2D edge_emb_adj = ws_.n2("edge_emb_adj", NM, D);
    Kokkos::parallel_for(
        "bw_comp", RangePolicy(0, (NM) * (D)),
        KOKKOS_LAMBDA(int _i) { const int k = _i / (D), d = _i % (D);
          edge_emb_adj(k, d) = comp_in_adj(k, d);
          input_edge_adj(k, d) += comp_in_adj(k, (n_merge - 1) * D + d);
        });
    // edge_emb = Lin_ee(edge_in4)
    linear_bwd(edge_in4_adj, edge_emb_adj, mat(g + ".edge_embedder.weight"));

    // carry node adjoint to previous layer
    node_adj = node_in_adj;
  }

  // ---- geometry backward: edge_in4_adj + cutoff path -> edge vectors ----
  // total cutoff_factor adjoint = readout term + attention bias term (cols 1..M)
  // (cf_seq_adj col 0 is the constant central token -> ignored)
  const bool do_adapt = h_.adaptive();
  RView1D adapted_adj = ws_.r1("adapted_adj", N);  // dE/d(adapted_cutoff[a]) (accumulator, double)
  RView2D edge_grad = ws_.r2("edge_grad", NM, 3);  // dE/dv per kept edge (double)
  RView1D pc_adj_e = ws_.r1("pc_adj_e", NM);       // per-edge pair-cutoff adjoint (adaptive)
  {
    const bool is_bump = (h_.cutoff_function == CutoffFunction::Bump);
    const double width = h_.cutoff_width;
    Kokkos::parallel_for(
        "geom_bwd", RangePolicy(0, NM), KOKKOS_LAMBDA(int k) {
          const double mask = d_mask(k);
          if (mask <= 0.0) {
            edge_grad(k, 0) = edge_grad(k, 1) = edge_grad(k, 2) = 0.0;
            return;
          }
          const int n = k / M, m = k % M;
          const double cutoff_total = cutoff_adj(k) + cf_seq_adj(n, 1 + m);
          const double dist = d_dist(k);
          const double rc = d_pcut(k);
          const double dcut_dd = cutoff_ddist(dist, rc, width, is_bump);
          // direct dependence of cutoff_factor on its own edge distance (rc fixed)
          const double dist_adj = edge_in4_adj(k, 3) + cutoff_total * dcut_dd;
          const double invd = (dist > 0.0) ? 1.0 / dist : 0.0;
          for (int c = 0; c < 3; ++c)
            edge_grad(k, c) = edge_in4_adj(k, c) + dist_adj * d_edge_vec(k, c) * invd;
          // adaptive path: d cutoff_factor / d pair_cutoff = -d cutoff_factor / d dist.
          // Parked per edge here and gathered per atom below, rather than scattered
          // with atomics -- adapted_adj feeds the adaptive cutoff, whose value
          // decides a discrete keep/drop, so its rounding is worth pinning down.
          if (do_adapt) pc_adj_e(k) = -cutoff_total * dcut_dd;
        });
    if (do_adapt) {
      // atom a's share is half of each of its own edges plus half of each edge
      // pointing at it -- and the latter are exactly the reverses of the former.
      auto rev = d_reverse;
      Kokkos::parallel_for(
          "adapted_adj_gather", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            double s = 0.0;
            for (int m = 0; m < M; ++m) {
              const int k = a * M + m;
              if (d_mask(k) <= 0.0) continue;
              s += 0.5 * pc_adj_e(k);
              const int r = rev(k);
              if (r >= 0) s += 0.5 * pc_adj_e(r);
            }
            adapted_adj(a) += s;
          });
    }
  }

  // scatter per-edge gradient to atoms (device): F[i]+=g, F[j]-=g, scaled by energy_scale
  // and accumulate the virial W_ab = sum_edges v_a * (scale*dE/dv)_b.
  RView2D d_forces = ws_.r2("forces", N, 3);  // accumulator (double)
  const int NS = dev.n_struct;             // 1 = single structure; >1 = batched
  auto sid = dev.struct_id;                // [N] owning structure (valid when NS>1)
  RView2D dvir = ws_.r2("virial9", NS, 9);  // per-structure virial (scatter + adaptive)
  {
    fold_edge_gradients(ws_, "ff", edge_grad, d_edge_vec, d_mask, d_reverse, d_forces, dvir,
                        sid, N, M, NS, energy_scale_);
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

  // symmetrize the virial into Voigt order [xx,yy,zz,xy,xz,yz]. When the caller
  // requested a device virial, fill the per-structure View [NS,6] with one kernel
  // (no host copy) — this works for any NS>=1 (dvir is populated for NS==1 by the
  // binned scatter above). Otherwise (host single-structure callers) fill the
  // scalar res.virial[6].
  if (dev_virial) {
    // Pooled (see batch_energy): fully overwritten by the kernel below, so no zero fill.
    RView2D bvir = ws_.r2("batch_virial", NS, 6);
    Kokkos::parallel_for(
        "virial_voigt", RangePolicy(0, NS), KOKKOS_LAMBDA(int b) {
          bvir(b, 0) = dvir(b, 0);
          bvir(b, 1) = dvir(b, 4);
          bvir(b, 2) = dvir(b, 8);
          bvir(b, 3) = 0.5 * (dvir(b, 1) + dvir(b, 3));
          bvir(b, 4) = 0.5 * (dvir(b, 2) + dvir(b, 6));
          bvir(b, 5) = 0.5 * (dvir(b, 5) + dvir(b, 7));
        });
    *dev_virial = bvir;
  } else {
    auto h_vir = Kokkos::create_mirror_view(dvir);
    Kokkos::deep_copy(h_vir, dvir);
    res.virial[0] = h_vir(0, 0);                        // xx
    res.virial[1] = h_vir(0, 4);                        // yy
    res.virial[2] = h_vir(0, 8);                        // zz
    res.virial[3] = 0.5 * (h_vir(0, 1) + h_vir(0, 3));  // xy
    res.virial[4] = 0.5 * (h_vir(0, 2) + h_vir(0, 6));  // xz
    res.virial[5] = 0.5 * (h_vir(0, 5) + h_vir(0, 7));  // yz
  }
  // device force output: hand back the [N,3] device view directly
  if (dev_forces) *dev_forces = d_forces;
  // host force output (single-structure callers): one N*3 copy to host
  if (host_forces) {
    host_forces->resize(static_cast<std::size_t>(N) * 3);
    auto h_f = Kokkos::create_mirror_view(d_forces);
    Kokkos::deep_copy(h_f, d_forces);
    for (int i = 0; i < N; ++i)
      for (int c = 0; c < 3; ++c) (*host_forces)[3 * i + c] = h_f(i, c);
  }

  return res;
}

}  // namespace pet
