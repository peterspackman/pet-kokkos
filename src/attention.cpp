// Multi-head attention over each atom's tokens, forward and backward.
//
// Atom n has S = M+1 tokens: its own (s = 0) and one per neighbour slot. Per
// head, query q attends to key k with weight softmax_k(scale * q.k + log cf_k),
// where cf is the key's smooth cutoff factor; a key with cf <= 0 (padding, or an
// edge exactly at its cutoff) is left out, and a padding query (s > 0, cf <= 0)
// produces zero.
//
// Thread order: one thread per (atom, head, query), query fastest. A warp is
// then 32 queries of the same atom and head, which read the same key and value
// at the same time -- one broadcast from L1.
//
// Each thread holds q and its output accumulator in registers, sized exactly by
// a compile-time head_dim (16, 32, 48, 64 and 80, the sizes the published models
// use; anything else takes a generic path sized kMaxHeadDim).
#include "ops.hpp"

#include <algorithm>
#include <type_traits>

namespace pet {

namespace {

// Call f with head_dim as a compile-time constant, 0 meaning "generic".
template <class F>
void with_head_dim(int hd, F&& f) {
  switch (hd) {
    case 16: return f(std::integral_constant<int, 16>{});
    case 32: return f(std::integral_constant<int, 32>{});
    case 48: return f(std::integral_constant<int, 48>{});
    case 64: return f(std::integral_constant<int, 64>{});
    case 80: return f(std::integral_constant<int, 80>{});
    default: return f(std::integral_constant<int, 0>{});
  }
}

// Online softmax, one pass over the keys. Saves the output (key:merged) and each
// query's running max and 1/sum (key:ml) when `save`.
template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
                    const WeightRef& w_out, View1D b_out, int N, int S, int H, int hd,
                    double temperature, bool save) {
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", N * S, D) : ws.tmp(N * S, D);
  View2D ml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  Kokkos::parallel_for(
      "attention", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
        const int sq = i % S, h = (i / S) % H, n = i / (S * H);
        const int nd = HD > 0 ? HD : hd;
        const int row_q = n * S + sq, si = (n * H + h) * S + sq;
        if (sq > 0 && cf_seq(n, sq) <= Net(0)) {  // padding query
          for (int d = 0; d < nd; ++d) merged(row_q, h * hd + d) = Net(0);
          if (save) ml(si, 0) = -Net(1e30), ml(si, 1) = Net(0);
          return;
        }
        const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
        Net q[CAP], acc[CAP];
        for (int d = 0; d < nd; ++d) q[d] = qkv(row_q, qo + d), acc[d] = Net(0);
        Net m = -Net(1e30), l = Net(0);
        for (int sk = 0; sk < S; ++sk) {
          const Net cf = cf_seq(n, sk);
          if (cf <= Net(0)) continue;
          const int row_k = n * S + sk;
          Net dot = Net(0);
          for (int d = 0; d < nd; ++d) dot += q[d] * qkv(row_k, ko + d);
          const Net s = dot * sc + fast_log(cf);
          const Net new_m = (s > m) ? s : m;
          const Net c = fast_exp(m - new_m), p = fast_exp(s - new_m);
          l = l * c + p;
          for (int d = 0; d < nd; ++d) acc[d] = acc[d] * c + p * qkv(row_k, vo + d);
          m = new_m;
        }
        const Net invl = (l > Net(0)) ? Net(1) / l : Net(0);
        for (int d = 0; d < nd; ++d) merged(row_q, h * hd + d) = acc[d] * invl;
        if (save) ml(si, 0) = m, ml(si, 1) = invl;
      });
  linear(out, merged, w_out, b_out);
}

// The backward, as the forward runs: nothing S x S is ever stored. With
// A = softmax weights (recomputed from the saved max m and 1/sum per query) and
// dS = A (dO.v - dO.out), the score adjoint before the scale:
//   queries, per (atom, head, query): dQ = sc sum_k dS K, and dO.out (Dq), kept;
//   keys, per (atom, head, key): dK = sc sum_q dS Q, dV = sum_q A dO, and the
//     key's dS summed over queries, the cf adjoint's share from this head;
//   cf, per (atom, key): those shares, summed over heads.
// Each pass recomputes q.k and dO.v for its pairs, in registers; a warp is
// consecutive queries (or keys) of one atom and head, so the rows it reads are
// broadcasts. Every sum has one owner and a fixed order: nothing is atomic.
template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                        View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                        const WeightRef& w_out, int N, int S, int H, int hd, double temperature,
                        Net beta) {
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(N * S, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(N * S, 3 * D);
  View2D cfh = ws.tmp(N * S, H);  // per key and head: the cf adjoint's share
  View2D dq_out = ws.tmp(N * H * S, 1);  // per query and head: dO.out
  View2D merged = ws.peek2(key + ":merged"), ml = ws.peek2(key + ":ml");

  Kokkos::parallel_for(
      "attn_bwd_queries", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
        const int sq = i % S, h = (i / S) % H, n = i / (S * H);
        const int nd = HD > 0 ? HD : hd;
        const int row_q = n * S + sq;
        const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
        const Net m = ml(i, 0), invl = ml(i, 1);
        Net q[CAP], dout[CAP], dq[CAP];
        Net Dq = Net(0);
        for (int d = 0; d < nd; ++d) {
          q[d] = qkv(row_q, qo + d);
          dout[d] = dmerged(row_q, qo + d);
          Dq += dout[d] * merged(row_q, qo + d);
          dq[d] = Net(0);
        }
        dq_out(i, 0) = Dq;
        if (invl > Net(0)) {  // else a padding query: no weights, no adjoints
          for (int sk = 0; sk < S; ++sk) {
            const Net cf = cf_seq(n, sk);
            if (cf <= Net(0)) continue;  // padding key
            const int row_k = n * S + sk;
            Net dot = Net(0), dA = Net(0);
            for (int d = 0; d < nd; ++d) {
              dot += q[d] * qkv(row_k, ko + d);
              dA += dout[d] * qkv(row_k, vo + d);
            }
            const Net a = fast_exp(dot * sc + fast_log(cf) - m) * invl;
            const Net ds = a * (dA - Dq) * sc;
            for (int d = 0; d < nd; ++d) dq[d] += ds * qkv(row_k, ko + d);
          }
        }
        for (int d = 0; d < nd; ++d) qkv_adj(row_q, qo + d) = dq[d];
      });

  Kokkos::parallel_for(
      "attn_bwd_keys", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
        const int sk = i % S, h = (i / S) % H, n = i / (S * H);
        const int nd = HD > 0 ? HD : hd;
        const int row_k = n * S + sk;
        const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
        Net k[CAP], v[CAP], dk[CAP], dv[CAP];
        Net cf_acc = Net(0);
        for (int d = 0; d < nd; ++d) dk[d] = dv[d] = Net(0);
        const Net cf = cf_seq(n, sk);
        if (cf > Net(0)) {  // else a padding key: no query sees it
          const Net lcf = fast_log(cf);
          for (int d = 0; d < nd; ++d) k[d] = qkv(row_k, ko + d), v[d] = qkv(row_k, vo + d);
          for (int sq = 0; sq < S; ++sq) {
            const int si = (n * H + h) * S + sq, row_q = n * S + sq;
            const Net invl = ml(si, 1);
            if (invl <= Net(0)) continue;  // padding query
            Net dot = Net(0), dA = Net(0);
            for (int d = 0; d < nd; ++d) {
              dot += qkv(row_q, qo + d) * k[d];
              dA += dmerged(row_q, qo + d) * v[d];
            }
            const Net a = fast_exp(dot * sc + lcf - ml(si, 0)) * invl;
            const Net ds = a * (dA - dq_out(si, 0));
            for (int d = 0; d < nd; ++d) {
              dk[d] += ds * sc * qkv(row_q, qo + d);
              dv[d] += a * dmerged(row_q, qo + d);
            }
            cf_acc += ds;
          }
        }
        for (int d = 0; d < nd; ++d) qkv_adj(row_k, ko + d) = dk[d], qkv_adj(row_k, vo + d) = dv[d];
        cfh(row_k, h) = cf_acc;
      });

  Kokkos::parallel_for(
      "attn_bwd_cf", RangePolicy(0, N * S), KOKKOS_LAMBDA(int i) {
        const int sk = i % S, n = i / S;
        const Net cf = cf_seq(n, sk);
        if (cf <= Net(0)) return;
        Net total = Net(0);
        for (int h = 0; h < H; ++h) total += cfh(n * S + sk, h);
        cf_seq_adj(n, sk) += total / cf;
      });
  linear_bwd(in_adj, qkv_adj, w_in, beta);
}

}  // namespace

void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, int N, int S, int heads, int head_dim,
               double temperature, bool save) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_impl<decltype(hd)::value>(ws, key, out, qkv, cf_seq, w_out, b_out, N, S, heads,
                                        head_dim, temperature, save);
  });
}

void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, int N, int S, int heads, int head_dim,
                   double temperature, Net beta) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_bwd_impl<decltype(hd)::value>(ws, key, in_adj, cf_seq_adj, out_adj, qkv, cf_seq,
                                            w_in, w_out, N, S, heads, head_dim, temperature, beta);
  });
}

}  // namespace pet
