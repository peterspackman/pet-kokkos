// The exact Hessian-vector product: the derivatives of the forces and the virial
// along K directions, by differentiating the forward and the hand-written
// backward along each direction (forward-over-reverse), with no finite step.
//
// The energy depends on the positions only through the raw edge vectors v, so a
// direction is a tangent v' of every raw edge (v' = u_j - u_i + e v for atomic
// displacements u and a strain e) and the product is d/dt of every dE/dv. It
// splits in two:
//   the network's: the edges enter it as x4 = (v, |v|), the cutoff factor and
//     the attention bias. Their tangents run forward through the network beside
//     the values, then backward beside the adjoints, every op's tangent next to
//     the op;
//   the geometry's: the cutoff factor's second derivatives in the distance and
//     the pair cutoff, and the solver adaptive cutoff's second-order implicit
//     derivative, contracted with the adjoints.
// The forward detaches the solver's root r and slope dn, as metatrain does; that
// gives the right first derivative but not the second, so here r and dn carry
// their true tangents and the result is the Hessian of the actual energy.
//
// Tangents are stacked in rows: a [R, C] activation's K tangents are one
// [K * R, C] buffer, tangent t's row r at t * R + r, so a linear layer's tangents
// are one GEMM through the same weight (and no bias).
//
// So far: a single structure, the feedforward featurizer, the solver (or no)
// adaptive cutoff, every activation kept, no CUDA graph and no exchange. The
// kernels are a thread per value, a warp per norm row, and a thread per query
// or key for the attention; tests/test_hvp.cpp holds the whole product to
// finite differences.
#include "ops.hpp"

#include "pet/cutoff.hpp"

#include <stdexcept>
#include <type_traits>

namespace pet {

namespace {

KOKKOS_INLINE_FUNCTION Net dsilu(Net z) {
  const Net s = sigmoid(z);
  return s * (Net(1) + z * (Net(1) - s));
}
KOKKOS_INLINE_FUNCTION Net d2silu(Net z) {
  const Net s = sigmoid(z);
  return s * (Net(1) - s) * (Net(2) + z * (Net(1) - Net(2) * s));
}

// The smooth cutoff's first and second derivatives in d, f(x) with x = (d - rc +
// w) / w, in double; the factor depends on rc only through d - rc.
KOKKOS_INLINE_FUNCTION void cut_derivs(double d, double rc, double w, bool bump, double& f1, double& f2) {
  f1 = f2 = 0.0;
  const double x = (d - (rc - w)) / w;
  if (x <= 0.0 || x >= 1.0) return;
  const double pi = detail::PET_PI;
  if (bump) {  // f = (1 + tanh(cot(pi x))) / 2
    const double si = Kokkos::sin(pi * x), co = Kokkos::cos(pi * x);
    const double th = Kokkos::tanh(co / si), se = 1.0 - th * th;
    const double ud = -pi / (si * si) / w, udd = 2.0 * pi * pi * co / (si * si * si) / (w * w);
    f1 = 0.5 * se * ud;
    f2 = 0.5 * (se * udd - 2.0 * th * se * ud * ud);
  } else {  // f = (1 + cos(pi x)) / 2
    f1 = -0.5 * pi * Kokkos::sin(pi * x) / w;
    f2 = -0.5 * pi * pi * Kokkos::cos(pi * x) / (w * w);
  }
}

// ---- elementwise tangents -------------------------------------------------------

// out_t = silu'(z) z_t; z [R, C], z_t and out_t [K R, C].
void act_tan(View2D out_t, View2D z, View2D z_t) {
  const int R = z.extent(0), C = z.extent(1);
  Kokkos::parallel_for(
      "hv_act", RangePolicy(0, z_t.extent(0) * C), KOKKOS_LAMBDA(int i) {
        const int r = (i / C) % R, c = i % C;
        out_t(i / C, c) = dsilu(z(r, c)) * z_t(i / C, c);
      });
}

// The backward of silu at pre-activation z, from a, the adjoint of its output:
// g = a silu'(z) (unless g is empty) and its tangent g_t = a_t silu'(z) + a
// silu''(z) z_t (a_t empty: zero).
void act_bwd2(View2D g, View2D g_t, View2D a, View2D a_t, View2D z, View2D z_t) {
  const int R = z.extent(0), C = z.extent(1);
  const bool has_at = a_t.extent(0) > 0;
  if (g.extent(0) > 0)
    Kokkos::parallel_for(
        "hv_act_bwd", RangePolicy(0, R * C), KOKKOS_LAMBDA(int i) {
          g(i / C, i % C) = a(i / C, i % C) * dsilu(z(i / C, i % C));
        });
  Kokkos::parallel_for(
      "hv_act_bwd_t", RangePolicy(0, z_t.extent(0) * C), KOKKOS_LAMBDA(int i) {
        const int q = i / C, r = q % R, c = i % C;
        const Net x = z(r, c);
        g_t(q, c) = (has_at ? a_t(q, c) * dsilu(x) : Net(0)) + a(r, c) * d2silu(x) * z_t(q, c);
      });
}

// SwiGLU's h = v sigmoid(g), v and g interleaved in pre (2j, 2j+1): its tangent.
void swiglu_tan(View2D h_t, View2D pre, View2D pre_t) {
  const int R = pre.extent(0), F = h_t.extent(1);
  Kokkos::parallel_for(
      "hv_swiglu", RangePolicy(0, h_t.extent(0) * F), KOKKOS_LAMBDA(int i) {
        const int q = i / F, r = q % R, c = i % F;
        const Net v = pre(r, 2 * c), s = sigmoid(pre(r, 2 * c + 1));
        h_t(q, c) = pre_t(q, 2 * c) * s + v * s * (Net(1) - s) * pre_t(q, 2 * c + 1);
      });
}

// Its backward from ha, h's adjoint, into pa (pre's), and both tangents.
void swiglu_bwd2(View2D pa, View2D pa_t, View2D ha, View2D ha_t, View2D pre, View2D pre_t) {
  const int R = pre.extent(0), F = ha.extent(1);
  Kokkos::parallel_for(
      "hv_swiglu_bwd", RangePolicy(0, R * F), KOKKOS_LAMBDA(int i) {
        const int r = i / F, c = i % F;
        const Net v = pre(r, 2 * c), s = sigmoid(pre(r, 2 * c + 1));
        pa(r, 2 * c) = ha(r, c) * s;
        pa(r, 2 * c + 1) = ha(r, c) * v * s * (Net(1) - s);
      });
  Kokkos::parallel_for(
      "hv_swiglu_bwd_t", RangePolicy(0, pa_t.extent(0) * F), KOKKOS_LAMBDA(int i) {
        const int q = i / F, r = q % R, c = i % F;
        const Net v = pre(r, 2 * c), s = sigmoid(pre(r, 2 * c + 1));
        const Net s1 = s * (Net(1) - s), s2 = s1 * (Net(1) - Net(2) * s);
        const Net vt = pre_t(q, 2 * c), gt = pre_t(q, 2 * c + 1), a = ha(r, c), at = ha_t(q, c);
        pa_t(q, 2 * c) = at * s + a * s1 * gt;
        pa_t(q, 2 * c + 1) = at * v * s1 + a * vt * s1 + a * v * s2 * gt;
      });
}

// ---- norms ----------------------------------------------------------------------
//
// out = (x - mu) inv w (+ b), mu the mean for LayerNorm and 0 for RMSNorm, inv =
// 1/sqrt(mean((x - mu)^2) + eps). With xc = x - mu: xc' = x' - mean(x'), inv' =
// -inv^3 mean(xc xc'), out' = w (xc' inv + xc inv'). A warp's lanes on a row, as
// the forward's norms run.

void norm_tan(View2D out_t, View2D x, View2D x_t, View1D w, bool ln) {
  const int R = x.extent(0), D = x.extent(1), RK = x_t.extent(0);
  Kokkos::parallel_for(
      "hv_norm", norm_policy(RK), KOKKOS_LAMBDA(const Team& t) {
        const int q = t.league_rank() * kNormRows + t.team_rank();
        if (q >= RK) return;
        const int r = q % R;
        Net sx = Net(0), st = Net(0);
        if (ln)
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(t, D), [&](int d, Net& a, Net& b) { a += x(r, d), b += x_t(q, d); }, sx, st);
        const Net mu = sx / D, mt = st / D;
        Net sq = Net(0), cr = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& a, Net& b) {
              const Net c = x(r, d) - mu;
              a += c * c, b += c * (x_t(q, d) - mt);
            },
            sq, cr);
        const Net inv = norm_inv(sq, D, ln), inv_t = -inv * inv * inv * cr / D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          out_t(q, d) = w(d) * ((x_t(q, d) - mt) * inv + (x(r, d) - mu) * inv_t);
        });
      });
}

// The backward, g the output's adjoint: gin = inv (gw - [mean gw] - xh mean(gw
// xh)), gw = g w, xh = xc inv; and its tangent. Onto gin, gin_t when acc.
void norm_bwd2(View2D gin, View2D gin_t, View2D g, View2D g_t, View2D x, View2D x_t, View1D w, bool ln,
               bool acc) {
  const int R = x.extent(0), D = x.extent(1), RK = x_t.extent(0);
  Kokkos::parallel_for(
      "hv_norm_bwd", norm_policy(R), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank() * kNormRows + t.team_rank();
        if (r >= R) return;
        const Net mu = ln ? row_mean(t, x, r) : Net(0);
        Net sq = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D), [&](int d, Net& a) { a += (x(r, d) - mu) * (x(r, d) - mu); }, sq);
        const Net inv = norm_inv(sq, D, ln);
        Net mg = Net(0), a = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& s0, Net& s1) {
              const Net gw = g(r, d) * w(d);
              s0 += gw, s1 += gw * (x(r, d) - mu) * inv;
            },
            mg, a);
        mg = ln ? mg / D : Net(0), a /= D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          const Net v = inv * (g(r, d) * w(d) - mg - (x(r, d) - mu) * inv * a);
          gin(r, d) = acc ? gin(r, d) + v : v;
        });
      });
  Kokkos::parallel_for(
      "hv_norm_bwd_t", norm_policy(RK), KOKKOS_LAMBDA(const Team& t) {
        const int q = t.league_rank() * kNormRows + t.team_rank();
        if (q >= RK) return;
        const int r = q % R;
        Net sx = Net(0), st = Net(0);
        if (ln)
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(t, D), [&](int d, Net& a, Net& b) { a += x(r, d), b += x_t(q, d); }, sx, st);
        const Net mu = sx / D, mt = st / D;
        Net sq = Net(0), cr = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& a, Net& b) {
              const Net c = x(r, d) - mu;
              a += c * c, b += c * (x_t(q, d) - mt);
            },
            sq, cr);
        const Net inv = norm_inv(sq, D, ln), inv_t = -inv * inv * inv * cr / D;
        // the row means of gw, gw', gw xh and (gw xh)'
        Net mg = Net(0), mgt = Net(0), a = Net(0), at = Net(0);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& s0, Net& s1, Net& s2, Net& s3) {
              const Net xh = (x(r, d) - mu) * inv, xht = (x_t(q, d) - mt) * inv + (x(r, d) - mu) * inv_t;
              const Net gw = g(r, d) * w(d), gwt = g_t(q, d) * w(d);
              s0 += gw, s1 += gwt, s2 += gw * xh, s3 += gwt * xh + gw * xht;
            },
            mg, mgt, a, at);
        mg = ln ? mg / D : Net(0), mgt = ln ? mgt / D : Net(0), a /= D, at /= D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          const Net xh = (x(r, d) - mu) * inv, xht = (x_t(q, d) - mt) * inv + (x(r, d) - mu) * inv_t;
          const Net gw = g(r, d) * w(d), gwt = g_t(q, d) * w(d);
          const Net v = inv_t * (gw - mg - xh * a) + inv * (gwt - mgt - xht * a - xh * at);
          gin_t(q, d) = acc ? gin_t(q, d) + v : v;
        });
      });
}

// ---- attention --------------------------------------------------------------------
//
// Per (atom, head), query i over keys j: s_ij = sc q_i.k_j + log cf_j, p = softmax_j
// s, o_i = sum_j p_ij v_j, over the valid keys (cf > 0); a query on an edge at its
// cutoff gives zero. Tangents: s' = sc (q'.k + q.k') + cf'/cf, p' = p (s' - pbar)
// with pbar = sum p s', o' = sum p' v + p v'. The backward from go = dO: pb = go.v,
// delta = go.o, sb = p (pb - delta); dq = sc sum_j sb k, dk = sc sum_i sb q, dv =
// sum_i p go, and d(log cf_j) = sum_i sb_ij -- each with its tangent.
//
// A thread per (tangent, atom, head, query) -- or key, in the keys pass -- its
// rows in registers, sized by a compile-time head_dim as the forward's are. The
// softmax is online: one pass over the keys accumulates o, sum e s' v and sum e
// v' (e = exp(s - running max)), rescaled when the max moves, which give o and o'
// at the end: o' = (sum e s' v - pbar sum e v + sum e v') / l. Nothing S x S is
// kept: the queries pass then recomputes p for dq, and the keys pass for dk, dv.

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

struct AttnShape {
  Tokens tk;
  int N, S, H, hd, T;
  Net sc;
};

template <int HD>
KOKKOS_INLINE_FUNCTION Net dot(const Net* a, const View2D& m, int row, int col, int nd) {
  Net s = Net(0);
  for (int d = 0; d < (HD > 0 ? HD : nd); ++d) s += a[d] * m(row, col + d);
  return s;
}
template <int HD>
KOKKOS_INLINE_FUNCTION void load(Net* a, const View2D& m, int row, int col, int nd) {
  for (int d = 0; d < (HD > 0 ? HD : nd); ++d) a[d] = m(row, col + d);
}

// One query's online pass over its keys: o and o' (into o, ot; B is scratch),
// and its softmax's max, 1/sum and pbar. q, qt: the query and its tangent.
template <int HD>
KOKKOS_INLINE_FUNCTION void query_pass(const Net* q, const Net* qt, Net* o, Net* ot, Net* B, Net& m, Net& invl,
                                       Net& pbar, const View2D& qkv, const View2D& qkv_t, const View2D& cf,
                                       const View2D& cf_t, const AttnShape& a, int n, int h, int t) {
  const int nd = HD > 0 ? HD : a.hd, D = a.H * a.hd, ko = D + h * a.hd, vo = 2 * D + h * a.hd;
  const int cnt = a.tk.count(n);
  Net l = Net(0), ps = Net(0);
  m = -Net(1e30);
  for (int d = 0; d < nd; ++d) o[d] = ot[d] = B[d] = Net(0);
  for (int j = 0; j < cnt; ++j) {
    const Net c = cf(n, j);
    if (c <= Net(0)) continue;
    const int rj = a.tk.row(n, j), tj = t * a.T + rj;
    const Net s = a.sc * dot<HD>(q, qkv, rj, ko, nd) + Kokkos::log(c);
    const Net st = a.sc * (dot<HD>(qt, qkv, rj, ko, nd) + dot<HD>(q, qkv_t, tj, ko, nd)) + cf_t(t * a.N + n, j) / c;
    if (s > m) {
      const Net r = Kokkos::exp(m - s);
      l *= r, ps *= r;
      for (int d = 0; d < nd; ++d) o[d] *= r, ot[d] *= r, B[d] *= r;
      m = s;
    }
    const Net e = Kokkos::exp(s - m);
    l += e, ps += e * st;
    for (int d = 0; d < nd; ++d) {
      const Net v = qkv(rj, vo + d);
      o[d] += e * v, ot[d] += e * st * v, B[d] += e * qkv_t(tj, vo + d);
    }
  }
  invl = l > Net(0) ? Net(1) / l : Net(0);
  pbar = ps * invl;
  for (int d = 0; d < nd; ++d) {
    o[d] *= invl;
    ot[d] = ot[d] * invl - pbar * o[d] + B[d] * invl;
  }
}

// merged_t [K T, D]: the tangent of the heads' outputs, before the output linear.
template <int HD>
void attn_tan_impl(View2D merged_t, View2D qkv, View2D qkv_t, View2D cf, View2D cf_t, const AttnShape& a, int K) {
  const int N = a.N, S = a.S, H = a.H;
  Kokkos::parallel_for(
      "hv_attn", RangePolicy(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
        constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
        const int i = idx % S, h = (idx / S) % H, n = (idx / (S * H)) % N, t = idx / (S * H * N);
        const int nd = HD > 0 ? HD : a.hd, qo = h * a.hd;
        if (i >= a.tk.count(n)) return;
        const int ri = a.tk.row(n, i);
        Net q[CAP], qt[CAP], o[CAP], ot[CAP], B[CAP];
        for (int d = 0; d < nd; ++d) ot[d] = Net(0);
        if (i == 0 || cf(n, i) > Net(0)) {
          load<HD>(q, qkv, ri, qo, nd);
          load<HD>(qt, qkv_t, t * a.T + ri, qo, nd);
          Net m, invl, pbar;
          query_pass<HD>(q, qt, o, ot, B, m, invl, pbar, qkv, qkv_t, cf, cf_t, a, n, h, t);
        }
        for (int d = 0; d < nd; ++d) merged_t(t * a.T + ri, qo + d) = ot[d];
      });
}
void attn_tan(View2D merged_t, View2D qkv, View2D qkv_t, View2D cf, View2D cf_t, const AttnShape& a, int K) {
  with_head_dim(a.hd, [&](auto hd) { attn_tan_impl<decltype(hd)::value>(merged_t, qkv, qkv_t, cf, cf_t, a, K); });
}

// The backward from dm (the merged output's adjoint) and its tangent, into
// qkv_adj and onto cf_adj (both with tangents). stats [K N H S, 5] is scratch.
template <int HD>
void attn_bwd2_impl(View2D qkv_adj, View2D qkv_adj_t, View2D cf_adj, View2D cf_adj_t, View2D dm, View2D dm_t,
                    View2D qkv, View2D qkv_t, View2D cf, View2D cf_t, View2D stats, const AttnShape& a, int K) {
  const int N = a.N, S = a.S, H = a.H;
  // Queries: dq, and per query its softmax, delta = go.o and their tangents.
  Kokkos::parallel_for(
      "hv_attn_q", RangePolicy(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
        constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
        const int i = idx % S, h = (idx / S) % H, n = (idx / (S * H)) % N, t = idx / (S * H * N);
        const int nd = HD > 0 ? HD : a.hd, D = H * a.hd, T = a.T, qo = h * a.hd, ko = D + qo, vo = 2 * D + qo;
        const int cnt = a.tk.count(n);
        if (i >= cnt) return;
        const int ri = a.tk.row(n, i), ti = t * T + ri;
        Net q[CAP], qt[CAP], o[CAP], ot[CAP], dq[CAP], dqt[CAP];
        for (int d = 0; d < nd; ++d) dq[d] = dqt[d] = Net(0);
        Net m = Net(0), invl = Net(0), pbar = Net(0), delta = Net(0), delta_t = Net(0);
        if (i == 0 || cf(n, i) > Net(0)) {
          load<HD>(q, qkv, ri, qo, nd);
          load<HD>(qt, qkv_t, ti, qo, nd);
          query_pass<HD>(q, qt, o, ot, dq, m, invl, pbar, qkv, qkv_t, cf, cf_t, a, n, h, t);
          for (int d = 0; d < nd; ++d) {
            delta += dm(ri, qo + d) * o[d];
            delta_t += dm_t(ti, qo + d) * o[d] + dm(ri, qo + d) * ot[d];
            dq[d] = Net(0);
          }
          // the query's own row of dO and its tangent, in o and ot from here
          load<HD>(o, dm, ri, qo, nd);
          load<HD>(ot, dm_t, ti, qo, nd);
          for (int j = 0; j < cnt; ++j) {
            const Net c = cf(n, j);
            if (c <= Net(0)) continue;
            const int rj = a.tk.row(n, j), tj = t * T + rj;
            const Net s = a.sc * dot<HD>(q, qkv, rj, ko, nd) + Kokkos::log(c);
            const Net st =
                a.sc * (dot<HD>(qt, qkv, rj, ko, nd) + dot<HD>(q, qkv_t, tj, ko, nd)) + cf_t(t * a.N + n, j) / c;
            const Net p = Kokkos::exp(s - m) * invl, pt = p * (st - pbar);
            const Net pb = dot<HD>(o, qkv, rj, vo, nd);
            const Net pbt = dot<HD>(ot, qkv, rj, vo, nd) + dot<HD>(o, qkv_t, tj, vo, nd);
            const Net sb = a.sc * p * (pb - delta), sbt = a.sc * (pt * (pb - delta) + p * (pbt - delta_t));
            for (int d = 0; d < nd; ++d) {
              const Net k = qkv(rj, ko + d);
              dq[d] += sb * k;
              dqt[d] += sbt * k + sb * qkv_t(tj, ko + d);
            }
          }
        }
        for (int d = 0; d < nd; ++d) {
          if (t == 0) qkv_adj(ri, qo + d) = dq[d];
          qkv_adj_t(ti, qo + d) = dqt[d];
        }
        stats(idx, 0) = m, stats(idx, 1) = invl, stats(idx, 2) = pbar;
        stats(idx, 3) = delta, stats(idx, 4) = delta_t;
      });
  // Keys: dk, dv and the log cf adjoint's share from this head.
  Kokkos::parallel_for(
      "hv_attn_k", RangePolicy(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
        constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
        const int j = idx % S, h = (idx / S) % H, n = (idx / (S * H)) % N, t = idx / (S * H * N);
        const int nd = HD > 0 ? HD : a.hd, D = H * a.hd, T = a.T, qo = h * a.hd, ko = D + qo, vo = 2 * D + qo;
        const int cnt = a.tk.count(n);
        if (j >= cnt) return;
        const int rj = a.tk.row(n, j), tj = t * T + rj;
        Net k[CAP], kt[CAP], v[CAP], vt[CAP], dk[CAP], dkt[CAP], dv[CAP], dvt[CAP];
        for (int d = 0; d < nd; ++d) dk[d] = dkt[d] = dv[d] = dvt[d] = Net(0);
        const Net c = cf(n, j);
        if (c > Net(0)) {
          load<HD>(k, qkv, rj, ko, nd);
          load<HD>(kt, qkv_t, tj, ko, nd);
          load<HD>(v, qkv, rj, vo, nd);
          load<HD>(vt, qkv_t, tj, vo, nd);
          const Net lc = Kokkos::log(c), bt = cf_t(t * a.N + n, j) / c;
          Net bb = Net(0), bbt = Net(0);
          for (int i = 0; i < cnt; ++i) {
            const int si = ((t * N + n) * H + h) * S + i;
            const Net invl = stats(si, 1);
            if (invl <= Net(0)) continue;  // a query on an edge at its cutoff
            const int ri = a.tk.row(n, i), ti = t * T + ri;
            const Net s = a.sc * dot<HD>(k, qkv, ri, qo, nd) + lc;
            const Net st = a.sc * (dot<HD>(kt, qkv, ri, qo, nd) + dot<HD>(k, qkv_t, ti, qo, nd)) + bt;
            const Net p = Kokkos::exp(s - stats(si, 0)) * invl, pt = p * (st - stats(si, 2));
            const Net pb = dot<HD>(v, dm, ri, qo, nd), pbt = dot<HD>(v, dm_t, ti, qo, nd) + dot<HD>(vt, dm, ri, qo, nd);
            const Net sb = p * (pb - stats(si, 3)), sbt = pt * (pb - stats(si, 3)) + p * (pbt - stats(si, 4));
            bb += sb, bbt += sbt;
            for (int d = 0; d < nd; ++d) {
              const Net qd = qkv(ri, qo + d), qdt = qkv_t(ti, qo + d), g = dm(ri, qo + d), gt = dm_t(ti, qo + d);
              dk[d] += a.sc * sb * qd;
              dkt[d] += a.sc * (sbt * qd + sb * qdt);
              dv[d] += p * g;
              dvt[d] += pt * g + p * gt;
            }
          }
          const Net ct = cf_t(t * N + n, j);
          if (t == 0) Kokkos::atomic_add(&cf_adj(n, j), bb / c);
          Kokkos::atomic_add(&cf_adj_t(t * N + n, j), (bbt - bb * ct / c) / c);
        }
        for (int d = 0; d < nd; ++d) {
          if (t == 0) qkv_adj(rj, ko + d) = dk[d], qkv_adj(rj, vo + d) = dv[d];
          qkv_adj_t(tj, ko + d) = dkt[d], qkv_adj_t(tj, vo + d) = dvt[d];
        }
      });
}
void attn_bwd2(View2D qkv_adj, View2D qkv_adj_t, View2D cf_adj, View2D cf_adj_t, View2D dm, View2D dm_t, View2D qkv,
               View2D qkv_t, View2D cf, View2D cf_t, View2D stats, const AttnShape& a, int K) {
  with_head_dim(a.hd, [&](auto hd) {
    attn_bwd2_impl<decltype(hd)::value>(qkv_adj, qkv_adj_t, cf_adj, cf_adj_t, dm, dm_t, qkv, qkv_t, cf, cf_t, stats,
                                        a, K);
  });
}

}  // namespace

// ---- the network's tangents ---------------------------------------------------------

void PetModel::feedforward_tan(View2D out_t, View2D in_t, const std::string& key, View2D pre, View2D pre_t,
                               Net beta) {
  const WeightRef w_in = mat(key + ".w_in.weight"), w_out = mat(key + ".w_out.weight");
  const bool swi = h_.activation == Activation::SwiGLU;
  Workspace::Scope scope(ws_);
  linear(pre_t, in_t, w_in, View1D(), 0);
  View2D h_t = ws_.tmp(in_t.extent(0), swi ? pre.extent(1) / 2 : pre.extent(1));
  if (swi) swiglu_tan(h_t, pre, pre_t);
  else act_tan(h_t, pre, pre_t);
  linear(out_t, h_t, w_out, View1D(), beta);
}

void PetModel::feedforward_bwd2(View2D in_adj, View2D in_adj_t, View2D out_adj, View2D out_adj_t,
                                const std::string& key, View2D pre, View2D pre_t, Net beta) {
  const WeightRef w_in = mat(key + ".w_in.weight"), w_out = mat(key + ".w_out.weight");
  const bool swi = h_.activation == Activation::SwiGLU;
  const int R = pre.extent(0), RK = pre_t.extent(0), P = pre.extent(1), F = swi ? P / 2 : P;
  Workspace::Scope scope(ws_);
  View2D ha = ws_.tmp(R, F), ha_t = ws_.tmp(RK, F), pa = ws_.tmp(R, P), pa_t = ws_.tmp(RK, P);
  linear_bwd(ha, out_adj, w_out, 0);
  linear_bwd(ha_t, out_adj_t, w_out, 0);
  if (swi) swiglu_bwd2(pa, pa_t, ha, ha_t, pre, pre_t);
  else act_bwd2(pa, pa_t, ha, ha_t, pre, pre_t);
  linear_bwd(in_adj, pa, w_in, beta);
  linear_bwd(in_adj_t, pa_t, w_in, beta);
}

void PetModel::ff_layer_tan(const PackedEdges& pk, int L, const LayerSaves& sav, LayerTangents& ts, int K,
                            View2D node_t, View2D ie_t, View2D x4_t, View2D cf, View2D cf_t) {
  const int N = node_t.extent(0) / K, E = pk.E, D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  const int S = cf.extent(1), T = N + E;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L), tag = "hv" + ls;
  const bool ln = h_.normalization == Normalization::LayerNorm;
  auto off = pk.off, rev = pk.reverse;
  const Tokens tk{off, N, false};
  const AttnShape as{tk, N, S, h_.num_heads, h_.head_dim, T,
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature))};
  for (auto* v : {&ts.tokens, &ts.qkv, &ts.node_new, &ts.tmp_center, &ts.eps, &ts.tmp_edge}) v->assign(A, View2D());
  auto keep = [&](const std::string& name, int a, int r, int c) {
    return ws_.n2("t_" + name + (a < 0 ? "" : std::to_string(a)) + "_" + tag, K * r, c);
  };
  Workspace::Scope layer_scope(ws_);

  // compress: z' = input_edge' wi^T + wx (v', |v|'); et' = compress.2(silu'(z) z')
  const CompressFold f = compress_fold(L);
  ts.cpre = keep("cpre", -1, E, D);
  if (f.wi.extent(0) > 0) linear(ts.cpre, ie_t, f.wi, View1D(), 0);
  {
    auto zt = ts.cpre;
    auto wx = f.wx;
    const bool gemm = f.wi.extent(0) > 0;
    Kokkos::parallel_for(
        "hv_compress", RangePolicy(0, K * E * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D;
          Net z = gemm ? zt(q, d) : Net(0);
          for (int c = 0; c < 4; ++c) z += wx(d, c) * x4_t(q, c);
          zt(q, d) = z;
        });
  }
  View2D et_t = ws_.tmp(K * E, D);
  {
    Workspace::Scope s(ws_);
    View2D a_t = ws_.tmp(K * E, D);
    act_tan(a_t, sav.cpre, ts.cpre);
    linear(et_t, a_t, mat(g + ".compress.2.weight"), View1D(), 0);
  }

  View2D node_cur = node_t, edge_cur = et_t;
  for (int a = 0; a < A; ++a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    View2D node_next = ws_.n2("t_node_next_" + std::to_string((L * A + a) % 2), K * N, Dn);
    View2D edge_next = ws_.tmp(K * E, D);
    Workspace::Scope block_scope(ws_);
    View2D in_node = ws_.tmp(K * N, D);
    linear(in_node, node_cur, mat(tl + ".center_contraction.weight"), View1D(), 0);
    View2D tok = ts.tokens[a] = keep("tok", a, T, D);
    Kokkos::parallel_for(
        "hv_tokens", RangePolicy(0, K * T * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D, t = q / T, r = q % T;
          tok(q, d) = r < N ? in_node(t * N + r, d) : edge_cur(t * E + r - N, d);
        });
    View2D attn_in = ws_.tmp(K * T, D);
    norm_tan(attn_in, sav.tokens[a], tok, vec(tl + ".norm_attention.weight"), ln);
    View2D qkv = ts.qkv[a] = keep("qkv", a, T, 3 * D);
    linear(qkv, attn_in, mat(tl + ".attention.input_linear.weight"), View1D(), 0);
    View2D merged = ws_.tmp(K * T, D), attn_out = ws_.tmp(K * T, D);
    attn_tan(merged, sav.qkv[a], qkv, cf, cf_t, as, K);
    linear(attn_out, merged, mat(tl + ".attention.output_linear.weight"), View1D(), 0);

    View2D atom_rows = ws_.tmp(K * N, D), node_exp = ws_.tmp(K * N, Dn);
    View2D eps = ts.eps[a] = keep("eps", a, E, D);
    Kokkos::parallel_for(
        "hv_split", RangePolicy(0, K * T * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D, t = q / T, r = q % T;
          if (r < N) atom_rows(t * N + r, d) = attn_out(q, d);
          else eps(t * E + r - N, d) = attn_out(q, d) + edge_cur(t * E + r - N, d);
        });
    linear(node_exp, atom_rows, mat(tl + ".center_expansion.weight"), View1D(), 0);
    View2D node_res = ts.node_new[a] = keep("nn", a, N, Dn);
    Kokkos::parallel_for(
        "hv_nres", RangePolicy(0, K * N * Dn),
        KOKKOS_LAMBDA(int i) { node_res(i / Dn, i % Dn) = node_cur(i / Dn, i % Dn) + node_exp(i / Dn, i % Dn); });

    View2D node_norm = ws_.tmp(K * N, Dn), edge_norm = ws_.tmp(K * E, D);
    norm_tan(node_norm, sav.node_new[a], node_res, vec(tl + ".norm_center_features.weight"), ln);
    norm_tan(edge_norm, sav.eps[a], eps, vec(tl + ".norm_mlp.weight"), ln);
    Kokkos::deep_copy(ExecSpace(), node_next, node_res);
    Kokkos::deep_copy(ExecSpace(), edge_next, eps);
    ts.tmp_center[a] = keep("tc", a, N, sav.tmp_center[a].extent(1));
    ts.tmp_edge[a] = keep("te", a, E, sav.tmp_edge[a].extent(1));
    feedforward_tan(node_next, node_norm, tl + ".center_mlp", sav.tmp_center[a], ts.tmp_center[a], 1);
    feedforward_tan(edge_next, edge_norm, tl + ".mlp", sav.tmp_edge[a], ts.tmp_edge[a], 1);
    node_cur = node_next;
    edge_cur = edge_next;
  }
  Kokkos::deep_copy(ExecSpace(), node_t, node_cur);

  // input_edge' += out_edge' + comb_mlp'(layernorm'([out_edge' | its reverse's]))
  const View2D out_edge = edge_cur;
  View2D concat = ts.concat = keep("cc", -1, E, 2 * D);
  Kokkos::parallel_for(
      "hv_concat", RangePolicy(0, K * E * 2 * D), KOKKOS_LAMBDA(int i) {
        const int q = i / (2 * D), d = i % (2 * D), t = q / E, k = q % E;
        if (d < D) {
          concat(q, d) = out_edge(q, d);
          ie_t(q, d) += out_edge(q, d);
        } else {
          concat(q, d) = rev(k) >= 0 ? out_edge(t * E + rev(k), d - D) : Net(0);
        }
      });
  View2D cnorm = ws_.tmp(K * E, 2 * D), cph = ws_.tmp(K * E, 2 * D);
  norm_tan(cnorm, sav.concat, concat, vec("combination_norms." + ls + ".weight"), true);
  ts.cph = keep("cph", -1, E, 2 * D);
  linear(ts.cph, cnorm, mat("combination_mlps." + ls + ".0.weight"), View1D(), 0);
  act_tan(cph, sav.cph, ts.cph);
  linear(ie_t, cph, mat("combination_mlps." + ls + ".2.weight"), View1D(), 1);
}

void PetModel::ff_layer_bwd2(const PackedEdges& pk, int L, const LayerSaves& sav, const LayerTangents& ts, int K,
                             View2D cf, View2D cf_t, View2D node_adj, View2D node_adj_t, View2D ie_adj,
                             View2D ie_adj_t, View2D x4_adj, View2D x4_adj_t, View2D cf_adj, View2D cf_adj_t) {
  const int N = node_adj.extent(0), E = pk.E, D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  const int S = cf.extent(1), T = N + E;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  const bool ln = h_.normalization == Normalization::LayerNorm;
  auto off = pk.off, rev = pk.reverse;
  const Tokens tk{off, N, false};
  const AttnShape as{tk, N, S, h_.num_heads, h_.head_dim, T,
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature))};
  Workspace::Scope layer_scope(ws_);
  View2D edge_adj = ws_.tmp(E, D), edge_adj_t = ws_.tmp(K * E, D);

  // The combination MLP; input_edge's adjoint passes through to the layer's input.
  {
    Workspace::Scope s(ws_);
    View2D a = ws_.tmp(E, 2 * D), a_t = ws_.tmp(K * E, 2 * D), c = ws_.tmp(E, 2 * D), c_t = ws_.tmp(K * E, 2 * D);
    const WeightRef w2 = mat("combination_mlps." + ls + ".2.weight"), w0 = mat("combination_mlps." + ls + ".0.weight");
    linear_bwd(a, ie_adj, w2, 0);
    linear_bwd(a_t, ie_adj_t, w2, 0);
    act_bwd2(c, c_t, a, a_t, sav.cph, ts.cph);
    View2D cn = ws_.tmp(E, 2 * D), cn_t = ws_.tmp(K * E, 2 * D);
    linear_bwd(cn, c, w0, 0);
    linear_bwd(cn_t, c_t, w0, 0);
    View2D ca = ws_.tmp(E, 2 * D), ca_t = ws_.tmp(K * E, 2 * D);
    norm_bwd2(ca, ca_t, cn, cn_t, sav.concat, ts.concat, vec("combination_norms." + ls + ".weight"), true, false);
    Kokkos::parallel_for(
        "hv_bw_concat", RangePolicy(0, (K + 1) * E * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D, t = q / E - 1, k = q % E, r = rev(k);
          if (t < 0) {
            edge_adj(k, d) = ie_adj(k, d) + ca(k, d) + (r >= 0 ? ca(r, D + d) : Net(0));
          } else {
            const int tq = t * E + k;
            edge_adj_t(tq, d) = ie_adj_t(tq, d) + ca_t(tq, d) + (r >= 0 ? ca_t(t * E + r, D + d) : Net(0));
          }
        });
  }

  for (int a = A - 1; a >= 0; --a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    // The two MLPs and their norms: edge_adj, node_adj become eps's and node_res's.
    {
      Workspace::Scope s(ws_);
      View2D en = ws_.tmp(E, D), en_t = ws_.tmp(K * E, D), cn = ws_.tmp(N, Dn), cn_t = ws_.tmp(K * N, Dn);
      feedforward_bwd2(en, en_t, edge_adj, edge_adj_t, tl + ".mlp", sav.tmp_edge[a], ts.tmp_edge[a], 0);
      norm_bwd2(edge_adj, edge_adj_t, en, en_t, sav.eps[a], ts.eps[a], vec(tl + ".norm_mlp.weight"), ln, true);
      feedforward_bwd2(cn, cn_t, node_adj, node_adj_t, tl + ".center_mlp", sav.tmp_center[a], ts.tmp_center[a], 0);
      norm_bwd2(node_adj, node_adj_t, cn, cn_t, sav.node_new[a], ts.node_new[a],
                vec(tl + ".norm_center_features.weight"), ln, true);
    }
    Workspace::Scope block_scope(ws_);
    // The attention's output adjoint: the expansion's input on the atom rows, eps's
    // on the edge rows; then back through the output linear, the heads, the input
    // linear and the token norm.
    View2D ao = ws_.tmp(T, D), ao_t = ws_.tmp(K * T, D);
    {
      Workspace::Scope s(ws_);
      View2D an = ws_.tmp(N, D), an_t = ws_.tmp(K * N, D);
      const WeightRef we = mat(tl + ".center_expansion.weight");
      linear_bwd(an, node_adj, we, 0);
      linear_bwd(an_t, node_adj_t, we, 0);
      Kokkos::parallel_for(
          "hv_bw_ao", RangePolicy(0, (K + 1) * T * D), KOKKOS_LAMBDA(int i) {
            const int q = i / D, d = i % D, t = q / T - 1, r = q % T;
            if (t < 0) ao(r, d) = r < N ? an(r, d) : edge_adj(r - N, d);
            else ao_t(t * T + r, d) = r < N ? an_t(t * N + r, d) : edge_adj_t(t * E + r - N, d);
          });
    }
    View2D tin = ws_.tmp(T, D), tin_t = ws_.tmp(K * T, D);
    {
      Workspace::Scope s(ws_);
      const WeightRef wo = mat(tl + ".attention.output_linear.weight"), wi = mat(tl + ".attention.input_linear.weight");
      View2D dm = ws_.tmp(T, D), dm_t = ws_.tmp(K * T, D);
      linear_bwd(dm, ao, wo, 0);
      linear_bwd(dm_t, ao_t, wo, 0);
      View2D qa = ws_.tmp(T, 3 * D), qa_t = ws_.tmp(K * T, 3 * D), stats = ws_.tmp(K * N * h_.num_heads * S, 5);
      attn_bwd2(qa, qa_t, cf_adj, cf_adj_t, dm, dm_t, sav.qkv[a], ts.qkv[a], cf, cf_t, stats, as, K);
      View2D ai = ws_.tmp(T, D), ai_t = ws_.tmp(K * T, D);
      linear_bwd(ai, qa, wi, 0);
      linear_bwd(ai_t, qa_t, wi, 0);
      norm_bwd2(tin, tin_t, ai, ai_t, sav.tokens[a], ts.tokens[a], vec(tl + ".norm_attention.weight"), ln, false);
    }
    // Tokens back where they came from: edge rows onto eps's residual share, atom
    // rows through the contraction onto node_in's.
    View2D inn = ws_.tmp(N, D), inn_t = ws_.tmp(K * N, D);
    Kokkos::parallel_for(
        "hv_bw_tok", RangePolicy(0, (K + 1) * T * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D, t = q / T - 1, r = q % T;
          if (t < 0) {
            if (r < N) inn(r, d) = tin(r, d);
            else edge_adj(r - N, d) += tin(r, d);
          } else {
            if (r < N) inn_t(t * N + r, d) = tin_t(t * T + r, d);
            else edge_adj_t(t * E + r - N, d) += tin_t(t * T + r, d);
          }
        });
    const WeightRef wc = mat(tl + ".center_contraction.weight");
    linear_bwd(node_adj, inn, wc, 1);
    linear_bwd(node_adj_t, inn_t, wc, 1);
  }

  // edge_adj is et's: back through compress.2 and compress.0.
  const CompressFold f = compress_fold(L);
  View2D ca = ws_.tmp(E, D), ca_t = ws_.tmp(K * E, D), gz = ws_.tmp(E, D), gz_t = ws_.tmp(K * E, D);
  const WeightRef w2 = mat(g + ".compress.2.weight");
  linear_bwd(ca, edge_adj, w2, 0);
  linear_bwd(ca_t, edge_adj_t, w2, 0);
  act_bwd2(gz, gz_t, ca, ca_t, sav.cpre, ts.cpre);
  auto wx = f.wx;
  Kokkos::parallel_for(
      "hv_bw_x4", RangePolicy(0, (K + 1) * E), KOKKOS_LAMBDA(int q) {
        const int t = q / E - 1, k = q % E;
        for (int c = 0; c < 4; ++c) {
          Net s = Net(0);
          for (int d = 0; d < D; ++d) s += (t < 0 ? gz(k, d) : gz_t(t * E + k, d)) * wx(d, c);
          if (t < 0) x4_adj(k, c) += s;
          else x4_adj_t(t * E + k, c) += s;
        }
      });
  if (f.wi.extent(0) > 0) {
    linear_bwd(ie_adj, gz, f.wi, 1);
    linear_bwd(ie_adj_t, gz_t, f.wi, 1);
  }
}

// The readout's tangents: of the heads' pre-activations, and of the edge
// predictions, which the cutoff factor's adjoint is.
void PetModel::readout_tan(const ReadoutSaves& sav, ReadoutTangents& ts, int K, View2D node_t, View2D edge_t) {
  const int N = sav.nh0[0].extent(0), E = sav.eh0[0].extent(0), Dh = h_.d_head;
  const std::string nh = "node_heads.energy.0", eh = "edge_heads.energy.0";
  ts.nh0 = ws_.n2("t_nh0", K * N, Dh), ts.nh1 = ws_.n2("t_nh1", K * N, Dh);
  ts.eh0 = ws_.n2("t_eh0", K * E, Dh), ts.eh1 = ws_.n2("t_eh1", K * E, Dh), ts.epred = ws_.n2("t_epred", K * E, 1);
  Workspace::Scope s(ws_);
  View2D a = ws_.tmp(K * N, Dh), b = ws_.tmp(K * E, Dh);
  linear(ts.nh0, node_t, mat(nh + ".0.weight"), View1D(), 0);
  act_tan(a, sav.nh0[0], ts.nh0);
  linear(ts.nh1, a, mat(nh + ".2.weight"), View1D(), 0);
  linear(ts.eh0, edge_t, mat(eh + ".0.weight"), View1D(), 0);
  act_tan(b, sav.eh0[0], ts.eh0);
  linear(ts.eh1, b, mat(eh + ".2.weight"), View1D(), 0);
  act_tan(b, sav.eh1[0], ts.eh1);
  linear(ts.epred, b, mat("edge_last_layers.energy.0.energy___0.weight"), View1D(), 0);
}

void PetModel::readout_bwd2(const ReadoutSaves& sav, const ReadoutTangents& ts, const PackedEdges& pk, int K,
                            View1D cut_t, View2D node_adj, View2D node_adj_t, View2D edge_adj, View2D edge_adj_t,
                            View1D cutoff_adj, View1D cutoff_adj_t) {
  const int N = node_adj.extent(0), E = pk.E, Dh = h_.d_head;
  Workspace::Scope s(ws_);
  View2D npa = ws_.tmp(N, 1), epa = ws_.tmp(E, 1), epa_t = ws_.tmp(K * E, 1);
  auto cut = pk.cut;
  auto epred = sav.epred[0], epred_t = ts.epred;
  Kokkos::parallel_for("hv_ro_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { npa(n, 0) = Net(1); });
  Kokkos::parallel_for(
      "hv_ro_e", RangePolicy(0, (K + 1) * E), KOKKOS_LAMBDA(int q) {
        const int t = q / E - 1, k = q % E;
        if (t < 0) epa(k, 0) = cut(k), cutoff_adj(k) = epred(k, 0);
        else epa_t(t * E + k, 0) = cut_t(t * E + k), cutoff_adj_t(t * E + k) = epred_t(t * E + k, 0);
      });
  const std::string nh = "node_heads.energy.0", eh = "edge_heads.energy.0";
  // node: the seed is constant, so the tangents start at the activations'.
  View2D a1 = ws_.tmp(N, Dh), g1 = ws_.tmp(N, Dh), g1_t = ws_.tmp(K * N, Dh);
  linear_bwd(a1, npa, mat("node_last_layers.energy.0.energy___0.weight"), 0);
  act_bwd2(g1, g1_t, a1, View2D(), sav.nh1[0], ts.nh1);
  View2D a0 = ws_.tmp(N, Dh), a0_t = ws_.tmp(K * N, Dh), g0 = ws_.tmp(N, Dh), g0_t = ws_.tmp(K * N, Dh);
  linear_bwd(a0, g1, mat(nh + ".2.weight"), 0);
  linear_bwd(a0_t, g1_t, mat(nh + ".2.weight"), 0);
  act_bwd2(g0, g0_t, a0, a0_t, sav.nh0[0], ts.nh0);
  linear_bwd(node_adj, g0, mat(nh + ".0.weight"), 0);
  linear_bwd(node_adj_t, g0_t, mat(nh + ".0.weight"), 0);
  // edges: the seed is the cutoff factor, whose tangent enters here.
  View2D b1 = ws_.tmp(E, Dh), b1_t = ws_.tmp(K * E, Dh), h1 = ws_.tmp(E, Dh), h1_t = ws_.tmp(K * E, Dh);
  const WeightRef wl = mat("edge_last_layers.energy.0.energy___0.weight");
  linear_bwd(b1, epa, wl, 0);
  linear_bwd(b1_t, epa_t, wl, 0);
  act_bwd2(h1, h1_t, b1, b1_t, sav.eh1[0], ts.eh1);
  View2D b0 = ws_.tmp(E, Dh), b0_t = ws_.tmp(K * E, Dh), h0 = ws_.tmp(E, Dh), h0_t = ws_.tmp(K * E, Dh);
  linear_bwd(b0, h1, mat(eh + ".2.weight"), 0);
  linear_bwd(b0_t, h1_t, mat(eh + ".2.weight"), 0);
  act_bwd2(h0, h0_t, b0, b0_t, sav.eh0[0], ts.eh0);
  linear_bwd(edge_adj, h0, mat(eh + ".0.weight"), 0);
  linear_bwd(edge_adj_t, h0_t, mat(eh + ".0.weight"), 0);
}

// ---- the whole product --------------------------------------------------------------

HvpOut PetModel::hvp(const DeviceEdgeData& dev, int K, RView2D vdot_raw) {
  if (h_.featurizer_type != FeaturizerType::FeedForward)
    throw std::runtime_error("pet: the Hessian-vector product supports the feedforward featurizer only (so far)");
  if (h_.adaptive() && h_.adaptive_cutoff_method != AdaptiveCutoffMethod::Solver)
    throw std::runtime_error("pet: the Hessian-vector product supports the solver adaptive cutoff only (so far)");
  if (h_.num_readout_layers != 1 || dev.exchange || dev.n_struct != 1 || dev.edge_capacity || dev.padded ||
      (dev.n_local >= 0 && dev.n_local != dev.n_atoms) || dev.centres() != dev.n_atoms)
    throw std::runtime_error("pet: the Hessian-vector product takes one whole structure");
  const int R = dev.n_raw;
  if ((int) dev.raw_slot.extent(0) != R || (int) vdot_raw.extent(0) != K * R)
    throw std::runtime_error("pet: the Hessian-vector product needs the raw list's slots and K tangents per raw edge");
  if (h_.adaptive() && ((int) dev.raw_off.extent(0) != dev.n_atoms + 1 || (int) dev.adapt_r.extent(0) != dev.n_atoms))
    throw std::runtime_error("pet: the Hessian-vector product needs the raw list grouped by centre, and the solver's roots");

  const int N = dev.n_atoms, S = dev.max_neighbors + 1, D = h_.d_pet, Dn = h_.d_node, G = h_.num_gnn_layers;
  Workspace::Scope scope(ws_);
  ws_.set_zero(false);
  const PackedEdges pk = pack_edges(dev);
  const int E = pk.E;
  const bool adaptive = h_.adaptive(), bump = h_.cutoff_function == CutoffFunction::Bump;
  const double width = h_.cutoff_width, wad = h_.cutoff_width_adaptive, target = h_.num_neighbors_adaptive,
               rmax = h_.cutoff, scale = energy_scale_;

  // ---- the geometry's tangents: each packed edge's raw edge, d', the cutoffs'
  auto rcen = dev.raw_center, rnb = dev.raw_neigh, rslot = dev.raw_slot, roff = dev.raw_off;
  auto rvec = dev.raw_vec;
  auto rdist = dev.raw_dist;
  auto slot_edge = pk.slot_edge, off = pk.off, center = pk.center, rev = pk.reverse;
  auto pvec = pk.vec;
  auto pdist = pk.dist, pcut = pk.pcut;
  auto ar = dev.adapt_r, adn = dev.adapt_dn;
  IView1D praw = ws_.i1("hv:praw", E);
  RView1D ddot = ws_.r1("hv:ddot", K * R), rcdot = ws_.r1("hv:rcdot", K * N), pcdot = ws_.r1("hv:pcdot", K * E);
  Kokkos::parallel_for(
      "hv_praw", RangePolicy(0, R), KOKKOS_LAMBDA(int e) {
        if (rslot(e) >= 0 && slot_edge(rslot(e)) >= 0) praw(slot_edge(rslot(e))) = e;
      });
  Kokkos::parallel_for(
      "hv_ddot", RangePolicy(0, K * R), KOKKOS_LAMBDA(int q) {
        const int e = q % R;
        const double d = rdist(e);
        ddot(q) = d > 0.0 ? (rvec(e, 0) * vdot_raw(q, 0) + rvec(e, 1) * vdot_raw(q, 1) + rvec(e, 2) * vdot_raw(q, 2)) / d
                          : 0.0;
      });
  // An atom's cutoff c(d): c' = sum over its raw edges of dc/dd d', dc/dd = -n_d / n_r
  // at the root (0 when clamped).
  Kokkos::parallel_for(
      "hv_rcdot", RangePolicy(0, K * N), KOKKOS_LAMBDA(int q) {
        const int t = q / N, a = q % N;
        double s = 0.0;
        if (adaptive && adn(a) > 0.0)
          for (int e = roff(a); e < roff(a + 1); ++e) {
            double f1, f2;
            cut_derivs(rdist(e), ar(a), wad, true, f1, f2);
            s += -f1 / adn(a) * ddot(t * R + e);
          }
        rcdot(q) = s;
      });
  View2D x4_t = ws_.n2("hv:x4_t", K * E, 4), cf_t = ws_.n2("hv:cf_t", K * N, S);
  View1D cut_t = ws_.n1("hv:cut_t", K * E);
  Kokkos::deep_copy(ExecSpace(), cf_t, Net(0));
  auto cf_seq = dev.cf_seq;
  Kokkos::parallel_for(
      "hv_x4_t", RangePolicy(0, K * E), KOKKOS_LAMBDA(int q) {
        const int t = q / E, k = q % E, e = praw(k), n = center(k);
        for (int c = 0; c < 3; ++c) x4_t(q, c) = (Net) vdot_raw(t * R + e, c);
        const double dd = ddot(t * R + e);
        x4_t(q, 3) = (Net) dd;
        const double pc_t = adaptive ? 0.5 * (rcdot(t * N + n) + rcdot(t * N + rnb(e))) : 0.0;
        pcdot(q) = pc_t;
        double f1, f2;
        cut_derivs(pdist(k), pcut(k), width, bump, f1, f2);
        cut_t(q) = (Net) (f1 * (dd - pc_t));
        cf_t(t * N + n, 1 + k - off(n)) = cut_t(q);
      });

  // ---- the network: the forward with every save, and its tangents layer by layer
  View2D node = ws_.n2("node", N, Dn), input_edge = ws_.n2("input_edge", E, D);
  gather(node, mat("node_embedders.0.weight").v, dev.species);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);
  const View2D cond = conditioning(dev);
  View2D node_t = ws_.n2("hv:node_t", K * N, Dn), ie_t = ws_.n2("hv:ie_t", K * E, D);
  Kokkos::deep_copy(ExecSpace(), node_t, Net(0));
  Kokkos::deep_copy(ExecSpace(), ie_t, Net(0));
  std::vector<LayerSaves> sav(G);
  std::vector<LayerTangents> tan(G);
  for (int L = 0; L < G; ++L) {
    sav[L].tag = std::to_string(L);
    ff_layer(dev, pk, L, node, input_edge, cond, &sav[L], true);
    ff_layer_tan(pk, L, sav[L], tan[L], K, node_t, ie_t, x4_t, cf_seq, cf_t);
  }
  View1D net = ws_.n1("per_atom_net", N);
  ReadoutSaves rs;
  readout({node}, {input_edge}, View2D(net.data(), N, 1), pk, &rs, "", "energy");
  ReadoutTangents rt;
  readout_tan(rs, rt, K, node_t, ie_t);

  // ---- the backward, adjoints and their tangents together
  View2D x4_adj = ws_.n2("x4_adj", E, 4), x4_adj_t = ws_.n2("hv:x4_adj_t", K * E, 4);
  View2D cf_adj = ws_.n2("cf_seq_adj", N, S), cf_adj_t = ws_.n2("hv:cf_adj_t", K * N, S);
  for (auto v : {x4_adj, x4_adj_t, cf_adj, cf_adj_t}) Kokkos::deep_copy(ExecSpace(), v, Net(0));
  View2D node_adj = ws_.n2("node_adj", N, Dn), node_adj_t = ws_.n2("hv:node_adj_t", K * N, Dn);
  View2D ie_adj = ws_.n2("ie_adj", E, D), ie_adj_t = ws_.n2("hv:ie_adj_t", K * E, D);
  View1D cutoff_adj = ws_.n1("cutoff_adj", E), cutoff_adj_t = ws_.n1("hv:cutoff_adj_t", K * E);
  readout_bwd2(rs, rt, pk, K, cut_t, node_adj, node_adj_t, ie_adj, ie_adj_t, cutoff_adj, cutoff_adj_t);
  for (int L = G - 1; L >= 0; --L)
    ff_layer_bwd2(pk, L, sav[L], tan[L], K, cf_seq, cf_t, node_adj, node_adj_t, ie_adj, ie_adj_t, x4_adj, x4_adj_t,
                  cf_adj, cf_adj_t);

  // ---- the geometry: dE/dv per raw edge and its tangents
  //   T = cutoff_adj + cf_adj (the factor's adjoint), dist_adj = x4_adj[3] + T f1,
  //   g = x4_adj[:3] + dist_adj v / d, pc_adj = -T f1,
  // and d/dt of each, f1' = f2 (d' - pc').
  RView2D g = ws_.r2("hv:g", R, 3), g_t = ws_.r2("hv:g_t", K * R, 3);
  RView1D pc_adj = ws_.r1("hv:pc_adj", E), pc_adj_t = ws_.r1("hv:pc_adj_t", K * E);
  Kokkos::deep_copy(ExecSpace(), g, 0.0);
  Kokkos::deep_copy(ExecSpace(), g_t, 0.0);
  Kokkos::parallel_for(
      "hv_edge_grad", RangePolicy(0, (K + 1) * E), KOKKOS_LAMBDA(int q) {
        const int t = q / E - 1, k = q % E, n = center(k), m = k - off(n), e = praw(k);
        const double d = pdist(k), invd = d > 0.0 ? 1.0 / d : 0.0;
        double f1, f2;
        cut_derivs(d, pcut(k), width, bump, f1, f2);
        const double T0 = (double) cutoff_adj(k) + (double) cf_adj(n, 1 + m);
        const double dist_adj = (double) x4_adj(k, 3) + T0 * f1;
        if (t < 0) {
          for (int c = 0; c < 3; ++c) g(e, c) = scale * ((double) x4_adj(k, c) + dist_adj * pvec(k, c) * invd);
          pc_adj(k) = -T0 * f1;
          return;
        }
        const int tk = t * E + k;
        const double Tt = (double) cutoff_adj_t(tk) + (double) cf_adj_t(t * N + n, 1 + m);
        const double dd = ddot(t * R + e), f1t = f2 * (dd - pcdot(tk));
        const double dist_adj_t = (double) x4_adj_t(tk, 3) + Tt * f1 + T0 * f1t;
        for (int c = 0; c < 3; ++c) {
          const double vt = vdot_raw(t * R + e, c);
          g_t(t * R + e, c) = scale * ((double) x4_adj_t(tk, c) + dist_adj_t * pvec(k, c) * invd +
                                       dist_adj * (vt * invd - pvec(k, c) * dd * invd * invd));
        }
        pc_adj_t(tk) = -(Tt * f1 + T0 * f1t);
      });

  if (adaptive) {
    // Each atom's cutoff adjoint A, half from each end of its edges; then per raw
    // edge e of atom c, A dc/dd_e v / d with q = dc/dd = -n_d / n_r and
    //   q' = -n_d' / n_r + n_d n_r' / n_r^2,
    //   n_d' = f2 (d' - r'),  n_r' = sum over c's raw edges of f2 (r' - d') + b''(r) r',
    // b(r) = target (r / rmax)^3 the count's baseline: the root moves, and so does
    // the slope there.
    RView1D A = ws_.r1("hv:A", N), A_t = ws_.r1("hv:A_t", K * N), nr_t = ws_.r1("hv:nr_t", K * N);
    Kokkos::parallel_for(
        "hv_adapted_adj", RangePolicy(0, (K + 1) * N), KOKKOS_LAMBDA(int q) {
          const int t = q / N - 1, a = q % N;
          double s = 0.0;
          for (int k = off(a); k < off(a + 1); ++k) {
            const double own = t < 0 ? pc_adj(k) : pc_adj_t(t * E + k);
            const double back = rev(k) < 0 ? 0.0 : t < 0 ? pc_adj(rev(k)) : pc_adj_t(t * E + rev(k));
            s += 0.5 * (own + back);
          }
          if (t < 0) A(a) = s;
          else A_t(t * N + a) = s;
        });
    Kokkos::parallel_for(
        "hv_nr_t", RangePolicy(0, K * N), KOKKOS_LAMBDA(int q) {
          const int t = q / N, a = q % N;
          if (adn(a) <= 0.0) return (void) (nr_t(q) = 0.0);
          const double r = ar(a), rt = rcdot(q);
          double s = 6.0 * target * r / (rmax * rmax * rmax) * rt;
          for (int e = roff(a); e < roff(a + 1); ++e) {
            double f1, f2;
            cut_derivs(rdist(e), r, wad, true, f1, f2);
            s += f2 * (rt - ddot(t * R + e));
          }
          nr_t(q) = s;
        });
    Kokkos::parallel_for(
        "hv_adaptive_g", RangePolicy(0, (K + 1) * R), KOKKOS_LAMBDA(int q) {
          const int t = q / R - 1, e = q % R, c = rcen(e);
          const double nr = adn(c), d = rdist(e);
          if (nr <= 0.0 || d <= 0.0) return;
          double f1, f2;
          cut_derivs(d, ar(c), wad, true, f1, f2);
          const double qe = -f1 / nr, invd = 1.0 / d;
          if (t < 0) {
            for (int x = 0; x < 3; ++x) g(e, x) += scale * A(c) * qe * rvec(e, x) * invd;
            return;
          }
          const int tn = t * N + c;
          const double dd = ddot(t * R + e);
          const double qt = -f2 * (dd - rcdot(tn)) / nr + f1 * nr_t(tn) / (nr * nr);
          for (int x = 0; x < 3; ++x) {
            const double vt = vdot_raw(t * R + e, x);
            g_t(t * R + e, x) += scale * (A_t(tn) * qe * rvec(e, x) * invd + A(c) * qt * rvec(e, x) * invd +
                                          A(c) * qe * (vt * invd - rvec(e, x) * dd * invd * invd));
          }
        });
  }

  // ---- folds: F(a) = sum over a's edges of g - sum over edges into a of g; the
  // virial W = sum v g^T, and W' = sum v' g^T + v g'^T.
  HvpOut out;
  out.forces = ws_.r2("hv:forces", N, 3);
  out.dforces = ws_.r2("hv:dforces", K * N, 3);
  out.dvir9 = ws_.r2("hv:dvir9", K, 9);
  for (auto v : {out.forces, out.dforces, out.dvir9}) Kokkos::deep_copy(ExecSpace(), v, 0.0);
  auto F = out.forces, dF = out.dforces, dW = out.dvir9;
  Kokkos::parallel_for(
      "hv_fold", RangePolicy(0, (K + 1) * R), KOKKOS_LAMBDA(int q) {
        const int t = q / R - 1, e = q % R, i = rcen(e), j = rnb(e);
        for (int c = 0; c < 3; ++c) {
          if (t < 0) {
            Kokkos::atomic_add(&F(i, c), g(e, c));
            Kokkos::atomic_add(&F(j, c), -g(e, c));
          } else {
            Kokkos::atomic_add(&dF(t * N + i, c), g_t(t * R + e, c));
            Kokkos::atomic_add(&dF(t * N + j, c), -g_t(t * R + e, c));
          }
        }
      });
  Kokkos::parallel_for(
      "hv_virial", RangePolicy(0, K * 9), KOKKOS_LAMBDA(int q) {
        const int t = q / 9, x = (q % 9) / 3, y = q % 3;
        double s = 0.0;
        for (int e = 0; e < R; ++e) s += vdot_raw(t * R + e, x) * g(e, y) + rvec(e, x) * g_t(t * R + e, y);
        dW(t, x * 3 + y) = s;
      });
  return out;
}

}  // namespace pet
