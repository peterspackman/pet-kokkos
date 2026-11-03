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
// Both featurizers, and the solver, grid or no adaptive cutoff; a single
// structure, every activation kept, no CUDA graph and no exchange. The
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
KOKKOS_INLINE_FUNCTION void cut_derivs(double d, double rc, double w, int kind, double& f1, double& f2) {
  f1 = f2 = 0.0;
  const double x = (d - (rc - w)) / w;
  if (x <= 0.0 || x >= 1.0) return;
  const double pi = detail::PET_PI;
  if (kind == 2) {  // f = 1 - (35x^4 - 84x^5 + 70x^6 - 20x^7)
    const double y = x * (1.0 - x);
    f1 = -140.0 * y * y * y / w;
    f2 = -420.0 * y * y * (1.0 - 2.0 * x) / (w * w);
  } else if (kind == 0) {  // f = (1 + tanh(cot(pi x))) / 2
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

// The attention passes hold several head-sized arrays a thread. Under HIP's
// default launch bound of 1024 an MI250X caps a thread at 128 registers and they
// spill (as attention.cpp found for the forward's passes); 256 lets them fit.
#if defined(KOKKOS_ENABLE_HIP)
using AttnRange = Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                      Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
#else
using AttnRange = RangePolicy;
#endif

struct AttnShape {
  Tokens tk;
  int N, S, H, hd, T;
  Net sc;
  double temperature;
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
      "hv_attn", AttnRange(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
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

// The backward's tangents, from dm (the merged output's adjoint) and its
// tangent, into qkv_adj_t and onto cf_adj_t. stats [K N H S, 5] is scratch.
template <int HD>
void attn_bwd2_impl(View2D qkv_adj_t, View2D cf_adj_t, View2D dm, View2D dm_t,
                    View2D qkv, View2D qkv_t, View2D cf, View2D cf_t, View2D stats, const AttnShape& a, int K) {
  const int N = a.N, S = a.S, H = a.H;
  // Queries: dq, and per query its softmax, delta = go.o and their tangents.
  Kokkos::parallel_for(
      "hv_attn_q", AttnRange(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
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
          qkv_adj_t(ti, qo + d) = dqt[d];
        }
        stats(idx, 0) = m, stats(idx, 1) = invl, stats(idx, 2) = pbar;
        stats(idx, 3) = delta, stats(idx, 4) = delta_t;
      });
  // Keys: dk, dv and the log cf adjoint's share from this head.
  Kokkos::parallel_for(
      "hv_attn_k", AttnRange(0, K * N * H * S), KOKKOS_LAMBDA(int idx) {
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
          Kokkos::atomic_add(&cf_adj_t(t * N + n, j), (bbt - bb * ct / c) / c);
        }
        for (int d = 0; d < nd; ++d) {
          qkv_adj_t(tj, ko + d) = dkt[d], qkv_adj_t(tj, vo + d) = dvt[d];
        }
      });
}
void attn_bwd2(View2D qkv_adj_t, View2D cf_adj_t, View2D dm, View2D dm_t, View2D qkv,
               View2D qkv_t, View2D cf, View2D cf_t, View2D stats, const AttnShape& a, int K) {
  with_head_dim(a.hd, [&](auto hd) {
    attn_bwd2_impl<decltype(hd)::value>(qkv_adj_t, cf_adj_t, dm, dm_t, qkv, qkv_t, cf, cf_t, stats,
                                        a, K);
  });
}

// The attention's forward tangent: merged_t (before the output linear), and the
// queries' pbar for the backward -- on AMD's matrix cores, or on NVIDIA's tensor
// cores in TF32, where it can, from the forward's saves under `key`.
void attention_tangent(Workspace& ws, const std::string& key, View2D merged_t, View2D pbar, View2D qkv, View2D qkv_t,
                       View2D cf, View2D cf_t, const AttnShape& as, int K) {
  if (attention_tan_fast(merged_t, pbar, qkv, qkv_t, ws.peek2(key + ":merged"), ws.peek2(key + ":ml"), cf, cf_t,
                         as.tk, as.S, as.H, as.hd, as.temperature, K))
    return;
  attn_tan(merged_t, qkv, qkv_t, cf, cf_t, as, K);
}

// Its backward, adjoints and tangents: from ao and ao_t (the output's adjoints,
// before the output linear) onto beta times in_adj and in_adj_t (the input
// linear's input), and onto cf_adj and cf_adj_t. The adjoint by the forward's own
// backward (attention.cpp), whose saves are under `key`; in_adj may be ao.
void attention_backward2(Workspace& ws, const std::string& key, View2D in_adj, View2D in_adj_t, View2D ao, View2D ao_t,
                         View2D qkv, View2D qkv_t, View2D merged_t, View2D pbar, View2D cf, View2D cf_t,
                         View2D cf_adj, View2D cf_adj_t, const WeightRef& wi, const WeightRef& wo,
                         const AttnShape& as, int K, Net beta) {
  const int T = as.T, D = as.H * as.hd;
  Workspace::Scope scope(ws);
  View2D dm = ws.tmp(T, D), dm_t = ws.tmp(K * T, D);
  linear_bwd(dm, ao, wo, 0);  // before in_adj, which may be ao, changes
  linear_bwd(dm_t, ao_t, wo, 0);
  attention_bwd(ws, key, in_adj, cf_adj, ao, qkv, cf, wi, wo, as.tk, as.S, as.H, as.hd, as.temperature, beta);
  View2D qa_t = ws.tmp(K * T, 3 * D);
  if (!attention_bwd_tan_fast(ws, qa_t, cf_adj_t, dm, dm_t, qkv, qkv_t, ws.peek2(key + ":merged"), merged_t,
                              ws.peek2(key + ":ml"), pbar, cf, cf_t, as.tk, as.S, as.H, as.hd, as.temperature, K)) {
    View2D stats = ws.tmp(K * as.N * as.H * as.S, 5);
    attn_bwd2(qa_t, cf_adj_t, dm, dm_t, qkv, qkv_t, cf, cf_t, stats, as, K);
  }
  linear_bwd(in_adj_t, qa_t, wi, beta);
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
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature)),
                     h_.attention_temperature};
  for (auto* v : {&ts.tokens, &ts.qkv, &ts.node_new, &ts.tmp_center, &ts.eps, &ts.tmp_edge, &ts.merged, &ts.pbar})
    v->assign(A, View2D());
  auto keep = [&](const std::string& name, int a, int r, int c) {
    return ws_.n2("t_" + name + (a < 0 ? "" : std::to_string(a)) + "_" + tag, K * r, c);
  };
  Workspace::Scope layer_scope(ws_);

  // compress: z' = input_edge' wi^T + wx (v', |v|'); et' = compress.2(silu'(z) z')
  ts.cpre = keep("cpre", -1, E, D);
  View2D et_t = ws_.tmp(K * E, D);
  compress_tan(ts.cpre, et_t, L, sav.cpre, ie_t, x4_t);

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
    View2D merged = ts.merged[a] = keep("mg", a, T, D), attn_out = ws_.tmp(K * T, D);
    ts.pbar[a] = keep("pb", a, N * h_.num_heads * S, 1);
    attention_tangent(ws_, "attn_" + sav.tag + "_" + std::to_string(a), merged, ts.pbar[a], sav.qkv[a], qkv, cf, cf_t,
                      as, K);
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
    copy(node_next, node_res);
    copy(edge_next, eps);
    ts.tmp_center[a] = keep("tc", a, N, sav.tmp_center[a].extent(1));
    ts.tmp_edge[a] = keep("te", a, E, sav.tmp_edge[a].extent(1));
    feedforward_tan(node_next, node_norm, tl + ".center_mlp", sav.tmp_center[a], ts.tmp_center[a], 1);
    feedforward_tan(edge_next, edge_norm, tl + ".mlp", sav.tmp_edge[a], ts.tmp_edge[a], 1);
    node_cur = node_next;
    edge_cur = edge_next;
  }
  copy(node_t, node_cur);

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
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature)),
                     h_.attention_temperature};
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
      View2D ai = ws_.tmp(T, D), ai_t = ws_.tmp(K * T, D);
      attention_backward2(ws_, "attn_" + sav.tag + "_" + std::to_string(a), ai, ai_t, ao, ao_t, sav.qkv[a], ts.qkv[a],
                          ts.merged[a], ts.pbar[a], cf, cf_t, cf_adj, cf_adj_t, wi, wo, as, K, 0);
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
  compress_bwd2(L, edge_adj, edge_adj_t, sav.cpre, ts.cpre, x4_adj, x4_adj_t, ie_adj, ie_adj_t);
}

void PetModel::compress_tan(View2D z_t, View2D et_t, int L, View2D z, View2D ie_t, View2D x4_t) {
  const CompressFold f = compress_fold(L);
  const int D = z_t.extent(1), KE = z_t.extent(0);
  const bool gemm = f.wi.extent(0) > 0;
  if (gemm) linear(z_t, ie_t, f.wi, View1D(), 0);
  auto wx = f.wx;
  Kokkos::parallel_for(
      "hv_compress", RangePolicy(0, KE * D), KOKKOS_LAMBDA(int i) {
        const int q = i / D, d = i % D;
        Net v = gemm ? z_t(q, d) : Net(0);
        for (int c = 0; c < 4; ++c) v += wx(d, c) * x4_t(q, c);
        z_t(q, d) = v;
      });
  Workspace::Scope s(ws_);
  View2D a_t = ws_.tmp(KE, D);
  act_tan(a_t, z, z_t);
  linear(et_t, a_t, mat("gnn_layers." + std::to_string(L) + ".compress.2.weight"), View1D(), 0);
}

void PetModel::compress_bwd2(int L, View2D et_adj, View2D et_adj_t, View2D z, View2D z_t, View2D x4_adj,
                             View2D x4_adj_t, View2D ie_adj, View2D ie_adj_t) {
  const CompressFold f = compress_fold(L);
  const int E = z.extent(0), D = z.extent(1), K = z_t.extent(0) / E;
  Workspace::Scope s(ws_);
  View2D ca = ws_.tmp(E, D), ca_t = ws_.tmp(K * E, D), gz = ws_.tmp(E, D), gz_t = ws_.tmp(K * E, D);
  const WeightRef w2 = mat("gnn_layers." + std::to_string(L) + ".compress.2.weight");
  linear_bwd(ca, et_adj, w2, 0);
  linear_bwd(ca_t, et_adj_t, w2, 0);
  act_bwd2(gz, gz_t, ca, ca_t, z, z_t);
  // x4_adj += gz wx, wx [D, 4]: a narrow GEMM, as the other weights' products are
  const WeightRef wx(f.wx);
  linear_bwd(x4_adj, gz, wx, 1);
  linear_bwd(x4_adj_t, gz_t, wx, 1);
  if (f.wi.extent(0) > 0) {
    linear_bwd(ie_adj, gz, f.wi, 1);
    linear_bwd(ie_adj_t, gz_t, f.wi, 1);
  }
}

// Readout layer i's tangents: of the heads' pre-activations, and of the edge
// predictions, which the cutoff factor's adjoint is.
void PetModel::readout_tan(const ReadoutSaves& sav, ReadoutTangents& ts, int i, int K, View2D node_t,
                           View2D edge_t) {
  const int N = sav.nh0[i].extent(0), E = sav.eh0[i].extent(0), Dh = h_.d_head, R = sav.nh0.size();
  const std::string si = std::to_string(i), nh = "node_heads.energy." + si, eh = "edge_heads.energy." + si;
  for (auto* v : {&ts.nh0, &ts.nh1, &ts.eh0, &ts.eh1, &ts.epred})
    if ((int) v->size() != R) v->assign(R, View2D());
  ts.nh0[i] = ws_.n2("t_nh0_" + si, K * N, Dh), ts.nh1[i] = ws_.n2("t_nh1_" + si, K * N, Dh);
  ts.eh0[i] = ws_.n2("t_eh0_" + si, K * E, Dh), ts.eh1[i] = ws_.n2("t_eh1_" + si, K * E, Dh);
  ts.epred[i] = ws_.n2("t_epred_" + si, K * E, 1);
  Workspace::Scope s(ws_);
  View2D a = ws_.tmp(K * N, Dh), b = ws_.tmp(K * E, Dh);
  linear(ts.nh0[i], node_t, mat(nh + ".0.weight"), View1D(), 0);
  act_tan(a, sav.nh0[i], ts.nh0[i]);
  linear(ts.nh1[i], a, mat(nh + ".2.weight"), View1D(), 0);
  linear(ts.eh0[i], edge_t, mat(eh + ".0.weight"), View1D(), 0);
  act_tan(b, sav.eh0[i], ts.eh0[i]);
  linear(ts.eh1[i], b, mat(eh + ".2.weight"), View1D(), 0);
  act_tan(b, sav.eh1[i], ts.eh1[i]);
  linear(ts.epred[i], b, mat("edge_last_layers.energy." + si + ".energy___0.weight"), View1D(), 0);
}

void PetModel::readout_bwd2(const ReadoutSaves& sav, const ReadoutTangents& ts, int i, const PackedEdges& pk,
                            int K, View1D cut_t, View2D node_adj, View2D node_adj_t, View2D edge_adj,
                            View2D edge_adj_t, View1D cutoff_adj, View1D cutoff_adj_t, bool acc) {
  const int N = node_adj.extent(0), E = pk.E, Dh = h_.d_head;
  Workspace::Scope s(ws_);
  View2D npa = ws_.tmp(N, 1), epa = ws_.tmp(E, 1), epa_t = ws_.tmp(K * E, 1);
  auto cut = pk.cut;
  auto epred = sav.epred[i], epred_t = ts.epred[i];
  Kokkos::parallel_for("hv_ro_n", RangePolicy(0, N), KOKKOS_LAMBDA(int n) { npa(n, 0) = Net(1); });
  Kokkos::parallel_for(
      "hv_ro_e", RangePolicy(0, (K + 1) * E), KOKKOS_LAMBDA(int q) {
        const int t = q / E - 1, k = q % E;
        if (t < 0) {
          epa(k, 0) = cut(k);
          cutoff_adj(k) = (acc ? cutoff_adj(k) : Net(0)) + epred(k, 0);
        } else {
          const int tk = t * E + k;
          epa_t(tk, 0) = cut_t(tk);
          cutoff_adj_t(tk) = (acc ? cutoff_adj_t(tk) : Net(0)) + epred_t(tk, 0);
        }
      });
  const std::string si = std::to_string(i), nh = "node_heads.energy." + si, eh = "edge_heads.energy." + si;
  // node: the seed is constant, so the tangents start at the activations'.
  View2D a1 = ws_.tmp(N, Dh), g1 = ws_.tmp(N, Dh), g1_t = ws_.tmp(K * N, Dh);
  linear_bwd(a1, npa, mat("node_last_layers.energy." + si + ".energy___0.weight"), 0);
  act_bwd2(g1, g1_t, a1, View2D(), sav.nh1[i], ts.nh1[i]);
  View2D a0 = ws_.tmp(N, Dh), a0_t = ws_.tmp(K * N, Dh), g0 = ws_.tmp(N, Dh), g0_t = ws_.tmp(K * N, Dh);
  linear_bwd(a0, g1, mat(nh + ".2.weight"), 0);
  linear_bwd(a0_t, g1_t, mat(nh + ".2.weight"), 0);
  act_bwd2(g0, g0_t, a0, a0_t, sav.nh0[i], ts.nh0[i]);
  linear_bwd(node_adj, g0, mat(nh + ".0.weight"), 0);
  linear_bwd(node_adj_t, g0_t, mat(nh + ".0.weight"), 0);
  // edges: the seed is the cutoff factor, whose tangent enters here.
  View2D b1 = ws_.tmp(E, Dh), b1_t = ws_.tmp(K * E, Dh), h1 = ws_.tmp(E, Dh), h1_t = ws_.tmp(K * E, Dh);
  const WeightRef wl = mat("edge_last_layers.energy." + si + ".energy___0.weight");
  linear_bwd(b1, epa, wl, 0);
  linear_bwd(b1_t, epa_t, wl, 0);
  act_bwd2(h1, h1_t, b1, b1_t, sav.eh1[i], ts.eh1[i]);
  View2D b0 = ws_.tmp(E, Dh), b0_t = ws_.tmp(K * E, Dh), h0 = ws_.tmp(E, Dh), h0_t = ws_.tmp(K * E, Dh);
  linear_bwd(b0, h1, mat(eh + ".2.weight"), 0);
  linear_bwd(b0_t, h1_t, mat(eh + ".2.weight"), 0);
  act_bwd2(h0, h0_t, b0, b0_t, sav.eh0[i], ts.eh0[i]);
  linear_bwd(edge_adj, h0, mat(eh + ".0.weight"), 0);
  linear_bwd(edge_adj_t, h0_t, mat(eh + ".0.weight"), 0);
}

// ---- the residual featurizer ----------------------------------------------------------
//
// tokens = [species embedding ; compress(input_edge, geometry)], then per block
// (PostLN) s1 = tokens + attention(tokens), tok_a = norm(s1), s2 = tok_a +
// mlp(tok_a), tokens = norm(s2). Tangents stacked as [K T, D] token rows; the
// atom rows' embedding is constant.

void PetModel::residual_layer_tan(const PackedEdges& pk, int L, const ResidualSaves& sav, ResidualTangents& ts,
                                  int K, View2D ie_t, View2D tokens_t, View2D x4_t, View2D cf, View2D cf_t) {
  const int N = cf.extent(0), S = cf.extent(1), E = pk.E, D = h_.d_pet, A = h_.num_attention_layers, T = N + E;
  const std::string g = "gnn_layers." + std::to_string(L), tag = "hr" + std::to_string(L);
  const bool ln = h_.normalization == Normalization::LayerNorm;
  const AttnShape as{Tokens{pk.off, N, false}, N, S, h_.num_heads, h_.head_dim, T,
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature)),
                     h_.attention_temperature};
  for (auto* v : {&ts.qkv, &ts.pre, &ts.s1, &ts.s2, &ts.merged, &ts.pbar}) v->assign(A, View2D());
  auto keep = [&](const std::string& name, int a, int r, int c) {
    return ws_.n2("t_" + name + std::to_string(a) + "_" + tag, K * r, c);
  };
  Workspace::Scope layer_scope(ws_);
  ts.cpre = keep("cpre", -1, E, D);
  {
    Workspace::Scope s(ws_);
    View2D et_t = ws_.tmp(K * E, D);
    compress_tan(ts.cpre, et_t, L, sav.cpre, ie_t, x4_t);
    Kokkos::parallel_for(
        "hv_r_tokens", RangePolicy(0, K * T * D), KOKKOS_LAMBDA(int i) {
          const int q = i / D, d = i % D, t = q / T, r = q % T;
          tokens_t(q, d) = r < N ? Net(0) : et_t(t * E + r - N, d);
        });
  }
  for (int a = 0; a < A; ++a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    Workspace::Scope s(ws_);
    View2D qkv = ts.qkv[a] = keep("qkv", a, T, 3 * D), s1 = ts.s1[a] = keep("s1", a, T, D);
    View2D s2 = ts.s2[a] = keep("s2", a, T, D), pre = ts.pre[a] = keep("pre", a, T, sav.pre[a].extent(1));
    linear(qkv, tokens_t, mat(tl + ".attention.input_linear.weight"), View1D(), 0);
    View2D merged = ts.merged[a] = keep("mg", a, T, D);
    ts.pbar[a] = keep("pb", a, N * h_.num_heads * S, 1);
    attention_tangent(ws_, "re_attn_" + sav.tag + "_" + std::to_string(a), merged, ts.pbar[a], sav.qkv[a], qkv, cf,
                      cf_t, as, K);
    copy(s1, tokens_t);
    linear(s1, merged, mat(tl + ".attention.output_linear.weight"), View1D(), 1);
    View2D tok_a = ws_.tmp(K * T, D);
    norm_tan(tok_a, sav.s1[a], s1, vec(tl + ".norm_attention.weight"), ln);
    copy(s2, tok_a);
    feedforward_tan(s2, tok_a, tl + ".mlp", sav.pre[a], pre, 1);
    norm_tan(tokens_t, sav.s2[a], s2, vec(tl + ".norm_mlp.weight"), ln);
  }
}

// tokens_adj (+_t): the adjoint of the layer's output tokens, turned into that of
// its input tokens; then the edge rows go back through compress onto x4_adj and
// ie_adj (+_t), the adjoint of the layer's input_edge.
void PetModel::residual_layer_bwd2(const PackedEdges& pk, int L, const ResidualSaves& sav,
                                   const ResidualTangents& ts, int K, View2D cf, View2D cf_t, View2D tokens_adj,
                                   View2D tokens_adj_t, View2D ie_adj, View2D ie_adj_t, View2D x4_adj,
                                   View2D x4_adj_t, View2D cf_adj, View2D cf_adj_t) {
  const int N = cf.extent(0), S = cf.extent(1), E = pk.E, D = h_.d_pet, A = h_.num_attention_layers, T = N + E;
  const std::string g = "gnn_layers." + std::to_string(L);
  const bool ln = h_.normalization == Normalization::LayerNorm;
  const AttnShape as{Tokens{pk.off, N, false}, N, S, h_.num_heads, h_.head_dim, T,
                     Net(1.0 / (Kokkos::sqrt((double) h_.head_dim) * h_.attention_temperature)),
                     h_.attention_temperature};
  Workspace::Scope layer_scope(ws_);
  for (int a = A - 1; a >= 0; --a) {
    const std::string tl = g + ".trans.layers." + std::to_string(a);
    Workspace::Scope s(ws_);
    // tokens = norm(s2), s2 = tok_a + mlp(tok_a)
    View2D s2a = ws_.tmp(T, D), s2a_t = ws_.tmp(K * T, D), ta = ws_.tmp(T, D), ta_t = ws_.tmp(K * T, D);
    norm_bwd2(s2a, s2a_t, tokens_adj, tokens_adj_t, sav.s2[a], ts.s2[a], vec(tl + ".norm_mlp.weight"), ln, false);
    copy(ta, s2a);
    copy(ta_t, s2a_t);
    feedforward_bwd2(ta, ta_t, s2a, s2a_t, tl + ".mlp", sav.pre[a], ts.pre[a], 1);
    // tok_a = norm(s1), s1 = tokens_in + attention(tokens_in)
    norm_bwd2(tokens_adj, tokens_adj_t, ta, ta_t, sav.s1[a], ts.s1[a], vec(tl + ".norm_attention.weight"), ln,
              false);
    const WeightRef wo = mat(tl + ".attention.output_linear.weight"), wi = mat(tl + ".attention.input_linear.weight");
    // s1 = tokens_in + attention(tokens_in): the attention's share onto the
    // residual's, in place, as the forward's own backward does it
    attention_backward2(ws_, "re_attn_" + sav.tag + "_" + std::to_string(a), tokens_adj, tokens_adj_t, tokens_adj,
                        tokens_adj_t, sav.qkv[a], ts.qkv[a], ts.merged[a], ts.pbar[a], cf, cf_t, cf_adj, cf_adj_t, wi,
                        wo, as, K, 1);
  }
  View2D et = ws_.tmp(E, D), et_t = ws_.tmp(K * E, D);
  Kokkos::parallel_for(
      "hv_r_et", RangePolicy(0, (K + 1) * E * D), KOKKOS_LAMBDA(int i) {
        const int q = i / D, d = i % D, t = q / E - 1, k = q % E;
        if (t < 0) et(k, d) = tokens_adj(N + k, d);
        else et_t(t * E + k, d) = tokens_adj_t(t * T + N + k, d);
      });
  compress_bwd2(L, et, et_t, sav.cpre, ts.cpre, x4_adj, x4_adj_t, ie_adj, ie_adj_t);
}

// ---- the whole product --------------------------------------------------------------

// Recorded as a CUDA graph like the forward (graph.hpp): a Krylov solver asks
// for product after product at one geometry, and at a few dozen atoms the
// launches are most of a call.
HvpOut PetModel::hvp(const DeviceEdgeData& dev, int K, RView2D vdot_raw) {
  auto key = graph_key(dev, true);
  key.insert(key.end(), {0x4876ull, (std::uintptr_t) K, (std::uintptr_t) vdot_raw.data(), vdot_raw.size()});
  auto pass = [&] { return hvp_pass(dev, K, vdot_raw); };
  return graphs_ && !ozaki_active() ? hv_graph_.run(key, pass) : pass();
}

HvpOut PetModel::hvp_pass(const DeviceEdgeData& dev, int K, RView2D vdot_raw) {
  if ((h_.featurizer_type == FeaturizerType::FeedForward && h_.num_readout_layers != 1) || dev.exchange ||
      dev.n_struct != 1 || dev.edge_capacity || dev.padded || (dev.n_local >= 0 && dev.n_local != dev.n_atoms) ||
      dev.centres() != dev.n_atoms)
    throw std::runtime_error("pet: the Hessian-vector product takes one whole structure");
  const int R = dev.n_raw;
  if ((int) dev.raw_slot.extent(0) != R || (int) vdot_raw.extent(0) != K * R)
    throw std::runtime_error("pet: the Hessian-vector product needs the raw list's slots and K tangents per raw edge");
  const bool solver = h_.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver;
  if (h_.adaptive() && ((int) dev.raw_off.extent(0) != dev.n_atoms + 1 ||
                        (solver && (int) dev.adapt_r.extent(0) != dev.n_atoms) || (!solver && n_probes_ < 2)))
    throw std::runtime_error(
        "pet: the Hessian-vector product needs the raw list grouped by centre, and the solver's roots");

  const int N = dev.n_atoms, S = dev.max_neighbors + 1, D = h_.d_pet, Dn = h_.d_node, G = h_.num_gnn_layers;
  if (K > max_index_directions(N, dev.max_neighbors, R))
    throw std::runtime_error("pet: " + std::to_string(K) + " tangent directions over " + std::to_string(N) +
                             " atoms make tensors past the 2^31 elements the kernels index");
  Workspace::Scope scope(ws_);
  ws_.set_zero(false);
  const PackedEdges pk = pack_edges(dev);
  const int E = pk.E;
  const bool adaptive = h_.adaptive();
  const int kind = int(h_.cutoff_function);
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
  // An atom's cutoff c(d) and its tangent c'. The solver's: c' = sum over its raw
  // edges of dc/dd d', dc/dd = -n_d / n_r at the root (0 when clamped).
  const int P = n_probes_;
  auto probes = probes_;
  RView2D geff, gdiff, ggrad, gsgn, gw, gdt, gwt;
  RView1D gc;
  // The solver's count derivatives at the root, per raw edge, the same for every
  // direction: f1, f2 of bump(d_e; r of its centre).
  RView2D sf = ws_.r2("hv:sf", adaptive && solver ? R : 0, 2);
  if (adaptive && solver) {
    Kokkos::parallel_for(
        "hv_sf", RangePolicy(0, R), KOKKOS_LAMBDA(int e) {
          double f1, f2;
          cut_derivs(rdist(e), ar(rcen(e)), wad, 0, f1, f2);
          sf(e, 0) = f1, sf(e, 1) = f2;
        });
    Kokkos::parallel_for(
        "hv_rcdot", TeamPolicy(K * N, 1, kLanes), KOKKOS_LAMBDA(const Team& m) {
          const int q = m.league_rank(), t = q / N, a = q % N;
          double s = 0.0;
          if (adn(a) > 0.0)
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(m, roff(a), roff(a + 1)),
                [&](int e, double& acc) { acc += -sf(e, 0) / adn(a) * ddot(t * R + e); }, s);
          Kokkos::single(Kokkos::PerThread(m), [&] { rcdot(q) = s; });
        });
  } else if (adaptive) {
    // The grid's: eff_p = sum_e bump(d_e, probe p), diff = eff - target + target
    // (p/(P-1))^3, grad = |gradient(diff)| (clamped at 1e-12, sign gsgn), u =
    // -(diff/grad)^2 / 2, w = softmax(u), c = sum_p probe_p w_p; and along each
    // direction eff' = sum_e f1 d', grad' = gsgn gradient(eff'), u' = -r r' with r =
    // diff/grad, w' = w (u' - sum w u'), c' = sum_p probe_p w'_p.
    geff = ws_.r2("hv:geff", N, P), gdiff = ws_.r2("hv:gdiff", N, P), ggrad = ws_.r2("hv:ggrad", N, P);
    gsgn = ws_.r2("hv:gsgn", N, P), gw = ws_.r2("hv:gw", N, P), gc = ws_.r1("hv:gc", N);
    gdt = ws_.r2("hv:gdt", K * N, P), gwt = ws_.r2("hv:gwt", K * N, P);
    auto eff = geff, diff = gdiff, grad = ggrad, sgn = gsgn, w = gw, dt = gdt, wt = gwt;
    auto c = gc;
    Kokkos::parallel_for(
        "hv_grid", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
          for (int p = 0; p < P; ++p) {
            double sum = 0.0;
            for (int e = roff(a); e < roff(a + 1); ++e) sum += detail::dev_bump_cutoff(rdist(e), probes(p), wad);
            eff(a, p) = sum;
            const double x = (double) p / (P - 1);
            diff(a, p) = sum - target + target * x * x * x;
          }
          double mx = -1e300;
          for (int p = 0; p < P; ++p) {
            const double g = p == 0 ? diff(a, 1) - diff(a, 0)
                             : p == P - 1 ? diff(a, P - 1) - diff(a, P - 2)
                                          : 0.5 * (diff(a, p + 1) - diff(a, p - 1));
            const double ag = Kokkos::fabs(g);
            grad(a, p) = ag > 1e-12 ? ag : 1e-12;
            sgn(a, p) = ag >= 1e-12 ? (g > 0.0 ? 1.0 : (g < 0.0 ? -1.0 : 0.0)) : 0.0;
            const double r = diff(a, p) / grad(a, p);
            w(a, p) = -0.5 * r * r;
            if (w(a, p) > mx) mx = w(a, p);
          }
          double sum = 0.0, cs = 0.0;
          for (int p = 0; p < P; ++p) sum += (w(a, p) = Kokkos::exp(w(a, p) - mx));
          for (int p = 0; p < P; ++p) cs += probes(p) * (w(a, p) /= sum);
          c(a) = cs;
        });
    Kokkos::parallel_for(
        "hv_grid_t", RangePolicy(0, K * N), KOKKOS_LAMBDA(int q) {
          const int t = q / N, a = q % N;
          const double p0 = probes(0), dp = probes(1) - probes(0);
          for (int p = 0; p < P; ++p) dt(q, p) = 0.0;
          for (int e = roff(a); e < roff(a + 1); ++e) {
            const double d = rdist(e), dd = ddot(t * R + e);
            const int lo = Kokkos::max((int) Kokkos::floor((d - p0) / dp), 0);
            const int hi = Kokkos::min((int) Kokkos::ceil((d + wad - p0) / dp) + 1, P);
            for (int p = lo; p < hi; ++p) {
              double f1, f2;
              cut_derivs(d, probes(p), wad, 0, f1, f2);
              dt(q, p) += f1 * dd;
            }
          }
          double ubar = 0.0;
          for (int p = 0; p < P; ++p) {
            const double gtp = sgn(a, p) * (p == 0 ? dt(q, 1) - dt(q, 0)
                                            : p == P - 1 ? dt(q, P - 1) - dt(q, P - 2)
                                                         : 0.5 * (dt(q, p + 1) - dt(q, p - 1)));
            const double gd = grad(a, p), r = diff(a, p) / gd;
            const double rt = dt(q, p) / gd - diff(a, p) * gtp / (gd * gd);
            wt(q, p) = -r * rt;  // u' for now
            ubar += w(a, p) * wt(q, p);
          }
          double ct = 0.0;
          for (int p = 0; p < P; ++p) ct += probes(p) * (wt(q, p) = w(a, p) * (wt(q, p) - ubar));
          rcdot(q) = ct;
        });
  }
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
        cut_derivs(pdist(k), pcut(k), width, kind, f1, f2);
        cut_t(q) = (Net) (f1 * (dd - pc_t));
        cf_t(t * N + n, 1 + k - off(n)) = cut_t(q);
      });

  // ---- the network: the forward with every save and its tangents, then the
  // backward with the adjoints' tangents, leaving x4_adj, cutoff_adj and cf_adj
  View2D x4_adj = ws_.n2("x4_adj", E, 4), x4_adj_t = ws_.n2("hv:x4_adj_t", K * E, 4);
  View2D cf_adj = ws_.n2("cf_seq_adj", N, S), cf_adj_t = ws_.n2("hv:cf_adj_t", K * N, S);
  for (auto v : {x4_adj, x4_adj_t, cf_adj, cf_adj_t}) Kokkos::deep_copy(ExecSpace(), v, Net(0));
  View1D cutoff_adj = ws_.n1("cutoff_adj", E), cutoff_adj_t = ws_.n1("hv:cutoff_adj_t", K * E);
  const View2D cond = conditioning(dev);
  View1D net = ws_.n1("per_atom_net", N);
  ReadoutSaves rs;
  ReadoutTangents rt;
  View2D ie_t = ws_.n2("hv:ie_t", K * E, D);
  Kokkos::deep_copy(ExecSpace(), ie_t, Net(0));
  if (h_.featurizer_type == FeaturizerType::FeedForward) {
    View2D node = ws_.n2("node", N, Dn), input_edge = ws_.n2("input_edge", E, D);
    gather(node, mat("node_embedders.0.weight").v, dev.species);
    gather(input_edge, mat("edge_embedder.weight").v, pk.species);
    View2D node_t = ws_.n2("hv:node_t", K * N, Dn);
    Kokkos::deep_copy(ExecSpace(), node_t, Net(0));
    std::vector<LayerSaves> sav(G);
    std::vector<LayerTangents> tan(G);
    for (int L = 0; L < G; ++L) {
      sav[L].tag = std::to_string(L), sav[L].keep_concat = true;
      ff_layer(dev, pk, L, node, input_edge, cond, &sav[L], true);
      ff_layer_tan(pk, L, sav[L], tan[L], K, node_t, ie_t, x4_t, cf_seq, cf_t);
    }
    readout({node}, {input_edge}, View2D(net.data(), N, 1), pk, &rs, "", "energy");
    readout_tan(rs, rt, 0, K, node_t, ie_t);
    View2D node_adj = ws_.n2("node_adj", N, Dn), node_adj_t = ws_.n2("hv:node_adj_t", K * N, Dn);
    View2D ie_adj = ws_.n2("ie_adj", E, D), ie_adj_t = ws_.n2("hv:ie_adj_t", K * E, D);
    readout_bwd2(rs, rt, 0, pk, K, cut_t, node_adj, node_adj_t, ie_adj, ie_adj_t, cutoff_adj, cutoff_adj_t, false);
    for (int L = G - 1; L >= 0; --L)
      ff_layer_bwd2(pk, L, sav[L], tan[L], K, cf_seq, cf_t, node_adj, node_adj_t, ie_adj, ie_adj_t, x4_adj,
                    x4_adj_t, cf_adj, cf_adj_t);
  } else {
    const int T = N + E, RL = h_.num_readout_layers;
    View2D input_edge = ws_.n2("re_input_edge", E, D);
    gather(input_edge, mat("edge_embedder.weight").v, pk.species);
    std::vector<ResidualSaves> sav(G);
    std::vector<ResidualTangents> tan(G);
    std::vector<View2D> tokens(G), tokens_t(G), node_feat(G), edge_feat(G);
    for (int L = 0; L < G; ++L) {
      const std::string ls = std::to_string(L);
      sav[L].tag = ls;
      tokens[L] = ws_.n2("re_tokens_" + ls, T, D);
      tokens_t[L] = ws_.n2("hv:tokens_t_" + ls, K * T, D);
      node_feat[L] = View2D(tokens[L].data(), N, D);
      edge_feat[L] = View2D(tokens[L].data() + std::size_t(N) * D, E, D);
      residual_layer(dev, pk, L, input_edge, tokens[L], &sav[L], true);
      residual_layer_tan(pk, L, sav[L], tan[L], K, ie_t, tokens_t[L], x4_t, cf_seq, cf_t);
      View2D node_out = node_feat[L];
      if (cond.extent(0) == (std::size_t) N)
        Kokkos::parallel_for(
            "hv_r_cond", RangePolicy(0, N * D), KOKKOS_LAMBDA(int i) { node_out(i / D, i % D) += cond(i / D, i % D); });
      if (L + 1 < G) {  // input_edge = (input_edge + out_edge of the reverse) / 2, and its tangent
        View2D tk = tokens[L], tkt = tokens_t[L];
        Kokkos::parallel_for(
            "hv_r_msg", RangePolicy(0, (K + 1) * E * D), KOKKOS_LAMBDA(int i) {
              const int q = i / D, d = i % D, t = q / E - 1, k = q % E, r = rev(k);
              if (t < 0) input_edge(k, d) = Net(0.5) * (input_edge(k, d) + (r >= 0 ? tk(N + r, d) : Net(0)));
              else
                ie_t(t * E + k, d) =
                    Net(0.5) * (ie_t(t * E + k, d) + (r >= 0 ? tkt(t * T + N + r, d) : Net(0)));
            });
      }
    }
    readout(node_feat, edge_feat, View2D(net.data(), N, 1), pk, &rs, "re_", "energy");
    // per readout layer: its tokens' tangents as [K N] node and [K E] edge rows,
    // and back the same way for the adjoints
    std::vector<View2D> tok_adj(G), tok_adj_t(G);
    for (int L = 0; L < G; ++L) {
      tok_adj[L] = ws_.n2("hv:tok_adj_" + std::to_string(L), T, D);
      tok_adj_t[L] = ws_.n2("hv:tok_adj_t_" + std::to_string(L), K * T, D);
      Kokkos::deep_copy(ExecSpace(), tok_adj[L], Net(0));
      Kokkos::deep_copy(ExecSpace(), tok_adj_t[L], Net(0));
    }
    for (int i = 0; i < RL; ++i) {
      Workspace::Scope s(ws_);
      View2D nt = ws_.tmp(K * N, D), et = ws_.tmp(K * E, D), tt = tokens_t[i];
      Kokkos::parallel_for(
          "hv_r_ro_split", RangePolicy(0, K * T * D), KOKKOS_LAMBDA(int j) {
            const int q = j / D, d = j % D, t = q / T, r = q % T;
            if (r < N) nt(t * N + r, d) = tt(q, d);
            else et(t * E + r - N, d) = tt(q, d);
          });
      readout_tan(rs, rt, i, K, nt, et);
      View2D na = ws_.tmp(N, D), na_t = ws_.tmp(K * N, D), ea = ws_.tmp(E, D), ea_t = ws_.tmp(K * E, D);
      readout_bwd2(rs, rt, i, pk, K, cut_t, na, na_t, ea, ea_t, cutoff_adj, cutoff_adj_t, i > 0);
      View2D ta = tok_adj[i], tat = tok_adj_t[i];
      Kokkos::parallel_for(
          "hv_r_ro_join", RangePolicy(0, (K + 1) * T * D), KOKKOS_LAMBDA(int j) {
            const int q = j / D, d = j % D, t = q / T - 1, r = q % T;
            if (t < 0) ta(r, d) = r < N ? na(r, d) : ea(r - N, d);
            else tat(t * T + r, d) = r < N ? na_t(t * N + r, d) : ea_t(t * E + r - N, d);
          });
    }
    // ie_adj: the adjoint of the next layer's input_edge, zero past the last
    View2D ie_adj = ws_.n2("hv:r_ie_adj", E, D), ie_adj_t = ws_.n2("hv:r_ie_adj_t", K * E, D);
    for (int L = G - 1; L >= 0; --L) {
      View2D ta = tok_adj[L], tat = tok_adj_t[L];
      View2D ie_next = ws_.n2("hv:r_ie_" + std::to_string(L % 2), E, D);
      View2D ie_next_t = ws_.n2("hv:r_ie_t_" + std::to_string(L % 2), K * E, D);
      const bool msg = L + 1 < G;
      Kokkos::parallel_for(
          "hv_r_msg_bwd", RangePolicy(0, (K + 1) * E * D), KOKKOS_LAMBDA(int i) {
            const int q = i / D, d = i % D, t = q / E - 1, k = q % E, r = rev(k);
            if (t < 0) {
              if (msg && r >= 0) ta(N + k, d) += Net(0.5) * ie_adj(r, d);
              ie_next(k, d) = msg ? Net(0.5) * ie_adj(k, d) : Net(0);
            } else {
              if (msg && r >= 0) tat(t * T + N + k, d) += Net(0.5) * ie_adj_t(t * E + r, d);
              ie_next_t(t * E + k, d) = msg ? Net(0.5) * ie_adj_t(t * E + k, d) : Net(0);
            }
          });
      residual_layer_bwd2(pk, L, sav[L], tan[L], K, cf_seq, cf_t, ta, tat, ie_next, ie_next_t, x4_adj, x4_adj_t,
                          cf_adj, cf_adj_t);
      ie_adj = ie_next, ie_adj_t = ie_next_t;
    }
  }

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
        cut_derivs(d, pcut(k), width, kind, f1, f2);
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
    if (solver) {
      Kokkos::parallel_for(
          "hv_nr_t", TeamPolicy(K * N, 1, kLanes), KOKKOS_LAMBDA(const Team& m) {
            const int q = m.league_rank(), t = q / N, a = q % N;
            const double rt = rcdot(q);
            double s = 0.0;
            if (adn(a) > 0.0)
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(m, roff(a), roff(a + 1)),
                  [&](int e, double& acc) { acc += sf(e, 1) * (rt - ddot(t * R + e)); }, s);
            Kokkos::single(Kokkos::PerThread(m), [&] {
              nr_t(q) = adn(a) > 0.0 ? s + 6.0 * target * ar(a) / (rmax * rmax * rmax) * rt : 0.0;
            });
          });
      Kokkos::parallel_for(
          "hv_adaptive_g", RangePolicy(0, (K + 1) * R), KOKKOS_LAMBDA(int q) {
            const int t = q / R - 1, e = q % R, c = rcen(e);
            const double nr = adn(c), d = rdist(e);
            if (nr <= 0.0 || d <= 0.0) return;
            const double f1 = sf(e, 0), f2 = sf(e, 1);
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
    } else {
      // The grid's backward, as forces.cpp's: per probe A_p = A w_p (probe_p - c),
      // diff_adj = A_p (-diff/grad^2) + gradient^T(Cq), Cq = A_p gsgn diff^2/grad^3,
      // then per raw edge gmag = scale sum_p diff_adj f1(d, probe_p) / d; and each one's
      // tangent.
      auto diff = gdiff, grad = ggrad, sgn = gsgn, w = gw, dt = gdt, wt = gwt;
      auto c = gc;
      RView2D da = ws_.r2("hv:gda", (K + 1) * N, P), cq = ws_.r2("hv:gcq", (K + 1) * N, P);
      Kokkos::parallel_for(
          "hv_grid_adj", RangePolicy(0, (K + 1) * N), KOKKOS_LAMBDA(int q) {
            const int t = q / N - 1, a = q % N, tq = t * N + a;
            for (int p = 0; p < P; ++p) {
              const double gd = grad(a, p), df = diff(a, p), Ap = A(a) * w(a, p) * (probes(p) - c(a));
              if (t < 0) {
                da(q, p) = Ap * (-df / (gd * gd));
                cq(q, p) = Ap * sgn(a, p) * df * df / (gd * gd * gd);
                continue;
              }
              const double gtp = sgn(a, p) * (p == 0 ? dt(tq, 1) - dt(tq, 0)
                                              : p == P - 1 ? dt(tq, P - 1) - dt(tq, P - 2)
                                                           : 0.5 * (dt(tq, p + 1) - dt(tq, p - 1)));
              const double Apt = A_t(tq) * w(a, p) * (probes(p) - c(a)) + A(a) * wt(tq, p) * (probes(p) - c(a)) -
                                 A(a) * w(a, p) * rcdot(tq);
              da(q, p) = Apt * (-df / (gd * gd)) + Ap * (-dt(tq, p) / (gd * gd) + 2.0 * df * gtp / (gd * gd * gd));
              cq(q, p) = sgn(a, p) * (Apt * df * df / (gd * gd * gd) +
                                      Ap * (2.0 * df * dt(tq, p) / (gd * gd * gd) - 3.0 * df * df * gtp / (gd * gd * gd * gd)));
            }
            for (int r = 0; r < P; ++r) {  // the adjoint of the gradient stencil
              double B = 0.0;
              if (r - 1 >= 1 && r - 1 <= P - 2) B += 0.5 * cq(q, r - 1);
              if (r + 1 >= 1 && r + 1 <= P - 2) B += -0.5 * cq(q, r + 1);
              if (r == 1) B += cq(q, 0);
              if (r == 0) B += -cq(q, 0);
              if (r == P - 1) B += cq(q, P - 1);
              if (r == P - 2) B += -cq(q, P - 1);
              da(q, r) += B;
            }
          });
      Kokkos::parallel_for(
          "hv_grid_g", RangePolicy(0, (K + 1) * R), KOKKOS_LAMBDA(int q) {
            const int t = q / R - 1, e = q % R, a = rcen(e);
            const double d = rdist(e);
            if (d <= 0.0) return;
            const double p0 = probes(0), dp = probes(1) - probes(0);
            const int lo = Kokkos::max((int) Kokkos::floor((d - p0) / dp), 0);
            const int hi = Kokkos::min((int) Kokkos::ceil((d + wad - p0) / dp) + 1, P);
            double gm = 0.0, gmt = 0.0;
            const double dd = t < 0 ? 0.0 : ddot(t * R + e);
            for (int p = lo; p < hi; ++p) {
              double f1, f2;
              cut_derivs(d, probes(p), wad, 0, f1, f2);
              gm += da(a, p) * f1;
              if (t >= 0) gmt += da((t + 1) * N + a, p) * f1 + da(a, p) * f2 * dd;
            }
            gm *= scale / d;
            if (t < 0) {
              for (int x = 0; x < 3; ++x) g(e, x) += gm * rvec(e, x);
              return;
            }
            gmt = scale * gmt / d - gm * dd / d;
            for (int x = 0; x < 3; ++x) g_t(t * R + e, x) += gmt * rvec(e, x) + gm * vdot_raw(t * R + e, x);
          });
    }
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
      "hv_virial", TeamPolicy(K * 9, Kokkos::AUTO), KOKKOS_LAMBDA(const Team& m) {
        const int q = m.league_rank(), t = q / 9, x = (q % 9) / 3, y = q % 3;
        double s = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(m, R),
            [&](int e, double& acc) { acc += vdot_raw(t * R + e, x) * g(e, y) + rvec(e, x) * g_t(t * R + e, y); }, s);
        Kokkos::single(Kokkos::PerTeam(m), [&] { dW(t, x * 3 + y) = s; });
      });
  return out;
}

}  // namespace pet
