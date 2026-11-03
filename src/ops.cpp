// Dense layers, norms, the feedforward blocks and compress.0: see ops.hpp.
#include "ops.hpp"

#include "pet/gemm.hpp"
#include "pet/ozaki.hpp"

#include <cstdint>

namespace pet {

// Every GEMM in the network has a weight as its second operand and comes through
// here: on narrow operands in the FP8, BF16 and FP16 modes where it can, else
// through gemm_ozaki, which is the vendor GEMM unless the Ozaki path is on. The
// backward takes the same path as the forward. In the native and TF32 modes,
// CUTLASS's GEMM where the build has it (fused_gemm.cpp); in the native mode its
// answer for a row does not depend on how many rows there are, so a structure
// gets the same bits whatever its batch.
static void weight_gemm(char tw, Net beta, View2D out, View2D in, const WeightRef& W, View1D b = {}) {
  if (gemm_lowp('N', tw, Net(1), in, W.v, beta, out, b)) return;
  if (!W.split && linear_fused(out, in, W.v, tw == 'T', b, beta)) return;
  gemm_ozaki('N', tw, Net(1), in, W.v, beta, out, W.for_orientation(tw == 'T'), b);
}

void linear(View2D out, View2D in, const WeightRef& W, View1D b, Net beta) { weight_gemm('T', beta, out, in, W, b); }

// The elementwise passes take four values a thread where their buffers allow
// (contiguous, aligned to four values, a multiple of four long): one 16-byte
// load or store, not four, and no row and column to work out. The arithmetic
// per value is the same either way. On an H200 that took pet-mad-s on 5184
// water 33.3 -> 31.2 ms; on an MI250X GCD the one-value passes were 2% faster,
// so AMD keeps them.
#if defined(KOKKOS_ENABLE_HIP)
constexpr bool kQuadPasses = false;
#else
constexpr bool kQuadPasses = true;
#endif
struct alignas(4 * sizeof(Net)) Quad {
  Net v[4];
};
static bool quads(const View2D& v) {
  return kQuadPasses && v.span_is_contiguous() && v.size() % 4 == 0 &&
         reinterpret_cast<std::uintptr_t>(v.data()) % sizeof(Quad) == 0;
}
static bool quad_rows(const View2D& v) {
  return kQuadPasses && v.extent(1) % 4 == 0 && v.stride(0) % 4 == 0 &&
         reinterpret_cast<std::uintptr_t>(v.data()) % sizeof(Quad) == 0;
}
KOKKOS_INLINE_FUNCTION Quad& quad(Net* p, std::size_t q) { return reinterpret_cast<Quad*>(p)[q]; }

void linear_bwd(View2D in_adj, View2D out_adj, const WeightRef& W, Net beta) {
  weight_gemm('N', beta, in_adj, out_adj, W);
}

void linear_silu(View2D out, View2D pre, View2D in, const WeightRef& W, View1D b) {
  // The fused kernels run in the native and TF32 modes only.
  if (!W.split && silu_in_fused(out, pre, in, W.v, b)) return;
  // The GEMM writes the pre-activation where it is kept, and the pass reads it
  // from there: no copy of it to write.
  const bool save = pre.data() != nullptr;
  const View2D z = save ? pre : out;
  linear(z, in, W, b);
  const int n = out.extent(0) * out.extent(1);
  if (quads(out) && quads(z)) {
    Net *o = out.data(), *p = z.data();
    Kokkos::parallel_for(
        "silu", RangePolicy(0, n / 4), KOKKOS_LAMBDA(int q) {
          Quad v = quad(p, q);
          for (int j = 0; j < 4; ++j) v.v[j] = silu(v.v[j]);
          quad(o, q) = v;
        });
    return;
  }
  View1D o(out.data(), n), p(z.data(), n);
  Kokkos::parallel_for("silu", RangePolicy(0, n), KOKKOS_LAMBDA(int i) { o(i) = silu(p(i)); });
}

void silu_bwd(View2D grad, View2D pre) {
  const int n = grad.extent(0) * grad.extent(1);
  if (quads(grad) && quads(pre)) {
    Net *g = grad.data(), *p = pre.data();
    Kokkos::parallel_for(
        "silu_bwd", RangePolicy(0, n / 4), KOKKOS_LAMBDA(int q) {
          const Quad x = quad(p, q);
          Quad y = quad(g, q);
          for (int j = 0; j < 4; ++j) {
            const Net sg = sigmoid(x.v[j]);
            y.v[j] *= sg * (Net(1) + x.v[j] * (Net(1) - sg));
          }
          quad(g, q) = y;
        });
    return;
  }
  View1D g(grad.data(), n), p(pre.data(), n);
  Kokkos::parallel_for(
      "silu_bwd", RangePolicy(0, n), KOKKOS_LAMBDA(int i) {
        const Net x = p(i), sg = sigmoid(x);
        g(i) *= sg * (Net(1) + x * (Net(1) - sg));
      });
}

void linear_silu_bwd(View2D grad, View2D out_adj, const WeightRef& W, View2D pre) {
  if (!W.split && silu_bwd_fused(grad, out_adj, W.v, pre)) return;
  linear_bwd(grad, out_adj, W, 0);
  silu_bwd(grad, pre);
}

void gather(View2D out, View2D table, IView1D idx) {
  const int D = out.extent(1);
  Kokkos::parallel_for(
      "gather", RangePolicy(0, out.extent(0) * D),
      KOKKOS_LAMBDA(int i) { out(i / D, i % D) = table(idx(i / D), i % D); });
}

void copy(View2D a, View2D b) {
  const int n = a.extent(0) * a.extent(1);
  View1D a1(a.data(), n), b1(b.data(), n);
  Kokkos::parallel_for("copy", RangePolicy(0, n), KOKKOS_LAMBDA(int i) { a1(i) = b1(i); });
}

// ---- norms --------------------------------------------------------------------
//
// The kernels are norm_rows and norm_rows_bwd (ops.hpp); these are the plain ones.

void norm_fwd(View2D out, View2D in, View1D weight, View1D bias, View2D add) {
  const bool sum = add.data() != nullptr;
  norm_rows("norm", out, in, weight, bias, KOKKOS_LAMBDA(int r, int d) {
    if (sum) in(r, d) += add(r, d);
  });
}

void norm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight, bool ln, bool acc, View2D also) {
  const bool twice = also.data() != nullptr;
  norm_rows_bwd("norm_bwd", out_adj, in, weight, ln, KOKKOS_LAMBDA(int r, int d, Net g) {
    const Net v = (acc ? in_adj(r, d) : Net(0)) + g;
    in_adj(r, d) = v;
    if (twice) also(r, d) = v;
  });
}

// The same arithmetic as norm_rows_bwd -- the same reductions in the same lane
// order, the same expression per value -- so the adjoints come out bit for bit
// as the norm pass and the gather after it gave them.
#if defined(PET_HIP_NORMS)
namespace {
// concat_norm_bwd a wavefront an edge, as hip_rows::Bwd: lane l holds columns
// l + 64 j of the concat row, the first V/2 of them oe's own, the rest its
// reverse's (D a multiple of 64).
struct ConcatBwd {
  View2D edge_adj, out_adj, oe, ie_adj;
  View1D weight;
  IView1D rev;
  bool ln;
  template <int V>
  __device__ void row(int k, int l) const {
    constexpr int H = V / 2;
    const int D = 64 * H, r = rev(k);
    const bool pair = r >= 0 && r != k && rev(r) == k;
    if (pair && r < k) return;  // the partner's wavefront has both
    auto load = [&](int q, Net* x, Net* ga) {
      const int p = rev(q);
#pragma unroll
      for (int j = 0; j < V; ++j) {
        const int d = l + 64 * j;
        x[j] = j < H ? oe(q, d) : p >= 0 ? oe(p, d - D) : Net(0);
        ga[j] = out_adj(q, d);
      }
    };
    Net xk[V], gk[V], xr[V], gr[V];
    load(k, xk, gk);
    const hip_rows::Stats sk = hip_rows::bwd_stats<V>(xk, gk, weight, l, 2 * D, ln);
    hip_rows::Stats sr = sk;
    if (r >= 0) {
      load(r, xr, gr);
      sr = hip_rows::bwd_stats<V>(xr, gr, weight, l, 2 * D, ln);
    }
#pragma unroll
    for (int j = 0; j < H; ++j) {
      const int d = l + 64 * j;
      const Net back = r >= 0 ? hip_rows::grad(sr, xr[H + j], gr[H + j], weight(D + d)) : Net(0);
      edge_adj(k, d) = ie_adj(k, d) + hip_rows::grad(sk, xk[j], gk[j], weight(d)) + back;
      if (pair) edge_adj(r, d) = ie_adj(r, d) + hip_rows::grad(sr, xr[j], gr[j], weight(d)) +
                                 hip_rows::grad(sk, xk[H + j], gk[H + j], weight(D + d));
    }
  }
};
}  // namespace
#endif

void concat_norm_bwd(View2D edge_adj, View2D out_adj, View2D oe, View1D weight, bool ln, View2D ie_adj,
                     IView1D rev) {
  const int E = oe.extent(0), D = oe.extent(1), W = 2 * D;
#if defined(PET_HIP_NORMS)
  if (D % 64 == 0 && hip_rows::launch("norm_cat_bwd", E, W, ConcatBwd{edge_adj, out_adj, oe, ie_adj, weight, rev, ln}))
    return;
#endif
  struct Stats {
    Net mu, inv, mg, mgx;
  };
  Kokkos::parallel_for(
      "norm_cat_bwd", norm_policy(E), KOKKOS_LAMBDA(const Team& t) {
        const int k = t.league_rank() * kNormRows + t.team_rank();
        if (k >= E) return;
        const int r = rev(k);
        const bool pair = r >= 0 && r != k && rev(r) == k;
        if (pair && r < k) return;  // the partner's team has both
        // Row q of the concat, read from oe.
        auto x = [&](int q, int d) {
          if (d < D) return oe(q, d);
          const int p = rev(q);
          return p >= 0 ? oe(p, d - D) : Net(0);
        };
        auto stats = [&](int q) {
          Net mu = Net(0);
          if (ln) {
            Net s = Net(0);
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, W), [&](int d, Net& a) { a += x(q, d); }, s);
            mu = s / W;
          }
          Net sq = Net(0), sg = Net(0), sgx = Net(0);
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(t, W),
              [&](int d, Net& a, Net& b, Net& c) {
                const Net xv = x(q, d) - mu, gw = out_adj(q, d) * weight(d);
                a += xv * xv;
                b += gw;
                c += gw * xv;
              },
              sq, sg, sgx);
          const Net inv = norm_inv(sq, W, ln);
          return Stats{mu, inv, ln ? sg / W : Net(0), sgx * inv / W};
        };
        auto g = [&](const Stats& s, int q, int d) {
          const Net a = Kokkos::fma(out_adj(q, d), weight(d), -s.mg), b = mul_rn(x(q, d) - s.mu, s.inv);
          return mul_rn(s.inv, Kokkos::fma(-b, s.mgx, a));
        };
        const Stats sk = stats(k), sr = r >= 0 ? stats(r) : sk;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          edge_adj(k, d) = ie_adj(k, d) + g(sk, k, d) + (r >= 0 ? g(sr, r, D + d) : Net(0));
          if (pair) edge_adj(r, d) = ie_adj(r, d) + g(sr, r, d) + g(sk, k, D + d);
        });
      });
}

// ---- SwiGLU -------------------------------------------------------------------

void swiglu_in(View2D pre, View2D h, View2D in, const WeightRef& w_in, View1D b_in) {
  // The fused kernels run in the native and TF32 modes only.
  if (!w_in.split && swiglu_in_fused(pre, h, in, w_in.v, b_in)) return;
  linear(pre, in, w_in, b_in);
  if (!h.data()) return;
  const int F = h.extent(1);
  if (quads(h) && quads(pre) && pre.size() == 2 * h.size()) {
    // Flat: h's value j is pre's 2j and 2j + 1 (pre rows are twice as wide).
    Net *hp = h.data(), *pp = pre.data();
    Kokkos::parallel_for(
        "swiglu", RangePolicy(0, int(h.size() / 4)), KOKKOS_LAMBDA(int q) {
          const Quad a = quad(pp, 2 * q), b = quad(pp, 2 * q + 1);
          quad(hp, q) = {{a.v[0] * sigmoid(a.v[1]), a.v[2] * sigmoid(a.v[3]), b.v[0] * sigmoid(b.v[1]),
                          b.v[2] * sigmoid(b.v[3])}};
        });
    return;
  }
  Kokkos::parallel_for(
      "swiglu", RangePolicy(0, h.extent(0) * F), KOKKOS_LAMBDA(int i) {
        const int r = i / F, c = i % F;
        h(r, c) = pre(r, 2 * c) * sigmoid(pre(r, 2 * c + 1));
      });
}

void swiglu(Workspace& ws, View2D out, View2D in, const WeightRef& w_in, View1D b_in,
            const WeightRef& w_out, View1D b_out, View2D pre, Net beta) {
  Workspace::Scope scope(ws);
  View2D h = ws.tmp(in.extent(0), pre.extent(1) / 2);
  swiglu_in(pre, h, in, w_in, b_in);
  linear(out, h, w_out, b_out, beta);
}

void swiglu_bwd(Workspace& ws, View2D in_adj, View2D out_adj, View2D pre, const WeightRef& w_in,
                const WeightRef& w_out, Net beta) {
  if (swiglu_bwd_deferred(ws, in_adj, out_adj, pre, w_in, w_out, beta)) return;
  const int R = out_adj.extent(0), F = pre.extent(1) / 2;
  Workspace::Scope scope(ws);
  View2D pre_adj = ws.tmp(R, 2 * F);
  if (w_out.split || !swiglu_bwd_fused(pre_adj, out_adj, w_out.v, pre)) {
    View2D h_adj = ws.tmp(R, F);
    linear_bwd(h_adj, out_adj, w_out, 0);
    if (quads(h_adj) && quads(pre) && quads(pre_adj)) {
      Net *ha = h_adj.data(), *pp = pre.data(), *pa = pre_adj.data();
      Kokkos::parallel_for(
          "swiglu_bwd", RangePolicy(0, R * F / 4), KOKKOS_LAMBDA(int q) {
            const Quad g = quad(ha, q);
            for (int half = 0; half < 2; ++half) {
              const Quad x = quad(pp, 2 * q + half);
              Quad y;
              for (int j = 0; j < 2; ++j) {
                const Net v = x.v[2 * j], sg = sigmoid(x.v[2 * j + 1]), gj = g.v[2 * half + j];
                y.v[2 * j] = gj * sg;
                y.v[2 * j + 1] = gj * v * sg * (Net(1) - sg);
              }
              quad(pa, 2 * q + half) = y;
            }
          });
    } else {
      Kokkos::parallel_for(
          "swiglu_bwd", RangePolicy(0, R * F), KOKKOS_LAMBDA(int i) {
            const int r = i / F, c = i % F;
            const Net v = pre(r, 2 * c), sg = sigmoid(pre(r, 2 * c + 1));
            pre_adj(r, 2 * c) = h_adj(r, c) * sg;
            pre_adj(r, 2 * c + 1) = h_adj(r, c) * v * sg * (Net(1) - sg);
          });
    }
  }
  linear_bwd(in_adj, pre_adj, w_in, beta);
}

// ---- compress.0 ---------------------------------------------------------------
//
// pre = [input_edge wi^T] + wx (v, |v|) + b + tab(species of the neighbour): the
// geometry and species blocks of compress.0 were folded into wx, b and tab at
// load time (CompressFold), so only input_edge past layer 0 needs a GEMM.

void compress_fwd(View2D out, View2D sav, const CompressFold& f, View2D input_edge,
                  const PackedEdges& pk) {
  const bool gemm = f.wi.extent(0) > 0, save = sav.data() != nullptr;
  if (gemm) weight_gemm('T', Net(0), out, input_edge, f.wi);
  const int D = out.extent(1);
  auto wx = f.wx;
  auto b = f.b;
  auto tab = f.tab;
  auto nsp = pk.species;
  auto ev = pk.vec;
  auto dist = pk.dist;
  if (quads(out) && (!save || quads(sav)) && quad_rows(tab) && D % 4 == 0 &&
      reinterpret_cast<std::uintptr_t>(b.data()) % sizeof(Quad) == 0) {
    Net *o = out.data(), *sv = sav.data();
    Kokkos::parallel_for(
        "compress", RangePolicy(0, int(out.size() / 4)), KOKKOS_LAMBDA(int q) {
          const int k = 4 * q / D, d0 = 4 * q % D;
          const Quad pre = gemm ? quad(o, q) : Quad{};
          const Net e0 = ev(k, 0), e1 = ev(k, 1), e2 = ev(k, 2), e3 = dist(k);
          Quad z, y;
          for (int j = 0; j < 4; ++j) {
            const int d = d0 + j;
            z.v[j] = (gemm ? pre.v[j] : Net(0)) + b(d) + tab(nsp(k), d) + wx(d, 0) * e0 + wx(d, 1) * e1 +
                     wx(d, 2) * e2 + wx(d, 3) * e3;
            y.v[j] = silu(z.v[j]);
          }
          if (save) quad(sv, q) = z;
          quad(o, q) = y;
        });
    return;
  }
  Kokkos::parallel_for(
      "compress", RangePolicy(0, out.extent(0) * D), KOKKOS_LAMBDA(int i) {
        const int k = i / D, d = i % D;
        const Net z = (gemm ? out(k, d) : Net(0)) + b(d) + tab(nsp(k), d) + wx(d, 0) * (Net) ev(k, 0) +
                      wx(d, 1) * (Net) ev(k, 1) + wx(d, 2) * (Net) ev(k, 2) + wx(d, 3) * (Net) dist(k);
        if (save) sav(k, d) = z;
        out(k, d) = silu(z);
      });
}

// One warp per edge, several edges a block (as the norms: one 32-thread block an
// edge leaves a large GPU at its limit of resident blocks): silu' in place,
// then the four geometry sums as one reduction. Nothing is computed for the
// species blocks, which are constants. Where the GEMM can apply silu' in its
// epilogue (CUDA's fused kernels) it does, and this pass only reads g for the
// sums: one write and one read of g fewer, the same arithmetic.
void compress_bwd(View2D g, View2D et_adj, const WeightRef& w2, View2D pre, const CompressFold& f, View2D x4_adj,
                  View2D ie_adj) {
  const int D = g.extent(1), E = g.extent(0);
  const bool fused = !w2.split && silu_bwd_fused(g, et_adj, w2.v, pre);
  if (!fused) linear_bwd(g, et_adj, w2, 0);
  auto wx = f.wx;
  Kokkos::parallel_for(
      "compress_bwd", norm_policy(E), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank() * kNormRows + t.team_rank();
        if (r >= E) return;
        Net a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& s0, Net& s1, Net& s2, Net& s3) {
              Net gd = g(r, d);
              if (!fused) {
                const Net x = pre(r, d), sg = sigmoid(x);
                g(r, d) = gd = gd * sg * (Net(1) + x * (Net(1) - sg));
              }
              s0 += gd * wx(d, 0), s1 += gd * wx(d, 1), s2 += gd * wx(d, 2), s3 += gd * wx(d, 3);
            },
            a0, a1, a2, a3);
        Kokkos::single(Kokkos::PerThread(t), [&] {
          x4_adj(r, 0) += a0, x4_adj(r, 1) += a1, x4_adj(r, 2) += a2, x4_adj(r, 3) += a3;
        });
      });
  if (f.wi.extent(0) > 0) linear_bwd(ie_adj, g, f.wi);
}

// ---- energy -------------------------------------------------------------------

// Accumulated in double: the sum runs over an atom's edges.
void readout_accumulate(View2D out, View2D node_pred, View2D edge_pred, const PackedEdges& pk,
                        bool acc) {
  auto off = pk.off;
  auto cut = pk.cut;
  const int C = out.extent(1);
  Kokkos::parallel_for(
      "readout", RangePolicy(0, out.extent(0) * C), KOKKOS_LAMBDA(int i) {
        const int n = i / C, c = i % C;
        double s = node_pred(n, c);
        for (int k = off(n); k < off(n + 1); ++k) s += (double) cut(k) * (double) edge_pred(k, c);
        out(n, c) = acc ? Net(out(n, c) + s) : Net(s);
      });
}

void assemble_energy(RView1D per_atom, View1D net, IView1D species, RView1D composition,
                     double scale, int n_local) {
  Kokkos::parallel_for(
      "assemble", RangePolicy(0, per_atom.extent(0)),
      KOKKOS_LAMBDA(int n) { per_atom(n) = n < n_local ? scale * net(n) + composition(species(n)) : 0.0; });
}

}  // namespace pet
