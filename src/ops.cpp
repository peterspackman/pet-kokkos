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
// load or store, not four, and no row and column to work out. An H200 ran the
// one-value passes at about half its bandwidth. The arithmetic per value is
// the same either way.
static bool quads(const View2D& v) {
  return v.span_is_contiguous() && v.size() % 4 == 0 &&
         reinterpret_cast<std::uintptr_t>(v.data()) % sizeof(Quad) == 0;
}
KOKKOS_INLINE_FUNCTION Quad& quad(Net* p, std::size_t q) { return reinterpret_cast<Quad*>(p)[q]; }

void linear_bwd(View2D in_adj, View2D out_adj, const WeightRef& W, Net beta) {
  weight_gemm('N', beta, in_adj, out_adj, W);
}

void linear_silu(View2D out, View2D pre, View2D in, const WeightRef& W, View1D b) {
  // The fused kernels run in the native and TF32 modes only.
  if (!W.split && silu_in_fused(out, pre, in, W.v, b)) return;
  linear(out, in, W, b);
  const int n = out.extent(0) * out.extent(1);
  const bool save = pre.data() != nullptr;
  if (quads(out) && (!save || quads(pre))) {
    Net *o = out.data(), *p = pre.data();
    Kokkos::parallel_for(
        "silu", RangePolicy(0, n / 4), KOKKOS_LAMBDA(int q) {
          Quad z = quad(o, q);
          if (save) quad(p, q) = z;
          for (int j = 0; j < 4; ++j) z.v[j] = silu(z.v[j]);
          quad(o, q) = z;
        });
    return;
  }
  View1D o(out.data(), n), p(pre.data(), save ? n : 0);
  Kokkos::parallel_for(
      "silu", RangePolicy(0, n), KOKKOS_LAMBDA(int i) {
        const Net z = o(i);
        if (save) p(i) = z;
        o(i) = silu(z);
      });
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
  Kokkos::parallel_for(
      "compress", RangePolicy(0, out.extent(0) * D), KOKKOS_LAMBDA(int i) {
        const int k = i / D, d = i % D;
        const Net z = (gemm ? out(k, d) : Net(0)) + b(d) + tab(nsp(k), d) + wx(d, 0) * (Net) ev(k, 0) +
                      wx(d, 1) * (Net) ev(k, 1) + wx(d, 2) * (Net) ev(k, 2) + wx(d, 3) * (Net) dist(k);
        if (save) sav(k, d) = z;
        out(k, d) = silu(z);
      });
}

// One warp per edge: silu' in place, then the four geometry sums as one reduction.
// Nothing is computed for the species blocks, which are constants.
void compress_bwd(View2D g, View2D pre, const CompressFold& f, View2D x4_adj, View2D ie_adj) {
  const int D = g.extent(1);
  auto wx = f.wx;
  Kokkos::parallel_for(
      "compress_bwd", TeamPolicy(g.extent(0), 1, kLanes), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank();
        Net a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& s0, Net& s1, Net& s2, Net& s3) {
              const Net x = pre(r, d), sg = sigmoid(x);
              const Net gd = g(r, d) * sg * (Net(1) + x * (Net(1) - sg));
              g(r, d) = gd;
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
void readout_accumulate(View1D e, View2D node_pred, View2D edge_pred, const PackedEdges& pk,
                        bool acc) {
  auto off = pk.off;
  auto cut = pk.cut;
  Kokkos::parallel_for(
      "readout", RangePolicy(0, e.extent(0)), KOKKOS_LAMBDA(int n) {
        double s = node_pred(n, 0);
        for (int k = off(n); k < off(n + 1); ++k) s += (double) cut(k) * (double) edge_pred(k, 0);
        e(n) = acc ? Net(e(n) + s) : Net(s);
      });
}

void assemble_energy(RView1D per_atom, View1D net, IView1D species, RView1D composition,
                     double scale, int n_local) {
  Kokkos::parallel_for(
      "assemble", RangePolicy(0, per_atom.extent(0)),
      KOKKOS_LAMBDA(int n) { per_atom(n) = n < n_local ? scale * net(n) + composition(species(n)) : 0.0; });
}

}  // namespace pet
