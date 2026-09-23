// Dense layers, norms, the feedforward blocks and compress.0: see ops.hpp.
#include "ops.hpp"

#include "pet/gemm.hpp"
#include "pet/ozaki.hpp"

namespace pet {

// Every GEMM goes through gemm_ozaki, which is the vendor GEMM unless the Ozaki
// path is on. The backward uses the same path as the forward, so a second
// derivative is not limited by a less accurate backward.
void linear(View2D out, View2D in, const WeightRef& W, View1D b, Net beta) {
  gemm_ozaki('N', 'T', Net(1), in, W.v, beta, out, W.for_orientation(true), b);
}

void linear_bwd(View2D in_adj, View2D out_adj, const WeightRef& W, Net beta) {
  gemm_ozaki('N', 'N', Net(1), out_adj, W.v, beta, in_adj, W.for_orientation(false));
}

void linear_silu(View2D out, View2D pre, View2D in, const WeightRef& W, View1D b) {
  linear(out, in, W, b);
  const int n = out.extent(0) * out.extent(1);
  const bool save = pre.data() != nullptr;
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
  View1D g(grad.data(), n), p(pre.data(), n);
  Kokkos::parallel_for(
      "silu_bwd", RangePolicy(0, n), KOKKOS_LAMBDA(int i) {
        const Net x = p(i), sg = sigmoid(x);
        g(i) *= sg * (Net(1) + x * (Net(1) - sg));
      });
}

void gather(View2D out, View2D table, IView1D idx) {
  const int D = out.extent(1);
  Kokkos::parallel_for(
      "gather", RangePolicy(0, out.extent(0) * D),
      KOKKOS_LAMBDA(int i) { out(i / D, i % D) = table(idx(i / D), i % D); });
}

void add_inplace(View2D a, View2D b) {
  const int n = a.extent(0) * a.extent(1);
  View1D a1(a.data(), n), b1(b.data(), n);
  Kokkos::parallel_for("add", RangePolicy(0, n), KOKKOS_LAMBDA(int i) { a1(i) += b1(i); });
}

void copy(View2D a, View2D b) {
  const int n = a.extent(0) * a.extent(1);
  View1D a1(a.data(), n), b1(b.data(), n);
  Kokkos::parallel_for("copy", RangePolicy(0, n), KOKKOS_LAMBDA(int i) { a1(i) = b1(i); });
}

// ---- norms --------------------------------------------------------------------
//
// RMSNorm and LayerNorm are one operation: out = (x - mu) * inv * weight (+ bias),
// where LayerNorm centres (mu = mean) and RMSNorm does not (mu = 0). One warp per
// row, in the network type (as torch does), and in two passes so LayerNorm's
// variance never subtracts two large squares; the second pass reads from L1.

constexpr double kRmsNormEps = 1.1920928955078125e-07;  // float32 epsilon
constexpr double kLayerNormEps = 1e-5;

KOKKOS_INLINE_FUNCTION void row_stats(const Team& t, const View2D& x, int r, bool ln, Net& mu,
                                      Net& inv) {
  const int D = x.extent(1);
  mu = Net(0);
  if (ln) {
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, D), [&](int d, Net& s) { s += x(r, d); }, mu);
    mu /= D;
  }
  Net v = Net(0);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(t, D), [&](int d, Net& s) { s += (x(r, d) - mu) * (x(r, d) - mu); }, v);
  inv = Net(1) / Kokkos::sqrt(v / D + Net(ln ? kLayerNormEps : kRmsNormEps));
}

void norm_fwd(View2D out, View2D in, View1D weight, View1D bias) {
  const bool ln = bias.extent(0) > 0;
  Kokkos::parallel_for(
      "norm", TeamPolicy(in.extent(0), 1, kLanes), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank();
        Net mu, inv;
        row_stats(t, in, r, ln, mu, inv);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, in.extent(1)), [&](int d) {
          out(r, d) = (in(r, d) - mu) * inv * weight(d) + (ln ? bias(d) : Net(0));
        });
      });
}

// With xhat = (x - mu) * inv and gw = out_adj * weight:
//   in_adj = inv * (gw - [mean(gw) for LayerNorm] - xhat * mean(gw * xhat)).
// The bias is a shift, so it does not appear.
void norm_bwd(View2D in_adj, View2D out_adj, View2D in, View1D weight, bool ln, bool acc) {
  const int D = in.extent(1);
  Kokkos::parallel_for(
      "norm_bwd", TeamPolicy(in.extent(0), 1, kLanes), KOKKOS_LAMBDA(const Team& t) {
        const int r = t.league_rank();
        Net mu, inv, sg = Net(0), sgx = Net(0);
        row_stats(t, in, r, ln, mu, inv);
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(t, D),
            [&](int d, Net& a, Net& b) {
              const Net gw = out_adj(r, d) * weight(d);
              a += gw;
              b += gw * (in(r, d) - mu) * inv;
            },
            sg, sgx);
        const Net mg = ln ? sg / D : Net(0), mgx = sgx / D;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, D), [&](int d) {
          in_adj(r, d) = (acc ? in_adj(r, d) : Net(0)) +
                         inv * (out_adj(r, d) * weight(d) - mg - (in(r, d) - mu) * inv * mgx);
        });
      });
}

// ---- SwiGLU -------------------------------------------------------------------

void swiglu_in(View2D pre, View2D h, View2D in, const WeightRef& w_in, View1D b_in) {
  if (!w_in.split && swiglu_in_fused(pre, h, in, w_in.v, b_in)) return;
  linear(pre, in, w_in, b_in);
  if (!h.data()) return;
  const int F = h.extent(1);
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
    Kokkos::parallel_for(
        "swiglu_bwd", RangePolicy(0, R * F), KOKKOS_LAMBDA(int i) {
          const int r = i / F, c = i % F;
          const Net v = pre(r, 2 * c), sg = sigmoid(pre(r, 2 * c + 1));
          pre_adj(r, 2 * c) = h_adj(r, c) * sg;
          pre_adj(r, 2 * c + 1) = h_adj(r, c) * v * sg * (Net(1) - sg);
        });
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
  if (gemm) gemm_ozaki('N', 'T', Net(1), input_edge, f.wi.v, Net(0), out, f.wi.for_orientation(true));
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
                     double scale) {
  Kokkos::parallel_for(
      "assemble", RangePolicy(0, per_atom.extent(0)),
      KOKKOS_LAMBDA(int n) { per_atom(n) = scale * net(n) + composition(species(n)); });
}

}  // namespace pet
