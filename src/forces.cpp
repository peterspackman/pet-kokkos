// The geometry end of the backward: from the adjoints the network leaves on each
// edge to per-atom forces and a per-structure virial, including the adaptive
// cutoff's share.
//
// Every reduction here is a gather in a fixed order, never a float atomic, so a
// result is reproducible to the bit: a relaxation is a chaotic map, and last-bit
// noise in the forces grows into different relaxed structures. The one atomic
// fallback, for a raw edge list without per-atom ranges, says so.
#include "ops.hpp"

#include "pet/cutoff.hpp"

#include <stdexcept>

namespace pet {

namespace {


// The smooth cutoff factors' derivatives in the distance, in float like the
// factors themselves (pet/cutoff.hpp).
KOKKOS_INLINE_FUNCTION double bump_ddist(double d, double rc, double w) {
  const float x = (float) ((d - (rc - w)) / w);
  if (x <= 0.0f || x >= 1.0f) return 0.0;
  const float si = Kokkos::sin((float) detail::PET_PI * x);
  const float t = Kokkos::tanh(Kokkos::cos((float) detail::PET_PI * x) / si);
  const float du = (-(float) detail::PET_PI / (si * si)) / (float) w;
  return 0.5 * (1.0 - (double) (t * t)) * (double) du;
}
KOKKOS_INLINE_FUNCTION double cosine_ddist(double d, double rc, double w) {
  const float x = (float) ((d - (rc - w)) / w);
  if (x <= 0.0f || x >= 1.0f) return 0.0;
  return 0.5 * (double) (-(float) detail::PET_PI * Kokkos::sin((float) detail::PET_PI * x)) / w;
}
KOKKOS_INLINE_FUNCTION double cutoff_ddist(double d, double rc, double w, bool bump) {
  return bump ? bump_ddist(d, rc, w) : cosine_ddist(d, rc, w);
}

// Forces and a per-atom virial from per-edge gradients g = dE/dv. The edge list
// is full directed -- every (i, j, shift) has its (j, i, -shift) -- so the edges
// that point AT atom a are the reverses of a's own, and
//   F(a) = sum over a's edges k of [g(k) - g(reverse(k))],
// which each atom gathers alone, in edge order.
void fold_edge_gradients(Workspace& ws, RView2D grad, const PackedEdges& pk, IView1D sid, int N,
                         int NS, double scale, RView2D forces, RView2D vir9) {
  RView2D vir_atom = ws.r2("fold:vir_atom", N, 9);
  auto off = pk.off, rev = pk.reverse;
  auto vec = pk.vec;
  const int nc = pk.off.extent(0) - 1;  // atoms with edges; the rest are only neighbours
  Kokkos::parallel_for(
      "fold_forces", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
        double f[3] = {0, 0, 0}, w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = i < nc ? off(i) : 0; k < (i < nc ? off(i + 1) : 0); ++k) {
          for (int c = 0; c < 3; ++c) f[c] += grad(k, c);
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) w[a * 3 + b] += vec(k, a) * grad(k, b);
          if (rev(k) >= 0)
            for (int c = 0; c < 3; ++c) f[c] -= grad(rev(k), c);
        }
        for (int c = 0; c < 3; ++c) forces(i, c) = scale * f[c];
        for (int t = 0; t < 9; ++t) vir_atom(i, t) = scale * w[t];
      });
  sum_by_structure(vir_atom, structure_offsets(ws, "fold", sid, N, NS), vir9, false);
}

// The adaptive cutoff's share of the forces and virial. Each atom's cutoff
// depends on the distances of all its raw (pre-cutoff) edges, so adapted_adj --
// dE/d(the cutoff of atom a) -- reaches every raw edge. Both schemes reduce to one
// per-edge scalar gmag = d(cutoff of the centre)/d(distance) * adapted_adj / d,
// computed by method; the gather into forces and virial is shared.
//
// The scheme always tapers with the bump function, whatever the model's own
// cutoff function: metatrain's adaptive_cutoff.py does, and so do the forwards.
void adaptive_backward(Workspace& ws, const DeviceEdgeData& dev, const Hypers& h, RView1D probes,
                       int P, RView1D adapted_adj, double scale, RView2D forces, RView2D vir9,
                       RView2D edge_grad) {
  const int N = dev.n_atoms, E = dev.n_raw, NS = dev.n_struct;
  const bool solver = h.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver;
  if (E == 0 || (!solver && P < 2)) return;
  auto center = dev.raw_center, neigh = dev.raw_neigh, roff = dev.raw_off, rrev = dev.raw_reverse;
  auto sid = dev.struct_id;
  auto dist = dev.raw_dist;
  auto vec = dev.raw_vec;
  const double width = h.cutoff_width_adaptive, target = h.num_neighbors_adaptive;
  RView1D gmag = ws.r1("ad:gmag", E);

  if (solver) {
    // The forward ends with one implicit-function step,
    //   cutoff = r - (n_total(r) - target) / dn,
    // with r and dn held constant, so d(cutoff)/d(d_j) = -(1/dn) dn_total/dd_j,
    // and dn_total/dd_j = -dbump/dr_j. dn = 0 marks a clamped cutoff.
    auto r = dev.adapt_r, dn = dev.adapt_dn;
    if ((int) r.extent(0) != N || (int) dn.extent(0) != N)
      throw std::runtime_error(
          "pet: the solver adaptive backward needs the forward's saved root and slope "
          "(DeviceEdgeData::adapt_r / adapt_dn)");
    Kokkos::parallel_for(
        "ad_gmag_solver", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const int c = center(e);
          const double d = dist(e);
          if (d <= 0.0 || dn(c) <= 0.0) return (void) (gmag(e) = 0.0);
          gmag(e) = scale * adapted_adj(c) * (-bump_ddist(d, r(c), width) / dn(c)) / d;
        });
  } else {
    // The grid scheme: the cutoff is a softmax-weighted average of probe radii,
    // weighted by how close each probe's smoothed neighbour count is to target.
    // Its backward, stage by stage:
    //   eff(a, p)  = sum over a's edges of bump(d, probe p)   (kept from the forward if we can)
    //   diff       = eff - target + target (p / (P-1))^3
    //   grad       = |torch.gradient(diff)|, clamped at 1e-12; gsign = its derivative
    //   w          = softmax_p(-0.5 (diff / grad)^2), cutoff = sum_p probe_p w_p
    // then the adjoints back through w, the gradient stencil, and eff.
    RView2D eff = dev.adapt_eff;
    const bool have_eff = (int) eff.extent(0) == N && (int) eff.extent(1) == P;
    if (!have_eff) {
      eff = ws.r2("ad:eff", N, P);
      Kokkos::deep_copy(ExecSpace(), eff, 0.0);
      Kokkos::parallel_for(
          "ad_eff", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
            for (int p = 0; p < P; ++p) Kokkos::atomic_add(&eff(center(e), p), detail::dev_bump_cutoff(dist(e), probes(p), width));
          });
    }
    RView2D diff = ws.r2("ad:diff", N, P), grad = ws.r2("ad:grad", N, P), gsign = ws.r2("ad:gsign", N, P),
            w = ws.r2("ad:w", N, P), Cq = ws.r2("ad:Cq", N, P), diff_adj = ws.r2("ad:diffadj", N, P);
    RView1D cutoff = ws.r1("ad:adapted", N);
    Kokkos::parallel_for(
        "ad_diff", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {
          const int a = i / P, p = i % P;
          const double x = (double) p / (P - 1);
          diff(a, p) = eff(a, p) - target + target * x * x * x;
        });
    Kokkos::parallel_for(
        "ad_grad", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {
          const int a = i / P, p = i % P;
          const double g = p == 0       ? diff(a, 1) - diff(a, 0)
                           : p == P - 1 ? diff(a, P - 1) - diff(a, P - 2)
                                        : 0.5 * (diff(a, p + 1) - diff(a, p - 1));
          const double ag = Kokkos::fabs(g);
          grad(a, p) = (ag > 1e-12) ? ag : 1e-12;
          gsign(a, p) = (ag >= 1e-12) ? ((g > 0.0) ? 1.0 : (g < 0.0 ? -1.0 : 0.0)) : 0.0;
        });
    Kokkos::parallel_for(
        "ad_w", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
          double mx = -1e300;
          for (int p = 0; p < P; ++p) {
            w(a, p) = -0.5 * (diff(a, p) / grad(a, p)) * (diff(a, p) / grad(a, p));
            if (w(a, p) > mx) mx = w(a, p);
          }
          double s = 0.0;
          for (int p = 0; p < P; ++p) s += (w(a, p) = Kokkos::exp(w(a, p) - mx));
          double c = 0.0;
          for (int p = 0; p < P; ++p) c += probes(p) * (w(a, p) /= s);
          cutoff(a) = c;
        });
    Kokkos::parallel_for(
        "ad_C", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {
          const int a = i / P, p = i % P;
          const double A = adapted_adj(a) * w(a, p) * (probes(p) - cutoff(a));
          const double gd = grad(a, p);
          diff_adj(a, p) = A * (-(diff(a, p) / (gd * gd)));
          Cq(a, p) = A * (diff(a, p) * diff(a, p) / (gd * gd * gd)) * gsign(a, p);
        });
    Kokkos::parallel_for(
        "ad_B", RangePolicy(0, N * P), KOKKOS_LAMBDA(int i) {  // adjoint of torch.gradient
          const int a = i / P, r = i % P;
          double B = 0.0;
          if (r - 1 >= 1 && r - 1 <= P - 2) B += 0.5 * Cq(a, r - 1);
          if (r + 1 >= 1 && r + 1 <= P - 2) B += -0.5 * Cq(a, r + 1);
          if (r == 1) B += Cq(a, 0);
          if (r == 0) B += -Cq(a, 0);
          if (r == P - 1) B += Cq(a, P - 1);
          if (r == P - 2) B += -Cq(a, P - 1);
          diff_adj(a, r) += B;
        });
    Kokkos::parallel_for(
        "ad_gmag_grid", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const double d = dist(e);
          if (d <= 0.0) return (void) (gmag(e) = 0.0);
          double da = 0.0;
          for (int p = 0; p < P; ++p) da += diff_adj(center(e), p) * bump_ddist(d, probes(p), width);
          gmag(e) = scale * da / d;
        });
  }

  if (edge_grad.extent(0) == (size_t) E)
    Kokkos::parallel_for(
        "ad_edge_grad", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          for (int c = 0; c < 3; ++c) edge_grad(e, c) += gmag(e) * vec(e, c);
        });

  if (roff.extent(0) != (size_t) (N + 1) || rrev.extent(0) != (size_t) E) {
    // No per-atom ranges: scatter with atomics. Correct, not reproducible.
    Kokkos::parallel_for(
        "ad_scatter", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const int c = center(e), j = neigh(e), b = NS > 1 ? sid(c) : 0;
          for (int x = 0; x < 3; ++x) {
            const double g = gmag(e) * vec(e, x);
            Kokkos::atomic_add(&forces(c, x), g);
            Kokkos::atomic_add(&forces(j, x), -g);
            for (int y = 0; y < 3; ++y) Kokkos::atomic_add(&vir9(b, y * 3 + x), vec(e, y) * g);
          }
        });
    return;
  }
  // The same gather as fold_edge_gradients, over the raw list. The partner's
  // vector is minus this edge's and it enters with a minus sign, so it adds.
  RView2D vir_atom = ws.r2("ad:vir_atom", N, 9);
  Kokkos::parallel_for(
      "ad_gather", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        double f[3] = {0, 0, 0}, w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int e = roff(a); e < roff(a + 1); ++e) {
          const double ge = gmag(e);
          for (int c = 0; c < 3; ++c) f[c] += ge * vec(e, c);
          for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y) w[x * 3 + y] += vec(e, x) * ge * vec(e, y);
          if (rrev(e) >= 0)
            for (int c = 0; c < 3; ++c) f[c] += gmag(rrev(e)) * vec(e, c);
        }
        for (int c = 0; c < 3; ++c) forces(a, c) += f[c];
        for (int t = 0; t < 9; ++t) vir_atom(a, t) = w[t];
      });
  sum_by_structure(vir_atom, structure_offsets(ws, "ad", sid, N, NS), vir9, true);
}

}  // namespace

IView1D structure_offsets(Workspace& ws, const std::string& key, IView1D sid, int N, int NS) {
  IView1D soff = ws.i1(key + ":soff", NS + 1);
  if (NS == 1) {
    Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 0), 0);
    Kokkos::deep_copy(ExecSpace(), Kokkos::subview(soff, 1), N);
    return soff;
  }
  IView1D count = ws.i1(key + ":scount", NS);
  Kokkos::deep_copy(ExecSpace(), count, 0);
  Kokkos::parallel_for(
      "struct_count", RangePolicy(0, N), KOKKOS_LAMBDA(int a) { Kokkos::atomic_inc(&count(sid(a))); });
  Kokkos::parallel_scan(
      "struct_scan", RangePolicy(0, NS), KOKKOS_LAMBDA(int b, int& upd, bool final) {
        if (final) soff(b) = upd;
        upd += count(b);
        if (final && b == NS - 1) soff(NS) = upd;
      });
  return soff;
}

void sum_by_structure(RView2D x, IView1D soff, RView2D out, bool acc) {
  const int T = x.extent(1);
  Kokkos::parallel_for(
      "struct_sum", RangePolicy(0, out.extent(0) * T), KOKKOS_LAMBDA(int i) {
        const int b = i / T, t = i % T;
        double s = 0.0;
        for (int a = soff(b); a < soff(b + 1); ++a) s += x(a, t);
        out(b, t) = acc ? out(b, t) + s : s;
      });
}

namespace {

// The adaptive cutoff's share: each kept edge's pair-cutoff adjoint, half to
// each end, then back through each atom's cutoff to its raw edges.
void adaptive_part(Workspace& ws, const DeviceEdgeData& dev, const Hypers& h, RView1D probes, int n_probes,
                   const PackedEdges& pk, RView1D pc_adj, double scale, RView2D forces, RView2D vir9,
                   RView2D edge_grad) {
  const int N = dev.n_atoms;
  auto off = pk.off, rev = pk.reverse;
  RView1D adapted_adj = ws.r1("adapted_adj", N);
  const int nc = pk.off.extent(0) - 1;
  Kokkos::parallel_for(
      "adapted_adj", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
        double s = 0.0;
        for (int k = a < nc ? off(a) : 0; k < (a < nc ? off(a + 1) : 0); ++k) {
          s += 0.5 * pc_adj(k);
          if (rev(k) >= 0) s += 0.5 * pc_adj(rev(k));
        }
        adapted_adj(a) = s;
      });
  // Over several ranks: an edge to a ghost owes the ghost half its pair-cutoff
  // adjoint too (the partner edge owes this rank's atom the other half, there);
  // gathered per ghost over the partnerless edges into it, then sent home.
  if (dev.exchange) {
    if (dev.orphan_edge.extent(0) > 0) {
      auto ooff = dev.orphan_off, oedge = dev.orphan_edge, slot = dev.raw_slot, slot_edge = pk.slot_edge;
      Kokkos::parallel_for(
          "remote_adapted", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            for (int k = ooff(a); k < ooff(a + 1); ++k) {
              const int s = slot(oedge(k)), p = s >= 0 ? slot_edge(s) : -1;
              if (p >= 0) adapted_adj(a) += 0.5 * pc_adj(p);
            }
          });
    }
    dev.exchange->atoms_reverse(adapted_adj);
  }
  adaptive_backward(ws, dev, h, probes, n_probes, adapted_adj, scale, forces, vir9, edge_grad);
}

}  // namespace

void forces_and_virial(Workspace& ws, const DeviceEdgeData& dev, const PackedEdges& pk,
                       const Hypers& h, RView1D probes, int n_probes, double scale, View2D x4_adj,
                       View1D cutoff_adj, View2D cf_seq_adj, RView2D& forces, RView2D& vir9,
                       RView2D& edge_grad) {
  const int N = dev.n_atoms, E = pk.E;
  const bool adaptive = h.adaptive(), bump = h.cutoff_function == CutoffFunction::Bump;
  const double width = h.cutoff_width;

  // dE/dv per edge: through (v, |v|) directly, and through |v| into the cutoff
  // factor, which the readout and the attention bias both use. With an adaptive
  // cutoff the factor also depends on the pair cutoff: that share is parked per
  // edge and gathered per atom into adapted_adj, half from each end.
  RView2D grad = ws.r2("edge_grad", E, 3);
  RView1D pc_adj = ws.r1("pc_adj", adaptive ? E : 0);
  auto off = pk.off, center = pk.center, rev = pk.reverse;
  auto vec = pk.vec;
  auto dist = pk.dist, pcut = pk.pcut;
  Kokkos::parallel_for(
      "edge_grad", RangePolicy(0, E), KOKKOS_LAMBDA(int k) {
        const int n = center(k), m = k - off(n);
        const double cutoff_total = cutoff_adj(k) + cf_seq_adj(n, 1 + m);
        const double dcut_dd = cutoff_ddist(dist(k), pcut(k), width, bump);
        const double dist_adj = x4_adj(k, 3) + cutoff_total * dcut_dd;
        const double invd = (dist(k) > 0.0) ? 1.0 / dist(k) : 0.0;
        for (int c = 0; c < 3; ++c) grad(k, c) = x4_adj(k, c) + dist_adj * vec(k, c) * invd;
        if (adaptive) pc_adj(k) = -cutoff_total * dcut_dd;
      });

  forces = ws.r2("forces", N, 3);
  vir9 = ws.r2("virial9", dev.n_struct, 9);
  fold_edge_gradients(ws, grad, pk, dev.struct_id, N, dev.n_struct, scale, forces, vir9);

  // The same per raw edge, in the raw list's order: the kept edge's gradient, 0
  // for a dropped one; the adaptive cutoff adds its share below.
  const int R = dev.n_raw;
  if (dev.raw_slot.extent(0) == (size_t) R && R > 0) {
    edge_grad = ws.r2("raw_edge_grad", R, 3);
    auto slot = dev.raw_slot, slot_edge = pk.slot_edge;
    Kokkos::parallel_for(
        "raw_edge_grad", RangePolicy(0, R), KOKKOS_LAMBDA(int e) {
          const int k = slot(e) >= 0 ? slot_edge(slot(e)) : -1;
          for (int c = 0; c < 3; ++c) edge_grad(e, c) = k >= 0 ? scale * grad(k, c) : 0.0;
        });
  } else {
    edge_grad = RView2D();
  }
  if (adaptive) adaptive_part(ws, dev, h, probes, n_probes, pk, pc_adj, scale, forces, vir9, edge_grad);

  // The folds above take an edge's share of its target's force from the partner
  // edge; an edge with none gives it here, from its own gradient.
  if (dev.orphan_edge.extent(0) > 0 && edge_grad.extent(0) > 0) {
    auto ooff = dev.orphan_off, oedge = dev.orphan_edge;
    auto eg = edge_grad;
    Kokkos::parallel_for(
        "orphan_forces", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
          for (int k = ooff(a); k < ooff(a + 1); ++k)
            for (int c = 0; c < 3; ++c) forces(a, c) -= eg(oedge(k), c);
        });
  }
}

}  // namespace pet
