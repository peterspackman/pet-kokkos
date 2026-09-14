// On-device NEF builder: turns a raw COO directed-edge list (already within the
// search cutoff) into a fully device-resident DeviceEdgeData, with the adaptive
// cutoff, NEF packing, and reverse-edge index all computed in Kokkos kernels.
//
// This is the "zero host round-trip" path, and the one a relaxation or MD driver
// should take: the raw edges are built on the device (build_nef_device, in
// device_geometry.hpp) and the output feeds PetModel::compute without ever
// touching the host. The tests also route
// through it (by uploading their host raw edges first), so the device build is
// validated on the Serial backend against the goldens.
//
// Slot assignment within an atom's neighbour row is a per-atom walk over that
// atom's own contiguous edge range, so a slot is a function of the edge list
// alone. It is deliberately NOT an atomic counter: the network is fp32, and
// handing out slots in thread-arrival order made every reduction over the
// neighbour axis round differently run to run. See the "pet_scatter" kernel.
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pet/config.hpp"
#include "pet/kokkos.hpp"
#include "pet/model.hpp"  // DeviceEdgeData

namespace pet {
namespace detail {

// pi as a literal: M_PI is a host <cmath> macro not guaranteed in device code.
constexpr double PET_PI = 3.14159265358979323846;

// Smooth cutoff factor (device). Mirrors host bump_cutoff / cosine_cutoff. The
// transcendental (tanh/tan/cos) is evaluated in float -- it is O(E*P) in the
// adaptive cutoff and double transcendentals are very slow on consumer GPUs; the
// cutoff factor only needs network (fp32) accuracy. Forward (here) and backward
// (cutoff_val in model.cpp) use the same float formula so the gradient stays
// consistent with the forward.
KOKKOS_INLINE_FUNCTION double dev_bump_cutoff(double d, double rc, double width) {
  const float scaled = (float) ((d - (rc - width)) / width);
  if (scaled <= 0.0f) return 1.0;
  if (scaled >= 1.0f) return 0.0;
  return 0.5 * (1.0 + (double) Kokkos::tanh(1.0f / Kokkos::tan((float) PET_PI * scaled)));
}
KOKKOS_INLINE_FUNCTION double dev_cosine_cutoff(double d, double rc, double width) {
  const float scaled = (float) ((d - (rc - width)) / width);
  const float x = Kokkos::fmin(Kokkos::fmax(scaled, 0.0f), 1.0f);
  return 0.5 * (1.0 + (double) Kokkos::cos((float) PET_PI * x));
}
KOKKOS_INLINE_FUNCTION double dev_cutoff_value(double d, double rc, double width, bool is_bump) {
  return is_bump ? dev_bump_cutoff(d, rc, width) : dev_cosine_cutoff(d, rc, width);
}

// d/d(rc) of dev_bump_cutoff at fixed distance. With s = (d - rc + w)/w and
// f = 0.5*(1 + tanh(cot(pi*s))),
//     df/ds  = -(pi/2) * sech^2(cot(pi s)) / sin^2(pi s)
//     ds/drc = -1/w
//   => df/drc = (pi / 2w) * sech^2(cot(pi s)) / sin^2(pi s),  positive.
//
// This is exactly -bump_ddist() from model.cpp, in the same float arithmetic, and
// that identity is load-bearing: the solver root-finds on dev_bump_cutoff, so its
// derivative has to be the derivative of THAT function rather than of the exact
// one, and the backward reuses bump_ddist for the same quantity.
//
// The clamp keeps sech^2/sin^2 off its 0*inf corner: as s -> 0 or 1 the cotangent
// diverges, tanh saturates to +-1 in float, and the product becomes 0 * inf = NaN
// unless the singular factor is kept finite. metatrain clamps to the same
// [1e-6, 1-1e-6] for the same reason.
KOKKOS_INLINE_FUNCTION double dev_bump_dcutoff_dr(double d, double rc, double width) {
  const float scaled = (float) ((d - (rc - width)) / width);
  if (scaled <= 0.0f || scaled >= 1.0f) return 0.0;
  const float safe = Kokkos::fmin(Kokkos::fmax(scaled, 1e-6f), 1.0f - 1e-6f);
  const float ps = (float) PET_PI * safe;
  const float si = Kokkos::sin(ps);
  const float tt = Kokkos::tanh(Kokkos::cos(ps) / si);
  const float dudr = ((float) PET_PI / (si * si)) / (float) width;
  return 0.5 * (1.0 - (double) (tt * tt)) * (double) dudr;
}

// Lower bound on a solver-chosen cutoff, as a fraction of the model cutoff, and
// the number of Newton-bisection iterations. Both match metatrain
// (adaptive_cutoff.py: min_cutoff_factor = 1/16, range(10)); changing either
// changes which cutoff an atom gets, so they are not tuning knobs.
constexpr double PET_SOLVER_MIN_CUTOFF_FACTOR = 1.0 / 16.0;
constexpr int PET_SOLVER_ITERS = 10;
constexpr double PET_SOLVER_DN_FLOOR = 1e-6;

// Pack (center, neigh-owner, cell-shift) into a 64-bit reverse-matching key.
// Layout: center:24 | neigh:24 | (sa+SHIFT_BIAS):5 | (sb+..):5 | (sc+..):5  = 63 bits.
// Supports up to 16,777,216 local+ghost atoms per rank and cell shifts in
// [-15, 15] per axis (far beyond any realistic neighbor image). Out-of-range
// values are detected on the host (see build_device_edge_data) and reported.
constexpr int PET_KEY_SHIFT_BIAS = 15;
constexpr int PET_KEY_ATOM_BITS = 24;
constexpr long long PET_KEY_ATOM_MAX = (1LL << PET_KEY_ATOM_BITS);  // exclusive
KOKKOS_INLINE_FUNCTION uint64_t pet_pack_key(int i, int j, int sa, int sb, int sc) {
  const uint64_t ui = static_cast<uint64_t>(i);
  const uint64_t uj = static_cast<uint64_t>(j);
  const uint64_t ua = static_cast<uint64_t>(sa + PET_KEY_SHIFT_BIAS);
  const uint64_t ub = static_cast<uint64_t>(sb + PET_KEY_SHIFT_BIAS);
  const uint64_t uc = static_cast<uint64_t>(sc + PET_KEY_SHIFT_BIAS);
  return (ui << 39) | (uj << 15) | (ua << 10) | (ub << 5) | uc;
}

}  // namespace detail

// Build a device-resident DeviceEdgeData from a raw COO directed-edge list.
//
//   N        number of (owned) atoms that act as centers
//   species  [N] device view, species index per atom (already Z -> [0,n_species))
//   re_i     [E] center local index
//   re_j     [E] neighbor owner local index (the atom a periodic/MPI ghost images)
//   re_shift [E,3] integer cell shift of the neighbor image (for reverse matching)
//   re_vec   [E,3] edge vector r_j(+image) - r_i (double)
//   re_dist  [E] edge distance (double)
//
// `re_*` must be the FULL list within the search cutoff (before adaptive masking);
// it is retained verbatim as DeviceEdgeData::raw_* for the adaptive backward.
using EdgeMap = Kokkos::UnorderedMap<uint64_t, int, ExecSpace>;

// Zero-allocation builder: all scratch and output buffers come from the persistent
// `ws` pool, the reverse-matching `edge_map` is reused (rehash-on-grow + clear),
// and the constant `probes`/P grid is precomputed by the caller. In steady-state
// MD (stable N and max-neighbor count) this performs no device allocation.
inline DeviceEdgeData build_device_edge_data(Workspace& ws, EdgeMap& edge_map, int& m_high,
                                             RView1D probes, int P, int N, IView1D species,
                                             IView1D re_i, IView1D re_j, IView2D re_shift,
                                             RView2D re_vec, RView1D re_dist, int E,
                                             const Hypers& h) {
  using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
  using Kokkos::MDRangePolicy;
  using Kokkos::Rank;

  const bool adaptive = h.adaptive();
  const bool is_bump = (h.cutoff_function == CutoffFunction::Bump);
  // Two different tapers, and they are not interchangeable: `width` tapers each
  // EDGE's own cutoff factor, `width_adaptive` tapers the SMOOTHED NEIGHBOUR
  // COUNT the adaptive scheme searches over. metatrain passes them separately
  // (cutoff_width / cutoff_width_adaptive); a checkpoint predating the split
  // carries only the first, and the loader mirrors it into the second.
  const double width = h.cutoff_width;
  const double width_adaptive = h.cutoff_width_adaptive;
  const double model_cutoff = h.cutoff;

  DeviceEdgeData dev;
  dev.n_atoms = N;
  dev.n_raw = E;
  dev.species = species;
  // raw_* alias the input COO list (the adaptive backward consumes them as-is)
  dev.raw_center = re_i;
  dev.raw_neigh = re_j;
  dev.raw_dist = re_dist;
  dev.raw_vec = re_vec;

  // ---- per-atom raw-edge ranges -------------------------------------------
  // The COO list is already segmented by centre atom: the caller counts atom gi's
  // edges exactly, prefix-sums the counts, and fills [eoff(gi), eoff(gi+1)) in a
  // fixed order (see pet_neigh_fill in device_geometry.hpp). So re_i is
  // non-decreasing with no gaps, and recovering the ranges here costs one integer
  // count plus a scan.
  //
  // This exists so the kernels below can walk ONE ATOM'S OWN EDGES IN ORDER rather
  // than scattering with atomics. Atomic order is nondeterministic, and the network
  // is fp32, so every reduction over the neighbour axis rounded differently run to
  // run -- the relaxer is a chaotic map and amplified that into multi-kJ/mol
  // differences in the relaxed energy, enough to reshuffle a landscape ranking
  // between two identical runs. Integer counts are order-independent in value, so
  // the count/scan below is itself safe.
  IView1D roff = ws.i1("nef:roff", N + 1);
  {
    IView1D rcnt = ws.i1("nef:rcnt", N);
    Kokkos::deep_copy(ExecSpace(), rcnt, 0);  // atomic accumulator, zeroed explicitly
    Kokkos::parallel_for(
        "pet_raw_count", RangePolicy(0, E),
        KOKKOS_LAMBDA(int e) { Kokkos::atomic_inc(&rcnt(re_i(e))); });
    Kokkos::parallel_scan(
        "pet_raw_scan", RangePolicy(0, N), KOKKOS_LAMBDA(int a, int& upd, bool final) {
          if (final) roff(a) = upd;
          upd += rcnt(a);
          if (final && a == N - 1) roff(N) = upd;
        });
    if (N == 0) Kokkos::deep_copy(ExecSpace(), roff, 0);
  }
  dev.raw_off = roff;

  // Partner of each raw edge: (i,j,shift) <-> (j,i,-shift). The list is full
  // directed and the keep test is symmetric in i and j, so the partner always
  // exists; -1 is kept as a guard rather than an expectation.
  //
  // Found by scanning atom j's own run rather than through a hash map: the runs are
  // short (one atom's neighbours within the cutoff), it costs no extra memory, and
  // it happens once per neighbour build rather than once per use.
  IView1D raw_rev = ws.i1("nef:rawrev", E);
  {
    Kokkos::parallel_for(
        "pet_raw_reverse", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          const int i = re_i(e), j = re_j(e);
          const int sa = -re_shift(e, 0), sb = -re_shift(e, 1), sc = -re_shift(e, 2);
          int found = -1;
          for (int f = roff(j); f < roff(j + 1); ++f) {
            if (re_j(f) == i && re_shift(f, 0) == sa && re_shift(f, 1) == sb &&
                re_shift(f, 2) == sc) {
              found = f;
              break;
            }
          }
          raw_rev(e) = found;
        });
  }
  dev.raw_reverse = raw_rev;

  // ---- adaptive per-atom cutoff (adaptive_cutoffs, on device) ----
  //
  // Two schemes, and they do NOT agree -- see AdaptiveCutoffMethod in config.hpp.
  // Both search the same smoothed neighbour count
  //     n_total(r) = sum_j bump(d_j, r, w) + target * (r / r_max)^3
  // for the radius where it reaches `target`; Grid samples it on a fixed probe
  // grid and takes a Gaussian-weighted average of the probes, Solver root-finds
  // it directly.
  //
  // Both use the BUMP taper whatever the model's cutoff_function is: metatrain's
  // adaptive_cutoff.py imports `cutoff_func_bump as cutoff_func` unconditionally,
  // so a Cosine model still gets a bump-shaped neighbour count here.
  RView1D acut = ws.r1("nef:acut", N);
  const bool solver = (h.adaptive_cutoff_method == AdaptiveCutoffMethod::Solver);
  if (adaptive && solver) {
    // Newton-bisection on f(r) = n_total(r) - target.
    //
    // The cubic baseline is what makes this well posed: it runs from 0 at r = 0
    // to `target` at r = r_max, so n_total is non-decreasing on [0, r_max] and
    // crosses `target` exactly once. [0, r_max] therefore brackets the root from
    // the start and never has to be widened.
    //
    // One thread per atom over its own contiguous raw-edge range. That is not
    // just convenient: the per-atom sums accumulate in a fixed order with no
    // atomics, so the root -- and hence which edges survive the keep test below
    // -- is reproducible run to run. A float-atomic reduction here would put
    // last-bit noise directly into a discrete keep/drop decision.
    const double target = h.num_neighbors_adaptive;
    const double rmax = model_cutoff;
    const double inv_rmax = 1.0 / rmax;
    const double lo_bound = rmax * detail::PET_SOLVER_MIN_CUTOFF_FACTOR;
    // The root and the slope there, kept for the backward's implicit-function
    // step rather than recomputed (the backward would otherwise redo the whole
    // solve just to recover dn/dr).
    RView1D ar = ws.r1("nef:adapt_r", N), adn = ws.r1("nef:adapt_dn", N);
    Kokkos::parallel_for(
        "pet_adapt_solver", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
          const int e0 = roff(a), e1 = roff(a + 1);
          double r_lo = 0.0, r_hi = rmax, r = 0.5 * rmax;
          double n = 0.0, dn = 0.0;
          for (int it = 0; it < detail::PET_SOLVER_ITERS; ++it) {
            n = 0.0;
            dn = 0.0;
            for (int e = e0; e < e1; ++e) {
              const double d = re_dist(e);
              n += detail::dev_bump_cutoff(d, r, width_adaptive);
              dn += detail::dev_bump_dcutoff_dr(d, r, width_adaptive);
            }
            const double x = r * inv_rmax;
            n += target * x * x * x;
            dn += 3.0 * target * x * x * inv_rmax;

            const double f = n - target;
            if (f <= 0.0) r_lo = r; else r_hi = r;
            // A Newton step, unless it would leave the bracket -- which happens
            // on the "shoulders" where one bump is crossing from active to
            // saturated and the local slope is tiny. Bisect there instead, so
            // the iteration cannot diverge however flat f gets.
            const double r_newton = r - f / Kokkos::fmax(dn, detail::PET_SOLVER_DN_FLOOR);
            r = (r_newton >= r_lo && r_newton <= r_hi) ? r_newton : 0.5 * (r_lo + r_hi);
          }
          // Re-evaluate at the final r: the loop's n/dn are from the previous
          // iterate, and the implicit-function step below needs both AT the root.
          n = 0.0;
          dn = 0.0;
          for (int e = e0; e < e1; ++e) {
            const double d = re_dist(e);
            n += detail::dev_bump_cutoff(d, r, width_adaptive);
            dn += detail::dev_bump_dcutoff_dr(d, r, width_adaptive);
          }
          {
            const double x = r * inv_rmax;
            n += target * x * x * x;
            dn += 3.0 * target * x * x * inv_rmax;
          }
          const double dn_root = Kokkos::fmax(dn, detail::PET_SOLVER_DN_FLOOR);
          // One trailing implicit-function step. In the converged regime the
          // residual is at float noise and this moves nothing; its purpose is to
          // be the point the BACKWARD differentiates, so gradients reach the
          // distances through the residual instead of through ten iterations.
          double adapted = r - (n - target) / dn_root;
          adapted = Kokkos::fmin(Kokkos::fmax(adapted, lo_bound), rmax);
          acut(a) = adapted;
          ar(a) = r;
          // Sign the slope so the backward knows the clamp was active without
          // recomputing the bound: a clamped cutoff has no gradient.
          adn(a) = (adapted > lo_bound && adapted < rmax) ? dn_root : 0.0;
        });
    dev.adapt_r = ar;
    dev.adapt_dn = adn;
  } else if (adaptive) {
    // Exec-space overload throughout this builder: the space-less deep_copy fences
    // the device before and after (see Workspace::get2).
    Kokkos::deep_copy(ExecSpace(), acut, model_cutoff);
    const double target = h.num_neighbors_adaptive;
    if (P > 0) {
      // eff[a][p] = sum over a's edges of bump(dist, probe_p, width_adaptive).
      // bump(d, probe_p, w) is exactly 1.0 for probe_p >= d+w, exactly 0 for
      // probe_p <= d, and smooth only on the ~w-wide transition in between. The
      // probe grid is uniform (probe_p = p0 + p*dp), so an edge contributes 1.0 to a
      // contiguous tail [p_full, P) and a smooth value to the few transition probes.
      // Mark the tail start in `eff` (one atomic) + add transition partials to `effp`
      // (~4 atomics), then a per-atom prefix sum turns tail-marks into counts and
      // folds in the partials -- O(E + N*P) vs the dense O(E*P) all-probes scatter
      // (which atomically added even the exact zeros). Result is bit-identical.
      RView2D eff = ws.r2("nef:eff", N, P);   // doubles as the tail-start diff array
      RView2D effp = ws.r2("nef:effp", N, P);  // smooth transition partials
      // One thread per ATOM over its own contiguous edge range, so the partial sums
      // accumulate in a fixed order and no atomics are needed. The adaptive cutoff
      // acut(a) feeds a discrete keep/drop test below, so last-bit noise here could
      // flip an edge in or out of the neighbour list entirely.
      Kokkos::parallel_for(
          "pet_adapt_eff_scatter", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            const double p0 = probes(0);
            const double dp = (P > 1) ? (probes(1) - probes(0)) : 1.0;
            for (int e = roff(a); e < roff(a + 1); ++e) {
              const double d = re_dist(e);
              // p_full: first probe >= d+width_adaptive (tail of exact 1.0s starts here)
              int p_full = (int) Kokkos::ceil((d + width_adaptive - p0) / dp);
              if (p_full < 0) p_full = 0;
              if (p_full > P) p_full = P;
              // p_start: first probe > d (transition region [p_start, p_full))
              int p_start = (int) Kokkos::floor((d - p0) / dp) + 1;
              if (p_start < 0) p_start = 0;
              if (p_full < P) eff(a, p_full) += 1.0;
              for (int p = p_start; p < p_full && p < P; ++p)
                effp(a, p) += detail::dev_bump_cutoff(d, probes(p), width_adaptive);
            }
          });
      Kokkos::parallel_for(
          "pet_adapt_eff_scan", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            double run = 0.0;
            for (int p = 0; p < P; ++p) {
              run += eff(a, p);                 // cumulative count of exact-1.0 tails
              eff(a, p) = run + effp(a, p);      // + smooth transition partials
            }
          });
      dev.adapt_eff = eff;  // reuse in the adaptive backward (skip recomputing K1)

      // Flat RangePolicy with the SECOND index fastest, not MDRangePolicy. On CUDA
      // MDRangePolicy iterates in the execution space's default LayoutLeft, so with
      // bounds {N, P} the atom index varies fastest and consecutive threads stride
      // across a LayoutRight row instead of walking along it. Same issue, and the
      // same fix, as the attention kernels.
      RView2D diffv = ws.r2("nef:diffv", N, P), gradv = ws.r2("nef:gradv", N, P);
      Kokkos::parallel_for(
          "pet_adapt_diff", RangePolicy(0, N * P), KOKKOS_LAMBDA(int _i) {
            const int a = _i / P, p = _i % P;
            const double x = (P > 1) ? static_cast<double>(p) / (P - 1) : 0.0;
            diffv(a, p) = eff(a, p) - target + target * x * x * x;
          });
      Kokkos::parallel_for(
          "pet_adapt_grad", RangePolicy(0, N * P), KOKKOS_LAMBDA(int _i) {
            const int a = _i / P, p = _i % P;
            double g;
            if (P == 1) {
              gradv(a, 0) = Kokkos::fabs(diffv(a, 0)) * 0.5 + 1e-12;
              return;
            }
            if (p == 0) g = diffv(a, 1) - diffv(a, 0);
            else if (p == P - 1) g = diffv(a, P - 1) - diffv(a, P - 2);
            else g = 0.5 * (diffv(a, p + 1) - diffv(a, p - 1));
            gradv(a, p) = Kokkos::fmax(Kokkos::fabs(g), 1e-12);
          });
      Kokkos::parallel_for(
          "pet_adapt_softmax", RangePolicy(0, N), KOKKOS_LAMBDA(int a) {
            double mx = -1e300;
            for (int p = 0; p < P; ++p) {
              const double r = diffv(a, p) / gradv(a, p);
              const double lw = -0.5 * r * r;
              mx = (lw > mx) ? lw : mx;
            }
            double sum = 0.0, accw = 0.0;
            for (int p = 0; p < P; ++p) {
              const double r = diffv(a, p) / gradv(a, p);
              const double w = Kokkos::exp(-0.5 * r * r - mx);
              sum += w;
              accw += probes(p) * w;
            }
            acut(a) = accw / sum;
          });
    }
  }

  // ---- per-edge pair cutoff, keep flag, smooth factor ----
  RView1D rcv = ws.r1("nef:rcv", E), factorv = ws.r1("nef:factorv", E);
  IView1D keepv = ws.i1("nef:keepv", E);
  IView1D count = ws.i1("nef:count", N);
  Kokkos::deep_copy(ExecSpace(), count, 0);  // atomic accumulator, zeroed explicitly
  Kokkos::parallel_for(
      "pet_keep", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        double rc;
        if (adaptive) rc = 0.5 * (acut(re_i(e)) + acut(re_j(e)));
        else rc = model_cutoff;
        rcv(e) = rc;
        const double d = re_dist(e);
        const int keep = (d <= rc) ? 1 : 0;  // non-adaptive: all within cutoff already
        keepv(e) = keep;
        if (keep) {
          factorv(e) = detail::dev_cutoff_value(d, rc, width, is_bump);
          Kokkos::atomic_inc(&count(re_i(e)));
        }
      });

  // ---- M = max neighbors over atoms ----
  int M = 0;
  Kokkos::parallel_reduce(
      "pet_maxM", RangePolicy(0, N),
      KOKKOS_LAMBDA(int a, int& m) { m = (count(a) > m) ? count(a) : m; }, Kokkos::Max<int>(M));
  if (M < 1) M = 1;  // guard (matches host build)
  // Grow-only M (like GRACE's maxneigh): never shrink the NEF buffers, so steady-
  // state MD reuses fixed memory even as per-step neighbor counts fluctuate. The
  // extra slots are padding (mask=0), so results are unchanged.
  //
  // Opt-in: m_high == 0 disables it, an MD caller seeds it with 1. For a
  // structure-search workload the grow-only rule is the wrong trade: M is the max over
  // every structure the evaluator has EVER seen -- across chunks, space groups, and
  // the whole run -- so one clashy candidate permanently inflates the padded work
  // for every later batch, and M never drops as crowded structures converge out.
  // It also makes the GEMM shapes depend on process history rather than on the
  // input. The workspace pool already reuses capacity on shrink, which is the
  // reallocation storm m_high was introduced to prevent.
  if (m_high > 0) {
    if (M < m_high) M = m_high;
    else m_high = M;
  }
  const int S = M + 1;
  const int NM = N * M;
  dev.max_neighbors = M;

  // ---- NEF arrays from the persistent pool (zeroed on reuse) ----
  dev.neigh_species = ws.i1("nef:out:neigh_species", NM);
  dev.reverse_index = ws.i1("nef:out:reverse", NM);
  dev.edge_vec = ws.r2("nef:out:edge_vec", NM, 3);
  dev.dist = ws.r1("nef:out:dist", NM);
  dev.mask = ws.r1("nef:out:mask", NM);
  dev.pair_cutoff = ws.r1("nef:out:pcut", NM);
  dev.cutoff_factor = ws.n1("nef:out:cutoff", NM);
  dev.cf_seq = ws.n2("nef:out:cf_seq", N, S);
  Kokkos::deep_copy(ExecSpace(), dev.reverse_index, -1);  // padding + unmatched edges stay -1

  // ---- scatter kept edges into NEF slots (atomic per-atom slot counter) ----
  IView1D slot = ws.i1("nef:slot", N);  // running slot counter
  Kokkos::deep_copy(ExecSpace(), slot, 0);  // accumulator, zeroed explicitly
  IView1D flat_of_edge = ws.i1("nef:flat_edge", E);  // flat NEF index per kept edge (-1 if dropped)
  Kokkos::deep_copy(ExecSpace(), flat_of_edge, -1);
  // reuse the persistent reverse-matching map: grow capacity if needed, then clear
  const uint32_t need = static_cast<uint32_t>(E > 0 ? 2 * E + 16 : 16);
  if (edge_map.capacity() < need) edge_map.rehash(need);
  edge_map.clear();

  {
    auto neigh_species = dev.neigh_species;
    auto edge_vec = dev.edge_vec;
    auto dist = dev.dist;
    auto mask = dev.mask;
    auto pair_cutoff = dev.pair_cutoff;
    auto cutoff_factor = dev.cutoff_factor;
    // One thread per ATOM, walking its own edges in index order, so slot s is a
    // function of the edge list alone. This was an atomic_fetch_add over all edges,
    // which handed out slots in thread-arrival order -- so an atom's neighbours
    // landed in a different order every run, and since the network is fp32 every
    // reduction over the neighbour axis (attention softmax, message sums) rounded
    // differently. That was the single largest source of PET's run-to-run
    // nondeterminism, ~1000x the residual from the force scatter.
    Kokkos::parallel_for(
        "pet_scatter", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
          int s = 0;
          for (int e = roff(i); e < roff(i + 1); ++e) {
            if (!keepv(e)) continue;
            const int j = re_j(e);
            const int flat = i * M + s;
            ++s;
            flat_of_edge(e) = flat;
            edge_vec(flat, 0) = re_vec(e, 0);
            edge_vec(flat, 1) = re_vec(e, 1);
            edge_vec(flat, 2) = re_vec(e, 2);
            dist(flat) = re_dist(e);
            neigh_species(flat) = species(j);
            cutoff_factor(flat) = static_cast<Net>(factorv(e));
            pair_cutoff(flat) = rcv(e);
            mask(flat) = 1.0;
            edge_map.insert(
                detail::pet_pack_key(i, j, re_shift(e, 0), re_shift(e, 1), re_shift(e, 2)), flat);
          }
        });
  }

  // ---- reverse-edge index: match (i,j,shift) with (j,i,-shift) ----
  {
    auto reverse_index = dev.reverse_index;
    Kokkos::parallel_for(
        "pet_reverse", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
          if (!keepv(e)) return;
          const uint64_t rkey = detail::pet_pack_key(re_j(e), re_i(e), -re_shift(e, 0),
                                                     -re_shift(e, 1), -re_shift(e, 2));
          const uint32_t idx = edge_map.find(rkey);
          reverse_index(flat_of_edge(e)) = edge_map.valid_at(idx) ? edge_map.value_at(idx) : -1;
        });
  }

  // ---- attention-bias source cf_seq[n, 0]=1, cf_seq[n,1+m]=cutoff_factor(n*M+m) ----
  {
    auto cf_seq = dev.cf_seq;
    auto cutoff_factor = dev.cutoff_factor;
    Kokkos::parallel_for(
        "pet_cfseq", RangePolicy(0, N),
        KOKKOS_LAMBDA(int n) { cf_seq(n, 0) = static_cast<Net>(1.0); });
    Kokkos::parallel_for(
        "pet_cfseq_e", RangePolicy(0, N * M), KOKKOS_LAMBDA(int _i) {
          const int n = _i / M, m = _i % M;
          cf_seq(n, 1 + m) = cutoff_factor(n * M + m);
        });
  }

  return dev;
}

// Convenience overload for the standalone System path: enumerate the full
// periodic neighbor list on the host (detail::build_raw_edges), upload the raw
// COO edges, and build the DeviceEdgeData on-device. Used by the tests to
// validate the device builder against the host builder + goldens.
inline DeviceEdgeData build_device_edge_data(const System& sys, const Hypers& h,
                                             const std::vector<int>& species_to_index) {
  const int N = sys.n_atoms;
  if (N > detail::PET_KEY_ATOM_MAX)
    throw std::runtime_error("build_device_edge_data: atom count exceeds packed-key range");

  IView1D species("species", N);
  auto h_sp = Kokkos::create_mirror_view(species);
  for (int i = 0; i < N; ++i) {
    const int Z = sys.atomic_numbers[i];
    const int s = (Z >= 0 && Z < (int) species_to_index.size()) ? species_to_index[Z] : -1;
    if (s < 0) throw std::runtime_error("unsupported atomic number in system");
    h_sp(i) = s;
  }
  Kokkos::deep_copy(species, h_sp);

  auto raw = detail::build_raw_edges(sys, h.cutoff);
  const int E = static_cast<int>(raw.size());
  IView1D re_i("re_i", E), re_j("re_j", E);
  IView2D re_shift("re_shift", E, 3);
  RView2D re_vec("re_vec", E, 3);
  RView1D re_dist("re_dist", E);
  auto h_i = Kokkos::create_mirror_view(re_i);
  auto h_j = Kokkos::create_mirror_view(re_j);
  auto h_sh = Kokkos::create_mirror_view(re_shift);
  auto h_v = Kokkos::create_mirror_view(re_vec);
  auto h_d = Kokkos::create_mirror_view(re_dist);
  const int sbias = detail::PET_KEY_SHIFT_BIAS;
  for (int e = 0; e < E; ++e) {
    const auto& r = raw[e];
    if (std::abs(r.sa) > sbias || std::abs(r.sb) > sbias || std::abs(r.sc) > sbias)
      throw std::runtime_error("build_device_edge_data: cell shift exceeds packed-key range");
    h_i(e) = r.i;
    h_j(e) = r.j;
    h_sh(e, 0) = r.sa;
    h_sh(e, 1) = r.sb;
    h_sh(e, 2) = r.sc;
    h_v(e, 0) = r.vx;
    h_v(e, 1) = r.vy;
    h_v(e, 2) = r.vz;
    h_d(e) = r.dist;
  }
  Kokkos::deep_copy(re_i, h_i);
  Kokkos::deep_copy(re_j, h_j);
  Kokkos::deep_copy(re_shift, h_sh);
  Kokkos::deep_copy(re_vec, h_v);
  Kokkos::deep_copy(re_dist, h_d);

  // local scratch (this overload is the non-hot test/standalone path)
  Workspace ws;
  EdgeMap edge_map(E > 0 ? 2 * E + 16 : 16);
  int m_high = 0;
  std::vector<double> probes_h;
  const double min_cutoff = 0.5, spacing = h.cutoff_width_adaptive / 4.0;
  for (double p = min_cutoff; p < h.cutoff - 1e-12; p += spacing) probes_h.push_back(p);
  const int P = static_cast<int>(probes_h.size());
  RView1D probes("probes", P > 0 ? P : 1);
  auto h_probes = Kokkos::create_mirror_view(probes);
  for (int p = 0; p < P; ++p) h_probes(p) = probes_h[p];
  Kokkos::deep_copy(probes, h_probes);

  return build_device_edge_data(ws, edge_map, m_high, probes, P, N, species, re_i, re_j, re_shift,
                                re_vec, re_dist, E, h);
}

}  // namespace pet
