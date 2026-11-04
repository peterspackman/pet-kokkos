// DFT-D3 on the device: see d3.hpp.
//
// The grid. Each structure is binned into cells about kBinFraction of the
// longest cutoff wide: along a periodic axis the cell's own vectors (a bin index
// past the box folds back with a lattice shift, as in the PET search,
// device_cell_list.hpp); a structure periodic along no axis (a molecule, an
// engine's atoms and ghosts) gets a box around its atoms. The binning is the
// PET search's (sort_into_bins), at a width chosen for D3's radii: the PET
// grid's bins are a PET cutoff wide, and walking them out to the CN cutoff
// visits ~25 atoms for every one within it (see README "D3 dispersion").
//
// Each pass has one thread per atom, in bin order, so a warp's atoms walk the
// same bins and read the same neighbours. Within a thread the neighbours come
// in a fixed order: bins in turn, each bin's atoms by index. No atomics.
#include "pet/d3.hpp"

#include "pet/calculator.hpp"
#include "pet/device_cell_list.hpp"

#include "d3_data.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace pet {

namespace {

constexpr double kBohr = 0.529177210903;      // Angstrom, CODATA 2018
constexpr double kHartree = 27.211386245988;  // eV, CODATA 2018
constexpr double kKcn = 16.0;                 // the counting function's steepness
constexpr double kK3 = 4.0;                   // the reference weights' Gaussian width
constexpr int kZ = d3data::kMaxZ;
constexpr int kR = d3data::kMaxRef;
// Bin width over the longest radius walked; PET_D3_BIN=width (Angstrom) instead.
constexpr double kBinFraction = 0.5;
// Room a kept molecule box leaves around its atoms (Angstrom): an engine's atoms
// move less than its skin between rebuilds.
constexpr double kBoxMargin = 2.0;

using DV1 = Kokkos::View<double*, LR, MemSpace>;
using DV2 = Kokkos::View<double**, LR, MemSpace>;

std::string lower(std::string s) {
  for (auto& c : s) c = (char) std::tolower((unsigned char) c);
  return s;
}
std::string squash(const std::string& s) {
  std::string o;
  for (char c : s)
    if (c != '-' && c != '_' && c != ' ') o += c;
  return o;
}

const d3data::Functional* find_functional(const std::string& name) {
  std::string n = lower(name);
  for (int k = 0; k < d3data::n_aliases; ++k)
    if (n == d3data::aliases[k].name) n = d3data::aliases[k].target;
  for (int k = 0; k < d3data::n_functionals; ++k)
    if (n == d3data::functionals[k].name) return &d3data::functionals[k];
  const std::string q = squash(n);
  for (int k = 0; k < d3data::n_aliases; ++k)
    if (q == squash(d3data::aliases[k].name)) return find_functional(d3data::aliases[k].target);
  for (int k = 0; k < d3data::n_functionals; ++k)
    if (q == squash(d3data::functionals[k].name)) return &d3data::functionals[k];
  return nullptr;
}

// The smooth step: 1 below rc - w, 0 past rc, 1 - (35x^4 - 84x^5 + 70x^6 - 20x^7)
// between (cutoff.hpp's smooth cutoff, here with its derivative). w = 0: hard.
KOKKOS_INLINE_FUNCTION void taper(double r, double rc, double w, double& t, double& dt) {
  dt = 0.0;
  if (r > rc) {
    t = 0.0;
    return;
  }
  const double lo = rc - w;
  if (w <= 0.0 || r <= lo) {
    t = 1.0;
    return;
  }
  const double x = (r - lo) / w, x2 = x * x, x3 = x2 * x;
  t = 1.0 - x3 * x * (35.0 - 84.0 * x + 70.0 * x2 - 20.0 * x3);
  dt = -140.0 * x3 * (1.0 - 3.0 * x + 3.0 * x2 - x3) / w;  // -140 x^3 (1 - x)^3 / w
}

// The CN's counting function 1 / (1 + exp(-k (rco / r - 1))) and its r-derivative.
KOKKOS_INLINE_FUNCTION void count(double r, double rco, double& f, double& df) {
  const double ex = Kokkos::exp(-kKcn * (rco / r - 1.0));
  f = 1.0 / (1.0 + ex);
  df = -kKcn * rco / (r * r) * ex * f * f;
}

}  // namespace

// ---- parameters -------------------------------------------------------------

D3Damping d3_damping(const std::string& name) {
  const std::string n = lower(name);
  if (n == "bj" || n == "rational" || n == "d3bj" || n == "d3(bj)") return D3Damping::BJ;
  if (n == "zero" || n == "d3zero" || n == "d3(0)" || n == "0") return D3Damping::Zero;
  throw std::invalid_argument("pet: unknown D3 damping '" + name + "' (bj or zero)");
}

D3Params d3_parameters(const std::string& functional, D3Damping damping, bool atm) {
  const d3data::Functional* f = find_functional(functional);
  const char* dn = damping == D3Damping::BJ ? "rational (BJ)" : "zero";
  if (!f || (damping == D3Damping::BJ ? !f->has_bj : !f->has_zero))
    throw std::invalid_argument("pet: no D3 " + std::string(dn) + "-damping parameters for '" + functional +
                                "' (pet::d3_functionals lists them, or give the parameters)");
  D3Params p;
  p.damping = damping;
  if (damping == D3Damping::BJ) {
    p.s6 = f->bj_s6, p.s8 = f->bj_s8, p.a1 = f->bj_a1, p.a2 = f->bj_a2 * kBohr;
  } else {
    p.s6 = f->zero_s6, p.s8 = f->zero_s8, p.rs6 = f->zero_rs6, p.rs8 = f->zero_rs8;
  }
  p.alpha = 14.0;
  p.s9 = atm ? 1.0 : 0.0;
  return p;
}

std::vector<std::string> d3_functionals(D3Damping damping) {
  std::vector<std::string> v;
  for (int k = 0; k < d3data::n_functionals; ++k) {
    const auto& f = d3data::functionals[k];
    if (damping == D3Damping::BJ ? f.has_bj : f.has_zero) v.push_back(f.name);
  }
  return v;
}

D3Cutoffs D3Cutoffs::reference() {
  D3Cutoffs c;
  c.pair = 95.0 * kBohr, c.cn = 40.0 * kBohr, c.atm = 40.0 * kBohr;
  c.pair_taper = c.cn_taper = c.atm_taper = 0.0;
  return c;
}

D3Settings D3Settings::functional(const std::string& name, D3Damping damping, bool atm) {
  D3Settings s;
  s.params = d3_parameters(name, damping, atm);
  return s;
}

// ---- the kernels --------------------------------------------------------------

namespace {

struct TagCN {};
struct TagWeights {};
template <bool F>
struct TagPair {};
template <bool F>
struct TagATM {};
struct TagChain {};

template <class Tag>
using Range = Kokkos::RangePolicy<ExecSpace, Tag, Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;

struct Kernels {
  // tables, pet units
  IView1D nref, ztab;
  DV1 cnref, rcov, r4r2, r0, c6;
  // atoms
  IView1D type, sid, sper;
  // grid
  DV2 frame, width;  // [B, 12] (lattice or box rows, then origin), [B, 3]
  IView2D nc;        // [B, 3]
  IView1D goff, abin, bstart, batom;
  DV2 posw;          // [N, 3] wrapped along periodic axes
  // state and outputs
  DV1 cn, dedcn, eat;
  DV2 gw, dgw, fat, wat;
  int n_local = -1;
  // parameters
  bool bj = true;
  double s6 = 1, s8 = 0, a1 = 0, a2 = 0, rs6 = 1, rs8 = 1, alp = 14, s9 = 0;
  double rp = 0, wp = 0, rc = 0, wc = 0, ra = 0, wa = 0;

  KOKKOS_INLINE_FUNCTION double owned(int i) const { return (n_local < 0 || i < n_local) ? 1.0 : 0.0; }
  KOKKOS_INLINE_FUNCTION int z(int i) const { return ztab(type(i)); }

  // f(j, v, d2, q, k) for every atom image within R of atom i (itself excluded):
  // v = r_j + shift - r_i, q the bin's place in this walk, k the atom's slot in
  // batom. With (q0, k0), only bins from q0 on, and in bin q0 slots past k0.
  template <class F>
  KOKKOS_INLINE_FUNCTION void visit(int i, double R, F&& f, int q0 = -1, int k0 = -1) const {
    const int b = sid(i), per = sper(b);
    const int n0 = nc(b, 0), n1 = nc(b, 1), n2 = nc(b, 2);
    const int loc = abin(i) - goff(b);
    const int c[3] = {loc / (n1 * n2), (loc / n2) % n1, loc % n2};
    const int n[3] = {n0, n1, n2};
    int r[3];
    for (int d = 0; d < 3; ++d) {
      r[d] = (int) Kokkos::ceil(R / width(b, d));
      if (!(per >> d & 1)) r[d] = Kokkos::min(r[d], n[d] - 1);
    }
    const double R2 = R * R;
    const double xi = posw(i, 0), yi = posw(i, 1), zi = posw(i, 2);
    int q = -1;
    for (int da = -r[0]; da <= r[0]; ++da) {
      const int ta = c[0] + da;
      if (!(per & 1) && (ta < 0 || ta >= n0)) {
        q += (2 * r[1] + 1) * (2 * r[2] + 1);
        continue;
      }
      const int sa = (per & 1) ? floor_div(ta, n0) : 0, ma = ta - sa * n0;
      for (int db = -r[1]; db <= r[1]; ++db) {
        const int tb = c[1] + db;
        if (!(per & 2) && (tb < 0 || tb >= n1)) {
          q += 2 * r[2] + 1;
          continue;
        }
        const int sb = (per & 2) ? floor_div(tb, n1) : 0, mb = tb - sb * n1;
        for (int dc = -r[2]; dc <= r[2]; ++dc) {
          ++q;
          const int tc = c[2] + dc;
          if (!(per & 4) && (tc < 0 || tc >= n2)) continue;
          if (q < q0) continue;
          const int sc = (per & 4) ? floor_div(tc, n2) : 0, mc = tc - sc * n2;
          const double ox = sa * frame(b, 0) + sb * frame(b, 3) + sc * frame(b, 6) - xi;
          const double oy = sa * frame(b, 1) + sb * frame(b, 4) + sc * frame(b, 7) - yi;
          const double oz = sa * frame(b, 2) + sb * frame(b, 5) + sc * frame(b, 8) - zi;
          const int bin = goff(b) + (ma * n1 + mb) * n2 + mc;
          const int k1 = bstart(bin + 1);
          for (int k = (q == q0 ? k0 + 1 : bstart(bin)); k < k1; ++k) {
            const int j = batom(k);
            const double v[3] = {posw(j, 0) + ox, posw(j, 1) + oy, posw(j, 2) + oz};
            const double d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (d2 < 1e-20 || d2 > R2) continue;
            f(j, v, d2, q, k);
          }
        }
      }
    }
  }

  // C6 between i (weights wi, dwi over its ni references) and j, and dC6/dCN_i.
  KOKKOS_INLINE_FUNCTION void c6_pair(int zi, int ni, const double* wi, const double* dwi, int j, int zj, double& C,
                                      double& dC) const {
    const int nj = nref(zj);
    const int base = (zi * kZ + zj) * (kR * kR);
    C = 0.0, dC = 0.0;
    for (int a = 0; a < ni; ++a) {
      double u = 0.0;
      for (int bb = 0; bb < nj; ++bb) u += c6(base + a * kR + bb) * gw(j, bb);
      C += wi[a] * u;
      dC += dwi[a] * u;
    }
  }
  KOKKOS_INLINE_FUNCTION double c6_plain(int i, int zi, int j, int zj) const {
    const int ni = nref(zi), nj = nref(zj);
    const int base = (zi * kZ + zj) * (kR * kR);
    double C = 0.0;
    for (int a = 0; a < ni; ++a) {
      double u = 0.0;
      for (int bb = 0; bb < nj; ++bb) u += c6(base + a * kR + bb) * gw(j, bb);
      C += gw(i, a) * u;
    }
    return C;
  }

  KOKKOS_INLINE_FUNCTION void operator()(TagCN, int t) const {
    const int i = batom(t), zi = z(i);
    const double rci = rcov(zi);
    double s = 0.0;
    visit(i, rc, [&](int j, const double*, double d2, int, int) {
      const double r = Kokkos::sqrt(d2);
      double f, df, tp, dtp;
      count(r, rci + rcov(z(j)), f, df);
      taper(r, rc, wc, tp, dtp);
      s += f * tp;
    });
    cn(i) = s;
  }

  // s-dftd3's weights: Gaussians in CN - CN_ref, normalised; where the sum
  // underflows, all weight on the reference with the largest CN.
  KOKKOS_INLINE_FUNCTION void operator()(TagWeights, int i) const {
    const int zi = z(i), ni = nref(zi);
    const double x = cn(i);
    double norm = 0.0, dnorm = 0.0;
    for (int a = 0; a < ni; ++a) {
      const double d = x - cnref(zi * kR + a), e = Kokkos::exp(-kK3 * d * d);
      norm += e;
      dnorm += -2.0 * kK3 * d * e;
    }
    double maxref = -1.0;
    for (int a = 0; a < ni; ++a) maxref = Kokkos::fmax(maxref, cnref(zi * kR + a));
    const double inv = 1.0 / norm;
    for (int a = 0; a < kR; ++a) {
      double g = 0.0, dg = 0.0;
      if (a < ni) {
        const double d = x - cnref(zi * kR + a), e = Kokkos::exp(-kK3 * d * d), de = -2.0 * kK3 * d * e;
        g = e * inv;
        dg = inv * (de - e * dnorm * inv);
        if (!(norm > 0.0) || !Kokkos::isfinite(g)) g = cnref(zi * kR + a) == maxref ? 1.0 : 0.0;
        if (!Kokkos::isfinite(dg)) dg = 0.0;
      }
      gw(i, a) = g;
      dgw(i, a) = dg;
    }
  }

  // The two-body term per unit C6, untapered, and its r-derivative.
  KOKKOS_INLINE_FUNCTION void damped(double r, double d2, int zi, int zj, double& e, double& de) const {
    const double Q = 3.0 * r4r2(zi) * r4r2(zj);
    if (bj) {
      const double R0 = a1 * Kokkos::sqrt(Q) + a2;
      const double R2 = R0 * R0, R6 = R2 * R2 * R2, R8 = R6 * R2;
      const double r6 = d2 * d2 * d2, r8 = r6 * d2;
      const double t6 = 1.0 / (r6 + R6), t8 = 1.0 / (r8 + R8);
      e = -(s6 * t6 + s8 * Q * t8);
      de = 6.0 * s6 * (r6 / r) * t6 * t6 + 8.0 * s8 * Q * (r8 / r) * t8 * t8;
    } else {
      const double R0 = r0(zi * kZ + zj);
      const double q6 = Kokkos::pow(r / (rs6 * R0), -alp), q8 = Kokkos::pow(r / (rs8 * R0), -(alp + 2.0));
      const double f6 = 1.0 / (1.0 + 6.0 * q6), f8 = 1.0 / (1.0 + 6.0 * q8);
      const double df6 = 6.0 * alp * q6 / r * f6 * f6, df8 = 6.0 * (alp + 2.0) * q8 / r * f8 * f8;
      const double r6 = d2 * d2 * d2, r8 = r6 * d2;
      e = -(s6 * f6 / r6 + s8 * Q * f8 / r8);
      de = -(s6 * (df6 - 6.0 * f6 / r) / r6 + s8 * Q * (df8 - 8.0 * f8 / r) / r8);
    }
  }

  template <bool F>
  KOKKOS_INLINE_FUNCTION void operator()(TagPair<F>, int t) const {
    const int i = batom(t), zi = z(i), ni = nref(zi);
    const double oi = owned(i);
    double wi[kR], dwi[kR];
    for (int a = 0; a < kR; ++a) wi[a] = gw(i, a), dwi[a] = dgw(i, a);
    double e = 0.0, de = 0.0, f[3] = {0, 0, 0}, w[6] = {0, 0, 0, 0, 0, 0};
    visit(i, rp, [&](int j, const double* v, double d2, int, int) {
      const double wt = 0.5 * (oi + owned(j));
      if (wt == 0.0) return;
      const int zj = z(j);
      const double r = Kokkos::sqrt(d2);
      double C, dC, u, du, tp, dtp;
      c6_pair(zi, ni, wi, dwi, j, zj, C, dC);
      damped(r, d2, zi, zj, u, du);
      taper(r, rp, wp, tp, dtp);
      e += 0.5 * oi * C * u * tp;
      if (F) {
        const double g = wt * C * (du * tp + u * dtp) / r;  // dE/dv = g v
        for (int x = 0; x < 3; ++x) f[x] += g * v[x];
        const double h = 0.5 * g;
        w[0] += h * v[0] * v[0], w[1] += h * v[1] * v[1], w[2] += h * v[2] * v[2];
        w[3] += h * v[0] * v[1], w[4] += h * v[0] * v[2], w[5] += h * v[1] * v[2];
        de += wt * dC * u * tp;
      }
    });
    eat(i) = e;
    if (F) {
      for (int x = 0; x < 3; ++x) fat(i, x) = f[x];
      for (int x = 0; x < 6; ++x) wat(i, x) = w[x];
      dedcn(i) = de;
    }
  }

  // Axilrod-Teller-Muto, zero-damped as s-dftd3 has it: for each triangle i, j,
  // k with every side within the cutoff,
  //   E = C9 (3 cos a cos b cos c + 1) / (r_ij r_ik r_jk)^3 f(r_ij r_ik r_jk),
  //   C9 = s9 sqrt(C6_ij C6_ik C6_jk / Eh), f(P) = 1 / (1 + 6 (R0 / P)^((alpha + 2) / 3)),
  // R0 the product of (4/3) R0_ab. Atom i sums every triangle it is a corner of,
  // each pair {j, k} once (k after j in its walk); a third of each goes to it.
  template <bool F>
  KOKKOS_INLINE_FUNCTION void operator()(TagATM<F>, int t) const {
    const int i = batom(t), zi = z(i), ni = nref(zi);
    const double oi = owned(i);
    double wi[kR], dwi[kR];
    for (int a = 0; a < kR; ++a) wi[a] = gw(i, a), dwi[a] = dgw(i, a);
    const double ex = (alp + 2.0) / 3.0, ra2 = ra * ra;
    double e = 0.0, de = 0.0, f[3] = {0, 0, 0}, w[6] = {0, 0, 0, 0, 0, 0};
    visit(i, ra, [&](int j, const double* vij, double a, int qj, int kj) {
      const int zj = z(j);
      const double oj = owned(j), rij = Kokkos::sqrt(a);
      double Cij, dCij;
      c6_pair(zi, ni, wi, dwi, j, zj, Cij, dCij);
      double tij, dtij;
      taper(rij, ra, wa, tij, dtij);
      const double r0ij = (4.0 / 3.0) * r0(zi * kZ + zj);
      visit(
          i, ra,
          [&](int k, const double* vik, double b, int, int) {
            const double vjk[3] = {vik[0] - vij[0], vik[1] - vij[1], vik[2] - vij[2]};
            const double c = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
            if (c > ra2 || c < 1e-20) return;
            const double wt = (oi + oj + owned(k)) / 3.0;
            if (wt == 0.0) return;
            const int zk = z(k);
            const double rik = Kokkos::sqrt(b), rjk = Kokkos::sqrt(c);
            double Cik, dCik;
            c6_pair(zi, ni, wi, dwi, k, zk, Cik, dCik);
            const double Cjk = c6_plain(j, zj, k, zk);
            const double C9 = s9 * Kokkos::sqrt(Kokkos::fabs(Cij * Cik * Cjk) / kHartree);
            const double R0 = r0ij * (4.0 / 3.0) * r0(zi * kZ + zk) * (4.0 / 3.0) * r0(zj * kZ + zk);
            const double P = rij * rik * rjk, P2 = P * P, P3 = P2 * P, P5 = P3 * P2;
            const double u = Kokkos::pow(R0 / P, ex), fd = 1.0 / (1.0 + 6.0 * u);
            const double X = a - b + c, Y = a + b - c, Zt = -a + b + c, s = X * Y * Zt;
            const double ang = 0.375 * s / P5 + 1.0 / P3;
            double tik, dtik, tjk, dtjk;
            taper(rik, ra, wa, tik, dtik);
            taper(rjk, ra, wa, tjk, dtjk);
            const double T = tij * tik * tjk;
            const double E = C9 * ang * fd * T;
            e += oi * E / 3.0;
            if (!F) return;
            // d/d(squared side) of ang; dfd/dP; then each side's dE/dr.
            const double s_a = Y * Zt + X * Zt - X * Y, s_b = -Y * Zt + X * Zt + X * Y, s_c = Y * Zt - X * Zt + X * Y;
            const double dang_a = 0.375 * (s_a - 2.5 * s / a) / P5 - 1.5 / (P3 * a);
            const double dang_b = 0.375 * (s_b - 2.5 * s / b) / P5 - 1.5 / (P3 * b);
            const double dang_c = 0.375 * (s_c - 2.5 * s / c) / P5 - 1.5 / (P3 * c);
            const double dfd_dlnP = 6.0 * ex * u * fd * fd;  // P dfd/dP
            // Each side's dE/dr: through the angle factor, the damping and the tapers.
            const double gij = C9 * (2.0 * rij * dang_a * fd * T + ang * dfd_dlnP / rij * T + ang * fd * dtij * tik * tjk);
            const double gik = C9 * (2.0 * rik * dang_b * fd * T + ang * dfd_dlnP / rik * T + ang * fd * tij * dtik * tjk);
            const double gjk = C9 * (2.0 * rjk * dang_c * fd * T + ang * dfd_dlnP / rjk * T + ang * fd * tij * tik * dtjk);
            const double hij = wt * gij / rij, hik = wt * gik / rik, hjk = wt * gjk / rjk;  // dE/dv = h v
            for (int x = 0; x < 3; ++x) f[x] += hij * vij[x] + hik * vik[x];
            const double third = 1.0 / 3.0;
            auto add = [&](const double* v, double h) {
              h *= third;
              w[0] += h * v[0] * v[0], w[1] += h * v[1] * v[1], w[2] += h * v[2] * v[2];
              w[3] += h * v[0] * v[1], w[4] += h * v[0] * v[2], w[5] += h * v[1] * v[2];
            };
            add(vij, hij), add(vik, hik), add(vjk, hjk);
            // dE/dCN_i through C6_ij and C6_ik: C9 ~ sqrt(C6_ij C6_ik C6_jk).
            de += wt * 0.5 * E * (dCij / Cij + dCik / Cik);
          },
          qj, kj);
    });
    eat(i) += e;
    if (F) {
      for (int x = 0; x < 3; ++x) fat(i, x) += f[x];
      for (int x = 0; x < 6; ++x) wat(i, x) += w[x];
      dedcn(i) += de;
    }
  }

  // The CN's chain rule: each pair i-j moves both CNs, so the force on i is
  // sum_j (dE/dCN_i + dE/dCN_j) dcn_ij/dv.
  KOKKOS_INLINE_FUNCTION void operator()(TagChain, int t) const {
    const int i = batom(t), zi = z(i);
    const double di = dedcn(i), rci = rcov(zi);
    double f[3] = {0, 0, 0}, w[6] = {0, 0, 0, 0, 0, 0};
    visit(i, rc, [&](int j, const double* v, double d2, int, int) {
      const double s = di + dedcn(j);
      if (s == 0.0) return;
      const double r = Kokkos::sqrt(d2);
      double c, dc, tp, dtp;
      count(r, rci + rcov(z(j)), c, dc);
      taper(r, rc, wc, tp, dtp);
      const double g = s * (dc * tp + c * dtp) / r;
      for (int x = 0; x < 3; ++x) f[x] += g * v[x];
      const double h = 0.5 * g;
      w[0] += h * v[0] * v[0], w[1] += h * v[1] * v[1], w[2] += h * v[2] * v[2];
      w[3] += h * v[0] * v[1], w[4] += h * v[0] * v[2], w[5] += h * v[1] * v[2];
    });
    for (int x = 0; x < 3; ++x) fat(i, x) += f[x];
    for (int x = 0; x < 6; ++x) wat(i, x) += w[x];
  }
};

}  // namespace

// ---- the D3 object ----------------------------------------------------------------

struct D3::Impl {
  D3Settings s;
  Workspace ws;
  // tables
  IView1D nref, ztab;
  DV1 cnref, rcov, r4r2, r0, c6;
  std::vector<int> ztab_host;
  // the grid as last sized
  bool sized = false;
  int sized_N = -1, sized_B = -1, nbins = 0;
  IView1D overflow{"d3:overflow", 1};
  D3::Out out;

  explicit Impl(D3Settings st) : s(st) {
    ws.set_zero(false);
    const D3Cutoffs& c = s.cutoffs;
    if (!(c.pair > 0 && c.cn > 0 && c.atm > 0) || c.pair_taper < 0 || c.cn_taper < 0 || c.atm_taper < 0 ||
        c.pair_taper > c.pair || c.cn_taper > c.cn || c.atm_taper > c.atm)
      throw std::invalid_argument("pet: D3 cutoffs must be positive, with tapers no wider than their cutoffs");
    auto hnref = Kokkos::create_mirror_view(nref = IView1D("d3:nref", kZ));
    auto hcn = Kokkos::create_mirror_view(cnref = DV1("d3:cnref", kZ * kR));
    auto hrcov = Kokkos::create_mirror_view(rcov = DV1("d3:rcov", kZ));
    auto hr4 = Kokkos::create_mirror_view(r4r2 = DV1("d3:r4r2", kZ));
    auto hr0 = Kokkos::create_mirror_view(r0 = DV1("d3:r0", kZ * kZ));
    auto hc6 = Kokkos::create_mirror_view(c6 = DV1("d3:c6", std::size_t(kZ) * kZ * kR * kR));
    Kokkos::deep_copy(hr0, 0.0), Kokkos::deep_copy(hc6, 0.0);
    const double c6_unit = kHartree * std::pow(kBohr, 6);
    for (int z = 0; z < kZ; ++z) {
      hnref(z) = d3data::nref[z];
      hrcov(z) = d3data::rcov_bohr[z] * kBohr;
      hr4(z) = d3data::r4r2_bohr[z] * kBohr;
      for (int a = 0; a < kR; ++a) hcn(z * kR + a) = d3data::cnref_e4[z * kR + a] / 1e4;
    }
    for (int i = 1; i < kZ; ++i)
      for (int j = 1; j <= i; ++j) {
        const int p = d3data::pair_index(i, j);
        hr0(i * kZ + j) = hr0(j * kZ + i) = d3data::r0_e4[p] / 1e4;
        const int* blk = &d3data::c6_e4[d3data::c6_off[p]];
        for (int a = 0; a < d3data::nref[i]; ++a)
          for (int b = 0; b < d3data::nref[j]; ++b) {
            const double v = blk[a * d3data::nref[j] + b] / 1e4 * c6_unit;
            hc6((std::size_t(i) * kZ + j) * kR * kR + a * kR + b) = v;
            hc6((std::size_t(j) * kZ + i) * kR * kR + b * kR + a) = v;
          }
      }
    Kokkos::deep_copy(nref, hnref), Kokkos::deep_copy(cnref, hcn), Kokkos::deep_copy(rcov, hrcov);
    Kokkos::deep_copy(r4r2, hr4), Kokkos::deep_copy(r0, hr0), Kokkos::deep_copy(c6, hc6);
    std::vector<int> id(kZ);
    for (int z = 0; z < kZ; ++z) id[z] = z;
    set_types(id);
    Kokkos::deep_copy(overflow, 0);
  }

  void set_types(const std::vector<int>& t) {
    for (int z : t)
      if (z < 0 || z >= kZ) throw std::invalid_argument("pet: D3 covers elements 1-103");
    ztab_host = t;
    ztab = IView1D("d3:ztab", std::max<std::size_t>(1, t.size()));
    Kokkos::deep_copy(Kokkos::subview(ztab, std::make_pair(std::size_t(0), t.size())),
                      Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(t.data(), t.size()));
  }

  double longest() const {
    const D3Cutoffs& c = s.cutoffs;
    return std::max({c.pair, c.cn, s.params.s9 != 0.0 ? c.atm : 0.0});
  }
  double bin_width() const {
    if (const char* e = std::getenv("PET_D3_BIN"))
      if (std::atof(e) > 0) return std::atof(e);
    return kBinFraction * std::max(s.cutoffs.cn, s.params.s9 != 0.0 ? s.cutoffs.atm : 0.0);
  }

  // The grid: each structure's frame and bins, every atom binned and sorted.
  struct Grid {
    DV2 frame, width, posw;
    IView2D nc;
    IView1D goff, abin, bstart, batom;
  };
  Grid grid(const D3::Atoms& at, bool keep) {
    const int N = at.N, B = at.B;
    Grid g;
    g.frame = ws.r2("d3:frame", B, 12);
    g.width = ws.r2("d3:width", B, 3);
    g.nc = ws.i2("d3:nc", B, 3);
    g.goff = ws.i1("d3:goff", B + 1);
    DV2 finv = ws.r2("d3:finv", B, 9);
    auto sper = at.sper, scnt = at.scnt, soff = at.soff;
    auto scell = at.scell;
    auto pos = at.pos;
    auto frame = g.frame, width = g.width;
    auto nc = g.nc;
    auto goff = g.goff;
    const bool resize = !keep || !sized || sized_N != N || sized_B != B;
    const double w = bin_width(), margin = keep ? kBoxMargin : 0.0;
    // Bins at most a few per atom, and no more than an int holds.
    const long cap = std::max<long>(64, 8L * N);
    if (resize) {
      // The box around each molecule's atoms.
      DV2 bb = ws.r2("d3:bbox", B, 6);
      Kokkos::parallel_for(
          "d3_bbox", TeamPolicy(B, 1, kLanes), KOKKOS_LAMBDA(const TeamPolicy::member_type& m) {
            const int b = m.league_rank(), a0 = soff(b), n = scnt(b);
            for (int d = 0; d < 3; ++d) {
              double lo = 0.0, hi = 0.0;
              if (n > 0 && sper(b) == 0) {
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(m, n), [&](int q, double& l) { l = Kokkos::fmin(l, pos(a0 + q, d)); },
                    Kokkos::Min<double>(lo));
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(m, n), [&](int q, double& h) { h = Kokkos::fmax(h, pos(a0 + q, d)); },
                    Kokkos::Max<double>(hi));
              }
              Kokkos::single(Kokkos::PerTeam(m), [&] { bb(b, d) = lo, bb(b, 3 + d) = hi; });
            }
          });
      IView1D gbins = ws.i1("d3:gbins", B);
      Kokkos::parallel_for(
          "d3_frames", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
            double A[9], o[3] = {0, 0, 0};
            if (sper(b)) {
              for (int k = 0; k < 9; ++k) A[k] = scell(b, k);
            } else {
              for (int k = 0; k < 9; ++k) A[k] = 0.0;
              for (int d = 0; d < 3; ++d) {
                o[d] = bb(b, d) - margin;
                A[4 * d] = Kokkos::fmax(bb(b, 3 + d) - bb(b, d) + 2.0 * margin, 1e-3);
              }
            }
            for (int k = 0; k < 9; ++k) frame(b, k) = A[k];
            for (int d = 0; d < 3; ++d) frame(b, 9 + d) = o[d];
            const double bc[3] = {A[4] * A[8] - A[5] * A[7], A[5] * A[6] - A[3] * A[8], A[3] * A[7] - A[4] * A[6]};
            const double ca[3] = {A[7] * A[2] - A[8] * A[1], A[8] * A[0] - A[6] * A[2], A[6] * A[1] - A[7] * A[0]};
            const double ab[3] = {A[1] * A[5] - A[2] * A[4], A[2] * A[3] - A[0] * A[5], A[0] * A[4] - A[1] * A[3]};
            const double vol = Kokkos::fabs(A[0] * bc[0] + A[1] * bc[1] + A[2] * bc[2]);
            const double* cr[3] = {bc, ca, ab};
            int n[3];
            double perp[3];
            for (int d = 0; d < 3; ++d) {
              const double area = Kokkos::sqrt(cr[d][0] * cr[d][0] + cr[d][1] * cr[d][1] + cr[d][2] * cr[d][2]);
              perp[d] = (vol > 1e-12 && area > 0.0) ? vol / area : 1e30;
              const bool binned = sper(b) == 0 || (sper(b) >> d & 1);
              n[d] = binned ? (int) Kokkos::fmin(Kokkos::fmax(Kokkos::floor(perp[d] / w), 1.0), 1e6) : 1;
            }
            const long cap_b = Kokkos::max(64L, 8L * scnt(b));
            while ((long) n[0] * n[1] * n[2] > Kokkos::min(cap_b, cap)) {
              const int d = (n[0] >= n[1] && n[0] >= n[2]) ? 0 : (n[1] >= n[2] ? 1 : 2);
              n[d] = Kokkos::max(1, (int) (n[d] * 0.8));
            }
            for (int d = 0; d < 3; ++d) nc(b, d) = n[d];
            gbins(b) = n[0] * n[1] * n[2];
          });
      Kokkos::parallel_scan(
          "d3_binscan", RangePolicy(0, B), KOKKOS_LAMBDA(int b, int& upd, bool final) {
            if (final) goff(b) = upd;
            upd += gbins(b);
            if (final && b == B - 1) goff(B) = upd;
          });
      Kokkos::deep_copy(nbins, Kokkos::subview(goff, B));
      sized = true, sized_N = N, sized_B = B;
    } else {
      // Kept: a periodic structure's bins follow its cell (the counts stay put).
      Kokkos::parallel_for(
          "d3_frames_keep", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
            if (sper(b))
              for (int k = 0; k < 9; ++k) frame(b, k) = scell(b, k);
          });
    }
    // Inverses and bin widths, from the frames as they now are.
    Kokkos::parallel_for(
        "d3_widths", RangePolicy(0, B), KOKKOS_LAMBDA(int b) {
          const double a0 = frame(b, 0), a1 = frame(b, 1), a2 = frame(b, 2), b0 = frame(b, 3), b1 = frame(b, 4),
                       b2 = frame(b, 5), c0 = frame(b, 6), c1 = frame(b, 7), c2 = frame(b, 8);
          const double det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
          const double id = det != 0.0 ? 1.0 / det : 0.0;
          finv(b, 0) = (b1 * c2 - b2 * c1) * id, finv(b, 1) = (a2 * c1 - a1 * c2) * id, finv(b, 2) = (a1 * b2 - a2 * b1) * id;
          finv(b, 3) = (b2 * c0 - b0 * c2) * id, finv(b, 4) = (a0 * c2 - a2 * c0) * id, finv(b, 5) = (a2 * b0 - a0 * b2) * id;
          finv(b, 6) = (b0 * c1 - b1 * c0) * id, finv(b, 7) = (a1 * c0 - a0 * c1) * id, finv(b, 8) = (a0 * b1 - a1 * b0) * id;
          const double vol = Kokkos::fabs(det);
          const double bc[3] = {b1 * c2 - b2 * c1, b2 * c0 - b0 * c2, b0 * c1 - b1 * c0};
          const double ca[3] = {c1 * a2 - c2 * a1, c2 * a0 - c0 * a2, c0 * a1 - c1 * a0};
          const double ab[3] = {a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0};
          const double* cr[3] = {bc, ca, ab};
          for (int d = 0; d < 3; ++d) {
            const double area = Kokkos::sqrt(cr[d][0] * cr[d][0] + cr[d][1] * cr[d][1] + cr[d][2] * cr[d][2]);
            const double perp = (vol > 1e-12 && area > 0.0) ? vol / area : 1e30;
            width(b, d) = perp / nc(b, d);
          }
        });
    // Bin every atom: fractional coordinates in the frame, wrapped along periodic
    // axes; an atom outside a kept box is clamped in and flagged.
    g.posw = ws.r2("d3:posw", N, 3);
    g.abin = ws.i1("d3:abin", N);
    auto posw = g.posw;
    auto abin = g.abin;
    auto sid = at.sid;
    auto over = overflow;
    Kokkos::parallel_for(
        "d3_assign", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
          const int b = sid(i), per = sper(b);
          double x[3], f[3];
          for (int d = 0; d < 3; ++d) x[d] = pos(i, d) - (per ? 0.0 : frame(b, 9 + d));
          for (int d = 0; d < 3; ++d) f[d] = x[0] * finv(b, d) + x[1] * finv(b, 3 + d) + x[2] * finv(b, 6 + d);
          int ijk[3];
          bool out_of_box = false;
          double p[3] = {pos(i, 0), pos(i, 1), pos(i, 2)};
          for (int d = 0; d < 3; ++d) {
            if (per >> d & 1) {  // wrapped into the cell
              const double n = Kokkos::floor(f[d]);
              f[d] -= n;
              for (int e = 0; e < 3; ++e) p[e] -= n * frame(b, 3 * d + e);
            } else if (!per && (f[d] < -1e-9 || f[d] > 1.0 + 1e-9)) {
              out_of_box = true;
            }
            const int c = (int) Kokkos::floor(f[d] * (double) nc(b, d));
            ijk[d] = Kokkos::min(Kokkos::max(c, 0), nc(b, d) - 1);
          }
          for (int e = 0; e < 3; ++e) posw(i, e) = p[e];
          if (out_of_box) over(0) = 1;
          abin(i) = goff(b) + (ijk[0] * nc(b, 1) + ijk[1]) * nc(b, 2) + ijk[2];
        });
    sort_into_bins(ws, "d3", g.abin, N, nbins, g.bstart, g.batom);
    return g;
  }

  D3::Out evaluate(const D3::Atoms& at, bool forces, bool keep) {
    const int N = at.N, B = at.B;
    D3::Out o;
    o.energy = ws.r1("d3:energy", B);
    o.per_atom = ws.r1("d3:eat", N);
    if (forces) {
      o.forces = ws.r2("d3:fat", N, 3);
      o.virial = ws.r2("d3:virial", B, 6);
      o.atom_virial = ws.r2("d3:wat", N, 6);
    }
    if (B == 0) return out = o;
    if (N == 0) {
      Kokkos::deep_copy(ExecSpace(), o.energy, 0.0);
      if (forces) Kokkos::deep_copy(ExecSpace(), o.virial, 0.0);
      return out = o;
    }
    if (at.n_local >= 0 && B != 1) throw std::invalid_argument("pet: D3 with ghosts takes one structure");
    if (!keep) Kokkos::deep_copy(ExecSpace(), overflow, 0);
    const Grid g = grid(at, keep);

    Kernels k;
    k.nref = nref, k.ztab = ztab, k.cnref = cnref, k.rcov = rcov, k.r4r2 = r4r2, k.r0 = r0, k.c6 = c6;
    k.type = at.type, k.sid = at.sid, k.sper = at.sper;
    k.frame = g.frame, k.width = g.width, k.nc = g.nc, k.goff = g.goff, k.abin = g.abin, k.bstart = g.bstart;
    k.batom = g.batom, k.posw = g.posw;
    k.cn = ws.r1("d3:cn", N), k.dedcn = ws.r1("d3:dedcn", N), k.eat = o.per_atom;
    k.gw = ws.r2("d3:gw", N, kR), k.dgw = ws.r2("d3:dgw", N, kR);
    k.fat = forces ? o.forces : DV2(), k.wat = forces ? o.atom_virial : DV2();
    k.n_local = at.n_local;
    const D3Params& p = s.params;
    const D3Cutoffs& c = s.cutoffs;
    k.bj = p.damping == D3Damping::BJ;
    k.s6 = p.s6, k.s8 = p.s8, k.a1 = p.a1, k.a2 = p.a2, k.rs6 = p.rs6, k.rs8 = p.rs8, k.alp = p.alpha, k.s9 = p.s9;
    k.rp = c.pair, k.wp = c.pair_taper, k.rc = c.cn, k.wc = c.cn_taper, k.ra = c.atm, k.wa = c.atm_taper;

    Kokkos::parallel_for("d3_cn", Range<TagCN>(0, N), k);
    if (at.exchange) forward(at.exchange, k.cn);
    Kokkos::parallel_for("d3_weights", Range<TagWeights>(0, N), k);
    const bool atm = p.s9 != 0.0;
    if (forces) {
      Kokkos::parallel_for("d3_pair", Range<TagPair<true>>(0, N), k);
      if (atm) Kokkos::parallel_for("d3_atm", Range<TagATM<true>>(0, N), k);
      if (at.exchange) reverse(at.exchange, k.dedcn);
      Kokkos::parallel_for("d3_chain", Range<TagChain>(0, N), k);
    } else {
      Kokkos::parallel_for("d3_pair_energy", Range<TagPair<false>>(0, N), k);
      if (atm) Kokkos::parallel_for("d3_atm_energy", Range<TagATM<false>>(0, N), k);
    }
    // Each structure's sums, a warp per (structure, column) in a fixed order.
    auto soff = at.soff, scnt = at.scnt;
    auto eat = o.per_atom, en = o.energy;
    auto wat = k.wat, vir = o.virial;
    const int T = forces ? 7 : 1;
    Kokkos::parallel_for(
        "d3_sums", TeamPolicy(B * T, 1, kLanes), KOKKOS_LAMBDA(const TeamPolicy::member_type& m) {
          const int b = m.league_rank() / T, col = m.league_rank() % T;
          const int a0 = soff(b), a1 = soff(b) + scnt(b);
          double sum = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(m, kLanes),
              [&](int l, double& part) {
                for (int a = a0 + l; a < a1; a += kLanes) part += col == 0 ? eat(a) : wat(a, col - 1);
              },
              sum);
          Kokkos::single(Kokkos::PerTeam(m), [&] {
            if (col == 0) en(b) = sum;
            else vir(b, col - 1) = sum;
          });
        });
    return out = o;
  }

  // The exchange moves Real views; D3 keeps double ones.
  template <class V>
  void forward(Exchange* ex, const V& a) {
    if constexpr (std::is_same_v<V, RView1D>) {
      ex->atoms_forward(a);
    } else {
      RView1D t = ws.r1("d3:xfer", a.extent(0));
      Kokkos::deep_copy(ExecSpace(), t, a);
      ex->atoms_forward(t);
      Kokkos::deep_copy(ExecSpace(), a, t);
    }
  }
  template <class V>
  void reverse(Exchange* ex, const V& a) {
    if constexpr (std::is_same_v<V, RView1D>) {
      ex->atoms_reverse(a);
    } else {
      RView1D t = ws.r1("d3:xfer", a.extent(0));
      Kokkos::deep_copy(ExecSpace(), t, a);
      ex->atoms_reverse(t);
      Kokkos::deep_copy(ExecSpace(), a, t);
    }
  }
};

D3::D3(D3Settings settings) {
  if (!Kokkos::is_initialized()) throw std::runtime_error("pet::D3: Kokkos must be initialized first");
  impl_ = std::make_unique<Impl>(settings);
}
D3::~D3() = default;
D3::D3(D3&&) noexcept = default;
D3& D3::operator=(D3&&) noexcept = default;

const D3Settings& D3::settings() const { return impl_->s; }
void D3::set_types(const std::vector<int>& t) { impl_->set_types(t); }
bool D3::overflowed() const {
  int o = 0;
  Kokkos::deep_copy(o, Kokkos::subview(impl_->overflow, 0));
  return o != 0;
}
IView1D D3::overflow_flag() const { return impl_->overflow; }
void D3::reset_grid() const {
  impl_->sized = false;
  Kokkos::deep_copy(impl_->overflow, 0);
}
std::size_t D3::generation() const { return impl_->ws.generation(); }

double D3::ghost_cutoff(bool exchange) const {
  const D3Cutoffs& c = impl_->s.cutoffs;
  const double reach = std::max(c.pair, impl_->s.params.s9 != 0.0 ? c.atm : 0.0);
  return exchange ? reach : reach + c.cn;
}

D3::Out D3::evaluate(const Atoms& atoms, bool compute_forces, bool keep_grid) const {
  return impl_->evaluate(atoms, compute_forces, keep_grid);
}

DeviceGeom D3::stage(const std::vector<System>& systems) const {
  std::vector<int> table(kZ, -1);
  for (int z = 1; z < kZ; ++z) table[z] = z;
  if (impl_->ztab_host.size() != std::size_t(kZ)) {
    std::vector<int> id(kZ);
    for (int z = 0; z < kZ; ++z) id[z] = z;
    impl_->set_types(id);
  }
  return stage_systems(impl_->ws, systems, table);
}

BatchResult D3::compute_device(const DeviceGeom& g, bool compute_forces) const {
  Atoms a;
  a.pos = g.pos, a.type = g.spec, a.sid = g.sid, a.soff = g.soff, a.scnt = g.scnt, a.sper = g.sper;
  a.scell = g.scell, a.N = g.Ntot, a.B = g.B;
  const Out o = impl_->evaluate(a, compute_forces, false);
  BatchResult br;
  br.n_struct = g.B, br.n_atoms = g.Ntot, br.struct_id = g.sid;
  auto conv1 = [&](const char* key, const Kokkos::View<double*, LR, MemSpace>& v) {
    RView1D r = impl_->ws.r1(key, v.extent(0));
    Kokkos::deep_copy(ExecSpace(), r, v);
    return r;
  };
  auto conv2 = [&](const char* key, const Kokkos::View<double**, LR, MemSpace>& v) {
    RView2D r = impl_->ws.r2(key, v.extent(0), v.extent(1));
    Kokkos::deep_copy(ExecSpace(), r, v);
    return r;
  };
  if constexpr (std::is_same_v<Real, double>) {
    br.energy = o.energy, br.per_atom = o.per_atom;
    if (compute_forces) br.forces = o.forces, br.virial = o.virial;
  } else {
    br.energy = conv1("d3:r_energy", o.energy), br.per_atom = conv1("d3:r_eat", o.per_atom);
    if (compute_forces) br.forces = conv2("d3:r_fat", o.forces), br.virial = conv2("d3:r_virial", o.virial);
  }
  return br;
}

Results D3::compute(const std::vector<System>& systems, bool compute_forces) const {
  Results out;
  if (systems.empty()) return out;
  const DeviceGeom g = stage(systems);
  Atoms a;
  a.pos = g.pos, a.type = g.spec, a.sid = g.sid, a.soff = g.soff, a.scnt = g.scnt, a.sper = g.sper;
  a.scell = g.scell, a.N = g.Ntot, a.B = g.B;
  const Out o = impl_->evaluate(a, compute_forces, false);
  auto copy = [](std::vector<double>& to, const auto& v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    to.assign(h.data(), h.data() + h.size());
  };
  copy(out.energy, o.energy);
  copy(out.per_atom_energy, o.per_atom);
  for (std::size_t b = 0; b < systems.size(); ++b) {
    out.n_atoms.push_back(systems[b].n_atoms);
    out.struct_id.insert(out.struct_id.end(), systems[b].n_atoms, int(b));
  }
  if (compute_forces) {
    copy(out.forces, o.forces);
    copy(out.virial, o.virial);
  } else {
    out.struct_id.clear();
  }
  return out;
}

Results D3::compute(const System& system, bool compute_forces) const {
  return compute(std::vector<System>{system}, compute_forces);
}

}  // namespace pet
