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
#include <chrono>
#include <cstdio>
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
// Bin width over the longest radius walked (half: the pair walk covers five bins
// a side, and a run along the third axis is one range however many bins it
// spans); PET_D3_BIN=width (Angstrom) instead.
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
// In T: double, or float in mixed precision.
template <class T>
KOKKOS_INLINE_FUNCTION void taper(T r, T rc, T w, T& t, T& dt) {
  dt = T(0);
  if (r > rc) {
    t = T(0);
    return;
  }
  const T lo = rc - w;
  if (w <= T(0) || r <= lo) {
    t = T(1);
    return;
  }
  const T x = (r - lo) / w, x2 = x * x, x3 = x2 * x;
  t = T(1) - x3 * x * (T(35) - T(84) * x + T(70) * x2 - T(20) * x3);
  dt = T(-140) * x3 * (T(1) - T(3) * x + T(3) * x2 - x3) / w;  // -140 x^3 (1 - x)^3 / w
}

// The CN's counting function 1 / (1 + exp(-k (rco / r - 1))) and its r-derivative.
template <class T>
KOKKOS_INLINE_FUNCTION void count(T r, T rco, T& f, T& df) {
  const T ex = Kokkos::exp(T(-kKcn) * (rco / r - T(1)));
  f = T(1) / (T(1) + ex);
  df = T(-kKcn) * rco / (r * r) * ex * f * f;
}

constexpr double kPi = 3.14159265358979323846;

// Gauss-Legendre nodes and weights on [-1, 1].
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0), w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(kPi * (i + 0.75) / (n + 0.5)), dp = 1.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j;
      }
      dp = n * (z * p0 - p1) / (z * z - 1.0);
      const double dz = p0 / dp;
      z -= dz;
      if (std::fabs(dz) < 1e-16) break;
    }
    x[i] = z, w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
}

}  // namespace

// ---- parameters -------------------------------------------------------------

D3Damping d3_damping(const std::string& name) {
  const std::string n = lower(name);
  if (n == "bj" || n == "rational" || n == "d3bj" || n == "d3(bj)") return D3Damping::BJ;
  if (n == "zero" || n == "d3zero" || n == "d3(0)" || n == "0") return D3Damping::Zero;
  throw std::invalid_argument("pet: unknown D3 damping '" + name + "' (bj or zero)");
}

D3Precision d3_precision(const std::string& name) {
  const std::string n = lower(name);
  if (n == "double" || n == "fp64") return D3Precision::Double;
  if (n == "mixed") return D3Precision::Mixed;
  if (n == "float" || n == "single" || n == "fp32") return D3Precision::Float;
  throw std::invalid_argument("pet: unknown D3 precision '" + name + "' (double, mixed or float)");
}

const char* d3_precision_name(D3Precision p) {
  return p == D3Precision::Float ? "float" : p == D3Precision::Mixed ? "mixed" : "double";
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
  c.tail = false;
  return c;
}

D3Settings D3Settings::functional(const std::string& name, D3Damping damping, bool atm) {
  D3Settings s;
  s.params = d3_parameters(name, damping, atm);
  return s;
}

// ---- the kernels --------------------------------------------------------------
//
// Every kernel is a template on two scalar types (D3Precision): P, each pair's
// and triangle's arithmetic; A, everything kept per atom or summed -- positions
// and the vectors between them, CNs, reference weights, dE/dCN, energies,
// forces, virials, the tail, and the lanes' and warps' sums. Double is <double,
// double>, Mixed <float, double>, Float <float, float>. Left in double whatever
// the precision: the grid's setup (each structure's frame and bin widths, and
// wrapping each atom into its cell: O(N) once a call, from the caller's double
// positions), the per-element tables' preparation on the host, and the results
// handed back (D3::Out), each structure's energy and virial summed over its
// atoms in double from the A values (a cheap pass, and the caller's types).

namespace {

template <class T>
using TV1 = Kokkos::View<T*, LR, MemSpace>;
template <class T>
using TV2 = Kokkos::View<T**, LR, MemSpace>;

// A named workspace buffer of any scalar type (the pool's are Real).
template <class T>
TV1<T> wsv1(Workspace& ws, const std::string& key, int n) {
  const std::size_t words = (std::size_t(n) * sizeof(T) + sizeof(Real) - 1) / sizeof(Real);
  return TV1<T>(reinterpret_cast<T*>(ws.r1(key, int(std::max<std::size_t>(words, 1))).data()), n);
}
template <class T>
TV2<T> wsv2(Workspace& ws, const std::string& key, int r, int c) {
  const std::size_t words = (std::size_t(r) * c * sizeof(T) + sizeof(Real) - 1) / sizeof(Real);
  return TV2<T>(reinterpret_cast<T*>(ws.r1(key, int(std::max<std::size_t>(words, 1))).data()), r, c);
}

// A lane's or warp's K sums.
template <int K, class T>
struct Sums {
  T v[K];
  KOKKOS_INLINE_FUNCTION Sums() {
    for (int k = 0; k < K; ++k) v[k] = T(0);
  }
  KOKKOS_INLINE_FUNCTION Sums& operator+=(const Sums& o) {
    for (int k = 0; k < K; ++k) v[k] += o.v[k];
    return *this;
  }
};

}  // namespace
}  // namespace pet

namespace Kokkos {
template <int K, class T>
struct reduction_identity<pet::Sums<K, T>> {
  KOKKOS_FORCEINLINE_FUNCTION static pet::Sums<K, T> sum() { return pet::Sums<K, T>(); }
};
}  // namespace Kokkos

namespace pet {
namespace {

struct TagGather {};
struct TagCN {};
struct TagWeights {};
template <bool F>
struct TagPair {};
template <bool F>
struct TagATM {};
struct TagChain {};
template <bool F>
struct TagTail {};
struct TagCount {};

template <class Tag>
using Range = Kokkos::RangePolicy<ExecSpace, Tag, Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
template <class Tag>
using Teams = Kokkos::TeamPolicy<ExecSpace, Tag, Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
template <class Tag>
using Member = typename Teams<Tag>::member_type;

// A warp per atom (a team of one thread with kLanes vector lanes); the lanes
// share out the atom's candidates and each sums its own, and the warp adds the
// lanes' sums in the reduction's fixed tree at the end. Positions, types and
// reference weights are gathered into bin order first (slot k holds atom
// batom(k)), so a run of bins along the third axis is one contiguous range of
// slots, and the lanes read it coalesced.
template <class P, class A>
struct Kernels {
  // tables, pet units: the pair arithmetic's in P, the weights' and tail's in A
  IView1D nref, ztab;
  TV1<P> c6, pairtab, rcov, lr0;  // pairtab [kZ * kZ * 3]: Q = C8/C6, then BJ's R0^6, R0^8 or zero's rs6 R0, rs8 R0
  TV1<A> c6a, cnref, tailtab;     // tailtab [kZ * kZ]: the pair term per unit C6 beyond the pair cutoff, integrated
  // The tail correction's per-structure state: elements present, and each
  // element's reference weights summed over the structure's atoms.
  IView2D present;   // [B, kZ]
  TV2<A> wsum;       // [B, kZ * kR]
  IView1D soff, scnt;
  RView2D scell;
  bool tail = false;
  // atoms
  IView1D type, sid, sper;
  // grid
  TV2<A> frame, width;  // [B, 12] (lattice or box rows, then origin), [B, 3]
  IView2D nc;           // [B, 3]
  IView1D goff, abin, bstart, batom;
  TV2<double> posw;     // [N, 3] wrapped along periodic axes, by atom
  // by slot (bin order)
  TV2<A> xs;            // [N, 3] positions
  TV2<float> xf;        // [N, 3] the same in float, to pass over far candidates cheaply (A = double)
  IView1D zs;           // [N] atomic numbers
  TV2<A> gws, dgws;     // [N, kR] reference weights and their CN-derivatives
  TV2<P> gwp, dgwp;     // the same in P (the same views when P is A)
  TV1<A> dedcns;        // [N] dE/dCN
  // by atom: state and outputs
  TV1<A> cn, dedcn, eat;
  TV2<A> fat, wat;
  int n_local = -1;
  // parameters
  bool bj = true;
  double s6 = 1, s8 = 0, a1 = 0, a2 = 0, rs6 = 1, rs8 = 1, alp = 14, s9 = 0;
  double rp = 0, wp = 0, rc = 0, wc = 0, ra = 0, wa = 0;
  double isEh = 1.0 / std::sqrt(kHartree);  // sqrt(C6^3) is a C9 in atomic units only
  // The same in the types they are used in, set once (typed()): a conversion
  // from double costs as much as double arithmetic on a GPU short of it.
  P s6p = 1, s8p = 0, alpp = 14, s9p = 0, exp9 = 16.0f / 3, isEhp = 0;
  P rpp = 0, wpp = 0, rcp = 0, wcp = 0, rap = 0, wap = 0, ra2p = 0;
  A rpa = 0, rca = 0, raa = 0;
  void typed() {
    s6p = P(s6), s8p = P(s8), alpp = P(alp), s9p = P(s9), exp9 = P((alp + 2.0) / 3.0), isEhp = P(isEh);
    rpp = P(rp), wpp = P(wp), rcp = P(rc), wcp = P(wc), rap = P(ra), wap = P(wa), ra2p = P(ra * ra);
    rpa = A(rp), rca = A(rc), raa = A(ra);
  }
  int atm_cap = 0;

  static constexpr bool kPrefilter = !std::is_same_v<A, float>;  // positions already float: no prefilter

  KOKKOS_INLINE_FUNCTION A owned_atom(int a) const { return (n_local < 0 || a < n_local) ? A(1) : A(0); }
  KOKKOS_INLINE_FUNCTION A owned(int k) const { return owned_atom(batom(k)); }

  // g(k0, k1, shift, q) for each run of slots holding the candidates within R of
  // the atom in slot ks: the bins out to ceil(R / width) along each axis, those
  // consecutive along the third (with the same lattice image) as one run. q
  // numbers the runs in walk order.
  template <class G>
  KOKKOS_INLINE_FUNCTION void runs(int ks, A R, G&& g) const {
    const int i = batom(ks), b = sid(i), per = sper(b);
    const int n0 = nc(b, 0), n1 = nc(b, 1), n2 = nc(b, 2);
    const int loc = abin(i) - goff(b);
    const int c0 = loc / (n1 * n2), c1 = (loc / n2) % n1, c2 = loc % n2;
    const int n[3] = {n0, n1, n2};
    int r[3];
    for (int d = 0; d < 3; ++d) {
      r[d] = (int) Kokkos::ceil(R / width(b, d));
      if (!(per >> d & 1)) r[d] = Kokkos::min(r[d], n[d] - 1);
    }
    int q = 0;
    for (int ta = c0 - r[0]; ta <= c0 + r[0]; ++ta) {
      if (!(per & 1) && (ta < 0 || ta >= n0)) continue;
      const int sa = (per & 1) ? floor_div(ta, n0) : 0, ma = ta - sa * n0;
      for (int tb = c1 - r[1]; tb <= c1 + r[1]; ++tb) {
        if (!(per & 2) && (tb < 0 || tb >= n1)) continue;
        const int sb = (per & 2) ? floor_div(tb, n1) : 0, mb = tb - sb * n1;
        const int row = goff(b) + (ma * n1 + mb) * n2;
        int lo = c2 - r[2], hi = c2 + r[2];
        if (!(per & 4)) lo = Kokkos::max(lo, 0), hi = Kokkos::min(hi, n2 - 1);
        for (int t = lo; t <= hi;) {
          const int sc = (per & 4) ? floor_div(t, n2) : 0, m0 = t - sc * n2;
          const int m1 = Kokkos::min(n2 - 1, m0 + (hi - t));
          A sh[3];
          for (int e = 0; e < 3; ++e) sh[e] = A(sa) * frame(b, e) + A(sb) * frame(b, 3 + e) + A(sc) * frame(b, 6 + e);
          g(bstart(row + m0), bstart(row + m1 + 1), sh, q++);
          t += m1 - m0 + 1;
        }
      }
    }
  }

  // The candidate in slot k from the atom in slot ks, offset (o, of): true and
  // v = r_k + shift - r_ks, d2 = |v|^2 when within R (itself excluded). With
  // double positions, a float test with room for its rounding passes over the
  // many candidates far past R first.
  KOKKOS_INLINE_FUNCTION bool candidate(int k, const A* o, const float* of, A R2, float Rf2, A* v, A& d2) const {
    if constexpr (kPrefilter) {
      const float ux = xf(k, 0) + of[0], uy = xf(k, 1) + of[1], uz = xf(k, 2) + of[2];
      if (ux * ux + uy * uy + uz * uz > Rf2) return false;
    }
    for (int x = 0; x < 3; ++x) v[x] = xs(k, x) + o[x];
    d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    return d2 >= A(1e-20) && d2 <= R2;
  }
  struct Origin {
    A o[3];
    float of[3];
  };
  KOKKOS_INLINE_FUNCTION Origin origin(int ks, const A* sh) const {
    Origin g;
    for (int x = 0; x < 3; ++x) {
      g.o[x] = sh[x] - xs(ks, x);
      g.of[x] = kPrefilter ? float(sh[x]) - xf(ks, x) : 0.0f;
    }
    return g;
  }

  // f(k, v, d2, q) for this lane's share of the candidates within R of slot ks.
  template <class F>
  KOKKOS_INLINE_FUNCTION void lane_walk(int ks, A R, int lane, F&& f) const {
    const A R2 = R * R;
    const float Rf2 = float(R2) * 1.0001f + 1e-3f;
    runs(ks, R, [&](int k0, int k1, const A* sh, int q) {
      const Origin g = origin(ks, sh);
      for (int k = k0 + lane; k < k1; k += kLanes) {
        A v[3], d2;
        if (candidate(k, g.o, g.of, R2, Rf2, v, d2)) f(k, v, d2, q);
      }
    });
  }

  // The same for the whole warp, the hits compacted first. Lanes test kLanes
  // candidates at a time and an ordered scan packs those within R into a
  // buffer; each full kLanes of hits is then evaluated with every lane busy --
  // most candidates miss, and evaluating them in place would idle most lanes
  // through the expensive part. f(k, v, d2, sums) adds into this lane's K sums;
  // out gets their total, in the reduction's fixed tree. Hits go to lanes in
  // walk order, so the result is as reproducible as the walk.
  template <class T>
  using Scratch2 = Kokkos::View<T**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  using ScratchI1 = Kokkos::View<int*, ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  template <int K>
  static std::size_t walk_scratch() {
    return Scratch2<A>::shmem_size(2 * kLanes, 4) + ScratchI1::shmem_size(2 * kLanes) +
           Scratch2<A>::shmem_size(kLanes, K);
  }
  template <int K, class M, class F>
  KOKKOS_INLINE_FUNCTION void team_walk(const M& t, int ks, A R, F&& f, A* out) const {
    Scratch2<A> bv(t.team_scratch(0), 2 * kLanes, 4);
    ScratchI1 bk(t.team_scratch(0), 2 * kLanes);
    Scratch2<A> acc(t.team_scratch(0), kLanes, K);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, kLanes), [&](int l) {
      for (int c = 0; c < K; ++c) acc(l, c) = A(0);
    });
    const A R2 = R * R;
    const float Rf2 = float(R2) * 1.0001f + 1e-3f;
    int count = 0;
    auto evaluate = [&](int n) {
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, n), [&](int l) {
        const A v[3] = {bv(l, 0), bv(l, 1), bv(l, 2)};
        A sums[K];
        for (int c = 0; c < K; ++c) sums[c] = acc(l, c);
        f(bk(l), v, bv(l, 3), sums);
        for (int c = 0; c < K; ++c) acc(l, c) = sums[c];
      });
      t.team_barrier();
    };
    runs(ks, R, [&](int k0, int k1, const A* sh, int) {
      const Origin g = origin(ks, sh);
      for (int base = k0; base < k1; base += kLanes) {
        int added = 0;
        Kokkos::parallel_scan(
            Kokkos::ThreadVectorRange(t, kLanes),
            [&](int l, int& pos, bool final) {
              const int k = base + l;
              A v[3] = {A(0), A(0), A(0)}, d2 = A(0);
              const bool hit = k < k1 && candidate(k, g.o, g.of, R2, Rf2, v, d2);
              if (final && hit) {
                const int w = count + pos;
                bk(w) = k;
                for (int c = 0; c < 3; ++c) bv(w, c) = v[c];
                bv(w, 3) = d2;
              }
              pos += hit;
            },
            added);
        count += added;
        if (count >= kLanes) {
          evaluate(kLanes);
          const int rest = count - kLanes;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, rest), [&](int m) {
            bk(m) = bk(kLanes + m);
            for (int c = 0; c < 4; ++c) bv(m, c) = bv(kLanes + m, c);
          });
          t.team_barrier();
          count = rest;
        }
      }
    });
    if (count > 0) evaluate(count);
    for (int c = 0; c < K; ++c) {
      A x = A(0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(t, kLanes), [&](int l, A& y) { y += acc(l, c); }, x);
      out[c] = x;
    }
  }

  // C6 between slots ki (its weights wi, dwi over ni references) and kj, and dC6/dCN_i.
  KOKKOS_INLINE_FUNCTION void c6_pair(int zi, int ni, const P* wi, const P* dwi, int kj, int zj, P& C, P& dC) const {
    const int nj = nref(zj);
    const int base = (zi * kZ + zj) * (kR * kR);
    C = P(0), dC = P(0);
    for (int a = 0; a < ni; ++a) {
      P u = P(0);
      for (int bb = 0; bb < nj; ++bb) u += c6(base + a * kR + bb) * gwp(kj, bb);
      C += wi[a] * u;
      dC += dwi[a] * u;
    }
  }
  KOKKOS_INLINE_FUNCTION P c6_plain(int ki, int zi, int kj, int zj) const {
    const int ni = nref(zi), nj = nref(zj);
    const int base = (zi * kZ + zj) * (kR * kR);
    P C = P(0);
    for (int a = 0; a < ni; ++a) {
      P u = P(0);
      for (int bb = 0; bb < nj; ++bb) u += c6(base + a * kR + bb) * gwp(kj, bb);
      C += gwp(ki, a) * u;
    }
    return C;
  }

  // Slot order: positions and atomic numbers.
  KOKKOS_INLINE_FUNCTION void operator()(TagGather, int k) const {
    const int a = batom(k);
    for (int d = 0; d < 3; ++d) {
      xs(k, d) = A(posw(a, d));
      if constexpr (kPrefilter) xf(k, d) = float(posw(a, d));
    }
    zs(k) = ztab(type(a));
  }

  KOKKOS_INLINE_FUNCTION void operator()(TagCN, const Member<TagCN>& t) const {
    const int ks = t.league_rank(), zi = zs(ks);
    const P rci = rcov(zi);
    A s[1];
    team_walk<1>(t, ks, rca, [&](int k, const A*, A d2, A* acc) {
      const P r = Kokkos::sqrt(P(d2));
      P f, df, tp, dtp;
      count<P>(r, rci + rcov(zs(k)), f, df);
      taper<P>(r, rcp, wcp, tp, dtp);
      acc[0] += A(f * tp);
    }, s);
    Kokkos::single(Kokkos::PerThread(t), [&] { cn(batom(ks)) = s[0]; });
  }

  // s-dftd3's weights: Gaussians in CN - CN_ref, normalised; where the sum
  // underflows, all weight on the reference with the largest CN.
  KOKKOS_INLINE_FUNCTION void operator()(TagWeights, int k) const {
    const int zi = zs(k), ni = nref(zi);
    const A x = cn(batom(k)), k3 = A(kK3);
    A norm = A(0), dnorm = A(0);
    for (int a = 0; a < ni; ++a) {
      const A d = x - cnref(zi * kR + a), e = Kokkos::exp(-k3 * d * d);
      norm += e;
      dnorm += A(-2) * k3 * d * e;
    }
    A maxref = A(-1);
    for (int a = 0; a < ni; ++a) maxref = Kokkos::fmax(maxref, cnref(zi * kR + a));
    const A inv = A(1) / norm;
    for (int a = 0; a < kR; ++a) {
      A g = A(0), dg = A(0);
      if (a < ni) {
        const A d = x - cnref(zi * kR + a), e = Kokkos::exp(-k3 * d * d), de = A(-2) * k3 * d * e;
        g = e * inv;
        dg = inv * (de - e * dnorm * inv);
        if (!(norm > A(0)) || !Kokkos::isfinite(g)) g = cnref(zi * kR + a) == maxref ? A(1) : A(0);
        if (!Kokkos::isfinite(dg)) dg = A(0);
      }
      gws(k, a) = g;
      dgws(k, a) = dg;
      if constexpr (!std::is_same_v<P, A>) gwp(k, a) = P(g), dgwp(k, a) = P(dg);
    }
  }

  // The two-body term per unit C6, untapered, and its r-derivative; ir = 1/r.
  KOKKOS_INLINE_FUNCTION void damped(P r, P ir, P d2, int zi, int zj, P& e, P& de) const {
    const int p = 3 * (zi * kZ + zj);
    const P Q = pairtab(p), S6 = s6p, S8 = s8p;
    const P r6 = d2 * d2 * d2, r8 = r6 * d2;
    if (bj) {
      const P Aa = r6 + pairtab(p + 1), Bd = r8 + pairtab(p + 2), id = P(1) / (Aa * Bd);
      const P t6 = Bd * id, t8 = Aa * id;  // 1/(r^6 + R0^6), 1/(r^8 + R0^8)
      e = -(S6 * t6 + S8 * Q * t8);
      de = (P(6) * S6 * r6 * t6 * t6 + P(8) * S8 * Q * r8 * t8 * t8) * ir;
    } else {
      const P al = alpp;
      const P q6 = Kokkos::pow(r / pairtab(p + 1), -al), q8 = Kokkos::pow(r / pairtab(p + 2), -(al + P(2)));
      const P f6 = P(1) / (P(1) + P(6) * q6), f8 = P(1) / (P(1) + P(6) * q8);
      const P df6 = P(6) * al * q6 * ir * f6 * f6, df8 = P(6) * (al + P(2)) * q8 * ir * f8 * f8;
      const P i6 = P(1) / r6, i8 = i6 / d2;
      e = -(S6 * f6 * i6 + S8 * Q * f8 * i8);
      de = -(S6 * (df6 - P(6) * f6 * ir) * i6 + S8 * Q * (df8 - P(8) * f8 * ir) * i8);
    }
  }

  KOKKOS_INLINE_FUNCTION static void add_virial(A* w, A h, const A* v) {
    w[0] += h * v[0] * v[0], w[1] += h * v[1] * v[1], w[2] += h * v[2] * v[2];
    w[3] += h * v[0] * v[1], w[4] += h * v[0] * v[2], w[5] += h * v[1] * v[2];
  }

  // The pairs: energy, forces, virial and dE/dCN, for atom i's full
  // neighbourhood. Sums: e, f[3], w[6], dE/dCN.
  template <bool F>
  KOKKOS_INLINE_FUNCTION void operator()(TagPair<F>, const Member<TagPair<F>>& t) const {
    const int ks = t.league_rank(), i = batom(ks), zi = zs(ks), ni = nref(zi);
    const A oi = owned_atom(i);
    P wi[kR], dwi[kR];
    for (int a = 0; a < kR; ++a) wi[a] = gwp(ks, a), dwi[a] = dgwp(ks, a);
    constexpr int K = F ? 11 : 1;
    A s[K];
    team_walk<K>(t, ks, rpa, [&](int k, const A* v, A d2, A* acc) {
      const A wt = A(0.5) * (oi + owned(k));
      if (wt == A(0)) return;
      const int zj = zs(k);
      const P r = Kokkos::sqrt(P(d2)), ir = P(1) / r;
      P C, dC, u, du, tp, dtp;
      c6_pair(zi, ni, wi, dwi, k, zj, C, dC);
      damped(r, ir, P(d2), zi, zj, u, du);
      taper<P>(r, rpp, wpp, tp, dtp);
      acc[0] += A(0.5) * oi * A(C * u * tp);
      if constexpr (F) {
        const A g = wt * A(C * (du * tp + u * dtp) * ir);  // dE/dv = g v
        for (int x = 0; x < 3; ++x) acc[1 + x] += g * v[x];
        add_virial(acc + 4, A(0.5) * g, v);
        acc[10] += wt * A(dC * u * tp);
      }
    }, s);
    Kokkos::single(Kokkos::PerThread(t), [&] {
      eat(i) = s[0];
      if constexpr (F) {
        for (int x = 0; x < 3; ++x) fat(i, x) = s[1 + x];
        for (int x = 0; x < 6; ++x) wat(i, x) = s[4 + x];
        dedcn(i) = s[10];
      }
    });
  }

  // Axilrod-Teller-Muto, zero-damped as s-dftd3 has it: for each triangle i, j,
  // k with every side within the cutoff,
  //   E = C9 (3 cos a cos b cos c + 1) / (r_ij r_ik r_jk)^3 f(r_ij r_ik r_jk),
  //   C9 = s9 sqrt(C6_ij C6_ik C6_jk / Eh), f(P) = 1 / (1 + 6 (R0 / P)^((alpha + 2) / 3)),
  // R0 the product of (4/3) R0_ab. Atom i sums every triangle it is a corner of,
  // each pair {j, k} once; a third of each is its own.
  //
  // Atom i's neighbours within the cutoff, with what each triangle needs of
  // them: slot, v = r_j - r_i, |v|^2, |v|, 1/|v|, the taper and its derivative,
  // sqrt(C6_ij), dC6_ij/dCN_i / C6_ij, and the logs of |v| and (4/3) R0_ij.
  struct Nb {
    int k;
    P v[3], d2, r, ir, t, dt, sC, dCr, lr, lr0;
  };
  static constexpr int kNb = 12;  // the values of an Nb in the list
  KOKKOS_INLINE_FUNCTION Nb neighbour(int k, const A* v, A d2, int zi, int ni, const P* wi, const P* dwi) const {
    Nb n;
    n.k = k;
    for (int x = 0; x < 3; ++x) n.v[x] = P(v[x]);
    n.d2 = P(d2), n.r = Kokkos::sqrt(P(d2)), n.ir = P(1) / n.r;
    const int zj = zs(k);
    P C, dC;
    c6_pair(zi, ni, wi, dwi, k, zj, C, dC);
    n.sC = Kokkos::sqrt(C), n.dCr = C > P(0) ? dC / C : P(0);
    taper<P>(n.r, rap, wap, n.t, n.dt);
    n.lr = Kokkos::log(n.r), n.lr0 = lr0(zi * kZ + zj);
    return n;
  }
  // One triangle i, j, k into atom i's sums: e, f[3], w[6], dE/dCN_i.
  template <bool F>
  KOKKOS_INLINE_FUNCTION void triangle(A oi, const Nb& J, const Nb& K, A* acc) const {
    const P vjk[3] = {K.v[0] - J.v[0], K.v[1] - J.v[1], K.v[2] - J.v[2]};
    const P c = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
    if (c > ra2p || c < P(1e-20)) return;
    const A wt = (oi + owned(J.k) + owned(K.k)) / A(3);
    if (wt == A(0)) return;
    const int zj = zs(J.k), zk = zs(K.k);
    const P a = J.d2, b = K.d2, rij = J.r, rik = K.r, rjk = Kokkos::sqrt(c), irjk = P(1) / rjk;
    const P ex = exp9;
    const P Cjk = c6_plain(J.k, zj, K.k, zk);
    const P C9 = s9p * J.sC * K.sC * Kokkos::sqrt(Kokkos::fabs(Cjk)) * isEhp;
    // (R0 / P)^ex from logs: R0 the product of the sides' (4/3) R0, P of the sides.
    const P u = Kokkos::exp(ex * (J.lr0 + K.lr0 + lr0(zj * kZ + zk) - J.lr - K.lr - Kokkos::log(rjk)));
    const P fd = P(1) / (P(1) + P(6) * u);
    const P iP = J.ir * K.ir * irjk, iP3 = iP * iP * iP, iP5 = iP3 * iP * iP;
    const P X = a - b + c, Y = a + b - c, Zt = -a + b + c, sgeo = X * Y * Zt;
    const P ang = P(0.375) * sgeo * iP5 + iP3;
    P tjk, dtjk;
    taper<P>(rjk, rap, wap, tjk, dtjk);
    const P Tp = J.t * K.t * tjk;
    const P E = C9 * ang * fd * Tp;
    acc[0] += oi * A(E) / A(3);
    if constexpr (F) {
      // d(ang)/d(each squared side), P d(fd)/dP, then each side's dE/dr:
      // through the angle factor, the damping and the tapers.
      const P s_a = Y * Zt + X * Zt - X * Y, s_b = -Y * Zt + X * Zt + X * Y, s_c = Y * Zt - X * Zt + X * Y;
      const P ia = J.ir * J.ir, ib = K.ir * K.ir, ic = irjk * irjk;
      const P dang_a = (P(0.375) * (s_a - P(2.5) * sgeo * ia) * iP5 - P(1.5) * iP3 * ia);
      const P dang_b = (P(0.375) * (s_b - P(2.5) * sgeo * ib) * iP5 - P(1.5) * iP3 * ib);
      const P dang_c = (P(0.375) * (s_c - P(2.5) * sgeo * ic) * iP5 - P(1.5) * iP3 * ic);
      const P dfd = P(6) * ex * u * fd * fd;
      const P gij = C9 * (P(2) * rij * dang_a * fd * Tp + ang * dfd * J.ir * Tp + ang * fd * J.dt * K.t * tjk);
      const P gik = C9 * (P(2) * rik * dang_b * fd * Tp + ang * dfd * K.ir * Tp + ang * fd * J.t * K.dt * tjk);
      const P gjk = C9 * (P(2) * rjk * dang_c * fd * Tp + ang * dfd * irjk * Tp + ang * fd * J.t * K.t * dtjk);
      const A hij = wt * A(gij * J.ir), hik = wt * A(gik * K.ir), hjk = wt * A(gjk * irjk);  // dE/dv = h v
      const A vj[3] = {A(J.v[0]), A(J.v[1]), A(J.v[2])}, vk[3] = {A(K.v[0]), A(K.v[1]), A(K.v[2])};
      const A vd[3] = {A(vjk[0]), A(vjk[1]), A(vjk[2])};
      for (int x = 0; x < 3; ++x) acc[1 + x] += hij * vj[x] + hik * vk[x];
      add_virial(acc + 4, hij / A(3), vj);
      add_virial(acc + 4, hik / A(3), vk);
      add_virial(acc + 4, hjk / A(3), vd);
      // dE/dCN_i through C6_ij and C6_ik: C9 ~ sqrt(C6_ij C6_ik C6_jk).
      acc[10] += wt * A(0.5) * A(E * (J.dCr + K.dCr));
    }
  }

  // Each atom's neighbours within the ATM cutoff into a list in level-1
  // scratch (atm_cap of them, compacted in walk order), then the lanes share
  // out its pairs {p, q}, p < q: lane l takes p = l, l + kLanes, ... An atom
  // with more neighbours than the list holds walks the grid again for each j
  // instead (slow, but a fixed capacity holds for a recorded step).
  static std::size_t atm_scratch(int cap) {
    return Scratch2<P>::shmem_size(cap, kNb) + ScratchI1::shmem_size(cap);
  }
  template <bool F>
  KOKKOS_INLINE_FUNCTION void operator()(TagATM<F>, const Member<TagATM<F>>& t) const {
    const int ks = t.league_rank(), i = batom(ks), zi = zs(ks), ni = nref(zi);
    const A oi = owned_atom(i);
    P wi[kR], dwi[kR];
    for (int a = 0; a < kR; ++a) wi[a] = gwp(ks, a), dwi[a] = dgwp(ks, a);
    Scratch2<P> L(t.team_scratch(1), atm_cap, kNb);
    ScratchI1 Lk(t.team_scratch(1), atm_cap);
    const A R = raa, R2 = R * R;
    const float Rf2 = float(R2) * 1.0001f + 1e-3f;
    // The list: ordered compaction, as team_walk's.
    int n = 0;
    runs(ks, R, [&](int k0, int k1, const A* sh, int) {
      const Origin g = origin(ks, sh);
      for (int base = k0; base < k1; base += kLanes) {
        int added = 0;
        Kokkos::parallel_scan(
            Kokkos::ThreadVectorRange(t, kLanes),
            [&](int l, int& pos, bool final) {
              const int k = base + l;
              A v[3] = {A(0), A(0), A(0)}, d2 = A(0);
              const bool hit = k < k1 && candidate(k, g.o, g.of, R2, Rf2, v, d2);
              if (final && hit && n + pos < atm_cap) {
                const int w = n + pos;
                Lk(w) = k;
                for (int c = 0; c < 3; ++c) L(w, c) = P(v[c]);
                L(w, 3) = P(d2);
              }
              pos += hit;
            },
            added);
        n += added;
      }
    });
    Sums<11, A> s;
    if (n <= atm_cap) {
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, n), [&](int m) {
        const A v[3] = {A(L(m, 0)), A(L(m, 1)), A(L(m, 2))};
        const Nb b = neighbour(Lk(m), v, A(L(m, 3)), zi, ni, wi, dwi);
        L(m, 4) = b.r, L(m, 5) = b.ir, L(m, 6) = b.t, L(m, 7) = b.dt, L(m, 8) = b.sC, L(m, 9) = b.dCr;
        L(m, 10) = b.lr, L(m, 11) = b.lr0;
      });
      t.team_barrier();
      auto get = [&](int m) {
        Nb b;
        b.k = Lk(m);
        for (int x = 0; x < 3; ++x) b.v[x] = L(m, x);
        b.d2 = L(m, 3), b.r = L(m, 4), b.ir = L(m, 5), b.t = L(m, 6), b.dt = L(m, 7), b.sC = L(m, 8), b.dCr = L(m, 9);
        b.lr = L(m, 10), b.lr0 = L(m, 11);
        return b;
      };
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(t, kLanes),
          [&](int lane, Sums<11, A>& sum) {
            for (int p = lane; p < n; p += kLanes) {
              const Nb J = get(p);
              for (int q = p + 1; q < n; ++q) triangle<F>(oi, J, get(q), sum.v);
            }
          },
          Kokkos::Sum<Sums<11, A>>(s));
    } else {
      // Past the list's capacity: j from this lane's share of the walk, k from a
      // walk of its own from j on.
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(t, kLanes),
          [&](int lane, Sums<11, A>& sum) {
            lane_walk(ks, R, lane, [&](int kj, const A* vij, A a, int qj) {
              const Nb J = neighbour(kj, vij, a, zi, ni, wi, dwi);
              runs(ks, R, [&](int k0, int k1, const A* sh, int qk) {
                if (qk < qj) return;
                const Origin g = origin(ks, sh);
                for (int kk = (qk == qj ? kj + 1 : k0); kk < k1; ++kk) {
                  A vik[3], b;
                  if (candidate(kk, g.o, g.of, R2, Rf2, vik, b))
                    triangle<F>(oi, J, neighbour(kk, vik, b, zi, ni, wi, dwi), sum.v);
                }
              });
            });
          },
          Kokkos::Sum<Sums<11, A>>(s));
    }
    Kokkos::single(Kokkos::PerThread(t), [&] {
      eat(i) += s.v[0];
      if (F) {
        for (int x = 0; x < 3; ++x) fat(i, x) += s.v[1 + x];
        for (int x = 0; x < 6; ++x) wat(i, x) += s.v[4 + x];
        dedcn(i) += s.v[10];
      }
    });
  }

  // How many neighbours each atom has within the ATM cutoff (for atm_cap).
  KOKKOS_INLINE_FUNCTION void operator()(TagCount, const Member<TagCount>& t) const {
    const int ks = t.league_rank();
    A s[1];
    team_walk<1>(t, ks, raa, [&](int, const A*, A, A* acc) { acc[0] += A(1); }, s);
    Kokkos::single(Kokkos::PerThread(t), [&] { dedcns(ks) = s[0]; });
  }

  // The tail beyond the pair cutoff, for a 3D-periodic structure of volume V,
  // as a uniform continuum of every atom of the structure (g(r) = 1):
  //   E_i = 1/(2V) sum_j C6_ij T_ij,  T_ij = int (1 - taper) e_ij(r) 4 pi r^2 dr,
  // with C6_ij = sum_ab w_ia w_jb C6ref_ab summed over j through each element's
  // total weights. Its virial is -E on the diagonal (E ~ 1/V), its forces come
  // through the CNs (dE/dCN, to the chain rule).
  template <bool F>
  KOKKOS_INLINE_FUNCTION void operator()(TagTail<F>, int k) const {
    const int i = batom(k), b = sid(i);
    if (sper(b) != 7) return;
    const double c0 = scell(b, 0), c1 = scell(b, 1), c2 = scell(b, 2), c3 = scell(b, 3), c4 = scell(b, 4),
                 c5 = scell(b, 5), c6v = scell(b, 6), c7 = scell(b, 7), c8 = scell(b, 8);
    const A V = A(Kokkos::fabs(c0 * (c4 * c8 - c5 * c7) - c1 * (c3 * c8 - c5 * c6v) + c2 * (c3 * c7 - c4 * c6v)));
    if (!(V > A(0))) return;
    const int zi = zs(k), ni = nref(zi);
    A e = A(0), de = A(0);
    for (int zj = 1; zj < kZ; ++zj) {
      if (!present(b, zj)) continue;
      const int nj = nref(zj), base = (zi * kZ + zj) * (kR * kR);
      const A T = tailtab(zi * kZ + zj);
      for (int a = 0; a < ni; ++a) {
        A u = A(0);
        for (int bb = 0; bb < nj; ++bb) u += c6a(base + a * kR + bb) * wsum(b, zj * kR + bb);
        e += gws(k, a) * u * T;
        de += dgws(k, a) * u * T;
      }
    }
    e *= A(0.5) / V;
    de *= A(1) / V;
    eat(i) += e;
    if (F) {
      wat(i, 0) -= e, wat(i, 1) -= e, wat(i, 2) -= e;
      dedcn(i) += de;
    }
  }

  // The CN's chain rule: each pair i-j moves both CNs, so the force on i is
  // sum_j (dE/dCN_i + dE/dCN_j) dcn_ij/dv. Sums: f[3], w[6].
  KOKKOS_INLINE_FUNCTION void operator()(TagChain, const Member<TagChain>& t) const {
    const int ks = t.league_rank(), i = batom(ks), zi = zs(ks);
    const A di = dedcns(ks);
    const P rci = rcov(zi);
    A s[9];
    team_walk<9>(t, ks, rca, [&](int k, const A* v, A d2, A* acc) {
      const A sum = di + dedcns(k);
      if (sum == A(0)) return;
      const P r = Kokkos::sqrt(P(d2));
      P c, dc, tp, dtp;
      count<P>(r, rci + rcov(zs(k)), c, dc);
      taper<P>(r, rcp, wcp, tp, dtp);
      const A g = sum * A((dc * tp + c * dtp) / r);
      for (int x = 0; x < 3; ++x) acc[x] += g * v[x];
      add_virial(acc + 3, A(0.5) * g, v);
    }, s);
    Kokkos::single(Kokkos::PerThread(t), [&] {
      for (int x = 0; x < 3; ++x) fat(i, x) += s[x];
      for (int x = 0; x < 6; ++x) wat(i, x) += s[3 + x];
    });
  }
};

}  // namespace

// ---- the D3 object ----------------------------------------------------------------

namespace d3detail {

TV1<float> to_float(const DV1& d, const char* label) {
  const int n = int(d.extent(0));
  TV1<float> f(Kokkos::view_alloc(std::string(label)), n);
  auto out = f;
  auto in = d;
  Kokkos::parallel_for(
      "d3_to_float", RangePolicy(0, n), KOKKOS_LAMBDA(int q) { out(q) = float(in(q)); });
  return f;
}

// A copy of a double view in T (the same view when T is double).
template <class T>
TV2<T> as(Workspace& ws, const std::string& key, const TV2<double>& d) {
  if constexpr (std::is_same_v<T, double>) {
    return d;
  } else {
    TV2<T> t = wsv2<T>(ws, key, d.extent(0), d.extent(1));
    auto in = d;
    const int c = d.extent(1);
    Kokkos::parallel_for(
        "d3_as", RangePolicy(0, int(d.extent(0))), KOKKOS_LAMBDA(int r) {
          for (int k = 0; k < c; ++k) t(r, k) = T(in(r, k));
        });
    return t;
  }
}

void fill_identity(IView1D id) {
  Kokkos::parallel_for("d3_identity", RangePolicy(0, (int) id.extent(0)), KOKKOS_LAMBDA(int z) { id(z) = z; });
}

// The implementation, at namespace scope: nvcc takes no device lambda in a
// private nested class (D3::Impl derives from it).
struct Core {
  D3Settings s;
  Workspace ws;
  // tables
  IView1D nref, identity;
  DV1 cnref, rcov, r4r2, r0, c6, pairtab, tailtab, lr0;
  TV1<float> c6f, pairtabf, rcovf, lr0f, cnreff, tailtabf;  // float copies, for Mixed and Float
  template <class T>
  static TV1<T> pick(const DV1& d, const TV1<float>& f) {
    if constexpr (std::is_same_v<T, double>) return d; else return f;
  }
  // the grid as last sized
  bool sized = false;
  std::size_t resizes = 0;
  int sized_N = -1, sized_B = -1, nbins = 0;
  IView1D overflow{"d3:overflow", 1};
  D3::Out out;

  explicit Core(D3Settings st) : s(st) {
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
    identity = IView1D("d3:identity", kZ);
    fill_identity(identity);
    // Per element pair: C8/C6, and the damping radii's powers.
    auto hp = Kokkos::create_mirror_view(pairtab = DV1("d3:pairtab", std::size_t(kZ) * kZ * 3));
    const D3Params& P = s.params;
    for (int i = 0; i < kZ; ++i)
      for (int j = 0; j < kZ; ++j) {
        const std::size_t q = 3 * (std::size_t(i) * kZ + j);
        const double Q = 3.0 * hr4(i) * hr4(j);
        hp(q) = Q;
        if (P.damping == D3Damping::BJ) {
          const double R0 = P.a1 * std::sqrt(Q) + P.a2;
          hp(q + 1) = std::pow(R0, 6), hp(q + 2) = std::pow(R0, 8);
        } else {
          hp(q + 1) = P.rs6 * hr0(i * kZ + j), hp(q + 2) = P.rs8 * hr0(i * kZ + j);
        }
      }
    Kokkos::deep_copy(pairtab, hp);
    // The tail integrals: the region past cutoff - taper weighted by 1 - taper,
    // then cutoff to infinity (r = cutoff / x), by Gauss-Legendre in each.
    auto ht = Kokkos::create_mirror_view(tailtab = DV1("d3:tailtab", std::size_t(kZ) * kZ));
    std::vector<double> gx, gw;
    gauss_legendre(48, gx, gw);
    const double rc = s.cutoffs.pair, w = s.cutoffs.pair_taper;
    for (int i = 0; i < kZ; ++i)
      for (int j = 0; j < kZ; ++j) {
        const std::size_t q = 3 * (std::size_t(i) * kZ + j);
        auto e = [&](double r) {  // the pair term per unit C6, as Kernels::damped
          const double d2 = r * r, r6 = d2 * d2 * d2, r8 = r6 * d2;
          if (P.damping == D3Damping::BJ) return -(P.s6 / (r6 + hp(q + 1)) + P.s8 * hp(q) / (r8 + hp(q + 2)));
          const double f6 = 1.0 / (1.0 + 6.0 * std::pow(r / hp(q + 1), -P.alpha));
          const double f8 = 1.0 / (1.0 + 6.0 * std::pow(r / hp(q + 2), -(P.alpha + 2.0)));
          return -(P.s6 * f6 / r6 + P.s8 * hp(q) * f8 / r8);
        };
        double T = 0.0;
        for (std::size_t g = 0; g < gx.size(); ++g) {
          if (w > 0) {
            const double r = rc - w + 0.5 * w * (gx[g] + 1.0);
            double t, dt;
            taper(r, rc, w, t, dt);
            T += 0.5 * w * gw[g] * (1.0 - t) * e(r) * 4.0 * kPi * r * r;
          }
          const double x = 0.5 * (gx[g] + 1.0), r = rc / x;
          T += 0.5 * gw[g] * e(r) * 4.0 * kPi * rc * rc * rc / (x * x * x * x);
        }
        ht(std::size_t(i) * kZ + j) = (i && j) ? T : 0.0;
      }
    Kokkos::deep_copy(tailtab, ht);
    auto hl = Kokkos::create_mirror_view(lr0 = DV1("d3:lr0", std::size_t(kZ) * kZ));
    for (int i = 0; i < kZ * kZ; ++i) hl(i) = hr0(i) > 0 ? std::log(4.0 / 3.0 * hr0(i)) : 0.0;
    Kokkos::deep_copy(lr0, hl);
    if (s.precision != D3Precision::Double) {
      c6f = to_float(c6, "d3:c6f"), pairtabf = to_float(pairtab, "d3:pairtabf");
      rcovf = to_float(rcov, "d3:rcovf"), lr0f = to_float(lr0, "d3:lr0f");
      cnreff = to_float(cnref, "d3:cnreff"), tailtabf = to_float(tailtab, "d3:tailtabf");
    }
    Kokkos::deep_copy(overflow, 0);
  }

  double longest() const {
    const D3Cutoffs& c = s.cutoffs;
    return std::max({c.pair, c.cn, s.params.s9 != 0.0 ? c.atm : 0.0});
  }
  double bin_width() const {
    if (const char* e = std::getenv("PET_D3_BIN"))
      if (std::atof(e) > 0) return std::atof(e);
    return kBinFraction * longest();
  }

  // The grid: each structure's frame and bins, every atom binned and sorted.
  struct Grid {
    TV2<double> frame, width, posw;
    IView2D nc;
    IView1D goff, abin, bstart, batom;
  };
  Grid grid(const D3::Atoms& at, bool keep) {
    const int N = at.N, B = at.B;
    Grid g;
    g.frame = wsv2<double>(ws, "d3:frame", B, 12);
    g.width = wsv2<double>(ws, "d3:width", B, 3);
    g.nc = ws.i2("d3:nc", B, 3);
    g.goff = ws.i1("d3:goff", B + 1);
    TV2<double> finv = wsv2<double>(ws, "d3:finv", B, 9);
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
      TV2<double> bb = wsv2<double>(ws, "d3:bbox", B, 6);
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
      sized = true, sized_N = N, sized_B = B, ++resizes;
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
    g.posw = wsv2<double>(ws, "d3:posw", N, 3);
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

  // The ATM pass: the neighbour lists' capacity from the most crowded atom (a
  // count and a read back), kept with the grid for a stepping engine.
  int atm_cap = 0;
  template <bool F, class Kn>
  void atm_pass(Kn& k, int N, bool keep) {
    if (!keep || atm_cap == 0) {
      Kokkos::parallel_for("d3_atm_count",
                           Teams<TagCount>(N, 1, kLanes).set_scratch_size(0, Kokkos::PerTeam(Kn::template walk_scratch<1>())),
                           k);
      double most = 0.0;
      auto cnt = k.dedcns;
      Kokkos::parallel_reduce(
          "d3_atm_most", RangePolicy(0, N), KOKKOS_LAMBDA(int q, double& m) { m = Kokkos::fmax(m, double(cnt(q))); },
          Kokkos::Max<double>(most));
      atm_cap = std::max(32, int(most) + int(most) / 8 + 8);
      if (const char* e = std::getenv("PET_D3_ATM_CAP"))  // tests: force the walk
        if (std::atoi(e) > 0) atm_cap = std::atoi(e);
    }
    k.atm_cap = atm_cap;
    Kokkos::parallel_for(F ? "d3_atm" : "d3_atm_energy",
                         Teams<TagATM<F>>(N, 1, kLanes).set_scratch_size(1, Kokkos::PerTeam(Kn::atm_scratch(atm_cap))),
                         k);
  }

  // Each periodic structure's elements and their summed reference weights, for
  // the tail. A warp per (structure, element), its lanes over the structure's
  // slots: the sums in a fixed order.
  template <class Kn>
  void tail_sums(Kn& k, int B) {
    using A = typename std::remove_reference_t<decltype(k.wsum)>::non_const_value_type;
    k.present = ws.i2("d3:present", B, kZ);
    k.wsum = wsv2<A>(ws, "d3:wsum", B, kZ * kR);
    auto present = k.present;
    auto wsum = k.wsum;
    auto zs = k.zs, sid = k.sid, batom = k.batom;
    auto gws = k.gws;
    auto soff = k.soff, scnt = k.scnt;
    Kokkos::deep_copy(ExecSpace(), present, 0);
    const int N = zs.extent(0);
    // Every atom flags its element: the same value from every writer, so no atomics.
    Kokkos::parallel_for(
        "d3_present", RangePolicy(0, N), KOKKOS_LAMBDA(int q) { present(sid(batom(q)), zs(q)) = 1; });
    Kokkos::parallel_for(
        "d3_wsum", TeamPolicy(B * kZ, 1, kLanes), KOKKOS_LAMBDA(const TeamPolicy::member_type& m) {
          const int b = m.league_rank() / kZ, z = m.league_rank() % kZ;
          const int q0 = soff(b), q1 = soff(b) + scnt(b);
          for (int a = 0; a < kR; ++a) {
            A x = A(0);
            if (present(b, z))
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(m, kLanes),
                  [&](int l, A& y) {
                    for (int q = q0 + l; q < q1; q += kLanes)
                      if (zs(q) == z) y += gws(q, a);
                  },
                  x);
            Kokkos::single(Kokkos::PerTeam(m), [&] { wsum(b, z * kR + a) = x; });
          }
        });
  }

  D3::Out evaluate(const D3::Atoms& at, bool forces, bool keep) {
    switch (s.precision) {
      case D3Precision::Float: return run<float, float>(at, forces, keep);
      case D3Precision::Mixed: return run<float, double>(at, forces, keep);
      default: return run<double, double>(at, forces, keep);
    }
  }

  template <class P, class A>
  D3::Out run(const D3::Atoms& at, bool forces, bool keep) {
    const int N = at.N, B = at.B;
    D3::Out o;
    o.energy = wsv1<double>(ws, "d3:energy", B);
    o.per_atom = wsv1<double>(ws, "d3:eat", N);
    if (forces) {
      o.forces = wsv2<double>(ws, "d3:fat", N, 3);
      o.virial = wsv2<double>(ws, "d3:virial", B, 6);
      o.atom_virial = wsv2<double>(ws, "d3:wat", N, 6);
    }
    if (B == 0) return out = o;
    if (N == 0) {
      Kokkos::deep_copy(ExecSpace(), o.energy, 0.0);
      if (forces) Kokkos::deep_copy(ExecSpace(), o.virial, 0.0);
      return out = o;
    }
    if (at.n_local >= 0 && B != 1) throw std::invalid_argument("pet: D3 with ghosts takes one structure");
    if (!keep) Kokkos::deep_copy(ExecSpace(), overflow, 0);
    static const bool prof = [] { const char* e = std::getenv("PET_D3_PROFILE"); return e && e[0] == '1'; }();
    auto t_last = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
      if (!prof) return;
      ExecSpace().fence();
      const auto t = std::chrono::steady_clock::now();
      std::fprintf(stderr, "d3 %-10s %8.3f ms\n", what, std::chrono::duration<double, std::milli>(t - t_last).count());
      t_last = t;
    };
    lap("start");
    const Grid g = grid(at, keep);
    lap("grid");

    using Kn = Kernels<P, A>;
    Kn k;
    k.nref = nref, k.ztab = at.ztab.extent(0) > 0 ? at.ztab : identity;
    k.c6 = pick<P>(c6, c6f), k.pairtab = pick<P>(pairtab, pairtabf), k.rcov = pick<P>(rcov, rcovf);
    k.lr0 = pick<P>(lr0, lr0f);
    k.c6a = pick<A>(c6, c6f), k.cnref = pick<A>(cnref, cnreff), k.tailtab = pick<A>(tailtab, tailtabf);
    k.soff = at.soff, k.scnt = at.scnt, k.scell = at.scell;
    k.tail = s.cutoffs.tail && at.n_local < 0;
    k.type = at.type, k.sid = at.sid, k.sper = at.sper;
    k.frame = as<A>(ws, "d3:frame_a", g.frame), k.width = as<A>(ws, "d3:width_a", g.width);
    k.nc = g.nc, k.goff = g.goff, k.abin = g.abin, k.bstart = g.bstart, k.batom = g.batom, k.posw = g.posw;
    k.xs = wsv2<A>(ws, "d3:xs", N, 3), k.zs = ws.i1("d3:zs", N);
    if constexpr (Kn::kPrefilter) k.xf = wsv2<float>(ws, "d3:xf", N, 3);
    k.gws = wsv2<A>(ws, "d3:gws", N, kR), k.dgws = wsv2<A>(ws, "d3:dgws", N, kR);
    if constexpr (std::is_same_v<P, A>) k.gwp = k.gws, k.dgwp = k.dgws;
    else k.gwp = wsv2<P>(ws, "d3:gwp", N, kR), k.dgwp = wsv2<P>(ws, "d3:dgwp", N, kR);
    k.dedcns = wsv1<A>(ws, "d3:dedcns", N), k.cn = wsv1<A>(ws, "d3:cn", N), k.dedcn = wsv1<A>(ws, "d3:dedcn", N);
    // Per-atom results in A: the outputs themselves when A is double.
    if constexpr (std::is_same_v<A, double>) {
      k.eat = o.per_atom;
      if (forces) k.fat = o.forces, k.wat = o.atom_virial;
    } else {
      k.eat = wsv1<A>(ws, "d3:eat_a", N);
      if (forces) k.fat = wsv2<A>(ws, "d3:fat_a", N, 3), k.wat = wsv2<A>(ws, "d3:wat_a", N, 6);
    }
    k.n_local = at.n_local;
    const D3Params& p = s.params;
    const D3Cutoffs& c = s.cutoffs;
    k.bj = p.damping == D3Damping::BJ;
    k.s6 = p.s6, k.s8 = p.s8, k.a1 = p.a1, k.a2 = p.a2, k.rs6 = p.rs6, k.rs8 = p.rs8, k.alp = p.alpha, k.s9 = p.s9;
    k.rp = c.pair, k.wp = c.pair_taper, k.rc = c.cn, k.wc = c.cn_taper, k.ra = c.atm, k.wa = c.atm_taper;
    k.typed();

    Kokkos::parallel_for("d3_gather", Range<TagGather>(0, N), k);
    auto teams = [&](auto tag, std::size_t scratch) {
      using Tag = decltype(tag);
      return Teams<Tag>(N, 1, kLanes).set_scratch_size(0, Kokkos::PerTeam(scratch));
    };
    Kokkos::parallel_for("d3_cn", teams(TagCN{}, Kn::template walk_scratch<1>()), k);
    lap("cn");
    if (at.exchange) forward(at.exchange, k.cn);
    Kokkos::parallel_for("d3_weights", Range<TagWeights>(0, N), k);
    if (k.tail) tail_sums(k, B);
    lap("weights");
    const bool atm = p.s9 != 0.0;
    if (forces) {
      Kokkos::parallel_for("d3_pair", teams(TagPair<true>{}, Kn::template walk_scratch<11>()), k);
      lap("pair");
      if (atm) atm_pass<true>(k, N, keep);
      lap("atm");
      if (k.tail) Kokkos::parallel_for("d3_tail", Range<TagTail<true>>(0, N), k);
      if (at.exchange) reverse(at.exchange, k.dedcn);
      auto dedcn = k.dedcn, dedcns = k.dedcns;
      auto batom = g.batom;
      Kokkos::parallel_for(
          "d3_dedcn_slots", RangePolicy(0, N), KOKKOS_LAMBDA(int q) { dedcns(q) = dedcn(batom(q)); });
      Kokkos::parallel_for("d3_chain", teams(TagChain{}, Kn::template walk_scratch<9>()), k);
      lap("chain");
    } else {
      Kokkos::parallel_for("d3_pair_energy", teams(TagPair<false>{}, Kn::template walk_scratch<1>()), k);
      if (atm) atm_pass<false>(k, N, keep);
      if (k.tail) Kokkos::parallel_for("d3_tail_energy", Range<TagTail<false>>(0, N), k);
    }
    // The results in double: each atom's (copied when A is not double), and
    // each structure's sums of them, a warp per (structure, column) in a fixed
    // order, in double.
    if constexpr (!std::is_same_v<A, double>) {
      auto ea = k.eat;
      auto fa = k.fat, wa = k.wat;
      auto eo = o.per_atom;
      auto fo = o.forces, wo = o.atom_virial;
      Kokkos::parallel_for(
          "d3_out", RangePolicy(0, N), KOKKOS_LAMBDA(int i) {
            eo(i) = double(ea(i));
            if (forces) {
              for (int x = 0; x < 3; ++x) fo(i, x) = double(fa(i, x));
              for (int x = 0; x < 6; ++x) wo(i, x) = double(wa(i, x));
            }
          });
    }
    auto soff = at.soff, scnt = at.scnt;
    auto eat = o.per_atom, en = o.energy;
    auto wat = o.atom_virial, vir = o.virial;
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
    lap("sums");
    return out = o;
  }

  // The exchange moves Real views; D3 keeps A ones.
  template <class To, class From>
  static void copy(const To& to, const From& from) {
    using T = typename To::non_const_value_type;
    Kokkos::parallel_for(
        "d3_xfer", RangePolicy(0, int(from.extent(0))), KOKKOS_LAMBDA(int i) { to(i) = T(from(i)); });
  }
  template <class V>
  void forward(Exchange* ex, const V& a) {
    if constexpr (std::is_same_v<V, RView1D>) {
      ex->atoms_forward(a);
    } else {
      RView1D t = ws.r1("d3:xfer", a.extent(0));
      copy(t, a);
      ex->atoms_forward(t);
      copy(a, t);
    }
  }
  template <class V>
  void reverse(Exchange* ex, const V& a) {
    if constexpr (std::is_same_v<V, RView1D>) {
      ex->atoms_reverse(a);
    } else {
      RView1D t = ws.r1("d3:xfer", a.extent(0));
      copy(t, a);
      ex->atoms_reverse(t);
      copy(a, t);
    }
  }
};

}  // namespace d3detail

struct D3::Impl : d3detail::Core {
  using Core::Core;
};

D3::D3(D3Settings settings) {
  if (!Kokkos::is_initialized()) throw std::runtime_error("pet::D3: Kokkos must be initialized first");
  impl_ = std::make_unique<Impl>(settings);
}
D3::~D3() = default;
D3::D3(D3&&) noexcept = default;
D3& D3::operator=(D3&&) noexcept = default;

const D3Settings& D3::settings() const { return impl_->s; }
bool D3::overflowed() const {
  int o = 0;
  Kokkos::deep_copy(o, Kokkos::subview(impl_->overflow, 0));
  return o != 0;
}
IView1D D3::overflow_flag() const { return impl_->overflow; }
void D3::reset_grid() const {
  impl_->sized = false;
  impl_->atm_cap = 0;
  Kokkos::deep_copy(impl_->overflow, 0);
}
std::size_t D3::generation() const { return impl_->ws.generation() * 1000003u + impl_->resizes; }

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
  for (const auto& s : systems)
    for (int z : s.atomic_numbers)
      if (z < 1 || z >= kZ) throw std::invalid_argument("pet: D3 covers elements 1-103, not Z = " + std::to_string(z));
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
