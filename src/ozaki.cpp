// fp64-accurate GEMMs on integer tensor cores: see ozaki.hpp.
#include "pet/ozaki.hpp"

#include "pet/gemm.hpp"

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <map>

namespace pet {

namespace {
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using Team = TeamPolicy::member_type;

int env_int(const char* name, int fallback, int lo, int hi) {
  const char* e = std::getenv(name);
  if (!e || !*e) return fallback;
  const int v = std::atoi(e);
  return (v < lo || v > hi) ? fallback : v;
}
}  // namespace

OzakiConfig& ozaki_config() {
  static OzakiConfig cfg = [] {
    OzakiConfig c;
    if (const char* m = std::getenv("PET_GEMM_MODE"))
      c.mode = (std::string(m) == "ozaki") ? GemmMode::Ozaki : GemmMode::Native;
    c.slices = env_int("PET_OZAKI_SLICES", kOzakiMaxSlices, 1, kOzakiMaxSlices);
    return c;
  }();
  return cfg;
}

bool ozaki_available() {
#if defined(KOKKOS_ENABLE_CUDA)
  // In fp32 the native GEMM is already the speed Ozaki buys.
  return sizeof(Net) == sizeof(double);
#else
  return false;
#endif
}

// Splitting. Each line is scaled by a power of two into [-1/2, 1/2] and peeled
// into 7-bit signed digits. At 1/2 rather than 1 the first digit is at most 64,
// not 128 (which int8 cannot hold), so |d_t d_v| <= 4096 and a k-term sum stays
// far inside int32: the accumulation is exact.

namespace {

// The smallest power of two >= 2 mu, so mu / scale <= 1/2; 0 for a zero line.
// An ulp of error in log2 can only push ceil up, costing one bit.
KOKKOS_INLINE_FUNCTION double norm_scale(double mu) {
  if (!(mu > 0.0)) return 0.0;
  const int e = (int) Kokkos::ceil(Kokkos::log2(mu));
  return Kokkos::pow(2.0, (double) (e + 1));
}

// Entry (r, c), already scaled to |u| <= 1/2, into its S digits. The clamp only
// guards against a NaN or denormal making an out-of-range int8.
KOKKOS_INLINE_FUNCTION void peel(double u, const I8View2D* sl, int S, int r, int c) {
  for (int t = 0; t < S; ++t) {
    const double v = u * (1 << kOzakiSliceBits);
    const double d = Kokkos::fmin(Kokkos::fmax(Kokkos::round(v), -127.0), 127.0);
    sl[t](r, c) = (int8_t) d;
    u = v - d;
  }
}

void split_into(Workspace* ws, const std::string& key, const View2D& src, int n_slices,
                bool by_column, bool uniform, OzakiSplit& out) {
  const int R = (int) src.extent(0), C = (int) src.extent(1);
  const int lines = uniform ? 1 : (by_column ? C : R);
  out.rows = R;
  out.cols = C;
  out.n_slices = n_slices;
  out.uniform = uniform;

  RView1D scale = ws ? ws->r1(key + ":oscale", lines) : RView1D(key + ":oscale", lines);
  out.scale = scale;

  // An activation's buffers come from the pool; a weight's are its own.
  out.slices.resize(n_slices);
  for (int t = 0; t < n_slices; ++t) {
    const std::string sk = key + ":osl" + std::to_string(t);
    out.slices[t] = ws ? ws->i8(sk, R, C) : I8View2D(sk, R, C);
  }
  const int S = n_slices;
  // A fixed array, which a kernel can capture and a std::vector cannot.
  I8View2D sl[kOzakiMaxSlices];
  for (int t = 0; t < S; ++t) sl[t] = out.slices[t];

  // Per row, the hot case (every activation): one team per row finds the max
  // and peels the digits in one coalesced pass.
  if (!uniform && !by_column) {
    Kokkos::parallel_for(
        key + ":osplit_row", TeamPolicy(R, Kokkos::AUTO), KOKKOS_LAMBDA(const Team& team) {
          const int r = team.league_rank();
          double mu = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, C),
              [&](int c, double& acc) { acc = Kokkos::max(acc, Kokkos::fabs((double) src(r, c))); },
              Kokkos::Max<double>(mu));
          const double sc = norm_scale(mu);
          if (team.team_rank() == 0) scale(r) = sc;
          // The scale is a power of two, so its reciprocal is exact.
          const double inv = (sc > 0.0) ? 1.0 / sc : 0.0;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, C),
                               [&](int c) { peel((double) src(r, c) * inv, sl, S, r, c); });
        });
    return;
  }

  if (uniform) {
    double mu = 0.0;
    Kokkos::parallel_reduce(
        key + ":omax_u", RangePolicy(0, R * C),
        KOKKOS_LAMBDA(int idx, double& acc) {
          acc = Kokkos::max(acc, Kokkos::fabs((double) src(idx / C, idx % C)));
        },
        Kokkos::Max<double>(mu));
    Kokkos::parallel_for(
        key + ":oscale_u", RangePolicy(0, 1), KOKKOS_LAMBDA(int) { scale(0) = norm_scale(mu); });
  } else if (by_column) {
    Kokkos::parallel_for(
        key + ":omax_c", RangePolicy(0, C), KOKKOS_LAMBDA(int c) {
          double mu = 0.0;
          for (int r = 0; r < R; ++r) mu = Kokkos::max(mu, Kokkos::fabs((double) src(r, c)));
          scale(c) = norm_scale(mu);
        });
  }
  // One thread per entry, the residual kept in a register.
  Kokkos::parallel_for(
      key + ":osplit", RangePolicy(0, R * C), KOKKOS_LAMBDA(int idx) {
        const int r = idx / C, c = idx % C;
        const double s = scale(uniform ? 0 : (by_column ? c : r));
        peel((s > 0.0) ? ((double) src(r, c) / s) : 0.0, sl, S, r, c);
      });
}

}  // namespace

OzakiSplit ozaki_split(Workspace& ws, const std::string& key, const View2D& src, int n_slices,
                       bool by_column) {
  OzakiSplit out;
  split_into(&ws, key, src, n_slices, by_column, /*uniform=*/false, out);
  return out;
}

OzakiSplit ozaki_split_weight(const View2D& src, int n_slices, bool transposed) {
  OzakiSplit out;
  if (!transposed) {
    split_into(nullptr, "w", src, n_slices, /*by_column=*/false, /*uniform=*/true, out);
    return out;
  }
  // Under one scale, splitting src^T is transposing the slices.
  const int R = (int) src.extent(0), C = (int) src.extent(1);
  View2D srcT("wT", C, R);
  Kokkos::parallel_for(
      "oz_wT", RangePolicy(0, R * C),
      KOKKOS_LAMBDA(int idx) { srcT(idx % C, idx / C) = src(idx / C, idx % C); });
  split_into(nullptr, "wT", srcT, n_slices, /*by_column=*/false, /*uniform=*/true, out);
  return out;
}

#if defined(KOKKOS_ENABLE_CUDA)
namespace {
// int8 x int8 -> int32 on tensor cores; beta = 1 accumulates, exactly. Returns
// the status: NOT_SUPPORTED is an answer for some shapes, not an error.
cublasStatus_t igemm_try(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k,
                         const int8_t* A, int lda, const int8_t* B, int ldb, int32_t beta,
                         int32_t* C, int ldc) {
  const int32_t alpha = 1;
  return cublasGemmEx(blas_handle(), oa, ob, m, n, k, &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I,
                      ldb, &beta, C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
}

void igemm(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, const int8_t* A,
           int lda, const int8_t* B, int ldb, int32_t beta, int32_t* C, int ldc) {
  const cublasStatus_t st = igemm_try(oa, ob, m, n, k, A, lda, B, ldb, beta, C, ldc);
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("pet: cublasGemmEx (int8) failed with status " +
                             std::to_string((int) st) + " (m=" + std::to_string(m) +
                             " n=" + std::to_string(n) + " k=" + std::to_string(k) + ")");
}
}  // namespace
#endif

namespace {
// The slice grid's scratch, under fixed keys.
Workspace& ozaki_ws() {
  // Leaked, like blas_handle(): a static destroyed after Kokkos::finalize would
  // free Views too late. Every buffer is written before it is read (the first
  // GEMM of each shift group has beta = 0), so zeroing is waste.
  static Workspace* ws = [] {
    auto* w = new Workspace();
    w->set_zero(false);
    return w;
  }();
  return *ws;
}
}  // namespace

#if defined(KOKKOS_ENABLE_CUDA)
namespace {
// Shapes the ~37-GEMM grid cannot win: a tiny contraction (the geometry
// embedder's k = 4), a GEMV (the readout heads' n = 1, no tensor cores), or too
// few outputs to pay for the launches. The fallback, DGEMM, is exact.
bool ozaki_shape_worthwhile(int m, int n, int k) {
  return k >= 32 && n >= 32 && (std::size_t) m * (std::size_t) n >= 4096u;
}

// Whether cuBLAS takes int8 for this shape: asked, once per shape, since the
// rule depends on the cuBLAS version and the architecture.
bool igemm_shape_supported(int m, int n, int k, cublasOperation_t ob, const int8_t* a, int lda,
                           const int8_t* b, int ldb, int32_t* c, int ldc) {
  static std::map<std::array<int, 4>, bool> cache;
  const std::array<int, 4> key{m, n, k, (int) ob};
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  const cublasStatus_t st = igemm_try(CUBLAS_OP_T, ob, m, n, k, a, lda, b, ldb, 0, c, ldc);
  if (st != CUBLAS_STATUS_SUCCESS && st != CUBLAS_STATUS_NOT_SUPPORTED)
    throw std::runtime_error("pet: cublasGemmEx (int8) probe failed with status " +
                             std::to_string((int) st));
  const bool ok = (st == CUBLAS_STATUS_SUCCESS);
  cache.emplace(key, ok);
  return ok;
}
}  // namespace
#endif

namespace {
void gemm_ozaki_nobias(char transA, char transB, Net alpha, const View2D& A, const View2D& B,
                       Net beta, const View2D& C, const OzakiSplit* bsplit) {
  const OzakiConfig& cfg = ozaki_config();
  Workspace& ws = ozaki_ws();
  const std::string key = "oz";

#if defined(KOKKOS_ENABLE_CUDA)
  const bool ta = (transA == 'T' || transA == 't');
  const bool tb = (transB == 'T' || transB == 't');
  const int m = (int) C.extent(0), n = (int) C.extent(1);
  const int k = ta ? (int) A.extent(0) : (int) A.extent(1);
  const int S = cfg.slices;

  if (!ozaki_shape_worthwhile(m, n, k)) {
    gemm(transA, transB, alpha, A, B, beta, C);
    return;
  }

  // Each operand is scaled along the index the product sums over.
  OzakiSplit asp = ozaki_split(ws, key + ":a", A, S, /*by_column=*/ta);
  // cuBLAS computes C^T = B^T A^T, so B is its first operand, which int8 wants
  // transposed (contraction index contiguous): use the split in that layout.
  OzakiSplit bown;
  const OzakiSplit* bs = bsplit;
  if (!bs || !bs->valid() || bs->n_slices != S) {
    bown = ozaki_split_weight(B, S, /*transposed=*/!tb);
    bs = &bown;
  }
  const int ldb_eff = tb ? (int) B.extent(1) : (int) B.extent(0);

  IView2D G = ws.i2(key + ":og", m, n);
  static_assert(sizeof(int) == 4, "the Ozaki accumulator assumes 32-bit int");

  const int lda = (int) A.extent(1), ldc = n;

  if (!igemm_shape_supported(n, m, k, ta ? CUBLAS_OP_T : CUBLAS_OP_N, bs->slices[0].data(),
                             ldb_eff, asp.slices[0].data(), lda,
                             reinterpret_cast<int32_t*>(G.data()), ldc)) {
    gemm(transA, transB, alpha, A, B, beta, C);
    return;
  }

  auto scaleA = asp.scale;
  auto scaleB = bs->scale;
  const bool uniA = asp.uniform, uniB = bs->uniform;

  // Products with the same t + v share their power of two, so each shift group
  // accumulates exactly into one int32 buffer and is folded into C once.
  const int max_shift = cfg.triangular ? S - 1 : 2 * (S - 1);
  for (int shift = 0; shift <= max_shift; ++shift) {
    int used = 0;
    for (int t = 0; t <= shift; ++t) {
      const int v = shift - t;
      if (t >= S || v >= S) continue;
      igemm(CUBLAS_OP_T, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, bs->slices[v].data(), ldb_eff,
            asp.slices[t].data(), lda, used ? 1 : 0, reinterpret_cast<int32_t*>(G.data()), ldc);
      ++used;
    }
    if (!used) continue;

    // C += alpha * 2^(-7(shift+2)) * scaleA_i * scaleB_j * G
    const double w = Kokkos::pow(2.0, -(double) (kOzakiSliceBits * (shift + 2)));
    const double al = (double) alpha;
    const double be = (double) beta;
    const bool first = (shift == 0);
    Kokkos::parallel_for(
        key + ":oacc", RangePolicy(0, m * n), KOKKOS_LAMBDA(int idx) {
          const int i = idx / n, j = idx % n;
          const double term =
              al * w * scaleA(uniA ? 0 : i) * scaleB(uniB ? 0 : j) * (double) G(i, j);
          C(i, j) = first ? (Net)(term + be * (double) C(i, j)) : (Net)((double) C(i, j) + term);
        });
  }
#else
  (void) cfg, (void) ws, (void) key, (void) bsplit;
  gemm(transA, transB, alpha, A, B, beta, C);
#endif
}
}  // namespace

void gemm_ozaki(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                const View2D& C, const OzakiSplit* bsplit, const View1D& bias) {
  if (!ozaki_active()) return gemm(transA, transB, alpha, A, B, beta, C, bias);
  gemm_ozaki_nobias(transA, transB, alpha, A, B, beta, C, bsplit);
  if (bias.extent(0) > 0) add_bias(C, bias);
}

}  // namespace pet
