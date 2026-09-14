#include "pet/ozaki.hpp"

#include "pet/gemm.hpp"

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <stdexcept>

namespace pet {

namespace {
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

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
  // Only worth doing when the alternative is fp64 arithmetic. In mixed or fp32
  // mode the network is already running at the speed Ozaki would be buying, and
  // the extra slice GEMMs would be a pure loss.
  return sizeof(Net) == sizeof(double);
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------
//
// Each row (or column) is normalised by a power of two so every entry lands in
// [-1/2, 1/2], then peeled into 7-bit signed digits:
//
//     r = sum_{t=0}^{s-1} d_t * 2^(-7(t+1)) + residual,  |residual| <= 2^(-7s-1)
//
// The half-scale normalisation is not cosmetic. At |r| <= 1 the first digit
// would be round(r * 128) = 128, which does not fit int8, and clamping it
// cascades into every later digit. Starting at |r| <= 1/2 bounds every digit by
// 64, so |d_t * d_v| <= 4096 and a k-term sum stays far inside int32 for any k
// this network produces -- which is what makes the accumulation exact, and the
// exactness is the whole point.

namespace {

// 2^ceil(log2(mu)) * 2, i.e. the smallest power of two strictly greater than
// 2*mu, so that mu / scale <= 1/2. Zero rows get scale 0 and split to nothing.
KOKKOS_INLINE_FUNCTION double norm_scale(double mu) {
  if (!(mu > 0.0)) return 0.0;
  // The smallest power of two >= mu, doubled, so that mu / scale <= 1/2.
  //
  // A one-ulp error in log2 can push ceil() up by one, which costs a single bit
  // of the mantissa and nothing else -- the split simply starts one binade
  // lower. Erring the other way would be the problem (an entry above 1/2 would
  // make the first digit overflow int8), and ceil cannot round down.
  const int e = (int) Kokkos::ceil(Kokkos::log2(mu));
  return Kokkos::pow(2.0, (double) (e + 1));
}

void split_into(Workspace* ws, const std::string& key, const View2D& src, int n_slices,
                bool by_column, bool uniform, OzakiSplit& out) {
  const int R = (int) src.extent(0), C = (int) src.extent(1);
  const int lines = uniform ? 1 : (by_column ? C : R);
  out.rows = R;
  out.cols = C;
  out.n_slices = n_slices;
  out.uniform = uniform;

  // Per-line (or matrix-wide) magnitude, then the normalising power of two.
  RView1D scale = ws ? ws->r1(key + ":oscale", lines) : RView1D(key + ":oscale", lines);
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
  } else {
    Kokkos::parallel_for(
        key + ":omax_r", RangePolicy(0, R), KOKKOS_LAMBDA(int r) {
          double mu = 0.0;
          for (int c = 0; c < C; ++c) mu = Kokkos::max(mu, Kokkos::fabs((double) src(r, c)));
          scale(r) = norm_scale(mu);
        });
  }
  out.scale = scale;

  // Activation slices come from the pool (reused every call); a weight's are
  // owned, because they are computed once at load and kept for the model's life.
  out.slices.resize(n_slices);
  for (int t = 0; t < n_slices; ++t) {
    const std::string sk = key + ":osl" + std::to_string(t);
    out.slices[t] = ws ? ws->i8(sk, R, C) : I8View2D(sk, R, C);
  }

  // Peel the digits. One thread per entry, walking the whole chain, so the
  // residual stays in a register rather than round-tripping through memory
  // n_slices times.
  const int S = n_slices;
  // Copy the slice handles into a fixed-size array the device lambda can hold;
  // a std::vector cannot cross into a kernel.
  I8View2D sl[kOzakiMaxSlices];
  for (int t = 0; t < S; ++t) sl[t] = out.slices[t];

  Kokkos::parallel_for(
      key + ":osplit", RangePolicy(0, R * C), KOKKOS_LAMBDA(int idx) {
        const int r = idx / C, c = idx % C;
        const double s = scale(uniform ? 0 : (by_column ? c : r));
        double u = (s > 0.0) ? ((double) src(r, c) / s) : 0.0;  // |u| <= 1/2
        for (int t = 0; t < S; ++t) {
          const double v = u * 128.0;  // 2^kOzakiSliceBits
          double d = Kokkos::round(v);
          // Defensive: the normalisation bounds this by 64, but a denormal or a
          // NaN slipping through must not become an out-of-range int8 store.
          d = Kokkos::fmin(Kokkos::fmax(d, -127.0), 127.0);
          sl[t](r, c) = (int8_t) d;
          u = v - d;
        }
      });
}

}  // namespace

OzakiSplit ozaki_split(Workspace& ws, const std::string& key, const View2D& src, int n_slices,
                       bool by_column) {
  OzakiSplit out;
  split_into(&ws, key, src, n_slices, by_column, /*uniform=*/false, out);
  return out;
}

OzakiSplit ozaki_split_weight(const View2D& src, int n_slices) {
  OzakiSplit out;
  // by_column is irrelevant under a uniform scale; pass false for definiteness.
  split_into(nullptr, "w", src, n_slices, /*by_column=*/false, /*uniform=*/true, out);
  return out;
}

// ---------------------------------------------------------------------------
// The slice-grid product
// ---------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
namespace {
// int8 x int8 -> int32, on tensor cores. beta = 1 accumulates, and because the
// accumulator is integer that accumulation is exact -- no ordering concern, and
// no reason to keep separate buffers per slice pair.
void igemm(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, const int8_t* A,
           int lda, const int8_t* B, int ldb, int32_t beta, int32_t* C, int ldc) {
  const int32_t alpha = 1;
  const cublasStatus_t st = cublasGemmEx(
      blas_handle(), oa, ob, m, n, k, &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, &beta, C,
      CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("pet: cublasGemmEx (int8) failed with status " +
                             std::to_string((int) st));
}
}  // namespace
#endif

void gemm_ozaki(Workspace& ws, const std::string& key, char transA, char transB, Net alpha,
                const View2D& A, const View2D& B, Net beta, const View2D& C,
                const OzakiSplit* bsplit) {
  const OzakiConfig& cfg = ozaki_config();
  if (!ozaki_active()) {
    gemm(transA, transB, alpha, A, B, beta, C);
    return;
  }

#if defined(KOKKOS_ENABLE_CUDA)
  const bool ta = (transA == 'T' || transA == 't');
  const bool tb = (transB == 'T' || transB == 't');
  const int m = (int) C.extent(0), n = (int) C.extent(1);
  const int k = ta ? (int) A.extent(0) : (int) A.extent(1);
  const int S = cfg.slices;

  // A is normalised along the direction the inner product sums over: by row when
  // it enters untransposed, by column when transposed. Same for B, mirrored.
  // Getting this backwards still produces a number, just a much less accurate
  // one, because the digits of the two operands would be scaled inconsistently
  // across the summation index.
  OzakiSplit asp = ozaki_split(ws, key + ":a", A, S, /*by_column=*/ta);
  OzakiSplit bown;
  const OzakiSplit* bs = bsplit;
  if (!bs || !bs->valid() || bs->n_slices != S) {
    bown = ozaki_split(ws, key + ":b", B, S, /*by_column=*/!tb);
    bs = &bown;
  }

  // Pooled: this is [m, n] int32, which at the widths the heavy models use is
  // hundreds of megabytes. Allocating it per GEMM would reintroduce exactly the
  // device-synchronising cudaMalloc/cudaFree churn the Workspace exists to
  // avoid.
  IView2D G = ws.i2(key + ":og", m, n);
  static_assert(sizeof(int) == 4, "the Ozaki accumulator assumes 32-bit int");

  const int lda = (int) A.extent(1), ldb = (int) B.extent(1), ldc = n;
  auto scaleA = asp.scale;
  auto scaleB = bs->scale;
  const bool uniA = asp.uniform, uniB = bs->uniform;

  // Group the slice grid by total shift. Every (t, v) with t + v == shift shares
  // the factor 2^(-7(shift+2)), so they can all accumulate into one int32 buffer
  // -- exactly -- and be folded into C once. That turns S*(S+1)/2 separate
  // rescale-and-add passes over an [m, n] matrix into S of them, which matters
  // because those passes are pure memory traffic.
  const int max_shift = cfg.triangular ? S - 1 : 2 * (S - 1);
  for (int shift = 0; shift <= max_shift; ++shift) {
    int used = 0;
    for (int t = 0; t <= shift; ++t) {
      const int v = shift - t;
      if (t >= S || v >= S) continue;
      // cuBLAS is column-major and these Views are row-major, so the operands
      // swap and the transposes follow -- the same inversion gemm() does.
      igemm(tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k,
            bs->slices[v].data(), ldb, asp.slices[t].data(), lda, used ? 1 : 0,
            reinterpret_cast<int32_t*>(G.data()), ldc);
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
          // beta is applied on the first shift only; later ones accumulate onto
          // what this loop has already written.
          C(i, j) = first ? (Net)(term + be * (double) C(i, j)) : (Net)((double) C(i, j) + term);
        });
  }
#else
  gemm(transA, transB, alpha, A, B, beta, C);
#endif
}

}  // namespace pet
