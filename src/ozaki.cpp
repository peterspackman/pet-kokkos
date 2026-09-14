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

OzakiSplit ozaki_split_weight(const View2D& src, int n_slices, bool transposed) {
  OzakiSplit out;
  if (!transposed) {
    // by_column is irrelevant under a uniform scale; pass false for definiteness.
    split_into(nullptr, "w", src, n_slices, /*by_column=*/false, /*uniform=*/true, out);
    return out;
  }
  // Materialise src^T and split that. Transposing the fp64 weight and splitting
  // is equivalent to transposing the slices -- the scale is matrix-wide, so the
  // digits do not depend on position -- and it is far less code than a
  // slice-wise transpose. The temporary lives only for this call, at load.
  const int R = (int) src.extent(0), C = (int) src.extent(1);
  View2D srcT("wT", C, R);
  Kokkos::parallel_for(
      "oz_wT", Kokkos::RangePolicy<ExecSpace>(0, R * C),
      KOKKOS_LAMBDA(int idx) { srcT(idx % C, idx / C) = src(idx / C, idx % C); });
  split_into(nullptr, "wT", srcT, n_slices, /*by_column=*/false, /*uniform=*/true, out);
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
//
// Returns the status rather than throwing, because CUBLAS_STATUS_NOT_SUPPORTED
// is an expected answer for some shapes and the caller falls back rather than
// failing. See gemm_ozaki.
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
  // The shape was probed before the loop, so an unsupported one has already
  // fallen back; anything failing here is a genuine error.
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("pet: cublasGemmEx (int8) failed with status " +
                             std::to_string((int) st) + " (m=" + std::to_string(m) +
                             " n=" + std::to_string(n) + " k=" + std::to_string(k) + ")");
}
}  // namespace
#endif

namespace {
// Scratch for the slice grid. Private and keyed fixed, because only one
// gemm_ozaki is ever in flight -- see the header. Pooled rather than allocated
// per call for the usual reason: a cudaMalloc/cudaFree pair per GEMM would
// synchronise the device on every layer of every evaluation.
Workspace& ozaki_ws() {
  // Leaked at exit, deliberately, exactly as blas_handle() is: a function-local
  // static is destroyed during static destruction, which runs AFTER
  // Kokkos::finalize(), and Kokkos rejects a View freed at that point. Leaking
  // it is the standard way out and costs nothing -- the process is ending.
  static Workspace* ws = new Workspace();
  return *ws;
}
}  // namespace

std::size_t ozaki_workspace_bytes() { return ozaki_ws().capacity_bytes(); }

void gemm_ozaki(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                const View2D& C, const OzakiSplit* bsplit) {
  const OzakiConfig& cfg = ozaki_config();
  if (!ozaki_active()) {
    gemm(transA, transB, alpha, A, B, beta, C);
    return;
  }
  Workspace& ws = ozaki_ws();
  const std::string key = "oz";

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
  // B is handed to cuBLAS as the FIRST operand (it computes C^T = B^T A^T for our
  // row-major data), and int8 tensor cores need that operand transposed -- the
  // contraction index has to be contiguous. When the caller already wants B
  // transposed the stored layout is right; when it does not, the transposed
  // decomposition is the one to use, and the GEMM is issued as OP_T either way.
  OzakiSplit bown;
  const OzakiSplit* bs = bsplit;
  if (!bs || !bs->valid() || bs->n_slices != S) {
    // No pre-split supplied: decompose here, in whichever layout the int8 GEMM
    // can take. This is the fallback; a model's weights arrive pre-split.
    bown = ozaki_split_weight(B, S, /*transposed=*/!tb);
    bs = &bown;
  }
  const int ldb_eff = tb ? (int) B.extent(1) : (int) B.extent(0);

  // Pooled: this is [m, n] int32, which at the widths the heavy models use is
  // hundreds of megabytes. Allocating it per GEMM would reintroduce exactly the
  // device-synchronising cudaMalloc/cudaFree churn the Workspace exists to
  // avoid.
  IView2D G = ws.i2(key + ":og", m, n);
  static_assert(sizeof(int) == 4, "the Ozaki accumulator assumes 32-bit int");

  const int lda = (int) A.extent(1), ldb = (int) B.extent(1), ldc = n;

  // Not every shape can go through int8 tensor cores. The exact rule is a
  // function of the cuBLAS version and the architecture's IMMA fragment size, so
  // rather than encode one, ask: try the shape, and on NOT_SUPPORTED use the
  // vendor fp64 GEMM for this call instead.
  //
  // That fallback is always safe -- DGEMM is exact, so falling back can only
  // improve accuracy -- and it is the right answer anyway for the shapes that
  // fail. In this network they are PET's geometry embedder, the Linear(4 ->
  // d_pet) on [edge_vector, distance] that gives the architecture its name: at
  // k = 4 there is nothing for a slice decomposition to win, and a 45-GEMM grid
  // over a k = 4 contraction would be pure overhead.
  {
    const cublasStatus_t probe =
        igemm_try(CUBLAS_OP_T, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, bs->slices[0].data(),
                  ldb_eff, asp.slices[0].data(), lda, 0, reinterpret_cast<int32_t*>(G.data()), ldc);
    if (probe == CUBLAS_STATUS_NOT_SUPPORTED) {
      gemm(transA, transB, alpha, A, B, beta, C);
      return;
    }
    if (probe != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("pet: cublasGemmEx (int8) probe failed with status " +
                               std::to_string((int) probe));
  }

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
      // Always OP_T on B: its slices are stored with the contraction index
      // contiguous, whichever layout that took.
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
