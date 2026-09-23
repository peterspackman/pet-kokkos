// Dense GEMM for PET, dispatched to the vendor BLAS of the active Kokkos backend
// (cuBLAS on CUDA, rocBLAS on HIP) with a portable Kokkos fallback for host
// builds. This is the ONLY linear-algebra primitive PET needs, so going straight
// to the vendor library lets us drop the heavy KokkosKernels dependency (which we
// pulled in solely for KokkosBlas::gemm, and which only wrapped these same vendor
// calls when its TPL was enabled).
//
// Semantics: C[m,n] = alpha * op(A) * op(B) + beta * C, with A/B/C row-major
// (LayoutRight) `Net` Views. transA/transB are 'N' or 'T'. An optional `bias`
// (length n) is added to every row of C: inside the GEMM's epilogue where the
// vendor library can do that (cuBLASLt, fp32), as a second pass otherwise.
//
// Mapping to column-major BLAS: a row-major [r,c] matrix is the column-major
// matrix [c,r] with leading dim c. Computing the column-major transpose
// C^T = op(B)^T op(A)^T means passing B then A with their op flags, swapping
// (m,n), and using the row-major column counts as the leading dims.
#pragma once

#include "pet/kokkos.hpp"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <tuple>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublasLt.h>
#include <cublas_v2.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <rocblas/rocblas.h>
#endif

namespace pet {

// TF32 tensor-core GEMMs for the fp32 (Net=float) path: ~10-bit mantissa
// truncation on the inputs, fp32 accumulate. The PET GEMMs are tall-skinny and
// memory-bound, so the win is modest -- but it CHANGES THE ANSWER, and not
// subtly: measured on the 8-atom pet-mad-xs crystal golden, enabling it moves
// the total energy by 9.4 meV (1.2 meV/atom), which is ~2800x the fp32 noise
// floor the same golden otherwise sits at. So it is off by default, and it must
// stay off for anything compared against a reference or checked for
// determinism.
//
// Process-global, not per-Calculator, because the cuBLAS handle below is a
// process-wide singleton and its math mode is fixed when it is created. That
// means this must be set BEFORE the first GEMM; pet::Calculator applies its
// Options::allow_tf32 at construction, which is early enough.
// PET_TF32=1 in the environment sets the initial value.
inline bool& tf32_flag() {
  static bool enabled = [] {
    const char* e = std::getenv("PET_TF32");
    return e && e[0] == '1';
  }();
  return enabled;
}
inline bool tf32_enabled() { return tf32_flag(); }
inline void set_tf32(bool on) { tf32_flag() = on; }

#if defined(KOKKOS_ENABLE_CUDA)
inline cublasHandle_t blas_handle() {
  // Leaked-at-exit singleton bound to Kokkos's default stream, so every GEMM is
  // ordered with the surrounding Kokkos kernels without explicit fences. Not
  // destroyed (avoids static-destruction-vs-Kokkos::finalize ordering issues).
  static cublasHandle_t h = [] {
    cublasHandle_t hh;
    cublasCreate(&hh);
    cublasSetStream(hh, Kokkos::DefaultExecutionSpace().cuda_stream());
    // Read once, here: the math mode is a property of the handle, and the
    // handle is created once. See tf32_flag() above.
    if (tf32_enabled()) cublasSetMathMode(hh, CUBLAS_TF32_TENSOR_OP_MATH);
    return hh;
  }();
  return h;
}
// Scalar-typed overloads so only the call matching `Net` is compiled (a non-
// template `if constexpr` would still type-check the other branch under nvcc).
inline void vendor_gemm(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, float al,
                        const float* A, int lda, const float* B, int ldb, float be, float* C,
                        int ldc) {
  cublasSgemm(blas_handle(), oa, ob, m, n, k, &al, A, lda, B, ldb, &be, C, ldc);
}
inline void vendor_gemm(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, double al,
                        const double* A, int lda, const double* B, int ldb, double be, double* C,
                        int ldc) {
  cublasDgemm(blas_handle(), oa, ob, m, n, k, &al, A, lda, B, ldb, &be, C, ldc);
}
#elif defined(KOKKOS_ENABLE_HIP)
inline rocblas_handle blas_handle() {
  static rocblas_handle h = [] {
    rocblas_handle hh;
    rocblas_create_handle(&hh);
    rocblas_set_stream(hh, Kokkos::DefaultExecutionSpace().hip_stream());
    return hh;
  }();
  return h;
}
inline void vendor_gemm(rocblas_operation oa, rocblas_operation ob, int m, int n, int k, float al,
                        const float* A, int lda, const float* B, int ldb, float be, float* C,
                        int ldc) {
  rocblas_sgemm(blas_handle(), oa, ob, m, n, k, &al, A, lda, B, ldb, &be, C, ldc);
}
inline void vendor_gemm(rocblas_operation oa, rocblas_operation ob, int m, int n, int k, double al,
                        const double* A, int lda, const double* B, int ldb, double be, double* C,
                        int ldc) {
  rocblas_dgemm(blas_handle(), oa, ob, m, n, k, &al, A, lda, B, ldb, &be, C, ldc);
}
#endif

// C(i, j) += b(j): the bias pass for every path that cannot fuse it.
inline void add_bias(const View2D& C, const View1D& b) {
  const int n = C.extent(1);
  Kokkos::parallel_for(
      "bias", Kokkos::RangePolicy<ExecSpace>(0, C.extent(0) * n),
      KOKKOS_LAMBDA(int i) { C(i / n, i % n) += b(i % n); });
}

#if defined(KOKKOS_ENABLE_CUDA)
// cuBLASLt, used only for what plain cuBLAS cannot do: add the bias in the GEMM
// epilogue rather than in a second full pass over C (which was ~35 extra
// kernels, and ~6% of an evaluation, per call). A plan -- descriptors plus the
// heuristic's algorithm -- is cached per shape: the heuristic query alone costs
// tens of microseconds, and a fixed choice keeps repeated runs bit-identical.
// Arguments are already in column-major (cuBLAS) order.
inline void lt_gemm_bias(bool ta, bool tb, int m, int n, int k, float alpha, const float* A,
                         int lda, const float* B, int ldb, float beta, float* C, int ldc,
                         const float* bias) {
  struct Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
  };
  constexpr std::size_t kWs = 32u << 20;
  static cublasLtHandle_t lt = [] { cublasLtHandle_t h; cublasLtCreate(&h); return h; }();
  static void* ws = [] { void* p = nullptr; cudaMalloc(&p, kWs); return p; }();  // leaked, like the handle
  static std::map<std::tuple<bool, bool, int, int, int, int, int, int>, Plan> plans;
  auto [it, fresh] = plans.try_emplace({ta, tb, m, n, k, lda, ldb, ldc});
  Plan& p = it->second;
  if (fresh) {
    const auto opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N, opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescCreate(&p.op, tf32_enabled() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                             CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof opA);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof opB);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_32F, ta ? k : m, ta ? m : k, lda);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_32F, tb ? n : k, tb ? k : n, ldb);
    cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F, m, n, ldc);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWs,
                                         sizeof kWs);
    cublasLtMatmulHeuristicResult_t r{};
    int found = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.a, p.b, p.c, p.c, pref, 1, &r, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (!found) throw std::runtime_error("pet: cuBLASLt found no algorithm for a bias GEMM");
    p.algo = r.algo;
  }
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
  cublasLtMatmul(lt, p.op, &alpha, A, p.a, B, p.b, &beta, C, p.c, C, p.c, &p.algo, ws, kWs,
                 ExecSpace().cuda_stream());
}
#endif

inline void gemm(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                 const View2D& C, const View1D& bias = {}) {
  const int m = C.extent(0), n = C.extent(1);
  const bool ta = (transA == 'T' || transA == 't');
  const bool tb = (transB == 'T' || transB == 't');
  const int k = ta ? A.extent(0) : A.extent(1);
  const bool has_bias = bias.extent(0) > 0;
  // row-major leading dims (unused by the portable fallback, which indexes directly)
  [[maybe_unused]] const int lda = A.extent(1), ldb = B.extent(1), ldc = C.extent(1);

#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Net, float>) {
    if (has_bias)
      return lt_gemm_bias(tb, ta, n, m, k, alpha, B.data(), ldb, A.data(), lda, beta, C.data(),
                          ldc, bias.data());
  }
  vendor_gemm(tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, alpha,
              B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
  if (has_bias) add_bias(C, bias);
#elif defined(KOKKOS_ENABLE_HIP)
  vendor_gemm(tb ? rocblas_operation_transpose : rocblas_operation_none,
              ta ? rocblas_operation_transpose : rocblas_operation_none, n, m, k, alpha, B.data(),
              ldb, A.data(), lda, beta, C.data(), ldc);
  if (has_bias) add_bias(C, bias);
#else
  // portable host/fallback gemm (Serial/OpenMP test builds): naive but correct.
  Kokkos::parallel_for(
      "gemm_naive", Kokkos::RangePolicy<ExecSpace>(0, m * n), KOKKOS_LAMBDA(int idx) {
        const int i = idx / n, j = idx % n;
        Net acc = Net(0);
        for (int p = 0; p < k; ++p) {
          const Net a = ta ? A(p, i) : A(i, p);
          const Net b = tb ? B(j, p) : B(p, j);
          acc += a * b;
        }
        C(i, j) = alpha * acc + beta * C(i, j) + (has_bias ? bias(j) : Net(0));
      });
#endif
}

}  // namespace pet
