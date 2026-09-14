// Dense GEMM for PET, dispatched to the vendor BLAS of the active Kokkos backend
// (cuBLAS on CUDA, rocBLAS on HIP) with a portable Kokkos fallback for host
// builds. This is the ONLY linear-algebra primitive PET needs, so going straight
// to the vendor library lets us drop the heavy KokkosKernels dependency (which we
// pulled in solely for KokkosBlas::gemm, and which only wrapped these same vendor
// calls when its TPL was enabled).
//
// Semantics: C[m,n] = alpha * op(A) * op(B) + beta * C, with A/B/C row-major
// (LayoutRight) `Net` Views. transA/transB are 'N' or 'T'.
//
// Mapping to column-major BLAS: a row-major [r,c] matrix is the column-major
// matrix [c,r] with leading dim c. Computing the column-major transpose
// C^T = op(B)^T op(A)^T means passing B then A with their op flags, swapping
// (m,n), and using the row-major column counts as the leading dims.
#pragma once

#include "pet/kokkos.hpp"

#include <cstdlib>

#if defined(KOKKOS_ENABLE_CUDA)
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

inline void gemm(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                 const View2D& C) {
  const int m = C.extent(0), n = C.extent(1);
  const bool ta = (transA == 'T' || transA == 't');
  const bool tb = (transB == 'T' || transB == 't');
  const int k = ta ? A.extent(0) : A.extent(1);
  // row-major leading dims (unused by the portable fallback, which indexes directly)
  [[maybe_unused]] const int lda = A.extent(1), ldb = B.extent(1), ldc = C.extent(1);

#if defined(KOKKOS_ENABLE_CUDA)
  vendor_gemm(tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, alpha,
              B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
#elif defined(KOKKOS_ENABLE_HIP)
  vendor_gemm(tb ? rocblas_operation_transpose : rocblas_operation_none,
              ta ? rocblas_operation_transpose : rocblas_operation_none, n, m, k, alpha, B.data(),
              ldb, A.data(), lda, beta, C.data(), ldc);
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
        C(i, j) = alpha * acc + beta * C(i, j);
      });
#endif
}

}  // namespace pet
