// Dense GEMM: C = alpha op(A) op(B) + beta C (+ bias on every row), on
// row-major Net views, through the vendor BLAS (cuBLAS, rocBLAS) or a naive
// Kokkos loop on the host. The only linear algebra PET needs.
//
// A row-major [r,c] matrix is the column-major [c,r] one, so each call computes
// C^T = op(B)^T op(A)^T: B and A swap places, and so do m and n.
#pragma once

#include "pet/kokkos.hpp"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <tuple>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublasLt.h>
#include <cublas_v2.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <rocblas/rocblas.h>
#endif

namespace pet {

// TF32 tensor cores for the fp32 GEMMs: faster, and it changes the answer (by
// ~0.1 meV/atom, a few meV/A), so it is off unless asked for. Process-global,
// because cuBLAS fixes a handle's math mode when the handle is created: set it
// before the first GEMM. PET_TF32=1 sets the initial value.
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
  // On Kokkos's stream, so GEMMs are ordered with the kernels around them. Never
  // destroyed: static destruction would run after Kokkos::finalize.
  static cublasHandle_t h = [] {
    cublasHandle_t hh;
    cublasCreate(&hh);
    cublasSetStream(hh, Kokkos::DefaultExecutionSpace().cuda_stream());
    if (tf32_enabled()) cublasSetMathMode(hh, CUBLAS_TF32_TENSOR_OP_MATH);
    return hh;
  }();
  return h;
}
// Overloads by scalar type, not `if constexpr`: nvcc type-checks the discarded
// branch of one in a non-template function.
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
// The bias in cuBLASLt's GEMM epilogue, instead of a second pass over C.
// Arguments in column-major order. A plan (descriptors and the heuristic's
// algorithm) is cached per shape: the heuristic is slow, and a fixed choice keeps
// repeated runs bit-identical. Returns false for double, which it does not do.
inline bool lt_gemm_bias(bool ta, bool tb, int m, int n, int k, float alpha, const float* A,
                         int lda, const float* B, int ldb, float beta, float* C, int ldc,
                         const float* bias) {
  struct Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
  };
  constexpr std::size_t kWs = 32u << 20;
  static cublasLtHandle_t lt = [] { cublasLtHandle_t h; cublasLtCreate(&h); return h; }();
  static void* ws = [] { void* p = nullptr; cudaMalloc(&p, kWs); return p; }();
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
  return true;
}
inline bool lt_gemm_bias(bool, bool, int, int, int, double, const double*, int, const double*, int,
                         double, double*, int, const double*) {
  return false;
}
#endif

inline void gemm(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                 const View2D& C, const View1D& bias = {}) {
  const int m = C.extent(0), n = C.extent(1);
  const bool ta = (transA == 'T' || transA == 't');
  const bool tb = (transB == 'T' || transB == 't');
  const int k = ta ? A.extent(0) : A.extent(1);
  const bool has_bias = bias.extent(0) > 0;
  [[maybe_unused]] const int lda = A.extent(1), ldb = B.extent(1), ldc = C.extent(1);

#if defined(KOKKOS_ENABLE_CUDA)
  if (has_bias && lt_gemm_bias(tb, ta, n, m, k, alpha, B.data(), ldb, A.data(), lda, beta,
                               C.data(), ldc, bias.data()))
    return;
  vendor_gemm(tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, alpha,
              B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
  if (has_bias) add_bias(C, bias);
#elif defined(KOKKOS_ENABLE_HIP)
  vendor_gemm(tb ? rocblas_operation_transpose : rocblas_operation_none,
              ta ? rocblas_operation_transpose : rocblas_operation_none, n, m, k, alpha, B.data(),
              ldb, A.data(), lda, beta, C.data(), ldc);
  if (has_bias) add_bias(C, bias);
#else
  Kokkos::parallel_for(
      "gemm_naive", Kokkos::RangePolicy<ExecSpace>(0, m * n), KOKKOS_LAMBDA(int ij) {
        const int i = ij / n, j = ij % n;
        Net acc = Net(0);
        for (int p = 0; p < k; ++p) acc += (ta ? A(p, i) : A(i, p)) * (tb ? B(j, p) : B(p, j));
        C(i, j) = alpha * acc + beta * C(i, j) + (has_bias ? bias(j) : Net(0));
      });
#endif
}

}  // namespace pet
