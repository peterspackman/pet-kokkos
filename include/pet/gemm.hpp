// Dense GEMM: C = alpha op(A) op(B) + beta C (+ bias on every row), on
// row-major Net views, through the vendor BLAS (cuBLAS, rocBLAS) or a naive
// Kokkos loop on the host. The only linear algebra PET needs.
//
// A row-major [r,c] matrix is the column-major [c,r] one, so each call computes
// C^T = op(B)^T op(A)^T: B and A swap places, and so do m and n.
#pragma once

#include "pet/kokkos.hpp"

#include <cstdlib>
#include <cstring>
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

// How the network's GEMMs run. Anything but Native changes the answer, so it is
// off unless asked for:
//   Native           fp32 (fp64 in an fp64 build), the reference
//   TF32             fp32 in and out, the products on TF32 tensor cores
//   BF16, FP16       operands converted to 16 bits, on tensor or matrix cores,
//                    fp32 accumulation and output (gemm_lowp.cpp)
//   FP8              E4M3 operands, each scaled by its largest entry, on FP8
//                    tensor cores (Ada, Hopper), fp32 accumulation and output
//   Ozaki            fp64-accurate products on integer tensor cores, for an
//                    fp64 build (ozaki.hpp)
// A mode the build or device cannot do runs Native. All but Native and Ozaki
// need an fp32 network; TF32 and FP8 need CUDA (compute capability 8.0 and
// 8.9); BF16 and FP16 run on CUDA (8.0, 7.0) or on AMD's matrix cores (rocBLAS).
// A GEMM the narrow path does not take (a shape cuBLASLt refuses) runs fp32.
// Process-global, from PET_GEMM=native|tf32|bf16|fp16|fp8|ozaki.
enum class GemmMode { Native, TF32, BF16, FP16, FP8, Ozaki };

inline GemmMode parse_gemm_mode(const char* s) {
  const char* names[] = {"native", "tf32", "bf16", "fp16", "fp8", "ozaki"};
  for (int i = 0; i < 6; ++i)
    if (std::strcmp(s, names[i]) == 0) return GemmMode(i);
  throw std::runtime_error(std::string("pet: unknown PET_GEMM mode '") + s + "'");
}

inline GemmMode& gemm_mode() {
  static GemmMode m = [] {
    const char* e = std::getenv("PET_GEMM");
    return e && *e ? parse_gemm_mode(e) : GemmMode::Native;
  }();
  return m;
}

#if defined(KOKKOS_ENABLE_CUDA)
inline cublasHandle_t blas_handle() {
  // On Kokkos's stream, so GEMMs are ordered with the kernels around them. Never
  // destroyed: static destruction would run after Kokkos::finalize.
  static cublasHandle_t h = [] {
    cublasHandle_t hh;
    cublasCreate(&hh);
    cublasSetStream(hh, Kokkos::DefaultExecutionSpace().cuda_stream());
    return hh;
  }();
  return h;
}
// The compute type for an fp32 GEMM: TF32 tensor cores in that mode. (cuBLAS's
// "fast" bf16 and fp16 compute types only permit down-conversion, and in
// practice choose the same TF32 kernels; those modes convert explicitly, in
// gemm_lowp.cpp.)
inline cublasComputeType_t fp32_compute_type() {
  return gemm_mode() == GemmMode::TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}
// Overloads by scalar type, not `if constexpr`: nvcc type-checks the discarded
// branch of one in a non-template function.
inline void vendor_gemm(cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, float al,
                        const float* A, int lda, const float* B, int ldb, float be, float* C,
                        int ldc) {
  const cublasComputeType_t compute = fp32_compute_type();
  if (compute == CUBLAS_COMPUTE_32F)
    cublasSgemm(blas_handle(), oa, ob, m, n, k, &al, A, lda, B, ldb, &be, C, ldc);
  else
    cublasGemmEx(blas_handle(), oa, ob, m, n, k, &al, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &be, C,
                 CUDA_R_32F, ldc, compute, CUBLAS_GEMM_DEFAULT);
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
      "bias", RangePolicy(0, C.extent(0) * n),
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
  static std::map<std::tuple<int, bool, bool, int, int, int, int, int, int>, Plan> plans;
  const cublasComputeType_t compute = fp32_compute_type();
  auto [it, fresh] = plans.try_emplace({int(compute), ta, tb, m, n, k, lda, ldb, ldc});
  Plan& p = it->second;
  if (fresh) {
    const auto opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N, opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescCreate(&p.op, compute, CUDA_R_32F);
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

// The FP8, BF16 and FP16 modes (gemm_lowp.cpp): C = alpha op(A) op(W) + beta C
// (+ bias), W a weight, which it converts on first use and keeps. False when it
// cannot -- another mode, not CUDA, an fp64 network, too old a device, or a
// shape cuBLASLt does not take -- and the caller runs the GEMM itself.
bool gemm_lowp(char transA, char transW, Net alpha, const View2D& A, const View2D& W, Net beta,
               const View2D& C, const View1D& bias = {});
// Counts moves of its activation scratch: part of a recorded graph's key.
std::size_t lowp_generation();

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
      "gemm_naive", RangePolicy(0, m * n), KOKKOS_LAMBDA(int ij) {
        const int i = ij / n, j = ij % n;
        Net acc = Net(0);
        for (int p = 0; p < k; ++p) acc += (ta ? A(p, i) : A(i, p)) * (tb ? B(j, p) : B(p, j));
        C(i, j) = alpha * acc + beta * C(i, j) + (has_bias ? bias(j) : Net(0));
      });
#endif
}

}  // namespace pet
