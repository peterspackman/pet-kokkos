// GEMMs on 8- and 16-bit operands (GemmMode::FP8, BF16, FP16; see gemm.hpp).
//
// Both operands are converted to the narrow type and multiplied on tensor cores
// by cuBLASLt, accumulating and writing fp32. FP8 is E4M3 with one fp32 scale per
// operand -- its largest entry over E4M3's -- and cuBLASLt takes it only with the
// first operand transposed and the second not, leading dimensions multiples of
// 16 and compute capability 8.9. In row-major terms that is C = X W^T, X the
// activations and W a weight: linear's GEMM. linear_bwd's, C = X W, is the same
// with W^T, so each weight is converted once per layout it is used in, on first
// use, and kept; the 16-bit types take the same route. Activations are
// converted on every call, the FP8 scale computed on the device: no host sync,
// so it can sit inside a CUDA graph.
#include "pet/gemm.hpp"

#if defined(KOKKOS_ENABLE_CUDA) && !defined(PET_KOKKOS_FP64)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace pet {
namespace {

constexpr float kE4M3Max = 448.0f;

// The three operand types: what cuBLASLt calls them, their size, and the
// conversion, which for E4M3 divides by the scale first.
struct E4M3 {
  using T = __nv_fp8_storage_t;
  static constexpr cudaDataType_t type = CUDA_R_8F_E4M3;
  static constexpr bool scaled = true;
  __host__ __device__ static T from(float x) { return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3); }
};
struct BF16 {
  using T = __nv_bfloat16;
  static constexpr cudaDataType_t type = CUDA_R_16BF;
  static constexpr bool scaled = false;
  __host__ __device__ static T from(float x) { return __float2bfloat16(x); }
};
struct FP16 {
  using T = __half;
  static constexpr cudaDataType_t type = CUDA_R_16F;
  static constexpr bool scaled = false;
  __host__ __device__ static T from(float x) { return __float2half(x); }
};

int compute_capability() {
  static const int cc = [] {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor;
  }();
  return cc;
}

// A weight in the narrow type, [rows, k] row-major, and its scale (1 unscaled).
struct Weight {
  Kokkos::View<char*, MemSpace> q;
  Kokkos::View<float*, MemSpace> scale;
};

// W, or W^T when `transposed`, converted on the host: once per weight, type and
// layout, outside any graph capture (a graph records a call only after running
// it eagerly). Never destroyed, since that would happen after Kokkos::finalize.
template <class F>
const Weight& weight(const View2D& W, bool transposed) {
  static auto* cache = new std::map<std::pair<const void*, bool>, Weight>;
  auto [it, fresh] = cache->try_emplace({W.data(), transposed});
  if (!fresh) return it->second;
  const int r = W.extent(0), c = W.extent(1);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
  float amax = 0.0f;
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) amax = std::fmax(amax, std::fabs(float(h(i, j))));
  const float s = F::scaled && amax > 0.0f ? amax / kE4M3Max : 1.0f;
  std::vector<typename F::T> q(std::size_t(r) * c);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) q[transposed ? std::size_t(j) * r + i : std::size_t(i) * c + j] = F::from(float(h(i, j)) / s);
  Weight& w = it->second;
  const std::size_t bytes = q.size() * sizeof(typename F::T);
  w.q = Kokkos::View<char*, MemSpace>("lowp_weight", bytes);
  w.scale = Kokkos::View<float*, MemSpace>("lowp_weight_scale", 1);
  Kokkos::deep_copy(w.q, Kokkos::View<const char*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                             reinterpret_cast<const char*>(q.data()), bytes));
  Kokkos::deep_copy(w.scale, s);
  return w;
}

// Scratch for one activation at a time (GEMMs run in stream order). It only
// grows, and `generation` counts its moves for the graph key. Never destroyed.
struct Scratch {
  Kokkos::View<char*, MemSpace> q;
  Kokkos::View<unsigned*, MemSpace> amax;  // the largest |x|, as float bits
  Kokkos::View<float*, MemSpace> scale;
  std::size_t generation = 0;
};
Scratch& scratch_state() {
  static auto* s = new Scratch;
  return *s;
}
Scratch& scratch(std::size_t bytes) {
  Scratch& s = scratch_state();
  if (s.q.extent(0) < bytes) {
    s.q = Kokkos::View<char*, MemSpace>();
    s.q = Kokkos::View<char*, MemSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "lowp_act"), bytes + bytes / 4);
    ++s.generation;
  }
  if (s.amax.extent(0) == 0) {
    s.amax = Kokkos::View<unsigned*, MemSpace>("lowp_amax", 1);
    s.scale = Kokkos::View<float*, MemSpace>("lowp_scale", 1);
    Kokkos::deep_copy(s.scale, 1.0f);
    ++s.generation;
  }
  return s;
}

// Non-negative floats order like their bits, so an atomic max on the bits is
// the max of |x|, the same in any order: the result is deterministic.
__global__ void amax_kernel(const float* x, long n, unsigned* amax) {
  float m = 0.0f;
  for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x)
    m = fmaxf(m, fabsf(x[i]));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}

template <class F>
__global__ void convert_kernel(const float* x, long n, const unsigned* amax, typename F::T* q, float* scale) {
  float inv = 1.0f;
  const long i0 = blockIdx.x * long(blockDim.x) + threadIdx.x;
  if (F::scaled) {
    const float a = __uint_as_float(*amax), s = a > 0.0f ? a / kE4M3Max : 1.0f;
    inv = 1.0f / s;
    if (i0 == 0) *scale = s;
  }
  for (long i = i0; i < n; i += long(gridDim.x) * blockDim.x) q[i] = F::from(x[i] * inv);
}

template <class F>
void convert(const float* x, long n, Scratch& s, cudaStream_t st) {
  const int threads = 256, blocks = int(std::min<long>((n + threads - 1) / threads, 1024));
  auto* q = reinterpret_cast<typename F::T*>(s.q.data());
  if (F::scaled) {
    cudaMemsetAsync(s.amax.data(), 0, sizeof(unsigned), st);
    amax_kernel<<<blocks, threads, 0, st>>>(x, n, s.amax.data());
  }
  convert_kernel<F><<<blocks, threads, 0, st>>>(x, n, s.amax.data(), q, s.scale.data());
}

struct Plan {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t w{}, x{}, c{};
  cublasLtMatmulAlgo_t algo{};
  bool ok = false, bias = false;  // bias: in the epilogue
};

// C^T = W' X'^T in cuBLAS's column-major terms: W' the [n,k] weight (row-major),
// X' the [m,k] activations, C [m,n] row-major with leading dimension ldc. A bias
// goes in the epilogue where cuBLASLt takes one, else in a pass after.
const Plan& plan(cublasLtHandle_t lt, cudaDataType_t type, int m, int n, int k, int ldc, bool want_bias,
                 std::size_t ws_bytes) {
  static auto* plans = new std::map<std::tuple<int, int, int, int, int, bool>, Plan>;
  auto [it, fresh] = plans->try_emplace({int(type), m, n, k, ldc, want_bias});
  Plan& p = it->second;
  if (!fresh) return p;
  const cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
  cublasLtMatrixLayoutCreate(&p.w, type, k, n, k);
  cublasLtMatrixLayoutCreate(&p.x, type, k, m, k);
  cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F, n, m, ldc);
  for (const bool bias : {want_bias, false}) {
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N);
    if (bias) {
      const auto epi = CUBLASLT_EPILOGUE_BIAS;
      const auto bt = CUDA_R_32F;
      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
      cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof bt);
    }
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                         sizeof ws_bytes);
    cublasLtMatmulHeuristicResult_t r{};
    int found = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.w, p.x, p.c, p.c, pref, 1, &r, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (found) {
      p.algo = r.algo, p.ok = true, p.bias = bias;
      break;
    }
  }
  return p;
}

template <class F>
bool run(Net alpha, const View2D& A, const View2D& W, bool tw, Net beta, const View2D& C, const View1D& bias) {
  const int m = C.extent(0), n = C.extent(1), k = A.extent(1), ldc = C.extent(1);
  if (m == 0 || k % 16 || n % 16 || reinterpret_cast<std::uintptr_t>(C.data()) % 16) return false;
  if (int(tw ? W.extent(1) : W.extent(0)) != k) return false;

  constexpr std::size_t kWs = 32u << 20;
  static cublasLtHandle_t lt = [] { cublasLtHandle_t h; cublasLtCreate(&h); return h; }();
  static void* ws = [] { void* p = nullptr; cudaMalloc(&p, kWs); return p; }();
  const bool has_bias = bias.extent(0) > 0;
  const Plan& p = plan(lt, F::type, m, n, k, ldc, has_bias, kWs);
  if (!p.ok) return false;

  const Weight& w = weight<F>(W, !tw);  // the [n,k] layout: W itself when C = A W^T
  Scratch& s = scratch(std::size_t(m) * k * sizeof(typename F::T));
  const cudaStream_t st = ExecSpace().cuda_stream();
  convert<F>(A.data(), long(m) * k, s, st);

  if (F::scaled) {
    const float* wsc = w.scale.data();
    const float* xsc = s.scale.data();
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsc, sizeof wsc);
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xsc, sizeof xsc);
  }
  if (p.bias) {
    const float* b = bias.data();
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b, sizeof b);
  }
  const float al = alpha, be = beta;
  cublasLtMatmul(lt, p.op, &al, w.q.data(), p.w, s.q.data(), p.x, &be, C.data(), p.c, C.data(), p.c, &p.algo, ws,
                 kWs, st);
  if (has_bias && !p.bias) add_bias(C, bias);
  return true;
}

}  // namespace

bool gemm_lowp(char transA, char transW, Net alpha, const View2D& A, const View2D& W, Net beta, const View2D& C,
               const View1D& bias) {
  if (transA == 'T' || transA == 't') return false;
  const bool tw = transW == 'T' || transW == 't';
  switch (gemm_mode()) {
    case GemmMode::FP8: return compute_capability() >= 89 && run<E4M3>(alpha, A, W, tw, beta, C, bias);
    case GemmMode::BF16: return compute_capability() >= 80 && run<BF16>(alpha, A, W, tw, beta, C, bias);
    case GemmMode::FP16: return compute_capability() >= 70 && run<FP16>(alpha, A, W, tw, beta, C, bias);
    default: return false;
  }
}

std::size_t lowp_generation() { return scratch_state().generation; }

}  // namespace pet

#else

namespace pet {
bool gemm_lowp(char, char, Net, const View2D&, const View2D&, Net, const View2D&, const View1D&) { return false; }
std::size_t lowp_generation() { return 0; }
}  // namespace pet

#endif
