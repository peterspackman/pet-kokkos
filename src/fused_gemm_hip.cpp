// The fp32 weight GEMMs on AMD's CDNA GPUs, every one in the network: our own
// main loop on the fp32 matrix cores (mfma_gemm.hpp) with the epilogue walking
// each lane's accumulators -- the bias, or the elementwise neighbours fused in,
// as fused_gemm.cpp does with CUTLASS on CUDA. rocBLAS and hipBLASLt have no
// SiLU-with-saved-input or SwiGLU epilogue, so without this each of those
// activations is a second pass over a wide activation.
//
// Every tile sums each output's k in the same order, and the few shapes the
// main loop does not take (a K or N not a multiple of four: the energy heads)
// go to a plain per-element loop, chosen by shape alone: so a row's answer does
// not depend on how many rows the GEMM has, and a structure gets the same bits
// whatever batch it is in (batch_invariant_gemms).
//
// Each function returns false where it does not run -- not a CDNA GPU, a
// double network, a 16-bit GEMM mode, PET_MFMA_GEMM=0, or an operand the kernels
// cannot read -- and the caller takes the unfused path (rocBLAS / hipBLASLt).
#include "ops.hpp"

#include "pet/gemm.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

#if defined(PET_FUSED_HIP)
#include "mfma_gemm.hpp"

namespace pet {

namespace {

namespace mg = mfma_gemm;
struct RM {};  // B = W, a [K, N] weight
struct CM {};  // B = W^T, a [N, K] weight

bool aligned(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("PET_MFMA_GEMM");
    return !(e && e[0] == '0');
  }();
  return on;
}
// The fp32 modes: native, and TF32, which this hardware does not have (it runs
// fp32). The 16-bit modes keep their own path (gemm_lowp.cpp).
bool fp32_mode() { return gemm_mode() == GemmMode::Native || gemm_mode() == GemmMode::TF32; }

// detail::set_cutlass_tile: a pinned block tile (0 large, 1 mid, 2 small) or -1.
int& forced_tile() {
  static int t = [] {
    const char* e = std::getenv("PET_CUTLASS_TILE");
    if (!e) return -1;
    const std::string s(e);
    return s == "large" ? 0 : s == "mid" ? 1 : s == "small" ? 2 : -1;
  }();
  return t;
}

using Large = mg::Tile<128, 128, 64, 64>;
using Mid = mg::Tile<128, 64, 64, 32>;
using Small = mg::Tile<64, 64, 32, 32>;

int cu_count() {
  static const int n = [] {
    int dev = 0, c = 0;
    (void) hipGetDevice(&dev);
    (void) hipDeviceGetAttribute(&c, hipDeviceAttributeMultiprocessorCount, dev);
    return c > 0 ? c : 110;
  }();
  return n;
}
template <class T>
long blocks(int M, int N) {
  return long(M + T::BM - 1) / T::BM * ((N + T::BN - 1) / T::BN);
}

// Reported to Kokkos Tools as a kernel called `name`, like any Kokkos kernel.
// The tile by shape, timed on an MI250X GCD: 128 x 64 at k = 256, 128 x 128
// from 512, each only while it gives two blocks a CU, else the next smaller
// (the tile changes the speed, never a bit of the answer). Eight-wave 128 x 128
// blocks were faster alone but 1% slower in the network.
template <class LayoutB, class Epi>
void launch(const char* name, int M, int N, int K, const float* A, const float* B, Epi epi) {
  if (M == 0 || N == 0) return;
  constexpr bool NN = std::is_same_v<LayoutB, RM>;
  int t = forced_tile();
  if (t < 0) {
    const long enough = 2L * cu_count();
    t = K <= 256 ? 1 : 0;
    if (t == 0 && blocks<Large>(M, N) < enough) t = 1;
    if (t == 1 && blocks<Mid>(M, N) < enough) t = 2;
  }
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  const hipStream_t st = ExecSpace().hip_stream();
  if (t == 0) mg::launch<Large, NN>(st, M, N, K, A, B, epi);
  else if (t == 1) mg::launch<Mid, NN>(st, M, N, K, A, B, epi);
  else mg::launch<Small, NN>(st, M, N, K, A, B, epi);
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

// The main loop reads both operands' rows 16 bytes at a time and the epilogues
// write whole runs of four: contiguous, 16-byte aligned, K and N multiples of 4.
bool vec_ok(const View2D& v) { return v.span_is_contiguous() && v.extent(1) % 4 == 0 && aligned(v.data()); }
bool vec_ok(const View1D& v) { return aligned(v.data()); }
bool usable(const View2D& in, const View2D& w) { return enabled() && fp32_mode() && vec_ok(in) && vec_ok(w); }

template <int L>
__device__ void load(const float* p, float* v) {
  static_assert(L == 4 || L == 2);
  if constexpr (L == 4) *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(p);
  else *reinterpret_cast<float2*>(v) = *reinterpret_cast<const float2*>(p);
}
template <int L>
__device__ void store(float* p, const float* v) {
  static_assert(L == 4 || L == 2);
  if constexpr (L == 4) *reinterpret_cast<float4*>(p) = *reinterpret_cast<const float4*>(v);
  else *reinterpret_cast<float2*>(p) = *reinterpret_cast<const float2*>(v);
}

// The epilogues, as fused_gemm.cpp's: run(r, c, v) gets output row r's
// columns c..c+3.

// pre = acc + b; h(j) = pre(2j) * sigmoid(pre(2j+1)).
struct SwigluFwd {
  float *pre, *h;
  const float* b;
  int N;
  __device__ void run(int r, int c, const float* v) const {
    float bb[4], p[4], q[2];
    load<4>(b + c, bb);
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = v[j] + bb[j];
    store<4>(pre + std::size_t(r) * N + c, p);
    if (!h) return;
#pragma unroll
    for (int j = 0; j < 2; ++j) q[j] = p[2 * j] * sigmoid(p[2 * j + 1]);
    store<2>(h + std::size_t(r) * (N / 2) + c / 2, q);
  }
};

// acc = h_adj(r, c..c+3); pre_adj(2j) = h_adj * sg, pre_adj(2j+1) = h_adj v sg (1 - sg).
struct SwigluBwd {
  const float* pre;
  float* pre_adj;
  int F;
  __device__ void run(int r, int c, const float* ha) const {
    const std::size_t o = std::size_t(r) * 2 * F + 2 * c;
    float p[8], q[8];
    load<4>(pre + o, p), load<4>(pre + o + 4, p + 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = p[2 * j], sg = sigmoid(p[2 * j + 1]);
      q[2 * j] = ha[j] * sg;
      q[2 * j + 1] = ha[j] * v * sg * (1.0f - sg);
    }
    store<4>(pre_adj + o, q), store<4>(pre_adj + o + 4, q + 4);
  }
};

// p = acc + b; out = silu(p), and pre = p when kept.
struct SiluFwd {
  float *out, *pre;
  const float* b;
  int N;
  __device__ void run(int r, int c, const float* v) const {
    float p[4], o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = v[j];
    if (b) {
      float bb[4];
      load<4>(b + c, bb);
#pragma unroll
      for (int j = 0; j < 4; ++j) p[j] += bb[j];
    }
    const std::size_t at = std::size_t(r) * N + c;
    if (pre) store<4>(pre + at, p);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = p[j] * sigmoid(p[j]);
    store<4>(out + at, o);
  }
};

// acc = the adjoint of silu's output; grad = acc * silu'(pre).
struct SiluBwd {
  const float* pre;
  float* grad;
  int N;
  __device__ void run(int r, int c, const float* a) const {
    const std::size_t at = std::size_t(r) * N + c;
    float x[4], g[4];
    load<4>(pre + at, x);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float sg = sigmoid(x[j]);
      g[j] = a[j] * sg * (1.0f + x[j] * (1.0f - sg));
    }
    store<4>(grad + at, g);
  }
};

// out = acc + b (+ beta out): the GEMM with nothing fused.
struct Plain {
  float* out;
  const float* b;
  float beta;
  int N;
  __device__ void run(int r, int c, const float* v) const {
    float* o = out + std::size_t(r) * N + c;
    float x[4], y[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = v[j];
    if (b) {
      load<4>(b + c, y);
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] += y[j];
    }
    if (beta != 0.0f) {
      load<4>(o, y);
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] += beta * y[j];
    }
    store<4>(o, x);
  }
};

// The shapes the main loop does not take. Against a [K, N] weight (K small: the
// energy heads' backward), an output a thread, its k in order. Against an
// [N, K] one (N small: the heads' forward), a wavefront a row, the lanes
// reading it together: lane l sums k = l, l + 64, ..., then a fixed tree.
__global__ void __launch_bounds__(256) gemm_plain_nn(int M, int N, int K, const float* A, int lda, const float* W,
                                                     float* out, int ldo, const float* b, float beta) {
  const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (i >= std::size_t(M) * N) return;
  const int r = int(i / N), c = int(i % N);
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[std::size_t(r) * lda + k] * W[std::size_t(k) * N + c];
  float* o = out + std::size_t(r) * ldo + c;
  float x = acc;
  if (b) x += b[c];
  if (beta != 0.0f) x += beta * *o;
  *o = x;
}
__global__ void __launch_bounds__(256) gemm_plain_nt(int M, int N, int K, const float* A, int lda, const float* W,
                                                     float* out, int ldo, const float* b, float beta) {
  const int r = blockIdx.x * 4 + threadIdx.x / 64, l = threadIdx.x % 64;
  if (r >= M) return;
  const float* a = A + std::size_t(r) * lda;
  for (int c = 0; c < N; ++c) {
    const float* w = W + std::size_t(c) * K;
    float acc = 0.0f;
    for (int k = l; k < K; k += 64) acc += a[k] * w[k];
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (l == 0) {
      float* p = out + std::size_t(r) * ldo + c;
      float x = acc;
      if (b) x += b[c];
      if (beta != 0.0f) x += beta * *p;
      *p = x;
    }
  }
}

}  // namespace

bool batch_invariant_gemms() { return enabled() && gemm_mode() == GemmMode::Native; }
void detail::set_cutlass_tile(int tile) { forced_tile() = tile; }
int detail::tf32_tile_count() { return 0; }
void detail::set_tf32_tile(int) {}

bool linear_fused(View2D out, View2D in, const View2D& w, bool w_transposed, View1D b, Net beta) {
  if (!enabled() || !fp32_mode() || !w.span_is_contiguous()) return false;
  const int M = out.extent(0), N = out.extent(1), K = in.extent(1);
  if (K % 4 == 0 && N % 4 == 0) {  // the main loop's shapes: unless an operand is laid out oddly
    if (!usable(in, w) || !vec_ok(out) || (b.data() && !vec_ok(b))) return false;
    const Plain epi{out.data(), b.data(), float(beta), N};
    if (w_transposed) launch<CM>("linear", M, N, K, in.data(), w.data(), epi);
    else launch<RM>("linear_bwd", M, N, K, in.data(), w.data(), epi);
    return true;
  }
  if (in.stride(1) != 1 || out.stride(1) != 1) return false;
  if (M == 0 || N == 0) return true;
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor("linear_plain", 0, &id);
  const std::size_t n = std::size_t(M) * N;
  if (w_transposed)
    hipLaunchKernelGGL(gemm_plain_nt, dim3((M + 3) / 4), dim3(256), 0, ExecSpace().hip_stream(), M, N, K, in.data(),
                       int(in.stride(0)), w.data(), out.data(), int(out.stride(0)), b.data(), float(beta));
  else
    hipLaunchKernelGGL(gemm_plain_nn, dim3((n + 255) / 256), dim3(256), 0, ExecSpace().hip_stream(), M, N, K,
                       in.data(), int(in.stride(0)), w.data(), out.data(), int(out.stride(0)), b.data(), float(beta));
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
  return true;
}

bool silu_in_fused(View2D out, View2D pre, View2D in, const View2D& w, View1D b) {
  if (!usable(in, w) || !vec_ok(out) || (pre.data() && !vec_ok(pre)) || (b.data() && !vec_ok(b))) return false;
  launch<CM>("fused linear_silu", out.extent(0), out.extent(1), in.extent(1), in.data(), w.data(),
             SiluFwd{out.data(), pre.data(), b.data(), int(out.extent(1))});
  return true;
}

bool silu_bwd_fused(View2D grad, View2D out_adj, const View2D& w, View2D pre) {
  if (!usable(out_adj, w) || !vec_ok(grad) || !vec_ok(pre)) return false;
  launch<RM>("fused silu_bwd", grad.extent(0), grad.extent(1), out_adj.extent(1), out_adj.data(), w.data(),
             SiluBwd{pre.data(), grad.data(), int(grad.extent(1))});
  return true;
}

bool swiglu_in_fused(View2D pre, View2D h, View2D in, const View2D& w, View1D b) {
  if (!usable(in, w) || !vec_ok(pre) || (h.data() && !vec_ok(h)) || !vec_ok(b)) return false;
  launch<CM>("fused swiglu_in", pre.extent(0), pre.extent(1), in.extent(1), in.data(), w.data(),
             SwigluFwd{pre.data(), h.data(), b.data(), int(pre.extent(1))});
  return true;
}

bool swiglu_bwd_fused(View2D pre_adj, View2D out_adj, const View2D& w_out, View2D pre) {
  if (!usable(out_adj, w_out) || !vec_ok(pre) || !vec_ok(pre_adj) || w_out.extent(1) % 4) return false;
  launch<RM>("fused swiglu_bwd", out_adj.extent(0), w_out.extent(1), out_adj.extent(1), out_adj.data(), w_out.data(),
             SwigluBwd{pre.data(), pre_adj.data(), int(w_out.extent(1))});
  return true;
}

}  // namespace pet

#endif
