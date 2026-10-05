// GEMMs with their elementwise neighbours fused into the epilogue: CUTLASS's
// fp32 SIMT main loop (bit-identical to cuBLAS's SGEMM, and within a few percent
// of its speed), with our own epilogue walking each thread's accumulators.
//
// Each function returns false when it cannot run here -- no CUTLASS, a double
// network, PET_CUTLASS=0, or an unaligned operand -- and the caller takes the
// unfused path.
#include "ops.hpp"

#include <cstdlib>

#if defined(PET_HAVE_CUTLASS) && defined(KOKKOS_ENABLE_CUDA) && !defined(PET_KOKKOS_FP64)
#define PET_FUSED 1
#include <cutlass/cutlass.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#endif

namespace pet {

#ifdef PET_FUSED

namespace {

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using TileShape = cutlass::gemm::GemmShape<256, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;

// out[M, N] = A[M, K] . B, B = W^T for a [N, K] weight (ColumnMajor) or W for a
// [K, N] one (RowMajor).
template <class LayoutB>
using Mma = typename cutlass::gemm::threadblock::DefaultMma<
    float, RM, 1, float, LayoutB, 1, float, RM, cutlass::arch::OpClassSimt, cutlass::arch::Sm80, TileShape,
    WarpShape, cutlass::gemm::GemmShape<1, 1, 1>, 3, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

// One CTA per output tile. The epilogue gets each thread's accumulators as runs
// of kLane contiguous columns, with the run's (row, col); rows and runs outside
// the output are skipped (N is a multiple of kLane).
template <class Mma, class Epi>
__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32)
    gemm_kernel(int M, int N, int K, const float* A, const float* B, Epi epi) {
  extern __shared__ char smem[];
  using Tile = typename Mma::Shape;
  using Policy = typename Mma::Operator::Policy;
  using Warp = typename Mma::Operator::Shape;
  using LayoutB = typename Mma::IteratorB::Layout;
  const int m0 = blockIdx.x * Tile::kM, n0 = blockIdx.y * Tile::kN;
  const int ldb = std::is_same_v<LayoutB, CM> ? K : N;
  typename Mma::IteratorA ia({RM(K)}, const_cast<float*>(A), {M, K}, threadIdx.x, {m0, 0});
  typename Mma::IteratorB ib({LayoutB(ldb)}, const_cast<float*>(B), {K, N}, threadIdx.x, {0, n0});
  const int warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0), lane = threadIdx.x % 32;
  Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(smem), threadIdx.x, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma((K + Tile::kK - 1) / Tile::kK, acc, ia, ib, acc);

  // The layout MmaSimtTileIterator stores C in: lanes tile the warp in
  // LaneMmaShape blocks, repeated every lanes x LaneMmaShape.
  constexpr int LM = Policy::LaneMmaShape::kM, LN = Policy::LaneMmaShape::kN;
  constexpr int DM = Policy::WarpShape::kRow * LM, DN = Policy::WarpShape::kColumn * LN;
  constexpr int IM = Warp::kM / DM, IN = Warp::kN / DN;
  const auto lo = Policy::get_lane_layout().inverse(lane);
  const int r0 = m0 + warp % Mma::WarpCount::kM * Warp::kM + lo.row() * LM;
  const int c0 = n0 + warp / Mma::WarpCount::kM * Warp::kN + lo.column() * LN;
#pragma unroll
  for (int im = 0; im < IM; ++im)
#pragma unroll
    for (int m = 0; m < LM; ++m)
#pragma unroll
      for (int in = 0; in < IN; ++in) {
        const int r = r0 + im * DM + m, c = c0 + in * DN;
        if (r < M && c < N) epi(r, c, acc.data() + LN * (in + IN * (m + im * LM)));
      }
}

constexpr int kLane = 4;  // LaneMmaShape::kN for fp32 at this warp shape

// Reported to Kokkos Tools as a kernel called `name`, like any Kokkos kernel.
template <class LayoutB, class Epi>
void launch(const char* name, int M, int N, int K, const float* A, const float* B, Epi epi) {
  using G = Mma<LayoutB>;
  static_assert(G::Operator::Policy::LaneMmaShape::kN == kLane);
  constexpr int smem = sizeof(typename G::SharedStorage);
  static const bool attr =
      cudaFuncSetAttribute(gemm_kernel<G, Epi>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
  (void) attr;
  const dim3 grid((M + G::Shape::kM - 1) / G::Shape::kM, (N + G::Shape::kN - 1) / G::Shape::kN);
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  gemm_kernel<G, Epi><<<grid, G::WarpCount::kCount * 32, smem, ExecSpace().cuda_stream()>>>(M, N, K, A, B, epi);
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("PET_CUTLASS");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Enough 256x128 tiles to fill the device a few times over; below that cuBLAS's
// smaller tiles win.
bool big_enough(int M, int N) {
  static const int sms = [] {
    int n = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n;
  }();
  return long(M + TileShape::kM - 1) / TileShape::kM * ((N + TileShape::kN - 1) / TileShape::kN) >= 4L * sms;
}

// Contiguous rows, 16-byte aligned, a whole number of kLane runs wide.
bool vec_ok(const View2D& v) {
  return v.span_is_contiguous() && v.extent(1) % kLane == 0 && reinterpret_cast<std::uintptr_t>(v.data()) % 16 == 0;
}
bool vec_ok(const View1D& v) { return reinterpret_cast<std::uintptr_t>(v.data()) % 16 == 0; }

using F4 = float4;
__device__ F4& at4(float* p, std::size_t i) { return *reinterpret_cast<F4*>(p + i); }

// pre = acc + b; h(j) = pre(2j) * sigmoid(pre(2j+1)).
struct SwigluFwd {
  float *pre, *h;
  const float* b;
  int N;
  __device__ void operator()(int r, int c, const float* v) const {
    const F4 bb = *reinterpret_cast<const F4*>(b + c);
    const F4 p = {v[0] + bb.x, v[1] + bb.y, v[2] + bb.z, v[3] + bb.w};
    at4(pre, std::size_t(r) * N + c) = p;
    if (h) *reinterpret_cast<float2*>(h + std::size_t(r) * (N / 2) + c / 2) = {p.x * sigmoid(p.y), p.z * sigmoid(p.w)};
  }
};

// acc = h_adj(r, c..c+3); pre_adj(2j) = h_adj * sg, pre_adj(2j+1) = h_adj v sg (1 - sg).
struct SwigluBwd {
  const float* pre;
  float* pre_adj;
  int F;
  __device__ void operator()(int r, int c, const float* ha) const {
    const std::size_t o = std::size_t(r) * 2 * F + 2 * c;
    float p[8], q[8];
    *reinterpret_cast<F4*>(p) = at4(const_cast<float*>(pre), o);
    *reinterpret_cast<F4*>(p + 4) = at4(const_cast<float*>(pre), o + 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = p[2 * j], sg = sigmoid(p[2 * j + 1]);
      q[2 * j] = ha[j] * sg;
      q[2 * j + 1] = ha[j] * v * sg * (1.0f - sg);
    }
    at4(pre_adj, o) = *reinterpret_cast<F4*>(q);
    at4(pre_adj, o + 4) = *reinterpret_cast<F4*>(q + 4);
  }
};

}  // namespace

bool swiglu_in_fused(View2D pre, View2D h, View2D in, const View2D& w, View1D b) {
  if (!enabled() || !big_enough(pre.extent(0), pre.extent(1)) || !vec_ok(pre) || (h.data() && !vec_ok(h)) || !vec_ok(b) || !in.span_is_contiguous()) return false;
  launch<CM>("fused swiglu_in", pre.extent(0), pre.extent(1), in.extent(1), in.data(), w.data(),
             SwigluFwd{pre.data(), h.data(), b.data(), int(pre.extent(1))});
  return true;
}

bool swiglu_bwd_fused(View2D pre_adj, View2D out_adj, const View2D& w_out, View2D pre) {
  if (!enabled() || !big_enough(out_adj.extent(0), w_out.extent(1)) || !vec_ok(pre) || !vec_ok(pre_adj) || !out_adj.span_is_contiguous() ||
      w_out.extent(1) % kLane)
    return false;
  launch<RM>("fused swiglu_bwd", out_adj.extent(0), w_out.extent(1), out_adj.extent(1), out_adj.data(), w_out.data(),
             SwigluBwd{pre.data(), pre_adj.data(), int(w_out.extent(1))});
  return true;
}

#else

bool swiglu_in_fused(View2D, View2D, View2D, const View2D&, View1D) { return false; }
bool swiglu_bwd_fused(View2D, View2D, const View2D&, View2D) { return false; }

#endif

}  // namespace pet
