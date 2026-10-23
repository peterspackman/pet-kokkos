// The fp32 and TF32 modes' weight GEMMs on CUDA, every one in the network:
// CUTLASS's main loop -- SIMT fp32 in the native mode, TF32 tensor cores in the
// TF32 mode -- with our own epilogue walking each thread's accumulators: the
// bias, or the elementwise neighbours fused in. In the native mode one kernel
// family serves every shape, so a structure's answer does not depend on its
// batch.
//
// Each function returns false when it cannot run here -- no CUTLASS, a double
// network, another GEMM mode, PET_CUTLASS=0, or an unaligned operand -- and the
// caller takes the unfused path (cuBLAS, whose kernel and split-k depend on the
// shape).
#include "ops.hpp"

#include "pet/gemm.hpp"

#include <cstdlib>
#include <map>
#include <string>
#include <tuple>

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
// Three SIMT CTA tiles, the largest that fills the device twice over. All are
// the same SIMT main loop: each thread runs one FMA chain over k, in order, per
// output element, so the tile changes the speed and never a bit of the answer.
// That makes a row's result independent of how many rows the GEMM has -- of the
// batch it came in -- which no split-k or tensor-core kernel promises. Timed on
// an RTX 4080 SUPER against cuBLAS: Large is level with it from a few hundred
// rows of tokens up, and a single small structure is ~0.3 ms slower.
template <int TM, int TN, int WM, int WN>
struct Tiles {
  using Tile = cutlass::gemm::GemmShape<TM, TN, 8>;
  using Warp = cutlass::gemm::GemmShape<WM, WN, 8>;
  static constexpr bool kTensor = false;
  static constexpr int kStages = 3;
};
using Large = Tiles<128, 128, 32, 64>;
using Mid = Tiles<64, 64, 32, 32>;
using Small = Tiles<64, 32, 32, 32>;

// TF32 tensor-core tiles (m16n8k8 instructions, operands 16-byte aligned), all
// with three or more stages: those share CUTLASS's multistage main loop, which
// sums each output's k in the same order whatever the tile, so the tile changes
// the speed and never a bit of the answer (two stages take the pipelined loop,
// whose order differs). No one tile is fastest across the network's shapes --
// they are memory-bound at k = 128-256, and a tile narrower than a row rereads
// the large activation operand -- so each shape takes the one timed fastest on
// its first call (pick_tile).
template <int TM, int TN, int TK, int WM, int WN, int S>
struct TcTiles {
  using Tile = cutlass::gemm::GemmShape<TM, TN, TK>;
  using Warp = cutlass::gemm::GemmShape<WM, WN, TK>;
  static constexpr bool kTensor = true;
  static constexpr int kStages = S;
};
using Tc64x64 = TcTiles<64, 64, 32, 32, 32, 4>;
using Tc64x64s = TcTiles<64, 64, 16, 32, 32, 3>;  // the s tiles: small shared memory, more CTAs per SM
using Tc64x128s = TcTiles<64, 128, 16, 32, 64, 3>;
using Tc128x64s = TcTiles<128, 64, 16, 64, 32, 3>;
using Tc128x128 = TcTiles<128, 128, 16, 64, 64, 3>;
using Tc128x128w8 = TcTiles<128, 128, 32, 32, 64, 3>;
using Tc128x256 = TcTiles<128, 256, 16, 64, 64, 3>;
using Tc256x128 = TcTiles<256, 128, 16, 64, 64, 3>;

// out[M, N] = A[M, K] . B, B = W^T for a [N, K] weight (ColumnMajor) or W for a
// [K, N] one (RowMajor).
template <class Cfg, class LayoutB, bool Tensor = Cfg::kTensor>
struct MmaOf {
  using type = typename cutlass::gemm::threadblock::DefaultMma<
      float, RM, 1, float, LayoutB, 1, float, RM, cutlass::arch::OpClassSimt, cutlass::arch::Sm80, typename Cfg::Tile,
      typename Cfg::Warp, cutlass::gemm::GemmShape<1, 1, 1>, Cfg::kStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
};
template <class Cfg, class LayoutB>
struct MmaOf<Cfg, LayoutB, true> {
  using type = typename cutlass::gemm::threadblock::DefaultMma<
      float, RM, 4, float, LayoutB, 4, float, RM, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      typename Cfg::Tile, typename Cfg::Warp, cutlass::gemm::GemmShape<16, 8, 8>, Cfg::kStages,
      cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
};
template <class Cfg, class LayoutB>
using Mma = typename MmaOf<Cfg, LayoutB>::type;

// Each thread's accumulators come in runs of L contiguous columns: 4 on SIMT
// (LaneMmaShape::kN at these warp shapes), 2 on tensor cores.
template <bool Tensor>
constexpr int kRun = Tensor ? 2 : 4;

// One CTA per output tile. The epilogue gets each thread's accumulators as runs
// of L contiguous columns, with the run's (row, col); rows and runs outside the
// output are skipped, and a run the edge cuts (N not a multiple of L) is the
// epilogue's to stop short.
template <class Mma, bool Tensor, class Epi>
__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32)
    gemm_kernel(int M, int N, int K, const float* A, const float* B, Epi epi) {
  extern __shared__ char smem[];
  using Tile = typename Mma::Shape;
  using Warp = typename Mma::Operator::Shape;
  using LayoutB = typename Mma::IteratorB::Layout;
  constexpr int L = kRun<Tensor>;
  const int m0 = blockIdx.x * Tile::kM, n0 = blockIdx.y * Tile::kN;
  const int ldb = std::is_same_v<LayoutB, CM> ? K : N;
  typename Mma::IteratorA ia({RM(K)}, const_cast<float*>(A), {M, K}, threadIdx.x, {m0, 0});
  typename Mma::IteratorB ib({LayoutB(ldb)}, const_cast<float*>(B), {K, N}, threadIdx.x, {0, n0});
  const int warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0), lane = threadIdx.x % 32;
  Mma mma(*reinterpret_cast<typename Mma::SharedStorage*>(smem), threadIdx.x, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma((K + Tile::kK - 1) / Tile::kK, acc, ia, ib, acc);

  const int wm = m0 + warp % Mma::WarpCount::kM * Warp::kM, wn = n0 + warp / Mma::WarpCount::kM * Warp::kN;
  if constexpr (Tensor) {
    // MmaTensorOp's order: the warp's m16n8 instruction tiles column-major,
    // four values each, (g, 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1) for
    // g = lane / 4, t = lane % 4.
    using Inst = typename Mma::Operator::InstructionShape;
    constexpr int IR = Warp::kM / Inst::kM, IC = Warp::kN / Inst::kN;
    const int r0 = wm + lane / 4, c0 = wn + 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < IR; ++i)
#pragma unroll
      for (int j = 0; j < IC; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = r0 + i * Inst::kM + 8 * h, c = c0 + j * Inst::kN;
          if (r < M && c < N) epi.template run<L>(r, c, acc.data() + 4 * (i + j * IR) + 2 * h);
        }
  } else {
    // The layout MmaSimtTileIterator stores C in: lanes tile the warp in
    // LaneMmaShape blocks, repeated every lanes x LaneMmaShape.
    using Policy = typename Mma::Operator::Policy;
    constexpr int LM = Policy::LaneMmaShape::kM, LN = Policy::LaneMmaShape::kN;
    static_assert(LN == L);
    constexpr int DM = Policy::WarpShape::kRow * LM, DN = Policy::WarpShape::kColumn * LN;
    constexpr int IM = Warp::kM / DM, IN = Warp::kN / DN;
    const auto lo = Policy::get_lane_layout().inverse(lane);
    const int r0 = wm + lo.row() * LM, c0 = wn + lo.column() * LN;
#pragma unroll
    for (int im = 0; im < IM; ++im)
#pragma unroll
      for (int m = 0; m < LM; ++m)
#pragma unroll
        for (int in = 0; in < IN; ++in) {
          const int r = r0 + im * DM + m, c = c0 + in * DN;
          if (r < M && c < N) epi.template run<L>(r, c, acc.data() + LN * (in + IN * (m + im * LM)));
        }
  }
}

template <class Cfg, class LayoutB, class Epi>
void launch_tile(const char* name, int M, int N, int K, const float* A, const float* B, Epi epi) {
  using G = Mma<Cfg, LayoutB>;
  constexpr int smem = sizeof(typename G::SharedStorage);
  constexpr auto kernel = gemm_kernel<G, Cfg::kTensor, Epi>;
  static const bool attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
  (void) attr;
  const dim3 grid((M + G::Shape::kM - 1) / G::Shape::kM, (N + G::Shape::kN - 1) / G::Shape::kN);
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  kernel<<<grid, G::WarpCount::kCount * 32, smem, ExecSpace().cuda_stream()>>>(M, N, K, A, B, epi);
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

int sm_count() {
  static const int sms = [] {
    int n = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n;
  }();
  return sms;
}

template <class Cfg>
long tiles(int M, int N) {
  return long(M + Cfg::Tile::kM - 1) / Cfg::Tile::kM * ((N + Cfg::Tile::kN - 1) / Cfg::Tile::kN);
}

// detail::set_cutlass_tile, from PET_CUTLASS_TILE: a pinned tile or -1.
int& forced_tile() {
  static int t = [] {
    const char* e = std::getenv("PET_CUTLASS_TILE");
    if (!e) return -1;
    const std::string s(e);
    return s == "large" ? 0 : s == "mid" ? 1 : s == "small" ? 2 : -1;
  }();
  return t;
}

// Tensor cores for the TF32 mode on Ampere and Ada. Hopper and later have
// faster instructions (wgmma) than this main loop's, which cuBLAS uses there.
bool sm8x() {
  static const bool yes = [] {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major == 8;
  }();
  return yes;
}
bool tensor() { return sm8x() && gemm_mode() == GemmMode::TF32; }

int& pinned_tf32_tile() {
  static int t = -1;
  return t;
}

constexpr int kTf32Tiles = 8;

// The tensor-core tile for a GEMM: the fastest of the candidates, timed once per
// (kernel, rows to within a factor of two, N, K) and kept. While a CUDA graph is
// being recorded nothing can be timed, so an untimed shape takes 128 x 128. An
// epilogue that reads its output (`keep`, n floats) has it restored between
// the timed runs.
template <class LayoutB, class Epi>
void launch_tc(const char* name, int M, int N, int K, const float* A, const float* B, Epi epi, float* keep,
               std::size_t n) {
  using Fn = void (*)(const char*, int, int, int, const float*, const float*, Epi);
  static constexpr Fn fns[] = {launch_tile<Tc64x64, LayoutB, Epi>,     launch_tile<Tc64x64s, LayoutB, Epi>,
                               launch_tile<Tc128x128, LayoutB, Epi>,   launch_tile<Tc64x128s, LayoutB, Epi>,
                               launch_tile<Tc128x64s, LayoutB, Epi>,   launch_tile<Tc128x128w8, LayoutB, Epi>,
                               launch_tile<Tc128x256, LayoutB, Epi>,   launch_tile<Tc256x128, LayoutB, Epi>};
  constexpr int kFns = sizeof fns / sizeof fns[0];
  static_assert(kFns == kTf32Tiles);
  if (pinned_tf32_tile() >= 0) return fns[pinned_tf32_tile()](name, M, N, K, A, B, epi);
  static std::map<std::tuple<int, int, int>, int> best;
  const cudaStream_t st = ExecSpace().cuda_stream();
  const std::tuple<int, int, int> key{32 - __builtin_clz(unsigned(M)), N, K};
  auto it = best.find(key);
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(st, &cap);
  if (it == best.end() && cap != cudaStreamCaptureStatusNone) return fns[2](name, M, N, K, A, B, epi);
  if (it == best.end()) {
    float* saved = nullptr;
    if (keep) {
      cudaMallocAsync(&saved, n * sizeof(float), st);
      cudaMemcpyAsync(saved, keep, n * sizeof(float), cudaMemcpyDeviceToDevice, st);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float fastest = 1e30f;
    int pick = 2;
    for (int rep = 0; rep < 2; ++rep)
      for (int i = 0; i < kFns; ++i) {
        if (keep) cudaMemcpyAsync(keep, saved, n * sizeof(float), cudaMemcpyDeviceToDevice, st);
        cudaEventRecord(e0, st);
        fns[i](name, M, N, K, A, B, epi);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (rep == 1 && ms < fastest) fastest = ms, pick = i;  // the first round warms up
      }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (keep) {
      cudaMemcpyAsync(keep, saved, n * sizeof(float), cudaMemcpyDeviceToDevice, st);
      cudaFreeAsync(saved, st);
    }
    it = best.emplace(key, pick).first;
  }
  fns[it->second](name, M, N, K, A, B, epi);
}

// Reported to Kokkos Tools as a kernel called `name`, like any Kokkos kernel.
template <class LayoutB, class Epi>
void launch(const char* name, int M, int N, int K, const float* A, const float* B, Epi epi, float* keep = nullptr,
            std::size_t n = 0) {
  if (M == 0 || N == 0) return;
  if (tensor()) return launch_tc<LayoutB>(name, M, N, K, A, B, epi, keep, n);
  int t = forced_tile();
  if (t < 0) t = tiles<Large>(M, N) >= 2L * sm_count() ? 0 : tiles<Mid>(M, N) >= 2L * sm_count() ? 1 : 2;
  if (t == 0) launch_tile<Large, LayoutB>(name, M, N, K, A, B, epi);
  else if (t == 1) launch_tile<Mid, LayoutB>(name, M, N, K, A, B, epi);
  else launch_tile<Small, LayoutB>(name, M, N, K, A, B, epi);
}

bool aligned(const float* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool enabled() {
  static const bool on = [] {
    const char* e = std::getenv("PET_CUTLASS");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Whether the GEMM can run here at all: CUTLASS on, an fp32 or TF32 mode, and
// for tensor cores the operands' rows 16-byte aligned (k and N multiples of 4).
// That leaves the energy heads' one-column GEMMs to cuBLAS, which runs them in
// fp32: on tensor cores they are what TF32's error in the energy comes from.
bool usable(const View2D& in, const View2D& w) {
  if (!enabled() || !in.span_is_contiguous() || !w.span_is_contiguous()) return false;
  if (gemm_mode() == GemmMode::Native) return true;
  return tensor() && aligned(in.data()) && aligned(w.data()) && w.extent(0) % 4 == 0 && w.extent(1) % 4 == 0;
}

// Contiguous rows, 16-byte aligned, a whole number of 4-wide runs wide.
bool vec_ok(const View2D& v) { return v.span_is_contiguous() && v.extent(1) % 4 == 0 && aligned(v.data()); }
bool vec_ok(const View1D& v) { return aligned(v.data()); }

// L consecutive floats as one access (L = 1, 2, 4).
template <int L>
struct VecOf;
template <>
struct VecOf<1> { using T = float; };
template <>
struct VecOf<2> { using T = float2; };
template <>
struct VecOf<4> { using T = float4; };
template <int L>
__device__ void load(const float* p, float* v) {
  *reinterpret_cast<typename VecOf<L>::T*>(v) = *reinterpret_cast<const typename VecOf<L>::T*>(p);
}
template <int L>
__device__ void store(float* p, const float* v) {
  *reinterpret_cast<typename VecOf<L>::T*>(p) = *reinterpret_cast<const typename VecOf<L>::T*>(v);
}

// pre = acc + b; h(j) = pre(2j) * sigmoid(pre(2j+1)).
struct SwigluFwd {
  float *pre, *h;
  const float* b;
  int N;
  template <int L>
  __device__ void run(int r, int c, const float* v) const {
    float bb[L], p[L], q[L / 2];
    load<L>(b + c, bb);
#pragma unroll
    for (int j = 0; j < L; ++j) p[j] = v[j] + bb[j];
    store<L>(pre + std::size_t(r) * N + c, p);
    if (!h) return;
#pragma unroll
    for (int j = 0; j < L / 2; ++j) q[j] = p[2 * j] * sigmoid(p[2 * j + 1]);
    store<L / 2>(h + std::size_t(r) * (N / 2) + c / 2, q);
  }
};

// acc = h_adj(r, c..c+L-1); pre_adj(2j) = h_adj * sg, pre_adj(2j+1) = h_adj v sg (1 - sg).
struct SwigluBwd {
  const float* pre;
  float* pre_adj;
  int F;
  template <int L>
  __device__ void run(int r, int c, const float* ha) const {
    const std::size_t o = std::size_t(r) * 2 * F + 2 * c;
    float p[2 * L], q[2 * L];
#pragma unroll
    for (int s = 0; s < 2 * L; s += 4) load<4>(pre + o + s, p + s);
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const float v = p[2 * j], sg = sigmoid(p[2 * j + 1]);
      q[2 * j] = ha[j] * sg;
      q[2 * j + 1] = ha[j] * v * sg * (1.0f - sg);
    }
#pragma unroll
    for (int s = 0; s < 2 * L; s += 4) store<4>(pre_adj + o + s, q + s);
  }
};

// p = acc + b; out = silu(p), and pre = p when kept.
struct SiluFwd {
  float *out, *pre;
  const float* b;
  int N;
  template <int L>
  __device__ void run(int r, int c, const float* v) const {
    float p[L], o[L];
#pragma unroll
    for (int j = 0; j < L; ++j) p[j] = v[j];
    if (b) {
      float bb[L];
      load<L>(b + c, bb);
#pragma unroll
      for (int j = 0; j < L; ++j) p[j] += bb[j];
    }
    const std::size_t at = std::size_t(r) * N + c;
    if (pre) store<L>(pre + at, p);
#pragma unroll
    for (int j = 0; j < L; ++j) o[j] = p[j] * sigmoid(p[j]);
    store<L>(out + at, o);
  }
};

// acc = the adjoint of silu's output; grad = acc * silu'(pre).
struct SiluBwd {
  const float* pre;
  float* grad;
  int N;
  template <int L>
  __device__ void run(int r, int c, const float* a) const {
    const std::size_t at = std::size_t(r) * N + c;
    float x[L], g[L];
    load<L>(pre + at, x);
#pragma unroll
    for (int j = 0; j < L; ++j) {
      const float sg = sigmoid(x[j]);
      g[j] = a[j] * sg * (1.0f + x[j] * (1.0f - sg));
    }
    store<L>(grad + at, g);
  }
};

// out = acc + b (+ beta out): the GEMM with nothing fused. A whole run on
// aligned rows is one access; a run the edge cuts (N not a multiple of L), or
// one on unaligned rows, goes element by element.
struct Plain {
  float* out;
  const float* b;
  float beta;
  int N;
  bool vec;  // out and b 16-byte aligned, N a multiple of 4
  template <int L>
  __device__ void run(int r, int c, const float* v) const {
    float* o = out + std::size_t(r) * N + c;
    if (vec) {
      float x[L], y[L];
#pragma unroll
      for (int j = 0; j < L; ++j) x[j] = v[j];
      if (b) {
        load<L>(b + c, y);
#pragma unroll
        for (int j = 0; j < L; ++j) x[j] += y[j];
      }
      if (beta != 0.0f) {
        load<L>(o, y);
#pragma unroll
        for (int j = 0; j < L; ++j) x[j] += beta * y[j];
      }
      store<L>(o, x);
      return;
    }
#pragma unroll
    for (int j = 0; j < L; ++j)
      if (c + j < N) {
        float x = v[j];
        if (b) x += b[c + j];
        if (beta != 0.0f) x += beta * o[j];
        o[j] = x;
      }
  }
};

}  // namespace

bool batch_invariant_gemms() { return enabled() && gemm_mode() == GemmMode::Native; }
void detail::set_cutlass_tile(int tile) { forced_tile() = tile; }
int detail::tf32_tile_count() { return enabled() && sm8x() ? kTf32Tiles : 0; }
void detail::set_tf32_tile(int tile) { pinned_tf32_tile() = tile; }

bool linear_fused(View2D out, View2D in, const View2D& w, bool w_transposed, View1D b, Net beta) {
  if (!usable(in, w) || !out.span_is_contiguous()) return false;
  const int M = out.extent(0), N = out.extent(1), K = in.extent(1);
  const bool vec = N % 4 == 0 && aligned(out.data()) && (!b.data() || aligned(b.data()));
  const Plain epi{out.data(), b.data(), float(beta), N, vec};
  float* keep = beta != Net(0) ? out.data() : nullptr;  // the epilogue reads out
  const std::size_t n = std::size_t(M) * N;
  if (w_transposed) launch<CM>("linear", M, N, K, in.data(), w.data(), epi, keep, n);
  else launch<RM>("linear_bwd", M, N, K, in.data(), w.data(), epi, keep, n);
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

#else

bool batch_invariant_gemms() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(PET_HAVE_CBLAS)
  return false;
#else
  return gemm_mode() == GemmMode::Native;  // the naive loop: one dot product per element
#endif
}
void detail::set_cutlass_tile(int) {}
int detail::tf32_tile_count() { return 0; }
void detail::set_tf32_tile(int) {}

bool linear_fused(View2D, View2D, const View2D&, bool, View1D, Net) { return false; }
bool silu_in_fused(View2D, View2D, View2D, const View2D&, View1D) { return false; }
bool silu_bwd_fused(View2D, View2D, const View2D&, View2D) { return false; }
bool swiglu_in_fused(View2D, View2D, View2D, const View2D&, View1D) { return false; }
bool swiglu_bwd_fused(View2D, View2D, const View2D&, View2D) { return false; }

#endif

}  // namespace pet
