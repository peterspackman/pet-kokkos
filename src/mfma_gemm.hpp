// The network's fp32 weight GEMMs on AMD's CDNA matrix cores
// (v_mfma_f32_16x16x4f32, exact fp32 products, fp32 sums), with an epilogue of
// the caller's walking each lane's accumulators: the bias, or the elementwise
// neighbours fused in (fused_gemm_hip.cpp). Self-contained HIP, so a standalone
// benchmark can include it.
//
//   out[M, N] = A[M, K] . B,  B = W^T for a weight W [N, K] (NT), or W [K, N] (NN),
// all row-major, rows contiguous, K a multiple of 4 and (NN) N a multiple of 4.
//
// The epilogue: col(c, Col&) loads what columns c..c+3 need whatever the row
// (once a lane), row(r, c, Row&) what output row r's columns c..c+3 need, and
// run(r, c, v, Col, Row) gets the accumulators v[4] and writes.
//
// The product is formed transposed, out^T = B^T A^T: the weight is the
// instruction's A operand and the activation its B, so a lane ends with four
// consecutive columns of one output row -- one 16-byte store, and a SwiGLU
// pair together. Lane l is (c, g) = (l % 16, l / 16); the instruction's A
// [16][4] has A[i][k] in lane i + 16k, B [4][16] has B[k][j] in lane j + 16k,
// and D [16][16] D[i][j] in lane j + 16(i/4), register i%4. Its four k slots
// may take any four of the summed index as long as both operands agree: in a
// tile of 16, slot g of instruction s is k = 4g + s, so a lane reads its four
// k of a K-contiguous row as one 16-byte LDS read.
//
// Every output's k runs in the same order -- tile by tile of 16, four products
// summed in an instruction, then added on -- whatever the block tile, so a
// row's answer does not depend on how many rows the GEMM has, nor on the tile.
//
// A block stages its k tile of both operands in LDS (the next tile's global
// loads in flight meanwhile), and each wave runs its own WM x WN of the block.
// Timed on an MI250X GCD at the network's shapes (k = 256-1024): 32-37 TF/s
// with a plain epilogue, level with rocBLAS's sgemm and hipBLASLt, and over a
// layer's GEMMs within 4% of the best of all rocBLAS's Tensile solutions for
// each shape, against 41.3 TF/s for a loop of nothing but these instructions
// (tools/roofline.md has the table). Slower, all measured: an LDS-free
// variant, each wave loading its own fragments from global memory (10-20%);
// the weight alone loaded straight to registers, the activation's LDS tile
// double-buffered (5-25%); 32-wide k tiles and double-buffered LDS (fewer
// blocks a CU); two k tiles of register prefetch (1%); a persistent grid
// prefetching its next tile during the epilogue (a pet-mad-s evaluation 139 ->
// 155 ms), or spreading each tile's epilogue over the next tile's main loop
// (two sets of accumulators: 2 waves a SIMD, 20-35%).
#pragma once

#include <hip/hip_runtime.h>

namespace pet::mfma_gemm {

typedef float f4 __attribute__((ext_vector_type(4)));

// A block of BM x BN outputs, waves of WM x WN, k in tiles of 16.
template <int BM_, int BN_, int WM_, int WN_>
struct Tile {
  static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_;
  static constexpr int kWaves = (BM / WM) * (BN / WN), kThreads = 64 * kWaves;
  static constexpr int FM = WM / 16, FN = WN / 16;
  static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0);
  static_assert((BM * 4) % kThreads == 0 && (BN * 4) % kThreads == 0);
};

// The LDS layouts, free of bank conflicts for both the stores and the reads.
// A K-contiguous [rows][16] tile (the activation, and an [N, K] weight) keeps
// its rows unpadded with the four 16-byte chunks of row r XOR-swizzled by
// (r / 2) % 4: a store's eight lanes (two rows, four chunks) and a fragment
// read's eight (eight rows, one chunk) then fall on eight distinct 4-bank
// groups. A [16][BN] weight tile (NN), read a float per lane, has its rows
// padded by 4 floats, so a read's g = 0 and g = 1 lanes are 16 banks apart.
__device__ inline int swz(int r, int chunk) { return r * 16 + ((chunk ^ (r >> 1)) & 3) * 4; }

// The activation operand: A's rows as they are, or (a prologue) four values a
// load computed from other buffers, row r's columns k..k+3.
struct Rows {
  const float* A;
  int K;
  __device__ float4 operator()(int r, int k) const { return *reinterpret_cast<const float4*>(A + size_t(r) * K + k); }
};

// Waves a SIMD the compiler must leave room for: the 128 x 128 tile's
// accumulators take 64 registers a lane, and with a small epilogue three of its
// waves fit a SIMD if the rest stay under 104; left alone the compiler took 106.
template <class T, class Epi>
constexpr int min_waves() {
  return T::BM * T::BN >= 128 * 128 && sizeof(typename Epi::Row) <= 16 ? 3 : 1;
}

template <class T, bool NN, class Epi, class Pro>
__global__ void __launch_bounds__(T::kThreads, (min_waves<T, Epi>())) gemm(int M, int N, int K, Pro A,
                                                                         const float* __restrict__ W, Epi epi) {
  constexpr int BM = T::BM, BN = T::BN, FM = T::FM, FN = T::FN, NT = T::kThreads, PB = BN + 4;
  constexpr int AQ = BM * 4 / NT, BQ = BN * 4 / NT;  // 16-byte loads a thread per k tile
  // The operand tiles, and after the main loop each wave's 16 x WN slice of the
  // output on its way to global memory (rows padded by 4 floats).
  constexpr int AS = BM * 16, BS = NN ? 16 * PB : BN * 16, WS = 16 * (T::WN + 4);
  __shared__ float smem[AS + BS > T::kWaves * WS ? AS + BS : T::kWaves * WS];
  float* const As = smem;
  float* const Bs = smem + AS;
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, c = lane % 16, g = lane / 16;
  const int ntiles = (N + BN - 1) / BN;
  const int m0 = (blockIdx.x / ntiles) * BM, n0 = (blockIdx.x % ntiles) * BN;
  const int wm = (wave / (BN / T::WN)) * T::WM, wn = (wave % (BN / T::WN)) * T::WN;

  f4 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f4{0, 0, 0, 0};

  float4 ra[AQ], rb[BQ];
  auto load = [&](int k0) {
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
      const int e = tid + q * NT, r = e / 4, k = k0 + (e % 4) * 4;
      ra[q] = (m0 + r < M && k < K) ? A(m0 + r, k) : float4{0, 0, 0, 0};
    }
#pragma unroll
    for (int q = 0; q < BQ; ++q) {
      const int e = tid + q * NT;
      if constexpr (NN) {
        const int r = e / (BN / 4), n = n0 + (e % (BN / 4)) * 4;
        rb[q] = (k0 + r < K && n < N) ? *reinterpret_cast<const float4*>(W + size_t(k0 + r) * N + n)
                                      : float4{0, 0, 0, 0};
      } else {
        const int r = e / 4, k = k0 + (e % 4) * 4;
        rb[q] = (n0 + r < N && k < K) ? *reinterpret_cast<const float4*>(W + size_t(n0 + r) * K + k)
                                      : float4{0, 0, 0, 0};
      }
    }
  };
  auto store = [&] {
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
      const int e = tid + q * NT;
      *reinterpret_cast<float4*>(As + swz(e / 4, e % 4)) = ra[q];
    }
#pragma unroll
    for (int q = 0; q < BQ; ++q) {
      const int e = tid + q * NT;
      if constexpr (NN) *reinterpret_cast<float4*>(Bs + (e / (BN / 4)) * PB + (e % (BN / 4)) * 4) = rb[q];
      else *reinterpret_cast<float4*>(Bs + swz(e / 4, e % 4)) = rb[q];
    }
  };
  auto compute = [&] {
    float4 af[FM], bf[FN];
#pragma unroll
    for (int i = 0; i < FM; ++i) af[i] = *reinterpret_cast<const float4*>(As + swz(wm + 16 * i + c, g));
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      if constexpr (NN) {
        const float* p = Bs + 4 * g * PB + wn + 16 * j + c;
        bf[j] = float4{p[0], p[PB], p[2 * PB], p[3 * PB]};
      } else {
        bf[j] = *reinterpret_cast<const float4*>(Bs + swz(wn + 16 * j + c, g));
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
          const float a = s == 0 ? bf[j].x : s == 1 ? bf[j].y : s == 2 ? bf[j].z : bf[j].w;
          const float x = s == 0 ? af[i].x : s == 1 ? af[i].y : s == 2 ? af[i].z : af[i].w;
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(a, x, acc[i][j], 0, 0, 0);
        }
  };

  const int nk = (K + 15) / 16;
  load(0);
  for (int t = 0; t < nk; ++t) {
    store();
    __syncthreads();
    if (t + 1 < nk) load((t + 1) * 16);
    compute();
    __syncthreads();
  }

  // The epilogue, 16 output rows of the wave's at a time, through LDS: lane l
  // reads back row l / (WN / 4) + p * (256 / WN) (p = 0, 1, ...) of them, four
  // consecutive columns, so a store writes whole WN-float runs of 256 / WN rows
  // rather than 64-byte pieces of 16 (the accumulators' layout). The epilogue
  // loads its per-column inputs (the bias) once, and a row group's inputs (the
  // output accumulated onto, a saved pre-activation) all before using any, so
  // their latencies overlap rather than add.
  constexpr int WN = T::WN, LPR = WN / 4, RPI = 64 / LPR, NP = 16 / RPI;
  float* const ws = smem + wave * WS;
  const int lc = (lane % LPR) * 4, lr = lane / LPR, col = n0 + wn + lc;
  typename Epi::Col cl;
  if (col < N) epi.col(col, cl);
#pragma unroll
  for (int i = 0; i < FM; ++i) {
    // acc[i][j][q]: out[m0 + wm + 16i + c][n0 + wn + 16j + 4g + q]
#pragma unroll
    for (int j = 0; j < FN; ++j)
      *reinterpret_cast<float4*>(ws + c * (WN + 4) + 16 * j + 4 * g) =
          float4{acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]};
    __builtin_amdgcn_wave_barrier();
    float4 v4[NP];
    typename Epi::Row in[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const int r = m0 + wm + 16 * i + lr + p * RPI;
      v4[p] = *reinterpret_cast<const float4*>(ws + (lr + p * RPI) * (WN + 4) + lc);
      if (r < M && col < N) epi.row(r, col, in[p]);
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const int r = m0 + wm + 16 * i + lr + p * RPI;
      if (r < M && col < N) {
        const float v[4] = {v4[p].x, v4[p].y, v4[p].z, v4[p].w};
        epi.run(r, col, v, cl, in[p]);
      }
    }
  }
}

template <class T, bool NN, class Epi, class Pro>
void launch(hipStream_t st, int M, int N, int K, Pro A, const float* W, Epi epi) {
  const int blocks = ((M + T::BM - 1) / T::BM) * ((N + T::BN - 1) / T::BN);
  hipLaunchKernelGGL((gemm<T, NN, Epi, Pro>), dim3(blocks), dim3(T::kThreads), 0, st, M, N, K, A, W, epi);
}
template <class T, bool NN, class Epi>
void launch(hipStream_t st, int M, int N, int K, const float* A, const float* W, Epi epi) {
  launch<T, NN>(st, M, N, K, Rows{A, K}, W, epi);
}

}  // namespace pet::mfma_gemm
