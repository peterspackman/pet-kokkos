// Multi-head attention over each atom's tokens, forward and backward.
//
// Atom n has S = M+1 tokens: its own (s = 0) and one per neighbour slot. Per
// head, query q attends to key k with weight softmax_k(scale * q.k + log cf_k),
// where cf is the key's smooth cutoff factor; a key with cf <= 0 (padding, or an
// edge exactly at its cutoff) is left out, and a padding query (s > 0, cf <= 0)
// produces zero.
//
// Each pass runs a team per (atom, head), a thread per token. The team first
// copies the rows every thread reads -- the keys and values, or in the keys pass
// the queries and output adjoints -- into scratch, so they are read from global
// memory once and not by each thread; then each thread holds its own row and
// accumulator in registers, sized exactly by a compile-time head_dim (16, 32,
// 48, 64 and 80, the sizes the published models use; anything else takes a
// generic path sized kMaxHeadDim).
#include "ops.hpp"

#include <algorithm>
#include <type_traits>

namespace pet {

namespace {

// Call f with head_dim as a compile-time constant, 0 meaning "generic".
template <class F>
void with_head_dim(int hd, F&& f) {
  switch (hd) {
    case 16: return f(std::integral_constant<int, 16>{});
    case 32: return f(std::integral_constant<int, 32>{});
    case 48: return f(std::integral_constant<int, 48>{});
    case 64: return f(std::integral_constant<int, 64>{});
    case 80: return f(std::integral_constant<int, 80>{});
    default: return f(std::integral_constant<int, 0>{});
  }
}

// Four consecutive values as one load: a head's slice of a row starts on a
// multiple of 4 (head_dim is, and so are the widths), in buffers the pool and
// the scratch align, so the load is one 16-byte instruction rather than four.
// The loops are limited by load instructions, not arithmetic, and the compiler
// cannot see the alignment to do this itself.
struct alignas(4 * sizeof(Net)) Net4 {
  Net v[4];
};
KOKKOS_INLINE_FUNCTION Net4 ld4(const Net* p) { return *reinterpret_cast<const Net4*>(p); }
// Between a row's two uses in the dK pass: read it again rather than keep it in
// registers.
KOKKOS_INLINE_FUNCTION void reread() { asm volatile("" ::: "memory"); }

constexpr Net kOff = -Net(1e30);  // the log cf of a key no query sees
KOKKOS_INLINE_FUNCTION bool off(Net lc) { return lc < -Net(1e29); }

// The passes hold several head-sized arrays per thread. On HIP, bounding the
// team at 256 threads lets the compiler give each thread the registers for them:
// under the default bound of 1024 an MI250X GCD caps a thread at 128 and they
// spill (a pbe0-pet relaxation step 1.9x slower). On CUDA the same bound lets
// nvcc spend registers that occupancy needs more.
#if defined(KOKKOS_ENABLE_HIP)
using AttnPolicy = Kokkos::TeamPolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                      Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
#else
using AttnPolicy = TeamPolicy;
#endif
using Member = AttnPolicy::member_type;

// A team per (atom, head), whole warps (wavefronts) enough for its S tokens, and
// `values` Nets of scratch: fast scratch where they fit, else global (level 1).
struct Teams {
  AttnPolicy policy;
  int level;
  std::size_t bytes;
};
Teams teams(int N, int H, int S, std::size_t values) {
  const std::size_t bytes = sizeof(Net) * values + 16;
  const int T = std::min(256, (S + kLanes - 1) / kLanes * kLanes);
  const int level = bytes <= std::size_t(AttnPolicy::scratch_size_max(0)) ? 0 : 1;
  return {AttnPolicy(N * H, T).set_scratch_size(level, Kokkos::PerTeam(bytes)), level, bytes};
}
KOKKOS_INLINE_FUNCTION Net* scratch(const Member& m, int level, std::size_t bytes) {
  return (Net*) m.team_scratch(level).get_shmem_aligned(bytes - 16, 16);
}

// Rows n*S .. n*S+S-1 of src, columns [col, col+hd), into dst [S][hd].
template <int HD>
KOKKOS_INLINE_FUNCTION void stage(const Member& m, Net* dst, const View2D& src, int n, int S, int col, int hd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * (HD / 4)), [&](int t) {
      const int r = t / (HD / 4), c = (t % (HD / 4)) * 4;
      *reinterpret_cast<Net4*>(dst + r * HD + c) = ld4(&src(n * S + r, col + c));
    });
  } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * hd), [&](int t) {
      dst[t] = src(n * S + t / hd, col + t % hd);
    });
  }
}
// A thread's own slice of a row into registers.
template <int HD>
KOKKOS_INLINE_FUNCTION void load_row(Net* reg, const Net* row, int nd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      for (int j = 0; j < 4; ++j) reg[d + j] = x.v[j];
    }
  } else {
    for (int d = 0; d < nd; ++d) reg[d] = row[d];
  }
}
// Each token's log cf, kOff for a key no query sees.
KOKKOS_INLINE_FUNCTION void stage_log_cf(const Member& m, Net* dst, const View2D& cf_seq, int n, int S) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
    const Net cf = cf_seq(n, r);
    dst[r] = cf > Net(0) ? fast_log(cf) : kOff;
  });
}

// row . reg over a head's slice: row (in memory) four at a time where head_dim
// allows, reg a thread's own registers.
template <int HD>
KOKKOS_INLINE_FUNCTION Net dot_row(const Net* row, const Net* reg, int nd) {
  Net s = Net(0);
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      s += x.v[0] * reg[d] + x.v[1] * reg[d + 1] + x.v[2] * reg[d + 2] + x.v[3] * reg[d + 3];
    }
  } else {
    for (int d = 0; d < nd; ++d) s += row[d] * reg[d];
  }
  return s;
}
// acc = acc * c + p * row, over a head's slice.
template <int HD>
KOKKOS_INLINE_FUNCTION void axpby_row(Net* acc, Net c, Net p, const Net* row, int nd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      for (int j = 0; j < 4; ++j) acc[d + j] = acc[d + j] * c + p * x.v[j];
    }
  } else {
    for (int d = 0; d < nd; ++d) acc[d] = acc[d] * c + p * row[d];
  }
}

// Where a pass reads the rows its threads share. On CUDA a team per (atom,
// head) copies them into scratch first (staged), so each is read from global
// memory once rather than by every thread: on an RTX 4080 the passes 15-35%
// faster. An MI250X CU has 64 KiB of LDS, which holds six such teams of one
// wavefront -- too few -- so there (and on the host) a flat range of threads
// reads them through L1 instead.
#if defined(KOKKOS_ENABLE_CUDA)
constexpr bool kStaged = true;
#else
constexpr bool kStaged = false;
#endif
#if defined(KOKKOS_ENABLE_HIP)
using FlatPolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                       Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
#else
using FlatPolicy = RangePolicy;
#endif

// A head's slice of S consecutive rows, in scratch or in a global array.
struct Rows {
  const Net* base;
  std::size_t stride;
  KOKKOS_INLINE_FUNCTION const Net* operator()(int r) const { return base + r * stride; }
};
KOKKOS_INLINE_FUNCTION Rows rows(const View2D& v, int row0, int col) { return {&v(row0, col), v.stride(0)}; }
// Each key's log cf: staged, or from the cutoff factors.
struct LogCf {
  const Net* staged;
  const Net* cf;
  KOKKOS_INLINE_FUNCTION Net operator()(int k) const {
    if (staged) return staged[k];
    return cf[k] > Net(0) ? fast_log(cf[k]) : kOff;
  }
};

// Runs pass f over every (atom, head, token): staged teams, or a flat range.
template <class F>
void launch(const char* name, F f, int N, int H, int S, std::size_t scratch_values) {
  if constexpr (kStaged) {
    const Teams t = teams(N, H, S, scratch_values);
    f.level = t.level, f.bytes = t.bytes;
    Kokkos::parallel_for(name, t.policy, f);
  } else {
    Kokkos::parallel_for(name, FlatPolicy(0, N * H * S), f);
  }
}

// The forward: online softmax, one pass over the keys. Saves the output
// (key:merged) and each query's running max and 1/sum (key:ml) when `save`.
template <int HD>
struct Forward {
  View2D qkv, cf_seq, merged, ml;
  int S, H, hd;
  Net sc;
  bool save;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd;
    const int row_q = n * S + sq, si = (n * H + h) * S + sq;
    if (sq > 0 && off(lcf(sq))) {  // padding query
      for (int d = 0; d < nd; ++d) merged(row_q, qo + d) = Net(0);
      if (save) ml(si, 0) = kOff, ml(si, 1) = Net(0);
      return;
    }
    Net q[CAP], acc[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    for (int d = 0; d < nd; ++d) acc[d] = Net(0);
    Net mx = kOff, l = Net(0);
    for (int k = 0; k < S; ++k) {
      const Net lc = lcf(k);
      if (off(lc)) continue;
      const Net s = dot_row<HD>(K(k), q, nd) * sc + lc;
      const Net nm = s > mx ? s : mx;
      const Net c = fast_exp(mx - nm), p = fast_exp(s - nm);
      l = l * c + p;
      axpby_row<HD>(acc, c, p, V(k), nd);
      mx = nm;
    }
    const Net invl = l > Net(0) ? Net(1) / l : Net(0);
    for (int d = 0; d < nd; ++d) merged(row_q, qo + d) = acc[d] * invl;
    if (save) ml(si, 0) = mx, ml(si, 1) = invl;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd, D = H * hd;
    Net* sk = scratch(m, level, bytes);
    Net* sv = sk + S * nd;
    Net* slc = sv + S * nd;
    stage<HD>(m, sk, qkv, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, Rows{sk, std::size_t(nd)}, Rows{sv, std::size_t(nd)}, LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, n * S, D + h * hd), rows(qkv, n * S, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
  }
};

#if defined(KOKKOS_ENABLE_HIP) && !defined(PET_KOKKOS_FP64) &&                                     \
    (defined(KOKKOS_ARCH_AMD_GFX908) || defined(KOKKOS_ARCH_AMD_GFX90A) || defined(KOKKOS_ARCH_AMD_GFX940) || \
     defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU))
#define PET_MFMA 1
// On AMD's CDNA GPUs the per-(atom, head) products run on the fp32 matrix cores
// (v_mfma_f32_16x16x4f32, exact fp32 products): one wavefront per (atom, head),
// tokens in blocks of 16. On an MI250X GCD this halves attention's time
// (pet-mols-s 2592 water 101 -> 82 ms; tested on gfx90a only). The
// instruction's lanes, measured on a GCD:
//   A [16][4]: A[i][k] in lane i + 16k;   B [4][16]: B[k][j] in lane j + 16k;
//   D [16][16]: D[i][j] in lane j + 16(i/4), register i%4.
// Lane l is (c, g) = (l%16, l/16). A product's four k-slots may take any four
// of the summed index as long as A and B agree, so in a dot over the head a
// lane's slot g walks its own contiguous slice [g W, g W + W), W = head_dim/4.
// Scores are formed transposed, S^T = K Q^T, so lane (c, g) holds query c
// against keys 4g + r: the softmax weights it ends with are then exactly its
// B operand for O^T = V^T P^T, and no shuffle or LDS sits between the two.
namespace mfma {

typedef float f4 __attribute__((ext_vector_type(4)));
__device__ inline f4 mma(float a, float b, f4 c) { return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0); }
__device__ inline float max16(float x) {  // over the lanes with this c (g = 0..3)
  x = fmaxf(x, __shfl_xor(x, 16));
  return fmaxf(x, __shfl_xor(x, 32));
}
__device__ inline float sum16(float x) {
  x += __shfl_xor(x, 16);
  return x + __shfl_xor(x, 32);
}

// A head's slice of S rows of a row-major array, rows past S reading zero.
struct Slice {
  const float* p;
  int ld, S;
  __device__ float at(int r, int d) const { return r < S ? p[(size_t) r * ld + d] : 0.0f; }
  template <int W>
  __device__ void chunk(int r, int d0, float* out) const {
    if (r >= S) {
      for (int j = 0; j < W; ++j) out[j] = 0.0f;
      return;
    }
    const float* q = p + (size_t) r * ld + d0;
    for (int j = 0; j < W; j += 4) {
      const float4 x = *reinterpret_cast<const float4*>(q + j);
      out[j] = x.x, out[j + 1] = x.y, out[j + 2] = x.z, out[j + 3] = x.w;
    }
  }
};
__device__ inline float log_cf(const float* cf, int j, int S) {
  if (j >= S) return -1e30f;
  const float c = cf[j];
  return c > 0.0f ? __logf(c) : -1e30f;
}

// The forward: each lane keeps its query's slice for every query block, and the
// key blocks stream past once, with a running max and sum per query (online
// softmax) -- each key and value row is read once per (atom, head).
template <int HD, int NB>
__global__ void __launch_bounds__(64) forward(const float* qkv, int ldq, const float* cf_seq, int ldc, float* merged,
                                              int ldm, float* ml, int S, int H, float sc, int save) {
  constexpr int W = HD / 4, DT = HD / 16;
  const int n = blockIdx.x / H, h = blockIdx.x % H, l = threadIdx.x, c = l % 16, g = l / 16, D = H * HD;
  const float* rows = qkv + (size_t) n * S * ldq;
  const Slice Q{rows + h * HD, ldq, S}, K{rows + D + h * HD, ldq, S}, V{rows + 2 * D + h * HD, ldq, S};
  const float* cf = cf_seq + (size_t) n * ldc;
  float q[NB][W], mx[NB], sum[NB];
  f4 acc[NB][DT];
  for (int I = 0; I < NB; ++I) {
    Q.chunk<W>(16 * I + c, g * W, q[I]);
    mx[I] = -1e30f, sum[I] = 0.0f;
    for (int t = 0; t < DT; ++t) acc[I][t] = f4{0, 0, 0, 0};
  }
  for (int J = 0; J < NB; ++J) {
    float k[W], v[DT][4], lc[4];
    K.chunk<W>(16 * J + c, g * W, k);
    for (int r = 0; r < 4; ++r) {
      lc[r] = log_cf(cf, 16 * J + 4 * g + r, S);
      for (int t = 0; t < DT; ++t) v[t][r] = V.at(16 * J + 4 * g + r, 16 * t + c);
    }
    for (int I = 0; I < NB; ++I) {
      f4 s = {0, 0, 0, 0};
      for (int t = 0; t < W; ++t) s = mma(k[t], q[I][t], s);  // s[r]: query 16I + c, key 16J + 4g + r
      float x[4], bm = -1e30f;
      for (int r = 0; r < 4; ++r) {
        x[r] = lc[r] > -1e29f ? s[r] * sc + lc[r] : -1e30f;
        bm = fmaxf(bm, x[r]);
      }
      const float nm = fmaxf(mx[I], max16(bm));
      const float alpha = __expf(mx[I] - nm);
      float p[4];
      for (int r = 0; r < 4; ++r) p[r] = lc[r] > -1e29f ? __expf(x[r] - nm) : 0.0f;
      sum[I] = sum[I] * alpha + p[0] + p[1] + p[2] + p[3];
      mx[I] = nm;
      for (int t = 0; t < DT; ++t) {
        acc[I][t] *= alpha;
        for (int r = 0; r < 4; ++r) acc[I][t] = mma(v[t][r], p[r], acc[I][t]);
      }
    }
  }
  for (int I = 0; I < NB; ++I) {
    const int i = 16 * I + c;
    const float tot = sum16(sum[I]);
    if (i >= S) continue;
    const bool live = i == 0 || cf[i] > 0.0f;  // else a padding query: zero out
    const float invl = live && tot > 0.0f ? 1.0f / tot : 0.0f;
    float* o = merged + (size_t) (n * S + i) * ldm + h * HD;
    for (int t = 0; t < DT; ++t)  // acc[I][t][r]: query i, dim 16t + 4g + r
      *reinterpret_cast<float4*>(o + 16 * t + 4 * g) =
          make_float4(acc[I][t][0] * invl, acc[I][t][1] * invl, acc[I][t][2] * invl, acc[I][t][3] * invl);
    if (save && g == 0) {
      float* m = ml + 2 * ((size_t) (n * H + h) * S + i);
      m[0] = live ? mx[I] : -1e30f, m[1] = invl;
    }
  }
}

// The backward of one (atom, head), with A = the weights recomputed from the
// saved max and 1/sum, dP = dO V^T and dS = A (dP - Dq):
//   queries (lane c a query, all query blocks held, key blocks streamed):
//     S^T and dP^T, dQ^T += K^T dS^T;
//   keys (lane c a key, all key blocks held, query blocks streamed): S and dP,
//     dK^T += Q^T dS, dV^T += dO^T A, and the key's dS summed over queries.
template <int HD, int NB>
__global__ void __launch_bounds__(64) backward(const float* qkv, int ldq, const float* dmerged, const float* merged,
                                               int ldm, const float* ml, const float* cf_seq, int ldc, float* qkv_adj,
                                               int lda, float* cfh, int S, int H, float sc) {
  constexpr int W = HD / 4, DT = HD / 16;
  __shared__ float s_m[16 * NB], s_l[16 * NB], s_dq[16 * NB], s_lc[16 * NB];
  const int n = blockIdx.x / H, h = blockIdx.x % H, l = threadIdx.x, c = l % 16, g = l / 16, D = H * HD;
  const float* rows = qkv + (size_t) n * S * ldq;
  const Slice Q{rows + h * HD, ldq, S}, K{rows + D + h * HD, ldq, S}, V{rows + 2 * D + h * HD, ldq, S};
  const Slice G{dmerged + (size_t) n * S * ldm + h * HD, ldm, S}, O{merged + (size_t) n * S * ldm + h * HD, ldm, S};
  const float* cf = cf_seq + (size_t) n * ldc;
  const float* mls = ml + 2 * (size_t) (n * H + h) * S;

  // Queries; on the way, each query's max, 1/sum and Dq = dO . O into LDS.
  {
    float q[NB][W], go[NB][W], mi[NB], li[NB], dqi[NB];
    f4 acc[NB][DT];
    for (int I = 0; I < NB; ++I) {
      const int i = 16 * I + c;
      float oo[W];
      Q.chunk<W>(i, g * W, q[I]);
      G.chunk<W>(i, g * W, go[I]);
      O.chunk<W>(i, g * W, oo);
      float dq = 0.0f;
      for (int t = 0; t < W; ++t) dq += go[I][t] * oo[t];
      dqi[I] = sum16(dq);
      mi[I] = i < S ? mls[2 * i] : -1e30f, li[I] = i < S ? mls[2 * i + 1] : 0.0f;
      if (g == 0) s_dq[i] = dqi[I], s_m[i] = mi[I], s_l[i] = li[I], s_lc[i] = log_cf(cf, i, S);
      for (int t = 0; t < DT; ++t) acc[I][t] = f4{0, 0, 0, 0};
    }
    for (int J = 0; J < NB; ++J) {
      float k[W], v[W], ka[DT][4], lc[4];
      K.chunk<W>(16 * J + c, g * W, k);
      V.chunk<W>(16 * J + c, g * W, v);
      for (int r = 0; r < 4; ++r) {
        lc[r] = log_cf(cf, 16 * J + 4 * g + r, S);
        for (int t = 0; t < DT; ++t) ka[t][r] = K.at(16 * J + 4 * g + r, 16 * t + c);
      }
      for (int I = 0; I < NB; ++I) {
        f4 s = {0, 0, 0, 0}, dp = {0, 0, 0, 0};
        for (int t = 0; t < W; ++t) s = mma(k[t], q[I][t], s), dp = mma(v[t], go[I][t], dp);
        float ds[4];
        for (int r = 0; r < 4; ++r) {  // query 16I + c, key 16J + 4g + r
          const float a = li[I] > 0.0f && lc[r] > -1e29f ? __expf(s[r] * sc + lc[r] - mi[I]) * li[I] : 0.0f;
          ds[r] = a * (dp[r] - dqi[I]);
        }
        for (int t = 0; t < DT; ++t)
          for (int r = 0; r < 4; ++r) acc[I][t] = mma(ka[t][r], ds[r], acc[I][t]);
      }
    }
    for (int I = 0; I < NB; ++I) {
      const int i = 16 * I + c;
      if (i >= S) continue;
      float* o = qkv_adj + (size_t) (n * S + i) * lda + h * HD;
      for (int t = 0; t < DT; ++t)
        *reinterpret_cast<float4*>(o + 16 * t + 4 * g) =
            make_float4(acc[I][t][0] * sc, acc[I][t][1] * sc, acc[I][t][2] * sc, acc[I][t][3] * sc);
    }
  }
  __syncthreads();

  // Keys.
  {
    float k[NB][W], v[NB][W], lcj[NB], cfa[NB];
    f4 dk[NB][DT], dv[NB][DT];
    for (int J = 0; J < NB; ++J) {
      K.chunk<W>(16 * J + c, g * W, k[J]);
      V.chunk<W>(16 * J + c, g * W, v[J]);
      lcj[J] = s_lc[16 * J + c], cfa[J] = 0.0f;
      for (int t = 0; t < DT; ++t) dk[J][t] = dv[J][t] = f4{0, 0, 0, 0};
    }
    for (int I = 0; I < NB; ++I) {
      float q[W], go[W], qa[DT][4], ga[DT][4], mi[4], li[4], dqi[4];
      Q.chunk<W>(16 * I + c, g * W, q);
      G.chunk<W>(16 * I + c, g * W, go);
      for (int r = 0; r < 4; ++r) {
        const int i = 16 * I + 4 * g + r;
        mi[r] = s_m[i], li[r] = s_l[i], dqi[r] = s_dq[i];
        for (int t = 0; t < DT; ++t) qa[t][r] = Q.at(i, 16 * t + c), ga[t][r] = G.at(i, 16 * t + c);
      }
      for (int J = 0; J < NB; ++J) {
        f4 s = {0, 0, 0, 0}, dp = {0, 0, 0, 0};
        for (int t = 0; t < W; ++t) s = mma(q[t], k[J][t], s), dp = mma(go[t], v[J][t], dp);
        float a[4], ds[4];
        for (int r = 0; r < 4; ++r) {  // query 16I + 4g + r, key 16J + c
          a[r] = li[r] > 0.0f && lcj[J] > -1e29f ? __expf(s[r] * sc + lcj[J] - mi[r]) * li[r] : 0.0f;
          ds[r] = a[r] * (dp[r] - dqi[r]);
          cfa[J] += ds[r];
        }
        for (int t = 0; t < DT; ++t)
          for (int r = 0; r < 4; ++r) dk[J][t] = mma(qa[t][r], ds[r], dk[J][t]), dv[J][t] = mma(ga[t][r], a[r], dv[J][t]);
      }
    }
    for (int J = 0; J < NB; ++J) {
      const int j = 16 * J + c;
      const float cft = sum16(cfa[J]);
      if (j >= S) continue;
      float* o = qkv_adj + (size_t) (n * S + j) * lda + D + h * HD;
      for (int t = 0; t < DT; ++t) {  // dk[J][t][r]: key j, dim 16t + 4g + r
        *reinterpret_cast<float4*>(o + 16 * t + 4 * g) =
            make_float4(dk[J][t][0] * sc, dk[J][t][1] * sc, dk[J][t][2] * sc, dk[J][t][3] * sc);
        *reinterpret_cast<float4*>(o + D + 16 * t + 4 * g) =
            make_float4(dv[J][t][0], dv[J][t][1], dv[J][t][2], dv[J][t][3]);
      }
      if (g == 0) cfh[(size_t) (n * S + j) * H + h] = cft;
    }
  }
}

// Calls f with the number of 16-token blocks as a compile-time constant, for
// as many as the kernels' register arrays hold (blocks x head_dim <= 256);
// false when there are more, for the general kernels.
template <int HD, class F>
bool with_blocks(int S, F&& f) {
  const int nb = (S + 15) / 16;
  bool done = false;
  auto try_nb = [&](auto c) {
    constexpr int NB = decltype(c)::value;
    if constexpr (NB * HD <= 256)
      if (nb == NB) f(c), done = true;
  };
  try_nb(std::integral_constant<int, 1>{}), try_nb(std::integral_constant<int, 2>{});
  try_nb(std::integral_constant<int, 3>{}), try_nb(std::integral_constant<int, 4>{});
  try_nb(std::integral_constant<int, 5>{}), try_nb(std::integral_constant<int, 6>{});
  try_nb(std::integral_constant<int, 7>{}), try_nb(std::integral_constant<int, 8>{});
  return done;
}

// Launched on Kokkos's stream, and reported to Kokkos Tools as a kernel.
template <class F>
void run(const char* name, F&& f) {
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  f(ExecSpace().hip_stream());
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

}  // namespace mfma
#endif

template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
                    const WeightRef& w_out, View1D b_out, int N, int S, int H, int hd,
                    double temperature, bool save) {
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", N * S, D) : ws.tmp(N * S, D);
  View2D ml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
#if defined(PET_MFMA)
  if constexpr (HD > 0 && HD % 16 == 0)
    if (mfma::with_blocks<HD>(S, [&](auto nb) {
          mfma::run("attention", [&](hipStream_t st) {
            hipLaunchKernelGGL((mfma::forward<HD, decltype(nb)::value>), dim3(N * H), dim3(64), 0, st, qkv.data(),
                               int(qkv.stride(0)), cf_seq.data(), int(cf_seq.stride(0)), merged.data(),
                               int(merged.stride(0)), save ? ml.data() : nullptr, S, H, float(sc), int(save));
          });
        })) {
      linear(out, merged, w_out, b_out);
      return;
    }
#endif
  launch("attention", Forward<HD>{qkv, cf_seq, merged, ml, S, H, hd, sc, save}, N, H, S, 2 * std::size_t(S) * hd + S);
  linear(out, merged, w_out, b_out);
}

// The backward, as the forward runs: nothing S x S is ever stored. With
// A = softmax weights (recomputed from the saved max m and 1/sum per query) and
// dS = A (dO.v - dO.out), the score adjoint before the scale:
//   queries, per (atom, head, query): dQ = sc sum_k dS K, and dO.out (Dq), kept;
//   keys, per (atom, head, key): dK = sc sum_q dS Q, dV = sum_q A dO, and the
//     key's dS summed over queries, the cf adjoint's share from this head;
//   cf, per (atom, key): those shares, summed over heads.
// Each pass recomputes q.k and dO.v for its pairs. Every sum has one owner and a
// fixed order: nothing is atomic.

template <int HD>
struct Queries {
  View2D qkv, dmerged, merged, ml, dq_out, cf_seq, qkv_adj;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd;
    const int row_q = n * S + sq, si = (n * H + h) * S + sq;
    const Net m = ml(si, 0), invl = ml(si, 1);
    Net q[CAP], dout[CAP], dq[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    load_row<HD>(dout, &dmerged(row_q, qo), nd);
    const Net Dq = dot_row<HD>(&merged(row_q, qo), dout, nd);
    for (int d = 0; d < nd; ++d) dq[d] = Net(0);
    dq_out(si, 0) = Dq;
    if (invl > Net(0))  // else a padding query: no weights, no adjoints
      for (int k = 0; k < S; ++k) {
        const Net lc = lcf(k);
        if (off(lc)) continue;  // padding key
        const Net a = fast_exp(dot_row<HD>(K(k), q, nd) * sc + lc - m) * invl;
        const Net ds = a * (dot_row<HD>(V(k), dout, nd) - Dq) * sc;
        axpby_row<HD>(dq, Net(1), ds, K(k), nd);
      }
    for (int d = 0; d < nd; ++d) qkv_adj(row_q, qo + d) = dq[d];
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd, D = H * hd;
    Net* sk = scratch(m, level, bytes);
    Net* sv = sk + S * nd;
    Net* slc = sv + S * nd;
    stage<HD>(m, sk, qkv, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, Rows{sk, std::size_t(nd)}, Rows{sv, std::size_t(nd)}, LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, n * S, D + h * hd), rows(qkv, n * S, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
  }
};

// The keys pass: dK and the key's cf share (DK), and dV (DV), from every query
// of its atom and head (their rows, output adjoints, max, 1/sum and Dq).
template <int HD, bool DK, bool DV>
struct Keys {
  View2D qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  // Per query: max, 1/sum and Dq, each at its stride.
  struct Stats {
    const Net *m, *l, *d;
    std::size_t sm, sd;
  };
  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sk, Rows Q, Rows G, Stats st) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, D = H * hd, ko = D + h * hd, vo = 2 * D + h * hd;
    const int row_k = n * S + sk;
    Net k[CAP], v[DK ? CAP : 1], dk[DK ? CAP : 1], dv[DV ? CAP : 1];
    Net cf_acc = Net(0);
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) dk[d] = Net(0);
      if constexpr (DV) dv[d] = Net(0);
    }
    const Net cf = cf_seq(n, sk);
    if (cf > Net(0)) {  // else a padding key: no query sees it
      const Net lcf = fast_log(cf);
      load_row<HD>(k, &qkv(row_k, ko), nd);
      if constexpr (DK) load_row<HD>(v, &qkv(row_k, vo), nd);
      for (int q = 0; q < S; ++q) {
        const Net invl = st.l[q * st.sm];
        if (invl <= Net(0)) continue;  // padding query
        const Net a = fast_exp(dot_row<HD>(Q(q), k, nd) * sc + lcf - st.m[q * st.sm]) * invl;
        Net ds = Net(0);
        if constexpr (DK) ds = a * (dot_row<HD>(G(q), v, nd) - st.d[q * st.sd]);
        if constexpr (!DV) reread();
        if constexpr (DK) axpby_row<HD>(dk, Net(1), ds * sc, Q(q), nd);
        if constexpr (DV) axpby_row<HD>(dv, Net(1), a, G(q), nd);
        cf_acc += ds;
      }
    }
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) qkv_adj(row_k, ko + d) = dk[d];
      if constexpr (DV) qkv_adj(row_k, vo + d) = dv[d];
    }
    if constexpr (DK) cfh(row_k, h) = cf_acc;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd;
    Net* sq_ = scratch(m, level, bytes);
    Net* sg = sq_ + S * nd;
    Net* sm = sg + S * nd;
    Net* sl = sm + S;
    Net* sd = sl + S;
    stage<HD>(m, sq_, qkv, n, S, h * hd, nd);
    stage<HD>(m, sg, dmerged, n, S, h * hd, nd);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
      const int si = (n * H + h) * S + r;
      sm[r] = ml(si, 0), sl[r] = ml(si, 1), sd[r] = dq_out(si, 0);
    });
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sk) {
      token(n, h, sk, Rows{sq_, std::size_t(nd)}, Rows{sg, std::size_t(nd)}, Stats{sm, sl, sd, 1, 1});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sk = i % S, h = (i / S) % H, n = i / (S * H), b = (n * H + h) * S;
    token(n, h, sk, rows(qkv, n * S, h * hd), rows(dmerged, n * S, h * hd),
          Stats{&ml(b, 0), &ml(b, 1), &dq_out(b, 0), ml.stride(0), dq_out.stride(0)});
  }
};

// On CUDA, when they fit fast scratch, the backward as small dense products in
// one team of kPairThreads per (atom, head): its queries, keys, values and
// output adjoints staged once, and S x S tables of the weights A and of dS, a
// thread per (query, key) pair; then dQ, dK, dV and the cf shares, a thread per
// output and four dims, each a sum over the S tokens in order. The rows are
// staged with a pitch of head_dim + 4, so a pair phase's threads, which read
// different rows, hit different banks. On an RTX 4080 SUPER, pet-mad-s water
// 648, 773 -> 485 us per layer against the two passes below, at a quarter
// occupancy, held there by the scratch.
constexpr int kPairThreads = 128;

std::size_t shared_per_sm() {
  int b = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&b, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
#endif
  return std::size_t(b);
}

template <int HD>
struct Pairs {
  View2D qkv, dmerged, merged, ml, cf_seq, qkv_adj, cfh;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  static int pitch(int hd) { return hd + 4; }
  static std::size_t values(int S, int hd) { return 4 * std::size_t(S) * pitch(hd) + 2 * std::size_t(S) * S + 4 * S; }

  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd, D = H * hd;
    const int P = HD > 0 ? HD + 4 : hd + 4;
    Net* sq_ = scratch(m, level, bytes);
    Net* sk = sq_ + S * P;
    Net* sv = sk + S * P;
    Net* sg = sv + S * P;
    Net* sa = sg + S * P;  // A[q][k]
    Net* sds = sa + S * S;  // dS[q][k], before the scale
    Net* slc = sds + S * S;
    Net* sm = slc + S;
    Net* sl = sm + S;
    Net* sd = sl + S;
    const int row0 = n * S;
    // The four rows' slices together, so a thread has all its loads in flight.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * (nd / 4)), [&](int t) {
      const int r = t / (nd / 4), c = (t % (nd / 4)) * 4, o = r * P + c;
      const Net4 q = ld4(&qkv(row0 + r, h * hd + c)), k = ld4(&qkv(row0 + r, D + h * hd + c));
      const Net4 v = ld4(&qkv(row0 + r, 2 * D + h * hd + c)), g = ld4(&dmerged(row0 + r, h * hd + c));
      *reinterpret_cast<Net4*>(sq_ + o) = q, *reinterpret_cast<Net4*>(sk + o) = k;
      *reinterpret_cast<Net4*>(sv + o) = v, *reinterpret_cast<Net4*>(sg + o) = g;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
      const int si = (n * H + h) * S + r;
      const Net cf = cf_seq(n, r);
      slc[r] = cf > Net(0) ? fast_log(cf) : kOff;
      sm[r] = ml(si, 0), sl[r] = ml(si, 1);
      Net dq = Net(0);  // Dq = dO.out, from global memory
      for (int d = 0; d < nd; d += 4) {
        const Net4 g = ld4(&dmerged(row0 + r, h * hd + d)), o = ld4(&merged(row0 + r, h * hd + d));
        dq += g.v[0] * o.v[0] + g.v[1] * o.v[1] + g.v[2] * o.v[2] + g.v[3] * o.v[3];
      }
      sd[r] = dq;
    });
    m.team_barrier();
    // A and dS per pair; zero for a padding query or key.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * S), [&](int t) {
      const int q = t / S, k = t % S;
      const Net invl = sl[q], lc = slc[k];
      Net a = Net(0), ds = Net(0);
      if (invl > Net(0) && !off(lc)) {
        Net s = Net(0), dp = Net(0);
        for (int d = 0; d < nd; d += 4) {
          const Net4 x = ld4(sq_ + q * P + d), y = ld4(sk + k * P + d), g = ld4(sg + q * P + d), v = ld4(sv + k * P + d);
          for (int j = 0; j < 4; ++j) s += x.v[j] * y.v[j], dp += g.v[j] * v.v[j];
        }
        a = fast_exp(s * sc + lc - sm[q]) * invl;
        ds = a * (dp - sd[q]);
      }
      sa[t] = a, sds[t] = ds;
    });
    m.team_barrier();
    // dQ = sc dS K, dK = sc dS^T Q, dV = A^T dO: a thread per (token, four dims).
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, 3 * S * (nd / 4)), [&](int t) {
      const int which = t / (S * (nd / 4)), r = (t / (nd / 4)) % S, d = (t % (nd / 4)) * 4;
      // Row r of dS (which 0), or column r of dS or A, against the rows of K, Q or dO.
      const Net* w = which == 0 ? sds + r * S : which == 1 ? sds + r : sa + r;
      const int ws = which == 0 ? 1 : S;
      const Net* x = (which == 0 ? sk : which == 1 ? sq_ : sg) + d;
      Net acc[4] = {Net(0), Net(0), Net(0), Net(0)};
      for (int j = 0; j < S; ++j) {
        const Net c = w[j * ws];
        const Net4 y = ld4(x + j * P);
        for (int i = 0; i < 4; ++i) acc[i] += c * y.v[i];
      }
      const Net f = which == 2 ? Net(1) : sc;
      *reinterpret_cast<Net4*>(&qkv_adj(row0 + r, which * D + h * hd + d)) = {{acc[0] * f, acc[1] * f, acc[2] * f, acc[3] * f}};
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int k) {
      Net acc = Net(0);
      for (int q = 0; q < S; ++q) acc += sds[q * S + k];
      cfh(row0 + k, h) = acc;
    });
  }
};

template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                        View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                        const WeightRef& w_out, int N, int S, int H, int hd, double temperature,
                        Net beta) {
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(N * S, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(N * S, 3 * D);
  View2D cfh = ws.tmp(N * S, H);  // per key and head: the cf adjoint's share
  View2D dq_out = ws.tmp(N * H * S, 1);  // per query and head: dO.out
  View2D merged = ws.peek2(key + ":merged"), ml = ws.peek2(key + ":ml");
  // Last: the cf shares summed over heads, and the input's adjoint.
  auto cf_and_input_adjoint = [&] {
    Kokkos::parallel_for(
        "attn_bwd_cf", RangePolicy(0, N * S), KOKKOS_LAMBDA(int i) {
          const int sk = i % S, n = i / S;
          const Net cf = cf_seq(n, sk);
          if (cf <= Net(0)) return;
          Net total = Net(0);
          for (int h = 0; h < H; ++h) total += cfh(n * S + sk, h);
          cf_seq_adj(n, sk) += total / cf;
        });
    linear_bwd(in_adj, qkv_adj, w_in, beta);
  };

#if defined(PET_MFMA)
  if constexpr (HD > 0 && HD % 16 == 0)
    if (mfma::with_blocks<HD>(S, [&](auto nb) {
          mfma::run("attn_bwd", [&](hipStream_t st) {
            hipLaunchKernelGGL((mfma::backward<HD, decltype(nb)::value>), dim3(N * H), dim3(64), 0, st, qkv.data(),
                               int(qkv.stride(0)), dmerged.data(), merged.data(), int(merged.stride(0)), ml.data(),
                               cf_seq.data(), int(cf_seq.stride(0)), qkv_adj.data(), int(qkv_adj.stride(0)),
                               cfh.data(), S, H, float(sc));
          });
        })) {
      cf_and_input_adjoint();
      return;
    }
#endif
  if constexpr (kStaged) {
    // Where two teams fit an SM's shared memory (with the driver's 1 KiB each),
    // or head_dim is 48 or more, where the two passes' per-thread arrays spill.
    // With one team per SM and a small head the two passes are faster: pbe0-pet
    // (S = 51, head_dim 32) 1.11 against 1.42 ms per layer; but pet-omat-l
    // (S = 49, head_dim 64) 4.9 against 2.3 for the pairs.
    static const std::size_t sm_bytes = shared_per_sm();
    const std::size_t pb = sizeof(Net) * Pairs<HD>::values(S, hd) + 16;
    if (hd % 4 == 0 && pb <= std::size_t(AttnPolicy::scratch_size_max(0)) && (2 * (pb + 1024) <= sm_bytes || hd >= 48)) {
      Kokkos::parallel_for("attn_bwd", AttnPolicy(N * H, kPairThreads).set_scratch_size(0, Kokkos::PerTeam(pb)),
                           Pairs<HD>{qkv, dmerged, merged, ml, cf_seq, qkv_adj, cfh, S, H, hd, sc, 0, pb});
      cf_and_input_adjoint();
      return;
    }
  }
  launch("attn_bwd_queries", Queries<HD>{qkv, dmerged, merged, ml, dq_out, cf_seq, qkv_adj, S, H, hd, sc}, N, H, S,
         2 * std::size_t(S) * hd + S);

  // From head_dim 32, one keys pass holding k, v, dk and dv takes 374 registers
  // on an MI250X GCD and runs one wave per SIMD; as two passes (dK with the cf
  // share, then dV) they run two and four, the keys' share of a pet-mols-s step
  // 40% less. At 16 the one pass already runs two, and on NVIDIA it fits.
#if defined(KOKKOS_ENABLE_HIP)
  constexpr bool split = HD >= 32;
#else
  constexpr bool split = false;
#endif
  const std::size_t kv = 2 * std::size_t(S) * hd + 3 * S;
  auto keys = [&](auto tag) { return decltype(tag){qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh, S, H, hd, sc}; };
  if constexpr (split) {
    launch("attn_bwd_keys", keys(Keys<HD, true, false>{}), N, H, S, kv);
    launch("attn_bwd_values", keys(Keys<HD, false, true>{}), N, H, S, kv);
  } else {
    launch("attn_bwd_keys", keys(Keys<HD, true, true>{}), N, H, S, kv);
  }

  cf_and_input_adjoint();
}

}  // namespace

void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, int N, int S, int heads, int head_dim,
               double temperature, bool save) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_impl<decltype(hd)::value>(ws, key, out, qkv, cf_seq, w_out, b_out, N, S, heads,
                                        head_dim, temperature, save);
  });
}

void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, int N, int S, int heads, int head_dim,
                   double temperature, Net beta) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_bwd_impl<decltype(hd)::value>(ws, key, in_adj, cf_seq_adj, out_adj, qkv, cf_seq,
                                            w_in, w_out, N, S, heads, head_dim, temperature, beta);
  });
}

}  // namespace pet
