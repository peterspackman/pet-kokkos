// Multi-head attention over each atom's tokens, forward and backward.
//
// Atom n has tk.count(n) tokens: its own (s = 0) and one per kept edge, rows of
// the token matrix (Tokens, ops.hpp); S, the most any atom has, sizes the
// per-slot values. Per head, query q attends to key k with weight
// softmax_k(scale * q.k + log cf_k), where cf is the key's smooth cutoff factor;
// a key with cf <= 0 (a slot past the atom's tokens, or an edge exactly at its
// cutoff) is left out, and a query on such an edge produces zero.
//
// Each pass runs a team per (atom, head), a thread per token. The team first
// copies the rows every thread reads -- the keys and values, or in the keys pass
// the queries and output adjoints -- into scratch, so they are read from global
// memory once and not by each thread; then each thread holds its own row and
// accumulator in registers, sized exactly by a compile-time head_dim (16, 32,
// 48, 64 and 80, the sizes the published models use; anything else takes a
// generic path sized kMaxHeadDim).
#include "ops.hpp"

#include "pet/gemm.hpp"

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

// Atom n's tokens' columns [col, col+hd) of src into dst [S][hd], the slots
// past its count zero.
template <int HD>
KOKKOS_INLINE_FUNCTION void stage(const Member& m, Net* dst, const View2D& src, const Tokens& tk, int n, int S,
                                  int col, int hd) {
  const int cnt = tk.count(n);
  if constexpr (HD > 0 && HD % 4 == 0) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * (HD / 4)), [&](int t) {
      const int r = t / (HD / 4), c = (t % (HD / 4)) * 4;
      *reinterpret_cast<Net4*>(dst + r * HD + c) = r < cnt ? ld4(&src(tk.row(n, r), col + c)) : Net4{};
    });
  } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * hd), [&](int t) {
      const int r = t / hd;
      dst[t] = r < cnt ? src(tk.row(n, r), col + t % hd) : Net(0);
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

// A head's slice of an atom's tokens: consecutive rows in scratch, or in the
// token matrix its own row and then its edges'.
struct Rows {
  const Net *own, *edges;
  std::size_t stride;
  KOKKOS_INLINE_FUNCTION const Net* operator()(int r) const { return r == 0 ? own : edges + (r - 1) * stride; }
};
KOKKOS_INLINE_FUNCTION Rows staged(const Net* base, int stride) { return {base, base + stride, std::size_t(stride)}; }
KOKKOS_INLINE_FUNCTION Rows rows(const View2D& v, const Tokens& tk, int n, int col) {
  const std::size_t ld = v.stride(0);
  return {v.data() + n * ld + col, v.data() + (tk.N + tk.off(n)) * ld + col, ld};
}
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
  Tokens tk;
  int S, H, hd;
  Net sc;
  bool save;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd, cnt = tk.count(n);
    const int si = (n * H + h) * S + sq;
    if (sq >= cnt) {  // a slot past the atom's tokens: no row
      if (save) ml(si, 0) = kOff, ml(si, 1) = Net(0);
      return;
    }
    const int row_q = tk.row(n, sq);
    if (sq > 0 && off(lcf(sq))) {  // a query on an edge at its cutoff
      for (int d = 0; d < nd; ++d) merged(row_q, qo + d) = Net(0);
      if (save) ml(si, 0) = kOff, ml(si, 1) = Net(0);
      return;
    }
    Net q[CAP], acc[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    for (int d = 0; d < nd; ++d) acc[d] = Net(0);
    Net mx = kOff, l = Net(0);
    for (int k = 0; k < cnt; ++k) {
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
    stage<HD>(m, sk, qkv, tk, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, tk, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, staged(sk, nd), staged(sv, nd), LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, tk, n, D + h * hd), rows(qkv, tk, n, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
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

// A block's atom: its token count and each slot's row (Tokens).
struct Atom {
  const int* off;
  int N, n, cnt;
  __device__ Atom(const int* o, int N_, int n_) : off(o), N(N_), n(n_), cnt(o[n_ + 1] - o[n_] + 1) {}
  __device__ size_t row(int s) const { return s == 0 ? size_t(n) : size_t(N + off[n] + s - 1); }
};
// A head's slice of an atom's tokens in a row-major token matrix, the slots
// past its tokens reading zero.
struct Slice {
  const float* p;
  int ld;
  Atom a;
  __device__ float at(int r, int d) const { return r < a.cnt ? p[a.row(r) * ld + d] : 0.0f; }
  template <int W>
  __device__ void chunk(int r, int d0, float* out) const {
    if (r >= a.cnt) {
      for (int j = 0; j < W; ++j) out[j] = 0.0f;
      return;
    }
    const float* q = p + a.row(r) * ld + d0;
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
__global__ void __launch_bounds__(64) forward(const float* qkv, int ldq, const int* off, int N, const float* cf_seq,
                                              int ldc, float* merged, int ldm, float* ml, int S, int H, float sc,
                                              int save) {
  constexpr int W = HD / 4, DT = HD / 16;
  const int n = blockIdx.x / H, h = blockIdx.x % H, l = threadIdx.x, c = l % 16, g = l / 16, D = H * HD;
  const Atom at(off, N, n);
  const Slice Q{qkv + h * HD, ldq, at}, K{qkv + D + h * HD, ldq, at}, V{qkv + 2 * D + h * HD, ldq, at};
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
    if (i >= at.cnt) {  // a slot past the atom's tokens: no row
      if (save && g == 0) ml[2 * ((size_t) (n * H + h) * S + i)] = -1e30f, ml[2 * ((size_t) (n * H + h) * S + i) + 1] = 0.0f;
      continue;
    }
    const bool live = i == 0 || cf[i] > 0.0f;  // else an edge at its cutoff: zero out
    const float invl = live && tot > 0.0f ? 1.0f / tot : 0.0f;
    float* o = merged + at.row(i) * ldm + h * HD;
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
__global__ void __launch_bounds__(64) backward(const float* qkv, int ldq, const int* off, int N, const float* dmerged,
                                               const float* merged, int ldm, const float* ml, const float* cf_seq,
                                               int ldc, float* qkv_adj, int lda, float* cfh, int S, int H, float sc) {
  constexpr int W = HD / 4, DT = HD / 16;
  __shared__ float s_m[16 * NB], s_l[16 * NB], s_dq[16 * NB], s_lc[16 * NB];
  const int n = blockIdx.x / H, h = blockIdx.x % H, l = threadIdx.x, c = l % 16, g = l / 16, D = H * HD;
  const Atom at(off, N, n);
  const Slice Q{qkv + h * HD, ldq, at}, K{qkv + D + h * HD, ldq, at}, V{qkv + 2 * D + h * HD, ldq, at};
  const Slice G{dmerged + h * HD, ldm, at}, O{merged + h * HD, ldm, at};
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
      if (i >= at.cnt) continue;
      float* o = qkv_adj + at.row(i) * lda + h * HD;
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
      if (j >= at.cnt) {  // no row; its cf share is zero
        if (g == 0) cfh[(size_t) (n * S + j) * H + h] = 0.0f;
        continue;
      }
      float* o = qkv_adj + at.row(j) * lda + D + h * HD;
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

#if defined(KOKKOS_ENABLE_CUDA) && !defined(PET_KOKKOS_FP64)
#define PET_TC 1
// On NVIDIA from compute capability 8.0, with PET_ATTN=tf32 (AttentionMode),
// the per-(atom, head) products on TF32 tensor cores (mma m16n8k8): a block per
// (atom, head), a warp per 16 queries (or keys), the shared rows staged once,
// rounded to TF32. The instruction's lanes, l = (g, t) = (l/4, l%4):
//   A [16][8]: a0 (g, t), a1 (g+8, t), a2 (g, t+4), a3 (g+8, t+4);
//   B [8][8]:  b0 (t, g), b1 (t+4, g);
//   C [16][8]: c0 (g, 2t), c1 (g, 2t+1), c2 (g+8, 2t), c3 (g+8, 2t+1).
// A product's eight k-slots may take the summed index in any order A and B
// agree on. Taking slot t as key (or query) 2t and slot t+4 as 2t+1, a score
// tile's C registers are, as they are, the A operand of the next product
// (P V, dS K, ...): (c0, c2, c1, c3).
namespace tc {

__device__ inline float tf32(float x) {
  std::uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r);
}
// c += A B, the operands already TF32.
__device__ inline void mma(float (&c)[4], float a0, float a1, float a2, float a3, float b0, float b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)), "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
        "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)));
#endif
}
__device__ inline float quad_max(float x) {  // over the four lanes with this g
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
__device__ inline float quad_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x + __shfl_xor_sync(0xffffffffu, x, 2);
}
// Rows' pitch in shared memory: head_dim + 4 puts a fragment's eight rows on
// different banks.
template <int HD>
constexpr int pitch = HD + 4;
// A block's atom: its token count and each slot's row (Tokens).
struct Atom {
  const int* off;
  int N, n, cnt;
  __device__ Atom(const int* o, int N_, int n_) : off(o), N(N_), n(n_), cnt(o[n_ + 1] - o[n_] + 1) {}
  __device__ size_t row(int s) const { return s == 0 ? size_t(n) : size_t(N + off[n] + s - 1); }
};
// Slots [0, R) of a head's slice at column col of a row-major token matrix into
// dst, rounded to TF32, slots past the atom's tokens zero.
template <int HD>
__device__ void stage(float* dst, const float* src, int ld, int col, const Atom& at, int R) {
  for (int i = threadIdx.x; i < R * (HD / 4); i += blockDim.x) {
    const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < at.cnt) x = *reinterpret_cast<const float4*>(src + at.row(r) * ld + col + c);
    *reinterpret_cast<float4*>(dst + r * pitch<HD> + c) = make_float4(tf32(x.x), tf32(x.y), tf32(x.z), tf32(x.w));
  }
}
__device__ inline float log_cf(const float* cf, int j, int S) {
  if (j >= S) return -1e30f;
  const float c = cf[j];
  return c > 0.0f ? __logf(c) : -1e30f;
}

// Shared memory, in floats, for S slots: two head slices of rows (K and V; in
// the backward's keys pass, Q and dO in their place) and per-token values. A
// block uses what its atom's tokens need.
template <int HD>
std::size_t forward_floats(int S) { return std::size_t((S + 15) / 16 * 16) * (2 * pitch<HD> + 1); }
template <int HD>
std::size_t backward_floats(int S) { return std::size_t((S + 15) / 16 * 16) * (2 * pitch<HD> + 4); }

// The forward, online softmax over blocks of 8 keys. Warp w: queries 16w + g
// (row 0 of its C tiles) and 16w + g + 8 (row 1).
template <int HD>
__global__ void forward(const float* qkv, int ldq, const int* off, int N, const float* cf_seq, int ldc, float* merged,
                        int ldm, float* ml, int ldml, int S, int H, float sc) {
  constexpr int P = pitch<HD>, DK = HD / 8;
  extern __shared__ float smem[];
  const int n = blockIdx.x / H, h = blockIdx.x % H, D = H * HD;
  const Atom at(off, N, n);
  const int R = (at.cnt + 15) / 16 * 16;
  float *sk = smem, *sv = sk + R * P, *slc = sv + R * P;
  const float* cf = cf_seq + (size_t) n * ldc;
  stage<HD>(sk, qkv, ldq, D + h * HD, at, R);
  stage<HD>(sv, qkv, ldq, 2 * D + h * HD, at, R);
  for (int r = threadIdx.x; r < R; r += blockDim.x) slc[r] = log_cf(cf, r, at.cnt);
  for (int q = at.cnt + threadIdx.x; ml && q < S; q += blockDim.x) {  // the slots past its tokens
    float* m = ml + (size_t) ((n * H + h) * S + q) * ldml;
    m[0] = -1e30f, m[1] = 0.0f;
  }
  const int w = threadIdx.x / 32, l = threadIdx.x % 32, g = l / 4, t = l % 4;
  const int qr[2] = {16 * w + g, 16 * w + g + 8};
  float qa[DK][4];
  for (int c = 0; c < DK; ++c)
    for (int j = 0; j < 4; ++j) {
      const int q = qr[j % 2], d = 8 * c + t + 4 * (j / 2);
      qa[c][j] = q < at.cnt ? tf32(qkv[at.row(q) * ldq + h * HD + d]) : 0.0f;
    }
  __syncthreads();
  if (16 * w >= at.cnt) return;  // a warp past the atom's tokens
  float acc[DK][4], mx[2] = {-1e30f, -1e30f}, sum[2] = {0.0f, 0.0f};
  for (int c = 0; c < DK; ++c)
    for (int j = 0; j < 4; ++j) acc[c][j] = 0.0f;
  for (int J = 0; J < R / 8; ++J) {
    const int k0 = 8 * J + 2 * t;  // this lane's keys: k0 and k0 + 1
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = 0; c < DK; ++c)
      mma(s, qa[c][0], qa[c][1], qa[c][2], qa[c][3], sk[(8 * J + g) * P + 8 * c + t], sk[(8 * J + g) * P + 8 * c + t + 4]);
    const float lc[2] = {slc[k0], slc[k0 + 1]};
    float p[4];
    for (int i = 0; i < 2; ++i) {  // query row i: s[2i], s[2i+1] against k0, k0 + 1
      float x[2];
      for (int j = 0; j < 2; ++j) x[j] = lc[j] > -1e29f ? s[2 * i + j] * sc + lc[j] : -1e30f;
      const float nm = fmaxf(mx[i], quad_max(fmaxf(x[0], x[1])));
      const float alpha = __expf(mx[i] - nm);
      for (int j = 0; j < 2; ++j) p[2 * i + j] = lc[j] > -1e29f ? __expf(x[j] - nm) : 0.0f;
      sum[i] = sum[i] * alpha + p[2 * i] + p[2 * i + 1];
      mx[i] = nm;
      for (int c = 0; c < DK; ++c) acc[c][2 * i] *= alpha, acc[c][2 * i + 1] *= alpha;
    }
    const float a0 = tf32(p[0]), a1 = tf32(p[2]), a2 = tf32(p[1]), a3 = tf32(p[3]);
    for (int c = 0; c < DK; ++c)
      mma(acc[c], a0, a1, a2, a3, sv[k0 * P + 8 * c + g], sv[(k0 + 1) * P + 8 * c + g]);
  }
  for (int i = 0; i < 2; ++i) {
    const int q = qr[i];
    const float tot = quad_sum(sum[i]);
    if (q >= at.cnt) continue;
    const bool live = q == 0 || cf[q] > 0.0f;  // else an edge at its cutoff: zero out
    const float invl = live && tot > 0.0f ? 1.0f / tot : 0.0f;
    float* o = merged + at.row(q) * ldm + h * HD;
    for (int c = 0; c < DK; ++c)  // acc[c][2i + j]: query q, dim 8c + 2t + j
      *reinterpret_cast<float2*>(o + 8 * c + 2 * t) = make_float2(acc[c][2 * i] * invl, acc[c][2 * i + 1] * invl);
    if (ml && t == 0) {
      float* m = ml + (size_t) ((n * H + h) * S + q) * ldml;
      m[0] = live ? mx[i] : -1e30f, m[1] = invl;
    }
  }
}

// The backward, with A recomputed from the saved max and 1/sum, dP = dO V^T,
// dS = A (dP - Dq) and Dq = dO . O:
//   queries (warp w: queries 16w + g, + 8; keys in blocks of 8): dQ = sc dS K;
//   keys (warp w: keys 16w + g, + 8; queries in blocks of 8): dK = sc dS^T Q,
//     dV = A^T dO, and each key's dS summed over queries (its cf share).
// Each pass reads two slices from shared memory and the other two only as its
// warp's own fragments, so one buffer holds K and V, then Q and dO: two blocks
// fit an SM where four slices would leave one.
template <int HD>
__global__ void backward(const float* qkv, int ldq, const int* off, int N, const float* dmerged, const float* merged,
                         int ldm, const float* ml, int ldml, const float* cf_seq, int ldc, float* qkv_adj, int lda,
                         float* cfh, int S, int H, float sc) {
  constexpr int P = pitch<HD>, DK = HD / 8;
  extern __shared__ float smem[];
  const int n = blockIdx.x / H, h = blockIdx.x % H, D = H * HD;
  const Atom at(off, N, n);
  const int R = (at.cnt + 15) / 16 * 16;
  float *sk = smem, *sv = sk + R * P, *sq = sk, *sg = sv;  // K, V, then Q, dO
  float *sm = sv + R * P, *sl = sm + R, *sd = sl + R, *slc = sd + R;
  const float* cf = cf_seq + (size_t) n * ldc;
  stage<HD>(sk, qkv, ldq, D + h * HD, at, R);
  stage<HD>(sv, qkv, ldq, 2 * D + h * HD, at, R);
  for (int q = at.cnt + threadIdx.x; q < S; q += blockDim.x) cfh[(size_t) (n * S + q) * H + h] = 0.0f;
  for (int r = threadIdx.x; r < R; r += blockDim.x) {
    float dq = 0.0f, m = -1e30f, li = 0.0f;
    if (r < at.cnt) {  // Dq in fp32, from the unrounded rows
      const float *a = dmerged + at.row(r) * ldm + h * HD, *b = merged + at.row(r) * ldm + h * HD;
      for (int d = 0; d < HD; d += 4) {
        const float4 x = *reinterpret_cast<const float4*>(a + d), y = *reinterpret_cast<const float4*>(b + d);
        dq += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
      }
      const float* mm = ml + (size_t) ((n * H + h) * S + r) * ldml;
      m = mm[0], li = mm[1];
    }
    sd[r] = dq, sm[r] = m, sl[r] = li, slc[r] = log_cf(cf, r, at.cnt);
  }
  __syncthreads();
  const int w = threadIdx.x / 32, l = threadIdx.x % 32, g = l / 4, t = l % 4;
  const int r0 = 16 * w + g, rr[2] = {r0, r0 + 8};
  const bool busy = 16 * w < at.cnt;  // else a warp past the atom's tokens, there for the barriers
  // A fragments of rows r0, r0 + 8: of a staged slice, or from global memory.
  auto frags = [&](const float* x, float (&f)[DK][4]) {
    for (int c = 0; c < DK; ++c)
      for (int j = 0; j < 4; ++j) f[c][j] = x[rr[j % 2] * P + 8 * c + t + 4 * (j / 2)];
  };
  auto global_frags = [&](const float* x, int ld, float (&f)[DK][4]) {
    for (int c = 0; c < DK; ++c)
      for (int j = 0; j < 4; ++j) {
        const int r = rr[j % 2];
        f[c][j] = r < at.cnt ? tf32(x[at.row(r) * ld + 8 * c + t + 4 * (j / 2)]) : 0.0f;
      }
  };
  float ka[DK][4], va[DK][4];  // the keys pass's own rows, taken before Q and dO replace them

  if (busy) {  // Queries.
    float qa[DK][4], ga[DK][4], dq[DK][4];
    global_frags(qkv + h * HD, ldq, qa), global_frags(dmerged + h * HD, ldm, ga);
    for (int c = 0; c < DK; ++c)
      for (int j = 0; j < 4; ++j) dq[c][j] = 0.0f;
    const float mi[2] = {sm[rr[0]], sm[rr[1]]}, li[2] = {sl[rr[0]], sl[rr[1]]}, di[2] = {sd[rr[0]], sd[rr[1]]};
    for (int J = 0; J < R / 8; ++J) {
      const int k0 = 8 * J + 2 * t, kb = 8 * J + g;
      float s[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int c = 0; c < DK; ++c) {
        mma(s, qa[c][0], qa[c][1], qa[c][2], qa[c][3], sk[kb * P + 8 * c + t], sk[kb * P + 8 * c + t + 4]);
        mma(dp, ga[c][0], ga[c][1], ga[c][2], ga[c][3], sv[kb * P + 8 * c + t], sv[kb * P + 8 * c + t + 4]);
      }
      float ds[4];
      for (int e = 0; e < 4; ++e) {  // query rr[e / 2], key k0 + e % 2
        const int i = e / 2;
        const float lc = slc[k0 + e % 2];
        const float a = li[i] > 0.0f && lc > -1e29f ? __expf(s[e] * sc + lc - mi[i]) * li[i] : 0.0f;
        ds[e] = tf32(a * (dp[e] - di[i]));
      }
      for (int c = 0; c < DK; ++c)
        mma(dq[c], ds[0], ds[2], ds[1], ds[3], sk[k0 * P + 8 * c + g], sk[(k0 + 1) * P + 8 * c + g]);
    }
    for (int i = 0; i < 2; ++i) {
      if (rr[i] >= at.cnt) continue;
      float* o = qkv_adj + at.row(rr[i]) * lda + h * HD;
      for (int c = 0; c < DK; ++c)
        *reinterpret_cast<float2*>(o + 8 * c + 2 * t) = make_float2(dq[c][2 * i] * sc, dq[c][2 * i + 1] * sc);
    }
  }
  if (busy) frags(sk, ka), frags(sv, va);
  __syncthreads();
  stage<HD>(sq, qkv, ldq, h * HD, at, R);
  stage<HD>(sg, dmerged, ldm, h * HD, at, R);
  __syncthreads();

  if (busy) {  // Keys.
    float dk[DK][4], dv[DK][4];
    for (int c = 0; c < DK; ++c)
      for (int j = 0; j < 4; ++j) dk[c][j] = dv[c][j] = 0.0f;
    const float lc[2] = {slc[rr[0]], slc[rr[1]]};
    float cfa[2] = {0.0f, 0.0f};
    for (int I = 0; I < R / 8; ++I) {
      const int q0 = 8 * I + 2 * t, qb = 8 * I + g;
      float s[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int c = 0; c < DK; ++c) {
        mma(s, ka[c][0], ka[c][1], ka[c][2], ka[c][3], sq[qb * P + 8 * c + t], sq[qb * P + 8 * c + t + 4]);
        mma(dp, va[c][0], va[c][1], va[c][2], va[c][3], sg[qb * P + 8 * c + t], sg[qb * P + 8 * c + t + 4]);
      }
      float a[4], ds[4];
      for (int e = 0; e < 4; ++e) {  // key rr[e / 2], query q0 + e % 2
        const int i = e / 2, q = q0 + e % 2;
        a[e] = sl[q] > 0.0f && lc[i] > -1e29f ? __expf(s[e] * sc + lc[i] - sm[q]) * sl[q] : 0.0f;
        ds[e] = a[e] * (dp[e] - sd[q]);
        cfa[i] += ds[e];
        a[e] = tf32(a[e]), ds[e] = tf32(ds[e]);
      }
      for (int c = 0; c < DK; ++c) {
        mma(dk[c], ds[0], ds[2], ds[1], ds[3], sq[q0 * P + 8 * c + g], sq[(q0 + 1) * P + 8 * c + g]);
        mma(dv[c], a[0], a[2], a[1], a[3], sg[q0 * P + 8 * c + g], sg[(q0 + 1) * P + 8 * c + g]);
      }
    }
    for (int i = 0; i < 2; ++i) {
      const float cft = quad_sum(cfa[i]);
      if (rr[i] >= at.cnt) continue;
      float* o = qkv_adj + at.row(rr[i]) * lda + D + h * HD;
      for (int c = 0; c < DK; ++c) {
        *reinterpret_cast<float2*>(o + 8 * c + 2 * t) = make_float2(dk[c][2 * i] * sc, dk[c][2 * i + 1] * sc);
        *reinterpret_cast<float2*>(o + D + 8 * c + 2 * t) = make_float2(dv[c][2 * i], dv[c][2 * i + 1]);
      }
      if (t == 0) cfh[(size_t) (n * S + rr[i]) * H + h] = cft;
    }
  }
}

// Whether the tensor-core kernels run here: the mode, the device, and shared
// memory for `floats`.
inline bool usable(std::size_t floats) {
  static const int smem_max = [] {
    int dev = 0, major = 0, b = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&b, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return major >= 8 ? b : 0;
  }();
  return attention_mode() == AttentionMode::TF32 && sizeof(float) * floats <= std::size_t(smem_max);
}
// Launched on Kokkos's stream with `bytes` of shared memory, and reported to
// Kokkos Tools as a kernel.
template <class K, class... A>
void run(const char* name, K kernel, int blocks, int threads, std::size_t bytes, A... args) {
  static int set = 0;  // the largest opt-in so far, per kernel
  if (int(bytes) > set) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)), set = int(bytes);
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  kernel<<<blocks, threads, bytes, ExecSpace().cuda_stream()>>>(args...);
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

}  // namespace tc
#endif

// A capacity's dead edge rows of x (Tokens): no atom's, so no kernel above
// writes them; zero, so what reads every row reads numbers.
void zero_dead(View2D x, const Tokens& tk) {
  if (!tk.dead) return;
  const int N = tk.N, W = x.extent(1), E = x.extent(0) - N;
  auto off = tk.off;
  Kokkos::parallel_for(
      "attn_dead", RangePolicy(0, E * W), KOKKOS_LAMBDA(int i) {
        if (i / W >= off(N)) x(N + i / W, i % W) = Net(0);
      });
}

template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
                    const WeightRef& w_out, View1D b_out, const Tokens& tk, int S, int H, int hd,
                    double temperature, bool save) {
  const int D = H * hd, N = tk.N, T = qkv.extent(0);
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", T, D) : ws.tmp(T, D);
  View2D ml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  auto finish = [&] {
    zero_dead(merged, tk);
    linear(out, merged, w_out, b_out);
  };
#if defined(PET_MFMA)
  if constexpr (HD > 0 && HD % 16 == 0)
    if (mfma::with_blocks<HD>(S, [&](auto nb) {
          mfma::run("attention", [&](hipStream_t st) {
            hipLaunchKernelGGL((mfma::forward<HD, decltype(nb)::value>), dim3(N * H), dim3(64), 0, st, qkv.data(),
                               int(qkv.stride(0)), tk.off.data(), N, cf_seq.data(), int(cf_seq.stride(0)),
                               merged.data(), int(merged.stride(0)), save ? ml.data() : nullptr, S, H, float(sc),
                               int(save));
          });
        }))
      return finish();
#endif
#if defined(PET_TC)
  if constexpr (HD > 0 && HD % 8 == 0)
    if (tc::usable(tc::forward_floats<HD>(S))) {
      tc::run("attention", tc::forward<HD>, N * H, (S + 15) / 16 * 32, sizeof(float) * tc::forward_floats<HD>(S),
              (const float*) qkv.data(), int(qkv.stride(0)), (const int*) tk.off.data(), N,
              (const float*) cf_seq.data(), int(cf_seq.stride(0)), merged.data(), int(merged.stride(0)),
              save ? ml.data() : nullptr, save ? int(ml.stride(0)) : 0, S, H, float(sc));
      return finish();
    }
#endif
  launch("attention", Forward<HD>{qkv, cf_seq, merged, ml, tk, S, H, hd, sc, save}, N, H, S,
         2 * std::size_t(S) * hd + S);
  finish();
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
  Tokens tk;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd, cnt = tk.count(n);
    const int si = (n * H + h) * S + sq;
    if (sq >= cnt) return (void) (dq_out(si, 0) = Net(0));  // no row
    const int row_q = tk.row(n, sq);
    const Net m = ml(si, 0), invl = ml(si, 1);
    Net q[CAP], dout[CAP], dq[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    load_row<HD>(dout, &dmerged(row_q, qo), nd);
    const Net Dq = dot_row<HD>(&merged(row_q, qo), dout, nd);
    for (int d = 0; d < nd; ++d) dq[d] = Net(0);
    dq_out(si, 0) = Dq;
    if (invl > Net(0))  // else a query on an edge at its cutoff: no weights, no adjoints
      for (int k = 0; k < cnt; ++k) {
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
    stage<HD>(m, sk, qkv, tk, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, tk, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, staged(sk, nd), staged(sv, nd), LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, tk, n, D + h * hd), rows(qkv, tk, n, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
  }
};

// The keys pass: dK and the key's cf share (DK), and dV (DV), from every query
// of its atom and head (their rows, output adjoints, max, 1/sum and Dq).
template <int HD, bool DK, bool DV>
struct Keys {
  View2D qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh;
  Tokens tk;
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
    const int nd = HD > 0 ? HD : hd, D = H * hd, ko = D + h * hd, vo = 2 * D + h * hd, cnt = tk.count(n);
    if (sk >= cnt) {  // no row; its cf share is zero
      if constexpr (DK) cfh(n * S + sk, h) = Net(0);
      return;
    }
    const int row_k = tk.row(n, sk);
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
      for (int q = 0; q < cnt; ++q) {
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
    if constexpr (DK) cfh(n * S + sk, h) = cf_acc;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd;
    Net* sq_ = scratch(m, level, bytes);
    Net* sg = sq_ + S * nd;
    Net* sm = sg + S * nd;
    Net* sl = sm + S;
    Net* sd = sl + S;
    stage<HD>(m, sq_, qkv, tk, n, S, h * hd, nd);
    stage<HD>(m, sg, dmerged, tk, n, S, h * hd, nd);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
      const int si = (n * H + h) * S + r;
      sm[r] = ml(si, 0), sl[r] = ml(si, 1), sd[r] = dq_out(si, 0);
    });
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sk) {
      token(n, h, sk, staged(sq_, nd), staged(sg, nd), Stats{sm, sl, sd, 1, 1});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sk = i % S, h = (i / S) % H, n = i / (S * H), b = (n * H + h) * S;
    token(n, h, sk, rows(qkv, tk, n, h * hd), rows(dmerged, tk, n, h * hd),
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
  Tokens tk;
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
    const int cnt = tk.count(n);
    // The four rows' slices together, so a thread has all its loads in flight;
    // the slots past the atom's tokens zero.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * (nd / 4)), [&](int t) {
      const int r = t / (nd / 4), c = (t % (nd / 4)) * 4, o = r * P + c, row = r < cnt ? tk.row(n, r) : 0;
      Net4 q{}, k{}, v{}, g{};
      if (r < cnt) {
        q = ld4(&qkv(row, h * hd + c)), k = ld4(&qkv(row, D + h * hd + c));
        v = ld4(&qkv(row, 2 * D + h * hd + c)), g = ld4(&dmerged(row, h * hd + c));
      }
      *reinterpret_cast<Net4*>(sq_ + o) = q, *reinterpret_cast<Net4*>(sk + o) = k;
      *reinterpret_cast<Net4*>(sv + o) = v, *reinterpret_cast<Net4*>(sg + o) = g;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
      const int si = (n * H + h) * S + r;
      const Net cf = cf_seq(n, r);
      slc[r] = cf > Net(0) ? fast_log(cf) : kOff;
      sm[r] = ml(si, 0), sl[r] = ml(si, 1);
      Net dq = Net(0);  // Dq = dO.out, from global memory
      for (int d = 0; d < (r < cnt ? nd : 0); d += 4) {
        const int row = tk.row(n, r);
        const Net4 g = ld4(&dmerged(row, h * hd + d)), o = ld4(&merged(row, h * hd + d));
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
      if (r >= cnt) return;
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
      *reinterpret_cast<Net4*>(&qkv_adj(tk.row(n, r), which * D + h * hd + d)) = {{acc[0] * f, acc[1] * f, acc[2] * f, acc[3] * f}};
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int k) {
      Net acc = Net(0);
      for (int q = 0; q < S; ++q) acc += sds[q * S + k];
      cfh(n * S + k, h) = acc;
    });
  }
};

template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                        View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                        const WeightRef& w_out, const Tokens& tk, int S, int H, int hd, double temperature,
                        Net beta) {
  const int D = H * hd, N = tk.N, T = qkv.extent(0);
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(T, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(T, 3 * D);
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
    zero_dead(qkv_adj, tk);
    linear_bwd(in_adj, qkv_adj, w_in, beta);
  };

#if defined(PET_MFMA)
  if constexpr (HD > 0 && HD % 16 == 0)
    if (mfma::with_blocks<HD>(S, [&](auto nb) {
          mfma::run("attn_bwd", [&](hipStream_t st) {
            hipLaunchKernelGGL((mfma::backward<HD, decltype(nb)::value>), dim3(N * H), dim3(64), 0, st, qkv.data(),
                               int(qkv.stride(0)), tk.off.data(), N, dmerged.data(), merged.data(),
                               int(merged.stride(0)), ml.data(), cf_seq.data(), int(cf_seq.stride(0)), qkv_adj.data(),
                               int(qkv_adj.stride(0)), cfh.data(), S, H, float(sc));
          });
        })) {
      cf_and_input_adjoint();
      return;
    }
#endif
#if defined(PET_TC)
  if constexpr (HD > 0 && HD % 8 == 0)
    if (tc::usable(tc::backward_floats<HD>(S))) {
      tc::run("attn_bwd", tc::backward<HD>, N * H, (S + 15) / 16 * 32, sizeof(float) * tc::backward_floats<HD>(S),
              (const float*) qkv.data(), int(qkv.stride(0)), (const int*) tk.off.data(), N,
              (const float*) dmerged.data(), (const float*) merged.data(), int(merged.stride(0)),
              (const float*) ml.data(), int(ml.stride(0)),
              (const float*) cf_seq.data(), int(cf_seq.stride(0)), qkv_adj.data(), int(qkv_adj.stride(0)),
              cfh.data(), S, H, float(sc));
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
                           Pairs<HD>{qkv, dmerged, merged, ml, cf_seq, qkv_adj, cfh, tk, S, H, hd, sc, 0, pb});
      cf_and_input_adjoint();
      return;
    }
  }
  launch("attn_bwd_queries", Queries<HD>{qkv, dmerged, merged, ml, dq_out, cf_seq, qkv_adj, tk, S, H, hd, sc}, N, H, S,
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
  auto keys = [&](auto tag) { return decltype(tag){qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh, tk, S, H, hd, sc}; };
  if constexpr (split) {
    launch("attn_bwd_keys", keys(Keys<HD, true, false>{}), N, H, S, kv);
    launch("attn_bwd_values", keys(Keys<HD, false, true>{}), N, H, S, kv);
  } else {
    launch("attn_bwd_keys", keys(Keys<HD, true, true>{}), N, H, S, kv);
  }

  cf_and_input_adjoint();
}

}  // namespace

bool detail::tf32_attention_available() {
#if defined(PET_TC)
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  return major >= 8;
#else
  return false;
#endif
}

void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, const Tokens& tk, int S, int heads, int head_dim,
               double temperature, bool save) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_impl<decltype(hd)::value>(ws, key, out, qkv, cf_seq, w_out, b_out, tk, S, heads,
                                        head_dim, temperature, save);
  });
}

void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, const Tokens& tk, int S, int heads, int head_dim,
                   double temperature, Net beta) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_bwd_impl<decltype(hd)::value>(ws, key, in_adj, cf_seq_adj, out_adj, qkv, cf_seq,
                                            w_in, w_out, tk, S, heads, head_dim, temperature, beta);
  });
}

}  // namespace pet
