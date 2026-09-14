// Ozaki scheme: fp64-accuracy GEMM on integer tensor cores.
//
// The problem this solves. On a consumer GPU fp64 arithmetic runs at 1/64 of
// fp32, so a model evaluated in PET_PRECISION=fp64 pays for its accuracy in
// wall clock -- measured on an RTX 4080 SUPER, 7.6x for pet-mad-xs and 14.5x for
// pet-mad-s once the neighbour build is taken out of the comparison. The penalty
// grows with model width, which is exactly backwards: the big models are the
// ones worth running accurately.
//
// The Ozaki scheme (Ozaki, Ogita, Oishi & Rump 2012; the integer tensor-core
// form is Ootomo/Uchino/Mukunoki 2024-25) buys that accuracy back. Split each
// operand into several narrow integer "slices", multiply the slices on hardware
// that is fast and EXACT for them, and sum the partial products:
//
//     A = sum_t D^A_t * 2^(-B t) * rowscale       (D integer, |D| < 2^B)
//     B = sum_v D^B_v * 2^(-B v) * colscale
//     A.B = sum_{t,v} (D^A_t . D^B_v) * 2^(-B(t+v)) * scales
//
// Why int8 and not fp32 slices. The scheme's accuracy comes from the
// ACCUMULATION being exact, not merely the products. An fp32 accumulator has 24
// bits, so exactness over k terms needs k * 2^(2B) < 2^24 -- for k = 1024 that
// caps B at 6 bits, wanting ~9 slices per operand and ~45 slice products, which
// is slower than just calling DGEMM. int8 inputs with an int32 accumulator are
// exact for free (k * 127^2 << 2^31 for any k here), allowing B = 7, ~8 slices,
// and running on tensor cores at ~8x SGEMM throughput. That is where the ~14x
// comes from.
//
// What PET makes free. Every GEMM in this network is `activation x weight`:
// linear, linear_silu and linear_bwd are the only three call sites, and each has
// a constant weight as one operand. The slice decomposition of an operand
// depends only on that operand (its per-row/column magnitude), so the WEIGHT
// SIDE CAN BE SPLIT ONCE AT MODEL LOAD and reused for every evaluation forever.
// Only the activation side is split at run time, halving the splitting work and
// removing it entirely from the weights' side of the inner loop.
//
// What this does NOT cover. The attention is a fused flash-attention kernel --
// an online softmax over a register-resident dot product -- not a GEMM, and both
// its matmul-shaped operations (Q.K^T and softmax.V) have two runtime operands,
// so neither the pre-split nor the library-reuse argument applies there. Its
// accuracy is limited by the fp32 softmax accumulation and by fast_exp/fast_log
// (~2^-22), and the fix there is compensated accumulation and accurate
// transcendentals, not this. See PLAN.md.
#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pet/kokkos.hpp"

namespace pet {

// I8View2D (the slice payload) comes from kokkos.hpp, where the Workspace pools
// it. The accumulator is IView2D: `int` is 32-bit on every platform this builds
// for, and reusing the existing int pool keeps the Ozaki path inside the
// no-allocation-per-step rule the rest of the evaluator follows.

// How the Ozaki path is used for a given evaluation.
enum class GemmMode {
  Native,  // the vendor GEMM in whatever `Net` is. The default.
  Ozaki,   // split into int8 slices, accumulate exactly, recombine in fp64.
};

// Bits carried per slice. 7 leaves the sign bit and keeps every partial product
// inside int8 x int8 -> int32 without saturation.
constexpr int kOzakiSliceBits = 7;

// Slices needed to represent an fp64 mantissa in full. 53 / 7 rounded up.
constexpr int kOzakiMaxSlices = 8;

// One operand, decomposed. `slices[t]` holds the t-th 7-bit digit of every
// entry; `scale` is the per-row (or per-column, for a transposed operand) power
// of two that normalised it before splitting.
//
// A weight's decomposition is computed once at load; an activation's is computed
// per call into workspace buffers.
struct OzakiSplit {
  std::vector<I8View2D> slices;  // [n_slices] each [rows, cols]
  // The power of two each entry was divided by before splitting. Length `lines`
  // for a per-line split, or 1 when `uniform` -- see below.
  RView1D scale;
  int n_slices = 0;
  int rows = 0, cols = 0;
  // A single scale for the whole matrix rather than one per row/column.
  //
  // Correctness only requires each operand's scale to be constant along the
  // index the inner product sums over; a scalar satisfies that in BOTH
  // directions, which is what makes it the right choice for a weight. A weight
  // is used transposed by `linear` and untransposed by `linear_bwd`, and those
  // want normalisation along opposite axes -- so a per-line split would have to
  // be stored twice, at 8 slices costing 2x the weight memory again on top of
  // the fp64 original.
  //
  // The price is accuracy: a matrix-wide scale spends mantissa on whichever row
  // is largest. For trained weights the per-row spread is narrow and it costs a
  // bit or two, which one extra slice recovers. Activations do NOT use this --
  // their rows are atoms and edges with genuinely different magnitudes, and they
  // only ever enter untransposed, so the dual-direction problem does not arise.
  bool uniform = false;

  bool valid() const { return n_slices > 0 && !slices.empty(); }
};

// Runtime configuration. Slice count is the accuracy/speed knob: fewer slices
// means fewer tensor-core GEMMs and less of the mantissa retained. 8 reproduces
// fp64 to within its own rounding; 4 gives roughly fp32-and-a-half at half the
// cost; 2 is a fast preview and should not be used for anything conserved.
struct OzakiConfig {
  GemmMode mode = GemmMode::Native;
  int slices = kOzakiMaxSlices;
  // Drop partial products whose combined shift puts them below the target
  // precision: only (t, v) with t + v < `slices` contribute. Turning this off
  // computes the full slices^2 grid, which is a correctness reference rather
  // than something to run.
  bool triangular = true;
};

// Process-wide configuration, initialised from the environment:
//   PET_GEMM_MODE=native|ozaki
//   PET_OZAKI_SLICES=1..8
// A Calculator applies its Options over the top at construction.
OzakiConfig& ozaki_config();

// True when this build can actually run the Ozaki path (CUDA with integer
// tensor-core GEMM available). Everything else falls back to the native GEMM,
// so asking for Ozaki is never an error -- but `ozaki_active()` says whether it
// is doing anything, and a benchmark should report it.
bool ozaki_available();
inline bool ozaki_active() {
  return ozaki_config().mode == GemmMode::Ozaki && ozaki_available();
}

// Decompose `src` into `cfg.slices` int8 slices, row-normalised. Used at load
// time for weights (into owned Views) and per call for activations (into
// workspace buffers keyed by `key`).
//
// `by_column` splits along columns instead of rows, for an operand that enters
// the product transposed -- the normalisation has to follow the direction the
// inner product sums over, or the slices of one operand do not line up with the
// other's.
OzakiSplit ozaki_split(Workspace& ws, const std::string& key, const View2D& src, int n_slices,
                       bool by_column);

// As above but owning its buffers and using a single matrix-wide scale, for a
// weight that is split once at load and then used in either direction. See
// OzakiSplit::uniform.
//
// `transposed` decomposes src^T instead. Both layouts are needed because int8
// tensor-core GEMM only accepts operands with the contraction index contiguous
// -- cuBLAS's "TN" case -- and a weight is contracted along opposite axes by
// linear() (transposed) and linear_bwd() (not). Feeding it the other layout
// returns CUBLAS_STATUS_NOT_SUPPORTED rather than running slowly.
//
// The two share a scale, so the transposed slices are a permutation of the same
// digits: no accuracy is lost, only memory. That memory is not trivial -- eight
// int8 slices in two layouts is 16 bytes per weight element against 8 for the
// fp64 weight itself -- so this is done only when the Ozaki path is actually on.
OzakiSplit ozaki_split_weight(const View2D& src, int n_slices, bool transposed = false);

// C = alpha * op(A) * op(B) + beta * C, computed through the slice grid.
// `bsplit` is B's precomputed decomposition (null to split it here). Falls back
// to the native GEMM when the Ozaki path is unavailable, so it is safe to call
// unconditionally.
//
// The scratch it needs -- the activation's slices and the int32 accumulator --
// comes from a pool private to this translation unit, under fixed keys. That is
// sound because no two of these calls are ever live at once: each completes
// before the next begins. It also keeps the signature small enough to drop into
// linear()/linear_bwd() without threading a Workspace through the ~80 call sites
// that use them.
void gemm_ozaki(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                const View2D& C, const OzakiSplit* bsplit);

// Bytes the Ozaki scratch pool is currently holding. A caller sizing a batch has
// to count this alongside the model's own workspace.
std::size_t ozaki_workspace_bytes();

}  // namespace pet
