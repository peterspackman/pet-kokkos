// The Ozaki scheme: fp64-accurate GEMM on int8 tensor cores (Ozaki, Ogita,
// Oishi & Rump 2012; the integer form Ootomo, Uchino & Mukunoki 2024).
//
// Each operand is scaled by a power of two per row (or column) and split into
// 7-bit integer digits; the digit products are summed exactly in int32 and
// recombined in fp64:
//
//     A = rowscale * sum_t D^A_t 2^(-7(t+1)),   B likewise,
//     A.B = scales * sum_{t,v} (D^A_t . D^B_v) 2^(-7(t+v+2)).
//
// int8 x int8 -> int32 is exact for any k here, which fp32 slices are not. On a
// consumer GPU, where fp64 runs at 1/64 of fp32, that makes an fp64 build's
// GEMMs several times faster. Every GEMM in PET has a weight as one operand, so
// the weights are split once at load and only the activations per call.
//
// Only in fp64 builds on CUDA; everywhere else gemm_ozaki is the vendor GEMM.
// The attention is not a GEMM and is not covered.
#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include "pet/kokkos.hpp"

namespace pet {

enum class GemmMode { Native, Ozaki };

constexpr int kOzakiSliceBits = 7;  // leaves the sign bit of an int8
constexpr int kOzakiMaxSlices = 8;  // ceil(53 / 7): a full fp64 mantissa

// One operand, split: slices[t] holds every entry's t-th digit, scaled by
// `scale` (per row, per column, or one for the whole matrix if `uniform`).
//
// A weight is split uniformly because it is contracted along both axes (by
// linear and linear_bwd), and only a scalar scale is constant along both. It
// costs a bit or two of the largest rows' mantissa. Activations, whose rows
// differ genuinely in magnitude, are split per row.
struct OzakiSplit {
  std::vector<I8View2D> slices;  // [n_slices] of [rows, cols]
  RView1D scale;
  int n_slices = 0;
  int rows = 0, cols = 0;
  bool uniform = false;

  bool valid() const { return n_slices > 0 && !slices.empty(); }
};

// The slice count trades accuracy for speed: 8 is fp64 to its own rounding, 4
// about fp32-and-a-half, 2 a preview. `triangular` keeps only the products with
// t + v < slices (the rest fall below the precision); off, it is a reference.
struct OzakiConfig {
  GemmMode mode = GemmMode::Native;
  int slices = kOzakiMaxSlices;
  bool triangular = true;
};

// Process-wide, from PET_GEMM_MODE=native|ozaki and PET_OZAKI_SLICES=1..8.
OzakiConfig& ozaki_config();

// Whether this build can run the Ozaki path (CUDA, fp64 Net). Asking for it
// elsewhere is not an error; it just does nothing.
bool ozaki_available();
inline bool ozaki_active() { return ozaki_config().mode == GemmMode::Ozaki && ozaki_available(); }

// Split `src` per row, or per column when `by_column` (for an operand that
// enters transposed: the scale must be constant along the contracted index).
// Buffers come from `ws` under `key`.
OzakiSplit ozaki_split(Workspace& ws, const std::string& key, const View2D& src, int n_slices,
                       bool by_column);

// Split a weight once, uniformly, into owned buffers; `transposed` splits src^T.
// Both layouts are kept because int8 cuBLAS takes only operands with the
// contraction index contiguous, and linear and linear_bwd contract opposite
// axes.
OzakiSplit ozaki_split_weight(const View2D& src, int n_slices, bool transposed = false);

// C = alpha op(A) op(B) + beta C (+ bias) through the slice grid, with `bsplit`
// B's split (null: split here). The vendor GEMM when the path is off or the
// shape cannot gain. Scratch is a pool private to ozaki.cpp: one call is in
// flight at a time.
void gemm_ozaki(char transA, char transB, Net alpha, const View2D& A, const View2D& B, Net beta,
                const View2D& C, const OzakiSplit* bsplit, const View1D& bias = {});

// Bytes that pool holds.
std::size_t ozaki_workspace_bytes();

}  // namespace pet
