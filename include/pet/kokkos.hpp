// Shared Kokkos View typedefs and small device-inline math helpers.
#pragma once

#include <Kokkos_Core.hpp>

#include <string>
#include <unordered_map>

namespace pet {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using LR = Kokkos::LayoutRight;

// Precision mode (compile-time). `Net` is the network type (embeddings,
// transformer, all GEMMs); `Real` is the geometry / energy / force / virial
// accumulation type. Three modes, selected by macro:
//   (default) MIXED : Net=float,  Real=double  -- fp32 network, fp64 geometry.
//   PET_KOKKOS_FP32 : Net=float,  Real=float   -- all single precision (fastest;
//                     on GeForce especially, fp32 GEMM + fp32 accumulation).
//   PET_KOKKOS_FP64 : Net=double, Real=double  -- all double (most accurate;
//                     fp64 GEMM is slow on consumer GPUs). PET_KOKKOS_DOUBLE_NET
//                     is kept as a legacy alias for this mode.
#if defined(PET_KOKKOS_FP64) || defined(PET_KOKKOS_DOUBLE_NET)
using Net = double;
using Real = double;
#elif defined(PET_KOKKOS_FP32)
using Net = float;
using Real = float;
#else
using Net = float;
using Real = double;
#endif

// Network (Net precision) views. View1D/View2D keep their names so the network
// kernels read naturally.
using View1D = Kokkos::View<Net*, LR, MemSpace>;
using View2D = Kokkos::View<Net**, LR, MemSpace>;
using View3D = Kokkos::View<Net***, LR, MemSpace>;

// Geometry / accumulation (Real precision) views.
using RView1D = Kokkos::View<Real*, LR, MemSpace>;
using RView2D = Kokkos::View<Real**, LR, MemSpace>;

using IView1D = Kokkos::View<int*, LR, MemSpace>;
using IView2D = Kokkos::View<int**, LR, MemSpace>;

// Persistent scratch-buffer pool reused across compute() calls. Each compute()
// otherwise allocates ~70 Kokkos Views (cudaMalloc/cudaFree, which synchronize
// the device) -- the dominant per-step cost on GPU for small/medium systems.
//
// A buffer is keyed by name (callers use distinct names for buffers that must be
// simultaneously live). The cached View is a *capacity* buffer: it is reused (no
// reallocation) whenever it has at least the requested rows AND exactly the
// requested columns, returning a correctly-sized [r,c] view over the first r rows
// (a contiguous, LayoutRight-equivalent prefix, since columns match). It only
// reallocates when the row capacity is too small or the column count changes.
//
// This matters for the batched relaxer/CSP path: as structures converge and the
// active set shrinks, the row count (N, N*M, N*(M+1)) drops every round while the
// column count is always a fixed model dim (d_pet, 2*d_pet, d_node, d_feedforward,
// the constant probe grid, ...). Exact-shape matching would therefore reallocate
// every one of the ~70 buffers on every shrink -- device-synchronizing cudaMalloc/
// cudaFree that dominated the step (measured: per-step time roughly doubled as the
// set shrank). Capacity reuse sizes the pool to the first (full) round and never
// reallocates again; the returned [r,c] prefix is byte-for-byte a fresh zero-
// initialized View (only the used r*c elements are zeroed), so callers are
// unaffected. The only caller obligation is the usual one: do not hand the same
// key to two buffers that are live at the same time (e.g. values carried across a
// layer loop need per-iteration keys).
class Workspace {
 public:
  View2D n2(const std::string& k, int r, int c) {
    View2D v = get2<View2D, Net>(n2_, k, r, c);
    cur2_[k] = v;  // remember the current logical view so peek2 returns the same prefix
    return v;
  }
  View1D n1(const std::string& k, int n) { return get1<View1D, Net>(n1_, k, n); }
  RView2D r2(const std::string& k, int r, int c) { return get2<RView2D, double>(r2_, k, r, c); }
  RView1D r1(const std::string& k, int n) { return get1<RView1D, double>(r1_, k, n); }
  IView2D i2(const std::string& k, int r, int c) { return get2<IView2D, int>(i2_, k, r, c); }
  IView1D i1(const std::string& k, int n) { return get1<IView1D, int>(i1_, k, n); }
  // Return an existing buffer without zeroing it (for reading data written by an
  // earlier kernel this step, e.g. saved forward activations). Returns the same
  // [r,c] prefix the matching n2() handed out this step, NOT the capacity buffer.
  View2D peek2(const std::string& k) { return cur2_.at(k); }

  // Zeroing policy for reused buffers. Reused buffers are zeroed by default so a
  // caller can treat them as fresh zero-init Views. But a buffer that is fully
  // overwritten before any read (every forward activation: a gemm beta=0 output, a
  // gather, a norm, or a full-coverage elementwise write) does not need it -- the
  // ~70 per-step zero memsets were measured at ~22% of a relax step. The forward
  // pass has no from-zero accumulators, so it runs with zeroing off; the backward
  // pass (linear_bwd beta=1 / norm_bwd += / atomic accumulators) keeps it on.
  // Freshly allocated buffers are always zero-initialized by Kokkos regardless.
  void set_zero(bool z) { zero_ = z; }

  // Total device memory held by the pool. This is what a PET evaluation actually
  // costs, and it is the only honest basis for choosing a batch width: the
  // dominant buffers scale with the NEIGHBOUR COUNT, which depends on the cutoff,
  // the density and the periodic images, and is not predictable from the model
  // hypers. Estimating it from the cutoff and a typical crystal density gave ~38
  // neighbours where the real list had ~119, i.e. a budget 3x too generous.
  std::size_t capacity_bytes() const {
    std::size_t b = 0;
    for (const auto& kv : n2_) b += kv.second.span() * sizeof(Net);
    for (const auto& kv : n1_) b += kv.second.span() * sizeof(Net);
    for (const auto& kv : r2_) b += kv.second.span() * sizeof(double);
    for (const auto& kv : r1_) b += kv.second.span() * sizeof(double);
    for (const auto& kv : i2_) b += kv.second.span() * sizeof(int);
    for (const auto& kv : i1_) b += kv.second.span() * sizeof(int);
    return b;
  }

 private:
  // Reuse the capacity buffer when it has enough rows and matching columns; hand
  // back an (unmanaged) [r,c] view over its first r rows. The capacity buffer stays
  // alive in the map, so the prefix view is valid for the step's lifetime.
  //
  // The zero fill passes ExecSpace() explicitly. The deep_copy overload WITHOUT an
  // execution space is specified to fence the whole device both before and after
  // the fill; the backward pass pulls ~200 buffers per call, so that was ~400 full
  // device syncs per optimizer round (measured: 18430 scalar deep_copies => 36860
  // fences over a 76-round relax, and nsys counted roughly one cudaDeviceSynchronize
  // per kernel launch). The CPU could never run ahead, so every kernel's submission
  // latency was exposed -- about 22 us per zeroed buffer, ~20% of a round. The
  // exec-space overload skips the fences when the memory is accessible from that
  // space, and everything here runs on the default instance, so ordering is
  // unchanged.
  template <class V, class T, class Map>
  V get2(Map& m, const std::string& k, int r, int c) {
    V& slot = m[k];
    if (slot.extent(0) >= (size_t) r && slot.extent(1) == (size_t) c) {
      V view(slot.data(), r, c);  // contiguous prefix (columns match -> LayoutRight)
      if (zero_) Kokkos::deep_copy(ExecSpace(), view, T(0));
      return view;
    }
    slot = V(k, r, c);  // (re)allocate capacity; fresh allocation is zero-initialized
    return slot;
  }
  template <class V, class T, class Map>
  V get1(Map& m, const std::string& k, int n) {
    V& slot = m[k];
    if (slot.extent(0) >= (size_t) n) {
      V view(slot.data(), n);
      if (zero_) Kokkos::deep_copy(ExecSpace(), view, T(0));
      return view;
    }
    slot = V(k, n);
    return slot;
  }
  bool zero_ = true;  // zeroing policy for reused buffers (see set_zero)
  std::unordered_map<std::string, View2D> n2_;
  std::unordered_map<std::string, View1D> n1_;
  std::unordered_map<std::string, RView2D> r2_;
  std::unordered_map<std::string, RView1D> r1_;
  std::unordered_map<std::string, IView2D> i2_;
  std::unordered_map<std::string, IView1D> i1_;
  std::unordered_map<std::string, View2D> cur2_;  // current logical n2 view per key (for peek2)
};

template <class T>
KOKKOS_INLINE_FUNCTION T sigmoidd(T x) {
  return T(1) / (T(1) + Kokkos::exp(-x));
}
template <class T>
KOKKOS_INLINE_FUNCTION T silud(T x) {
  return x * sigmoidd(x);
}

// Fast single-precision exp/log for the attention softmax. Maps to the native
// hardware transcendental (NVIDIA SFU / AMD v_exp_f32,v_log_f32) via __expf/__logf
// in device code, vs the accurate libm/ocml polynomial that Kokkos::exp(float)
// uses. On CDNA2 the accurate ocml expf/logf is many instructions and makes the
// softmax the dominant cost; the ~2^-22 relative error of the intrinsics is
// negligible for softmax weights. Host builds fall back to expf/logf. Used only
// in the attention math -- the fp64 geometry/force accumulation is untouched.
KOKKOS_INLINE_FUNCTION float fast_exp(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __expf(x);
#else
  return expf(x);
#endif
}
KOKKOS_INLINE_FUNCTION float fast_log(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __logf(x);
#else
  return logf(x);
#endif
}
// fp64-mode passthrough (PET_KOKKOS_FP64): keep full double accuracy -- on the HPC
// GPUs where fp64 mode makes sense, fp64 transcendentals are not the weak path.
KOKKOS_INLINE_FUNCTION double fast_exp(double x) { return Kokkos::exp(x); }
KOKKOS_INLINE_FUNCTION double fast_log(double x) { return Kokkos::log(x); }

}  // namespace pet
