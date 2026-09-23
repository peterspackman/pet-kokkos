// Shared Kokkos View typedefs and small device-inline math helpers.
#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Narrow integer views, for the Ozaki slice decomposition (ozaki.hpp). int8 is
// the widest type whose pairwise products accumulate exactly in int32 over the
// inner dimensions this network uses.
using I8View2D = Kokkos::View<int8_t**, LR, MemSpace>;
using I8View1D = Kokkos::View<int8_t*, LR, MemSpace>;

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
  I8View2D i8(const std::string& k, int r, int c) { return get2<I8View2D, int8_t>(i8_, k, r, c); }
  IView1D i1(const std::string& k, int n) { return get1<IView1D, int>(i1_, k, n); }

  // Temporaries whose lifetime is a C++ scope. tmp() hands out a pooled buffer
  // that no open Scope holds -- the smallest that fits, else the largest free one
  // regrown -- and a Scope gives back everything handed out since it opened.
  //
  // Keying buffers by name holds each name's buffer for the whole evaluation, so
  // forward temporaries and backward temporaries -- never live at the same time
  // -- each cost their own memory, as did every layer's copy of a per-layer
  // label. With lexical lifetimes they share: on pet-omat-l the named working
  // set was ~50 [edges x d_pet] buffers. The same call sequence gets the same
  // buffers, so this is deterministic; steady state allocates nothing.
  //
  // Use it only for values that die with the enclosing Scope. A saved activation,
  // or anything peek2() retrieves, keeps a name.
  class Scope {
   public:
    explicit Scope(Workspace& w) : ws_(w), mark_(w.held_.size()) {}
    ~Scope() {
      for (; ws_.held_.size() > mark_; ws_.held_.pop_back()) ws_.busy_[ws_.held_.back()] = false;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    Workspace& ws_;
    std::size_t mark_;
  };
  View2D tmp(int r, int c) {
    const std::size_t need = std::size_t(r) * std::size_t(c);
    int fit = -1, grow = -1;
    for (int i = 0; i < (int) tmp_.size(); ++i) {
      if (busy_[i]) continue;
      const std::size_t cap = tmp_[i].extent(0);
      if (cap >= need && (fit < 0 || cap < tmp_[fit].extent(0))) fit = i;
      if (cap < need && (grow < 0 || cap > tmp_[grow].extent(0))) grow = i;
    }
    if (fit < 0) {
      fit = grow >= 0 ? grow : (int) tmp_.size();
      if (grow < 0) tmp_.emplace_back(), busy_.push_back(false);
      tmp_[fit] = View1D();  // free before growing, so the old and new never coexist
      tmp_[fit] = View1D("ws:tmp", need);
    }
    busy_[fit] = true;
    held_.push_back(fit);
    View2D v(tmp_[fit].data(), r, c);
    if (zero_) Kokkos::deep_copy(ExecSpace(), v, Net(0));
    return v;
  }
  // Capacity-keyed 2-D scratch: reuse whenever the buffer holds enough ELEMENTS,
  // whatever shape they were last used in. i2()/i8() above reuse only when the
  // column count matches exactly, because they hand back a row prefix and a row
  // prefix of a LayoutRight buffer is only contiguous when the columns agree.
  // That rule is right for the model's activations, whose width is a fixed model
  // dimension, but wrong for the Ozaki slice grid: its buffers are [m,k] and
  // [m,n] with k and n taking half a dozen values across one evaluation, so a
  // fixed key reallocated on almost every call -- ~590 cudaMalloc/cudaFree per
  // evaluation, each cudaFree synchronising the device, which is precisely the
  // churn this pool exists to prevent.
  //
  // Reshaping is safe here because the view is unmanaged over a contiguous
  // buffer: any [r,c] with r*c <= capacity is a valid LayoutRight view of it.
  IView2D i2_any(const std::string& k, int r, int c) {
    return get2_any<IView2D, IView1D, int>(i2cap_, k, r, c);
  }
  I8View2D i8_any(const std::string& k, int r, int c) {
    return get2_any<I8View2D, I8View1D, int8_t>(i8cap_, k, r, c);
  }
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
    for (const auto& kv : i8_) b += kv.second.span() * sizeof(int8_t);
    for (const auto& v : tmp_) b += v.span() * sizeof(Net);
    for (const auto& kv : i2cap_) b += kv.second.span() * sizeof(int);
    for (const auto& kv : i8cap_) b += kv.second.span() * sizeof(int8_t);
    return b;
  }

  // Every buffer the pool holds, largest first, as (label, bytes). The pool is
  // grow-only and keyed by label, so this is the honest answer to "what is using
  // the memory" -- an evaluation's footprint is the sum of its labels, and a
  // label that appears once per layer is paid for once per layer.
  std::vector<std::pair<std::string, std::size_t>> capacity_breakdown() const {
    std::vector<std::pair<std::string, std::size_t>> v;
    auto add = [&v](const auto& m, std::size_t esz) {
      for (const auto& kv : m) v.emplace_back(kv.first, kv.second.span() * esz);
    };
    add(n2_, sizeof(Net));
    add(n1_, sizeof(Net));
    add(r2_, sizeof(double));
    add(r1_, sizeof(double));
    add(i2_, sizeof(int));
    add(i1_, sizeof(int));
    add(i8_, sizeof(int8_t));
    for (const auto& t : tmp_) v.emplace_back("ws:tmp", t.span() * sizeof(Net));
    add(i2cap_, sizeof(int));
    add(i8cap_, sizeof(int8_t));
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return v;
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
  // Capacity pool backing i2_any/i8_any: a flat buffer per key, viewed as
  // whatever 2-D shape the caller asked for.
  template <class V, class Base, class T, class Map>
  V get2_any(Map& m, const std::string& k, int r, int c) {
    Base& slot = m[k];
    const std::size_t need = (std::size_t) r * (std::size_t) c;
    if (slot.extent(0) < need) slot = Base(k, need);  // fresh allocations zero-init
    V view(slot.data(), r, c);
    if (zero_) Kokkos::deep_copy(ExecSpace(), view, T(0));
    return view;
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
  std::unordered_map<std::string, I8View2D> i8_;
  std::vector<View1D> tmp_;          // tmp() pool
  std::vector<bool> busy_;           // held by an open Scope
  std::vector<std::size_t> held_;    // tmp_ indices, in hand-out order
  std::unordered_map<std::string, IView1D> i2cap_;
  std::unordered_map<std::string, I8View1D> i8cap_;
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
