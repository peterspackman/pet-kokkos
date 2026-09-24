// Kokkos types, the scratch pool every evaluation draws from, the host->device
// upload, and a few device math helpers.
#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pet {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using LR = Kokkos::LayoutRight;

// Precision, chosen at compile time. Net is the network's type (every GEMM and
// activation), Real the geometry's and the accumulated energy/force/virial's.
//   default          Net=float  Real=double   fp32 network, fp64 geometry
//   PET_KOKKOS_FP32  Net=float  Real=float    all single, fastest
//   PET_KOKKOS_FP64  Net=double Real=double   all double, a correctness instrument
//                    (PET_KOKKOS_DOUBLE_NET is a legacy alias)
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

using View1D = Kokkos::View<Net*, LR, MemSpace>;
using View2D = Kokkos::View<Net**, LR, MemSpace>;
using RView1D = Kokkos::View<Real*, LR, MemSpace>;
using RView2D = Kokkos::View<Real**, LR, MemSpace>;
using IView1D = Kokkos::View<int*, LR, MemSpace>;
using IView2D = Kokkos::View<int**, LR, MemSpace>;
// int8 slices for the Ozaki GEMM (ozaki.hpp).
using I8View1D = Kokkos::View<int8_t*, LR, MemSpace>;
using I8View2D = Kokkos::View<int8_t**, LR, MemSpace>;
using U64View1D = Kokkos::View<uint64_t*, LR, MemSpace>;

// The scratch pool. An evaluation needs ~100 buffers, and allocating them per
// call means cudaMalloc/cudaFree, which synchronise the device; the pool keeps
// them between calls, so a warm evaluation allocates nothing.
//
// Two kinds of buffer:
//
//   Named -- n2("key", r, c) and friends. One flat buffer per key, grown when too
//   small and otherwise reused whatever shape it is asked for, so a shrinking
//   batch never reallocates. Two buffers live at the same time need two keys.
//   Saved activations are named, because the backward finds them again (peek2).
//
//   Scoped -- tmp(r, c). Any pooled buffer that no open Scope holds; a Scope
//   returns everything handed out since it opened. Forward and backward
//   temporaries, never live together, share memory this way. The same call
//   sequence gets the same buffers, so this is deterministic.
//
// A handed-out buffer is zeroed while set_zero(true), and otherwise left as it
// was -- or, with PET_WS_POISON=1, filled with NaN, so a read before a write
// shows up in the result instead of passing as a plausible stale number.
class Workspace {
 public:
  View2D n2(const std::string& k, int r, int c) { return cur2_[k] = get<View2D>(k, r, c); }
  View1D n1(const std::string& k, int n) { return get<View1D>(k, n); }
  RView2D r2(const std::string& k, int r, int c) { return get<RView2D>(k, r, c); }
  RView1D r1(const std::string& k, int n) { return get<RView1D>(k, n); }
  IView2D i2(const std::string& k, int r, int c) { return get<IView2D>(k, r, c); }
  IView1D i1(const std::string& k, int n) { return get<IView1D>(k, n); }
  I8View2D i8(const std::string& k, int r, int c) { return get<I8View2D>(k, r, c); }
  U64View1D u64(const std::string& k, int n) { return get<U64View1D>(k, n); }

  // The view the last n2(k, ...) handed out, unfilled: a saved activation.
  View2D peek2(const std::string& k) const { return cur2_.at(k); }

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

  // The smallest free buffer that fits, else the largest free one regrown.
  View2D tmp(int r, int c) {
    const std::size_t need = std::size_t(r) * c;
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
      const bool regrow = tmp_[fit].extent(0) > 0;
      tmp_[fit] = View1D();  // free the old one first
      tmp_[fit] = View1D("ws:tmp", regrow ? need + need / 4 : need);
      ++generation_;
    }
    busy_[fit] = true;
    held_.push_back(fit);
    View2D v(tmp_[fit].data(), r, c);
    fill(v);
    return v;
  }

  void set_zero(bool z) { zero_ = z; }

  // Bumped whenever the pool allocates, i.e. whenever a pointer it handed out
  // may have moved. A recorded CUDA graph keys on it.
  std::size_t generation() const { return generation_; }

  // Every buffer the pool holds, largest first, as (label, bytes).
  std::vector<std::pair<std::string, std::size_t>> capacity_breakdown() const {
    std::vector<std::pair<std::string, std::size_t>> v;
    for (const auto& [k, b] : named_) v.emplace_back(k, b.span());
    for (const auto& t : tmp_) v.emplace_back("ws:tmp", t.span() * sizeof(Net));
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return v;
  }
  std::size_t capacity_bytes() const {
    std::size_t n = 0;
    for (const auto& [k, b] : capacity_breakdown()) n += b;
    return n;
  }

 private:
  static bool poison() {
    static const bool p = [] { const char* e = std::getenv("PET_WS_POISON"); return e && e[0] == '1'; }();
    return p;
  }
  template <class V, class... Ext>
  V get(const std::string& k, Ext... n) {
    using T = typename V::non_const_value_type;
    const std::size_t bytes = (std::size_t(n) * ... * sizeof(T));
    // Exact the first time; a quarter over when it has to grow again, so a
    // stepping loop whose sizes wander (MD) stops reallocating.
    auto& buf = named_[k];
    if (buf.extent(0) < bytes) buf = Bytes(k, buf.extent(0) ? bytes + bytes / 4 : bytes), ++generation_;
    V v(reinterpret_cast<T*>(buf.data()), n...);
    fill(v);
    return v;
  }
  template <class V>
  void fill(const V& v) {
    using T = typename V::non_const_value_type;
    if (v.size() == 0) return;  // deep_copy fences on an empty view, which a graph capture cannot hold
    if (zero_) Kokkos::deep_copy(ExecSpace(), v, T(0));
    else if constexpr (std::is_floating_point_v<T>)
      if (poison()) Kokkos::deep_copy(ExecSpace(), v, std::numeric_limits<T>::quiet_NaN());
  }

  using Bytes = Kokkos::View<char*, MemSpace>;
  bool zero_ = true;
  std::size_t generation_ = 0;
  std::unordered_map<std::string, Bytes> named_;
  std::unordered_map<std::string, View2D> cur2_;  // for peek2
  std::vector<View1D> tmp_;                       // the tmp() pool
  std::vector<bool> busy_;                        // held by an open Scope
  std::vector<std::size_t> held_;                 // tmp_ indices, in hand-out order
};

// Many small host arrays to the device as one asynchronous copy. add() converts
// an array to its device type into a staging area; send() copies it all through
// pinned memory into a device buffer kept between calls; get() views an array
// where it landed. Views are valid until the next send().
class Upload {
 public:
  void clear() { host_.clear(); }
  template <class T, class U>
  std::size_t add(const std::vector<U>& v) {
    const std::size_t off = (host_.size() + 15) & ~std::size_t(15);
    host_.resize(off + v.size() * sizeof(T));
    T* dst = reinterpret_cast<T*>(host_.data() + off);
    for (std::size_t i = 0; i < v.size(); ++i) dst[i] = static_cast<T>(v[i]);
    return off;
  }
  void send() {
    const std::size_t n = host_.size();
    const auto all = std::make_pair(std::size_t(0), n);
    if (pinned_.extent(0) < n) pinned_ = Pinned(Kokkos::view_alloc(Kokkos::WithoutInitializing, "upload:host"), n);
    if (dev_.extent(0) < n) dev_ = Device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "upload"), n);
    ExecSpace().fence();  // the previous send may still be reading pinned_
    std::copy(host_.begin(), host_.end(), pinned_.data());
    Kokkos::deep_copy(ExecSpace(), Kokkos::subview(dev_, all), Kokkos::subview(pinned_, all));
  }
  template <class V, class... Ext>
  V get(std::size_t off, Ext... n) const {
    return V(reinterpret_cast<typename V::pointer_type>(dev_.data() + off), n...);
  }

 private:
  using Pinned = Kokkos::View<char*, Kokkos::SharedHostPinnedSpace>;
  using Device = Kokkos::View<char*, MemSpace>;
  std::vector<char> host_;
  Pinned pinned_;
  Device dev_;
};

template <class T>
KOKKOS_INLINE_FUNCTION T sigmoid(T x) {
  return T(1) / (T(1) + Kokkos::exp(-x));
}
template <class T>
KOKKOS_INLINE_FUNCTION T silu(T x) {
  return x * sigmoid(x);
}

// The hardware exp/log for the attention softmax (~2^-22 relative error, which
// softmax weights do not notice). In fp64 builds, the accurate ones.
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
KOKKOS_INLINE_FUNCTION double fast_exp(double x) { return Kokkos::exp(x); }
KOKKOS_INLINE_FUNCTION double fast_log(double x) { return Kokkos::log(x); }

}  // namespace pet
