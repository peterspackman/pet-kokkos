// A metatensor array over a Kokkos View of doubles (values) or int32 (labels),
// in device memory or on the host, so the plugin's outputs can stay where they
// were computed and reach the engine through DLPack without a copy.
//
// The data is contiguous, row-major. DLPack exports share it (the export keeps
// the View alive); every other operation makes a new array in the same place.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <metatensor.hpp>

#include "pet/kokkos.hpp"

namespace pet_metatomic {

// Where this build's device memory is, as DLPack calls it.
inline DLDevice kokkos_device() {
#if defined(KOKKOS_ENABLE_CUDA)
  return {kDLCUDA, Kokkos::device_id()};
#elif defined(KOKKOS_ENABLE_HIP)
  return {kDLROCM, Kokkos::device_id()};
#else
  return {kDLCPU, 0};
#endif
}

inline bool same_device(DLDevice a, DLDevice b) { return a.device_type == b.device_type && a.device_id == b.device_id; }

template <typename T>
class KokkosArray final : public metatensor::DataArrayBase {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, int32_t>, "float64 or int32");

 public:
  using DeviceView = Kokkos::View<T*, pet::MemSpace>;
  using HostView = Kokkos::View<T*, Kokkos::HostSpace>;

  // Zero-filled, on the device (or on the host).
  explicit KokkosArray(std::vector<uintptr_t> shape, bool on_host = false) : shape_(std::move(shape)), on_host_(on_host) {
    if (on_host_)
      host_ = HostView("mta:host", count(shape_));
    else
      device_ = DeviceView("mta:device", count(shape_));
  }

  // The device data, to write into; empty for an array on the host.
  DeviceView device_view() const { return device_; }

  mts_data_origin_t origin() const override {
    static const mts_data_origin_t origin = [] {
      mts_data_origin_t o = 0;
      metatensor::details::check_status(mts_register_data_origin("pet_kokkos::KokkosArray", &o));
      return o;
    }();
    return origin;
  }

  DLDevice device() const override { return on_host_ ? DLDevice{kDLCPU, 0} : kokkos_device(); }

  DLDataType dtype() const override {
    if constexpr (std::is_same_v<T, double>) return {kDLFloat, 64, 1};
    else return {kDLInt, 32, 1};
  }

  DLManagedTensorVersioned* as_dlpack(DLDevice to, const int64_t* stream, DLPackVersion max_version) override {
    if (max_version.major < 1) throw metatensor::Error("pet_kokkos::KokkosArray exports DLPack 1.x only");
    // The writes that made this data are on Kokkos's stream: done before anyone
    // reads it, unless the caller asks for no synchronization.
    if (!on_host_ && !(stream && *stream == -1)) Kokkos::fence("KokkosArray::as_dlpack");
    if (same_device(to, device()))
      return on_host_ ? export_view(host_.data(), true, host_, DeviceView())
                      : export_view(device_.data(), false, HostView(), device_);
    if (to.device_type == kDLCPU) {  // a host copy, owned by the export
      HostView copy("mta:host", device_.extent(0));
      Kokkos::deep_copy(copy, device_);
      return export_view(copy.data(), true, copy, DeviceView());
    }
    throw metatensor::Error("pet_kokkos::KokkosArray: can not export to device type " +
                            std::to_string(to.device_type) + " id " + std::to_string(to.device_id));
  }

  std::unique_ptr<DataArrayBase> from_dlpack(DLManagedTensorVersioned* tensor) const override {
    struct Guard {
      DLManagedTensorVersioned* t;
      ~Guard() {
        if (t && t->deleter) t->deleter(t);
      }
    } guard{tensor};
    if (!tensor) throw metatensor::Error("pet_kokkos::KokkosArray::from_dlpack: null tensor");
    const DLTensor& t = tensor->dl_tensor;
    const DLDataType want = dtype();
    if (t.dtype.code != want.code || t.dtype.bits != want.bits || t.dtype.lanes != 1)
      throw metatensor::Error("pet_kokkos::KokkosArray::from_dlpack: unexpected dtype");
    std::vector<uintptr_t> shape(t.shape, t.shape + t.ndim);
    for (int k = 0; t.strides && k < t.ndim; ++k)
      if (shape[k] > 1 && t.strides[k] != row_major_stride(shape, k))
        throw metatensor::Error("pet_kokkos::KokkosArray::from_dlpack: the tensor must be contiguous");
    const auto* data = reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
    const bool host = t.device.device_type == kDLCPU;
    if (!host && !same_device(t.device, kokkos_device()))
      throw metatensor::Error("pet_kokkos::KokkosArray::from_dlpack: data on a device this build does not use");
    auto out = std::make_unique<KokkosArray>(shape, host);
    const std::size_t n = count(shape);
    if (host)
      Kokkos::deep_copy(out->host_, Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(data, n));
    else
      Kokkos::deep_copy(out->device_, Kokkos::View<const T*, pet::MemSpace, Kokkos::MemoryUnmanaged>(data, n));
    return out;
  }

  std::unique_ptr<DataArrayBase> copy(DLDevice device) const override {
    const bool host = device.device_type == kDLCPU;
    if (!host && !same_device(device, kokkos_device()))
      throw metatensor::Error("pet_kokkos::KokkosArray: can not copy to device type " +
                              std::to_string(device.device_type));
    auto out = std::make_unique<KokkosArray>(shape_, host);
    if (host && on_host_) Kokkos::deep_copy(out->host_, host_);
    if (host && !on_host_) Kokkos::deep_copy(out->host_, device_);
    if (!host && on_host_) Kokkos::deep_copy(out->device_, host_);
    if (!host && !on_host_) Kokkos::deep_copy(out->device_, device_);
    return out;
  }

  std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape, metatensor::MtsArray fill_value) const override {
    auto fill = fill_value.as_dlpack_array<T>({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
    const T x = fill.data()[0];
    auto out = std::make_unique<KokkosArray>(std::move(shape), on_host_);
    if (x != T(0)) {
      if (on_host_)
        Kokkos::deep_copy(out->host_, x);
      else
        Kokkos::deep_copy(out->device_, x);
    }
    return out;
  }

  const std::vector<uintptr_t>& shape() const& override { return shape_; }

  void reshape(const std::vector<uintptr_t>& shape) override {
    if (count(shape) != count(shape_)) throw metatensor::Error("pet_kokkos::KokkosArray::reshape: wrong number of elements");
    shape_ = shape;
  }

  void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
    const int nd = int(shape_.size());
    if (int(axis_1) >= nd || int(axis_2) >= nd) throw metatensor::Error("pet_kokkos::KokkosArray::swap_axes: no such axis");
    if (nd > kMaxDims) throw metatensor::Error("pet_kokkos::KokkosArray: too many dimensions");
    std::vector<uintptr_t> shape = shape_;
    std::swap(shape[axis_1], shape[axis_2]);
    // out[index] = in[index with the two axes swapped]
    Dims d{};
    d.n = nd;
    for (int k = 0; k < nd; ++k) d.out_shape[k] = int64_t(shape[k]), d.in_stride[k] = row_major_stride(shape_, k);
    std::swap(d.in_stride[axis_1], d.in_stride[axis_2]);
    if (on_host_) {
      HostView out("mta:host", host_.extent(0));
      transpose<Kokkos::DefaultHostExecutionSpace>(d, host_, out);
      host_ = out;
    } else {
      DeviceView out("mta:device", device_.extent(0));
      transpose<pet::ExecSpace>(d, device_, out);
      device_ = out;
    }
    shape_ = shape;
  }

  void move_data(const DataArrayBase& input, std::vector<mts_data_movement_t> moves) override {
    const auto* in = dynamic_cast<const KokkosArray*>(&input);
    if (!in || in->on_host_ != on_host_)
      throw metatensor::Error("pet_kokkos::KokkosArray::move_data: the input must be a KokkosArray in the same place");
    if (shape_.size() < 2 || in->shape_.size() != shape_.size())
      throw metatensor::Error("pet_kokkos::KokkosArray::move_data: mismatched dimensions");
    // [samples, components..., properties]
    int64_t C = 1;
    for (std::size_t k = 1; k + 1 < shape_.size(); ++k) C *= int64_t(shape_[k]);
    const int64_t P_out = int64_t(shape_.back()), P_in = int64_t(in->shape_.back());
    const int M = int(moves.size());
    Kokkos::View<const mts_data_movement_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> host_moves(moves.data(), M);
    if (on_host_) {
      move<Kokkos::DefaultHostExecutionSpace>(host_moves, C, P_in, P_out, in->host_, host_);
    } else {
      Kokkos::View<mts_data_movement_t*, pet::MemSpace> device_moves("mta:moves", M);
      Kokkos::deep_copy(device_moves, host_moves);
      move<pet::ExecSpace>(device_moves, C, P_in, P_out, in->device_, device_);
    }
  }

 // Public only because nvcc wants a device lambda's enclosing function to be.
  static constexpr int kMaxDims = 8;
  struct Dims {
    int n;
    int64_t out_shape[kMaxDims], in_stride[kMaxDims];
  };

  static std::size_t count(const std::vector<uintptr_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<std::size_t>());
  }
  static int64_t row_major_stride(const std::vector<uintptr_t>& shape, int axis) {
    int64_t s = 1;
    for (std::size_t k = axis + 1; k < shape.size(); ++k) s *= int64_t(shape[k]);
    return s;
  }

  template <class Exec, class View>
  static void transpose(const Dims d, View in, View out) {
    Kokkos::parallel_for(
        "mta_swap_axes", Kokkos::RangePolicy<Exec>(0, out.extent(0)), KOKKOS_LAMBDA(int64_t flat) {
          int64_t rest = flat, src = 0;
          for (int k = d.n - 1; k >= 0; --k) {
            src += (rest % d.out_shape[k]) * d.in_stride[k];
            rest /= d.out_shape[k];
          }
          out(flat) = in(src);
        });
  }

  template <class Exec, class Moves, class In, class Out>
  static void move(Moves moves, int64_t C, int64_t P_in, int64_t P_out, In in, Out out) {
    Kokkos::parallel_for(
        "mta_move_data", Kokkos::RangePolicy<Exec>(0, moves.extent(0)), KOKKOS_LAMBDA(int m) {
          const auto mv = moves(m);
          for (int64_t c = 0; c < C; ++c)
            for (int64_t x = 0; x < int64_t(mv.properties_length); ++x)
              out((int64_t(mv.sample_out) * C + c) * P_out + int64_t(mv.properties_start_out) + x) =
                  in((int64_t(mv.sample_in) * C + c) * P_in + int64_t(mv.properties_start_in) + x);
        });
    Exec().fence();
  }

 private:
  // A DLPack view of `data`, keeping whichever View owns it alive.
  DLManagedTensorVersioned* export_view(T* data, bool on_host, HostView host, DeviceView device) const {
    struct Context {
      HostView host;
      DeviceView device;
      std::vector<int64_t> shape, strides;
    };
    auto ctx = std::make_unique<Context>();
    ctx->host = std::move(host), ctx->device = std::move(device);
    for (std::size_t k = 0; k < shape_.size(); ++k)
      ctx->shape.push_back(int64_t(shape_[k])), ctx->strides.push_back(row_major_stride(shape_, int(k)));

    auto managed = std::make_unique<DLManagedTensorVersioned>();
    std::memset(managed.get(), 0, sizeof(DLManagedTensorVersioned));
    managed->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed->deleter = [](DLManagedTensorVersioned* self) {
      delete static_cast<Context*>(self->manager_ctx);
      delete self;
    };
    DLTensor& t = managed->dl_tensor;
    t.data = data;
    t.device = on_host ? DLDevice{kDLCPU, 0} : kokkos_device();
    t.ndim = int32_t(shape_.size());
    t.dtype = dtype();
    t.shape = t.ndim ? ctx->shape.data() : nullptr;
    t.strides = t.ndim ? ctx->strides.data() : nullptr;
    t.byte_offset = 0;
    managed->manager_ctx = ctx.release();
    return managed.release();
  }

  std::vector<uintptr_t> shape_;
  bool on_host_ = false;
  HostView host_;
  DeviceView device_;
};

}  // namespace pet_metatomic
