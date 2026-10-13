// pet-kokkos as a metatomic plugin: any engine that loads metatomic plugins
// (mta_load_plugin) gets PET's energy, forces and virial from pet::Calculator.
//
// The engine hands over each system with a full pair list at the model cutoff,
// and the engine's pairs are the edges (the adaptive cutoff prunes them as
// usual). The selected atoms are moved to the front of a system and become its
// owned atoms: the energy is theirs, the gradients are with respect to every
// atom.
//
// A system in this build's device memory is read there (Calculator::
// compute_pairs) and its outputs stay there, in KokkosArrays; a system on the
// host goes through compute_edges, with host outputs. Either way the evaluation
// runs on whatever device this build's Kokkos has.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <metatomic.hpp>

#include "kokkos_array.hpp"
#include "pet/calculator.hpp"
#include "pet/gemm.hpp"

namespace {

// --- Kokkos ------------------------------------------------------------------
//
// The plugin carries its own Kokkos (linked in, symbols hidden), so it brings it
// up with the first model -- one GPU per node-local MPI rank, unless the model's
// options name one -- and takes it down at exit. Kokkos cannot start again once
// finalized, so it is not finalized when the last model goes; and a model the
// host never unloaded holds device memory Kokkos must not outlive, so then it is
// left up.

std::mutex kokkos_mutex;
int live_models = 0;

void kokkos_acquire(int device_id) {
  std::lock_guard<std::mutex> lock(kokkos_mutex);
  if (!Kokkos::is_initialized()) {
    if (Kokkos::is_finalized()) throw metatomic::Error("pet-kokkos: Kokkos was already finalized in this process");
    // Quiet: the host process did not ask for Kokkos, nor for its advice.
    auto settings = Kokkos::InitializationSettings().set_disable_warnings(true);
    if (device_id >= 0)
      settings.set_device_id(device_id);
    else
      settings.set_map_device_id_by("mpi_rank");
    Kokkos::initialize(settings);
    std::atexit([] {
      std::lock_guard<std::mutex> lock(kokkos_mutex);
      if (live_models == 0 && Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
    });
  }
  ++live_models;
}

void kokkos_release() {
  std::lock_guard<std::mutex> lock(kokkos_mutex);
  --live_models;
}

// --- options -----------------------------------------------------------------

struct PetOptions {
  pet::Options calc;
  // The GEMM mode, when given. pet::Options::gemm = Native means "the default"
  // (TF32 on recent NVIDIA GPUs), so an explicit "native" is set directly.
  // Process-global, as in pet-kokkos: the last model loaded sets it.
  std::optional<pet::GemmMode> gemm;
  int device_id = -1;  // -1: by node-local MPI rank
};

double parse_number(const std::string& value, const std::string& name) {
  std::size_t used = 0;
  double x = 0;
  try {
    x = std::stod(value, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != value.size() || !std::isfinite(x))
    throw metatomic::Error("pet-kokkos option '" + name + "' must be a number, got '" + value + "'");
  return x;
}

PetOptions parse_options(const std::map<std::string, std::string>& options) {
  PetOptions o;
  for (const auto& [key, value] : options) {
    if (key == "gemm") {
      o.gemm = pet::parse_gemm_mode(value.c_str());
    } else if (key == "recompute") {
      bool found = false;
      for (auto r : {pet::Recompute::Auto, pet::Recompute::Never, pet::Recompute::Wide, pet::Recompute::Layers})
        if (value == pet::recompute_name(r)) o.calc.recompute = r, found = true;
      if (!found) throw metatomic::Error("pet-kokkos option 'recompute' must be auto, never, wide or layers");
    } else if (key == "memory_budget_gib") {
      o.calc.memory_budget_bytes = std::size_t(parse_number(value, key) * (1ull << 30));
    } else if (key == "device_id") {
      o.device_id = int(parse_number(value, key));
    } else {
      throw metatomic::Error("unknown pet-kokkos option '" + key +
                             "' (known: gemm, recompute, memory_budget_gib, device_id)");
    }
  }
  return o;
}

// --- reading the engine's arrays -----------------------------------------------

// A DLPack tensor on the host, copied out in row-major order whatever its
// strides; the dtype must be T's.
template <typename T>
std::vector<T> host_values(const metatomic::DLPackTensor& tensor, const char* what) {
  const DLTensor& t = tensor->dl_tensor;
  if (t.device.device_type != kDLCPU)
    throw metatomic::Error(std::string("pet-kokkos: the system's ") + what + " must be on the CPU");
  DLDataType want{};
  if constexpr (std::is_same_v<T, double>) want = {kDLFloat, 64, 1};
  else if constexpr (std::is_same_v<T, int32_t>) want = {kDLInt, 32, 1};
  else if constexpr (std::is_same_v<T, bool>) want = {kDLBool, 8, 1};
  if (t.dtype.code != want.code || t.dtype.bits != want.bits || t.dtype.lanes != 1)
    throw metatomic::Error(std::string("pet-kokkos: unexpected dtype for the system's ") + what);

  std::size_t n = 1;
  for (int d = 0; d < t.ndim; ++d) n *= std::size_t(t.shape[d]);
  std::vector<T> out(n);
  const auto* base = reinterpret_cast<const char*>(t.data) + t.byte_offset;
  for (std::size_t flat = 0; flat < n; ++flat) {
    // Row-major index -> element offset through the strides (none: contiguous).
    std::int64_t offset = 0, rest = std::int64_t(flat), stride = 1;
    for (int d = t.ndim - 1; d >= 0; --d) {
      const std::int64_t i = rest % t.shape[d];
      rest /= t.shape[d];
      offset += i * (t.strides ? t.strides[d] : stride);
      stride *= t.shape[d];
    }
    if constexpr (std::is_same_v<T, bool>)
      out[flat] = *reinterpret_cast<const std::uint8_t*>(base + offset) != 0;
    else
      out[flat] = *reinterpret_cast<const T*>(base + offset * std::int64_t(sizeof(T)));
  }
  return out;
}

// A per-system scalar input ("charge", "spin_multiplicity").
double system_scalar(const metatomic::System& system, const std::string& name) {
  auto map = system.custom_data(name);
  auto block = map.block_by_id(0);
  auto values = block.values<double>();
  if (values.shape().empty() || values.shape()[0] != 1)
    throw metatomic::Error("pet-kokkos: input '" + name + "' must hold one value per system");
  return values.data()[0];
}

// --- building outputs --------------------------------------------------------

bool has_names(const metatensor::Labels& labels, const std::vector<std::string>& names) {
  const auto got = labels.names();
  return std::equal(got.begin(), got.end(), names.begin(), names.end(),
                    [](const char* a, const std::string& b) { return b == a; });
}

metatensor::Labels labels(const std::vector<std::string>& names, const std::vector<int32_t>& values) {
  return metatensor::Labels(names, values.data(), values.size() / names.size(), metatensor::assume_unique{});
}

using DeviceInts = pet_metatomic::KokkosArray<int32_t>;

// Labels over int32 values in device memory: metatensor wants a block's labels
// where its values are.
metatensor::Labels device_labels(const std::vector<std::string>& names, std::unique_ptr<DeviceInts> values) {
  return metatensor::Labels(names, metatensor::DataArrayBase::to_mts_array(std::move(values)), metatensor::assume_unique{});
}

// Small labels, uploaded; on the host when `device` is false.
metatensor::Labels labels_on(bool device, const std::vector<std::string>& names, const std::vector<int32_t>& values) {
  if (!device) return labels(names, values);
  auto array = std::make_unique<DeviceInts>(std::vector<uintptr_t>{values.size() / names.size(), names.size()});
  Kokkos::deep_copy(array->device_view(),
                    Kokkos::View<const int32_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(values.data(), values.size()));
  return device_labels(names, std::move(array));
}

std::unique_ptr<metatensor::SimpleDataArray<double>> array(std::vector<uintptr_t> shape, std::vector<double> data) {
  return std::make_unique<metatensor::SimpleDataArray<double>>(std::move(shape), std::move(data));
}

metatensor::TensorMap single_block_map(metatensor::TensorBlock block, bool device) {
  std::vector<metatensor::TensorBlock> blocks;
  blocks.emplace_back(std::move(block));
  return metatensor::TensorMap(labels_on(device, {"_"}, {0}), std::move(blocks));
}

// --- the model ---------------------------------------------------------------

// One system's answer, packed in the engine's atom order: the energy, dE/d(eps)
// (3x3, row-major), the selected atoms' energies, then dE/dr [n_atoms, 3] when
// gradients were asked for. On the host, or on the device for a system that was.
struct SystemResult {
  static constexpr std::size_t kEnergy = 0, kStrain = 1, kPerAtom = 10;
  std::vector<int32_t> selected;  // the atoms whose energy is counted, as selected
  std::size_t n_atoms = 0;
  std::vector<double> host;
  pet_metatomic::KokkosArray<double>::DeviceView device;
  std::size_t gradient() const { return kPerAtom + selected.size(); }
  std::size_t size(bool gradients) const { return gradient() + (gradients ? 3 * n_atoms : 0); }
};

// An output array, on the host or the device, filled from the systems' results.
class Output {
 public:
  Output(bool on_device, std::vector<uintptr_t> shape) : shape_(std::move(shape)) {
    std::size_t n = 1;
    for (auto d : shape_) n *= d;
    if (on_device)
      device_ = std::make_unique<pet_metatomic::KokkosArray<double>>(shape_);
    else
      host_.assign(n, 0.0);
  }
  // n values of r from `from`, to `at`.
  void put(std::size_t at, const SystemResult& r, std::size_t from, std::size_t n) {
    if (n == 0) return;
    if (!device_) {
      std::copy(r.host.begin() + from, r.host.begin() + from + n, host_.begin() + at);
    } else {
      using Range = std::pair<std::size_t, std::size_t>;
      Kokkos::deep_copy(pet::ExecSpace(), Kokkos::subview(device_->device_view(), Range(at, at + n)),
                        Kokkos::subview(r.device, Range(from, from + n)));
    }
  }
  std::unique_ptr<metatensor::DataArrayBase> release() {
    if (device_) return std::move(device_);
    return array(shape_, std::move(host_));
  }

 private:
  std::vector<uintptr_t> shape_;
  std::vector<double> host_;
  std::unique_ptr<pet_metatomic::KokkosArray<double>> device_;
};

// Whether a system's arrays are in this build's device memory (else they must
// be on the host).
bool on_device(const metatomic::System& system) {
  const DLDevice where = system.positions()->dl_tensor.device;
  if (where.device_type == kDLCPU) return false;
  const DLDevice ours = pet_metatomic::kokkos_device();
  if (where.device_type != ours.device_type)
    throw metatomic::Error("pet-kokkos: the system is on a device this build does not run on");
  if (where.device_id != ours.device_id)
    throw metatomic::Error("pet-kokkos: the system is on device " + std::to_string(where.device_id) +
                           ", the model on device " + std::to_string(ours.device_id) + " (see the device_id option)");
  return true;
}

// Whatever the engine wrote to device memory is finished before we read it: its
// work may be on another stream.
void sync_with_engine() {
#if defined(KOKKOS_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(KOKKOS_ENABLE_HIP)
  (void) hipDeviceSynchronize();
#endif
}

// A device DLPack tensor of T as a contiguous device pointer: its own data when
// it is contiguous, else a contiguous copy in `keep`.
template <typename T>
const T* device_contiguous(const metatomic::DLPackTensor& tensor, Kokkos::View<T*, pet::MemSpace>& keep, const char* what) {
  const DLTensor& t = tensor->dl_tensor;
  DLDataType want{};
  if constexpr (std::is_same_v<T, double>) want = {kDLFloat, 64, 1};
  else want = {kDLInt, 32, 1};
  if (t.dtype.code != want.code || t.dtype.bits != want.bits || t.dtype.lanes != 1)
    throw metatomic::Error(std::string("pet-kokkos: unexpected dtype for the system's ") + what);
  const T* data = reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
  int64_t n = 1, rows = t.ndim > 0 ? t.shape[0] : 1, cols = t.ndim > 1 ? t.shape[1] : 1;
  for (int d = 0; d < t.ndim; ++d) n *= t.shape[d];
  const int64_t s0 = t.strides && t.ndim > 0 ? t.strides[0] : cols, s1 = t.strides && t.ndim > 1 ? t.strides[1] : 1;
  if ((rows <= 1 || s0 == cols) && (cols <= 1 || s1 == 1)) return data;
  keep = Kokkos::View<T*, pet::MemSpace>(std::string("mta:") + what, n);
  auto out = keep;
  Kokkos::parallel_for(
      "mta_contiguous", Kokkos::RangePolicy<pet::ExecSpace>(0, n),
      KOKKOS_LAMBDA(int64_t k) { out(k) = data[(k / cols) * s0 + (k % cols) * s1]; });
  return keep.data();
}

// (row, system, atom) for every atom of the systems with rows; on the device,
// written there.
metatensor::Labels gradient_samples(const std::vector<SystemResult>& results, const std::vector<int32_t>& ids,
                                    bool device) {
  const std::vector<std::string> names = {"sample", "system", "atom"};
  if (!device) {
    std::vector<int32_t> samples;
    for (std::size_t r = 0; r < ids.size(); ++r)
      for (std::size_t a = 0; a < results[ids[r]].n_atoms; ++a) samples.insert(samples.end(), {int32_t(r), ids[r], int32_t(a)});
    return labels(names, samples);
  }
  std::size_t total = 0;
  for (int32_t s : ids) total += results[s].n_atoms;
  auto array = std::make_unique<DeviceInts>(std::vector<uintptr_t>{total, 3});
  auto v = array->device_view();
  std::size_t at = 0;
  for (std::size_t r = 0; r < ids.size(); ++r) {
    const int row = int(r), system = ids[r], start = int(at);
    Kokkos::parallel_for(
        "mta_gradient_samples", Kokkos::RangePolicy<pet::ExecSpace>(0, int(results[ids[r]].n_atoms)),
        KOKKOS_LAMBDA(int a) {
          v(3 * (start + a)) = row, v(3 * (start + a) + 1) = system, v(3 * (start + a) + 2) = a;
        });
    at += results[ids[r]].n_atoms;
  }
  return device_labels(names, std::move(array));
}

// (system, atom) for every selected atom, in selection order: on the device,
// written there when the selection is every atom, else uploaded.
metatensor::Labels atom_samples(const std::vector<SystemResult>& results, bool device) {
  const std::vector<std::string> names = {"system", "atom"};
  std::size_t rows = 0;
  bool everything = true;
  for (const auto& r : results) {
    rows += r.selected.size();
    everything = everything && r.selected.size() == r.n_atoms &&
                 std::is_sorted(r.selected.begin(), r.selected.end());
  }
  if (!device || !everything) {
    std::vector<int32_t> samples;
    samples.reserve(2 * rows);
    for (std::size_t s = 0; s < results.size(); ++s)
      for (int32_t a : results[s].selected) samples.insert(samples.end(), {int32_t(s), a});
    return labels_on(device, names, samples);
  }
  auto array = std::make_unique<DeviceInts>(std::vector<uintptr_t>{rows, 2});
  auto v = array->device_view();
  std::size_t at = 0;
  for (std::size_t s = 0; s < results.size(); ++s) {
    const int system = int(s), start = int(at);
    Kokkos::parallel_for(
        "mta_atom_samples", Kokkos::RangePolicy<pet::ExecSpace>(0, int(results[s].n_atoms)),
        KOKKOS_LAMBDA(int a) { v(2 * (start + a)) = system, v(2 * (start + a) + 1) = a; });
    at += results[s].n_atoms;
  }
  return device_labels(names, std::move(array));
}

class PetModel final : public metatomic::BaseModel {
 public:
  PetModel(std::string name, const std::string& json, const std::string& weights, PetOptions options)
      : name_(std::move(name)) {
    kokkos_acquire(options.device_id);
    try {
      if (options.gemm) pet::gemm_mode() = *options.gemm;
      calc_ = std::make_unique<pet::Calculator>(json, weights, options.calc);
    } catch (...) {
      kokkos_release();
      throw;
    }
    pairs_ = metatomic::PairListOptions::builder()
                 .cutoff(calc_->cutoff())
                 .full_list(true)
                 .strict(false)
                 .add_requestor("pet-kokkos")
                 .build();
  }

  ~PetModel() override {
    calc_.reset();  // its device memory, before Kokkos may go
    kokkos_release();
  }

  PetModel(const PetModel&) = delete;
  PetModel& operator=(const PetModel&) = delete;

  metatomic::ModelCapabilities capabilities() const override {
    const std::string energy_unit = calc_->energy_unit();
    auto energy = metatomic::Quantity::builder()
                      .name("energy")
                      .unit(energy_unit)
                      .sample_kind(metatomic::SampleKind::System)
                      .add_gradient(metatomic::Gradients::Positions)
                      .add_gradient(metatomic::Gradients::Strain)
                      .build();
    auto atomic_energy =
        metatomic::Quantity::builder().name("energy").unit(energy_unit).sample_kind(metatomic::SampleKind::Atom).build();
    std::vector<int32_t> types(calc_->atomic_types().begin(), calc_->atomic_types().end());
    // On a CUDA build, systems in device memory are read there and the outputs
    // left there. metatomic cannot yet execute a model on ROCm data (it checks
    // the inputs with kernels it has only for CUDA and Metal), so a HIP build
    // takes host systems.
    using Device = metatomic::ModelCapabilities::Device;
#if defined(KOKKOS_ENABLE_CUDA)
    const std::vector<Device> devices = {Device::CUDA, Device::CPU};
#else
    const std::vector<Device> devices = {Device::CPU};
#endif
    // Everything an atom's energy depends on: one cutoff per message-passing
    // layer, and one more for the adaptive cutoff (ghost_cutoff).
    return metatomic::ModelCapabilities::builder()
        .atomic_types(types)
        .interaction_range(calc_->ghost_cutoff())
        .length_unit(calc_->length_unit())
        .supported_devices(devices)
        .dtype(metatomic::ModelCapabilities::DType::Float64)
        .add_output(energy)
        .add_output(atomic_energy)
        .build();
  }

  metatomic::ModelMetadata metadata() const override {
    return metatomic::ModelMetadata::builder()
        .name(name_)
        .description("PET (Point Edge Transformer) interatomic potential, evaluated by pet-kokkos")
        .add_reference("architecture", "https://arxiv.org/abs/2305.19302")
        .add_reference("implementation", "https://github.com/peterspackman/pet-kokkos")
        .build();
  }

  std::vector<metatomic::PairListOptions> requested_pair_lists() const override { return {pairs_}; }

  std::vector<metatomic::Quantity> requested_inputs() const override {
    if (!calc_->hypers().system_conditioning) return {};
    return {
        metatomic::Quantity::builder().name("charge").unit("e").sample_kind(metatomic::SampleKind::System).build(),
        metatomic::Quantity::builder().name("spin_multiplicity").unit("").sample_kind(metatomic::SampleKind::System).build(),
    };
  }

  std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System>& systems,
                                                   std::optional<metatensor::Labels> selected_atoms,
                                                   const std::vector<metatomic::Quantity>& requested_outputs) override {
    bool positions = false, strain = false;
    for (const auto& output : requested_outputs) {
      if (output.name() != "energy") throw metatomic::Error("pet-kokkos: unsupported output '" + output.name() + "'");
      const auto kind = output.sample_kind();
      if (kind != metatomic::SampleKind::System && kind != metatomic::SampleKind::Atom)
        throw metatomic::Error("pet-kokkos: energy must have system or atom samples");
      for (auto g : output.gradients()) {
        if (kind != metatomic::SampleKind::System)
          throw metatomic::Error("pet-kokkos: gradients are only available for the per-system energy");
        (g == metatomic::Gradients::Positions ? positions : strain) = true;
      }
    }
    const bool gradients = positions || strain;

    // Each system's selected atoms, in the order the selection lists them. The
    // selection is metadata, read on the host.
    std::vector<std::vector<int32_t>> selection(systems.size());
    if (selected_atoms) {
      if (!has_names(*selected_atoms, {"system", "atom"}))
        throw metatomic::Error("pet-kokkos: selected_atoms must have the dimensions 'system' and 'atom'");
      auto values = selected_atoms->values();
      for (std::size_t r = 0; r < selected_atoms->count(); ++r) {
        const int32_t s = values(r, 0), a = values(r, 1);
        if (s < 0 || std::size_t(s) >= systems.size() || a < 0 || std::size_t(a) >= systems[s].size())
          throw metatomic::Error("pet-kokkos: selected_atoms refers to an atom that does not exist");
        selection[s].push_back(a);
      }
    } else {
      for (std::size_t s = 0; s < systems.size(); ++s) {
        selection[s].resize(systems[s].size());
        for (std::size_t a = 0; a < systems[s].size(); ++a) selection[s][a] = int32_t(a);
      }
    }

    // metatomic hands over the systems on one device.
    const bool device = !systems.empty() && on_device(systems[0]);
    if (device) sync_with_engine();
    std::vector<SystemResult> results;
    results.reserve(systems.size());
    for (std::size_t s = 0; s < systems.size(); ++s) {
      if (on_device(systems[s]) != device) throw metatomic::Error("pet-kokkos: the systems are on different devices");
      results.push_back(device ? evaluate_device(systems[s], std::move(selection[s]), gradients)
                               : evaluate_host(systems[s], std::move(selection[s]), gradients));
    }

    std::vector<metatensor::TensorMap> outputs;
    for (const auto& output : requested_outputs)
      outputs.push_back(output.sample_kind() == metatomic::SampleKind::System
                            ? system_energy(results, device, gradients, selected_atoms.has_value(), output)
                            : atom_energy(results, device));
    return outputs;
  }

 private:
  // The selected atoms first, as the owned atoms: where[engine atom] = ours.
  // Empty when that changes nothing (every atom selected, in order).
  static std::vector<int> owned_first(const std::vector<int32_t>& selected, int n) {
    bool identity = int(selected.size()) == n;
    for (int i = 0; identity && i < n; ++i) identity = selected[i] == i;
    if (identity) return {};
    std::vector<int> where(n, -1);
    int next = 0;
    for (int32_t a : selected) {
      if (where[a] >= 0) throw metatomic::Error("pet-kokkos: an atom is selected twice");
      where[a] = next++;
    }
    for (int a = 0; a < n; ++a)
      if (where[a] < 0) where[a] = next++;
    return where;
  }

  int system_charge(const metatomic::System& system) const {
    return calc_->hypers().system_conditioning ? int(std::lround(system_scalar(system, "charge"))) : 0;
  }
  int system_spin(const metatomic::System& system) const {
    return calc_->hypers().system_conditioning ? int(std::lround(system_scalar(system, "spin_multiplicity"))) : 1;
  }

  SystemResult evaluate_host(const metatomic::System& system, std::vector<int32_t> selected, bool gradients) const {
    SystemResult out;
    const int n = int(system.size());
    out.selected = std::move(selected);
    out.n_atoms = n;
    out.host.assign(out.size(gradients), 0.0);
    if (out.selected.empty()) return out;  // nothing counts: zero energy, zero gradient

    const auto types = host_values<int32_t>(system.types(), "types");
    const auto positions = host_values<double>(system.positions(), "positions");
    const auto cell = host_values<double>(system.cell(), "cell");

    std::vector<int> where = owned_first(out.selected, n), order(n);
    if (where.empty())
      for (int a = 0; a < n; ++a) where.push_back(a);
    for (int a = 0; a < n; ++a) order[where[a]] = a;

    std::vector<int> z(n);
    std::vector<double> pos(std::size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
      z[i] = types[order[i]];
      for (int d = 0; d < 3; ++d) pos[3 * i + d] = positions[3 * std::size_t(order[i]) + d];
    }

    // The engine's pairs, grouped by centre in list order. The vectors are
    // recomputed from the positions, shifts and cell (r_j + S.H - r_i).
    auto pairs = system.pairs(pairs_);
    auto samples = pairs.samples();
    if (!has_names(samples, {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"}))
      throw metatomic::Error("pet-kokkos: unexpected pair list samples");
    auto p = samples.values();
    const std::size_t P = samples.count();
    std::vector<int> offsets(n + 1, 0), neighbors(P), shifts(3 * P);
    bool shifted = false;
    for (std::size_t k = 0; k < P; ++k) ++offsets[where[p(k, 0)] + 1];
    for (int i = 0; i < n; ++i) offsets[i + 1] += offsets[i];
    std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t k = 0; k < P; ++k) {
      const int e = cursor[where[p(k, 0)]]++;
      neighbors[e] = where[p(k, 1)];
      for (int d = 0; d < 3; ++d) shifts[3 * e + d] = p(k, 2 + d), shifted |= shifts[3 * e + d] != 0;
    }

    const int n_local = int(out.selected.size());
    pet::EdgeListView view;
    view.n_atoms = n;
    view.n_local = n_local == n ? -1 : n_local;
    view.positions = pos.data();
    view.atomic_numbers = z.data();
    view.offsets = offsets.data();
    view.neighbors = neighbors.data();
    view.shifts = shifted ? shifts.data() : nullptr;
    view.cell = shifted ? cell.data() : nullptr;
    view.charge = system_charge(system);
    view.spin_multiplicity = system_spin(system);

    const pet::Results r = calc_->compute_edges(view, gradients, false);
    auto& v = out.host;
    v[SystemResult::kEnergy] = r.energy.at(0);
    for (int i = 0; i < n_local; ++i) v[SystemResult::kPerAtom + i] = r.per_atom_energy.at(i);
    if (gradients) {
      const std::size_t g = out.gradient();
      for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) v[g + 3 * std::size_t(order[i]) + d] = -r.forces.at(3 * std::size_t(i) + d);
      // The virial W = dE/d(eps), Voigt [xx yy zz xy xz yz], symmetric.
      const auto& w = r.virial;
      const double strain[9] = {w.at(0), w.at(3), w.at(4), w.at(3), w.at(1), w.at(5), w.at(4), w.at(5), w.at(2)};
      std::copy(strain, strain + 9, v.begin() + SystemResult::kStrain);
    }
    return out;
  }

  // Public only because nvcc wants a device lambda's enclosing function to be.
 public:
  // The same with everything in device memory: the engine's arrays are read in
  // place (the pair indices too, when its labels live there) and the answer
  // stays on the device. Only the cell, charge and spin, and a selection's
  // renumbering, cross the bus. (The labels of the outputs are uploaded too
  // where they are a handful of numbers, and written on the device otherwise.)
  SystemResult evaluate_device(const metatomic::System& system, std::vector<int32_t> selected, bool gradients) const {
    using DView = pet_metatomic::KokkosArray<double>::DeviceView;
    using IView = Kokkos::View<int*, pet::MemSpace>;
    SystemResult out;
    const int n = int(system.size());
    out.selected = std::move(selected);
    out.n_atoms = n;
    out.device = DView("mta:result", out.size(gradients));
    if (out.selected.empty()) return out;

    auto types_t = system.types();
    auto positions_t = system.positions();
    IView types_copy;
    DView positions_copy;
    const int* types = device_contiguous<int>(types_t, types_copy, "types");
    const double* positions = device_contiguous<double>(positions_t, positions_copy, "positions");
    const auto cell = cell_on_host(system);

    auto pairs = system.pairs(pairs_);
    auto samples = pairs.samples();
    if (!has_names(samples, {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"}))
      throw metatomic::Error("pet-kokkos: unexpected pair list samples");
    const std::size_t P = samples.count();
    // The pair indices in place: metatensor keeps a block's labels where its
    // values are, and metatomic the pairs' values on the system's device.
    std::optional<metatensor::DLPackArray<int32_t>> indices;
    const int* pair_data = nullptr;
    long stride_pair = 5, stride_col = 1;
    if (P > 0) {
      indices.emplace(samples.values(pet_metatomic::kokkos_device()));
      pair_data = indices->data();
      stride_pair = long(indices->strides()[0]), stride_col = long(indices->strides()[1]);
    }

    const std::vector<int> where = owned_first(out.selected, n);
    IView renumber;
    if (!where.empty()) {
      renumber = IView("mta:renumber", n);
      Kokkos::deep_copy(renumber, Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(where.data(), n));
    }

    const int n_local = int(out.selected.size());
    pet::DevicePairListView view;
    view.n_atoms = n;
    view.n_local = n_local == n ? -1 : n_local;
    view.n_pairs = int(P);
    view.pairs = pair_data;
    view.stride_pair = stride_pair, view.stride_col = stride_col;
    view.atomic_numbers = types;
    view.renumber = where.empty() ? nullptr : renumber.data();
    view.charge = system_charge(system);
    view.spin_multiplicity = system_spin(system);

    const pet::BatchResult r = calc_->compute_pairs(view, positions, cell.data(), gradients);

    // Packed as SystemResult says, in the engine's numbering.
    auto v = out.device;
    auto energy = r.energy, per_atom = r.per_atom;
    const int g = int(out.gradient());
    Kokkos::parallel_for(
        "mta_energies", Kokkos::RangePolicy<pet::ExecSpace>(0, n_local + 1), KOKKOS_LAMBDA(int i) {
          if (i == 0)
            v(SystemResult::kEnergy) = energy(0);
          else
            v(SystemResult::kPerAtom + i - 1) = per_atom(i - 1);
        });
    if (gradients) {
      auto forces = r.forces, virial = r.virial;
      const bool has_ren = !where.empty();
      Kokkos::parallel_for(
          "mta_gradients", Kokkos::RangePolicy<pet::ExecSpace>(0, n), KOKKOS_LAMBDA(int a) {
            const int i = has_ren ? renumber(a) : a;
            for (int d = 0; d < 3; ++d) v(g + 3 * a + d) = -forces(i, d);
            if (a == 0) {
              constexpr int voigt[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
              for (int k = 0; k < 9; ++k) v(SystemResult::kStrain + k) = virial(0, voigt[k]);
            }
          });
    }
    pet::ExecSpace().fence();  // r's views are the Calculator's, and the inputs the engine's
    return out;
  }

 private:
  std::array<double, 9> cell_on_host(const metatomic::System& system) const {
    auto cell = system.cell();
    const DLTensor& t = cell->dl_tensor;
    std::array<double, 9> out{};
    if (t.device.device_type == kDLCPU) {
      const auto v = host_values<double>(cell, "cell");
      std::copy(v.begin(), v.end(), out.begin());
      return out;
    }
    pet_metatomic::KokkosArray<double>::DeviceView keep;
    const double* data = device_contiguous<double>(cell, keep, "cell");
    Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(out.data(), 9),
                      Kokkos::View<const double*, pet::MemSpace, Kokkos::MemoryUnmanaged>(data, 9));
    return out;
  }

  // One row per system; with a selection, only the systems it has atoms in.
  // A gradient's "sample" is the row.
  metatensor::TensorMap system_energy(const std::vector<SystemResult>& results, bool device, bool gradients,
                                      bool selection, const metatomic::Quantity& request) const {
    std::vector<int32_t> ids, rows;
    for (std::size_t s = 0; s < results.size(); ++s) {
      if (selection && results[s].selected.empty()) continue;
      ids.push_back(int32_t(s)), rows.push_back(int32_t(rows.size()));
    }
    const std::size_t R = ids.size();
    Output energy(device, {R, 1});
    for (std::size_t r = 0; r < R; ++r) energy.put(r, results[ids[r]], SystemResult::kEnergy, 1);
    const auto property = labels_on(device, {"energy"}, {0});
    metatensor::TensorBlock block(energy.release(), labels_on(device, {"system"}, ids), {}, property);

    for (auto g : request.gradients()) {
      if (!gradients) throw metatomic::Error("pet-kokkos: internal error, gradients were not computed");
      if (g == metatomic::Gradients::Positions) {
        std::size_t total = 0;
        for (int32_t s : ids) total += results[s].n_atoms;
        Output values(device, {total, 3, 1});
        std::size_t at = 0;
        for (std::size_t r = 0; r < R; ++r) {
          const auto& res = results[ids[r]];
          values.put(3 * at, res, res.gradient(), 3 * res.n_atoms);
          at += res.n_atoms;
        }
        block.add_gradient("positions", metatensor::TensorBlock(values.release(), gradient_samples(results, ids, device),
                                                                {labels_on(device, {"xyz"}, {0, 1, 2})}, property));
      } else {
        Output values(device, {R, 3, 3, 1});
        for (std::size_t r = 0; r < R; ++r) values.put(9 * r, results[ids[r]], SystemResult::kStrain, 9);
        block.add_gradient("strain", metatensor::TensorBlock(
                                         values.release(), labels_on(device, {"sample"}, rows),
                                         {labels_on(device, {"xyz_1"}, {0, 1, 2}), labels_on(device, {"xyz_2"}, {0, 1, 2})},
                                         property));
      }
    }
    return single_block_map(std::move(block), device);
  }

  metatensor::TensorMap atom_energy(const std::vector<SystemResult>& results, bool device) const {
    std::size_t rows = 0;
    for (const auto& r : results) rows += r.selected.size();
    Output values(device, {rows, 1});
    std::size_t at = 0;
    for (const auto& r : results) {
      values.put(at, r, SystemResult::kPerAtom, r.selected.size());
      at += r.selected.size();
    }
    return single_block_map(metatensor::TensorBlock(values.release(), atom_samples(results, device), {},
                                                    labels_on(device, {"energy"}, {0})),
                            device);
  }

 private:
  std::string name_;
  std::unique_ptr<pet::Calculator> calc_;
  metatomic::PairListOptions pairs_ = metatomic::PairListOptions::builder().cutoff(1.0).full_list(true).build();
};

// `load_from` is a model name or a path prefix to <spec>.json and
// <spec>.safetensors, resolved as pet::Calculator does. Anything that does not
// resolve is not ours: nullptr, so metatomic can try another plugin.
std::unique_ptr<metatomic::BaseModel> load_model(const std::string& load_from,
                                                 const std::map<std::string, std::string>& options) {
  std::string json, weights;
  try {
    pet::resolve_model(load_from, json, weights);
  } catch (const std::exception&) {
    return nullptr;
  }
  return std::make_unique<PetModel>(load_from, json, weights, parse_options(options));
}

}  // namespace

MTA_REGISTER_CXX_PLUGIN("pet-kokkos", load_model);
