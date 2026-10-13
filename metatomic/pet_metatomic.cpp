// pet-kokkos as a metatomic plugin: any engine that loads metatomic plugins
// (mta_load_plugin) gets PET's energy, forces and virial from pet::Calculator.
//
// The engine hands over each system with a full pair list at the model cutoff;
// the plugin turns it into an EdgeListView and calls compute_edges, so the
// engine's pairs are the edges (the adaptive cutoff prunes them as usual). The
// selected atoms are moved to the front of a system and become its owned atoms:
// the energy is theirs, the gradients are with respect to every atom.
//
// Inputs and outputs live on the host; the evaluation runs on whatever device
// this build's Kokkos has.
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

std::unique_ptr<metatensor::SimpleDataArray<double>> array(std::vector<uintptr_t> shape, std::vector<double> data) {
  return std::make_unique<metatensor::SimpleDataArray<double>>(std::move(shape), std::move(data));
}

metatensor::TensorMap single_block_map(metatensor::TensorBlock block) {
  std::vector<metatensor::TensorBlock> blocks;
  blocks.emplace_back(std::move(block));
  return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
}

// --- the model ---------------------------------------------------------------

// One system's answer, in the engine's atom order.
struct SystemResult {
  double energy = 0;
  std::vector<int32_t> selected;    // the atoms whose energy is counted, as selected
  std::vector<double> per_atom;     // [selected.size()]
  std::vector<double> gradient;     // [n_atoms * 3] dE/dr
  std::array<double, 9> strain{};   // dE/d(eps), 3x3 row-major
};

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
    // Everything an atom's energy depends on: one cutoff per message-passing
    // layer, and one more for the adaptive cutoff (ghost_cutoff).
    return metatomic::ModelCapabilities::builder()
        .atomic_types(types)
        .interaction_range(calc_->ghost_cutoff())
        .length_unit(calc_->length_unit())
        .supported_devices({metatomic::ModelCapabilities::Device::CPU})
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

    // Each system's selected atoms, in the order the selection lists them.
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

    std::vector<SystemResult> results;
    results.reserve(systems.size());
    for (std::size_t s = 0; s < systems.size(); ++s)
      results.push_back(evaluate(systems[s], std::move(selection[s]), positions || strain));

    std::vector<metatensor::TensorMap> outputs;
    for (const auto& output : requested_outputs)
      outputs.push_back(output.sample_kind() == metatomic::SampleKind::System
                            ? system_energy(systems, results, selected_atoms.has_value(), output)
                            : atom_energy(results));
    return outputs;
  }

 private:
  SystemResult evaluate(const metatomic::System& system, std::vector<int32_t> selected, bool gradients) const {
    SystemResult out;
    const int n = int(system.size());
    out.selected = std::move(selected);
    out.per_atom.assign(out.selected.size(), 0.0);
    if (gradients) out.gradient.assign(std::size_t(n) * 3, 0.0);
    if (out.selected.empty()) return out;  // nothing counts: zero energy, zero gradient

    const auto types = host_values<int32_t>(system.types(), "types");
    const auto positions = host_values<double>(system.positions(), "positions");
    const auto cell = host_values<double>(system.cell(), "cell");

    // The selected atoms first: they are the owned atoms. order[new] = old.
    std::vector<int> order, where(n, -1);
    order.reserve(n);
    for (int32_t a : out.selected) {
      if (where[a] >= 0) throw metatomic::Error("pet-kokkos: an atom is selected twice");
      where[a] = int(order.size()), order.push_back(a);
    }
    const int n_local = int(order.size());
    for (int a = 0; a < n; ++a)
      if (where[a] < 0) where[a] = int(order.size()), order.push_back(a);

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

    pet::EdgeListView view;
    view.n_atoms = n;
    view.n_local = n_local == n ? -1 : n_local;
    view.positions = pos.data();
    view.atomic_numbers = z.data();
    view.offsets = offsets.data();
    view.neighbors = neighbors.data();
    view.shifts = shifted ? shifts.data() : nullptr;
    view.cell = shifted ? cell.data() : nullptr;
    if (calc_->hypers().system_conditioning) {
      view.charge = int(std::lround(system_scalar(system, "charge")));
      view.spin_multiplicity = int(std::lround(system_scalar(system, "spin_multiplicity")));
    }

    const pet::Results r = calc_->compute_edges(view, gradients, false);
    out.energy = r.energy.at(0);
    for (int i = 0; i < n_local; ++i) out.per_atom[i] = r.per_atom_energy.at(i);
    if (gradients) {
      for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) out.gradient[3 * std::size_t(order[i]) + d] = -r.forces.at(3 * std::size_t(i) + d);
      // The virial W = dE/d(eps), Voigt [xx yy zz xy xz yz], symmetric.
      const auto& w = r.virial;
      out.strain = {w.at(0), w.at(3), w.at(4), w.at(3), w.at(1), w.at(5), w.at(4), w.at(5), w.at(2)};
    }
    return out;
  }

  // One row per system; with a selection, only the systems it has atoms in.
  // A gradient's "sample" is the row.
  metatensor::TensorMap system_energy(const std::vector<metatomic::System>& systems,
                                      const std::vector<SystemResult>& results, bool selection,
                                      const metatomic::Quantity& request) const {
    std::vector<int32_t> ids, rows;
    std::vector<double> energy;
    for (std::size_t s = 0; s < results.size(); ++s) {
      if (selection && results[s].selected.empty()) continue;
      ids.push_back(int32_t(s)), rows.push_back(int32_t(rows.size())), energy.push_back(results[s].energy);
    }
    const std::size_t R = ids.size();
    metatensor::TensorBlock block(array({R, 1}, std::move(energy)), labels({"system"}, ids), {},
                                  metatensor::Labels({"energy"}, {{0}}));

    for (auto g : request.gradients()) {
      if (g == metatomic::Gradients::Positions) {
        std::vector<int32_t> samples;
        std::vector<double> values;
        for (std::size_t r = 0; r < R; ++r) {
          const std::size_t s = std::size_t(ids[r]);
          for (std::size_t a = 0; a < systems[s].size(); ++a) {
            samples.insert(samples.end(), {int32_t(r), int32_t(s), int32_t(a)});
            for (int d = 0; d < 3; ++d) values.push_back(results[s].gradient[3 * a + d]);
          }
        }
        const std::size_t n = samples.size() / 3;
        block.add_gradient(
            "positions",
            metatensor::TensorBlock(array({n, 3, 1}, std::move(values)), labels({"sample", "system", "atom"}, samples),
                                    {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
                                    metatensor::Labels({"energy"}, {{0}})));
      } else {
        std::vector<double> values;
        for (int32_t s : ids) values.insert(values.end(), results[s].strain.begin(), results[s].strain.end());
        block.add_gradient("strain", metatensor::TensorBlock(
                                         array({R, 3, 3, 1}, std::move(values)), labels({"sample"}, rows),
                                         {metatensor::Labels({"xyz_1"}, {{0}, {1}, {2}}),
                                          metatensor::Labels({"xyz_2"}, {{0}, {1}, {2}})},
                                         metatensor::Labels({"energy"}, {{0}})));
      }
    }
    return single_block_map(std::move(block));
  }

  metatensor::TensorMap atom_energy(const std::vector<SystemResult>& results) const {
    std::vector<int32_t> samples;
    std::vector<double> values;
    for (std::size_t s = 0; s < results.size(); ++s)
      for (std::size_t i = 0; i < results[s].selected.size(); ++i) {
        samples.insert(samples.end(), {int32_t(s), results[s].selected[i]});
        values.push_back(results[s].per_atom[i]);
      }
    const std::size_t rows = values.size();
    return single_block_map(metatensor::TensorBlock(array({rows, 1}, std::move(values)), labels({"system", "atom"}, samples),
                                                    {}, metatensor::Labels({"energy"}, {{0}})));
  }

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
