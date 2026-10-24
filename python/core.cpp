// pet_kokkos._core: pet::Calculator for Python (nanobind).
//
// Inputs are NumPy arrays on the host; results come back as NumPy arrays that
// own their memory. Kokkos starts with the first Calculator; the package's
// atexit hook closes the live ones and finalizes it (pet_kokkos/__init__.py).
// The GEMM and attention modes are process-wide, as in pet-kokkos itself.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pet/calculator.hpp"
#include "pet/gemm.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

void start_kokkos() {
  if (Kokkos::is_finalized())
    throw std::runtime_error("pet_kokkos: Kokkos was finalized already (is the interpreter shutting down?)");
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings s;
    s.set_disable_warnings(true);
    Kokkos::initialize(s);
  }
}

void finalize_kokkos() {
  if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

// A NumPy array owning `v`, of the given shape.
template <class T>
nb::ndarray<nb::numpy, T> owned(std::vector<T> v, std::initializer_list<std::size_t> shape) {
  auto* p = new std::vector<T>(std::move(v));
  nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
  return nb::ndarray<nb::numpy, T>(p->data(), shape, owner);
}

using Numbers = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Cell = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;

pet::System make_system(Numbers numbers, Positions positions, std::optional<Cell> cell, std::array<bool, 3> pbc,
                        int charge, int spin_multiplicity) {
  const std::size_t n = numbers.shape(0);
  if (positions.shape(0) != n) throw std::invalid_argument("pet_kokkos: numbers and positions differ in length");
  pet::System s;
  s.n_atoms = int(n);
  s.atomic_numbers.assign(numbers.data(), numbers.data() + n);
  s.positions.assign(positions.data(), positions.data() + 3 * n);
  if (cell)
    for (int i = 0; i < 9; ++i) s.cell[i] = cell->data()[i];
  else if (pbc[0] || pbc[1] || pbc[2])
    throw std::invalid_argument("pet_kokkos: a periodic system needs a cell");
  s.pbc = pbc;
  s.charge = charge;
  s.spin_multiplicity = spin_multiplicity;
  return s;
}

// Structure b of r (atoms [a0, a0 + n)) as a dict.
nb::dict structure(const pet::Results& r, std::size_t b, std::size_t a0, std::size_t n, bool forces) {
  nb::dict d;
  d["energy"] = r.energy[b];
  d["energies"] = owned(std::vector<double>(r.per_atom_energy.begin() + a0, r.per_atom_energy.begin() + a0 + n), {n});
  if (forces && !r.forces.empty()) {
    d["forces"] = owned(std::vector<double>(r.forces.begin() + 3 * a0, r.forces.begin() + 3 * (a0 + n)), {n, 3});
    if (r.virial.size() >= 6 * (b + 1))
      d["virial"] = owned(std::vector<double>(r.virial.begin() + 6 * b, r.virial.begin() + 6 * (b + 1)), {6});
  }
  return d;
}

class Calculator {
 public:
  Calculator(const std::string& model, std::optional<std::string> gemm, std::optional<std::string> attention,
             bool device_neighbors, bool cache_neighbors, std::optional<std::string> recompute,
             double memory_budget_gib, int max_batch_atoms) {
    start_kokkos();
    pet::Options o;
    o.device_neighbors = device_neighbors;
    o.cache_neighbors = cache_neighbors;
    o.memory_budget_bytes = std::size_t(memory_budget_gib * double(1ull << 30));
    o.max_batch_atoms = max_batch_atoms;
    if (recompute) {
      bool found = false;
      for (auto r : {pet::Recompute::Auto, pet::Recompute::Never, pet::Recompute::Wide, pet::Recompute::Layers})
        if (*recompute == pet::recompute_name(r)) o.recompute = r, found = true;
      if (!found) throw std::invalid_argument("pet_kokkos: recompute must be auto, never, wide or layers");
    }
    if (gemm) pet::gemm_mode() = pet::parse_gemm_mode(gemm->c_str());
    if (attention) {
      if (*attention != "native" && *attention != "tf32")
        throw std::invalid_argument("pet_kokkos: attention must be native or tf32");
      pet::attention_mode() = *attention == "tf32" ? pet::AttentionMode::TF32 : pet::AttentionMode::Native;
    }
    calc_ = std::make_unique<pet::Calculator>(model, o);
  }

  nb::dict compute(Numbers numbers, Positions positions, std::optional<Cell> cell, std::array<bool, 3> pbc,
                   bool forces, int charge, int spin_multiplicity) const {
    const pet::System s = make_system(numbers, positions, cell, pbc, charge, spin_multiplicity);
    pet::Results r;
    {
      nb::gil_scoped_release release;
      r = get().compute(s, forces);
    }
    return structure(r, 0, 0, std::size_t(s.n_atoms), forces);
  }

  // Each item (numbers, positions, cell or None, pbc, charge, spin_multiplicity).
  nb::list compute_batch(const std::vector<std::tuple<Numbers, Positions, std::optional<Cell>, std::array<bool, 3>,
                                                      int, int>>& items,
                         bool forces) const {
    std::vector<pet::System> systems;
    for (const auto& [z, x, c, p, q, m] : items) systems.push_back(make_system(z, x, c, p, q, m));
    pet::Results r;
    {
      nb::gil_scoped_release release;
      r = get().compute(systems, forces);
    }
    nb::list out;
    std::size_t a0 = 0;
    for (std::size_t b = 0; b < systems.size(); ++b) {
      out.append(structure(r, b, a0, std::size_t(systems[b].n_atoms), forces));
      a0 += systems[b].n_atoms;
    }
    return out;
  }

  void close() { calc_.reset(); }
  bool closed() const { return !calc_; }
  double cutoff() const { return get().cutoff(); }
  std::vector<int> atomic_types() const { return get().atomic_types(); }
  std::string length_unit() const { return get().length_unit(); }
  std::string energy_unit() const { return get().energy_unit(); }
  double energy_scale() const { return get().energy_scale(); }
  std::string memory_report() const { return get().memory_report(); }

 private:
  const pet::Calculator& get() const {
    if (!calc_) throw std::runtime_error("pet_kokkos: this Calculator was closed");
    return *calc_;
  }
  std::unique_ptr<pet::Calculator> calc_;
};

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "pet-kokkos: PET machine-learned potentials on Kokkos";

  m.def("backend", [] { return std::string(Kokkos::DefaultExecutionSpace::name()); },
        "The Kokkos execution space this build computes on (Serial, OpenMP, Threads, Cuda, HIP).");
  m.def("model_search_dirs", &pet::model_search_dirs, "Where named models are looked for, in order.");
  m.def(
      "find_model",
      [](const std::string& spec) -> std::optional<std::pair<std::string, std::string>> {
        try {
          std::string j, w;
          pet::resolve_model(spec, j, w);
          return std::make_pair(j, w);
        } catch (const std::exception&) {
          return std::nullopt;
        }
      },
      "spec"_a, "A model's (json, safetensors) paths, or None when it is not found.");
  m.def("_finalize", &finalize_kokkos);

  nb::class_<Calculator>(m, "Calculator")
      .def(nb::init<const std::string&, std::optional<std::string>, std::optional<std::string>, bool, bool,
                    std::optional<std::string>, double, int>(),
           "model"_a, "gemm"_a = nb::none(), "attention"_a = nb::none(), "device_neighbors"_a = true,
           "cache_neighbors"_a = false, "recompute"_a = nb::none(), "memory_budget_gib"_a = 0.0,
           "max_batch_atoms"_a = 0)
      .def("compute", &Calculator::compute, "numbers"_a, "positions"_a, "cell"_a = nb::none(),
           "pbc"_a = std::array<bool, 3>{false, false, false}, "forces"_a = true, "charge"_a = 0,
           "spin_multiplicity"_a = 1)
      .def("compute_batch", &Calculator::compute_batch, "items"_a, "forces"_a = true)
      .def("close", &Calculator::close)
      .def_prop_ro("closed", &Calculator::closed)
      .def_prop_ro("cutoff", &Calculator::cutoff)
      .def_prop_ro("atomic_types", &Calculator::atomic_types)
      .def_prop_ro("length_unit", &Calculator::length_unit)
      .def_prop_ro("energy_unit", &Calculator::energy_unit)
      .def_prop_ro("energy_scale", &Calculator::energy_scale)
      .def("memory_report", &Calculator::memory_report);
}
