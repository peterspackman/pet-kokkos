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
#include "pet/d3.hpp"
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
  if (b < r.energy_uncertainty.size()) {
    d["energy_uncertainty"] = r.energy_uncertainty[b];
    d["energies_uncertainty"] = owned(
        std::vector<double>(r.per_atom_energy_uncertainty.begin() + a0, r.per_atom_energy_uncertainty.begin() + a0 + n),
        {n});
    if (r.n_ensemble > 0) {
      const std::size_t K = r.n_ensemble;
      d["energy_ensemble"] = owned(
          std::vector<double>(r.energy_ensemble.begin() + b * K, r.energy_ensemble.begin() + (b + 1) * K), {K});
    }
  }
  if (forces && !r.forces.empty()) {
    d["forces"] = owned(std::vector<double>(r.forces.begin() + 3 * a0, r.forces.begin() + 3 * (a0 + n)), {n, 3});
    if (r.virial.size() >= 6 * (b + 1))
      d["virial"] = owned(std::vector<double>(r.virial.begin() + 6 * b, r.virial.begin() + 6 * (b + 1)), {6});
  }
  return d;
}

// pet::D3: settings at construction, the correction made on first use.
class D3 {
 public:
  D3(std::optional<std::string> functional, const std::string& damping, bool atm, std::optional<double> s6,
     std::optional<double> s8, std::optional<double> a1, std::optional<double> a2_bohr, std::optional<double> rs6,
     std::optional<double> rs8, std::optional<double> alpha, std::optional<double> s9, std::optional<double> cutoff,
     std::optional<double> cutoff_taper, std::optional<double> cn_cutoff, std::optional<double> cn_taper,
     std::optional<double> atm_cutoff, std::optional<double> atm_taper, bool tail, bool reference,
     bool mixed_precision) {
    const pet::D3Damping d = pet::d3_damping(damping);
    if (functional) s_ = pet::D3Settings::functional(*functional, d, atm);
    else s_.params.damping = d, s_.params.s9 = atm ? 1.0 : 0.0;
    pet::D3Params& p = s_.params;
    if (s6) p.s6 = *s6;
    if (s8) p.s8 = *s8;
    if (a1) p.a1 = *a1;
    if (a2_bohr) p.a2 = *a2_bohr * 0.529177210903;
    if (rs6) p.rs6 = *rs6;
    if (rs8) p.rs8 = *rs8;
    if (alpha) p.alpha = *alpha;
    if (s9) p.s9 = *s9;
    pet::D3Cutoffs& c = s_.cutoffs;
    if (reference) c = pet::D3Cutoffs::reference();
    if (cutoff) c.pair = *cutoff;
    if (cutoff_taper) c.pair_taper = *cutoff_taper;
    if (cn_cutoff) c.cn = *cn_cutoff;
    if (cn_taper) c.cn_taper = *cn_taper;
    if (atm_cutoff) c.atm = *atm_cutoff;
    if (atm_taper) c.atm_taper = *atm_taper;
    c.tail = tail && !reference;
    s_.mixed_precision = mixed_precision;
    if (!functional && !(a1 || rs6))
      throw std::invalid_argument("pet_kokkos.D3: give a functional, or its parameters (s8 and a1, a2 or rs6)");
  }
  const pet::D3Settings& settings() const { return s_; }

  nb::dict compute(Numbers numbers, Positions positions, std::optional<Cell> cell, std::array<bool, 3> pbc,
                   bool forces) {
    const pet::System s = make_system(numbers, positions, cell, pbc, 0, 1);
    pet::Results r;
    {
      nb::gil_scoped_release release;
      r = get().compute(s, forces);
    }
    return structure(r, 0, 0, std::size_t(s.n_atoms), forces);
  }
  nb::list compute_batch(
      const std::vector<std::tuple<Numbers, Positions, std::optional<Cell>, std::array<bool, 3>>>& items,
      bool forces) {
    std::vector<pet::System> systems;
    for (const auto& [z, x, c, p] : items) systems.push_back(make_system(z, x, c, p, 0, 1));
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
  nb::dict params() const {
    const pet::D3Params& p = s_.params;
    nb::dict d;
    d["damping"] = p.damping == pet::D3Damping::BJ ? "bj" : "zero";
    d["s6"] = p.s6, d["s8"] = p.s8, d["a1"] = p.a1, d["a2"] = p.a2 / 0.529177210903;
    d["rs6"] = p.rs6, d["rs8"] = p.rs8, d["alpha"] = p.alpha, d["s9"] = p.s9;
    return d;
  }
  nb::dict cutoffs() const {
    const pet::D3Cutoffs& c = s_.cutoffs;
    nb::dict d;
    d["cutoff"] = c.pair, d["cutoff_taper"] = c.pair_taper, d["cn_cutoff"] = c.cn, d["cn_taper"] = c.cn_taper;
    d["atm_cutoff"] = c.atm, d["atm_taper"] = c.atm_taper, d["tail"] = c.tail;
    return d;
  }
  double ghost_cutoff(bool exchange) {
    const pet::D3Cutoffs& c = s_.cutoffs;
    const double reach = std::max(c.pair, s_.params.s9 != 0.0 ? c.atm : 0.0);
    return exchange ? reach : reach + c.cn;
  }
  void close() { d3_.reset(); }

 private:
  pet::D3& get() {
    if (!d3_) {
      start_kokkos();
      d3_ = std::make_unique<pet::D3>(s_);
    }
    return *d3_;
  }
  pet::D3Settings s_;
  std::unique_ptr<pet::D3> d3_;
};

class Calculator {
 public:
  Calculator(const std::string& model, std::optional<std::string> gemm, std::optional<std::string> attention,
             bool device_neighbors, bool cache_neighbors, std::optional<std::string> recompute,
             double memory_budget_gib, int max_batch_atoms, bool non_conservative, bool uncertainty,
             bool smooth_cutoff, const D3* d3) {
    start_kokkos();
    pet::Options o;
    if (d3) o.d3 = d3->settings();
    o.non_conservative = non_conservative;
    o.uncertainty = uncertainty;
    o.smooth_cutoff = smooth_cutoff;
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
      if (*attention != "native" && *attention != "tf32" && *attention != "3xtf32")
        throw std::invalid_argument("pet_kokkos: attention must be native, tf32 or 3xtf32");
      pet::attention_mode() = pet::parse_attention_mode(attention->c_str());
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

  // d(forces)/dt [K, N, 3] and d(virial)/dt [K, 6] along K directions, exactly
  // (pet::Calculator::response): displacements [K, N, 3], strains [K, 3, 3].
  nb::dict response(Numbers numbers, Positions positions, std::optional<Cell> cell, std::array<bool, 3> pbc,
                    nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu> displacements,
                    nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu> strains, int charge,
                    int spin_multiplicity) const {
    const pet::System s = make_system(numbers, positions, cell, pbc, charge, spin_multiplicity);
    const std::size_t n = s.n_atoms, K = std::max(displacements.shape(0), strains.shape(0));
    if ((displacements.size() && (displacements.shape(0) != K || displacements.shape(1) != n ||
                                  displacements.shape(2) != 3)) ||
        (strains.size() && (strains.shape(0) != K || strains.shape(1) != 3 || strains.shape(2) != 3)))
      throw std::invalid_argument("pet_kokkos: displacements must be [K, N, 3] and strains [K, 3, 3]");
    std::vector<double> u(displacements.data(), displacements.data() + displacements.size());
    std::vector<double> e(strains.data(), strains.data() + strains.size());
    pet::Calculator::Response r;
    {
      nb::gil_scoped_release release;
      r = get().response(s, int(K), u, e);
    }
    nb::dict d;
    d["forces"] = owned(std::move(r.forces), {K, n, 3});
    d["virial"] = owned(std::move(r.virial), {K, 6});
    d["base_forces"] = owned(std::move(r.base), {n, 3});
    return d;
  }

  void close() { calc_.reset(); }
  bool closed() const { return !calc_; }
  double cutoff() const { return get().cutoff(); }
  std::vector<int> atomic_types() const { return get().atomic_types(); }
  std::string length_unit() const { return get().length_unit(); }
  std::string energy_unit() const { return get().energy_unit(); }
  double energy_scale() const { return get().energy_scale(); }
  bool has_non_conservative_forces() const { return get().has_non_conservative_forces(); }
  bool has_non_conservative_stress() const { return get().has_non_conservative_stress(); }
  bool has_uncertainty() const { return get().has_uncertainty(); }
  int n_ensemble() const { return get().n_ensemble(); }
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
      "precision_modes",
      [] {
        const char* gemm[] = {"native", "tf32", "bf16", "fp16", "fp8", "ozaki", "3xtf32"};
        return std::make_pair(std::string(gemm[int(pet::gemm_mode())]),
                              std::string(pet::attention_mode_name(pet::attention_mode())));
      },
      "The process-wide (gemm, attention) modes: native unless set by a Calculator or PET_GEMM / PET_ATTN.");
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
  m.def(
      "d3_functionals",
      [](const std::string& damping) { return pet::d3_functionals(pet::d3_damping(damping)); }, "damping"_a = "bj",
      "The functionals with D3 parameters for a damping (bj or zero).");

  nb::class_<D3>(m, "D3")
      .def(nb::init<std::optional<std::string>, const std::string&, bool, std::optional<double>, std::optional<double>,
                    std::optional<double>, std::optional<double>, std::optional<double>, std::optional<double>,
                    std::optional<double>, std::optional<double>, std::optional<double>, std::optional<double>,
                    std::optional<double>, std::optional<double>, std::optional<double>, std::optional<double>, bool,
                    bool, bool>(),
           "functional"_a.none(), "damping"_a = "bj", "atm"_a = false, "s6"_a = nb::none(), "s8"_a = nb::none(),
           "a1"_a = nb::none(), "a2"_a = nb::none(), "rs6"_a = nb::none(), "rs8"_a = nb::none(),
           "alpha"_a = nb::none(), "s9"_a = nb::none(), "cutoff"_a = nb::none(), "cutoff_taper"_a = nb::none(),
           "cn_cutoff"_a = nb::none(), "cn_taper"_a = nb::none(), "atm_cutoff"_a = nb::none(),
           "atm_taper"_a = nb::none(), "tail"_a = true, "reference"_a = false,
           "mixed_precision"_a = false)
      .def("compute", &D3::compute, "numbers"_a, "positions"_a, "cell"_a = nb::none(),
           "pbc"_a = std::array<bool, 3>{false, false, false}, "forces"_a = true)
      .def("compute_batch", &D3::compute_batch, "items"_a, "forces"_a = true)
      .def("ghost_cutoff", &D3::ghost_cutoff, "exchange"_a = false)
      .def("close", &D3::close)
      .def_prop_ro("params", &D3::params)
      .def_prop_ro("cutoffs", &D3::cutoffs);

  nb::class_<Calculator>(m, "Calculator")
      .def(nb::init<const std::string&, std::optional<std::string>, std::optional<std::string>, bool, bool,
                    std::optional<std::string>, double, int, bool, bool, bool, const D3*>(),
           "model"_a, "gemm"_a = nb::none(), "attention"_a = nb::none(), "device_neighbors"_a = true,
           "cache_neighbors"_a = false, "recompute"_a = nb::none(), "memory_budget_gib"_a = 0.0,
           "max_batch_atoms"_a = 0, "non_conservative"_a = false, "uncertainty"_a = false,
           "smooth_cutoff"_a = false, "d3"_a = nb::none())
      .def("compute", &Calculator::compute, "numbers"_a, "positions"_a, "cell"_a = nb::none(),
           "pbc"_a = std::array<bool, 3>{false, false, false}, "forces"_a = true, "charge"_a = 0,
           "spin_multiplicity"_a = 1)
      .def("compute_batch", &Calculator::compute_batch, "items"_a, "forces"_a = true)
      .def("response", &Calculator::response, "numbers"_a, "positions"_a, "cell"_a.none(), "pbc"_a,
           "displacements"_a, "strains"_a, "charge"_a = 0, "spin_multiplicity"_a = 1)
      .def("close", &Calculator::close)
      .def_prop_ro("closed", &Calculator::closed)
      .def_prop_ro("cutoff", &Calculator::cutoff)
      .def_prop_ro("atomic_types", &Calculator::atomic_types)
      .def_prop_ro("length_unit", &Calculator::length_unit)
      .def_prop_ro("energy_unit", &Calculator::energy_unit)
      .def_prop_ro("energy_scale", &Calculator::energy_scale)
      .def_prop_ro("has_non_conservative_forces", &Calculator::has_non_conservative_forces)
      .def_prop_ro("has_non_conservative_stress", &Calculator::has_non_conservative_stress)
      .def_prop_ro("has_uncertainty", &Calculator::has_uncertainty)
      .def_prop_ro("n_ensemble", &Calculator::n_ensemble)
      .def("memory_report", &Calculator::memory_report);
}
