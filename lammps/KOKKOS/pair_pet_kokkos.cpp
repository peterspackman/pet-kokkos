/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Peter Spackman
------------------------------------------------------------------------- */

#include "pair_pet_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "comm.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include "pet/calculator.hpp"

#include <type_traits>
#include <vector>

using namespace LAMMPS_NS;

template <class DeviceType> PairPETKokkos<DeviceType>::PairPETKokkos(LAMMPS *lmp) : PairPET(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  full_list = true;  // KOKKOS builds full lists on the device
  reverse_comm_device = 1;
}

template <class DeviceType> PairPETKokkos<DeviceType>::~PairPETKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  eatom = nullptr, vatom = nullptr;
}

template <class DeviceType> void PairPETKokkos<DeviceType>::init_style()
{
  PairPET::init_style();
  // The same list (full), built on the device.
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> && !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

// At a neighbour-list rebuild: LAMMPS's device list, copied to the host once, in
// pet-kokkos's form.
template <class DeviceType> void PairPETKokkos<DeviceType>::rebuild()
{
  atomKK->sync(Host, X_MASK | TAG_MASK | TYPE_MASK);
  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto ilist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_list->d_ilist);
  auto numneigh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_list->d_numneigh);
  auto neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_list->d_neighbors);
  List l;
  l.n = list->inum + (mode == Mode::Ghosts ? list->gnum : 0);
  l.ilist = ilist.data(), l.numneigh = numneigh.data();
  l.neighbor = [&](int i, int k) { return neighbors(i, k) & NEIGHMASK; };
  if (mode == Mode::Images) set_images(l, false);
  else if (mode == Mode::Exchange) set_exchange(l);
  else set_ghosts(l);
  listed = true;
}

template <class DeviceType> void PairPETKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  ev_init(eflag_in, vflag_in, 0);
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  if (neighbor->ago == 0 || !listed) rebuild();

  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  pet::Calculator::DeviceArrays a;
  if constexpr (std::is_same_v<typename decltype(x)::array_layout, Kokkos::LayoutRight>) {
    a.positions = x.data();
  } else {
    if (x_rows.extent(0) < x.extent(0)) x_rows = decltype(x_rows)("pet:x_rows", x.extent(0));
    Kokkos::deep_copy(Kokkos::subview(x_rows, std::make_pair(size_t(0), size_t(x.extent(0))), Kokkos::ALL), x);
    a.positions = x_rows.data();
  }
  a.forces = energy_only(0) ? nullptr : f.data();  // no forces: no backward pass
  a.per_atom_energy = eflag_atom ? k_eatom.template view<DeviceType>().data() : nullptr;
  a.per_atom_virial = vflag_atom && a.forces ? k_vatom.template view<DeviceType>().data() : nullptr;
  a.virial_scale = -1.0;  // LAMMPS's virial is minus pet-kokkos's (see pair_pet.cpp)
  double cell[9];
  cell_rows(cell);
  pet::Calculator::Totals t;
  try {
    t = calc->compute_step(a, cell);
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style pet/kk: {}", e.what());
  }

  if (eflag_global) eng_vdwl += t.energy;
  if (vflag_global && a.forces)
    for (int k = 0; k < 6; ++k) virial[k] -= t.virial[k];
  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }
  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }
  if (a.forces) atomKK->modified(execution_space, F_MASK);
}

// LAMMPS checks MPI only on several ranks; on one, its flag says nothing.
template <class DeviceType> bool PairPETKokkos<DeviceType>::mpi_gpu_aware() const
{
  return comm->nprocs > 1 ? lmp->kokkos->gpu_aware_flag : PairPET::mpi_gpu_aware();
}

// On the device through LAMMPS's device comm, unless it is set to the classic
// (host) one: then the host hooks.
template <class DeviceType> void PairPETKokkos<DeviceType>::atoms_forward(pet::RView1D a)
{
  if (lmp->kokkos->forward_pair_comm_classic) return PairPET::atoms_forward(a);
  d_atom = a;
  comm->forward_comm(this);
}

template <class DeviceType> void PairPETKokkos<DeviceType>::atoms_reverse(pet::RView1D a)
{
  if (lmp->kokkos->reverse_pair_comm_classic) return PairPET::atoms_reverse(a);
  d_atom = a;
  comm->reverse_comm(this);
  const int nlocal = atom->nlocal;
  Kokkos::parallel_for(
      "pet_zero_ghosts", Kokkos::RangePolicy<DeviceType>(nlocal, a.extent(0)), KOKKOS_LAMBDA(int i) { a(i) = 0; });
}

template <class DeviceType>
int PairPETKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_list, DAT::tdual_xfloat_1d &buf,
                                                         int, int *)
{
  auto list = k_list.view<DeviceType>();
  auto b = buf.view<DeviceType>();
  auto a = d_atom;
  Kokkos::parallel_for(
      "pet_pack_forward", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) { b(i) = a(list(i)); });
  return n;
}

template <class DeviceType>
void PairPETKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto b = buf.view<DeviceType>();
  auto a = d_atom;
  Kokkos::parallel_for(
      "pet_unpack_forward", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) { a(first + i) = b(i); });
}

template <class DeviceType>
int PairPETKokkos<DeviceType>::pack_reverse_comm_kokkos(int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto b = buf.view<DeviceType>();
  auto a = d_atom;
  Kokkos::parallel_for(
      "pet_pack_reverse", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) { b(i) = a(first + i); });
  return n;
}

template <class DeviceType>
void PairPETKokkos<DeviceType>::unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_list, DAT::tdual_xfloat_1d &buf)
{
  auto list = k_list.view<DeviceType>();
  auto b = buf.view<DeviceType>();
  auto a = d_atom;
  Kokkos::parallel_for(
      "pet_unpack_reverse", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) { a(list(i)) += b(i); });
}

namespace LAMMPS_NS {
template class PairPETKokkos<LMPDeviceType>;
}
