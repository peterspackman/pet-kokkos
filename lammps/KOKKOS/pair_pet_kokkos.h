/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pet/kk,PairPETKokkos<LMPDeviceType>);
PairStyle(pet/kk/device,PairPETKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_PET_KOKKOS_H
#define LMP_PAIR_PET_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "pair_pet.h"

namespace LAMMPS_NS {

// pair_style pet on LAMMPS's KOKKOS package: pet-kokkos shares LAMMPS's Kokkos
// and reads positions from, and adds forces and per-atom energy and virial to,
// LAMMPS's device arrays. The host sees the neighbour list only when LAMMPS
// rebuilds it, and the totals each step.
template <class DeviceType> class PairPETKokkos : public PairPET {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairPETKokkos(class LAMMPS *);
  ~PairPETKokkos() override;
  void compute(int, int) override;
  void init_style() override;

 private:
  typename AT::tdual_efloat_1d k_eatom;
  typename AT::tdual_virial_array k_vatom;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> x_rows;  // x, if LAMMPS's is not row-major
  void rebuild();
};

}    // namespace LAMMPS_NS

#endif
#endif
