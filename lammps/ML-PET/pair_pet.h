/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pet,PairPET);
// clang-format on
#else

#ifndef LMP_PAIR_PET_H
#define LMP_PAIR_PET_H

#include "pair.h"

#include <functional>
#include <memory>
#include <vector>

namespace pet {
class Calculator;
}

namespace LAMMPS_NS {

// PET machine-learning potentials through pet-kokkos, on LAMMPS's own neighbour
// list: handed over when LAMMPS rebuilds it, positions every step. Two ways to
// give it the atoms (see pair_pet.cpp): periodic images of the owned atoms, or
// ghosts. pet/kk (KOKKOS package) keeps every step on the device.
class PairPET : public Pair {
 public:
  PairPET(class LAMMPS *);
  ~PairPET() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  enum class Mode { Images, Ghosts };
  Mode mode = Mode::Images;
  std::unique_ptr<pet::Calculator> calc;
  double cutoff = 0.0;                   // the model's
  std::vector<int> type_z;               // atomic number of each LAMMPS type
  std::vector<int> off, nbr, z, shift;   // the list at the last rebuild, pet-kokkos's form
  bool listed = false;
  bool full_list = false;                // images mode on a full list (pet/kk)

  // A neighbour list, however it is held: centre ilist[ii] has numneigh[i]
  // neighbours, neighbor(i, k).
  struct List {
    int n = 0;
    const int *ilist = nullptr, *numneigh = nullptr;
    std::function<int(int, int)> neighbor;
  };
  // Energy only, no forces: LAMMPS versions that pass ENERGY_ONLY (Monte Carlo,
  // numdiff, FEP) set Pair::eflag_only after ev_init; others lack the member.
  template <class T = Pair> auto energy_only(int) -> decltype(bool(static_cast<T *>(this)->eflag_only))
  {
    return static_cast<T *>(this)->eflag_only;
  }
  bool energy_only(long) { return false; }

  void set_images(const List &l, bool half);
  void set_ghosts(const List &l);
  void cell_rows(double cell[9]) const;

 private:
  void allocate();
  void set(int n, bool half);
};

}    // namespace LAMMPS_NS

#endif
#endif
