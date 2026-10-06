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

#include "pet/exchange.hpp"

#include <Kokkos_Core.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace pet {
class Calculator;
}

namespace LAMMPS_NS {

// PET machine-learning potentials through pet-kokkos, on LAMMPS's own neighbour
// list: handed over when LAMMPS rebuilds it, positions every step. Three ways to
// give it the atoms (see pair_pet.cpp): periodic images of the owned atoms,
// ghosts one cutoff deep with rows exchanged between ranks, or a deep ghost
// shell. pet/kk (KOKKOS package) keeps every step on the device.
class PairPET : public Pair {
 public:
  PairPET(class LAMMPS *);
  ~PairPET() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // Mode exchange: pet::Exchange's moves over LAMMPS's comm, and its hooks.
  virtual void atoms_forward(pet::RView1D a);
  virtual void atoms_reverse(pet::RView1D a);
  void set_live(pet::IView1D live);
  void edges(pet::View2D out, pet::View2D in);
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  enum class Mode { Images, Ghosts, Exchange };
  Mode mode = Mode::Images;
  std::unique_ptr<pet::Calculator> calc;
  double cutoff = 0.0;                   // the model's
  std::vector<int> type_z;               // atomic number of each LAMMPS type
  std::vector<int> off, nbr, z, shift;   // the list at the last rebuild, pet-kokkos's form
  bool listed = false;
  bool full_list = false;                // images mode on a full list (pet/kk)
  // Whether MPI takes device pointers, so rows cross without host copies:
  // keyword gpu_aware yes|no, else asked of MPI (or, for pet/kk, LAMMPS's
  // -pk kokkos gpu/aware).
  enum class Aware { Auto, Yes, No } aware = Aware::Auto;
  bool gpu_aware = false;
  virtual bool mpi_gpu_aware() const;

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
  void set_exchange(const List &l);
  void cell_rows(double cell[9]) const;

  // Mode exchange. Atom values ride LAMMPS's forward and reverse comm (which
  // relay through ghosts of ghosts); edge rows go straight to the rank owning
  // the ghost, grouped by rank (MPI_Alltoallv). Once per rebuild, every edge to
  // a ghost is matched to that rank's partner edge: recv_map[q] is the edge the
  // q-th row arriving is for. Once per evaluation (set_live), the edges PET
  // keeps: only their rows cross, and where each lands is worked out then, so
  // each layer's swap is one pack, one MPI_Alltoallv and one unpack.
  int n_remote = 0;                                // edges to ghosts
  std::vector<int> send_order, recv_map;           // remote edges by destination; arrival -> edge
  std::vector<int> send_counts, send_displs, recv_counts, recv_displs;  // per rank, in rows
  pet::IView1D d_send_order, d_recv_map;            // the same, on the device
  pet::IView1D d_send_block, d_send_displs;        // each send position's block start; per rank
  pet::IView1D d_recv_displs;                      // where each rank's rows start, arriving
  std::vector<int> live_sc, live_sd, live_rc, live_rd;  // this evaluation's live rows, per rank
  int n_live_send = 0, n_live_arrive = 0;
  pet::IView1D d_row, d_live_at, d_live_counts;    // each remote edge's live row; compaction
  pet::IView1D d_send_row, d_send_tag, d_recv_tag; // live rows in send order, and their partners
  pet::IView1D d_arrive, d_arrive_row;             // per rank arrival offsets; each arrival's row
  pet::View2D d_send, d_recv;                      // rows in send / arrival order
  Kokkos::View<char *, Kokkos::SharedHostPinnedSpace> h_send, h_recv;  // staging without GPU-aware MPI
  std::vector<double> atom_buf;                    // one value per atom, for the comm hooks
  struct Link : pet::Exchange {                    // what pet-kokkos calls
    PairPET *p;
    explicit Link(PairPET *pair) : p(pair) {}
    void atoms_forward(pet::RView1D a) override { p->atoms_forward(a); }
    void atoms_reverse(pet::RView1D a) override { p->atoms_reverse(a); }
    void set_live(pet::IView1D live) override { p->set_live(live); }
    void edges(pet::View2D out, pet::View2D in) override { p->edges(out, in); }
  } link{this};
  void alltoallv_device(const void *out, void *in, int width, const std::vector<int> &sc,
                        const std::vector<int> &sd, const std::vector<int> &rc, const std::vector<int> &rd);
  void exchange_views();  // what set_live works in, sized for n_remote
  void alltoall_rows(const char *out, int width, char *in);
  void forward_atoms();

 private:
  void allocate();
  void set(int n, bool half);
};

}    // namespace LAMMPS_NS

#endif
#endif
