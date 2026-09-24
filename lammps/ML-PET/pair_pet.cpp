/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

// pair_style pet MODEL [mode images|ghosts]
// pair_coeff * * ELEMENT...        (one element symbol per atom type)
//
// MODEL is a pet-kokkos model name (searched on PET_MODEL_DIR and friends) or a
// path prefix to MODEL.json and MODEL.safetensors. Units metal, newton pair on.
//
// PET passes messages over its GNN layers, so an owned atom's energy depends on
// atoms several cutoffs away. Two ways to give it that:
//
//   images (the default on one MPI rank): every ghost is a periodic image of an
//     owned atom, so pet-kokkos sees only the owned atoms, with each edge to an
//     image labelled by its lattice shift. Exact, and costs the owned atoms only.
//   ghosts (the default on more): the ghost cutoff is raised to what the model
//     needs (pet::Calculator::ghost_cutoff), ghosts get neighbour lists too, and
//     pet-kokkos evaluates them all and counts only the owned atoms' energy;
//     ghost forces come home by reverse communication. It costs the ghost shell
//     too -- several cutoffs deep -- so it only pays on large subdomains.

#include "pair_pet.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include "pet/calculator.hpp"
#include "pet/io.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

using namespace LAMMPS_NS;

namespace {

// Kokkos belongs to the application; without LAMMPS's KOKKOS package there is
// none, so the first pet pair style brings it up (one GPU per node-local rank)
// and it goes down at exit, after every Calculator is gone.
void ensure_kokkos()
{
  if (Kokkos::is_initialized()) return;
  Kokkos::initialize(Kokkos::InitializationSettings().set_map_device_id_by("mpi_rank"));
  std::atexit([] {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  });
}

}    // namespace

/* ---------------------------------------------------------------------- */

PairPET::PairPET(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  ghostneigh = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairPET::~PairPET()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

/* ---------------------------------------------------------------------- */

void PairPET::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // The list goes to pet-kokkos when LAMMPS rebuilds it; every other step moves
  // positions only, and pet-kokkos replays the step it recorded.
  if (neighbor->ago == 0 || !listed) {
    List l;
    l.n = list->inum + (mode == Mode::Ghosts ? list->gnum : 0);
    l.ilist = list->ilist, l.numneigh = list->numneigh;
    l.neighbor = [this](int i, int k) { return list->firstneigh[i][k] & NEIGHMASK; };
    if (mode == Mode::Images) set_images(l, true);
    else set_ghosts(l);
    listed = true;
  }
  const int n = mode == Mode::Images ? atom->nlocal : atom->nlocal + atom->nghost;
  const int nlocal = atom->nlocal;
  double **x = atom->x, **f = atom->f;
  double cell[9];
  cell_rows(cell);
  const bool forces = !energy_only(0);
  pet::Results r;
  try {
    r = calc->compute_step(n ? &x[0][0] : nullptr, cell, forces, forces && vflag_atom);
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style pet: {}", e.what());
  }

  if (eflag_global) eng_vdwl += r.energy[0];
  if (eflag_atom)
    for (int i = 0; i < nlocal; ++i) eatom[i] += r.per_atom_energy[i];
  if (!forces) return;
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) f[i][c] += r.forces[3 * i + c];
  // LAMMPS's virial is sum (x_i - x_j) (x) F_i, minus pet-kokkos's sum v (x) dE/dv.
  // Not f . r: PET is not exactly rotation invariant, so sum x (x) F has an
  // antisymmetric part, and f . r keeps one off-diagonal of it.
  if (vflag_global)
    for (int k = 0; k < 6; ++k) virial[k] -= r.virial[k];

  // Per-atom: each pair's -v (x) g, symmetrised like the global one, half to each end.
  if (vflag_atom)
    for (int i = 0; i < n; ++i)
      for (int e = off[i]; e < off[i + 1]; ++e) {
        const int j = nbr[e];
        const int *s = mode == Mode::Images ? &shift[3 * e] : nullptr;
        double d[3];
        for (int c = 0; c < 3; ++c)
          d[c] = x[j][c] - x[i][c] + (s ? s[0] * cell[c] + s[1] * cell[3 + c] + s[2] * cell[6 + c] : 0.0);
        const double *g = &r.edge_gradient[3 * e];
        const double w[6] = {-d[0] * g[0], -d[1] * g[1], -d[2] * g[2], -0.5 * (d[0] * g[1] + d[1] * g[0]),
                             -0.5 * (d[0] * g[2] + d[2] * g[0]), -0.5 * (d[1] * g[2] + d[2] * g[1])};
        for (int k = 0; k < 6; ++k) vatom[i][k] += 0.5 * w[k], vatom[j][k] += 0.5 * w[k];
      }
}

// LAMMPS's h is xx yy zz yz xz xy; pet-kokkos wants the lattice vectors as rows.
void PairPET::cell_rows(double cell[9]) const
{
  const double *h = domain->h;
  const double c[9] = {h[0], 0.0, 0.0, h[5], h[1], 0.0, h[4], h[3], h[2]};
  std::copy(c, c + 9, cell);
}

// The owned atoms only: a neighbour that is a ghost becomes the owned atom it is
// an image of, found by its tag, with the lattice shift that tells images apart.
void PairPET::set_images(const List &l, bool half)
{
  const int nlocal = atom->nlocal;
  double **x = atom->x;
  const tagint *tag = atom->tag;
  const double *hi = domain->h_inv;
  std::unordered_map<tagint, int> owner;
  owner.reserve(nlocal);
  for (int i = 0; i < nlocal; ++i) owner[tag[i]] = i;

  off.assign(nlocal + 1, 0);
  for (int ii = 0; ii < l.n; ++ii) off[l.ilist[ii] + 1] = l.numneigh[l.ilist[ii]];
  for (int i = 0; i < nlocal; ++i) off[i + 1] += off[i];
  nbr.resize(off[nlocal]), shift.resize(3 * off[nlocal]);
  for (int ii = 0; ii < l.n; ++ii) {
    const int i = l.ilist[ii];
    for (int k = 0; k < l.numneigh[i]; ++k) {
      const int j = l.neighbor(i, k), e = off[i] + k;
      const auto it = owner.find(tag[j]);
      if (it == owner.end())
        error->one(FLERR, "pair_style pet mode images: atom {} is not an image of an owned atom", tag[j]);
      const int o = it->second;
      const double s[3] = {x[j][0] - x[o][0], x[j][1] - x[o][1], x[j][2] - x[o][2]};
      const double f[3] = {hi[0] * s[0] + hi[5] * s[1] + hi[4] * s[2], hi[1] * s[1] + hi[3] * s[2], hi[2] * s[2]};
      nbr[e] = o;
      for (int c = 0; c < 3; ++c) shift[3 * e + c] = (int) std::lround(f[c]);
    }
  }
  set(nlocal, half);
}

// Owned atoms and ghosts as they sit, from the full list with ghosts' neighbours.
void PairPET::set_ghosts(const List &l)
{
  const int nall = atom->nlocal + atom->nghost;
  off.assign(nall + 1, 0);
  for (int ii = 0; ii < l.n; ++ii) off[l.ilist[ii] + 1] = l.numneigh[l.ilist[ii]];
  for (int i = 0; i < nall; ++i) off[i + 1] += off[i];
  nbr.resize(off[nall]);
  shift.clear();
  for (int ii = 0; ii < l.n; ++ii) {
    const int i = l.ilist[ii];
    for (int k = 0; k < l.numneigh[i]; ++k) nbr[off[i] + k] = l.neighbor(i, k);
  }
  set(nall, false);
}

void PairPET::set(int n, bool half)
{
  z.resize(n);
  for (int i = 0; i < n; ++i) z[i] = type_z[atom->type[i]];
  pet::EdgeListView v;
  v.n_atoms = n, v.n_local = atom->nlocal, v.half = half;
  v.atomic_numbers = z.data(), v.offsets = off.data(), v.neighbors = nbr.data();
  v.shifts = shift.empty() ? nullptr : shift.data();
  try {
    calc->set_neighbors(v);
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style pet: {}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

void PairPET::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cutghost, n + 1, n + 1, "pair:cutghost");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairPET::settings(int narg, char **arg)
{
  if (narg != 1 && narg != 3) error->all(FLERR, "Illegal pair_style pet command: pair_style pet MODEL [mode images|ghosts]");
  mode = comm->nprocs == 1 ? Mode::Images : Mode::Ghosts;
  if (narg == 3) {
    if (strcmp(arg[1], "mode") != 0) error->all(FLERR, "Illegal pair_style pet keyword {}", arg[1]);
    if (strcmp(arg[2], "images") == 0) mode = Mode::Images;
    else if (strcmp(arg[2], "ghosts") == 0) mode = Mode::Ghosts;
    else error->all(FLERR, "Illegal pair_style pet mode {}: images or ghosts", arg[2]);
  }
  if (mode == Mode::Images && comm->nprocs > 1)
    error->all(FLERR, "pair_style pet mode images needs one MPI rank: other ranks' atoms are not images");
  ghostneigh = mode == Mode::Ghosts;
  no_virial_fdotr_compute = 1;  // pet-kokkos's symmetric virial instead, see compute()
  ensure_kokkos();

  // Ranks on one node share its GPUs, and each must budget for its share.
  MPI_Comm node;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int node_ranks = 1;
  MPI_Comm_size(node, &node_ranks);
  MPI_Comm_free(&node);
  pet::Options opts;
  const int devices = std::max(1, Kokkos::num_devices());
  opts.device_share = (node_ranks + devices - 1) / devices;
  try {
    calc = std::make_unique<pet::Calculator>(arg[0], opts);
  } catch (std::exception &e) {
    error->all(FLERR, "pair_style pet: {}", e.what());
  }
  cutoff = calc->cutoff();
}

void PairPET::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int n = atom->ntypes;
  if (narg != 2 + n)
    error->all(FLERR, "Incorrect args for pair coefficients: pair_coeff * * followed by {} elements", n);
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients: pair_style pet takes pair_coeff * *");
  type_z.assign(n + 1, 0);
  for (int t = 1; t <= n; ++t) {
    const int zt = pet::element_number(arg[1 + t]);
    if (!calc->supports(zt)) error->all(FLERR, "pair_style pet: the model does not support element {}", arg[1 + t]);
    type_z[t] = zt;
  }
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 1;
}

void PairPET::init_style()
{
  if (strcmp(update->unit_style, "metal") != 0) error->all(FLERR, "Pair style pet requires metal units");
  // Ghosts get forces to send home, and a half list needs newton's pair split;
  // images mode on a full list needs neither.
  if (force->newton_pair == 0 && !(mode == Mode::Images && full_list))
    error->all(FLERR, "Pair style pet requires newton pair on{}", full_list ? " in mode ghosts" : "");

  listed = false;
  if (mode == Mode::Images) {
    if (full_list) neighbor->add_request(this, NeighConst::REQ_FULL);
    else neighbor->add_request(this);  // half: pet-kokkos mirrors it
    return;
  }
  // Neighbours of ghosts too: their features feed the owned atoms' through the
  // message-passing layers.
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  const double ghost = calc->ghost_cutoff() + neighbor->skin;
  if (comm->cutghostuser < ghost) {
    comm->cutghostuser = ghost;
    if (comm->me == 0) utils::logmesg(lmp, "pair_style pet: ghost cutoff raised to {:.4g} A\n", ghost);
  }
}

double PairPET::init_one(int i, int j)
{
  cutghost[i][j] = cutghost[j][i] = cutoff;
  return cutoff;
}
