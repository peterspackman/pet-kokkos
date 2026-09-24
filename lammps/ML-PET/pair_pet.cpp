/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

// pair_style pet MODEL [mode images|exchange|ghosts] [gpu_aware auto|yes|no]
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
//   exchange (the default on more): each rank evaluates its owned atoms, ghosts
//     one cutoff deep, and at every layer the ranks swap the rows of edges that
//     cross between them (pet/exchange.hpp): ghosts' adaptive cutoffs and their
//     adjoints by LAMMPS's comm, edge rows straight to the owning rank. Ghost
//     forces come home by reverse communication.
//   ghosts: no exchange; the ghost cutoff is raised to what the message passing
//     needs (pet::Calculator::ghost_cutoff) and pet-kokkos evaluates that whole
//     shell. Kept for comparison: it costs several cutoffs of ghosts.

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

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>  // Open MPI's MPIX_Query_cuda_support
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
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
    else if (mode == Mode::Exchange) set_exchange(l);
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

// Owned atoms with their lists, ghosts one cutoff deep with none; and the
// routing of the edges to ghosts: to the rank owning each ghost, matched there
// by (that atom's tag, this atom's tag, the edge vector) to its own edges.
void PairPET::set_exchange(const List &l)
{
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost, nprocs = comm->nprocs;
  double **x = atom->x;
  const tagint *tag = atom->tag;
  off.assign(nall + 1, 0);
  for (int ii = 0; ii < l.n; ++ii) off[l.ilist[ii] + 1] = l.numneigh[l.ilist[ii]];
  for (int i = 0; i < nall; ++i) off[i + 1] += off[i];
  nbr.resize(off[nall]);
  shift.clear();
  for (int ii = 0; ii < l.n; ++ii) {
    const int i = l.ilist[ii];
    for (int k = 0; k < l.numneigh[i]; ++k) nbr[off[i] + k] = l.neighbor(i, k);
  }

  atom_buf.assign(nall, comm->me);  // each ghost's owner
  forward_atoms();
  std::vector<int> remote_of(off[nall], -1);
  std::vector<std::vector<int>> to_rank(nprocs);
  std::vector<double> key;
  int n_remote = 0;
  for (int i = 0; i < nlocal; ++i)
    for (int e = off[i]; e < off[i + 1]; ++e) {
      const int j = nbr[e];
      if (j < nlocal) continue;
      to_rank[(int) atom_buf[j]].push_back(remote_of[e] = n_remote++);
      key.insert(key.end(), {double(tag[j]), double(tag[i]), x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]});
    }
  send_order.clear(), send_counts.assign(nprocs, 0), recv_counts.assign(nprocs, 0);
  for (int p = 0; p < nprocs; ++p) {
    send_counts[p] = to_rank[p].size();
    send_order.insert(send_order.end(), to_rank[p].begin(), to_rank[p].end());
  }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, world);
  send_displs.assign(nprocs, 0), recv_displs.assign(nprocs, 0);
  for (int p = 1; p < nprocs; ++p)
    send_displs[p] = send_displs[p - 1] + send_counts[p - 1], recv_displs[p] = recv_displs[p - 1] + recv_counts[p - 1];
  const int n_recv = recv_displs[nprocs - 1] + recv_counts[nprocs - 1];

  std::vector<double> keys_out(5 * n_remote), keys_in(5 * n_recv);
  for (int q = 0; q < n_remote; ++q) std::copy_n(&key[5 * send_order[q]], 5, &keys_out[5 * q]);
  alltoall_rows((const char *) keys_out.data(), 5 * sizeof(double), (char *) keys_in.data());
  std::unordered_map<tagint, int> local;
  for (int i = 0; i < nlocal; ++i) local[tag[i]] = i;
  recv_map.assign(n_recv, -1);
  int unmatched = 0;
  for (int q = 0; q < n_recv; ++q) {
    const double *k = &keys_in[5 * q];
    const auto it = local.find((tagint) k[0]);
    if (it != local.end()) {
      const int o = it->second;
      for (int e = off[o]; e < off[o + 1] && recv_map[q] < 0; ++e) {
        const int h = nbr[e];
        if (h >= nlocal && tag[h] == (tagint) k[1] &&
            std::fabs(x[h][0] - x[o][0] + k[2]) + std::fabs(x[h][1] - x[o][1] + k[3]) + std::fabs(x[h][2] - x[o][2] + k[4]) < 1e-6)
          recv_map[q] = remote_of[e];
      }
    }
    unmatched += recv_map[q] < 0;
  }
  int all = 0;
  MPI_Allreduce(&unmatched, &all, 1, MPI_INT, MPI_SUM, world);
  if (all && comm->me == 0)
    error->warning(FLERR, "pair_style pet mode exchange: {} edge rows found no partner edge", all);
  using HostI = Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  d_send_order = pet::IView1D("pet:send_order", n_remote), d_recv_map = pet::IView1D("pet:recv_map", n_recv);
  Kokkos::deep_copy(d_send_order, HostI(send_order.data(), n_remote));
  Kokkos::deep_copy(d_recv_map, HostI(recv_map.data(), n_recv));
  std::vector<int> block(n_remote), rank(n_remote);
  for (int p = 0; p < nprocs; ++p)
    for (int q = send_displs[p]; q < send_displs[p] + send_counts[p]; ++q) block[q] = send_displs[p], rank[q] = p;
  d_send_block = pet::IView1D("pet:send_block", n_remote), d_send_rank = pet::IView1D("pet:send_rank", n_remote);
  d_recv_displs = pet::IView1D("pet:recv_displs", nprocs);
  Kokkos::deep_copy(d_send_block, HostI(block.data(), n_remote));
  Kokkos::deep_copy(d_send_rank, HostI(rank.data(), n_remote));
  Kokkos::deep_copy(d_recv_displs, HostI(recv_displs.data(), nprocs));
  d_live_at = pet::IView1D("pet:live_at", n_remote + 1), d_live_counts = pet::IView1D("pet:live_counts", nprocs);
  d_arrive = pet::IView1D("pet:arrive", nprocs + 1);
  set(nall, false);
}

// Rows of `width` bytes to each rank, in send_order; what arrives, in rank order.
void PairPET::alltoall_rows(const char *out, int width, char *in)
{
  const int nprocs = comm->nprocs;
  std::vector<int> sc(nprocs), sd(nprocs), rc(nprocs), rd(nprocs);
  for (int p = 0; p < nprocs; ++p)
    sc[p] = send_counts[p] * width, sd[p] = send_displs[p] * width, rc[p] = recv_counts[p] * width,
    rd[p] = recv_displs[p] * width;
  MPI_Alltoallv(out, sc.data(), sd.data(), MPI_BYTE, in, rc.data(), rd.data(), MPI_BYTE, world);
}

// Live rows only (edges PET keeps this step), compacted in send order on the
// device, each tagged in its extra last column with its place among the rows
// its destination expects from this rank; across by MPI from pinned host
// buffers; placed by tag on the device.
void PairPET::edges(pet::View2D out, pet::View2D in, pet::IView1D live)
{
  using Range = Kokkos::RangePolicy<pet::ExecSpace>;
  const int D = out.extent(1), W = D + 1, nprocs = comm->nprocs, n_send = send_order.size();
  auto order = d_send_order, block = d_send_block, rank = d_send_rank, at = d_live_at, counts = d_live_counts;
  Kokkos::parallel_scan(
      "pet_live_scan", Range(0, n_send + 1), KOKKOS_LAMBDA(int q, int &c, bool final) {
        if (final) at(q) = c;
        if (q < n_send) c += live(order(q)) >= 0;
      });
  Kokkos::deep_copy(counts, 0);
  Kokkos::parallel_for(
      "pet_live_counts", Range(0, n_send), KOKKOS_LAMBDA(int q) {
        if (live(order(q)) >= 0) Kokkos::atomic_inc(&counts(rank(q)));
      });
  std::vector<int> sc(nprocs), rc(nprocs), sd(nprocs, 0), rd(nprocs + 1, 0);
  Kokkos::deep_copy(Kokkos::View<int *, Kokkos::HostSpace>(sc.data(), nprocs), counts);
  MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, world);
  for (int p = 1; p < nprocs; ++p) sd[p] = sd[p - 1] + sc[p - 1];
  for (int p = 0; p < nprocs; ++p) rd[p + 1] = rd[p] + rc[p];
  const int n_live = sd[nprocs - 1] + sc[nprocs - 1], n_arrive = rd[nprocs];

  if (d_send.extent(0) < (size_t) n_live || d_send.extent(1) != (size_t) W)
    d_send = pet::View2D("pet:send", n_send, W);
  if (d_recv.extent(0) < (size_t) n_arrive || d_recv.extent(1) != (size_t) W)
    d_recv = pet::View2D("pet:recv", recv_map.size(), W);
  if (!gpu_aware && (h_send.extent(0) < d_send.extent(0) || h_send.extent(1) != (size_t) W))
    h_send = decltype(h_send)("pet:h_send", d_send.extent(0), W);
  if (!gpu_aware && (h_recv.extent(0) < d_recv.extent(0) || h_recv.extent(1) != (size_t) W))
    h_recv = decltype(h_recv)("pet:h_recv", d_recv.extent(0), W);
  auto send = d_send, recv = d_recv;
  Kokkos::parallel_for(
      "pet_send_rows", Range(0, n_send), KOKKOS_LAMBDA(int q) {
        const int r = order(q);
        if (live(r) < 0) return;
        const int c = at(q), tag = q - block(q);
        for (int d = 0; d < D; ++d) send(c, d) = out(r, d);
        pet::Net t = 0;
        memcpy(&t, &tag, sizeof(int));
        send(c, D) = t;
      });
  const auto rows = [](auto v, int n) { return Kokkos::subview(v, std::make_pair(0, n), Kokkos::ALL); };
  const int bytes = W * sizeof(pet::Net);
  std::vector<int> scb(nprocs), sdb(nprocs), rcb(nprocs), rdb(nprocs);
  for (int p = 0; p < nprocs; ++p) scb[p] = sc[p] * bytes, sdb[p] = sd[p] * bytes, rcb[p] = rc[p] * bytes, rdb[p] = rd[p] * bytes;
  if (gpu_aware) {  // device to device; MPI does not follow the stream, so fence first
    Kokkos::fence();
    MPI_Alltoallv(d_send.data(), scb.data(), sdb.data(), MPI_BYTE, d_recv.data(), rcb.data(), rdb.data(), MPI_BYTE, world);
  } else {
    Kokkos::deep_copy(rows(h_send, n_live), rows(d_send, n_live));
    MPI_Alltoallv(h_send.data(), scb.data(), sdb.data(), MPI_BYTE, h_recv.data(), rcb.data(), rdb.data(), MPI_BYTE, world);
    Kokkos::deep_copy(rows(d_recv, n_arrive), rows(h_recv, n_arrive));
  }

  Kokkos::deep_copy(Kokkos::subview(d_arrive, std::make_pair(0, nprocs + 1)),
                    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(rd.data(), nprocs + 1));
  Kokkos::deep_copy(in, pet::Net(0));
  auto map = d_recv_map, displs = d_recv_displs, arrive = d_arrive;
  Kokkos::parallel_for(
      "pet_recv_rows", Range(0, n_arrive), KOKKOS_LAMBDA(int a) {
        int p = 0;
        while (arrive(p + 1) <= a) ++p;  // the sending rank
        int tag = 0;
        const pet::Net t = recv(a, D);
        memcpy(&tag, &t, sizeof(int));
        const int r = map(displs(p) + tag);
        if (r >= 0)
          for (int d = 0; d < D; ++d) in(r, d) = recv(a, d);
      });
}

void PairPET::atoms_forward(pet::RView1D a)
{
  const int nall = atom->nlocal + atom->nghost;
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  atom_buf.assign(h.data(), h.data() + nall);
  forward_atoms();
  std::copy_n(atom_buf.data(), nall, h.data());
  Kokkos::deep_copy(a, h);
}

// atom_buf through LAMMPS's comm, by the host hooks below -- also under the
// KOKKOS package, whose comm would otherwise want device ones.
void PairPET::forward_atoms()
{
  const ExecutionSpace space = execution_space;
  execution_space = Host;
  comm->forward_comm(this);
  execution_space = space;
}

void PairPET::atoms_reverse(pet::RView1D a)
{
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost;
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  atom_buf.assign(h.data(), h.data() + nall);
  comm->reverse_comm(this);
  for (int i = 0; i < nall; ++i) h(i) = i < nlocal ? atom_buf[i] : 0.0;
  Kokkos::deep_copy(a, h);
}

int PairPET::pack_forward_comm(int n, int *list, double *buf, int, int *)
{
  for (int k = 0; k < n; ++k) buf[k] = atom_buf[list[k]];
  return n;
}

void PairPET::unpack_forward_comm(int n, int first, double *buf)
{
  for (int k = 0; k < n; ++k) atom_buf[first + k] = buf[k];
}

int PairPET::pack_reverse_comm(int n, int first, double *buf)
{
  for (int k = 0; k < n; ++k) buf[k] = atom_buf[first + k];
  return n;
}

void PairPET::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int k = 0; k < n; ++k) atom_buf[list[k]] += buf[k];
}

void PairPET::set(int n, bool half)
{
  z.resize(n);
  for (int i = 0; i < n; ++i) z[i] = type_z[atom->type[i]];
  pet::EdgeListView v;
  v.n_atoms = n, v.n_local = atom->nlocal, v.half = half;
  v.exchange = mode == Mode::Exchange ? &link : nullptr;
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
  if (narg < 1 || narg % 2 == 0)
    error->all(FLERR, "Illegal pair_style pet command: pair_style pet MODEL [mode images|exchange|ghosts] "
                      "[gpu_aware auto|yes|no]");
  mode = comm->nprocs == 1 ? Mode::Images : Mode::Exchange;
  for (int k = 1; k < narg; k += 2) {
    const std::string key = arg[k], val = arg[k + 1];
    if (key == "mode" && val == "images") mode = Mode::Images;
    else if (key == "mode" && val == "exchange") mode = Mode::Exchange;
    else if (key == "mode" && val == "ghosts") mode = Mode::Ghosts;
    else if (key == "gpu_aware" && val == "auto") aware = Aware::Auto;
    else if (key == "gpu_aware" && val == "yes") aware = Aware::Yes;
    else if (key == "gpu_aware" && val == "no") aware = Aware::No;
    else error->all(FLERR, "Illegal pair_style pet keyword: {} {}", key, val);
  }
  if (mode == Mode::Images && comm->nprocs > 1)
    error->all(FLERR, "pair_style pet mode images needs one MPI rank: other ranks' atoms are not images");
  ghostneigh = mode == Mode::Ghosts;
  comm_forward = comm_reverse = 1;  // mode exchange: one value per atom
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

// Asked of MPI where it can say; an MPI that cannot is taken not to.
bool PairPET::mpi_gpu_aware() const
{
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  if (MPIX_Query_cuda_support()) return true;
#endif
#if defined(MPIX_ROCM_AWARE_SUPPORT) && MPIX_ROCM_AWARE_SUPPORT
  if (MPIX_Query_rocm_support()) return true;
#endif
  return false;
}

void PairPET::init_style()
{
  gpu_aware = aware == Aware::Yes || (aware == Aware::Auto && mpi_gpu_aware());
  if (mode == Mode::Exchange && comm->nprocs > 1 && comm->me == 0)
    utils::logmesg(lmp, "pair_style pet: edge rows cross {}\n", gpu_aware ? "from device memory (GPU-aware MPI)"
                                                                           : "through pinned host memory");
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
  if (mode == Mode::Exchange) {  // owned atoms' lists; ghosts one cutoff deep
    neighbor->add_request(this, NeighConst::REQ_FULL);
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
