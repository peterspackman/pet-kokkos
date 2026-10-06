/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Peter Spackman
------------------------------------------------------------------------- */

// pair_style pet MODEL [mode images|exchange|ghosts] [gpu_aware auto|yes|no]
// pair_coeff * * ELEMENT...        (one element symbol per atom type)
//
// MODEL is a pet-kokkos model name (searched on PET_MODEL_DIR and friends) or a
// path prefix to MODEL.json and MODEL.safetensors. Units metal, newton pair on.
//
// PET passes messages over its GNN layers, so an owned atom's energy depends on
// atoms several cutoffs away. Three ways to give it that:
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
  n_remote = 0;
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
  set_peers();

  std::vector<double> keys_out(5 * n_remote), keys_in(5 * n_recv);
  for (int q = 0; q < n_remote; ++q) std::copy_n(&key[5 * send_order[q]], 5, &keys_out[5 * q]);
  alltoall_rows((const char *) keys_out.data(), 5 * sizeof(double), (char *) keys_in.data());
  // Each owned atom's edges to ghosts, sorted by the ghost's tag; a partner is
  // found by its owned atom, then by binary search. A small cell can hold several
  // images of one ghost: told apart by the edge vector.
  std::unordered_map<tagint, int> local;
  for (int i = 0; i < nlocal; ++i) local[tag[i]] = i;
  std::vector<int> by_tag, by_off(nlocal + 1, 0);
  by_tag.reserve(n_remote);
  for (int i = 0; i < nlocal; ++i) {
    by_off[i] = by_tag.size();
    for (int e = off[i]; e < off[i + 1]; ++e)
      if (nbr[e] >= nlocal) by_tag.push_back(e);
    std::sort(by_tag.begin() + by_off[i], by_tag.end(), [&](int a, int b) { return tag[nbr[a]] < tag[nbr[b]]; });
  }
  by_off[nlocal] = by_tag.size();
  recv_map.assign(n_recv, -1);
  int unmatched = 0;
  for (int q = 0; q < n_recv; ++q) {
    const double *k = &keys_in[5 * q];
    const auto it = local.find((tagint) k[0]);
    const int o = it == local.end() ? -1 : it->second;
    const tagint th = (tagint) k[1];
    auto e = o < 0 ? by_tag.end() : std::lower_bound(by_tag.begin() + by_off[o], by_tag.begin() + by_off[o + 1], th,
                                                     [&](int a, tagint t) { return tag[nbr[a]] < t; });
    for (; o >= 0 && e != by_tag.begin() + by_off[o + 1] && tag[nbr[*e]] == th && recv_map[q] < 0; ++e) {
      const int h = nbr[*e];
      if (std::fabs(x[h][0] - x[o][0] + k[2]) + std::fabs(x[h][1] - x[o][1] + k[3]) + std::fabs(x[h][2] - x[o][2] + k[4]) < 1e-6)
        recv_map[q] = remote_of[*e];
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
  std::vector<int> block(n_remote), sdisp(send_displs);
  sdisp.push_back(n_remote);
  for (int p = 0; p < nprocs; ++p)
    for (int q = send_displs[p]; q < send_displs[p] + send_counts[p]; ++q) block[q] = send_displs[p];
  d_send_block = pet::IView1D("pet:send_block", n_remote), d_send_displs = pet::IView1D("pet:send_displs", nprocs + 1);
  d_recv_displs = pet::IView1D("pet:recv_displs", nprocs);
  Kokkos::deep_copy(d_send_block, HostI(block.data(), n_remote));
  Kokkos::deep_copy(d_send_displs, HostI(sdisp.data(), nprocs + 1));
  Kokkos::deep_copy(d_recv_displs, HostI(recv_displs.data(), nprocs));
  exchange_views();
  set(nall, false);
}

void PairPET::exchange_views()
{
  const int nprocs = comm->nprocs;
  d_row = pet::IView1D("pet:row", n_remote), d_live_at = pet::IView1D("pet:live_at", n_remote + 1);
  d_live_counts = pet::IView1D("pet:live_counts", nprocs), d_arrive = pet::IView1D("pet:arrive", nprocs + 1);
  h_arrive = decltype(h_arrive)("pet:h_arrive", nprocs + 1);
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

void PairPET::set_peers()
{
  peers.clear();
  for (int p = 0; p < comm->nprocs; ++p)
    if (p != comm->me && (send_counts[p] > 0 || recv_counts[p] > 0)) peers.push_back(p);
}

// Rows `width` bytes wide between device buffers, point to point with the peers
// (a message each way with every peer, empty or not, so that sends and receives
// always pair up): straight from device memory when MPI is GPU-aware, else
// through pinned host buffers. Rows this rank sends itself (ghosts that are
// images of its own atoms) are copied on the device. Counts and displacements
// per rank, in rows; with `got`, rc is an upper bound and got is what came.
void PairPET::exchange_rows(const void *out, void *in, int width, const std::vector<int> &sc,
                            const std::vector<int> &sd, const std::vector<int> &rc, const std::vector<int> &rd,
                            std::vector<int> *got)
{
  using Bytes = Kokkos::View<char *, pet::MemSpace, Kokkos::MemoryUnmanaged>;
  const int me = comm->me, tag = 0x9e7;
  char *o = (char *) out, *i = (char *) in;
  const auto bytes = [width](int rows) { return std::size_t(rows) * width; };
  if (got) got->assign(comm->nprocs, 0);
  const pet::ExecSpace exec;
  if (sc[me] > 0)  // in stream order, like everything around it
    Kokkos::deep_copy(exec, Bytes(i + bytes(rd[me]), bytes(sc[me])), Bytes(o + bytes(sd[me]), bytes(sc[me])));
  if (got) (*got)[me] = sc[me];
  if (peers.empty()) return;
  exec.fence("pet: rows ready for MPI");  // MPI does not follow the stream

  std::size_t n_out = 0, n_in = 0;
  for (int p : peers) n_out = std::max(n_out, bytes(sd[p] + sc[p])), n_in = std::max(n_in, bytes(rd[p] + rc[p]));
  char *so = o, *si = i;
  if (!gpu_aware) {
    if (h_send.extent(0) < n_out) h_send = decltype(h_send)("pet:h_send", n_out);
    if (h_recv.extent(0) < n_in) h_recv = decltype(h_recv)("pet:h_recv", n_in);
    Kokkos::deep_copy(exec, Kokkos::subview(h_send, std::make_pair(std::size_t(0), n_out)), Bytes(o, n_out));
    exec.fence("pet: rows staged for MPI");
    so = h_send.data(), si = h_recv.data();
  }
  const int np = peers.size();
  std::vector<MPI_Request> req(2 * np);
  for (int k = 0; k < np; ++k) {
    const int p = peers[k];
    MPI_Irecv(si + bytes(rd[p]), (int) bytes(rc[p]), MPI_BYTE, p, tag, world, &req[k]);
  }
  for (int k = 0; k < np; ++k) {
    const int p = peers[k];
    MPI_Isend(so + bytes(sd[p]), (int) bytes(sc[p]), MPI_BYTE, p, tag, world, &req[np + k]);
  }
  std::vector<MPI_Status> st(2 * np);
  MPI_Waitall(2 * np, req.data(), st.data());
  for (int k = 0; k < np; ++k) {
    const int p = peers[k];
    int n = (int) bytes(rc[p]);
    if (got) {
      MPI_Get_count(&st[k], MPI_BYTE, &n);
      (*got)[p] = n / width;
    }
    if (!gpu_aware && n > 0)
      Kokkos::deep_copy(exec, Bytes(i + bytes(rd[p]), n),
                        Kokkos::subview(h_recv, std::make_pair(bytes(rd[p]), bytes(rd[p]) + n)));
  }
  // The staged rows reach the device in stream order; h_recv is not reused
  // before the next exchange's fence.
}

// The edges PET keeps this evaluation (pet::Exchange::set_live): their rows in
// send order, how many go to each rank, and -- told by each sender which of its
// partner edges a row is for -- the row each arrival lands in. One round with
// the peers: each receives at most its rebuild count, and learns from what
// arrives how many there are. The layers' swaps then move live rows only.
void PairPET::set_live(pet::IView1D live)
{
  using Range = Kokkos::RangePolicy<pet::ExecSpace>;
  const int nprocs = comm->nprocs, n_send = n_remote, n_live = live.extent(0);
  const pet::ExecSpace exec;
  auto row = d_row, order = d_send_order, at = d_live_at, displs = d_send_displs, counts = d_live_counts;
  Kokkos::deep_copy(exec, row, -1);
  Kokkos::parallel_for(
      "pet_live_rows", Range(0, n_live), KOKKOS_LAMBDA(int l) {
        if (live(l) >= 0) row(live(l)) = l;  // -1: a capacity's padding
      });
  Kokkos::parallel_scan(
      "pet_live_scan", Range(0, n_send + 1), KOKKOS_LAMBDA(int q, int &c, bool final) {
        if (final) at(q) = c;
        if (q < n_send) c += row(order(q)) >= 0;
      });
  Kokkos::parallel_for(
      "pet_live_counts", Range(0, nprocs), KOKKOS_LAMBDA(int p) { counts(p) = at(displs(p + 1)) - at(displs(p)); });
  live_sc.resize(nprocs), live_sd.assign(nprocs, 0), live_rd.assign(nprocs + 1, 0);
  if (h_counts.extent(0) < (size_t) nprocs + 1) h_counts = decltype(h_counts)("pet:h_counts", nprocs + 1);
  Kokkos::deep_copy(exec, Kokkos::subview(h_counts, std::make_pair(0, nprocs)), counts);
  exec.fence("pet: live counts");  // the one wait set_live needs: MPI sizes come from these
  for (int p = 0; p < nprocs; ++p) live_sc[p] = h_counts(p);
  for (int p = 1; p < nprocs; ++p) live_sd[p] = live_sd[p - 1] + live_sc[p - 1];
  n_live_send = live_sd[nprocs - 1] + live_sc[nprocs - 1];

  if (d_send_row.extent(0) < (size_t) n_live_send)
    d_send_row = pet::IView1D("pet:send_row", n_live_send), d_send_tag = pet::IView1D("pet:send_tag", n_live_send);
  auto block = d_send_block, send_row = d_send_row, send_tag = d_send_tag;
  Kokkos::parallel_for(
      "pet_live_send", Range(0, n_send), KOKKOS_LAMBDA(int q) {
        const int l = row(order(q));
        if (l >= 0) send_row(at(q)) = l, send_tag(at(q)) = q - block(q);
      });
  // Each partner's tags land at the rebuild's offset for it.
  const std::size_t n_recv = d_recv_map.extent(0);
  if (d_recv_tag_all.extent(0) < std::max<std::size_t>(n_recv, 1)) d_recv_tag_all = pet::IView1D("pet:recv_tag", std::max<std::size_t>(n_recv, 1));
  exchange_rows(send_tag.data(), d_recv_tag_all.data(), sizeof(int), live_sc, live_sd, recv_counts, recv_displs, &live_rc);
  for (int p = 0; p < nprocs; ++p) live_rd[p + 1] = live_rd[p] + live_rc[p];
  n_live_arrive = live_rd[nprocs];

  if (d_arrive_row.extent(0) < (size_t) n_live_arrive) d_arrive_row = pet::IView1D("pet:arrive_row", n_live_arrive);
  for (int p = 0; p <= nprocs; ++p) h_arrive(p) = live_rd[p];
  Kokkos::deep_copy(exec, d_arrive, Kokkos::subview(h_arrive, std::make_pair(0, nprocs + 1)));
  auto map = d_recv_map, rdisp = d_recv_displs, arrive = d_arrive, tag = d_recv_tag_all, arrive_row = d_arrive_row;
  Kokkos::parallel_for(
      "pet_live_arrive", Range(0, n_live_arrive), KOKKOS_LAMBDA(int a) {
        int p = 0;
        while (arrive(p + 1) <= a) ++p;  // the sending rank
        const int r = map(rdisp(p) + tag(rdisp(p) + a - arrive(p)));
        arrive_row(a) = r >= 0 ? row(r) : -1;
      });
}

// One layer's rows of the live edges to ghosts, across: packed in send order,
// one MPI_Alltoallv, placed as set_live worked out.
void PairPET::edges(pet::View2D out, pet::View2D in)
{
  using Range = Kokkos::RangePolicy<pet::ExecSpace>;
  const int D = out.extent(1);
  if (d_send.extent(0) < (size_t) n_live_send || d_send.extent(1) != (size_t) D)
    d_send = pet::View2D("pet:send", n_live_send, D);
  if (d_recv.extent(0) < (size_t) n_live_arrive || d_recv.extent(1) != (size_t) D)
    d_recv = pet::View2D("pet:recv", n_live_arrive, D);
  auto send = d_send, recv = d_recv;
  auto send_row = d_send_row, arrive_row = d_arrive_row;
  Kokkos::parallel_for(
      "pet_send_rows", Range(0, std::size_t(n_live_send) * D),
      KOKKOS_LAMBDA(std::size_t i) { send(i / D, i % D) = out(send_row(i / D), i % D); });
  exchange_rows(send.data(), recv.data(), D * sizeof(pet::Net), live_sc, live_sd, live_rc, live_rd);
  Kokkos::deep_copy(pet::ExecSpace(), in, pet::Net(0));  // a row with no partner (warned of at the rebuild) stays zero
  Kokkos::parallel_for(
      "pet_recv_rows", Range(0, std::size_t(n_live_arrive) * D), KOKKOS_LAMBDA(std::size_t i) {
        const int r = arrive_row(i / D);
        if (r >= 0) in(r, i % D) = recv(i / D, i % D);
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
  // Mode images on a small system: every step between rebuilds the same shapes,
  // recorded and replayed as one graph. It pays where launches dominate (~2000
  // atoms or fewer on an RTX 4080); past that the headroom the shapes need costs
  // more than the launches it saves. PET_MD_FIXED=0|1 decides instead.
  opts.md_fixed_shapes = mode == Mode::Images && atom->natoms <= 2000;
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

// Asked of MPI where it can say (Open MPI, MVAPICH); Cray MPICH says so in its
// environment instead. An MPI that cannot say is taken not to be.
bool PairPET::mpi_gpu_aware() const
{
  const char *cray = getenv("MPICH_GPU_SUPPORT_ENABLED");
  if (cray && atoi(cray) == 1) return true;
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
