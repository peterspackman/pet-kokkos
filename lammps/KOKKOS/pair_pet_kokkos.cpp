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

#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cstdint>

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

// At a neighbour-list rebuild: LAMMPS's device list, handed to pet-kokkos where
// it is, with each atom's element and, in mode images, the owned atom each ghost
// is an image of (by tag); in mode exchange, routed between ranks on the device
// too (route_on_device).
template <class DeviceType> void PairPETKokkos<DeviceType>::rebuild()
{
  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  atomKK->sync(execution_space, X_MASK | TAG_MASK | TYPE_MASK);
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost, ntypes = atom->ntypes;
  using Range = Kokkos::RangePolicy<DeviceType>;
  if ((int) d_type_z.extent(0) != ntypes + 1) {
    d_type_z = decltype(d_type_z)("pet:type_z", ntypes + 1);
    Kokkos::deep_copy(d_type_z, Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(type_z.data(), ntypes + 1));
  }
  if ((int) d_z.extent(0) < nall) d_z = decltype(d_z)("pet:z", nall), d_image_of = decltype(d_image_of)("pet:image_of", nall);
  auto type = atomKK->k_type.template view<DeviceType>();
  auto tag = atomKK->k_tag.template view<DeviceType>();
  auto z = d_z, type_z_d = d_type_z;
  Kokkos::parallel_for("pet_z", Range(0, nall), KOKKOS_LAMBDA(int i) { z(i) = type_z_d(type(i)); });

  pet::DeviceEdgeListView v;
  v.n_atoms = mode == Mode::Images ? nlocal : nall, v.n_local = nlocal;
  v.n_centres = list->inum + (mode == Mode::Ghosts ? list->gnum : 0);
  v.ilist = k_list->d_ilist.data(), v.numneigh = k_list->d_numneigh.data(), v.neighbors = k_list->d_neighbors.data();
  v.stride_i = k_list->d_neighbors.stride(0), v.stride_k = k_list->d_neighbors.stride(1), v.mask = NEIGHMASK;
  v.atomic_numbers = d_z.data();
  double cell[9];
  if (mode == Mode::Images) {  // one rank: every ghost an image of an owned atom, found by its tag
    tagint max_tag = 0;
    Kokkos::parallel_reduce(
        "pet_max_tag", Range(0, nall), KOKKOS_LAMBDA(int i, tagint &m) { m = tag(i) > m ? tag(i) : m; },
        Kokkos::Max<tagint>(max_tag));
    if ((tagint) d_owned_by_tag.extent(0) <= max_tag) d_owned_by_tag = decltype(d_owned_by_tag)("pet:owned_by_tag", max_tag + 1);
    auto by_tag = d_owned_by_tag, image_of = d_image_of;
    Kokkos::deep_copy(by_tag, -1);
    Kokkos::parallel_for("pet_owned_by_tag", Range(0, nlocal), KOKKOS_LAMBDA(int i) { by_tag(tag(i)) = i; });
    Kokkos::parallel_for("pet_image_of", Range(0, nall), KOKKOS_LAMBDA(int j) { image_of(j) = by_tag(tag(j)); });
    cell_rows(cell);
    v.image_of = d_image_of.data(), v.positions = positions(), v.cell = cell;
  }
  if (mode == Mode::Exchange) {
    route_on_device(v);
    v.exchange = &link;
  }
  try {
    calc->set_neighbors(v);
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style pet/kk: {}", e.what());
  }
  listed = true;
}

// Mode exchange's routing, as PairPET::set_exchange does it on the host, here on
// the device: the edges to ghosts numbered in the list's order (atom order, then
// each atom's neighbours in turn, as pet-kokkos numbers them), grouped by the
// rank owning the ghost; each sent with a key (the ghost's tag, this atom's tag,
// the edge vector) by which the owning rank finds the partner edge among its own.
template <class DeviceType> void PairPETKokkos<DeviceType>::route_on_device(const pet::DeviceEdgeListView &v)
{
  using Range = Kokkos::RangePolicy<DeviceType>;
  using Team = Kokkos::TeamPolicy<DeviceType>;
  using U64 = Kokkos::View<uint64_t *, DeviceType>;
  using IV = Kokkos::View<int *, DeviceType>;
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost, nprocs = comm->nprocs, me = comm->me;
  const int NC = v.n_centres;
  const int *ilist = v.ilist, *numneigh = v.numneigh, *nbr = v.neighbors;
  const long si = v.stride_i, sk = v.stride_k;
  auto tag = atomKK->k_tag.template view<DeviceType>();
  const double *x = positions();

  tagint max_tag = 0;
  Kokkos::parallel_reduce(
      "pet_max_tag", Range(0, nall), KOKKOS_LAMBDA(int i, tagint &m) { m = tag(i) > m ? tag(i) : m; },
      Kokkos::Max<tagint>(max_tag));
  if (int64_t(max_tag) >= (int64_t(1) << 31)) error->one(FLERR, "pair_style pet/kk mode exchange: atom IDs past 2^31");

  // Each ghost's owner: every rank's own number, forward to its ghosts.
  pet::RView1D owner("pet:owner", nall);
  Kokkos::parallel_for("pet_me", Range(0, nlocal), KOKKOS_LAMBDA(int i) { owner(i) = me; });
  lammps_forward(owner);

  // The edges to ghosts, in pet-kokkos's numbering: each atom's range, then its
  // pairs in list order, those to ghosts counted off by a scan.
  IV count("pet:count", nall + 1), off("pet:off", nall + 1);
  Kokkos::parallel_for("pet_ex_count", Range(0, NC), KOKKOS_LAMBDA(int ii) {
    const int i = ilist[ii];
    int c = 0;
    for (int k = 0; k < numneigh[i]; ++k) c += (nbr[i * si + k * sk] & NEIGHMASK) >= nlocal;
    count(i) = c;
  });
  Kokkos::parallel_scan("pet_ex_off", Range(0, nall + 1), KOKKOS_LAMBDA(int i, int &c, bool final) {
    if (final) off(i) = c;
    if (i < nall) c += count(i);
  });
  int R = 0;
  Kokkos::deep_copy(R, Kokkos::subview(off, nall));
  n_remote = R;
  IV rem_i("pet:rem_i", R), rem_j("pet:rem_j", R);
  Kokkos::parallel_for("pet_ex_remote", Range(0, NC), KOKKOS_LAMBDA(int ii) {
    const int i = ilist[ii];
    int r = off(i);
    for (int k = 0; k < numneigh[i]; ++k) {
      const int j = nbr[i * si + k * sk] & NEIGHMASK;
      if (j >= nlocal) rem_i(r) = i, rem_j(r) = j, ++r;
    }
  });

  // Grouped by destination, in that order within each: a sort of (rank, index).
  U64 by_rank("pet:by_rank", R);
  Kokkos::parallel_for("pet_ex_dest", Range(0, R), KOKKOS_LAMBDA(int r) {
    by_rank(r) = uint64_t(int(owner(rem_j(r)))) << 32 | uint32_t(r);
  });
  Kokkos::sort(by_rank);
  d_send_order = pet::IView1D("pet:send_order", R), d_send_block = pet::IView1D("pet:send_block", R);
  d_send_displs = pet::IView1D("pet:send_displs", nprocs + 1);
  auto order = d_send_order, block = d_send_block, displs = d_send_displs;
  Kokkos::parallel_for("pet_ex_displs", Range(0, nprocs + 1), KOKKOS_LAMBDA(int p) {
    int lo = 0, hi = R;  // the first entry for rank p or beyond
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (int(by_rank(mid) >> 32) < p) lo = mid + 1;
      else hi = mid;
    }
    displs(p) = lo;
  });
  Kokkos::parallel_for("pet_ex_order", Range(0, R), KOKKOS_LAMBDA(int q) {
    order(q) = int(by_rank(q) & 0xffffffffu);
    block(q) = displs(int(by_rank(q) >> 32));
  });
  std::vector<int> sd(nprocs + 1);
  Kokkos::deep_copy(Kokkos::View<int *, Kokkos::HostSpace>(sd.data(), nprocs + 1), displs);
  send_counts.resize(nprocs), send_displs.assign(sd.begin(), sd.end() - 1), recv_counts.resize(nprocs);
  for (int p = 0; p < nprocs; ++p) send_counts[p] = sd[p + 1] - sd[p];
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, world);
  recv_displs.assign(nprocs, 0);
  for (int p = 1; p < nprocs; ++p) recv_displs[p] = recv_displs[p - 1] + recv_counts[p - 1];
  const int n_recv = recv_displs[nprocs - 1] + recv_counts[nprocs - 1];
  set_peers();
  d_recv_displs = pet::IView1D("pet:recv_displs", nprocs);
  Kokkos::deep_copy(d_recv_displs, Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(recv_displs.data(), nprocs));

  // The keys, across.
  Kokkos::View<double *, DeviceType> keys_out("pet:keys_out", 5 * std::size_t(R)), keys_in("pet:keys_in", 5 * std::size_t(n_recv));
  Kokkos::parallel_for("pet_ex_keys", Range(0, R), KOKKOS_LAMBDA(int q) {
    const int r = order(q), i = rem_i(r), j = rem_j(r);
    double *k = &keys_out(5 * std::size_t(q));
    k[0] = tag(j), k[1] = tag(i);
    for (int c = 0; c < 3; ++c) k[2 + c] = x[3 * j + c] - x[3 * i + c];
  });
  exchange_rows(keys_out.data(), keys_in.data(), 5 * sizeof(double), send_counts, send_displs, recv_counts,
                recv_displs);

  // Each arriving key's partner among this rank's edges to ghosts: sorted by
  // (atom's tag, ghost's tag), found by binary search, told apart by the vector
  // where a small cell holds several images of one ghost.
  U64 mine("pet:mine", R);
  IV mine_r("pet:mine_r", R);
  Kokkos::parallel_for("pet_ex_mine", Range(0, R), KOKKOS_LAMBDA(int r) {
    mine(r) = uint64_t(tag(rem_i(r))) << 32 | uint64_t(tag(rem_j(r))), mine_r(r) = r;
  });
  if (R > 0) Kokkos::Experimental::sort_by_key(typename DeviceType::execution_space(), mine, mine_r);
  d_recv_map = pet::IView1D("pet:recv_map", n_recv);
  auto map = d_recv_map;
  int unmatched = 0;
  Kokkos::parallel_reduce(
      "pet_ex_match", Range(0, n_recv),
      KOKKOS_LAMBDA(int q, int &u) {
        const double *k = &keys_in(5 * std::size_t(q));
        const uint64_t want = uint64_t(k[0]) << 32 | uint64_t(k[1]);
        int lo = 0, hi = R;
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (mine(mid) < want) lo = mid + 1;
          else hi = mid;
        }
        int found = -1;
        for (int a = lo; a < R && mine(a) == want && found < 0; ++a) {
          const int r = mine_r(a), o = rem_i(r), h = rem_j(r);
          double d = 0;
          for (int c = 0; c < 3; ++c) d += Kokkos::fabs(x[3 * h + c] - x[3 * o + c] + k[2 + c]);
          if (d < 1e-6) found = r;
        }
        map(q) = found;
        u += found < 0;
      },
      unmatched);
  int all = 0;
  MPI_Allreduce(&unmatched, &all, 1, MPI_INT, MPI_SUM, world);
  if (all && comm->me == 0)
    error->warning(FLERR, "pair_style pet mode exchange: {} edge rows found no partner edge", all);
  atoms_map(owner);
  exchange_views();
}

// The point-to-point maps for atoms_forward/atoms_reverse: this rank's ghosts
// grouped by owner, and, told their tags, the owned atoms each rank asks for.
template <class DeviceType> void PairPETKokkos<DeviceType>::atoms_map(const pet::RView1D &owner)
{
  using Range = Kokkos::RangePolicy<DeviceType>;
  using U64 = Kokkos::View<uint64_t *, DeviceType>;
  using IV = Kokkos::View<int *, DeviceType>;
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost, nprocs = comm->nprocs, G = nall - nlocal;
  auto tag = atomKK->k_tag.template view<DeviceType>();
  // Lower bound of `v` in the high words of sorted keys: each rank's first entry.
  auto bounds = [](U64 keys, int n, IV out, int count) {
    Kokkos::parallel_for(
        "pet_bounds", Kokkos::RangePolicy<DeviceType>(0, count), KOKKOS_LAMBDA(int v) {
          int lo = 0, hi = n;
          while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (int(keys(mid) >> 32) < v) lo = mid + 1;
            else hi = mid;
          }
          out(v) = lo;
        });
  };

  U64 key("pet:gh_key", G);
  Kokkos::parallel_for(
      "pet_gh_key", Range(0, G),
      KOKKOS_LAMBDA(int q) { key(q) = uint64_t(int(owner(nlocal + q))) << 32 | uint32_t(nlocal + q); });
  if (G > 0) Kokkos::sort(key);
  d_gh_order = pet::IView1D("pet:gh_order", G);
  IV gd("pet:gh_displs", nprocs + 1);
  bounds(key, G, gd, nprocs + 1);
  auto gh = d_gh_order;
  Kokkos::parallel_for("pet_gh_order", Range(0, G), KOKKOS_LAMBDA(int q) { gh(q) = int(key(q) & 0xffffffffu); });
  std::vector<int> h(nprocs + 1);
  Kokkos::deep_copy(Kokkos::View<int *, Kokkos::HostSpace>(h.data(), nprocs + 1), gd);
  gh_sd.assign(h.begin(), h.end() - 1), gh_sc.resize(nprocs), srv_sc.resize(nprocs), srv_sd.assign(nprocs, 0);
  for (int p = 0; p < nprocs; ++p) gh_sc[p] = h[p + 1] - h[p];
  MPI_Alltoall(gh_sc.data(), 1, MPI_INT, srv_sc.data(), 1, MPI_INT, world);
  for (int p = 1; p < nprocs; ++p) srv_sd[p] = srv_sd[p - 1] + srv_sc[p - 1];
  const int S = srv_sd[nprocs - 1] + srv_sc[nprocs - 1];
  for (int p = 0; p < nprocs; ++p)  // the owners, and those this one owns for, alongside the edge peers
    if (p != comm->me && (gh_sc[p] > 0 || srv_sc[p] > 0) && std::find(peers.begin(), peers.end(), p) == peers.end())
      peers.push_back(p);
  std::sort(peers.begin(), peers.end());

  IV tags_out("pet:gh_tags", G), tags_in("pet:srv_tags", S);
  Kokkos::parallel_for("pet_gh_tags", Range(0, G), KOKKOS_LAMBDA(int q) { tags_out(q) = int(tag(gh(q))); });
  exchange_rows(tags_out.data(), tags_in.data(), sizeof(int), gh_sc, gh_sd, srv_sc, srv_sd);

  tagint max_tag = 0;
  Kokkos::parallel_reduce(
      "pet_max_tag", Range(0, nlocal), KOKKOS_LAMBDA(int i, tagint &m) { m = tag(i) > m ? tag(i) : m; },
      Kokkos::Max<tagint>(max_tag));
  IV by_tag("pet:by_tag", max_tag + 1);
  Kokkos::deep_copy(by_tag, -1);
  Kokkos::parallel_for("pet_by_tag", Range(0, nlocal), KOKKOS_LAMBDA(int i) { by_tag(tag(i)) = i; });
  d_srv_idx = pet::IView1D("pet:srv_idx", S);
  auto idx = d_srv_idx;
  int missing = 0;
  Kokkos::parallel_reduce(
      "pet_srv_idx", Range(0, S),
      KOKKOS_LAMBDA(int q, int &m) {
        const int t = tags_in(q);
        idx(q) = t >= 0 && t <= max_tag ? by_tag(t) : -1;
        m += idx(q) < 0;
        if (idx(q) < 0) idx(q) = 0;
      },
      missing);
  if (missing) error->one(FLERR, "pair_style pet/kk: {} ghosts asked of this rank are not its atoms", missing);

  // The reverse: each owned atom's arrivals, in arrival order.
  U64 rk("pet:rev_key", S);
  Kokkos::parallel_for(
      "pet_rev_key", Range(0, S), KOKKOS_LAMBDA(int q) { rk(q) = uint64_t(idx(q)) << 32 | uint32_t(q); });
  if (S > 0) Kokkos::sort(rk);
  d_rev_off = pet::IView1D("pet:rev_off", nlocal + 1), d_rev_pos = pet::IView1D("pet:rev_pos", S);
  bounds(rk, S, d_rev_off, nlocal + 1);
  auto pos = d_rev_pos;
  Kokkos::parallel_for("pet_rev_pos", Range(0, S), KOKKOS_LAMBDA(int k) { pos(k) = int(rk(k) & 0xffffffffu); });
  d_val_out = pet::RView1D("pet:val_out", S), d_val_in = pet::RView1D("pet:val_in", G);
  p2p_atoms = true;
}

// LAMMPS's positions as pet-kokkos takes them: [n, 3] row-major doubles.
template <class DeviceType> const double *PairPETKokkos<DeviceType>::positions()
{
  auto x = atomKK->k_x.template view<DeviceType>();
  if constexpr (std::is_same_v<typename decltype(x)::array_layout, Kokkos::LayoutRight>) {
    return x.data();
  } else {
    if (x_rows.extent(0) < x.extent(0)) x_rows = decltype(x_rows)("pet:x_rows", x.extent(0));
    Kokkos::deep_copy(Kokkos::subview(x_rows, std::make_pair(size_t(0), size_t(x.extent(0))), Kokkos::ALL), x);
    return x_rows.data();
  }
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

  auto f = atomKK->k_f.template view<DeviceType>();
  pet::Calculator::DeviceArrays a;
  a.positions = positions();
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
// Each ghost's value from its owner: point to point once atoms_map has run,
// else LAMMPS's forward comm.
template <class DeviceType> void PairPETKokkos<DeviceType>::atoms_forward(pet::RView1D a)
{
  if (!p2p_atoms) return lammps_forward(a);
  using Range = Kokkos::RangePolicy<DeviceType>;
  auto out = d_val_out, in = d_val_in;
  auto idx = d_srv_idx, gh = d_gh_order;
  Kokkos::parallel_for("pet_atoms_out", Range(0, idx.extent(0)), KOKKOS_LAMBDA(int q) { out(q) = a(idx(q)); });
  exchange_rows(out.data(), in.data(), sizeof(double), srv_sc, srv_sd, gh_sc, gh_sd);
  Kokkos::parallel_for("pet_atoms_in", Range(0, gh.extent(0)), KOKKOS_LAMBDA(int q) { a(gh(q)) = in(q); });
}

template <class DeviceType> void PairPETKokkos<DeviceType>::lammps_forward(pet::RView1D a)
{
  if (lmp->kokkos->forward_pair_comm_classic) return PairPET::atoms_forward(a);
  d_atom = a;
  comm->forward_comm(this);
}

template <class DeviceType> void PairPETKokkos<DeviceType>::atoms_reverse(pet::RView1D a)
{
  const int nlocal = atom->nlocal;
  using Range = Kokkos::RangePolicy<DeviceType>;
  if (p2p_atoms) {  // each ghost's value to its owner, summed there in a fixed order
    auto out = d_val_in, in = d_val_out;
    auto gh = d_gh_order, off = d_rev_off, pos = d_rev_pos;
    Kokkos::parallel_for("pet_atoms_back", Range(0, gh.extent(0)), KOKKOS_LAMBDA(int q) { out(q) = a(gh(q)); });
    exchange_rows(out.data(), in.data(), sizeof(double), gh_sc, gh_sd, srv_sc, srv_sd);
    Kokkos::parallel_for(
        "pet_atoms_sum", Range(0, nlocal), KOKKOS_LAMBDA(int i) {
          double s = 0.0;
          for (int k = off(i); k < off(i + 1); ++k) s += in(pos(k));
          a(i) += s;
        });
    Kokkos::parallel_for(
        "pet_zero_ghosts", Range(nlocal, a.extent(0)), KOKKOS_LAMBDA(int i) { a(i) = 0; });
    return;
  }
  if (lmp->kokkos->reverse_pair_comm_classic) return PairPET::atoms_reverse(a);
  d_atom = a;
  comm->reverse_comm(this);
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
