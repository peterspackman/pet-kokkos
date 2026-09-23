// The host neighbour list: a structure's edges, its adaptive cutoff, and the
// padded per-atom layout the model reads (NEF). It reproduces metatrain's
// preprocessing, so the network sees the inputs it was trained on. The device
// builder (device_neighbors.hpp) does the same without leaving the GPU.
#pragma once

#include <array>
#include <vector>

#include "pet/config.hpp"

namespace pet {

// A structure to evaluate.
struct System {
  int n_atoms = 0;
  std::vector<int> atomic_numbers;  // [N]
  std::vector<double> positions;    // [N*3] Angstrom
  std::array<double, 9> cell{};     // lattice vectors as rows
  std::array<bool, 3> pbc{false, false, false};
  // For a model trained with system_conditioning; the defaults are metatrain's
  // (neutral, singlet).
  int charge = 0;
  int spin_multiplicity = 1;
};

// A neighbour list in NEF: atom n's kept edges in slots [n*M, n*M + count),
// padding after, M the largest count. [N*M] arrays are flat, row-major.
struct EdgeData {
  int n_atoms = 0;
  int max_neighbors = 0;              // M
  std::vector<int> species;           // [N] species index
  std::vector<int> num_neigh;         // [N]
  std::vector<double> edge_vec;       // [N*M*3] r_j + shift - r_i
  std::vector<double> edge_dist;      // [N*M]
  std::vector<int> neigh_species;     // [N*M]
  std::vector<double> cutoff_factor;  // [N*M]
  std::vector<double> pair_cutoff;    // [N*M]
  std::vector<char> mask;             // [N*M] 1 kept, 0 padding
  std::vector<int> reverse_index;     // [N*M] slot of the (j, i, -shift) edge, -1 if none

  // Per structure, for a conditioned model: one entry per concatenated structure.
  std::vector<int> charge, spin_multiplicity;

  // The solver adaptive cutoff's root and slope per atom (0 slope: clamped),
  // which its backward needs. Empty otherwise.
  std::vector<double> adapt_r, adapt_dn;

  // Every edge within the cutoff, before the adaptive cutoff drops any: each
  // atom's cutoff depends on all of them, so its backward reaches all of them.
  int n_raw = 0;
  std::vector<int> raw_center, raw_neigh;  // [E]
  std::vector<double> raw_dist;            // [E]
  std::vector<double> raw_vec;             // [E*3]
};

// An edge supplied to build_edge_data_from_raw. center and neigh_owner are local
// atom indices (the owner of a periodic or ghost image); the tags are global ids,
// used only to match an edge with its reverse.
struct RawEdgeIn {
  int center, neigh_owner;
  long long tag_center, tag_neigh;
  int sa, sb, sc;     // lattice shift of the neighbour's image
  double vx, vy, vz;  // r_j + shift - r_i
  double dist;
};

// NEF from a full edge list within the cutoff, and species already mapped to
// indices.
EdgeData build_edge_data_from_raw(int N, const std::vector<int>& species, const Hypers& h,
                                  const std::vector<RawEdgeIn>& edges);
// NEF for one structure, with its own periodic neighbour search.
EdgeData build_edge_data(const System& sys, const Hypers& h, const std::vector<int>& species_to_index);
// Several structures' NEF as one, for a batch: atoms renumbered, rows padded to
// the largest M. struct_id receives each atom's structure. Edges never cross
// structures, so each gets the answer it would get alone.
EdgeData concat_edge_data(const std::vector<EdgeData>& parts, std::vector<int>& struct_id);

// The search build_edge_data uses: a property of the run, set by the Calculator
// from its Options, or by PET_NEIGHBORS=builtin|vesin.
enum class NeighborBackend { Builtin, Vesin };
NeighborBackend& neighbor_backend();

namespace detail {

// The smooth cutoff functions, and the bump's derivative in the cutoff radius.
double bump_cutoff(double d, double rc, double width);
double cosine_cutoff(double d, double rc, double width);
double bump_dcutoff_dr(double d, double rc, double width);

// A directed edge from the search.
struct RawEdge {
  int i, j;
  int sa, sb, sc;     // lattice shift applied to j
  double vx, vy, vz;  // r_j + shift - r_i
  double dist;
};
// How many periodic images to search along each lattice vector.
std::array<int, 3> image_ranges(const System& sys, double cutoff);
// Every edge within the cutoff, both directions, brute force over images; the
// same set as vesin's full list.
std::vector<RawEdge> build_raw_edges(const System& sys, double cutoff);
// build_raw_edges or vesin's cell list, per neighbor_backend().
std::vector<RawEdge> build_raw_edges_dispatch(const System& sys, double cutoff);

}  // namespace detail

}  // namespace pet
