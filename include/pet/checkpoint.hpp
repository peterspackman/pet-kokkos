// A converted checkpoint: the <model>.json from tools/convert_pet.py (hypers,
// species map, composition energies, energy scale) and its weights.
#pragma once

#include <string>
#include <vector>

#include "pet/config.hpp"
#include "pet/safetensors.hpp"

namespace pet {

struct Checkpoint {
  Hypers hypers;
  double energy_scale = 1.0;
  int n_species = 0;
  std::string length_unit = "angstrom";
  std::string energy_unit = "eV";
  std::vector<int> species_to_index;         // [Z] species index, -1 if unsupported
  std::vector<double> composition_energies;  // [n_species] eV
  // The non-conservative heads' output scales, empty for a model without them
  // (or converted before the converter kept them): forces per species, the
  // stress one number. The heads predict forces and stress directly, with no
  // backward pass.
  std::vector<double> nc_force_scale;   // [n_species] or empty
  std::vector<double> nc_stress_scale;  // [1] or empty
  // The LLPR energy uncertainty, for a model that has one (0 features: none):
  // sigma = multiplier * |inv_cholesky * f|, f the last-layer features. The
  // matrices are the weights' llpr.energy.inv_cholesky and llpr.energy.ensemble.
  int llpr_features = 0;
  int llpr_ensemble = 0;  // ensemble members, 0 without an ensemble
  double llpr_multiplier = 0.0;
  SafeTensors weights;

  // Throws on a model this build does not implement (long-range, ZBL) rather
  // than returning a plausible wrong number.
  Checkpoint(const std::string& json_path, const std::string& safetensors_path);
};

}  // namespace pet
