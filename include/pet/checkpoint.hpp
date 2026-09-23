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
  SafeTensors weights;

  // Throws on a model this build does not implement (long-range, ZBL) rather
  // than returning a plausible wrong number.
  Checkpoint(const std::string& json_path, const std::string& safetensors_path);
};

}  // namespace pet
