// Parses the <model>.json metadata emitted by tools/convert_pet.py into a
// Hypers struct plus species map / composition energies / energy scale, and
// owns the SafeTensors weight blob.
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "pet/config.hpp"
#include "pet/safetensors.hpp"

namespace pet {

class Checkpoint {
 public:
  Hypers hypers;
  double energy_scale = 1.0;
  int n_species = 0;
  std::string length_unit = "angstrom";
  std::string energy_unit = "eV";

  // species_to_index[Z] -> species index in [0, n_species), or -1 if unsupported.
  std::vector<int> species_to_index;
  // composition_energies[species_index] -> reference energy (eV), length n_species.
  std::vector<double> composition_energies;

  SafeTensors weights;

  Checkpoint(const std::string& json_path, const std::string& safetensors_path)
      : weights(safetensors_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("checkpoint: cannot open '" + json_path + "'");
    nlohmann::json j;
    f >> j;

    const auto& h = j.at("hypers");
    hypers.d_pet = h.at("d_pet").get<int>();
    hypers.d_head = h.at("d_head").get<int>();
    hypers.d_node = h.at("d_node").get<int>();
    hypers.d_feedforward = h.at("d_feedforward").get<int>();
    hypers.num_heads = h.at("num_heads").get<int>();
    hypers.head_dim = h.at("head_dim").get<int>();
    hypers.num_attention_layers = h.at("num_attention_layers").get<int>();
    hypers.num_gnn_layers = h.at("num_gnn_layers").get<int>();
    // num_readout_layers: explicit if present, else derived from the featurizer
    // (feedforward -> 1, residual -> num_gnn_layers).
    hypers.attention_temperature = h.at("attention_temperature").get<double>();
    hypers.cutoff = h.at("cutoff").get<double>();
    hypers.cutoff_width = h.at("cutoff_width").get<double>();
    if (!h.at("num_neighbors_adaptive").is_null())
      hypers.num_neighbors_adaptive = h.at("num_neighbors_adaptive").get<double>();
    // Absent -> the checkpoint predates the choice, which means Grid. Absent
    // cutoff_width_adaptive likewise predates the split from cutoff_width.
    hypers.adaptive_cutoff_method =
        parse_acm(h.value("adaptive_cutoff_method", std::string("grid")));
    hypers.cutoff_width_adaptive =
        h.value("cutoff_width_adaptive", hypers.cutoff_width);
    hypers.zbl = h.value("zbl", false);
    hypers.system_conditioning = h.value("system_conditioning", false);
    hypers.max_charge = h.value("max_charge", 10);
    hypers.max_spin_multiplicity = h.value("max_spin_multiplicity", 10);
    hypers.long_range_enabled = h.value("long_range_enabled", false);

    hypers.normalization = parse_norm(h.at("normalization").get<std::string>());
    hypers.activation = parse_act(h.at("activation").get<std::string>());
    hypers.transformer_type = parse_tt(h.at("transformer_type").get<std::string>());
    hypers.featurizer_type = parse_ft(h.at("featurizer_type").get<std::string>());
    hypers.cutoff_function = parse_cf(h.at("cutoff_function").get<std::string>());

    hypers.num_readout_layers = h.value(
        "num_readout_layers",
        hypers.featurizer_type == FeaturizerType::FeedForward ? 1 : hypers.num_gnn_layers);

    energy_scale = j.at("energy_scale").get<double>();
    n_species = j.at("n_species").get<int>();
    length_unit = j.value("length_unit", std::string("angstrom"));
    energy_unit = j.value("energy_unit", std::string("eV"));
    species_to_index = j.at("species_to_index").get<std::vector<int>>();
    composition_energies = j.at("composition_energies").get<std::vector<double>>();

    // Refuse, by name, anything this build does not implement -- rather than
    // running the wrong path and returning a plausible wrong number. An adaptive
    // cutoff chosen by the solver differs from one chosen on the probe grid, and
    // the difference is an energy, not a crash.
    if (hypers.long_range_enabled)
      throw std::runtime_error("checkpoint: long-range models not yet supported");
    if (hypers.zbl) throw std::runtime_error("checkpoint: ZBL term not yet supported");
  }

 private:
  static Normalization parse_norm(const std::string& s) {
    if (s == "RMSNorm") return Normalization::RMSNorm;
    if (s == "LayerNorm") return Normalization::LayerNorm;
    throw std::runtime_error("unknown normalization '" + s + "'");
  }
  static Activation parse_act(const std::string& s) {
    if (s == "SwiGLU") return Activation::SwiGLU;
    if (s == "SiLU") return Activation::SiLU;
    throw std::runtime_error("unknown activation '" + s + "'");
  }
  static TransformerType parse_tt(const std::string& s) {
    if (s == "PreLN") return TransformerType::PreLN;
    if (s == "PostLN") return TransformerType::PostLN;
    throw std::runtime_error("unknown transformer_type '" + s + "'");
  }
  static FeaturizerType parse_ft(const std::string& s) {
    if (s == "feedforward") return FeaturizerType::FeedForward;
    if (s == "residual") return FeaturizerType::Residual;
    throw std::runtime_error("unknown featurizer_type '" + s + "'");
  }
  static AdaptiveCutoffMethod parse_acm(const std::string& s) {
    if (s == "grid") return AdaptiveCutoffMethod::Grid;
    if (s == "solver") return AdaptiveCutoffMethod::Solver;
    throw std::runtime_error("unknown adaptive_cutoff_method '" + s + "'");
  }
  static CutoffFunction parse_cf(const std::string& s) {
    if (s == "Bump") return CutoffFunction::Bump;
    if (s == "Cosine") return CutoffFunction::Cosine;
    throw std::runtime_error("unknown cutoff_function '" + s + "'");
  }
};

}  // namespace pet
