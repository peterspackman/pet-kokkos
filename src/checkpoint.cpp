#include "pet/checkpoint.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace pet {

SafeTensors::SafeTensors(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("safetensors: cannot open '" + path + "'");
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error("safetensors: truncated header");
  const std::streampos data_start = f.tellg();

  const auto j = nlohmann::json::parse(header);
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& name = it.key();
    if (name == "__metadata__") continue;
    const auto& e = it.value();
    const std::string dtype = e.at("dtype").get<std::string>();
    const int64_t begin = e.at("data_offsets")[0].get<int64_t>();
    const int64_t end = e.at("data_offsets")[1].get<int64_t>();
    Tensor t;
    for (const auto& s : e.at("shape")) t.shape.push_back(s.get<int64_t>());
    const int64_t n = t.numel();
    const int64_t item = dtype == "F32" ? 4 : dtype == "F64" ? 8 : 0;
    if (!item) throw std::runtime_error("safetensors: unsupported dtype '" + dtype + "' for '" + name + "'");
    // The byte range and the shape come separately; a mismatch would overrun.
    if (begin < 0 || end - begin != n * item)
      throw std::runtime_error("safetensors: '" + name + "' byte range does not match its shape");

    f.seekg(data_start + static_cast<std::streamoff>(begin));
    t.data.resize(static_cast<std::size_t>(n));
    if (item == 8) {
      f.read(reinterpret_cast<char*>(t.data.data()), end - begin);
    } else {
      std::vector<float> raw(static_cast<std::size_t>(n));
      f.read(reinterpret_cast<char*>(raw.data()), end - begin);
      for (int64_t i = 0; i < n; ++i) t.data[i] = raw[i];
    }
    if (!f) throw std::runtime_error("safetensors: truncated data for '" + name + "'");
    tensors_.emplace(name, std::move(t));
  }
}

const Tensor& SafeTensors::at(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("safetensors: missing tensor '" + name + "'");
  return it->second;
}

namespace {

template <class E>
E parse(const std::string& key, const std::string& s, std::initializer_list<std::pair<const char*, E>> names) {
  for (const auto& [n, e] : names)
    if (s == n) return e;
  throw std::runtime_error("checkpoint: unknown " + key + " '" + s + "'");
}

}  // namespace

Checkpoint::Checkpoint(const std::string& json_path, const std::string& safetensors_path)
    : weights(safetensors_path) {
  std::ifstream f(json_path);
  if (!f) throw std::runtime_error("checkpoint: cannot open '" + json_path + "'");
  nlohmann::json j;
  f >> j;

  const auto& h = j.at("hypers");
  Hypers& p = hypers;
  p.d_pet = h.at("d_pet");
  p.d_head = h.at("d_head");
  p.d_node = h.at("d_node");
  p.d_feedforward = h.at("d_feedforward");
  p.num_heads = h.at("num_heads");
  p.head_dim = h.at("head_dim");
  p.num_attention_layers = h.at("num_attention_layers");
  p.num_gnn_layers = h.at("num_gnn_layers");
  p.attention_temperature = h.at("attention_temperature");
  p.cutoff = h.at("cutoff");
  p.cutoff_width = h.at("cutoff_width");
  if (!h.at("num_neighbors_adaptive").is_null()) p.num_neighbors_adaptive = h.at("num_neighbors_adaptive");
  // Older checkpoints predate these keys: Grid, and one taper width.
  p.adaptive_cutoff_method = parse<AdaptiveCutoffMethod>(
      "adaptive_cutoff_method", h.value("adaptive_cutoff_method", "grid"),
      {{"grid", AdaptiveCutoffMethod::Grid}, {"solver", AdaptiveCutoffMethod::Solver}});
  p.cutoff_width_adaptive = h.value("cutoff_width_adaptive", p.cutoff_width);
  p.zbl = h.value("zbl", false);
  p.system_conditioning = h.value("system_conditioning", false);
  p.max_charge = h.value("max_charge", 10);
  p.max_spin_multiplicity = h.value("max_spin_multiplicity", 10);
  p.long_range_enabled = h.value("long_range_enabled", false);
  p.normalization = parse<Normalization>("normalization", h.at("normalization"),
                                         {{"RMSNorm", Normalization::RMSNorm}, {"LayerNorm", Normalization::LayerNorm}});
  p.activation = parse<Activation>("activation", h.at("activation"),
                                   {{"SwiGLU", Activation::SwiGLU}, {"SiLU", Activation::SiLU}});
  p.transformer_type = parse<TransformerType>("transformer_type", h.at("transformer_type"),
                                              {{"PreLN", TransformerType::PreLN}, {"PostLN", TransformerType::PostLN}});
  p.featurizer_type = parse<FeaturizerType>("featurizer_type", h.at("featurizer_type"),
                                            {{"feedforward", FeaturizerType::FeedForward},
                                             {"residual", FeaturizerType::Residual}});
  p.cutoff_function = parse<CutoffFunction>("cutoff_function", h.at("cutoff_function"),
                                            {{"Bump", CutoffFunction::Bump}, {"Cosine", CutoffFunction::Cosine}});
  // Feedforward reads out once, residual after every GNN layer.
  p.num_readout_layers =
      h.value("num_readout_layers", p.featurizer_type == FeaturizerType::FeedForward ? 1 : p.num_gnn_layers);

  energy_scale = j.at("energy_scale");
  n_species = j.at("n_species");
  length_unit = j.value("length_unit", "angstrom");
  energy_unit = j.value("energy_unit", "eV");
  species_to_index = j.at("species_to_index").get<std::vector<int>>();
  composition_energies = j.at("composition_energies").get<std::vector<double>>();

  if (p.long_range_enabled) throw std::runtime_error("checkpoint: long-range models not yet supported");
  if (p.zbl) throw std::runtime_error("checkpoint: ZBL term not yet supported");
}

}  // namespace pet
