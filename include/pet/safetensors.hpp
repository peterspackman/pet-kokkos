// Minimal safetensors reader (header-only).
//
// safetensors layout:
//   [8 bytes little-endian uint64 header_len]
//   [header_len bytes of JSON: { name: {dtype, shape, data_offsets:[a,b]}, ... }]
//   [raw tensor bytes]  (data_offsets are relative to the start of this region)
//
// We only need float32/float64 reading; everything is returned as double so the
// rest of the code can stay precision-agnostic. The on-disk PET weights are f32.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace pet {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<double> data;  // row-major, length = prod(shape)

  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  int64_t dim(int i) const { return shape.at(i); }
  int ndim() const { return static_cast<int>(shape.size()); }
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path) { load(path); }

  bool has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
  }

  const Tensor& at(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
      throw std::runtime_error("safetensors: missing tensor '" + name + "'");
    return it->second;
  }

  std::size_t size() const { return tensors_.size(); }

 private:
  std::unordered_map<std::string, Tensor> tensors_;

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open '" + path + "'");

    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f) throw std::runtime_error("safetensors: truncated header length");

    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!f) throw std::runtime_error("safetensors: truncated header");

    const std::streampos data_start = f.tellg();

    auto j = nlohmann::json::parse(header);
    for (auto it = j.begin(); it != j.end(); ++it) {
      const std::string& name = it.key();
      if (name == "__metadata__") continue;
      const auto& entry = it.value();
      const std::string dtype = entry.at("dtype").get<std::string>();

      Tensor t;
      for (const auto& s : entry.at("shape")) t.shape.push_back(s.get<int64_t>());
      const auto offsets = entry.at("data_offsets");
      const int64_t begin = offsets[0].get<int64_t>();
      const int64_t end = offsets[1].get<int64_t>();
      const int64_t nbytes = end - begin;
      const int64_t n = t.numel();

      // The read below is sized from the header's byte range but lands in a buffer
      // sized from the header's shape. Nothing guarantees the two agree, so check:
      // a file whose byte range is longer than its shape would otherwise overrun
      // the heap rather than be rejected.
      auto expect_bytes = [&](int64_t itemsize) {
        if (begin < 0 || end < begin || nbytes != n * itemsize)
          throw std::runtime_error("safetensors: tensor '" + name +
                                   "' byte range does not match its shape");
      };

      f.seekg(data_start + static_cast<std::streamoff>(begin));
      t.data.resize(static_cast<std::size_t>(n));

      if (dtype == "F32") {
        expect_bytes(4);
        std::vector<float> raw(static_cast<std::size_t>(n));
        f.read(reinterpret_cast<char*>(raw.data()), nbytes);
        for (int64_t i = 0; i < n; ++i) t.data[i] = static_cast<double>(raw[i]);
      } else if (dtype == "F64") {
        expect_bytes(8);
        f.read(reinterpret_cast<char*>(t.data.data()), nbytes);
      } else {
        throw std::runtime_error("safetensors: unsupported dtype '" + dtype +
                                 "' for tensor '" + name + "'");
      }
      if (!f) throw std::runtime_error("safetensors: truncated data for '" + name + "'");
      tensors_.emplace(name, std::move(t));
    }
  }
};

}  // namespace pet
