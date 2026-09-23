// A minimal safetensors reader: an 8-byte little-endian header length, a JSON
// header { name: {dtype, shape, data_offsets: [a, b]} }, then the raw bytes.
// F32 and F64 only, everything read as double.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pet {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<double> data;  // row-major

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
  explicit SafeTensors(const std::string& path);
  bool has(const std::string& name) const { return tensors_.count(name) > 0; }
  const Tensor& at(const std::string& name) const;  // throws if missing
  std::size_t size() const { return tensors_.size(); }

 private:
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace pet
