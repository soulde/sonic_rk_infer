#include "rknn_api_compat.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Model {
 public:
  Model(const std::string& path, size_t input_count, size_t output_count)
      : input_count_(input_count), output_count_(output_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    auto size = static_cast<size_t>(f.tellg());
    blob_.resize(size); f.seekg(0); f.read(reinterpret_cast<char*>(blob_.data()), size);
    check(rknn_init(&ctx_, blob_.data(), static_cast<uint32_t>(blob_.size()), 0, nullptr), "rknn_init");
    rknn_tensor_attr in{}, out{}; in.index = out.index = 0;
    check(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)), "input_attr");
    check(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out, sizeof(out)), "output_attr");
    if (in.n_elems != input_count_ || out.n_elems != output_count_)
      throw std::runtime_error("model tensor size mismatch");
    std::cerr << "model input=" << in.n_elems << " output=" << out.n_elems << "\n";
  }
  ~Model() { if (ctx_) rknn_destroy(ctx_); }
  void run(const float* input, float* output) {
    rknn_input in{}; in.index = 0; in.buf = const_cast<float*>(input);
    in.size = static_cast<uint32_t>(input_count_ * sizeof(float));
    in.type = RKNN_TENSOR_FLOAT32; in.fmt = RKNN_TENSOR_UNDEFINED;
    check(rknn_inputs_set(ctx_, 1, &in), "inputs_set");
    check(rknn_run(ctx_, nullptr), "run");
    rknn_output out{}; out.index = 0; out.want_float = 1;
    check(rknn_outputs_get(ctx_, 1, &out, nullptr), "outputs_get");
    if (out.size < output_count_ * sizeof(float)) throw std::runtime_error("output too small");
    std::memcpy(output, out.buf, output_count_ * sizeof(float));
    check(rknn_outputs_release(ctx_, 1, &out), "outputs_release");
  }
 private:
  static void check(int ret, const char* what) { if (ret) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(ret)); }
  rknn_context ctx_ = 0; std::vector<uint8_t> blob_; size_t input_count_, output_count_;
};

int main(int argc, char** argv) {
  if (argc != 7) { std::cerr << "usage: runner MODEL INPUT_F32 OUTPUT_F32 COUNT INPUT_WIDTH OUTPUT_WIDTH\n"; return 2; }
  try {
    const size_t count = std::stoul(argv[4]), input_width = std::stoul(argv[5]);
    const size_t output_width = std::stoul(argv[6]);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output");
    Model model(argv[1], input_width, output_width);
    std::vector<float> in(input_width), out(output_width);
    for (size_t i = 0; i < count; ++i) {
      input.read(reinterpret_cast<char*>(in.data()), static_cast<std::streamsize>(in.size() * sizeof(float)));
      if (input.gcount() != static_cast<std::streamsize>(in.size() * sizeof(float))) throw std::runtime_error("short input");
      model.run(in.data(), out.data());
      output.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    }
    std::cerr << "processed " << count << " samples\n";
  } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 1; }
}
