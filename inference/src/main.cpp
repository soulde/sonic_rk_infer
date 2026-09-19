#include "rknn_api_compat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ModelConfig {
  std::string name;
  std::string path;
  size_t input_elements;
  size_t output_elements;
};

struct TimingStats {
  std::vector<double> us;

  void add(Clock::time_point begin, Clock::time_point end) {
    us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  double average() const {
    return std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());
  }

  double percentile(double fraction) const {
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const size_t index = std::min(sorted.size() - 1,
                                  static_cast<size_t>(fraction * sorted.size()));
    return sorted[index];
  }
};

class RknnModel {
 public:
  explicit RknnModel(ModelConfig config) : config_(std::move(config)) {}
  ~RknnModel() { close(); }

  void open() {
    std::ifstream file(config_.path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open model: " + config_.path);
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error("empty model: " + config_.path);
    model_.resize(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(model_.data()), size);

    const int ret = rknn_init(&ctx_, model_.data(), static_cast<uint32_t>(model_.size()), 0, nullptr);
    if (ret != 0) throw std::runtime_error(config_.name + " rknn_init failed: " + std::to_string(ret));

    rknn_input_output_num io{};
    check(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io");
    if (io.n_input != 1 || io.n_output != 1) {
      throw std::runtime_error(config_.name + " expected one input and one output");
    }
    input_attr_.index = 0;
    output_attr_.index = 0;
    check(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)), "query input");
    check(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_)), "query output");
    if (input_attr_.n_elems != config_.input_elements || output_attr_.n_elems != config_.output_elements) {
      throw std::runtime_error(config_.name + " tensor element count mismatch");
    }
    std::cout << config_.name << ": input=" << input_attr_.n_elems << " output=" << output_attr_.n_elems
              << " input_type=" << input_attr_.type << " output_type=" << output_attr_.type << '\n';
  }

  std::vector<float> run(const std::vector<float>& input) {
    if (input.size() != config_.input_elements) throw std::runtime_error("bad input size");
    rknn_input tensor{};
    tensor.index = 0;
    tensor.buf = const_cast<float*>(input.data());
    tensor.size = static_cast<uint32_t>(input.size() * sizeof(float));
    tensor.pass_through = 0;
    tensor.type = RKNN_TENSOR_FLOAT32;
    tensor.fmt = RKNN_TENSOR_UNDEFINED;
    check(rknn_inputs_set(ctx_, 1, &tensor), config_.name + " inputs_set");
    check(rknn_run(ctx_, nullptr), config_.name + " run");

    rknn_output output{};
    output.index = 0;
    output.want_float = 1;
    check(rknn_outputs_get(ctx_, 1, &output, nullptr), config_.name + " outputs_get");
    std::vector<float> result(config_.output_elements);
    if (output.size < result.size() * sizeof(float)) {
      rknn_outputs_release(ctx_, 1, &output);
      throw std::runtime_error(config_.name + " output is smaller than float output");
    }
    std::copy_n(static_cast<const float*>(output.buf), result.size(), result.begin());
    check(rknn_outputs_release(ctx_, 1, &output), config_.name + " outputs_release");
    return result;
  }

 private:
  static void check(int ret, const std::string& what) {
    if (ret != 0) throw std::runtime_error(what + " failed: " + std::to_string(ret));
  }

  void close() {
    if (ctx_ != 0) {
      rknn_destroy(ctx_);
      ctx_ = 0;
    }
  }

  ModelConfig config_;
  rknn_context ctx_ = 0;
  std::vector<uint8_t> model_;
  rknn_tensor_attr input_attr_{};
  rknn_tensor_attr output_attr_{};
};

struct Options {
  std::string encoder = "sonic_encoder_int8.rknn";
  std::string decoder = "sonic_decoder_int8.rknn";
  int warmup = 100;
  int iterations = 1000;
};

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (arg == "--encoder") options.encoder = next("--encoder");
    else if (arg == "--decoder") options.decoder = next("--decoder");
    else if (arg == "--warmup") options.warmup = std::stoi(next("--warmup"));
    else if (arg == "--iterations") options.iterations = std::stoi(next("--iterations"));
    else if (arg == "--help") {
      std::cout << "usage: sonic_rk3576_infer [--encoder path] [--decoder path] "
                   "[--warmup N] [--iterations N]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + arg);
  }
  if (options.warmup < 0 || options.iterations <= 0) throw std::runtime_error("invalid iteration counts");
  return options;
}

void print_stats(const std::string& name, const TimingStats& stats) {
  const double avg = stats.average();
  std::cout << std::fixed << std::setprecision(2) << name << ": avg=" << avg << " us, p50="
            << stats.percentile(0.50) << " us, p99=" << stats.percentile(0.99)
            << " us, avg_hz=" << (1e6 / avg) << '\n';
}

void fill_low_latency_encoder_observation(std::vector<float>& input, std::mt19937& rng) {
  // Layout mirrors policy/low_latency/observation_config.yaml and the C++
  // observation registry: mode(4), g1 pos/vel/ori(10 frames), teleop
  // single-frame fields, teleop lower-body pos/vel(10 frames), then SMPL
  // 4-frame fields. Use the g1 mode for this throughput smoke test; fields
  // belonging to other modes remain zero, exactly as mode filtering does.
  std::fill(input.begin(), input.end(), 0.0f);
  std::normal_distribution<float> noise(0.0f, 0.15f);
  input[0] = 0.0f;  // g1 encoder mode
  for (int frame = 0; frame < 10; ++frame) {
    for (int joint = 0; joint < 29; ++joint) {
      input[4 + frame * 29 + joint] = noise(rng);
      input[294 + frame * 29 + joint] = noise(rng);
    }
    for (int value = 0; value < 6; ++value) {
      input[584 + frame * 6 + value] = noise(rng);
    }
  }
}

void fill_low_latency_decoder_observation(std::vector<float>& input, std::mt19937& rng) {
  // Decoder layout is token_state(64) followed by five 10-frame history
  // groups: angular velocity(3), joint position(29), joint velocity(29),
  // last action(29), and gravity direction(3).
  std::normal_distribution<float> noise(0.0f, 0.15f);
  std::fill(input.begin(), input.end(), 0.0f);
  for (int frame = 0; frame < 10; ++frame) {
    for (int value = 0; value < 3; ++value) input[64 + frame * 3 + value] = noise(rng);
    for (int value = 0; value < 29; ++value) {
      input[94 + frame * 29 + value] = noise(rng);
      input[384 + frame * 29 + value] = noise(rng);
      input[674 + frame * 29 + value] = noise(rng);
    }
    for (int value = 0; value < 3; ++value) input[964 + frame * 3 + value] = noise(rng);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    RknnModel encoder({"encoder", options.encoder, 1247, 64});
    RknnModel decoder({"decoder", options.decoder, 994, 29});
    encoder.open();
    decoder.open();

    std::mt19937 rng(20260919);
    std::vector<float> encoder_input(1247);
    std::vector<float> decoder_input(994);
    fill_low_latency_encoder_observation(encoder_input, rng);
    fill_low_latency_decoder_observation(decoder_input, rng);

    for (int i = 0; i < options.warmup; ++i) {
      const auto token = encoder.run(encoder_input);
      std::copy(token.begin(), token.end(), decoder_input.begin());
      decoder.run(decoder_input);
    }

    TimingStats encoder_stats, decoder_stats, end_to_end_stats;
    std::vector<float> action;
    for (int i = 0; i < options.iterations; ++i) {
      const auto total_begin = Clock::now();
      const auto encoder_begin = Clock::now();
      const auto token = encoder.run(encoder_input);
      const auto encoder_end = Clock::now();
      std::copy(token.begin(), token.end(), decoder_input.begin());
      const auto decoder_begin = Clock::now();
      action = decoder.run(decoder_input);
      const auto decoder_end = Clock::now();
      encoder_stats.add(encoder_begin, encoder_end);
      decoder_stats.add(decoder_begin, decoder_end);
      end_to_end_stats.add(total_begin, decoder_end);
    }
    print_stats("encoder", encoder_stats);
    print_stats("decoder", decoder_stats);
    print_stats("end_to_end", end_to_end_stats);
    std::cout << "action[0..2]=" << action[0] << "," << action[1] << "," << action[2] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
