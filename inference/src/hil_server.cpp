#include "rknn_api_compat.h"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x434e4f53;  // bytes "SONC" on little-endian hosts
constexpr uint16_t kVersion = 1;
constexpr uint16_t kRequest = 1;
constexpr uint16_t kResponse = 2;
constexpr size_t kEncoderElements = 1247;
constexpr size_t kDecoderElements = 994;
constexpr size_t kActionElements = 29;
constexpr size_t kRequestFloats = kEncoderElements + kDecoderElements;

#pragma pack(push, 1)
struct PacketHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
  uint32_t sequence;
  uint32_t reserved;
};
#pragma pack(pop)

void check(int ret, const std::string& what) {
  if (ret != 0) throw std::runtime_error(what + " failed: " + std::to_string(ret));
}

void read_full(int fd, void* buffer, size_t size) {
  auto* bytes = static_cast<char*>(buffer);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t n = ::recv(fd, bytes + offset, size - offset, MSG_WAITALL);
    if (n <= 0) throw std::runtime_error("peer closed connection");
    offset += static_cast<size_t>(n);
  }
}

void write_full(int fd, const void* buffer, size_t size) {
  const auto* bytes = static_cast<const char*>(buffer);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t n = ::send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
    if (n <= 0) throw std::runtime_error("send failed");
    offset += static_cast<size_t>(n);
  }
}

class Model {
 public:
  Model(const std::string& name, const std::string& path, size_t input_size, size_t output_size)
      : name_(name), input_size_(input_size), output_size_(output_size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const auto file_size = file.tellg();
    model_.resize(static_cast<size_t>(file_size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(model_.data()), file_size);
    check(rknn_init(&ctx_, model_.data(), static_cast<uint32_t>(model_.size()), 0, nullptr), name_ + " init");
    rknn_input_output_num io{};
    check(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), name_ + " query io");
    if (io.n_input != 1 || io.n_output != 1) throw std::runtime_error(name_ + " expects one input/output");
  }

  ~Model() { if (ctx_) rknn_destroy(ctx_); }

  std::vector<float> run(const float* input) {
    rknn_input tensor{};
    tensor.index = 0;
    tensor.buf = const_cast<float*>(input);
    tensor.size = static_cast<uint32_t>(input_size_ * sizeof(float));
    tensor.pass_through = 0;
    tensor.type = RKNN_TENSOR_FLOAT32;
    tensor.fmt = RKNN_TENSOR_UNDEFINED;
    check(rknn_inputs_set(ctx_, 1, &tensor), name_ + " inputs_set");
    check(rknn_run(ctx_, nullptr), name_ + " run");
    rknn_output output{};
    output.index = 0;
    output.want_float = 1;
    check(rknn_outputs_get(ctx_, 1, &output, nullptr), name_ + " outputs_get");
    if (output.size < output_size_ * sizeof(float)) {
      rknn_outputs_release(ctx_, 1, &output);
      throw std::runtime_error(name_ + " output size mismatch");
    }
    std::vector<float> result(output_size_);
    std::memcpy(result.data(), output.buf, output_size_ * sizeof(float));
    check(rknn_outputs_release(ctx_, 1, &output), name_ + " outputs_release");
    return result;
  }

 private:
  std::string name_;
  size_t input_size_;
  size_t output_size_;
  rknn_context ctx_ = 0;
  std::vector<uint8_t> model_;
};

int make_server_socket(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket failed");
  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 || ::listen(fd, 1) < 0) {
    ::close(fd);
    throw std::runtime_error("bind/listen failed");
  }
  return fd;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: sonic_rk3576_hil_server <encoder.rknn> <decoder.rknn> <port> [max_steps]\n";
    return 2;
  }
  try {
    const int port = std::stoi(argv[3]);
    const uint64_t max_steps = argc == 5 ? std::stoull(argv[4]) : 0;
    Model encoder("encoder", argv[1], kEncoderElements, 64);
    Model decoder("decoder", argv[2], kDecoderElements, kActionElements);
    const int server = make_server_socket(port);
    std::cout << "HIL server listening on 0.0.0.0:" << port << "\n" << std::flush;
    const int client = ::accept(server, nullptr, nullptr);
    if (client < 0) throw std::runtime_error("accept failed");
    int no_delay = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    std::cout << "HIL client connected\n" << std::flush;

    std::vector<float> request(kRequestFloats);
    std::vector<float> decoder_input(kDecoderElements);
    uint64_t steps = 0;
    auto last_report = std::chrono::steady_clock::now();
    while (max_steps == 0 || steps < max_steps) {
      PacketHeader request_header{};
      read_full(client, &request_header, sizeof(request_header));
      if (request_header.magic != kMagic || request_header.version != kVersion || request_header.type != kRequest) {
        throw std::runtime_error("invalid request header");
      }
      read_full(client, request.data(), request.size() * sizeof(float));
      const auto token = encoder.run(request.data());
      std::memcpy(decoder_input.data(), request.data() + kEncoderElements, kDecoderElements * sizeof(float));
      std::memcpy(decoder_input.data(), token.data(), token.size() * sizeof(float));
      const auto action = decoder.run(decoder_input.data());

      PacketHeader response{kMagic, kVersion, kResponse, request_header.sequence, 0};
      std::vector<uint8_t> response_packet(sizeof(response) + action.size() * sizeof(float));
      std::memcpy(response_packet.data(), &response, sizeof(response));
      std::memcpy(response_packet.data() + sizeof(response), action.data(), action.size() * sizeof(float));
      write_full(client, response_packet.data(), response_packet.size());
      ++steps;
      if (steps % 100 == 0) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last_report).count();
        std::cout << "steps=" << steps << " server_hz=" << (100.0 / seconds) << "\n" << std::flush;
        last_report = now;
      }
    }
    ::close(client);
    ::close(server);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "HIL server error: " << error.what() << '\n';
    return 1;
  }
}
