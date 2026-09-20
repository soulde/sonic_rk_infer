#pragma once

#include <cstdint>

extern "C" {

using rknn_context = uint64_t;

enum rknn_query_cmd {
  RKNN_QUERY_IN_OUT_NUM = 0,
  RKNN_QUERY_INPUT_ATTR = 1,
  RKNN_QUERY_OUTPUT_ATTR = 2,
  RKNN_QUERY_PERF_RUN = 4,
  RKNN_QUERY_SDK_VERSION = 5,
};

enum rknn_tensor_type {
  RKNN_TENSOR_FLOAT32 = 0,
  RKNN_TENSOR_INT8 = 2,
};

enum rknn_tensor_format {
  RKNN_TENSOR_NCHW = 0,
  RKNN_TENSOR_NHWC = 1,
  RKNN_TENSOR_NC1HWC2 = 2,
  RKNN_TENSOR_UNDEFINED = 3,
};

constexpr uint32_t RKNN_MAX_DIMS = 16;
constexpr uint32_t RKNN_MAX_NAME_LEN = 256;

struct rknn_input_output_num {
  uint32_t n_input;
  uint32_t n_output;
};

struct rknn_tensor_attr {
  uint32_t index;
  uint32_t n_dims;
  uint32_t dims[RKNN_MAX_DIMS];
  char name[RKNN_MAX_NAME_LEN];
  uint32_t n_elems;
  uint32_t size;
  rknn_tensor_format fmt;
  rknn_tensor_type type;
  int32_t qnt_type;
  int8_t fl;
  int32_t zp;
  float scale;
  uint32_t w_stride;
  uint32_t size_with_stride;
  uint8_t pass_through;
  uint32_t h_stride;
};

struct rknn_input {
  uint32_t index;
  void* buf;
  uint32_t size;
  uint8_t pass_through;
  rknn_tensor_type type;
  rknn_tensor_format fmt;
};

struct rknn_output {
  uint8_t want_float;
  uint8_t is_prealloc;
  uint32_t index;
  void* buf;
  uint32_t size;
};

int rknn_init(rknn_context*, void*, uint32_t, uint32_t, void*);
int rknn_destroy(rknn_context);
int rknn_query(rknn_context, rknn_query_cmd, void*, uint32_t);
int rknn_inputs_set(rknn_context, uint32_t, rknn_input[]);
int rknn_run(rknn_context, void*);
int rknn_outputs_get(rknn_context, uint32_t, rknn_output[], void*);
int rknn_outputs_release(rknn_context, uint32_t, rknn_output[]);

}
