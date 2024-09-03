#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "my_micro_context.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
template <int SZ, class T>
struct TfArray {
  int sz;
  T elem[SZ];

  TFLMRegistration reg[1];
  void op_init(void) { reg[0] = tflite::Register_FULLY_CONNECTED(); }

  uint8_t buffer_1[1];
  const uint8_t buffer_2[] = {
      0xad,
      0x1,
      0x0,
      0x0,
  };
  const uint8_t buffer_3[] = {
      0xd9, 0x3b, 0x27, 0x15, 0x1c, 0xe0, 0xde, 0xdd,
      0xf,  0x1b, 0xc5, 0xd7, 0x12, 0xdd, 0xf9, 0x7f,
  };
  const uint8_t buffer_4[] = {
      0x27, 0xfd, 0xff, 0xff, 0xa2, 0x7,  0x0,  0x0,  0x62, 0x2,  0x0,
      0x0,  0x0,  0x0,  0x0,  0x0,  0xf1, 0x0,  0x0,  0x0,  0x29, 0xfe,
      0xff, 0xff, 0xdd, 0xff, 0xff, 0xff, 0x9d, 0xfc, 0xff, 0xff, 0x3b,
      0x2,  0x0,  0x0,  0x45, 0x2,  0x0,  0x0,  0xa4, 0x10, 0x0,  0x0,
      0x67, 0xf,  0x0,  0x0,  0x4f, 0x2,  0x0,  0x0,  0x0,  0x0,  0x0,
      0x0,  0x87, 0xfc, 0xff, 0xff, 0x11, 0xec, 0xff, 0xff,
  };
  const uint8_t buffer_5[] = {
      0xf4, 0x1a, 0xed, 0x9,  0x19, 0x21, 0xf4, 0x24, 0xe0, 0x21, 0xef, 0xbc,
      0xf7, 0xf5, 0xfa, 0x19, 0x3,  0xdc, 0xd2, 0x2,  0x6,  0xf9, 0xf4, 0x2,
      0xff, 0xfa, 0xef, 0xf1, 0xef, 0xd3, 0x27, 0xe1, 0xfb, 0x27, 0xdd, 0xeb,
      0xdb, 0xe4, 0x5,  0x1a, 0x17, 0xfc, 0x24, 0x12, 0x15, 0xef, 0x1e, 0xe4,
      0x10, 0xfe, 0x14, 0xda, 0x1c, 0xf8, 0xf3, 0xf1, 0xef, 0xe2, 0xf3, 0x9,
      0xe3, 0xe9, 0xed, 0xe3, 0xe4, 0x15, 0x7,  0xb,  0x4,  0x1b, 0x1a, 0xfe,
      0xeb, 0x1,  0xde, 0x21, 0xe6, 0xb,  0xec, 0x3,  0x23, 0xa,  0x22, 0x24,
      0x1e, 0x27, 0x3,  0xe6, 0x3,  0x24, 0xff, 0xc0, 0x11, 0xf8, 0xfc, 0xf1,
      0x11, 0xc,  0xf5, 0xe0, 0xf3, 0x7,  0x17, 0xe5, 0xe8, 0xed, 0xfa, 0xdc,
      0xe8, 0x23, 0xfb, 0x7,  0xdd, 0xfb, 0xfd, 0x0,  0x14, 0x26, 0x11, 0x17,
      0xe7, 0xf1, 0x11, 0xea, 0x2,  0x26, 0x4,  0x4,  0x25, 0x21, 0x1d, 0xa,
      0xdb, 0x1d, 0xdc, 0x20, 0x1,  0xfa, 0xe3, 0x37, 0xb,  0xf1, 0x1a, 0x16,
      0xef, 0x1c, 0xe7, 0x3,  0xe0, 0x16, 0x2,  0x3,  0x21, 0x18, 0x9,  0x2e,
      0xd9, 0xe5, 0x14, 0xb,  0xea, 0x1a, 0xfc, 0xd8, 0x13, 0x0,  0xc4, 0xd8,
      0xec, 0xd9, 0xfe, 0xd,  0x19, 0x20, 0xd8, 0xd6, 0xe2, 0x1f, 0xe9, 0xd7,
      0xca, 0xe2, 0xdd, 0xc6, 0x13, 0xe7, 0x4,  0x3e, 0x0,  0x1,  0x14, 0xc7,
      0xdb, 0xe7, 0x15, 0x15, 0xf5, 0x6,  0xd6, 0x1a, 0xdc, 0x9,  0x22, 0xfe,
      0x8,  0x2,  0x13, 0xef, 0x19, 0x1e, 0xe2, 0x9,  0xfd, 0xf3, 0x14, 0xdd,
      0xda, 0x20, 0xd9, 0xf,  0xe3, 0xf9, 0xf7, 0xee, 0xe9, 0x24, 0xe6, 0x29,
      0x0,  0x7,  0x16, 0xe2, 0x1e, 0xd,  0x23, 0xd3, 0xdd, 0xf7, 0x14, 0xfa,
      0x8,  0x22, 0x26, 0x21, 0x9,  0x8,  0xf,  0xb,  0xe0, 0x12, 0xf4, 0x7f,
      0xdc, 0x58, 0xe5, 0x26,
  };
  const uint8_t buffer_6[] = {
      0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0xc2, 0xea, 0xff,
      0xff, 0x75, 0xea, 0xff, 0xff, 0xb8, 0xfa, 0xff, 0xff, 0x24, 0xfa,
      0xff, 0xff, 0xc8, 0xef, 0xff, 0xff, 0xac, 0xff, 0xff, 0xff, 0x44,
      0xd,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0xbd, 0x7,  0x0,  0x0,
      0x33, 0xea, 0xff, 0xff, 0x0,  0x0,  0x0,  0x0,  0xcc, 0xe4, 0xff,
      0xff, 0x4f, 0xd,  0x0,  0x0,  0xcf, 0xe3, 0xff, 0xff,
  };
  const uint8_t buffer_7[] = {
      0xf7, 0xca, 0x39, 0x47, 0x68, 0x73, 0x62, 0x63,
      0x40, 0xe6, 0x7f, 0x19, 0xae, 0x44, 0x5f, 0x56,
  };
  uint8_t buffer_8[16];
  uint8_t buffer_9[16];
  uint8_t buffer_10[1];

  const TfArray<2, int> dim_0 = {2, {1, 1}};
  const TfArray<1, int> dim_1 = {1, {1}};
  const TfArray<2, int> dim_2 = {2, {1, 16}};
  const TfArray<1, int> dim_3 = {1, {16}};
  const TfArray<2, int> dim_4 = {2, {16, 16}};
  const TfArray<1, int> dim_5 = {1, {16}};
  const TfArray<2, int> dim_6 = {2, {16, 1}};
  const TfArray<2, int> dim_7 = {2, {1, 16}};
  const TfArray<2, int> dim_8 = {2, {1, 16}};
  const TfArray<2, int> dim_9 = {2, {1, 1}};
  TfLiteTensor tflTensors[10] = {
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.024480115622282028, .zero_point = -128},
          .data = {.data = (void*)buffer_1},
          .dims = (TfLiteIntArray*)&dim_0,
          .bytes = 1,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteArenaRw,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.00019670200708787888, .zero_point = 0},
          .data = {.data = (void*)buffer_2},
          .dims = (TfLiteIntArray*)&dim_1,
          .bytes = 4,
          .type = kTfLiteInt32,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.015397093258798122, .zero_point = 0},
          .data = {.data = (void*)buffer_3},
          .dims = (TfLiteIntArray*)&dim_2,
          .bytes = 16,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.00014517262752633542, .zero_point = 0},
          .data = {.data = (void*)buffer_4},
          .dims = (TfLiteIntArray*)&dim_3,
          .bytes = 64,
          .type = kTfLiteInt32,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.010894655250012875, .zero_point = 0},
          .data = {.data = (void*)buffer_5},
          .dims = (TfLiteIntArray*)&dim_4,
          .bytes = 256,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 9.88754109130241e-05, .zero_point = 0},
          .data = {.data = (void*)buffer_6},
          .dims = (TfLiteIntArray*)&dim_5,
          .bytes = 64,
          .type = kTfLiteInt32,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.004039009101688862, .zero_point = 0},
          .data = {.data = (void*)buffer_7},
          .dims = (TfLiteIntArray*)&dim_6,
          .bytes = 16,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteMmapRo,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.01332512404769659, .zero_point = -128},
          .data = {.data = (void*)buffer_8},
          .dims = (TfLiteIntArray*)&dim_7,
          .bytes = 16,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteArenaRw,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.012775269336998463, .zero_point = -128},
          .data = {.data = (void*)buffer_9},
          .dims = (TfLiteIntArray*)&dim_8,
          .bytes = 16,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteArenaRw,
          .is_variable = 0,
      },
      {
          .quantization =
              {
                  .type = kTfLiteAffineQuantization,
              },
          .params = {.scale = 0.008290956728160381, .zero_point = 5},
          .data = {.data = (void*)buffer_10},
          .dims = (TfLiteIntArray*)&dim_9,
          .bytes = 1,
          .type = kTfLiteInt8,
          .allocation_type = kTfLiteArenaRw,
          .is_variable = 0,
      },
  };
  TfLiteEvalTensor tflEvalTensors[10] = {
      {
          .data = {.data = (void*)buffer_1},
          .dims = (TfLiteIntArray*)&dim_0,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_2},
          .dims = (TfLiteIntArray*)&dim_1,
          .type = kTfLiteInt32,
      },
      {
          .data = {.data = (void*)buffer_3},
          .dims = (TfLiteIntArray*)&dim_2,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_4},
          .dims = (TfLiteIntArray*)&dim_3,
          .type = kTfLiteInt32,
      },
      {
          .data = {.data = (void*)buffer_5},
          .dims = (TfLiteIntArray*)&dim_4,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_6},
          .dims = (TfLiteIntArray*)&dim_5,
          .type = kTfLiteInt32,
      },
      {
          .data = {.data = (void*)buffer_7},
          .dims = (TfLiteIntArray*)&dim_6,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_8},
          .dims = (TfLiteIntArray*)&dim_7,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_9},
          .dims = (TfLiteIntArray*)&dim_8,
          .type = kTfLiteInt8,
      },
      {
          .data = {.data = (void*)buffer_10},
          .dims = (TfLiteIntArray*)&dim_9,
          .type = kTfLiteInt8,
      },
  };

  int op_code_index[] = {0, 0, 0};
  const TfArray<3, int> input_0 = {3, {0, 6, 5}};
  const TfArray<1, int> output_0 = {1, {7}};
  const TfArray<3, int> input_1 = {3, {7, 4, 3}};
  const TfArray<1, int> output_1 = {1, {8}};
  const TfArray<3, int> input_2 = {3, {8, 2, 1}};
  const TfArray<1, int> output_2 = {1, {9}};
  const TfLiteFullyConnectedParams opdata_0 = {
      kTfLiteActRelu,
      kTfLiteFullyConnectedWeightsFormatDefault,
      false,
      false,
  };
  const TfLiteFullyConnectedParams opdata_1 = {
      kTfLiteActRelu,
      kTfLiteFullyConnectedWeightsFormatDefault,
      false,
      false,
  };
  const TfLiteFullyConnectedParams opdata_2 = {
      kTfLiteActNone,
      kTfLiteFullyConnectedWeightsFormatDefault,
      false,
      false,
  };
  TfLiteNode my_node[] = {
      {
          .inputs = (TfLiteIntArray*)&input_0,
          .outputs = (TfLiteIntArray*)&output_0,
          .builtin_data = (void*)&opdata_0,
          .custom_initial_data = nullptr,
          .custom_initial_data_size = 0,
      },
      {
          .inputs = (TfLiteIntArray*)&input_1,
          .outputs = (TfLiteIntArray*)&output_1,
          .builtin_data = (void*)&opdata_1,
          .custom_initial_data = nullptr,
          .custom_initial_data_size = 0,
      },
      {
          .inputs = (TfLiteIntArray*)&input_2,
          .outputs = (TfLiteIntArray*)&output_2,
          .builtin_data = (void*)&opdata_2,
          .custom_initial_data = nullptr,
          .custom_initial_data_size = 0,
      },
  };

}

static void*
AllocatePersistentBuffer(TfLiteContext* ctx, size_t bytes) {
  return std::malloc(bytes);
}

static TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node,
                                             int index) {
  return &tflTensors[index];
}

static TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node,
                                              int index) {
  return &tflTensors[index];
}

static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                               int tensor_idx) {
  std::cout << "index is" << tensor_idx << std::endl;
  return &tflTensors[tensor_idx];
}

static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                       int tensor_idx) {
  std::cout << "index2 is" << tensor_idx << std::endl;
  return &tflEvalTensors[tensor_idx];
}

void autogen_init(void) {
  TfLiteStatus status;
  op_init();
  class tflite::MyMicroContext my_context =
      tflite::MyMicroContext(NULL, NULL, NULL, tflTensors);
  std::cout << sizeof(my_context) << std::endl;
  static TfLiteContext context;
  context.impl_ = &my_context;
  context.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  // context.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  // context.GetScratchBuffer = &GetScratchBuffer;
  context.GetTensor = &GetTensor;
  context.GetEvalTensor = &GetEvalTensor;
  context.tensors = tflTensors;
  context.tensors_size = 3;

  for (int i = 0; i < 3; i++) {
    my_node[i].user_data = reg[op_code_index[i]].init(
        &context, (const char*)my_node[i].builtin_data, 0);
    status = reg[op_code_index[i]].prepare(&context, &my_node[i]);
  }
}

void autogen_run(void) {
  TfLiteStatus status;
  for (int i = 0; i < 3; i++) {
    status = reg[op_code_index[i]].invoke(&context, &my_node[i]);
    if (status != kTfLiteOk) {
      std::cout << "cannot call invoke" << std::endl;
      return;
    }
  }
}
};  // namespace