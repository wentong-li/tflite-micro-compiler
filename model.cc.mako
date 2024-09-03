#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "my_micro_context.h"

namespace {
template <int SZ, class T>
struct TfArray {
  int sz;
  T elem[SZ];


${kernel_region}

${buffer_region}

${tensor_region}

${node_region}

}

static void* AllocatePersistentBuffer(TfLiteContext *ctx, size_t bytes) {
		return std::malloc(bytes);
}

static TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node, int index) {
	return &tflTensors[index];
}

static TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node, int index) {
	return &tflTensors[index];
}

static TfLiteTensor* GetTensor(const struct TfLiteContext* context,
                               int tensor_idx) {
	std::cout << "index is" << tensor_idx <<std::endl;
  return &tflTensors[tensor_idx];
}

static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                       int tensor_idx) {
	std::cout << "index2 is" << tensor_idx <<std::endl;
  return &tflEvalTensors[tensor_idx];
}

void autogen_init(void) {
	TfLiteStatus status;
	op_init();
	class tflite::MyMicroContext my_context = tflite::MyMicroContext(NULL, NULL, NULL, tflTensors);
	std::cout << sizeof(my_context) <<std::endl;
	static TfLiteContext context;
	context.impl_ = &my_context;
	context.AllocatePersistentBuffer = &AllocatePersistentBuffer;
	// context.RequestScratchBufferInArena = &RequestScratchBufferInArena;
	// context.GetScratchBuffer = &GetScratchBuffer;
	context.GetTensor = &GetTensor;
	context.GetEvalTensor = &GetEvalTensor;
	context.tensors = tflTensors;
	context.tensors_size = ${node_count};


	for (int i = 0; i < ${node_count}; i++) {
		my_node[i].user_data = reg[op_code_index[i]].init(&context, (const char*)my_node[i].builtin_data, 0);
		status = reg[op_code_index[i]].prepare(&context, &my_node[i]);
	}

}

void autogen_run(void){
	TfLiteStatus status;
	for (int i = 0; i < ${node_count}; i++) {
		status = reg[op_code_index[i]].invoke(&context, &my_node[i]);
		if (status != kTfLiteOk) {
			std::cout << "cannot call invoke" << std::endl;
			return;
		}
	}
}
};