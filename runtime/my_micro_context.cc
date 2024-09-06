#include "my_micro_context.h"
#include <cstdlib>
namespace tflite {
	MyMicroContext::MyMicroContext(MicroAllocator* allocator, const Model* model, MicroInterpreterGraph* graph, TfLiteTensor* tensor)
	: allocator_(*allocator),
      graph_(*graph),
      model_(model),
      state_(InterpreterState::kInit),
      tensor_(tensor) {
		// this->allocator_ = *allocator;
		// this->model_ = model;
		// this->graph_ = *graph;
	}
	MyMicroContext::~MyMicroContext() {}

	void* MyMicroContext::AllocatePersistentBuffer(size_t bytes) {
		return std::malloc(bytes);
	}

	TfLiteStatus MyMicroContext::RequestScratchBufferInArena(
	size_t bytes, int* buffer_idx) {
		return kTfLiteOk;
	}

void* MyMicroContext::GetScratchBuffer(int buffer_idx) {
  return NULL;
}

TfLiteTensor* MyMicroContext::AllocateTempTfLiteTensor(
    int tensor_idx) {
  return &tensor_[tensor_idx];
}

void MyMicroContext::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  return;
}

uint8_t* MyMicroContext::AllocateTempBuffer(size_t size,
                                                     size_t alignment) {

  return NULL;
}

void MyMicroContext::DeallocateTempBuffer(uint8_t* buffer) {

}

TfLiteEvalTensor* MyMicroContext::GetEvalTensor(int tensor_idx) {
  return NULL;
}

void MyMicroContext::SetScratchBufferHandles(
    ScratchBufferHandle* scratch_buffer_handles) {
}

TfLiteStatus MyMicroContext::set_external_context(
    void* external_context_payload) {


  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

void MyMicroContext::SetInterpreterState(InterpreterState state) {
  state_ = state;
}

MyMicroContext::InterpreterState
MyMicroContext::GetInterpreterState() const {
  return state_;
}
}
