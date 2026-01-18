#include <stdio.h>

#include "esp_heap_caps.h"
#include "model_runner.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// global vars
static tflite::MicroInterpreter *interpreter_ptr = nullptr;
static TfLiteTensor *input_tensor_ptr =
    nullptr; // pointer to model's input tensor

using NextKeyOpsResolver = tflite::MicroMutableOpResolver<20>;

// operations for Conv1D
static TfLiteStatus RegisterOps(NextKeyOpsResolver &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLess());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  return kTfLiteOk;
}

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t *tensor_arena = nullptr;

void model_setup() {

  // reading model byte array
  extern const unsigned char conv1d_nonhybrid_tflite[];
  const tflite::Model *model = tflite::GetModel(conv1d_nonhybrid_tflite);

  // adding operations required for model
  static NextKeyOpsResolver op_resolver;
  TfLiteStatus status = RegisterOps(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("Op registration failed");
    return;
  }

  // allocate in to PSRAM 
  MicroPrintf("Attempting to allocate %d bytes in PSRAM...", kTensorArenaSize);
  tensor_arena = (uint8_t *)heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  // fallback to heap
  if (tensor_arena == nullptr) {
    MicroPrintf("PSRAM allocation failed. Falling back to heap.");
    tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
  }

  if (tensor_arena == nullptr) {
    MicroPrintf("Failed to allocate tensor arena");
    return;
  }

  // initializing interpreter
  static tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                              kTensorArenaSize);
  // allocating memory
  interpreter.AllocateTensors();

  // setting globals
  interpreter_ptr = &interpreter;

  // checking input
  TfLiteTensor *temp_input_tensor_ptr = interpreter.input(0);

  if (temp_input_tensor_ptr == nullptr) {
    MicroPrintf("Error: input tensor is null\n");
    return;
  }

  int *input_dim_ptr = &temp_input_tensor_ptr->dims->size;
  int *input_shape_ptr = temp_input_tensor_ptr->dims->data;
  TfLiteType *input_type_ptr = &temp_input_tensor_ptr->type;

  if (*input_dim_ptr != 2) {
    MicroPrintf("Error: dim should be 2 but got %d\n", *input_dim_ptr);
    return;
  }
  if (input_shape_ptr[0] != 1 || input_shape_ptr[1] != 40) {
    MicroPrintf("Error: shape should be [1, 40] but got [%d, %d]",
                input_shape_ptr[0], input_shape_ptr[1]);
    return;
  }
  if (*input_type_ptr != kTfLiteInt32) {
    MicroPrintf("Error: type should be kTfLiteInt32");
    return;
  }

  input_tensor_ptr = temp_input_tensor_ptr;

  for (int i = 0; i < 40; i++) {
    input_tensor_ptr->data.i32[i] = 0; 
  }

  TfLiteStatus invoke_status = interpreter_ptr->Invoke();

  if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed");
  } else {
      MicroPrintf("Invoked successfully");
  }

}

int logTest(int x) {
  MicroPrintf("Printing using MicroPrintf");
  return 2 * x;
}
