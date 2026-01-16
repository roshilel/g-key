#include <stdio.h>

#include "model_runner.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// global vars
static tflite::MicroInterpreter *interpreter_ptr = nullptr;
static TfLiteTensor *input_tensor_ptr =
    nullptr; // pointer to model's input tensor

using NextKeyOpsResolver = tflite::MicroMutableOpResolver<20>;

// operations for LSTM
static TfLiteStatus RegisterOps(NextKeyOpsResolver &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
  TF_LITE_ENSURE_STATUS(op_resolver.AddUnidirectionalSequenceLSTM());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

void model_setup() {

  // reading model byte array
  extern const unsigned char nextkey_tflite[];
  extern const unsigned int nextkey_tflite_len;
  const tflite::Model *model = tflite::GetModel(nextkey_tflite);

  // adding operations required for model
  NextKeyOpsResolver op_resolver;

  TfLiteStatus status = RegisterOps(op_resolver);
  if (status != kTfLiteOk) {
    MicroPrintf("Op registration failed");
    return;
  }

  // defining mem needed
  // constexpr: no need to give this var mem address
  constexpr uint16_t tensor_arena_size = 4 * 1024; // 4MB
  uint8_t tensor_arena[tensor_arena_size];

  // initializing interpreter
  static tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                              tensor_arena_size);
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
}

int logTest(int x) {
  MicroPrintf("Printing using MicroPrintf");
  return 2 * x;
}
