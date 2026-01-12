#include <stdio.h>

#include "model_runner.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

using NextKeyOpsResolver = tflite::MicroMutableOpResolver<4>;

static TfLiteStatus RegisterOps(NextKeyOpsResolver &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
  TF_LITE_ENSURE_STATUS(op_resolver.AddUnidirectionalSequenceLSTM());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

tflite::MicroInterpreter *interpreter_ptr = nullptr;

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
  constexpr uint16_t tensor_arena_size = 4 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // initializing interpreter
  static tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                              tensor_arena_size);

  interpreter_ptr = &interpreter;

  // allocating memory
  interpreter.AllocateTensors();

  // pointer to model's input tnesor
  TfLiteTensor *input = interpreter.input(0);
}

int logTest(int x) {
  MicroPrintf("Printing using MicroPrintf");
  return 2 * x;
}
