#include "model_runner.h"

#include <stdio.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const unsigned char nextkey_tflite[];
extern const unsigned int nextkey_tflite_len;

const tflite::Model *model = tflite::GetModel(nextkey_tflite);

int logTest(int x) {
  MicroPrintf("Printing using MicroPrintf");
  return 2 * x;
}
