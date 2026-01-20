#ifndef MODELRUNNER_H
#define MODELRUNNER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void model_setup();

int32_t run_inference(const char* input, const size_t len);

int logTest(int x);

#ifdef __cplusplus
}
#endif

#endif // MODELRUNNER_H
