/**
 * No-op AIC Kernel for Task Scaling
 *
 * Minimal cube kernel that performs a trivial write. Each task writes 1.0
 * at its designated position in the output tensor, proving execution order.
 *
 * Args:
 *   args[0] = output tensor (INOUT) - single float32 element per task
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    *out = 1.0f;
}
