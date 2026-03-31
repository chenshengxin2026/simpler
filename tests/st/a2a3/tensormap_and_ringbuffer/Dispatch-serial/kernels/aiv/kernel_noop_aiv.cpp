/**
 * No-op AIV Kernel for Dispatch Throughput
 *
 * Minimal vector kernel that writes a single scalar to prove execution.
 * The kernel reads the current accumulated value, adds 1.0, and writes back.
 * With N tasks, the final output should be N.0.
 *
 * Args:
 *   args[0] = output tensor (INOUT) - single float32 element
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
    *out = *out + 1.0f;
}
