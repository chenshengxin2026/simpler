/**
 * Increment AIV Kernel for Graph Topology Tests
 *
 * Minimal vector kernel: reads one input scalar, writes output = input + 1.0.
 * Used to build DAG chains with verifiable accumulated values.
 *
 * Args:
 *   args[0] = input tensor  (INPUT)  - single float32 element
 *   args[1] = output tensor (OUTPUT/INOUT) - single float32 element
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
    __gm__ Tensor* in_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ float* in_ptr = reinterpret_cast<__gm__ float*>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float* out_ptr = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    *out_ptr = *in_ptr + 1.0f;
}
