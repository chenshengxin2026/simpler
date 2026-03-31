/**
 * No-op AIV Kernel for Graph Topology Tests
 *
 * Minimal vector kernel that increments first arg by 1.0 (INOUT pattern).
 * Additional args beyond args[0] are ignored by the kernel but create
 * runtime dependencies for barrier/merge tasks in fan-in and diamond topologies.
 *
 * Args:
 *   args[0] = output tensor (INOUT) - single float32 element
 *   args[1..N] = dependency inputs (INPUT, ignored by kernel)
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
