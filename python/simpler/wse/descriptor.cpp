/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include <cstring>
#include "tensor.h"

extern "C" size_t wse_tensor_size() { return sizeof(Tensor); }

extern "C" void wse_pack_tensor(void *storage, void *address, uint32_t rows, uint32_t cols, uint32_t element_bytes) {
    const uint32_t shape[] = {rows, cols};
    const auto dtype = element_bytes == 2 ? DataType::FLOAT16 : DataType::FLOAT32;
    const auto tensor = simpler::hbg::make_tensor_external(address, shape, 2, dtype);
    std::memcpy(storage, &tensor, sizeof(tensor));
}
