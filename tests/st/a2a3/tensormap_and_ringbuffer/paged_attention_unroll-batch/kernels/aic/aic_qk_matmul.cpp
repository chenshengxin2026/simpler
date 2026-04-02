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
// Batched QK Matmul Kernel: for each batch b, for each block i:
//   qi(M, K) @ kj[i].T(K, N) -> sij(M, N)
//
// Processes batch_count batches × n_blocks blocks in a single kernel invocation.
// Per-batch qi is hoisted (constant across all blocks for the same batch).
// Per-batch per-block kj addresses computed from key_cache base + block_table lookup.
//
// Output layout: batch_count × n_blocks contiguous (M, N) tiles in batch-major order.
// Tile (b, i) at address: sij_base + (b * n_blocks + i) * M * N
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// Template: M=q_tile, K=head_dim, N=block_size

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

template <int M, int K, int N>
static __aicore__ void qk_matmul_batch_unroll_impl(
    __gm__ Tensor *query, __gm__ Tensor *key_cache, __gm__ Tensor *sij_out, uint64_t block_table_ptr,
    uint64_t batch_count, uint64_t n_blocks, uint64_t bn, uint64_t q_offset, uint64_t block_num, uint64_t num_heads,
    uint64_t batch_start
) {
    __gm__ bfloat16_t *query_base = reinterpret_cast<__gm__ bfloat16_t *>(query->buffer.addr);
    __gm__ bfloat16_t *key_base = reinterpret_cast<__gm__ bfloat16_t *>(key_cache->buffer.addr);
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_out->buffer.addr);
    __gm__ int32_t *bt = reinterpret_cast<__gm__ int32_t *>(block_table_ptr);

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    for (uint64_t b = 0; b < batch_count; b++) {
        // Hoist qi TLOAD per batch (constant across all blocks)
        __gm__ bfloat16_t *qi_addr = query_base + ((batch_start + b) * num_heads + q_offset) * K;
        GlobalA qiGlobal(qi_addr);
        TLOAD(aMatTile, qiGlobal);

        for (uint64_t i = 0; i < n_blocks; i++) {
            int32_t phys_block = bt[(batch_start + b) * block_num + (bn + i)];
            __gm__ bfloat16_t *kj_addr = key_base + (uint64_t)phys_block * N * K;
            __gm__ float *sij_addr = sij_base + (b * n_blocks + i) * M * N;

            GlobalB kjGlobal(kj_addr);
            GlobalOut sijGlobal(sij_addr);

            TLOAD(bMatTile, kjGlobal);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);

            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            TMATMUL(cTile, aTile, bTile);

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            TSTORE(sijGlobal, cTile);

            if (i + 1 < n_blocks || b + 1 < batch_count) {
                pipe_barrier(PIPE_ALL);
            }
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *query = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *sij_out = reinterpret_cast<__gm__ Tensor *>(args[2]);
    uint64_t block_table_ptr = static_cast<uint64_t>(args[3]);
    uint64_t batch_count = static_cast<uint64_t>(args[4]);
    uint64_t n_blocks = static_cast<uint64_t>(args[5]);
    uint64_t bn = static_cast<uint64_t>(args[6]);
    uint64_t q_offset = static_cast<uint64_t>(args[7]);
    uint64_t block_num = static_cast<uint64_t>(args[8]);
    uint64_t num_heads = static_cast<uint64_t>(args[9]);
    uint64_t batch_start = static_cast<uint64_t>(args[10]);

    uint64_t q_tile_size = static_cast<uint64_t>(sij_out->shapes[0] / batch_count);

    if (q_tile_size == 16) {
        qk_matmul_batch_unroll_impl<16, 128, 128>(
            query, key_cache, sij_out, block_table_ptr, batch_count, n_blocks, bn, q_offset, block_num, num_heads,
            batch_start
        );
    } else {
        qk_matmul_batch_unroll_impl<64, 128, 64>(
            query, key_cache, sij_out, block_table_ptr, batch_count, n_blocks, bn, q_offset, block_num, num_heads,
            batch_start
        );
    }
}
