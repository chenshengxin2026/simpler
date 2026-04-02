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
// Batched SplitK PV Matmul Kernel: for each batch b, accumulated P @ V across n_blocks
//
// Processes batch_count batches × n_blocks blocks using SplitK accumulation:
//   Block 0: TMATMUL(C, A, B)       — initialize accumulator
//   Block i: TMATMUL_ACC(C, C, A, B) — accumulate into same C
//
// Per-batch output: oi_new(M, N) fp32 = sum of P_i @ V_i across all blocks.
// Input pij layout: batch_count × n_blocks contiguous (M, K) tiles in batch-major order.
// Per-block vj addresses: value_cache base + block_table lookup.
//
// Optimizations:
//   - Double-buffered L1 tiles (ping/pong for A and B via MTE2)
//   - Double-buffered L0 tiles (ping/pong for L0A and L0B via MTE1)
//   - TLOAD(next) overlaps with TMATMUL(current) via MTE2/M-pipe parallelism
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128) -> (16, 128)
//   Case2: (64,  64) @ ( 64, 128) -> (64, 128)
//
// Template: M=q_tile, K=block_size, N=head_dim

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
static __aicore__ void pv_matmul_batch_unroll_impl(
    __gm__ Tensor *pij_batch, __gm__ Tensor *value_cache, __gm__ Tensor *oi_new_batch, uint64_t block_table_ptr,
    uint64_t batch_count, uint64_t n_blocks, uint64_t bn, uint64_t block_num, uint64_t batch_start
) {
    __gm__ bfloat16_t *pij_base = reinterpret_cast<__gm__ bfloat16_t *>(pij_batch->buffer.addr);
    __gm__ bfloat16_t *val_base = reinterpret_cast<__gm__ bfloat16_t *>(value_cache->buffer.addr);
    __gm__ float *oi_base = reinterpret_cast<__gm__ float *>(oi_new_batch->buffer.addr);
    __gm__ int32_t *bt = reinterpret_cast<__gm__ int32_t *>(block_table_ptr);

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    constexpr int kATileBytes = M * K * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kBTileBytes = K * N * static_cast<int>(sizeof(bfloat16_t));

    TileMatA aMatTile[2];
    TileMatB bMatTile[2];
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], kATileBytes);
    TASSIGN(bMatTile[0], 2 * kATileBytes);
    TASSIGN(bMatTile[1], 2 * kATileBytes + kBTileBytes);

    LeftTile aTile[2];
    RightTile bTile[2];
    AccTile cTile;
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], kATileBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], kBTileBytes);
    TASSIGN(cTile, 0x0);

    for (uint64_t b = 0; b < batch_count; b++) {
        GlobalOut oiGlobal(oi_base + b * M * N);

        // Seed reverse-dependency flags
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        for (uint64_t i = 0; i < n_blocks; i++) {
            int cur = static_cast<int>(i % 2);
            __gm__ bfloat16_t *pij_addr = pij_base + (b * n_blocks + i) * M * K;
            int32_t phys_block = bt[(batch_start + b) * block_num + (bn + i)];
            __gm__ bfloat16_t *vj_addr = val_base + (uint64_t)phys_block * K * N;

            GlobalA pijGlobal(pij_addr);
            GlobalB vjGlobal(vj_addr);

            // Stage 1: TLOAD (MTE2: GM → L1[cur])
            wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);
            TLOAD(aMatTile[cur], pijGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            TLOAD(bMatTile[cur], vjGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

            // Stage 2: TMOV (MTE1: L1[cur] → L0[cur])
            wait_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            TMOV(aTile[cur], aMatTile[cur]);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
            TMOV(bTile[cur], bMatTile[cur]);
            set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);

            // Stage 3: TMATMUL (M-pipe: L0A[cur] × L0B[cur] → L0C)
            set_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
            wait_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
            if (i == 0) {
                TMATMUL(cTile, aTile[cur], bTile[cur]);
            } else {
                TMATMUL_ACC(cTile, cTile, aTile[cur], bTile[cur]);
            }
            set_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
        }

        // Drain outstanding reverse-dependency flags
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE(oiGlobal, cTile);

        if (b + 1 < batch_count) {
            set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *pij_batch = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *value_cache = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *oi_new_batch = reinterpret_cast<__gm__ Tensor *>(args[2]);
    uint64_t block_table_ptr = static_cast<uint64_t>(args[3]);
    uint64_t batch_count = static_cast<uint64_t>(args[4]);
    uint64_t n_blocks = static_cast<uint64_t>(args[5]);
    uint64_t bn = static_cast<uint64_t>(args[6]);
    uint64_t block_num = static_cast<uint64_t>(args[7]);
    uint64_t batch_start = static_cast<uint64_t>(args[8]);

    uint64_t q_tile_size = static_cast<uint64_t>(oi_new_batch->shapes[0] / batch_count);

    if (q_tile_size == 16) {
        pv_matmul_batch_unroll_impl<16, 128, 128>(
            pij_batch, value_cache, oi_new_batch, block_table_ptr, batch_count, n_blocks, bn, block_num, batch_start
        );
    } else {
        pv_matmul_batch_unroll_impl<64, 64, 128>(
            pij_batch, value_cache, oi_new_batch, block_table_ptr, batch_count, n_blocks, bn, block_num, batch_start
        );
    }
}
