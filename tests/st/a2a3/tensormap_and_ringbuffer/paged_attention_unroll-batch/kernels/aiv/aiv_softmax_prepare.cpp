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
// Batched Two-Pass Softmax Kernel (AIV) for batch_count × n_blocks tiles
//
// Processes batch_count batches in a single kernel invocation.
// For each batch b, performs two-pass softmax over n_blocks KV blocks:
//
//   Pass 1: Find global row max across all blocks for this batch.
//           For each block i, computes valid_len from context_lens[batch_start + b].
//           Masks invalid positions, scales, and accumulates global rowmax.
//
//   Pass 2: Compute exp(sij * scale - globalMax), convert to bf16,
//           store pij, and accumulate lij = rowsum across all blocks.
//
// Input layout:  sij tiles stored as batch_count × n_blocks contiguous (M, N) tiles
//                Tile (b, i) at: sij_base + (b * n_blocks + i) * M * N
// Output layout: pij tiles in same layout
//                mij/lij: (batch_count * M) contiguous scalars
//
// Supports two tile configurations via runtime dispatch:
//   Case1: M=16, N=128 (q_tile=16, block_size=128)
//   Case2: M=64, N=64  (q_tile=64, block_size=64)

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

template <int M, int N>
static __aicore__ void softmax_prepare_batch_unroll_impl(
    __gm__ Tensor *sij_batch, __gm__ Tensor *pij_batch, __gm__ Tensor *mij_batch, __gm__ Tensor *lij_batch,
    float scale_value, uint64_t context_lens_ptr, uint64_t batch_count, uint64_t n_blocks, uint64_t bn,
    uint64_t batch_start
) {
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_batch->buffer.addr);
    __gm__ bfloat16_t *pij_base = reinterpret_cast<__gm__ bfloat16_t *>(pij_batch->buffer.addr);
    __gm__ float *mij_base = reinterpret_cast<__gm__ float *>(mij_batch->buffer.addr);
    __gm__ float *lij_base = reinterpret_cast<__gm__ float *>(lij_batch->buffer.addr);
    __gm__ int32_t *ctx_lens = reinterpret_cast<__gm__ int32_t *>(context_lens_ptr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = M / kScalarCols;

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;

    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;
    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;
    using TileScalarRow = Tile<TileType::Vec, float, 1, M, BLayout::RowMajor, 1, M>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;

    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    TileVecMxN sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileVecMxN sumAccTile;
    TileScalarDN localMaxDN;
    TileScalarDN globalMaxDN;
    TileScalarDN sumDN;
    TileVecMxN_bf16 pijBf16Tile;
    TileScalarRow localMaxRow;
    TileScalarRow globalMaxRow;
    TileScalarND globalMaxND;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, kDataBytes);
    TASSIGN(tmpTile, 2 * kDataBytes);
    TASSIGN(sumAccTile, 3 * kDataBytes);
    int scalarBase = 4 * kDataBytes;
    TASSIGN(localMaxDN, scalarBase);
    TASSIGN(localMaxRow, scalarBase);
    TASSIGN(globalMaxDN, scalarBase + kScalarDNBytes);
    TASSIGN(globalMaxRow, scalarBase + kScalarDNBytes);
    TASSIGN(globalMaxND, scalarBase + kScalarDNBytes);
    TASSIGN(sumDN, scalarBase + 2 * kScalarDNBytes);
    TASSIGN(pijBf16Tile, scalarBase + 3 * kScalarDNBytes);

    for (uint64_t b = 0; b < batch_count; b++) {
        int32_t cur_seq = ctx_lens[batch_start + b];
        __gm__ float *mij_addr = mij_base + b * M;
        __gm__ float *lij_addr = lij_base + b * M;
        GlobalScalarND mijGlobalND(mij_addr);
        GlobalScalarDN lijGlobalDN(lij_addr);

        // Count valid blocks for this batch in the current group
        uint64_t valid_block_count = 0;
        for (uint64_t i = 0; i < n_blocks; i++) {
            uint64_t block_start = (bn + i) * N;
            if (block_start < (uint64_t)cur_seq) valid_block_count = i + 1;
        }

        if (valid_block_count == 0) {
            // All blocks beyond sequence: write mij=-1e30, lij=0, pij=0
            constexpr float NEG_LARGE = -1e30f;
            for (int r = 0; r < kAlignedRows; r++) {
                localMaxDN.SetValue(r, NEG_LARGE);
                sumDN.SetValue(r, 0.0f);
            }
            for (int e = 0; e < M * N; e++) {
                pijBf16Tile.SetValue(e, static_cast<bfloat16_t>(0.0f));
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(mijGlobalND, localMaxDN);
            TSTORE(lijGlobalDN, sumDN);
            for (uint64_t i = 0; i < n_blocks; i++) {
                GlobalDataMxN_bf16 pijGlobal(pij_base + (b * n_blocks + i) * M * N);
                TSTORE(pijGlobal, pijBf16Tile);
            }
            if (b + 1 < batch_count) {
                pipe_barrier(PIPE_ALL);
            }
            continue;
        }

        // ======== Pass 1: Find global row max across valid blocks ========
        for (uint64_t i = 0; i < valid_block_count; i++) {
            uint64_t block_start_pos = (bn + i) * N;
            uint64_t valid_len = N;
            if (block_start_pos + N > (uint64_t)cur_seq) {
                valid_len = (uint64_t)cur_seq - block_start_pos;
            }

            GlobalDataMxN sijGlobal(sij_base + (b * n_blocks + i) * M * N);
            TLOAD(sijTile, sijGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (valid_len < static_cast<uint64_t>(N)) {
                TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
                TASSIGN(sijDynTile, 0x0);
                TFILLPAD_INPLACE(sijPadTile, sijDynTile);
                pipe_barrier(PIPE_V);
            }

            TMULS(sijTile, sijTile, scale_value);
            pipe_barrier(PIPE_V);
            TROWMAX(localMaxDN, sijTile, tmpTile);
            pipe_barrier(PIPE_V);

            TRESHAPE(localMaxRow, localMaxDN);
            if (i == 0) {
                TMAX(globalMaxRow, localMaxRow, localMaxRow);
            } else {
                TMAX(globalMaxRow, globalMaxRow, localMaxRow);
            }
            pipe_barrier(PIPE_V);
        }

        // TRESHAPE back for Pass 2
        TRESHAPE(globalMaxDN, globalMaxRow);

        // Store mij
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(mijGlobalND, globalMaxND);

        // ======== Pass 2: Compute exp, pij, lij ========
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        bool sum_initialized = false;
        for (uint64_t i = 0; i < n_blocks; i++) {
            uint64_t block_start_pos = (bn + i) * N;
            GlobalDataMxN_bf16 pijGlobal(pij_base + (b * n_blocks + i) * M * N);

            if (block_start_pos >= (uint64_t)cur_seq) {
                // Block beyond sequence: write zero pij
                for (int e = 0; e < M * N; e++) {
                    pijBf16Tile.SetValue(e, static_cast<bfloat16_t>(0.0f));
                }
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(pijGlobal, pijBf16Tile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                continue;
            }

            uint64_t valid_len = N;
            if (block_start_pos + N > (uint64_t)cur_seq) {
                valid_len = (uint64_t)cur_seq - block_start_pos;
            }

            GlobalDataMxN sijGlobal(sij_base + (b * n_blocks + i) * M * N);
            TLOAD(sijTile, sijGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (valid_len < static_cast<uint64_t>(N)) {
                TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
                TASSIGN(sijDynTile, 0x0);
                TFILLPAD_INPLACE(sijPadTile, sijDynTile);
                pipe_barrier(PIPE_V);
            }

            TMULS(sijTile, sijTile, scale_value);
            pipe_barrier(PIPE_V);
            TROWEXPANDSUB(pijTile, sijTile, globalMaxDN);
            pipe_barrier(PIPE_V);
            TEXP(pijTile, pijTile);
            pipe_barrier(PIPE_V);
            TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
            pipe_barrier(PIPE_V);
            TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);

            pipe_barrier(PIPE_V);
            if (!sum_initialized) {
                TMULS(sumAccTile, pijTile, 1.0f);
                sum_initialized = true;
            } else {
                TADD(sumAccTile, sumAccTile, pijTile);
            }

            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(pijGlobal, pijBf16Tile);

            if (i + 1 < n_blocks) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }

        // Compute final row sum and store lij
        pipe_barrier(PIPE_V);
        TROWSUM(sumDN, sumAccTile, tmpTile);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(lijGlobalDN, sumDN);

        if (b + 1 < batch_count) {
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *sij_batch = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *pij_batch = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *mij_batch = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *lij_batch = reinterpret_cast<__gm__ Tensor *>(args[3]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[4]);
    float scale_value = scale_conv.f;
    uint64_t context_lens_ptr = static_cast<uint64_t>(args[5]);
    uint64_t batch_count = static_cast<uint64_t>(args[6]);
    uint64_t n_blocks = static_cast<uint64_t>(args[7]);
    uint64_t bn = static_cast<uint64_t>(args[8]);
    uint64_t batch_start = static_cast<uint64_t>(args[9]);

    uint64_t q_tile_size = static_cast<uint64_t>(sij_batch->shapes[0] / batch_count);

    if (q_tile_size == 16) {
        softmax_prepare_batch_unroll_impl<16, 128>(
            sij_batch, pij_batch, mij_batch, lij_batch, scale_value, context_lens_ptr, batch_count, n_blocks, bn,
            batch_start
        );
    } else {
        softmax_prepare_batch_unroll_impl<64, 64>(
            sij_batch, pij_batch, mij_batch, lij_batch, scale_value, context_lens_ptr, batch_count, n_blocks, bn,
            batch_start
        );
    }
}
