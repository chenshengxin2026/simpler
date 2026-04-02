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
/**
 * Paged Attention Orchestration - Chunked Batch + Block-Unroll Architecture
 *
 * Combines two tiling strategies for maximum task reduction:
 *   - IN_CORE_BATCH: number of batch items per kernel invocation
 *   - N_UNROLL: number of KV blocks grouped per kernel invocation
 *
 * The full batch is split into chunks of IN_CORE_BATCH size. Within each
 * chunk, blocks are grouped into N_UNROLL-sized groups. Each group submits
 * exactly 4 tasks (QK, SF, PV, UP), where each kernel processes
 * batch_count items across n_blocks blocks.
 *
 * Task count = num_chunks * (1 + ceil(max_bn / N_UNROLL) * 4)
 *
 * Configuration:
 *   IN_CORE_BATCH controls the number of batch items per kernel invocation.
 *   N_UNROLL controls the number of KV blocks grouped per kernel invocation.
 *
 * Memory Layout:
 *   Query: (batch * num_heads, head_dim) bf16
 *   Key:   (total_blocks, block_size, head_dim) bf16 (stored as K^T for QK)
 *   Value: (total_blocks, block_size, head_dim) bf16
 *
 * Per-group intermediate tensors:
 *   sij:     (chunk_bc * q_tile, n_blocks * block_size)  fp32 — batch_count × n_blocks contiguous (M, N) tiles
 *   pij:     (chunk_bc * q_tile, n_blocks * block_size)  bf16
 *   mij/lij: (chunk_bc * q_tile)                         fp32 — group-level per-batch
 *   oi_new:  (chunk_bc * q_tile, head_dim)               fp32 — SplitK accumulated per-batch
 *
 * Per-chunk accumulator tensors (persist across block groups):
 *   oi:      (chunk_bc * q_tile, head_dim)    fp32
 *   mi/li:   (chunk_bc * q_tile)              fp32
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define IN_CORE_BATCH 4
#define N_UNROLL 64

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    // Read dimensions from tensor metadata
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;

    LOG_INFO(
        "paged_attention_unroll_batch: batch=%" PRIu64 ", num_heads=%" PRIu64 ", IN_CORE_BATCH=%d, N_UNROLL=%d", batch,
        num_heads, IN_CORE_BATCH, N_UNROLL
    );

    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    int *host_block_table = orch_args.tensor(3).data_as<int>();
    int *host_context_lens = orch_args.tensor(4).data_as<int>();

    // Compute max block count across all batches
    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint64_t total_blocks_count = orch_args.tensor(1).shapes[0];
    uint64_t kv_total_rows = total_blocks_count * block_size;

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};

    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32, true);

    uint64_t bt_addr = reinterpret_cast<uintptr_t>(host_block_table);
    uint64_t cl_addr = reinterpret_cast<uintptr_t>(host_context_lens);

    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t chunk_idx = orch_thread_index; chunk_idx < num_chunks; chunk_idx += orch_thread_num) {
            uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;
            uint64_t batch_start = chunk_idx * IN_CORE_BATCH;

            PTO2_SCOPE() {
                // Accumulators persist across the block-group loop for the entire chunk
                uint32_t oi_acc_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t scalar_acc_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                TensorCreateInfo oi_batch_ci(oi_acc_shapes, 2, DataType::FLOAT32);
                TensorCreateInfo scalar_acc_ci(scalar_acc_shapes, 1, DataType::FLOAT32);

                Arg params_hub;
                params_hub.add_output(oi_batch_ci);
                params_hub.add_output(scalar_acc_ci);
                params_hub.add_output(scalar_acc_ci);
                TaskOutputTensors hub_outs = pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_hub);
                const Tensor &oi_batch = hub_outs.get_ref(0);
                const Tensor &li_batch = hub_outs.get_ref(1);
                const Tensor &mi_batch = hub_outs.get_ref(2);

                // Per-group oi_new shape is loop-invariant (hoist out of bn loop)
                uint32_t oi_new_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                TensorCreateInfo oi_new_ci(oi_new_shapes, 2, DataType::FLOAT32);
                uint32_t vec_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                TensorCreateInfo vec_ci(vec_shapes, 1, DataType::FLOAT32);

                for (uint64_t bn = 0; bn < max_bn; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min(static_cast<uint64_t>(N_UNROLL), max_bn - bn);

                    PTO2_SCOPE() {
                        // sij/pij shapes depend on n_blocks (varies for last group)
                        uint32_t sij_shapes[2] = {
                            static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(n_blocks * block_size)
                        };
                        TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
                        TensorCreateInfo pij_ci(sij_shapes, 2, data_type);

                        // === Task 1: Batched QK matmul (batch_count × n_blocks) ===
                        Arg params_qk;
                        params_qk.add_input(query);
                        params_qk.add_input(key_cache);
                        params_qk.add_output(sij_ci);
                        params_qk.add_scalar(bt_addr);
                        params_qk.add_scalar(chunk_bc);
                        params_qk.add_scalar(n_blocks);
                        params_qk.add_scalar(bn);
                        params_qk.add_scalar(q_offset);
                        params_qk.add_scalar(block_num);
                        params_qk.add_scalar(num_heads);
                        params_qk.add_scalar(batch_start);
                        TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                        const Tensor &sij_b = qk_outs.get_ref(0);

                        // === Task 2: Per-batch two-pass softmax over n_blocks ===
                        Arg params_sf;
                        params_sf.add_input(sij_b);
                        params_sf.add_output(pij_ci);
                        params_sf.add_output(vec_ci);
                        params_sf.add_output(vec_ci);
                        params_sf.add_scalar(scale_value);
                        params_sf.add_scalar(cl_addr);
                        params_sf.add_scalar(chunk_bc);
                        params_sf.add_scalar(n_blocks);
                        params_sf.add_scalar(bn);
                        params_sf.add_scalar(batch_start);
                        TaskOutputTensors sf_outs = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                        const Tensor &pij_b = sf_outs.get_ref(0);
                        const Tensor &mij_b = sf_outs.get_ref(1);
                        const Tensor &lij_b = sf_outs.get_ref(2);

                        // === Task 3: Per-batch SplitK PV matmul ===
                        Arg params_pv;
                        params_pv.add_input(pij_b);
                        params_pv.add_input(value_cache);
                        params_pv.add_output(oi_new_ci);
                        params_pv.add_scalar(bt_addr);
                        params_pv.add_scalar(chunk_bc);
                        params_pv.add_scalar(n_blocks);
                        params_pv.add_scalar(bn);
                        params_pv.add_scalar(block_num);
                        params_pv.add_scalar(batch_start);
                        TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                        const Tensor &oi_new_b = pv_outs.get_ref(0);

                        // === Task 4: Batched online update (unchanged interface) ===
                        uint64_t is_first = (bn == 0) ? 1 : 0;
                        uint64_t is_last = (bn + n_blocks >= max_bn) ? 1 : 0;
                        Arg params_up;
                        params_up.add_input(mij_b);
                        params_up.add_input(lij_b);
                        params_up.add_input(oi_new_b);
                        params_up.add_inout(mi_batch);
                        params_up.add_inout(li_batch);
                        params_up.add_inout(oi_batch);
                        params_up.add_inout(out);
                        params_up.add_scalar(is_first);
                        params_up.add_scalar(is_last);
                        params_up.add_scalar(chunk_bc);
                        params_up.add_scalar(q_offset);
                        params_up.add_scalar(num_heads);
                        params_up.add_scalar(batch_start);
                        pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                    }
                }
            }
        }
    }

    uint64_t num_groups = (max_bn + N_UNROLL - 1) / N_UNROLL;
    LOG_INFO(
        "paged_attention_unroll_batch: %" PRIu64 " tasks (batch=%" PRIu64 ", max_bn=%" PRIu64 ", chunks=%" PRIu64
        ", groups=%" PRIu64 ", IN_CORE_BATCH=%d, N_UNROLL=%d)",
        static_cast<uint64_t>(num_chunks * (1 + num_groups * 4)), batch, max_bn, num_chunks, num_groups, IN_CORE_BATCH,
        N_UNROLL
    );
}

}  // extern "C"
