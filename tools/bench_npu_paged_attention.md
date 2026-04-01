# bench_npu_paged_attention.py

Benchmark tool for measuring the latency of the AscendC-based paged attention operator (`torch_npu._npu_paged_attention`) on Ascend NPUs.

## Overview

This script measures the wall-clock latency of the paged attention operator provided by `torch_npu`, which is internally implemented using AscendC kernels and scheduled by the ATB (Ascend Transformer Boost) runtime. The benchmark generates random input tensors matching production model shapes, runs warmup iterations, then performs timed measurement rounds with `torch.npu.synchronize()` barriers to ensure accurate device-side timing.

**Output metrics**: Mean, Trimmed Mean (drops top/bottom 10%), Std, Min, P50, P90, P99, Max — all in microseconds.

---

## Prerequisites

### 1. Hardware

An Ascend NPU device (Atlas 800I A2 / Atlas A2 training / Atlas A3 series). Verify device availability:

```bash
npu-smi info
```

### 2. CANN Toolkit

Source the CANN environment before running:

```bash
source /usr/local/Ascend/cann-<version>/set_env.sh
```

This sets `ASCEND_HOME_PATH` and adds CANN libraries to `LD_LIBRARY_PATH`. The ATB library (`libasdsip.so`) requires `$ASCEND_HOME_PATH/runtime/data/platform_config/` to exist for SoC detection. If your CANN installation places `platform_config` under `aarch64-linux/data/` instead of `runtime/data/`, create a symlink:

```bash
# Only needed if you see "platform_config does not exist" in ATB logs
sudo ln -s /usr/local/Ascend/cann-<version>/aarch64-linux/data \
           /usr/local/Ascend/cann-<version>/runtime/data
```

### 3. NNAL / ATB

Source the NNAL ATB environment to make `libatb.so` available:

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4. Python Environment

Requires `torch` and `torch_npu` with ATB support:

```bash
conda activate <your_npu_env>
python -c "import torch_npu; print(torch_npu.__version__)"
```

### 5. Complete Setup

A typical session setup before running the benchmark:

```bash
source /usr/local/Ascend/cann-<version>/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
conda activate <your_npu_env>
```

---

## Usage

### Predefined Cases

Three predefined cases are included, aligned with production-scale model configurations:

| Case  | batch | num_heads | kv_head_num | head_dim | block_size | context_len | dtype    |
|-------|-------|-----------|-------------|----------|------------|-------------|----------|
| Case1 | 256   | 16        | 1           | 128      | 128        | 8192        | bfloat16 |
| Case2 | 64    | 64        | 1           | 128      | 64         | 8192        | bfloat16 |
| Case3 | 64    | 64        | 1           | 256      | 64         | 8192        | bfloat16 |

```bash
# Run a single predefined case
python tools/bench_npu_paged_attention.py --case Case1 --device 0

# Run all predefined cases and print a summary comparison table
python tools/bench_npu_paged_attention.py --case all --device 0
```

### Custom Parameters

```bash
python tools/bench_npu_paged_attention.py \
    --batch 32 --num-heads 40 --kv-head-num 8 --head-dim 128 \
    --block-size 16 --context-len 2048 --max-model-len 4096 \
    --device 0
```

### Variable Sequence Lengths

Use `--context-lens-list` to assign different context lengths per batch element (simulating real serving workloads with mixed sequence lengths):

```bash
python tools/bench_npu_paged_attention.py \
    --batch 4 --num-heads 32 --head-dim 128 --block-size 16 \
    --context-lens-list 512,1024,2048,4096 --max-model-len 8192
```

If fewer values are provided than `--batch`, they are repeated cyclically. If more are provided, they are truncated.

### Adjusting Warmup and Measurement Rounds

```bash
python tools/bench_npu_paged_attention.py --case Case1 --warmup 50 --rounds 200
```

---

## CLI Reference

### Tensor Shape Parameters

| Flag                  | Default  | Description                                               |
|-----------------------|----------|-----------------------------------------------------------|
| `--batch`             | 256      | Batch size (number of sequences)                          |
| `--num-heads`         | 16       | Number of query attention heads                           |
| `--kv-head-num`       | 1        | Number of KV heads (for GQA/MQA)                          |
| `--head-dim`          | 128      | Head dimension                                            |
| `--block-size`        | 128      | Tokens per KV cache block                                 |
| `--context-len`       | 8192     | Uniform context length for all sequences                  |
| `--max-model-len`     | 32768    | Maximum model sequence length (determines block_table width) |
| `--context-lens-list` | —        | Comma-separated per-batch context lengths (overrides `--context-len`) |
| `--dtype`             | bfloat16 | Data type for Q/K/V tensors (`float16` or `bfloat16`)     |

### Benchmark Parameters

| Flag       | Default | Description                      |
|------------|---------|----------------------------------|
| `--warmup` | 10      | Number of warmup iterations      |
| `--rounds` | 100     | Number of timed measurement rounds |
| `--device` | 0       | NPU device ID                    |
| `--case`   | —       | Predefined case name (`Case1`, `Case2`, `Case3`, or `all`) |

---

## Input Tensor Layout

The script generates random tensors following the ATB PagedAttention operator schema:

```
torch_npu._npu_paged_attention(
    query,        # [batch, num_heads, head_dim]           — on NPU
    key_cache,    # [total_blocks, block_size, kv_head_num, head_dim] — on NPU
    value_cache,  # [total_blocks, block_size, kv_head_num, head_dim] — on NPU
    num_kv_heads, # int
    num_heads,    # int
    scale_value,  # float  (1 / sqrt(head_dim))
    block_table,  # [batch, max_num_blocks_per_req]        — on NPU, int32
    context_lens, # [batch]                                — on CPU, int32
    out,          # [batch, num_heads, head_dim]            — on NPU
)
```

Key details:

- **`total_blocks`** = `batch * ceil(max_context_len / block_size)`. Each sequence gets non-overlapping random physical block indices.
- **`max_num_blocks_per_req`** = `max_model_len / block_size`. Unused slots in `block_table` are zero-filled.
- **`context_lens` must remain on CPU.** The ATB operator reads `context_lens` via `hostData` during `OperationSetup` to build the execution plan. Moving it to NPU causes `tensor.hostData is null` → `build param from host tensor fail`.
- **`scale_value`** = `1.0 / sqrt(head_dim)`, the standard attention scaling factor.

---

## Operator Constraints

The following constraints apply to `torch_npu._npu_paged_attention` (derived from CANN/ATB documentation and runtime validation):

| Constraint                    | Requirement                                  |
|-------------------------------|----------------------------------------------|
| `block_size`                  | Must be a multiple of 16; `block_size <= 128` |
| `head_dim`                    | Range: (0, 256]                              |
| GQA                           | `num_heads >= kv_head_num` and `num_heads % kv_head_num == 0` |
| `block_table` values          | Must be in `[0, total_blocks)`               |
| `context_lens` tensor         | Must be on CPU (host memory)                 |
| dtype                         | `float16` or `bfloat16` (Atlas A2/A3)        |

Violating these constraints results in `PagedAttentionOperation setup failed!` during the ATB `OperationSetup` phase. Check ATB logs at `~/ascend/log/atb/atb_<pid>_<timestamp>.log` for detailed error messages.

---

## Benchmark Methodology

1. **Input generation**: Random tensors are created on CPU, then transferred to the target NPU device (`context_lens` stays on CPU).
2. **Warmup phase**: `--warmup` iterations are run without timing to warm up the NPU pipeline, JIT compilation caches, and memory allocators. A single `torch.npu.synchronize()` follows.
3. **Measurement phase**: Each of the `--rounds` iterations is individually timed:
   - `torch.npu.synchronize()` before starting the timer
   - Execute `_npu_paged_attention`
   - `torch.npu.synchronize()` after execution
   - Record wall-clock elapsed time via `time.perf_counter()`
4. **Statistics**: Latencies are sorted and summarized. The trimmed mean drops the top and bottom 10% to reduce outlier impact.

---

## Example Output

```
Device: NPU:0
PyTorch: 2.8.0
torch_npu: 2.8.0

================================================================================
  Case1: batch=256, heads=16, ctx=8K, blk=128
================================================================================

  Input shapes:
  query:         [256, 16, 128]  dtype=torch.bfloat16
  key_cache:     [16384, 128, 1, 128]  dtype=torch.bfloat16
  value_cache:   [16384, 128, 1, 128]  dtype=torch.bfloat16
  block_table:   [256, 256]  dtype=torch.int32
  context_lens:  [256]  dtype=torch.int32
  context_len:   8192
  num_kv_heads:  1
  num_heads:     16
  scale_value:   0.088388

  Warmup: 10 rounds ...

  --- Latency ---
  Rounds:        100
  Mean:          1234.5 us
  Trimmed Mean:  1220.3 us  (drop 10% tails)
  Std:            45.2 us
  Min:           1180.1 us
  P50:           1215.0 us
  P90:           1290.3 us
  P99:           1350.7 us
  Max:           1402.1 us
```

When running `--case all`, a summary comparison table is appended:

```
================================================================================
  Summary
================================================================================
  Case                       Mean (us)   Trim (us)    P50 (us)    P99 (us)
  -------------------------  ----------  ----------  ----------  ----------
  Case1                        1234.5      1220.3      1215.0      1350.7
  Case2                        2456.8      2440.1      2445.2      2520.3
  Case3                        4012.3      3990.5      3998.1      4150.6
```

---

## Troubleshooting

### `PagedAttentionOperation setup failed!`

This is a generic ATB error. Check the detailed ATB log for the root cause:

```bash
ls -lt ~/ascend/log/atb/  # find the latest log file
cat ~/ascend/log/atb/atb_<pid>_<timestamp>.log
```

Common causes:

| ATB Log Message | Cause | Fix |
|-----------------|-------|-----|
| `tensor.hostData is null` | `context_lens` tensor is on NPU | Keep `context_lens` on CPU (do not call `.to(device)` on it) |
| `platform_config does not exist` | CANN env not sourced or missing symlink | `source set_env.sh`; create `runtime/data` symlink if needed (see [Prerequisites](#2-cann-toolkit)) |
| `PagedAttentionMaskNdKernel is not found` | SoC detection failed (follows platform_config error) | Fix the platform_config issue above |
| `build param from host tensor fail` | Same as `hostData is null` | Same fix |

### `libatb.so: cannot open shared object file`

NNAL ATB environment is not sourced:

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### `torch_npu is not installed`

The `torch_npu` package is not available in the current Python environment:

```bash
conda activate <your_npu_env>
```

### Synchronous Debugging

For accurate stack traces when diagnosing operator errors, enable synchronous execution:

```bash
ASCEND_LAUNCH_BLOCKING=1 python tools/bench_npu_paged_attention.py --case Case1
```

This forces all NPU operations to run synchronously. Remove the env var after debugging as it degrades performance.
