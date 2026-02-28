# TFILLPAD_INPLACE Bug Reproduction

## Summary

`TFILLPAD_INPLACE` produces incorrect padding results, causing downstream softmax
and attention computations to return wrong values. The severity differs by platform:

- **Hardware (a2a3)**: Bug reproduces only at N=16 (float32) when `valid_len <= N/2`
- **Simulator (a2a3sim)**: Bug reproduces at all tested N values (16, 32, 64, 128)

## Environment

- **Platform**: Ascend A2/A3
- **CANN**: 8.5.0
- **Data type**: float32
- **PTO source**: `include/pto/npu/a2a3/TFillPad.hpp`

## This Example

This directory (`batch_paged_attention_issue/`) is a modified copy of
`batch_paged_attention/` that removes the SetValue workaround, using **only**
`TFILLPAD_INPLACE` for padding. This makes the bug visible in test results.

### Changes from `batch_paged_attention/`

1. **`kernels/aiv/aiv_softmax_prepare.cpp`**: Removed the SetValue workaround
   loop. Only `TFILLPAD_INPLACE(sijPadTile, sijDynTile)` is called.

2. **All kernels**: Updated to support `head_dim=32` and variable `block_size`
   {16, 32, 64, 128} via runtime dispatch in `kernel_entry`.

3. **`kernels/orchestration/paged_attention_orch.cpp`**: Passes `block_size`
   as a scalar parameter to each kernel for dispatch.

4. **`golden.py`**: Test cases sweep `block_size` across {16, 32, 64, 128}
   with `context_len = block_size ± 1`.

### Key Code (the bug)

In `aiv_softmax_prepare.cpp`:
```cpp
TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
TASSIGN(sijDynTile, 0x0);

// BUG: TFILLPAD_INPLACE alone produces corrupted padding data.
// Columns [valid_len, N) should be filled with -inf but get wrong values.
TFILLPAD_INPLACE(sijPadTile, sijDynTile);

// The workaround (removed in this version) would fix it:
//   if (valid_len < N) {
//       for (r = 0; r < M; r++)
//           for (c = valid_len; c < N; c++)
//               sijTile.SetValue(r * N + c, -inf);
//   }
```

## Test Configuration

```
batch=1, num_heads=16, kv_head_num=1, head_dim=32, max_model_len=256
```

## Test Results

Tested with TFILLPAD_INPLACE only (no SetValue workaround), sweeping `block_size`
(= tile column count N) across {16, 32, 64, 128} with `context_len = block_size ± 1`:

| block_size (N) | context_len | valid_len in last block | a2a3 (hardware) | a2a3sim (simulator) |
|----------------|-------------|------------------------|-----------------|---------------------|
| 16             | 15          | 15 (pads 1)            | PASS            | FAIL (401/512)      |
| 16             | 17          | 1 (pads 15)            | FAIL (48/512)   | FAIL (476/512)      |
| 32             | 31          | 31 (pads 1)            | PASS            | FAIL (277/512)      |
| 32             | 33          | 1 (pads 31)            | PASS            | FAIL (233/512)      |
| 64             | 63          | 63 (pads 1)            | PASS            | FAIL (114/512)      |
| 64             | 65          | 1 (pads 63)            | PASS            | FAIL (461/512)      |
| 128            | 127         | 127 (pads 1)           | PASS            | FAIL (17/512)       |
| 128            | 129         | 1 (pads 127)           | PASS            | FAIL (9/512)        |

### Hardware (a2a3)

Only `block_size=16, context_len=17` (valid_len=1) fails. All N >= 32 pass.
This is consistent with earlier findings that the bug is limited to N=16 where
`valid_len <= N/2` triggers the buggy `PadRightRemainingRows` code path.

### Simulator (a2a3sim)

All 8 cases fail, regardless of block_size. The mismatch count generally
decreases with larger N (476 -> 233 -> 461 -> 9), but none pass. This indicates
the simulator's `TFILLPAD_INPLACE` implementation has broader correctness issues.

## Root Cause

The bug is in `TFillPad` (`include/pto/npu/a2a3/TFillPad.hpp`). The function
has two internal padding paths:

- **Path A** (`Handle32BAlignedPad_Other`): Fills the partial 32-byte block at
  the boundary using `vector_dup` with a bitmask. This path is reliable on hardware.

- **Path B** (`PadRightSingleRow` + `PadRightRemainingRows`): Fills complete
  32-byte blocks to the right of the boundary. Uses `vector_dup` for row 0, then
  `vcopy` with `srcRepeatStride=0` (broadcast) to replicate to remaining rows.
  This path produces incorrect results on hardware when N=16.

Which path runs depends on `valid_len`:

```
elements_per_block = 32 / sizeof(float) = 8
srcValidCol32B = ceil(valid_len / 8) * 8
padCols = N - srcValidCol32B    // columns for Path B

For N=16:
  valid_len in [1,8]  -> padCols = 8  -> Path B runs -> BUG
  valid_len in [9,15] -> padCols = 0  -> Path B is NO-OP -> OK
```

The problematic `vcopy` call in Path B:
```cpp
// dstRepeatStride=2 (64 bytes = 1 row at N=16), srcRepeatStride=0 (broadcast)
vcopy(_dstPtr, dstPtr + padOffset, 15, 1, 0, 2, 0);
```

On the simulator, even Path A appears to produce incorrect results, explaining
why all N values fail.

## Workaround

The working fix in `batch_paged_attention/` uses **both** `TFILLPAD_INPLACE`
and scalar `SetValue` writes:

```cpp
TFILLPAD_INPLACE(sijPadTile, sijDynTile);
if (valid_len < static_cast<uint64_t>(N)) {
    constexpr float NEG_INF = -__builtin_huge_valf();
    for (int r = 0; r < M; r++) {
        for (uint64_t c = valid_len; c < N; c++) {
            sijTile.SetValue(static_cast<uint32_t>(r * N + c), NEG_INF);
        }
    }
}
```

`TFILLPAD_INPLACE` is still needed even when overwritten by `SetValue`, because
it sets up vector pipeline state (mask modes, barriers) that subsequent vector
operations depend on.

## Impact

- **Hardware users with block_size >= 32**: Can use `TFILLPAD_INPLACE` alone
- **Hardware users with block_size = 16**: Must use the combined workaround
- **Simulator users**: Must use the combined workaround at all block sizes
- **float16/bfloat16 users**: Likely affected at N <= 32 (untested)

## Expected Behavior

`TFILLPAD_INPLACE(padTile, dynTile)` should correctly fill columns
`[valid_len, N)` with the pad value (`-inf` for `PadValue::Min`) for all
valid combinations of N and `valid_len`, on both hardware and simulator.

## How to Run

All commands assume the working directory is the simpler project root
(`simpler/`). The example directory is:

```
examples/tensormap_and_ringbuffer/batch_paged_attention_issue/
```

### Running a Single Test Case

Use `examples/scripts/run_example.py` to run a single test case.

**Simulation mode** (no hardware required):

```bash
python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/batch_paged_attention_issue/kernels \
    -g examples/tensormap_and_ringbuffer/batch_paged_attention_issue/golden.py \
    -p a2a3sim
```

**Hardware mode** (requires Ascend device):

```bash
python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/batch_paged_attention_issue/kernels \
    -g examples/tensormap_and_ringbuffer/batch_paged_attention_issue/golden.py \
    -p a2a3 -d 0
```

By default, these run the `DEFAULT_CASE` (`BS16_Pad15`). To run a specific case,
use `--case`:

```bash
# Run block_size=64, context_len=65 on hardware
python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/batch_paged_attention_issue/kernels \
    -g examples/tensormap_and_ringbuffer/batch_paged_attention_issue/golden.py \
    -p a2a3 -d=12 --case BS64_Pad63
```

```bash
# Run block_size=64, context_len=65 on simulator
python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/batch_paged_attention_issue/kernels \
    -g examples/tensormap_and_ringbuffer/batch_paged_attention_issue/golden.py \
    -p a2a3sim --case BS64_Pad63
```

Available test case names:

| Case name      | block_size | context_len | valid_len | pad_cols |
|----------------|-----------|-------------|-----------|----------|
| `BS16_Pad1`    | 16        | 15          | 15        | 1        |
| `BS16_Pad15`   | 16        | 17          | 1         | 15       |
| `BS32_Pad1`    | 32        | 31          | 31        | 1        |
| `BS32_Pad31`   | 32        | 33          | 1         | 31       |
| `BS64_Pad1`    | 64        | 63          | 63        | 1        |
| `BS64_Pad63`   | 64        | 65          | 1         | 63       |
| `BS128_Pad1`   | 128       | 127         | 127       | 1        |
| `BS128_Pad127` | 128       | 129         | 1         | 127      |

Common options for `run_example.py`:

| Option                | Description                                      |
|-----------------------|--------------------------------------------------|
| `-p a2a3` / `a2a3sim` | Platform: hardware or simulator                  |
| `-d <id>`             | Device ID (default: 0)                            |
| `--case <name>`       | Run a specific test case                          |
| `--all`               | Run all test cases                                |
| `-v`                  | Verbose output (debug log level)                  |
| `--silent`            | Only show errors                                  |
| `--enable-profiling`  | Enable profiling and generate swimlane.json       |


## Files

- **Bug location**: `include/pto/npu/a2a3/TFillPad.hpp`, functions
  `PadRightSingleRow` and `PadRightRemainingRows`
- **Bug reproduction (this example)**: `examples/tensormap_and_ringbuffer/batch_paged_attention_issue/`
- **Workaround version**: `examples/tensormap_and_ringbuffer/batch_paged_attention/`
- **Detailed analysis**: `examples/tensormap_and_ringbuffer/paged_attention/TFILLPAD_INPLACE_BUG.md`
