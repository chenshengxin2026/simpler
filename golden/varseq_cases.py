# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared variable-length sequence configurations for Case1 and Case2.

Both paged_attention_unroll/golden.py and tools/bench_npu_paged_attention.py
import from this module so that their Case1 and Case2 use identical per-batch
context lengths, enabling fair comparison between the two.

Generation: fixed-seed random, block-size aligned, range [512, 16384].
"""

import random


def _make_lens(batch: int, seed: int, min_blocks: int, max_blocks: int, block_size: int) -> list[int]:
    """Return a deterministic list of block-aligned context lengths."""
    rng = random.Random(seed)
    return [rng.randint(min_blocks, max_blocks) * block_size for _ in range(batch)]


# Case1: batch=256, block_size=128  →  context range [512, 16384]
VARSEQ_CASE1_LENS: list[int] = _make_lens(
    batch=256, seed=42, min_blocks=4, max_blocks=128, block_size=128
)

# Case2: batch=64, block_size=64  →  context range [512, 16384]
VARSEQ_CASE2_LENS: list[int] = _make_lens(
    batch=64, seed=42, min_blocks=8, max_blocks=256, block_size=64
)
