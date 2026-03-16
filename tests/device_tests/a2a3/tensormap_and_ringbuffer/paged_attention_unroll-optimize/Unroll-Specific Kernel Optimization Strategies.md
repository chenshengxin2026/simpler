# Unroll-Specific Kernel Optimization Strategies

## Premise

The three optimizations ported from `paged_attention-optimize` (TRESHAPE in online_update, event separation in QK/PV matmul, TSTORE overlap in softmax) were designed for a per-block-iteration model. Under the unrolled execution model (`N_UNROLL=64`, all blocks in internal loops), these optimizations are either irrelevant (TRESHAPE — the "else" path is never reached) or counterproductive (extra `pipe_barrier(PIPE_V)` in the 64-iteration softmax loop).

This document identifies **six optimization strategies specifically designed for the unrolled execution pattern**, targeting the three dominant bottlenecks: the 64-iteration internal loops in QK matmul, softmax, and PV matmul.

---

## Strategy 1: Revert Softmax Pass 2 TSTORE/Compute Overlap (Remove Degradation Source)

**Kernel**: `aiv_softmax_prepare.cpp` — Pass 2 inner loop
**Priority**: Critical (this is the direct cause of the observed degradation)
**Estimated Savings**: ~100K–340K cycles per batch (eliminates net-negative overhead)

### Problem

The current optimize version inserts 3 extra `pipe_barrier(PIPE_V)` per Pass 2 iteration to enable TSTORE overlap with downstream computation:

```cpp
// Current optimize version (Pass 2 loop body, lines 169-186):
TEXP(pijTile, pijTile);
pipe_barrier(PIPE_V);                                   // ADDED barrier #1
TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);                 // pij bf16 ready early
pipe_barrier(PIPE_V);                                   // ADDED barrier #2
TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
pipe_barrier(PIPE_V);                                   // ADDED barrier #3
if (i == 0) { TMULS(sumAccTile, pijTile, 1.0f); }
else         { TADD(sumAccTile, sumAccTile, pijTile); }
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(pijGlobal, pijBf16Tile);
```

Over 64 iterations × 256 batches = 49,152 extra pipeline stalls (~250K–500K cycles), exceeding the ~164K cycles saved by TSTORE overlap.

### Solution

Revert to the original sequential pattern where TSTORE happens after all computation completes. This eliminates the 3 extra barriers per iteration while maintaining correctness:

```cpp
// Reverted Pass 2 loop body:
TEXP(pijTile, pijTile);
TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
if (i == 0) { TMULS(sumAccTile, pijTile, 1.0f); }
else         { TADD(sumAccTile, sumAccTile, pijTile); }

set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(pijGlobal, pijBf16Tile);
```

### Impact

Eliminates 192 extra `pipe_barrier(PIPE_V)` per kernel call (3 × 64 iterations). The TSTORE of the 4KB pij tile (~10 cycles on MTE3) completes within the MTE3→MTE2 sync window anyway, so the "overlap" benefit was already marginal.

---

## Strategy 2: Eliminate GM Round-Trip in Softmax Pass 1 via TRESHAPE

**Kernel**: `aiv_softmax_prepare.cpp` — Pass 1 inner loop
**Priority**: High (largest single optimization target in the unroll kernel)
**Estimated Savings**: ~80K GM accesses per batch (× 256 batches)

### Problem

Pass 1 computes the global row max across 64 blocks. `TROWMAX` produces a ColMajor DN tile `localMaxDN`, but element-wise `TMAX` for accumulation requires ND layout. The current code uses a **GM round-trip** for this conversion in every iteration i > 0:

```cpp
// Current Pass 1 (i > 0, lines 122-141):
TROWMAX(localMaxDN, sijTile, tmpTile);          // → ColMajor (M,1) in UB

// GM round-trip to convert DN → ND for TMAX:
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(lijGlobalDN, localMaxDN);                // UB → GM (DN)      ← 1 GM write

set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
TLOAD(maxND_a, mijGlobalND);                    // GM → UB (ND)      ← 1 GM read
TLOAD(maxND_b, lijGlobalND);                    // GM → UB (ND)      ← 1 GM read

set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TMAX(maxND_a, maxND_a, maxND_b);                // element-wise max

set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(mijGlobalND, maxND_a);                   // UB → GM (ND)      ← 1 GM write
```

That's **4 GM operations + 6 set_flag/wait_flag pairs per iteration** for iterations 1–63 = **252 GM accesses + 378 sync instructions**.

### Solution

Use TRESHAPE to perform the DN↔RowMajor conversion entirely in UB, keeping the accumulated global max in UB across all iterations:

```cpp
// TRESHAPE-based Pass 1:
TileScalarRow globalMaxRow;   // persistent UB accumulator (RowMajor 1,M)
TileScalarRow localMaxRow;    // scratch for TRESHAPE output

for (uint64_t i = 0; i < n_blocks; i++) {
    // ... TLOAD sij, TFILLPAD if needed, TMULS ...
    pipe_barrier(PIPE_V);
    TROWMAX(localMaxDN, sijTile, tmpTile);

    if (i == 0) {
        TRESHAPE(globalMaxRow, localMaxDN);       // DN → RowMajor (zero-cost)
    } else {
        TRESHAPE(localMaxRow, localMaxDN);        // DN → RowMajor (zero-cost)
        TMAX(globalMaxRow, globalMaxRow, localMaxRow);  // element-wise in UB
    }
}

// After Pass 1: TRESHAPE back to DN for Pass 2's TROWEXPANDSUB
TRESHAPE(globalMaxDN, globalMaxRow);              // RowMajor → DN (zero-cost)

// Store to mij for output (optional, can skip if not needed downstream)
TileScalarND globalMaxND;
TASSIGN(globalMaxND, /* same UB as globalMaxRow */);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(mijGlobalND, globalMaxND);
```

### Impact

- **Eliminates**: 63 × 4 = 252 GM accesses, 63 × 6 = 378 sync instructions
- **Replaces with**: 63 TRESHAPE (zero-cost metadata) + 63 TMAX (pure UB compute)
- **Bonus**: globalMaxDN stays in UB for Pass 2, eliminating the TLOAD between passes (saves 1 additional GM read + sync pair)

---

## Strategy 3: Hoist qi TLOAD in QK Matmul (Eliminate Redundant Loads)

**Kernel**: `aic_qk_matmul.cpp` — inner loop
**Priority**: High (simple change, clear savings)
**Estimated Savings**: 63 × 4KB = 252KB GM bandwidth per kernel call (× 256 batches)

### Problem

The query tile `qi` is identical across all 64 iterations (same query head against different key blocks), but is reloaded from GM every iteration:

```cpp
for (uint64_t i = 0; i < n_blocks; i++) {
    GlobalA qiGlobal(qi_base);                           // same address every iteration!
    GlobalB kjGlobal(key_base + block_indices[i] * N * K); // different each iteration

    TLOAD(aMatTile, qiGlobal);    // redundant for i > 0
    // ...
    TLOAD(bMatTile, kjGlobal);
    // ...
}
```

### Solution

Hoist the qi TLOAD before the loop. Since `aMatTile` at L1 address 0x0 is only 4KB (Case1: 16×128×2) while `bMatTile` at 0x20000 is 32KB (128×128×2), they occupy non-overlapping L1 regions. As long as `bMatTile` load doesn't clobber `aMatTile`'s L1 slot, the hoisted load remains valid:

```cpp
// Pre-load qi once before the loop
GlobalA qiGlobal(qi_base);
TLOAD(aMatTile, qiGlobal);
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
TMOV(aTile, aMatTile);           // qi → L0A (persists if L0A not overwritten)

for (uint64_t i = 0; i < n_blocks; i++) {
    GlobalB kjGlobal(key_base + block_indices[i] * N * K);

    TLOAD(bMatTile, kjGlobal);   // only B load per iteration
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);       // re-TMOV from L1 (qi still in L1, skip GM)
    TMOV(bTile, bMatTile);
    // ... TMATMUL, TSTORE ...
}
```

Note: TMATMUL consumes L0A/L0B data, so `aTile` must be re-TMOV'd from L1 each iteration. But the L1→L0A TMOV (~4KB) is much cheaper than GM→L1 TLOAD (~4KB over MTE2).

### Impact

- **Eliminates**: 63 GM→L1 TLOAD operations (63 × 4KB = 252KB bandwidth)
- **Eliminates**: 63 × 1 set_flag for the qi-specific MTE2 event
- **Retains**: 63 L1→L0A TMOV operations (~4 cycles each vs. ~20+ cycles for GM TLOAD)

---

## Strategy 4: Overlap TSTORE/TLOAD in QK Matmul (Replace pipe_barrier(PIPE_ALL))

**Kernel**: `aic_qk_matmul.cpp` — inner loop
**Priority**: Medium-High (eliminates the most expensive sync in the QK loop)
**Estimated Savings**: ~30–60 cycles per iteration × 63 iterations = ~2K–4K cycles per call

### Problem

The current loop has `pipe_barrier(PIPE_ALL)` between iterations, which forces **all** pipelines to drain before the next iteration starts:

```cpp
TMATMUL(cTile, aTile, bTile);                  // PIPE_M

set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

TSTORE(sijGlobal, cTile);                      // PIPE_FIX (MTE3)

if (i + 1 < n_blocks) {
    pipe_barrier(PIPE_ALL);                    // ← kills all overlap with next iteration
}
```

The `pipe_barrier(PIPE_ALL)` is conservative. The actual dependencies are:
1. TSTORE(cTile) must complete before cTile is overwritten by next TMATMUL → **PIPE_FIX → PIPE_M dependency**
2. TLOAD(bMatTile) reuses L1 address 0x20000 → must wait for previous TMOV(bTile) to complete → **PIPE_MTE1 → PIPE_MTE2 dependency**

But TLOAD(next B) to L1 via MTE2 is **independent** of TSTORE(sij) via MTE3, so they can run in parallel.

### Solution

Replace `pipe_barrier(PIPE_ALL)` with targeted synchronization that allows TLOAD(next B) and TSTORE(current sij) to overlap:

```cpp
TMATMUL(cTile, aTile, bTile);

// Signal matmul done to both TSTORE pipe and TLOAD pipe
set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);     // TSTORE can start
set_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);    // next TLOAD can start (after L0 is consumed)

// Start TSTORE and next TLOAD in parallel
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
TSTORE(sijGlobal, cTile);                   // MTE3: store result to GM

if (i + 1 < n_blocks) {
    // Next B load (MTE2) runs in parallel with TSTORE (MTE3)
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
    GlobalB kjGlobal_next(key_base + block_indices[i+1] * N * K);
    TLOAD(bMatTile, kjGlobal_next);          // MTE2: load next B from GM

    // Wait for both TSTORE and TLOAD before TMOV
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);   // TSTORE → allow TMOV (cTile freed)
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);  // TLOAD → allow TMOV (B in L1)
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

    TMOV(aTile, aMatTile);                   // qi still in L1 (from Strategy 3)
    TMOV(bTile, bMatTile);
}
```

### Impact

- TLOAD(next B, 32KB via MTE2) runs in parallel with TSTORE(sij, 8KB via MTE3)
- Saves ~20–40 cycles per iteration (the TLOAD latency that was previously serialized)
- The `pipe_barrier(PIPE_ALL)` drained **all 5 pipelines** (~30–60 cycle stall); targeted sync drains only the required pipes

---

## Strategy 5: Double-Buffer sij TLOAD in Softmax Pass 2

**Kernel**: `aiv_softmax_prepare.cpp` — Pass 2 inner loop
**Priority**: Medium (requires UB memory budget, but meaningful latency hiding)
**Estimated Savings**: ~15–30 cycles per iteration × 63 iterations (TLOAD latency hidden)

### Problem

Each Pass 2 iteration begins with a blocking TLOAD of the next sij tile, which stalls PIPE_V:

```cpp
for (uint64_t i = 0; i < n_blocks; i++) {
    GlobalDataMxN sijGlobal(sij_base + i * M * N);
    TLOAD(sijTile, sijGlobal);                     // ← blocking: PIPE_V idle
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // ... compute (TMULS, TROWEXPANDSUB, TEXP, TCVT, accumulate) ...
    // ... TSTORE pij ...
}
```

Computation cannot start until TLOAD completes, leaving PIPE_V idle during the entire MTE2 transfer.

### Solution

Double-buffer `sijTile` so that the TLOAD of iteration i+1 overlaps with computation of iteration i. Requires allocating a second sij UB buffer (additional M×N×4 bytes = 8KB for Case1):

```cpp
TileVecMxN sijTile_A, sijTile_B;
TASSIGN(sijTile_A, 0x0);
TASSIGN(sijTile_B, /* new slot: kDataBytes offset */);

// Pre-load first tile
GlobalDataMxN sijGlobal_0(sij_base);
TLOAD(sijTile_A, sijGlobal_0);

for (uint64_t i = 0; i < n_blocks; i++) {
    TileVecMxN& curSij = (i % 2 == 0) ? sijTile_A : sijTile_B;
    TileVecMxN& nxtSij = (i % 2 == 0) ? sijTile_B : sijTile_A;

    // Wait for current tile to finish loading
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Kick off next tile load (overlaps with all computation below)
    if (i + 1 < n_blocks) {
        GlobalDataMxN sijGlobal_next(sij_base + (i + 1) * M * N);
        TLOAD(nxtSij, sijGlobal_next);             // MTE2 runs in background
    }

    // Compute on curSij (PIPE_V, fully overlapped with TLOAD above)
    TMULS(curSij, curSij, scale_value);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, curSij, globalMaxDN);
    pipe_barrier(PIPE_V);
    TEXP(pijTile, pijTile);
    TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
    TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
    if (i == 0) { TMULS(sumAccTile, pijTile, 1.0f); }
    else         { TADD(sumAccTile, sumAccTile, pijTile); }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(pijGlobal, pijBf16Tile);

    // Ensure TSTORE completes before next iteration's TLOAD reuses MTE path
    if (i + 1 < n_blocks) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
}
```

### UB Memory Budget

Case1 (M=16, N=128): extra 16×128×4 = 8,192 bytes. The current layout uses ~4×kDataBytes + scalars ≈ 35KB. UB is typically 256KB, so 8KB additional is well within budget.

### Impact

- Hides the TLOAD latency (~15–30 cycles for 8KB) behind PIPE_V computation (~40+ cycles for scale+sub+exp+cvt+acc)
- Net effect: 63 iterations × ~15 cycles = ~1K cycles saved per kernel call, modest but free

---

## Strategy 6: Overlap TLOAD/TMATMUL_ACC in PV Matmul

**Kernel**: `aic_pv_matmul.cpp` — inner loop
**Priority**: Medium (the PV loop has tighter data dependencies than QK)
**Estimated Savings**: ~20–40 cycles per iteration × 63 iterations = ~1.5K–2.5K cycles per call

### Problem

The PV matmul accumulates all 64 blocks via `TMATMUL_ACC`. The current loop waits for TMATMUL to complete before starting the next TLOAD:

```cpp
for (uint64_t i = 0; i < n_blocks; i++) {
    TLOAD(aMatTile, pijGlobal);    // MTE2
    TLOAD(bMatTile, vjGlobal);     // MTE2
    // ... TMOV to L0 ...
    if (i == 0) TMATMUL(cTile, aTile, bTile);
    else        TMATMUL_ACC(cTile, cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);    // ← wait for matmul before next load
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
}
```

TMATMUL_ACC writes to L0C (AccTile) and reads from L0A/L0B. The next iteration's TLOAD writes to L1, which is **independent** of L0C. So TLOAD can overlap with TMATMUL_ACC.

### Solution

Start the next iteration's TLOAD before the current TMATMUL_ACC completes, using double-buffered L1 tiles:

```cpp
TileMatA aMatTile_ping, aMatTile_pong;
TileMatB bMatTile_ping, bMatTile_pong;
TASSIGN(aMatTile_ping, 0x0);
TASSIGN(aMatTile_pong, 0x10000);     // separate L1 slot
TASSIGN(bMatTile_ping, 0x20000);
TASSIGN(bMatTile_pong, 0x30000);     // separate L1 slot

// Pre-load first iteration
TLOAD(aMatTile_ping, pijGlobal_0);
TLOAD(bMatTile_ping, vjGlobal_0);

for (uint64_t i = 0; i < n_blocks; i++) {
    auto& curA = (i % 2 == 0) ? aMatTile_ping : aMatTile_pong;
    auto& curB = (i % 2 == 0) ? bMatTile_ping : bMatTile_pong;
    auto& nxtA = (i % 2 == 0) ? aMatTile_pong : aMatTile_ping;
    auto& nxtB = (i % 2 == 0) ? bMatTile_pong : bMatTile_ping;

    // Wait for current loads, TMOV to L0
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, curA);
    TMOV(bTile, curB);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Matmul
    if (i == 0) TMATMUL(cTile, aTile, bTile);
    else        TMATMUL_ACC(cTile, cTile, aTile, bTile);

    // Start next TLOAD immediately (L1 nxt slots are independent of L0)
    if (i + 1 < n_blocks) {
        GlobalA pijGlobal_next(pij_base + (i+1) * M * K);
        GlobalB vjGlobal_next(val_base + block_indices[i+1] * K * N);
        TLOAD(nxtA, pijGlobal_next);    // overlaps with TMATMUL_ACC above
        TLOAD(nxtB, vjGlobal_next);     // overlaps with TMATMUL_ACC above
    }

    // Only wait for matmul before using nxt L1 slots in next TMOV
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
}
```

### L1 Memory Budget

Case1: aMatTile = 16×128×2 = 4KB, bMatTile = 128×128×2 = 32KB. Double-buffered: 2×(4+32) = 72KB. L1 is 1MB, well within budget.

### Impact

- TLOAD(pij + vj) for next iteration (~36KB total via MTE2) runs in parallel with TMATMUL_ACC (~matmul latency ~20–50 cycles)
- Hides most of the MTE2 transfer latency, saving ~20–40 cycles per iteration

---

## Combined Impact Summary

| # | Strategy | Target Kernel | Mechanism | Estimated Saving (per kernel call) |
|---|---------|--------------|-----------|-----------------------------------|
| 1 | Revert TSTORE overlap | softmax Pass 2 | Remove 192 extra `pipe_barrier(PIPE_V)` | +400–1300 cycles (eliminates degradation) |
| 2 | TRESHAPE in Pass 1 | softmax Pass 1 | Eliminate 252 GM accesses + 378 syncs | +5K–10K cycles |
| 3 | Hoist qi TLOAD | QK matmul | Eliminate 63 redundant GM loads | +1.5K–3K cycles |
| 4 | Overlap TSTORE/TLOAD | QK matmul | Replace `pipe_barrier(PIPE_ALL)` with targeted sync | +2K–4K cycles |
| 5 | Double-buffer sij | softmax Pass 2 | Hide TLOAD latency behind computation | +1K cycles |
| 6 | Double-buffer L1 | PV matmul | Hide TLOAD latency behind TMATMUL_ACC | +1.5K–2.5K cycles |
| | **Total** | | | **+11K–22K cycles per kernel call** |

Multiplied by 256 batches (Case1 config): **~2.8M–5.6M cycles total improvement**.

### Priority Ordering

1. **Strategy 1** (revert Pass 2 overlap) — immediate, zero-risk, eliminates the observed degradation
2. **Strategy 2** (TRESHAPE in Pass 1) — highest absolute benefit, well-understood TRESHAPE pattern
3. **Strategy 3** (hoist qi) — simple code change, clear GM bandwidth savings
4. **Strategy 4** (QK TSTORE/TLOAD overlap) — moderate complexity, removes the most expensive sync
5. **Strategy 6** (PV double-buffer) — moderate complexity, meaningful latency hiding
6. **Strategy 5** (softmax double-buffer) — lowest priority, smallest benefit, but still positive

### Key Design Principle

The unrolled execution model concentrates 64 blocks into tight internal loops. Optimizations must target **per-iteration overhead** (barriers, redundant loads, serialized operations) rather than **per-kernel-call overhead** (layout conversion in online_update). Every extra instruction in the inner loop body is amplified 64×, and every eliminated stall cycle compounds across all iterations.
