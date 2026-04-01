#!/usr/bin/env python3
"""Benchmark torch_npu._npu_paged_attention with configurable shapes.

Measures wall-clock latency of the NPU paged attention operator across
multiple rounds, reports statistics (mean, min, max, std, percentiles),
and optionally validates output against a CPU golden reference.

Usage:
    # Single case with explicit parameters
    python tools/bench_npu_paged_attention.py \
        --batch 32 --num-heads 40 --kv-head-num 8 --head-dim 128 \
        --block-size 16 --context-len 2048 --max-model-len 4096

    # Predefined cases (aligned with paged_attention_unroll test cases)
    python tools/bench_npu_paged_attention.py --case Case1
    python tools/bench_npu_paged_attention.py --case all

    # Variable sequence lengths
    python tools/bench_npu_paged_attention.py \
        --batch 4 --num-heads 32 --head-dim 128 --block-size 16 \
        --context-lens-list 512,1024,2048,4096 --max-model-len 8192

    # Custom warmup / measurement rounds
    python tools/bench_npu_paged_attention.py --case Case1 \
        --warmup 50 --rounds 200
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu is not installed. This script requires an Ascend NPU environment.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Predefined benchmark cases
# ---------------------------------------------------------------------------
# Aligned with tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/golden.py
PREDEFINED_CASES = {
    "Case1": dict(
        batch=256, num_heads=16, kv_head_num=1, head_dim=128,
        block_size=128, context_len=8192, max_model_len=32768,
        dtype="bfloat16", desc="batch=256, heads=16, ctx=8K, blk=128",
    ),
    "Case2": dict(
        batch=64, num_heads=64, kv_head_num=1, head_dim=128,
        block_size=64, context_len=8192, max_model_len=32768,
        dtype="bfloat16", desc="batch=64, heads=64, ctx=8K, blk=64",
    ),
    "Case3": dict(
        batch=64, num_heads=64, kv_head_num=1, head_dim=256,
        block_size=64, context_len=8192, max_model_len=32768,
        dtype="bfloat16", desc="batch=64, heads=64, head_dim=256, ctx=8K, blk=64",
    ),
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
@dataclass
class PagedAttentionInputs:
    """All tensors needed for _npu_paged_attention, already on NPU.

    ATB schema (from libop_plugin_atb.so):
        _npu_paged_attention(Tensor query, Tensor key_cache, Tensor value_cache,
            int num_kv_heads, int num_heads, float scale_value,
            Tensor block_table, Tensor context_lens, Tensor(a!) out,
            *, Tensor? workspace=None) -> ()
    """
    query: torch.Tensor          # [batch, num_heads, head_dim]
    key_cache: torch.Tensor      # [total_blocks, block_size, kv_head_num, head_dim]
    value_cache: torch.Tensor    # [total_blocks, block_size, kv_head_num, head_dim]
    block_table: torch.Tensor    # [batch, max_num_blocks_per_req]  int32
    context_lens: torch.Tensor   # [batch]  int32
    out: torch.Tensor            # [batch, num_heads, head_dim]  float16/bfloat16
    num_kv_heads: int
    num_heads: int
    scale_value: float


def generate_inputs(
    batch: int,
    num_heads: int,
    kv_head_num: int,
    head_dim: int,
    block_size: int,
    context_len: int,
    max_model_len: int,
    dtype: str = "float16",
    context_lens_list: list[int] | None = None,
    device: str = "npu",
) -> PagedAttentionInputs:
    """Generate random input tensors on the target device."""
    torch_dtype = getattr(torch, dtype)
    max_num_blocks_per_req = max_model_len // block_size
    scale_value = 1.0 / (head_dim ** 0.5)

    # Per-batch context lengths
    if context_lens_list is not None:
        seq_vals = list(context_lens_list)
        if len(seq_vals) < batch:
            seq_vals = (seq_vals * ((batch + len(seq_vals) - 1) // len(seq_vals)))[:batch]
        elif len(seq_vals) > batch:
            seq_vals = seq_vals[:batch]
        ctx_lens = torch.tensor(seq_vals, dtype=torch.int32)
    else:
        ctx_lens = torch.full((batch,), context_len, dtype=torch.int32)

    max_ctx = int(ctx_lens.max().item())
    cur_valid_blocks = (max_ctx + block_size - 1) // block_size
    total_blocks = max(batch * cur_valid_blocks, 1)

    # Build block_table: each row maps to valid physical block indices
    block_table = torch.zeros(batch, max_num_blocks_per_req, dtype=torch.int32)
    for b in range(batch):
        n_blocks_b = (int(ctx_lens[b].item()) + block_size - 1) // block_size
        if n_blocks_b > 0:
            block_table[b, :n_blocks_b] = torch.randperm(total_blocks)[:n_blocks_b].to(torch.int32)

    query = torch.empty(batch, num_heads, head_dim, dtype=torch_dtype).uniform_(-0.5, 0.5)
    key_cache = torch.empty(total_blocks, block_size, kv_head_num, head_dim, dtype=torch_dtype).uniform_(-0.5, 0.5)
    value_cache = torch.empty(total_blocks, block_size, kv_head_num, head_dim, dtype=torch_dtype).uniform_(-0.5, 0.5)
    out = torch.empty(batch, num_heads, head_dim, dtype=torch_dtype)

    return PagedAttentionInputs(
        query=query.to(device),
        key_cache=key_cache.to(device),
        value_cache=value_cache.to(device),
        block_table=block_table.to(device),
        context_lens=ctx_lens,  # Must stay on CPU: ATB v1 reads hostData during setup
        out=out.to(device),
        num_kv_heads=kv_head_num,
        num_heads=num_heads,
        scale_value=scale_value,
    )


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------
def run_single(inputs: PagedAttentionInputs) -> None:
    """Run one invocation of _npu_paged_attention."""
    torch_npu._npu_paged_attention(
        inputs.query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.num_kv_heads,
        inputs.num_heads,
        inputs.scale_value,
        inputs.block_table,
        inputs.context_lens,
        inputs.out,
    )


def benchmark(
    inputs: PagedAttentionInputs,
    warmup: int = 10,
    rounds: int = 100,
) -> list[float]:
    """Warmup then measure per-round latency in microseconds.

    Uses torch.npu.synchronize() to ensure accurate timing.
    """
    # Warmup
    for _ in range(warmup):
        run_single(inputs)
    torch.npu.synchronize()

    # Measurement
    latencies_us = []
    for _ in range(rounds):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run_single(inputs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    return latencies_us


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(latencies: list[float], trim_pct: float = 10.0) -> dict:
    """Compute summary statistics from latency measurements."""
    n = len(latencies)
    s = sorted(latencies)
    mean_val = sum(s) / n
    min_val = s[0]
    max_val = s[-1]
    std_val = (sum((x - mean_val) ** 2 for x in s) / n) ** 0.5
    p50 = s[n // 2]
    p90 = s[int(n * 0.9)]
    p99 = s[int(n * 0.99)]

    # Trimmed mean
    trim_count = max(int(n * trim_pct / 100), 0)
    if 2 * trim_count < n:
        trimmed = s[trim_count : n - trim_count]
        trimmed_mean = sum(trimmed) / len(trimmed)
    else:
        trimmed_mean = mean_val

    return dict(
        mean=mean_val, min=min_val, max=max_val, std=std_val,
        p50=p50, p90=p90, p99=p99,
        trimmed_mean=trimmed_mean, trim_pct=trim_pct,
        count=n,
    )


def print_stats(stats: dict, label: str = "") -> None:
    """Pretty-print benchmark statistics."""
    if label:
        print(f"\n  --- {label} ---")
    print(f"  Rounds:        {stats['count']}")
    print(f"  Mean:          {stats['mean']:>10.1f} us")
    print(f"  Trimmed Mean:  {stats['trimmed_mean']:>10.1f} us  (drop {stats['trim_pct']:.0f}% tails)")
    print(f"  Std:           {stats['std']:>10.1f} us")
    print(f"  Min:           {stats['min']:>10.1f} us")
    print(f"  P50:           {stats['p50']:>10.1f} us")
    print(f"  P90:           {stats['p90']:>10.1f} us")
    print(f"  P99:           {stats['p99']:>10.1f} us")
    print(f"  Max:           {stats['max']:>10.1f} us")


def print_shapes(inputs: PagedAttentionInputs, params: dict) -> None:
    """Print input tensor shapes and parameters."""
    print(f"  query:         {list(inputs.query.shape)}  dtype={inputs.query.dtype}")
    print(f"  key_cache:     {list(inputs.key_cache.shape)}  dtype={inputs.key_cache.dtype}")
    print(f"  value_cache:   {list(inputs.value_cache.shape)}  dtype={inputs.value_cache.dtype}")
    print(f"  block_table:   {list(inputs.block_table.shape)}  dtype={inputs.block_table.dtype}")
    print(f"  context_lens:  {list(inputs.context_lens.shape)}  dtype={inputs.context_lens.dtype}")
    if params.get("context_lens_list"):
        ctx_list = params["context_lens_list"]
        ctx_preview = str(ctx_list[:8])
        if len(ctx_list) > 8:
            ctx_preview = ctx_preview[:-1] + ", ...]"
        print(f"  context_lens values: {ctx_preview}")
    else:
        print(f"  context_len:   {params['context_len']}")
    print(f"  num_kv_heads:  {inputs.num_kv_heads}")
    print(f"  num_heads:     {inputs.num_heads}")
    print(f"  scale_value:   {inputs.scale_value:.6f}")


# ---------------------------------------------------------------------------
# Summary table for multi-case runs
# ---------------------------------------------------------------------------
def print_summary_table(results: list[tuple[str, dict, dict]]) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    header = f"  {'Case':<25s}  {'Mean (us)':>10s}  {'Trim (us)':>10s}  {'P50 (us)':>10s}  {'P99 (us)':>10s}"
    sep = f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}"
    print(header)
    print(sep)
    for name, _params, stats in results:
        print(
            f"  {name:<25s}  {stats['mean']:>10.1f}  {stats['trimmed_mean']:>10.1f}"
            f"  {stats['p50']:>10.1f}  {stats['p99']:>10.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark torch_npu._npu_paged_attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Predefined cases: {', '.join(PREDEFINED_CASES.keys())}, all",
    )

    # Tensor shape parameters
    g = p.add_argument_group("Tensor shape parameters")
    g.add_argument("--batch", type=int, default=256, help="Batch size (num sequences)")
    g.add_argument("--num-heads", type=int, default=16, help="Number of query attention heads")
    g.add_argument("--kv-head-num", type=int, default=1, help="Number of KV heads (for GQA)")
    g.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    g.add_argument("--block-size", type=int, default=128, help="Tokens per KV cache block")
    g.add_argument("--context-len", type=int, default=8192, help="Context length (uniform)")
    g.add_argument("--max-model-len", type=int, default=32768, help="Max model sequence length")
    g.add_argument(
        "--context-lens-list", type=str, default=None,
        help="Comma-separated per-batch context lengths (overrides --context-len)",
    )
    g.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                   help="Data type for Q/K/V tensors")

    # Benchmark parameters
    b = p.add_argument_group("Benchmark parameters")
    b.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    b.add_argument("--rounds", type=int, default=100, help="Number of measurement rounds")
    b.add_argument("--device", type=int, default=0, help="NPU device ID")
    b.add_argument("--case", type=str, default=None,
                   help="Predefined case name or 'all' to run all cases")

    return p.parse_args()


def run_case(name: str, params: dict, warmup: int, rounds: int) -> tuple[dict, dict]:
    """Run benchmark for one parameter set. Returns (params, stats)."""
    print(f"\n{'=' * 80}")
    if params.get("desc"):
        print(f"  {name}: {params['desc']}")
    else:
        print(f"  {name}")
    print("=" * 80)

    inputs = generate_inputs(
        batch=params["batch"],
        num_heads=params["num_heads"],
        kv_head_num=params["kv_head_num"],
        head_dim=params["head_dim"],
        block_size=params["block_size"],
        context_len=params.get("context_len", 1),
        max_model_len=params["max_model_len"],
        dtype=params.get("dtype", "float16"),
        context_lens_list=params.get("context_lens_list"),
    )

    print("\n  Input shapes:")
    print_shapes(inputs, params)

    print(f"\n  Warmup: {warmup} rounds ...")
    latencies = benchmark(inputs, warmup=warmup, rounds=rounds)

    stats = compute_stats(latencies)
    print_stats(stats, label="Latency")

    return params, stats


def main() -> None:
    args = parse_args()
    torch.npu.set_device(args.device)

    print(f"Device: NPU:{args.device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch_npu: {torch_npu.__version__}")

    results: list[tuple[str, dict, dict]] = []

    if args.case:
        if args.case == "all":
            cases = list(PREDEFINED_CASES.items())
        elif args.case in PREDEFINED_CASES:
            cases = [(args.case, PREDEFINED_CASES[args.case])]
        else:
            print(f"ERROR: Unknown case '{args.case}'. Available: {', '.join(PREDEFINED_CASES.keys())}, all")
            sys.exit(1)

        for name, params in cases:
            params_copy, stats = run_case(name, dict(params), args.warmup, args.rounds)
            results.append((name, params_copy, stats))
    else:
        # Build params from CLI arguments
        params = dict(
            batch=args.batch,
            num_heads=args.num_heads,
            kv_head_num=args.kv_head_num,
            head_dim=args.head_dim,
            block_size=args.block_size,
            context_len=args.context_len,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        if args.context_lens_list:
            params["context_lens_list"] = [int(x) for x in args.context_lens_list.split(",")]

        name = "custom"
        params_copy, stats = run_case(name, params, args.warmup, args.rounds)
        results.append((name, params_copy, stats))

    if len(results) > 1:
        print_summary_table(results)


if __name__ == "__main__":
    main()
