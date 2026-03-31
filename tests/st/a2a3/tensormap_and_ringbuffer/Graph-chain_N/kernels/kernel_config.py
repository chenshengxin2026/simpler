"""
Kernel configuration for chain_N test (tensormap_and_ringbuffer).

Linear dependency chain: seed -> Task_0 -> Task_1 -> ... -> Task_{N-1} -> result.
Uses a single AIV increment kernel (out = in + 1.0).

Kernels:
  func_id=0: kernel_inc_aiv (AIV) - reads input, writes output = input + 1.0
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "chain_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "INC",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_inc_aiv.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
