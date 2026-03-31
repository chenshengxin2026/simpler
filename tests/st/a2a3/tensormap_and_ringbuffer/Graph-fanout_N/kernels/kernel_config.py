"""
Kernel configuration for fanout_N test (tensormap_and_ringbuffer).

Fan-out topology: 1 source -> N consumers.
Reuses chain_N's AIV increment kernel.

Kernels:
  func_id=0: kernel_inc_aiv (AIV) - reads input, writes output = input + 1.0
"""

from pathlib import Path

_CHAIN_KERNELS = Path(__file__).parent / ".." / ".." / "Graph-chain_N" / "kernels"

ORCHESTRATION = {
    "source": str(Path(__file__).parent / "orchestration" / "fanout_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "INC",
        "source": str(_CHAIN_KERNELS / "aiv" / "kernel_inc_aiv.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
