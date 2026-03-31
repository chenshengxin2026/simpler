"""
Kernel configuration for dispatch_throughput test (tensormap_and_ringbuffer).

Measures scheduler throughput by submitting N noop tasks serially.

Kernels:
  func_id=0: kernel_noop_aic (AIC) - empty cube kernel, increments counter
  func_id=1: kernel_noop_aiv (AIV) - empty vector kernel, increments counter
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "dispatch_throughput_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "NOOP_AIC",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_noop_aic.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "NOOP_AIV",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_noop_aiv.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
