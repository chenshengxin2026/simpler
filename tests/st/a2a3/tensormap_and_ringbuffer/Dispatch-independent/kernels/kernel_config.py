"""
Kernel configuration for task_scaling test (tensormap_and_ringbuffer).

Measures dispatch overhead growth as task count scales from 1 to 1000.

Kernels:
  func_id=0: kernel_noop_aic (AIC) - trivial write kernel
  func_id=1: kernel_noop_aiv (AIV) - trivial write kernel
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "task_scaling_orch.cpp"),
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
