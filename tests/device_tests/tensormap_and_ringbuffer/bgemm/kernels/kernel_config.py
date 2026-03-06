"""
Kernel configuration for BGEMM (tensormap_and_ringbuffer Runtime).

Cube core (AIC) for matrix multiplication, Vector core (AIV) for element-wise addition.
"""

import sys
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_GOLDEN_PATH = _KERNELS_ROOT.parent / "golden.py"

# Import TILE_SIZE from golden.py
sys.path.insert(0, str(_GOLDEN_PATH.parent))
try:
    from golden import TILE_SIZE
finally:
    sys.path.pop(0)

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "bgemm_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "GEMM",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_gemm_tile.cpp"),
        "core_type": "aic",
        "compile_flags": [f"-DTILE_SIZE={TILE_SIZE}"],
    },
    {
        "func_id": 1,
        "name": "ADD",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_tile_add.cpp"),
        "core_type": "aiv",
        "compile_flags": [f"-DTILE_SIZE={TILE_SIZE}"],
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
