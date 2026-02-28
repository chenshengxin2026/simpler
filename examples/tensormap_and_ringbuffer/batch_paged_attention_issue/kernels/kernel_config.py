"""
Paged Attention Kernel and Orchestration Configuration -- TFILLPAD_INPLACE Bug Reproduction

Modified from the original to support variable block_size {16, 32, 64, 128}
with head_dim=32, enabling the TFILLPAD_INPLACE bug sweep.

Kernels use runtime dispatch on block_size via if/else in kernel_entry.

AIC Kernels (Matrix Multiplication):
  - aic_qk_matmul: Q @ K^T computation (M=16, K=32, N=block_size)
  - aic_pv_matmul: P @ V computation (M=16, K=block_size, N=32)

AIV Kernels (Vector Operations):
  - aiv_softmax_prepare: scale, pad (TFILLPAD_INPLACE only!), rowmax, exp, rowsum
  - aiv_online_update: online softmax accumulation + fused normalization (M=16, N=32)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

# Kernel configs
KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"),       "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"),       "core_type": "aic"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_KERNELS_ROOT / "aic" / "aic_hub.cpp"),       "core_type": "aic"},
    # AIV kernels (vector operations)
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"),   "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"),       "core_type": "aiv"},
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
