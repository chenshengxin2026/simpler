# Kernel and Orchestration Configuration

from pathlib import Path

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer
# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 4,
	"orch_thread_num": 1,
	"block_dim": 24,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "paged_attention.cpp"),
	"function_name": "aicpu_orchestration_entry"
}

KERNELS = [
	{"func_id": 0, "source": str(_ROOT_DIR / "kernels" / "aiv" / "kernel_init_inplace.cpp"), "core_type": "aiv"},
	{"func_id": 1, "source": str(_ROOT_DIR / "kernels" / "aic" / "kernel_qk_matmul.cpp"), "core_type": "aic"},
	{"func_id": 2, "source": str(_ROOT_DIR / "kernels" / "aiv" / "kernel_softmax_prepare.cpp"), "core_type": "aiv"},
	{"func_id": 3, "source": str(_ROOT_DIR / "kernels" / "aic" / "kernel_pv_matmul.cpp"), "core_type": "aic"},
	{"func_id": 4, "source": str(_ROOT_DIR / "kernels" / "aiv" / "kernel_online_update.cpp"), "core_type": "aiv"},
]
