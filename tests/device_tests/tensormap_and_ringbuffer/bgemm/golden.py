"""
BGEMM Golden Implementation (tensormap_and_ringbuffer Runtime)

Computation: C = A @ B (tiled matrix multiplication)

User-configurable parameters (per case):
  TILE_SIZE   - tile dimension (recommended power of 2)
  INCORE_NUM  - number of incores (must be even), BATCH = INCORE_NUM / 2

Note: All cases must use the same TILE_SIZE (kernel compile-time constant).

Fixed: GRID_K = 2

Args layout: [ptr_A, ptr_B, ptr_C, config, size_A, size_B, size_C]
Config tensor: [TILE_SIZE, INCORE_NUM, GRID_K]
"""

import ctypes
import torch

__outputs__ = ["C"]

RTOL = 1e-3
ATOL = 1e-3

GRID_K = 2

ALL_CASES = {
    "Case1": {
        "TILE_SIZE": 128,  # INCORE_SIZE
        "INCORE_NUM": 64,  # INCORE_NUM == GRID_K * batch
    }
}

DEFAULT_CASE = "Case1"

# Extract TILE_SIZE from first case for compile-time constant
# All cases must use the same TILE_SIZE (enforced at runtime)
TILE_SIZE = ALL_CASES[DEFAULT_CASE]["TILE_SIZE"]

# Validate that all cases use the same TILE_SIZE
for case_name, case_params in ALL_CASES.items():
    if case_params["TILE_SIZE"] != TILE_SIZE:
        raise ValueError(
            f"All cases must use the same TILE_SIZE. "
            f"Case '{case_name}' has TILE_SIZE={case_params['TILE_SIZE']}, "
            f"but expected {TILE_SIZE} (from '{DEFAULT_CASE}')"
        )


def generate_inputs(params: dict) -> list:
    """Generate input tensors with tile-first memory layout."""
    tile_size = params["TILE_SIZE"]
    incore_num = params["INCORE_NUM"]

    tile = tile_size
    batch = incore_num // 2

    config = torch.tensor(
        [tile_size, incore_num, GRID_K],
        dtype=torch.int64,
    )

    A = torch.randn(batch, GRID_K, tile, tile, dtype=torch.float32) * 0.01
    B = torch.randn(batch, GRID_K, tile, tile, dtype=torch.float32) * 0.01
    C = torch.zeros(batch, tile, tile, dtype=torch.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
        ("config", config),
        ("size_A", ctypes.c_int64(A_flat.nbytes)),
        ("size_B", ctypes.c_int64(B_flat.nbytes)),
        ("size_C", ctypes.c_int64(C_flat.nbytes)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden result: C[batch] = sum(k) A[batch,k] @ B[batch,k]."""
    tile_size = params["TILE_SIZE"]
    incore_num = params["INCORE_NUM"]

    tile = tile_size
    batch = incore_num // 2

    A = torch.as_tensor(tensors["A"]).reshape(batch, GRID_K, tile, tile)
    B = torch.as_tensor(tensors["B"]).reshape(batch, GRID_K, tile, tile)
    C = torch.as_tensor(tensors["C"]).reshape(batch, tile, tile)

    C[:] = 0.0

    for b in range(batch):
        for k_idx in range(GRID_K):
            C[b] += torch.matmul(A[b, k_idx], B[b, k_idx])

    tensors["C"][:] = C.flatten()


if __name__ == "__main__":
    params = {"name": DEFAULT_CASE, **ALL_CASES[DEFAULT_CASE]}
    result = generate_inputs(params)
    tensors = {name: tensor for name, tensor in result if isinstance(tensor, torch.Tensor)}
    compute_golden(tensors, params)

    tile = params["TILE_SIZE"]
    batch = params["INCORE_NUM"] // 2
    print(f"=== BGEMM Golden Test ({params['name']}) ===")
    print(f"TILE_SIZE={params['TILE_SIZE']}, INCORE_NUM={params['INCORE_NUM']}")
    print(f"TILE={tile}, BATCH={batch}, GRID_K={GRID_K}")

    C = tensors["C"].reshape(batch, tile, tile)
    print(f"Output shape: {C.shape}")
    print(f"Output range: [{C.min():.4f}, {C.max():.4f}]")
    print(f"Output mean: {C.mean():.4f}")
    print("Golden test passed!")
