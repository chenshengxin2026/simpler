"""
Golden script for chain_N test (linear dependency chain).

Builds a chain of N tasks where each adds 1.0 to its input:
  seed(0.0) -> Task_0 -> Task_1 -> ... -> Task_{N-1} -> result
  result = N.0

Measures dependency chain resolution overhead vs chain length.

Cases sweep chain length: 4, 8, 16, 32, 64.

Args layout: [seed, result, chain_len]
"""

import ctypes
import torch

__outputs__ = ["result"]

RTOL = 1e-5
ATOL = 1e-5

ALL_CASES = {
    "Chain4":  {"chain_len": 4},
    "Chain8":  {"chain_len": 8},
    "Chain16": {"chain_len": 16},
    "Chain32": {"chain_len": 32},
    "Chain64": {"chain_len": 64},
}

DEFAULT_CASE = "Chain32"


def generate_inputs(params: dict) -> list:
    chain_len = params["chain_len"]

    seed = torch.zeros(1, dtype=torch.float32)
    result = torch.zeros(1, dtype=torch.float32)

    return [
        ("seed", seed),
        ("result", result),
        ("chain_len", ctypes.c_int64(chain_len)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    chain_len = params["chain_len"]
    result = torch.as_tensor(tensors["result"])
    result[0] = float(chain_len)
