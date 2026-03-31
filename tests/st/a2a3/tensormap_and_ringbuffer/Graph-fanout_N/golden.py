"""
Golden script for fanout_N test (wide fan-out topology).

1 source task fans out to N independent consumer tasks:
  seed(0.0) -> [Source] -> intermediate(1.0) -> [Consumer_0] -> result[0] = 2.0
                                              -> [Consumer_1] -> result[1] = 2.0
                                              -> ...
                                              -> [Consumer_{N-1}] -> result[N-1] = 2.0

Tests parallel dispatch capability: can the runtime simultaneously issue
N independent tasks that all read from the same source output?

Consumer output slots are cache-line aligned (64B = 16 float32 elements)
to avoid false sharing.

Cases sweep fan-out width: 2, 4, 8, 16, 24.

Args layout: [seed, result, fanout_width]
"""

import ctypes
import torch

__outputs__ = ["result"]

RTOL = 1e-5
ATOL = 1e-5

CACHE_LINE_ELEMS = 16

ALL_CASES = {
    "Fanout2":  {"fanout_width": 2},
    "Fanout4":  {"fanout_width": 4},
    "Fanout8":  {"fanout_width": 8},
    "Fanout15": {"fanout_width": 15},
}

DEFAULT_CASE = "Fanout15"


def generate_inputs(params: dict) -> list:
    fanout_width = params["fanout_width"]

    seed = torch.zeros(1, dtype=torch.float32)
    result = torch.zeros(fanout_width * CACHE_LINE_ELEMS, dtype=torch.float32)

    return [
        ("seed", seed),
        ("result", result),
        ("fanout_width", ctypes.c_int64(fanout_width)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    fanout_width = params["fanout_width"]
    result = torch.as_tensor(tensors["result"])

    # Source output = seed(0.0) + 1.0 = 1.0
    # Each consumer output = source(1.0) + 1.0 = 2.0
    for i in range(fanout_width):
        result[i * CACHE_LINE_ELEMS] = 2.0
