# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the PTO-ISA two-member probe using the same deployment as compiled kernels."""

import argparse
import ctypes
import json

import torch
import torch_npu  # noqa: F401
from simpler.wse import Deployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    lib = ctypes.CDLL(args.library)
    lib.launch_smoke.argtypes = [ctypes.c_void_p] * 4
    lib.launch_smoke.restype = None
    source = torch.arange(64, dtype=torch.float16).reshape(2, 2, 16)
    golden = torch.cat(list(source), dim=1)
    x = source.to(device)
    for reverse in (False, True):
        deployment = Deployment.stub((2, 9), reverse=reverse).install(device)
        for _ in range(3):
            launch = deployment.prepare((2, 16))
            out = torch.empty((2, 2, 32), dtype=torch.float16, device=device)
            stream = torch.npu.current_stream()
            lib.launch_smoke(stream._as_parameter_, x.data_ptr(), out.data_ptr(), launch.contexts.data_ptr())
            stream.synchronize()
            status = launch.check()
            assert torch.equal(out.cpu(), golden.expand(2, -1, -1))
            print(json.dumps({"reverse": reverse, "status": status, "bit_exact": True}), flush=True)


if __name__ == "__main__":
    main()
