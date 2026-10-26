# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build and launch unchanged PyPTO kernel_entry wrappers on A3.

This adapter launches resident core groups directly. It deliberately has no
distributed scheduler: the example supplies a static sequence of kernel calls.
"""

import ctypes
import json
import subprocess
from pathlib import Path

from . import CalendarLaunch


def _include_dirs(simpler_root: Path, isa_root: Path) -> list[str]:
    return [
        f"-I{isa_root / 'include'}",
        f"-I{simpler_root / 'src/a2a3/runtime/host_build_graph/runtime'}",
        f"-I{simpler_root / 'src/common'}",
        f"-I{simpler_root / 'src/common/task_interface'}",
    ]


def build_kernel(source: Path, output: Path, *, simpler_root: Path, isa_root: Path, calendar: bool = False) -> Path:
    """Compile a generated kernel, preserving the compiler's Tensor dispatch ABI."""
    if source.parent.name not in ("aic", "aiv"):
        raise ValueError("Expected a generated kernels/aic or kernels/aiv source")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source.parents[2] / "compiled_meta.json").read_text())
    (output / "signature.json").write_text(
        json.dumps({"params": metadata["lowered_params"], "calendar": calendar}, indent=2) + "\n"
    )
    arch = "dav-c220-cube" if source.parent.name == "aic" else "dav-c220-vec"
    shim = output / "launch.cpp"
    shim.write_text(
        "#include <pto/pto-inst.hpp>\n"
        "#if defined(__CCE_AICORE__)\n"
        f'#include "{source.resolve().as_posix()}"\n'
        "#endif\n"
        'extern "C" __global__ AICORE void wse_entry(__gm__ int64_t* args, uint32_t stride) {\n'
        "#if defined(__CCE_AICORE__)\n"
        "  kernel_entry(args + get_block_idx() * stride);\n"
        "  pipe_barrier(PIPE_ALL);\n"
        "#endif\n"
        "}\n"
        'extern "C" void wse_launch(void* stream, void* args, uint32_t stride, uint32_t blocks) {\n'
        "  wse_entry<<<blocks, nullptr, stream>>>(static_cast<int64_t*>(args), stride);\n"
        "}\n"
    )
    library = output / "kernel.so"
    command = [
        "bisheng",
        "-fPIC",
        "-shared",
        "-xcce",
        f"--cce-aicore-arch={arch}",
        "-O2",
        "-std=c++17",
        *_include_dirs(simpler_root, isa_root),
        str(shim),
        "-o",
        str(library),
    ]
    (output / "build-command.txt").write_text("\n".join(command) + "\n")
    with (output / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return library


def build_descriptor(output: Path, *, simpler_root: Path, isa_root: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    library = output / "descriptor.so"
    subprocess.run(
        [
            "g++",
            "-fPIC",
            "-shared",
            "-O2",
            "-std=c++17",
            *_include_dirs(simpler_root, isa_root),
            str(Path(__file__).with_name("descriptor.cpp")),
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


class Runner:
    """Synchronous owner of Tensor descriptors, dispatch arrays and Calendar status."""

    def __init__(self, descriptor_library: Path, device: str):
        import torch  # noqa: PLC0415 - device dependencies load only at execution
        import torch_npu  # noqa: F401, PLC0415

        torch.npu.set_device(device)
        self.device = device
        self.abi = ctypes.CDLL(str(descriptor_library.resolve()))
        self.abi.wse_tensor_size.restype = ctypes.c_size_t
        self.abi.wse_pack_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.abi.wse_pack_tensor.restype = None
        self.libraries: dict[Path, ctypes.CDLL] = {}

    def run(self, library: Path, arguments: list[list], *, calendar: CalendarLaunch | None = None):
        import torch  # noqa: PLC0415 - device dependencies load only at execution

        if not arguments or not 1 <= len(arguments) <= 24:
            raise ValueError("A launch requires 1..24 resident participants")
        argc = len(arguments[0])
        signature = json.loads(library.with_name("signature.json").read_text())
        if argc != len(signature["params"]) or signature["calendar"] != (calendar is not None):
            raise ValueError("Argument count or Calendar binding does not match the compiled kernel")
        if any(len(row) != argc for row in arguments):
            raise ValueError("All participants must have the same signature")
        if calendar is not None and len(arguments) != len(calendar.deployment.members):
            raise ValueError("Calendar group and launch sizes differ")
        size = self.abi.wse_tensor_size()
        storage = torch.zeros((len(arguments) * argc, size), dtype=torch.uint8)
        for index, tensor in enumerate(tensor for row in arguments for tensor in row):
            parameter = signature["params"][index % argc]
            if (
                tensor.ndim != 2
                or tensor.dtype not in (torch.float16, torch.float32)
                or not tensor.is_contiguous()
                or str(tensor.device) != self.device
            ):
                raise ValueError("Kernel arguments must be contiguous FP16/FP32 matrices on the selected NPU")
            dtype = "fp16" if tensor.dtype == torch.float16 else "fp32"
            if list(tensor.shape) != parameter["shape"] or dtype != parameter["dtype"]:
                raise ValueError(f"Argument {parameter['name']} does not match the compiled shape/dtype")
            self.abi.wse_pack_tensor(
                storage.data_ptr() + index * size,
                tensor.data_ptr(),
                *tensor.shape,
                tensor.element_size(),
            )
        descriptors = storage.to(self.device)
        dispatch = []
        for physical in range(len(arguments)):
            row = [descriptors.data_ptr() + (physical * argc + arg) * size for arg in range(argc)]
            if calendar is not None:
                row.append(calendar.context_address(physical))
            dispatch.append(row)
        args = torch.tensor(dispatch, dtype=torch.int64).to(self.device)
        path = library.resolve()
        if path not in self.libraries:
            lib = ctypes.CDLL(str(path))
            lib.wse_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
            lib.wse_launch.restype = None
            self.libraries[path] = lib
        stream = torch.npu.current_stream()
        self.libraries[path].wse_launch(stream._as_parameter_, args.data_ptr(), len(dispatch[0]), len(arguments))
        stream.synchronize()
        return calendar.check() if calendar is not None else None
