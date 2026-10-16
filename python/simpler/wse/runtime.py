# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute generated PTO-ISA kernels; torch is used only for storage and streams."""

import ctypes
import json
import subprocess
from pathlib import Path

from .deployment import MockBatcher, file_hash, require


def build_device_library(artifact, pto_isa, *, compiler="bisheng"):
    artifact, pto_isa = Path(artifact).resolve(), Path(pto_isa).resolve()
    graph = json.loads((artifact / "program.json").read_text())
    sources = []
    for node in graph["nodes"]:
        if node["kind"] == "input":
            continue
        idx = node["id"]
        lines = [f'#include "op_{idx}.cpp"']
        if node["kind"] == "all_gather":
            types = ["half", "half", "int64_t", "int32_t", "int32_t", "int32_t", "int32_t"]
        else:
            src = "half" if node["kind"] == "gemm" else "float"
            dst = "half" if node["dtype"] == "f16" else "float"
            types = [src, src, dst, "int32_t"]
        host_params = ", ".join(f"void *p{i}" for i in range(len(types)))
        cast_names = ", ".join(f"({t} *)p{i}" for i, t in enumerate(types))
        lines += [
            f'extern "C" void launch_{idx}(uint32_t blocks, void *stream, {host_params}) {{',
            f"  wse_op_{idx}<<<blocks, nullptr, stream>>>({cast_names});",
            "}",
        ]
        source = artifact / f"launch_{idx}.cpp"
        source.write_text("\n".join(lines) + "\n")
        sources.append(source)
    # PTOAS emits complete global kernels, each with its own preamble.
    # Keep them in separate translation units and launch the emitted entry.
    (artifact / "kernels.cpp").unlink(missing_ok=True)
    command = [
        compiler,
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-O2",
        "-std=c++17",
        f"-I{pto_isa / 'include'}",
        *(str(source) for source in sources),
        "-o",
        str(artifact / "kernels.so"),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (artifact / "bisheng.log").write_text(result.stdout + result.stderr)
    (artifact / "bisheng_command.json").write_text(json.dumps(command, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"bisheng failed ({result.returncode}): {result.stderr[-5000:]}")
    receipt = {
        "schema": "wse.a3.build.v1",
        "files": {
            name: file_hash(artifact / name)
            for name in ("kernels.so", "program.json", "manifest.json", "binding_schema.json")
        },
        "generatedSources": {p.name: file_hash(p) for p in sorted(artifact.glob("op_*.cpp"))},
        "ptoIsaWseHeaders": {p.name: file_hash(p) for p in sorted((pto_isa / "include/pto/wse").glob("*.hpp"))},
    }
    (artifact / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return artifact / "kernels.so"


class A3Runtime:
    """Mock Batcher orchestration with actual native A3 compute and DMA kernels."""

    def __init__(self, artifact, device=0):
        # Deployment-only users do not need the optional NPU runtime packages.
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: F401, PLC0415

        self.torch = torch
        torch.npu.set_device(device)
        self.device = torch.device(f"npu:{device}")
        require("Ascend910_93" in torch.npu.get_device_name(device), "This runner requires actual A3 silicon")
        self.artifact = Path(artifact).resolve()
        self.graph = json.loads((self.artifact / "program.json").read_text())
        self.manifest = json.loads((self.artifact / "manifest.json").read_text())
        require((self.artifact / "build_receipt.json").is_file(), "Build the artifact with build_device_library first")
        self.build_receipt = json.loads((self.artifact / "build_receipt.json").read_text())
        require(
            self.build_receipt["schema"] == "wse.a3.build.v1"
            and set(self.build_receipt["files"])
            == {"kernels.so", "program.json", "manifest.json", "binding_schema.json"}
            and all(file_hash(self.artifact / name) == digest for name, digest in self.build_receipt["files"].items()),
            "Device library or compiler metadata differs from its successful build receipt",
        )
        self.lib = ctypes.CDLL(str(self.artifact / "kernels.so"))
        self.batcher = MockBatcher(self.artifact)
        self.snapshots = {}
        self.statuses = []
        self.binary_hash = file_hash(self.artifact / "kernels.so")

    def _launch(self, node, args):
        fn = getattr(self.lib, f"launch_{node['id']}")
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * len(args)
        fn.restype = None
        for tensor in args:
            require(
                tensor.device == self.device and tensor.is_contiguous(),
                "Kernel arguments must be local contiguous NPU tensors",
            )
        stream = self.torch.npu.current_stream(self.device)._as_parameter_
        fn(self.graph["workers"], stream, *(ctypes.c_void_p(t.data_ptr()) for t in args))

    def _upload_private(self, host):
        itemsize = host.element_size()
        allocation = self.torch.empty(host.numel() + 2048 // itemsize, dtype=host.dtype, device=self.device)
        offset = ((-allocation.data_ptr()) % 2048) // itemsize
        private = allocation[offset : offset + host.numel()].reshape(host.shape)
        private.copy_(host)
        require(private.data_ptr() % 2048 == 0, "Private local DRAM backing must be page aligned")
        return private

    def run(self, inputs, bundle, *, capture=True):
        import numpy as np  # noqa: PLC0415

        torch = self.torch
        require(
            all(file_hash(self.artifact / name) == digest for name, digest in self.build_receipt["files"].items()),
            "Artifacts changed after the device library was loaded",
        )
        require(bundle["binaryHash"] == self.binary_hash, "Deployment differs from the already loaded device binary")
        self.batcher.install(bundle)
        self.batcher.begin_launch()
        self.snapshots, self.statuses = {}, []
        mapping = bundle["mapping"]
        mapping_dev = torch.tensor(mapping, dtype=torch.int32, device=self.device)
        storage = {}
        for node in self.graph["nodes"]:
            rows, cols = node["shape"]
            dtype = torch.float16 if node["dtype"] == "f16" else torch.float32
            itemsize = 2 if dtype == torch.float16 else 4
            padded = ((rows * cols * itemsize + 2047) // 2048) * 2048 // itemsize
            host = torch.full((self.graph["route_nodes"], padded), float("nan"), dtype=dtype)
            if node["kind"] == "input":
                name = node["attrs"]["name"]
                require(name in inputs, f"Missing input {name}")
                value = torch.as_tensor(inputs[name])
                require(
                    tuple(value.shape) == tuple(node["attrs"]["global_shape"]) and value.dtype == dtype,
                    f"Input {name} shape/dtype differs from compiled contract",
                )
                require(torch.isfinite(value).all().item(), f"Input {name} contains nonfinite values")
                for logical, physical in enumerate(mapping):
                    shard = (
                        value
                        if node["attrs"]["placement"] == "replicated"
                        else value[:, logical * cols : (logical + 1) * cols]
                    )
                    host[physical, : rows * cols] = shard.reshape(-1)
            storage[node["id"]] = self._upload_private(host)
            if node["kind"] == "input":
                continue
            sources = [storage[i] for i in node["inputs"]]
            if node["kind"] == "all_gather":
                slot = node["attrs"]["plan_slot"]
                pair = bundle["pairs"][slot]
                call = self.manifest["calls"][slot]
                members = [None] * self.graph["workers"]
                for group in call["groups"]:
                    for logical in group:
                        members[logical] = group
                control = self.batcher.activate(slot)
                routes = torch.from_numpy(np.frombuffer(bytes.fromhex(pair["routes"]), dtype=np.int64).copy()).to(
                    self.device
                )
                group_dev = torch.tensor(members, dtype=torch.int32, device=self.device)
                ctrl_dev = torch.tensor(control, dtype=torch.int32, device=self.device)
                status = torch.full((self.graph["workers"], 16), -1, dtype=torch.int32, device=self.device)
                self._launch(node, [sources[0], storage[node["id"]], routes, mapping_dev, group_dev, ctrl_dev, status])
                self.batcher.drain(slot, torch.npu.synchronize)
                recorded = status.cpu()[:, :6].tolist()
                for logical, row in enumerate(recorded):
                    expected = [0, call["expectedRxBytes"], control[4], pair["bindingId"], logical, mapping[logical]]
                    require(
                        row == expected,
                        f"Device Calendar status mismatch: worker={logical}, actual={row}, expected={expected}",
                    )
                self.statuses.append({"slot": slot, "control": control, "records": recorded})
            else:
                rhs = sources[1] if len(sources) == 2 else sources[0]
                self._launch(node, [sources[0], rhs, storage[node["id"]], mapping_dev])
                torch.npu.synchronize()
            if capture or node["id"] == self.graph["output"]:
                values = storage[node["id"]].cpu()
                inactive = sorted(set(range(self.graph["route_nodes"])) - set(mapping))
                require(torch.isnan(values[inactive]).all().item(), f"Kernel {node['id']} wrote an unowned node")
                require(
                    torch.isnan(values[:, rows * cols :]).all().item(), f"Kernel {node['id']} overran a private slot"
                )
                ordered = torch.stack([values[p, : rows * cols].reshape(rows, cols) for p in mapping])
                require(
                    torch.isfinite(ordered).all().item(), f"Kernel {node['id']} produced nonfinite or unwritten data"
                )
                self.snapshots[node["id"]] = ordered.numpy().copy()
        output = self.snapshots[self.graph["output"]]
        spans = self.graph["nodes"][self.graph["output"]]["layout"]
        order = sorted(range(len(mapping)), key=lambda logical: spans[logical][0])
        return np.concatenate([output[logical] for logical in order], axis=1)
