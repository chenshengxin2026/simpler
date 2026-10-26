# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build PyPTO's WSE FFN artifact, or run and validate it on a locked A3 device."""

import argparse
import hashlib
import json
from pathlib import Path

from simpler.wse import Deployment
from simpler.wse.launch import Runner, build_descriptor, build_kernel


def build(artifact: Path, isa_root: Path):
    simpler_root = Path(__file__).resolve().parents[4]
    manifest = json.loads((artifact / "wse_manifest.json").read_text())
    build_descriptor(artifact / "native", simpler_root=simpler_root, isa_root=isa_root)
    for name, source in manifest["kernels"].items():
        build_kernel(
            artifact / source,
            artifact / "native" / name,
            simpler_root=simpler_root,
            isa_root=isa_root,
            calendar=name in manifest["calendars"],
        )
        print(f"Built {name}", flush=True)


def run(artifact: Path, device_index: int, repeats: int, seed: int):
    import torch  # noqa: PLC0415 - build subcommand requires no device packages

    device = f"npu:{device_index}"
    runner = Runner(artifact / "native/descriptor.so", device)
    manifest = json.loads((artifact / "wse_manifest.json").read_text())
    t, h, intermediate = manifest["shape"]
    members = manifest["members"]
    p = len(members)
    libraries = {name: artifact / "native" / name / "kernel.so" for name in manifest["kernels"]}
    records = []

    def allocate(rows, cols, dtype):
        return torch.empty((rows, cols), dtype=dtype, device=device)

    def gather(name, sources, installed):
        deployment = installed.deployment
        contract = manifest["calendars"][name]
        shape, axis = contract["shape"], contract["axis"]
        result_shape = list(shape)
        result_shape[axis] *= len(deployment.members)
        outputs = {bid: allocate(*result_shape, torch.float16) for bid in deployment.members}
        launch = installed.prepare(tuple(shape))
        arguments = [[sources[bid], outputs[bid]] for bid in deployment.launch_order]
        status = runner.run(libraries[name], arguments, calendar=launch)
        # Use actual device-produced inputs for the bit-exact transport oracle.
        golden = torch.cat([sources[bid].cpu() for bid in deployment.members], dim=axis)
        for output in outputs.values():
            assert torch.equal(output.cpu(), golden), f"{name}: AllGather is not bit exact"
        return outputs, status

    for reverse in (False, True):
        deployments = {
            name: Deployment.stub(tuple(spec["members"]), reverse=reverse).install(device)
            for name, spec in manifest["calendars"].items()
        }
        for iteration in range(repeats):
            generator = torch.Generator().manual_seed(seed + iteration)
            x = (torch.randn((t, h), generator=generator) * 0.2).half()
            wg = (torch.randn((h, intermediate), generator=generator) * 0.15).half()
            wu = (torch.randn((h, intermediate), generator=generator) * 0.15).half()
            wd = (torch.randn((intermediate, h), generator=generator) * 0.15).half()
            gate = x.float() @ wg.float()
            up = x.float() @ wu.float()
            reference_z = (gate / (1 + torch.exp(-gate)) * up).half()
            reference_y = (reference_z.float() @ wd.float()).half()

            local_x = {bid: x.to(device) for bid in members}
            local_wg = {
                bid: wg[:, rank * intermediate // p : (rank + 1) * intermediate // p].contiguous().to(device)
                for rank, bid in enumerate(members)
            }
            local_wu = {
                bid: wu[:, rank * intermediate // p : (rank + 1) * intermediate // p].contiguous().to(device)
                for rank, bid in enumerate(members)
            }
            local_wd = {
                bid: wd[:, rank * h // p : (rank + 1) * h // p].contiguous().to(device)
                for rank, bid in enumerate(members)
            }
            g = {bid: allocate(t, intermediate // p, torch.float32) for bid in members}
            u = {bid: allocate(t, intermediate // p, torch.float32) for bid in members}
            z = {bid: allocate(t, intermediate // p, torch.float16) for bid in members}
            order = members[::-1] if reverse else members
            runner.run(libraries["gate_up"], [[local_x[bid], local_wg[bid], g[bid]] for bid in order])
            runner.run(libraries["gate_up"], [[local_x[bid], local_wu[bid], u[bid]] for bid in order])
            runner.run(libraries["swiglu"], [[g[bid], u[bid], z[bid]] for bid in order])
            actual_z = torch.cat([z[bid].cpu() for bid in members], dim=1)
            torch.testing.assert_close(actual_z, reference_z, rtol=2e-3, atol=2e-5)

            b, full, status = {}, {}, {}
            for name in ("b0", "b1"):
                outputs, status[name] = gather(name, z, deployments[name])
                b.update(outputs)
            for name in ("c0", "c1"):
                outputs, status[name] = gather(name, b, deployments[name])
                full.update(outputs)
            for bid in members:
                assert torch.equal(full[bid].cpu(), actual_z), "B/C composition changed logical order"

            accumulators = {bid: allocate(t, h // p, torch.float32) for bid in members}
            y = {bid: allocate(t, h // p, torch.float16) for bid in members}
            runner.run(libraries["down"], [[full[bid], local_wd[bid], accumulators[bid]] for bid in order])
            runner.run(libraries["output_cast"], [[accumulators[bid], y[bid]] for bid in order])
            actual_y = torch.cat([y[bid].cpu() for bid in members], dim=1)
            torch.testing.assert_close(actual_y, reference_y, rtol=2e-3, atol=1e-4)
            error = actual_y.float() - reference_y.float()

            for name in ("axis0", "singleton"):
                deployment = deployments[name].deployment
                sources = {
                    bid: (torch.arange(32).reshape(2, 16) + 100 * rank + iteration).half().to(device)
                    for rank, bid in enumerate(deployment.members)
                }
                _, status[name] = gather(name, sources, deployments[name])
            record = {
                "reverse_mapping": reverse,
                "iteration": iteration,
                "seed": seed + iteration,
                "max_abs_error": error.abs().max().item(),
                "rms_error": error.square().mean().sqrt().item(),
                "z_max_abs_error": (actual_z.float() - reference_z.float()).abs().max().item(),
                "gather_bit_exact": True,
                "status": status,
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    paths = sorted(artifact.rglob("*.pto")) + sorted(artifact.rglob("*.cpp")) + sorted(artifact.rglob("*.so"))
    report = {
        "device": device,
        "device_name": torch.npu.get_device_name(device_index),
        "torch": torch.__version__,
        "shape": manifest["shape"],
        "members": members,
        "tolerance": {"rtol": 2e-3, "atol": 1e-4},
        "runs": records,
        "sha256": {str(path.relative_to(artifact)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (artifact / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(records)} FFN runs; report: {artifact / 'validation.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--artifact", required=True, type=Path)
    build_parser.add_argument("--isa-root", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--artifact", required=True, type=Path)
    run_parser.add_argument("--device", required=True, type=int)
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=1026)
    args = parser.parse_args()
    if args.command == "build":
        build(args.artifact.resolve(), args.isa_root.resolve())
    else:
        if args.repeats <= 0:
            parser.error("--repeats must be positive")
        run(args.artifact.resolve(), args.device, args.repeats, args.seed)


if __name__ == "__main__":
    main()
