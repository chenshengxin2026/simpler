# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end FFN validation, with CPU golden and device collective snapshots."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from simpler.wse import A3Runtime, MockPG, prepare_deployment


def metrics(actual, expected, *, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rms_relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.01)
        assert rms_relative < 0.002, f"RMS relative error {rms_relative} >= 0.002"
    return {
        "maxAbs": float(np.abs(delta).max()),
        "rmsRelative": rms_relative,
        "exact": bool(np.array_equal(actual, expected)),
    }


def make_inputs(shape, seed):
    t, h, i = shape
    rng = np.random.default_rng(seed)
    return {
        "X": rng.normal(0, 0.5, (t, h)).astype(np.float16),
        "W_gate": rng.normal(0, 0.05, (h, i)).astype(np.float16),
        "W_up": rng.normal(0, 0.05, (h, i)).astype(np.float16),
        "W_down": rng.normal(0, 0.05, (i, h)).astype(np.float16),
    }


def golden(inputs):
    x, wg, wu, wd = (torch.from_numpy(inputs[k]).float() for k in ("X", "W_gate", "W_up", "W_down"))
    gate, up = x @ wg, x @ wu
    z = (torch.nn.functional.silu(gate) * up).half()
    y = (z.float() @ wd).half()
    return {"gate": gate.numpy(), "up": up.numpy(), "z": z.numpy(), "y": y.numpy()}


def check_stages(runtime, expected):
    snapshots = runtime.snapshots
    results = {}
    for idx, label in ((4, "gate"), (5, "up"), (8, "z")):
        full = np.concatenate(list(snapshots[idx]), axis=1)
        results[label] = metrics(full, expected[label])
    for slot, idx in ((0, 9), (1, 10)):
        input_id = runtime.graph["nodes"][idx]["inputs"][0]
        src = snapshots[input_id]
        dst = snapshots[idx]
        for members in runtime.manifest["calls"][slot]["groups"]:
            correct = np.concatenate([src[logical] for logical in members], axis=1)
            for logical in members:
                metrics(dst[logical], correct, exact=True)
        results[f"calendar_{slot}"] = {
            "all32WorkersBitExact": True,
            "remoteBytes": sum(row[1] for row in runtime.statuses[slot]["records"]),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", type=int, required=True, help="Device reserved for this run")
    parser.add_argument("--report-dir", type=Path, help="Default: artifact directory")
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if "TASK_DEVICE" in os.environ and args.device != int(os.environ["TASK_DEVICE"]):
        parser.error("--device must match the task-submit locked device")
    torch.set_num_threads(2)
    out = Path(args.artifact).resolve()
    report_dir = args.report_dir.resolve() if args.report_dir else out
    report_dir.mkdir(exist_ok=True, parents=True)
    runtime = A3Runtime(out, args.device)
    graph = runtime.graph
    shape = [
        graph["nodes"][0]["shape"][0],
        graph["nodes"][0]["shape"][1],
        graph["nodes"][1]["attrs"]["global_shape"][1],
    ]
    inputs = make_inputs(shape, args.seed)
    expected = golden(inputs)
    report = {
        "status": "RUNNING",
        "task": os.environ.get("TASKQUEUE_TASK_ID"),
        "device": args.device,
        "soc": torch.npu.get_device_name(args.device),
        "shape": shape,
        "seed": args.seed,
        "binaryHash": runtime.binary_hash,
        "tolerance": {"atol": 0.002, "rtol": 0.01, "rmsRelativeLimit": 0.002},
        "inputHashes": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in inputs.items()},
        "runs": [],
    }
    report_path = report_dir / f"validation-seed-{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    rows = graph["route_nodes"] // 8
    configs = [
        ("identity", MockPG(rows=rows), "identity"),
        ("reverse", MockPG(rows=rows), "reverse"),
        (
            "pg_bad_compute_row",
            MockPG(rows=rows, bad_compute=tuple(range(8))),
            "identity",
        ),
    ]
    reference_output = None
    try:
        for name, pg, policy in configs:
            bundle = prepare_deployment(out, pg, policy=policy)
            (report_dir / f"deployment-{name}.json").write_text(json.dumps(bundle, indent=2) + "\n")
            for repetition in range(args.repeat):
                start = time.monotonic()
                actual = runtime.run(inputs, bundle)
                if reference_output is None:
                    reference_output = actual.copy()
                mapping_invariance = metrics(actual, reference_output, exact=True)
                result = {
                    "deployment": name,
                    "repetition": repetition,
                    "mapping": bundle["mapping"],
                    "secondsIncludingValidation": time.monotonic() - start,
                    "output": metrics(actual, expected["y"]),
                    "mappingAndRepeatInvariance": mapping_invariance,
                    "stages": check_stages(runtime, expected),
                    "deviceCalendarStatus": runtime.statuses,
                    "lifecycle": runtime.batcher.trace.copy(),
                }
                report["runs"].append(result)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                np.savez_compressed(
                    report_dir / f"ffn-{name}-seed-{args.seed}.npz",
                    actual=actual,
                    golden=expected["y"],
                )
                print(
                    json.dumps(
                        {
                            "deployment": name,
                            "repeat": repetition,
                            "output": result["output"],
                        }
                    ),
                    flush=True,
                )
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
