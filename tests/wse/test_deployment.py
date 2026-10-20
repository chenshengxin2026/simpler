# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import copy
import json

import numpy as np
import pytest
from simpler.wse import MockBatcher, MockPG, NoCStub, prepare_deployment, validate_bundle
from simpler.wse.deployment import decode_entry, digest, encode_entry, validate_tree
from wse_calendar_ffn import build_ffn


@pytest.fixture
def artifact(tmp_path):
    path = build_ffn().emit(tmp_path)
    (path / "kernels.so").write_bytes(b"unit-test binary identity; never loaded")
    return path


@pytest.mark.parametrize(
    "pg,policy", [(MockPG(), "identity"), (MockPG(), "reverse"), (MockPG(bad_compute=tuple(range(8))), "identity")]
)
def test_mapping_and_noc_boundary(artifact, pg, policy):
    solver = NoCStub()
    bundle = prepare_deployment(artifact, pg, policy=policy, solver=solver)
    assert solver.calls == 1 and validate_bundle(artifact, bundle)
    assert len(set(bundle["mapping"])) == 32
    text = json.dumps(solver.last_input)
    assert all(key not in text for key in ("rowBytes", "expectedRxBytes", "capacity", "recvStride", "shape"))
    assert bundle["binaryHash"] == prepare_deployment(artifact, MockPG())["binaryHash"]


def test_full_geometry_data_order_under_reversed_mapping(artifact):
    pg = MockPG()
    bundle = prepare_deployment(artifact, pg, policy="reverse")
    mapping = bundle["mapping"]
    manifest = json.loads((artifact / "manifest.json").read_text())
    payload = {mapping[logical]: np.full((8, 96), logical, dtype=np.float16) for logical in range(32)}
    for call, pair in zip(manifest["calls"], bundle["pairs"]):
        result = {}
        for group in call["groups"]:
            expected = np.concatenate([payload[mapping[logical]] for logical in group], axis=1)
            table = bytes.fromhex(pair["routes"])
            for logical in group:
                _, lands, received = decode_entry(
                    table[mapping[logical] * 16 : (mapping[logical] + 1) * 16], pg, mapping[logical]
                )
                assert lands == {mapping[s] for s in group if s != logical}
                assert received == call["expectedRxBytes"]
                result[mapping[logical]] = expected.copy()
        payload = result
    expected = np.repeat(np.arange(32, dtype=np.float16), 96)[None, :].repeat(8, axis=0)
    for value in payload.values():
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize(
    "mode",
    [
        "mapping",
        "expected",
        "pair",
        "tree",
        "offset",
        "slot_conflict",
        "slot_duplicate",
        "slot_period",
        "inactive_route",
    ],
)
def test_corrupted_binding_rejected(artifact, mode):
    b = prepare_deployment(artifact, MockPG())
    if mode == "mapping":
        b["mapping"][0] = b["mapping"][1]
    elif mode == "pair":
        b["pairs"][0]["slots"] = b["pairs"][1]["slots"]
    elif mode == "tree":
        image = bytearray.fromhex(b["pairs"][0]["routes"])
        image[0] |= 1
        b["pairs"][0]["routes"] = image.hex()
        b["pairs"][0]["slots"]["routesHash"] = digest(image.hex())
    elif mode == "expected":
        image = bytearray.fromhex(b["pairs"][0]["routes"])
        image[12] ^= 1
        b["pairs"][0]["routes"] = image.hex()
        b["pairs"][0]["slots"]["routesHash"] = digest(image.hex())
    elif mode == "offset":
        b["pairs"][0]["planSlot"] = 1
    elif mode == "slot_duplicate":
        b["pairs"][0]["slots"]["schedule"].append(b["pairs"][0]["slots"]["schedule"][0])
    elif mode == "slot_period":
        b["pairs"][0]["slots"]["period"] = 1
    elif mode == "inactive_route":
        image = bytearray.fromhex(b["pairs"][0]["routes"])
        image[-1] = 1
        b["pairs"][0]["routes"] = image.hex()
        b["pairs"][0]["slots"]["routesHash"] = digest(image.hex())
    else:
        b["pairs"][0]["slots"]["schedule"][0]["slot"] = 1
    b["seal"] = digest({k: v for k, v in b.items() if k != "seal"})
    with pytest.raises(ValueError):
        validate_bundle(artifact, b)


def test_hash_and_pg_failures(artifact):
    b = prepare_deployment(artifact, MockPG())
    b["mapping"][0] = 100
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(artifact, b)
    with pytest.raises(ValueError, match="footprint"):
        prepare_deployment(artifact, MockPG(bad_compute=tuple(range(24))))
    with pytest.raises(ValueError, match="failed link"):
        prepare_deployment(artifact, MockPG(bad_links=((0, 1),)))
    with pytest.raises(ValueError, match="ABI"):
        prepare_deployment(artifact, MockPG(rows=5))


def test_route_cycle_and_invalid_landing():
    pg = MockPG()
    with pytest.raises(ValueError, match="acyclic"):
        validate_tree(pg, {0, 1, 8, 9}, 0)
    with pytest.raises(ValueError, match="landing"):
        decode_entry(encode_entry(1, 32, 48), pg, 0)
    with pytest.raises(ValueError, match="width"):
        encode_entry(1 << 80, 32, 40)


def test_legacy_abi_is_explicit(tmp_path):
    path = build_ffn().emit(tmp_path, nodes=40)
    (path / "kernels.so").write_bytes(b"cpu-unit-test")
    b = prepare_deployment(path, MockPG(rows=5), policy="reverse")
    assert len(bytes.fromhex(b["pairs"][0]["routes"])) == 40 * 16
    assert validate_bundle(path, b)


def test_batcher_atomic_install_drain_and_epoch_wrap(artifact):
    b = prepare_deployment(artifact, MockPG())
    batcher = MockBatcher(artifact)
    batcher.install(b)
    with pytest.raises(ValueError, match="begin_launch"):
        batcher.activate(0)
    batcher.begin_launch()
    with pytest.raises(ValueError, match="dependency"):
        batcher.activate(1)
    batcher.epoch = 254
    assert batcher.activate(0)[4] == 255
    with pytest.raises(ValueError, match="matching"):
        batcher.drain(1, lambda: None)

    def failed_sync():
        raise RuntimeError("device failed")

    with pytest.raises(RuntimeError, match="device failed"):
        batcher.drain(0, failed_sync)
    assert batcher.inflight and not batcher.completed
    with pytest.raises(ValueError, match="in flight"):
        batcher.install(b)
    with pytest.raises(ValueError, match="drain"):
        batcher.activate(1)
    batcher.drain(0, lambda: None)
    assert batcher.activate(1)[4] == 1
    batcher.drain(1, lambda: None)
    old = copy.deepcopy(batcher.bundle)
    invalid = copy.deepcopy(b)
    invalid["seal"] = "bad"
    with pytest.raises(ValueError):
        batcher.install(invalid)
    assert batcher.bundle == old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
