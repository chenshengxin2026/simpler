# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validated mock PG, deterministic mapping/NoC stubs and Batcher lifecycle."""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class MockPG:
    rows: int = 6
    cols: int = 8
    bad_compute: tuple = ()
    bad_routers: tuple = ()
    bad_links: tuple = ()

    def __post_init__(self):
        require(self.rows in (5, 6) and self.cols == 8, "PG requires a 5x8 or 6x8 mesh")
        require(
            all(type(n) is int and 0 <= n < self.nodes for n in self.bad_compute + self.bad_routers),
            "PG defect node is outside topology",
        )
        for a, b in self.bad_links:
            require(
                type(a) is int and type(b) is int and 0 <= a < self.nodes and b in self.neighbours(a),
                "PG defect link is not a mesh edge",
            )

    @property
    def nodes(self):
        return self.rows * self.cols

    def neighbours(self, node):
        r, c = divmod(node, self.cols)
        return {
            rr * self.cols + cc
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= rr < self.rows and 0 <= cc < self.cols
        }

    def snapshot(self):
        return {
            "rows": self.rows,
            "cols": self.cols,
            "bad_compute": list(self.bad_compute),
            "bad_routers": list(self.bad_routers),
            "bad_links": [list(e) for e in self.bad_links],
        }


def mapping_stub(workers, pg, policy):
    require(workers == 32, "Mapping stub currently implements the 32-worker FFN footprint")
    require(policy in ("identity", "reverse"), f"Unknown mapping policy: {policy}")
    starts = range(pg.rows - 3) if policy == "identity" else range(pg.rows - 4, -1, -1)
    bad = set(pg.bad_compute) | set(pg.bad_routers)
    for row in starts:
        candidates = list(range(row * 8, (row + 4) * 8))
        if not bad.intersection(candidates):
            return candidates if policy == "identity" else candidates[::-1]
    raise ValueError("PG has no usable contiguous 4x8 compute footprint for mapping stub")


def validate_tree(pg, nodes, source):
    nodes = set(nodes)
    require(source in nodes and not nodes.intersection(pg.bad_routers), "Route misses source or uses failed router")
    visited, pending = set(), [source]
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend((pg.neighbours(node) & nodes) - visited)
    edges = {tuple(sorted((n, m))) for n in nodes for m in pg.neighbours(n) & nodes}
    require(visited == nodes and len(edges) == len(nodes) - 1, "P bitmap must induce a connected acyclic tree")
    require(not edges.intersection(tuple(sorted(e)) for e in pg.bad_links), "Route crosses a failed link")
    return edges


class NoCStub:
    """One call plans all groups; straight induced trees and rank-based TDM slots."""

    def __init__(self):
        self.calls = 0
        self.last_input = None

    def compile(self, *, topology, groups, semantics):
        self.calls += 1
        self.last_input = {"topology": topology.snapshot(), "groups": groups, "semantics": semantics}
        result = {}
        for slot, partition in groups.items():
            require(semantics[slot]["opcode"] == 1 and semantics[slot]["redOp"] == 0, "Stub only supports AllGather")
            routes, schedule, resources = {}, [], {}
            for group in partition:
                rows, cols = {p // 8 for p in group}, {p % 8 for p in group}
                require(len(rows) == 1 or len(cols) == 1, "NoC stub requires collinear groups")
                step = 1 if len(rows) == 1 else 8
                passed = set(range(min(group), max(group) + 1, step))
                for rank, source in enumerate(group):
                    edges = validate_tree(topology, passed, source)
                    occupied = resources.setdefault(rank, set())
                    footprint = {("node", p) for p in passed} | {("link", e) for e in edges}
                    require(not occupied.intersection(footprint), "Stub TDM schedule has a resource conflict")
                    occupied.update(footprint)
                    lands = set(group) - {source}
                    bits = sum((2 | int(p in lands)) << (2 * p) for p in passed)
                    routes[source] = bits
                    schedule.append({"source": source, "slot": rank, "tree": sorted(passed)})
            result[slot] = {
                "routes": routes,
                "slots": {
                    "planSlot": slot,
                    "opcode": 1,
                    "mode": "TDM-stub",
                    "period": max(len(g) for g in partition),
                    "schedule": schedule,
                    "backgroundPolicy": "no-background-in-this-fixture",
                },
            }
        return result


def encode_entry(bits, expected, nodes):
    require(nodes in (40, 48) and 0 <= bits < (1 << (nodes * 2)), "Route bits exceed profile width")
    require(0 <= expected < 2**32, "Expected receive bytes overflow uint32")
    payload = bits.to_bytes(nodes // 4, "little")
    if nodes == 40:
        lands = sum(bool(bits & (1 << (2 * p))) for p in range(nodes))
        payload += bytes([lands, 1])
    return payload + struct.pack("<I", expected)


def decode_entry(entry, pg, source):
    require(len(entry) == 16, "Route entry must be 16 bytes")
    bits = int.from_bytes(entry[: pg.nodes // 4], "little")
    pairs = [(bits >> (2 * p)) & 3 for p in range(pg.nodes)]
    require(1 not in pairs, "Illegal landing without transit bit")
    passed = {p for p, pair in enumerate(pairs) if pair & 2}
    lands = {p for p, pair in enumerate(pairs) if pair & 1}
    validate_tree(pg, passed, source)
    if pg.nodes == 40:
        require(entry[10] == len(lands) and entry[11] == 1, "Invalid v2 metadata")
    return passed, lands, struct.unpack_from("<I", entry, 12)[0]


def prepare_deployment(artifact, pg, *, policy="identity", solver=None):
    artifact = Path(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    require(manifest["target"]["nodes"] == pg.nodes, "Binary route ABI differs from PG topology")
    mapping = mapping_stub(manifest["logicalWorkers"], pg, policy)
    calls = manifest["calls"]
    groups = {c["planSlot"]: [[mapping[logical] for logical in g] for g in c["groups"]] for c in calls}
    semantics = {
        c["planSlot"]: {
            "opcode": c["opcode"],
            "redOp": c["redOp"],
            "dtype": c["dtype"],
            "dependencies": c["dependencies"],
            "allGroupsConcurrent": True,
        }
        for c in calls
    }
    solver = solver or NoCStub()
    plans = solver.compile(topology=pg, groups=groups, semantics=semantics)
    pairs = []
    for call in calls:
        slot = call["planSlot"]
        routes = [bytes(16)] * pg.nodes
        for source, bits in plans[slot]["routes"].items():
            routes[source] = encode_entry(bits, call["expectedRxBytes"], pg.nodes)
        image = b"".join(routes).hex()
        slots = dict(plans[slot]["slots"], routesHash=digest(image))
        pairs.append({"planSlot": slot, "bindingId": slot + 1, "routes": image, "slots": slots})
    bundle = {
        "schema": "wse.deployment.v1",
        "binaryHash": file_hash(artifact / "kernels.so"),
        "manifestHash": digest(manifest),
        "programHash": file_hash(artifact / "program.json"),
        "bindingSchemaHash": file_hash(artifact / "binding_schema.json"),
        "pg": pg.snapshot(),
        "mappingPolicy": policy,
        "mapping": mapping,
        "pairs": pairs,
        "solver": "collinear-induced-tree-stub-v1",
        "nocInput": solver.last_input,
        "solverCalls": 1,
    }
    bundle["seal"] = digest(bundle)
    validate_bundle(artifact, bundle)
    return bundle


def validate_bundle(artifact, bundle):
    artifact = Path(artifact)
    require(bundle.get("seal") == digest({k: v for k, v in bundle.items() if k != "seal"}), "Bundle checksum mismatch")
    manifest = json.loads((artifact / "manifest.json").read_text())
    require(bundle["binaryHash"] == file_hash(artifact / "kernels.so"), "Binary hash mismatch")
    require(bundle["manifestHash"] == digest(manifest), "Manifest hash mismatch")
    require(bundle["programHash"] == file_hash(artifact / "program.json"), "Program hash mismatch")
    require(bundle["bindingSchemaHash"] == file_hash(artifact / "binding_schema.json"), "Binding schema mismatch")
    graph = json.loads((artifact / "program.json").read_text())
    allocations = [
        {
            "tensor": n["id"],
            "bytes": ((n["shape"][0] * n["shape"][1] * (2 if n["dtype"] == "f16" else 4) + 2047) // 2048) * 2048,
        }
        for n in graph["nodes"]
    ]
    require(
        allocations == manifest["localAllocations"]
        and sum(a["bytes"] for a in allocations) == manifest["localDramBytesPerWorker"],
        "Local DRAM allocation accounting mismatch",
    )
    limits = {"UB": 192 * 1024, "L1": 512 * 1024, "L0A": 64 * 1024, "L0B": 64 * 1024, "L0C": 128 * 1024}
    require(
        all(0 <= manifest["scratchHighWaterBytes"][key] <= cap for key, cap in limits.items()),
        "Scratch memory exceeds the A3/WSE common resource budget",
    )
    p = bundle["pg"]
    pg = MockPG(
        p["rows"], p["cols"], tuple(p["bad_compute"]), tuple(p["bad_routers"]), tuple(map(tuple, p["bad_links"]))
    )
    require(pg.nodes == manifest["target"]["nodes"], "Route profile mismatch")
    mapping = bundle["mapping"]
    require(
        len(mapping) == manifest["logicalWorkers"] and len(set(mapping)) == len(mapping), "Mapping is not injective"
    )
    require(all(type(p) is int and 0 <= p < pg.nodes for p in mapping), "Physical node out of range")
    require(not set(mapping).intersection(pg.bad_compute + pg.bad_routers), "Mapping uses unusable compute node")
    calls = manifest["calls"]
    require(
        bundle["schema"] == "wse.deployment.v1" and bundle["solverCalls"] == 1,
        "Unsupported deployment schema or multiple NoC planning calls",
    )
    require([c["planSlot"] for c in calls] == list(range(len(calls))), "Plan slots must be contiguous")
    require(len(bundle["pairs"]) == len(calls), "Missing route/slot pair")
    for call, pair in zip(calls, bundle["pairs"]):
        require(call["opcode"] == 1 and call["redOp"] == 0, "Unsupported opcode/redOp")
        require(
            call["recvOffset"] >= call["rows"] * call["srcStride"] and call["recvOffset"] % 32 == 0,
            "Asymmetric or overlapping UB buffers",
        )
        require(
            call["capacity"] == call["rows"] * call["recvStride"]
            and call["recvOffset"] + call["capacity"] <= min(192 * 1024, 384 * 1024),
            "UB capacity violation",
        )
        require(
            call["srcStride"] == call["rowBytes"]
            and call["rowBytes"] % 32 == 0
            and call["recvStride"] == call["rowBytes"] * call["groupSize"],
            "Unsupported DMA geometry",
        )
        expected = call["rows"] * call["rowBytes"] * (call["groupSize"] - 1)
        require(call["expectedRxBytes"] == expected, "Incorrect incoming byte count")
        require(
            pair["planSlot"] == pair["slots"]["planSlot"] == call["planSlot"]
            and pair["bindingId"] == call["planSlot"] + 1
            and pair["slots"]["opcode"] == call["opcode"]
            and pair["slots"]["routesHash"] == digest(pair["routes"]),
            "Route/slot pair mismatch",
        )
        image = bytes.fromhex(pair["routes"])
        require(len(image) == pg.nodes * 16, "Route table size mismatch")
        require(
            all(image[p * 16 : (p + 1) * 16] == bytes(16) for p in set(range(pg.nodes)) - set(mapping)),
            "Inactive route entries must be zero",
        )
        require(
            sorted(logical for g in call["groups"] for logical in g) == list(range(len(mapping))),
            "Incomplete group partition",
        )
        scheduled = {entry["source"]: entry for entry in pair["slots"]["schedule"]}
        require(
            len(pair["slots"]["schedule"]) == len(mapping) and set(scheduled) == set(mapping),
            "Incomplete or duplicate slot installation",
        )
        require(
            pair["slots"]["period"] == call["groupSize"]
            and pair["slots"]["mode"] == "TDM-stub"
            and pair["slots"]["backgroundPolicy"] == "no-background-in-this-fixture",
            "Unsupported stub slot policy or period",
        )
        occupied = {}
        for group in call["groups"]:
            require(len(group) == call["groupSize"], "Nonuniform groups")
            for rank, logical in enumerate(group):
                physical = mapping[logical]
                passed, lands, received = decode_entry(image[physical * 16 : (physical + 1) * 16], pg, physical)
                require(
                    lands == {mapping[source] for source in group if source != logical},
                    "Route lands at incorrect group members",
                )
                require(received == expected, "Route receive amount mismatch")
                require(call["selfOffsets"][str(logical)] == rank * call["rowBytes"], "Invalid logical rank offset")
                entry = scheduled[physical]
                require(entry["tree"] == sorted(passed) and entry["slot"] == rank, "Slot/route geometry mismatch")
                used = occupied.setdefault(rank, set())
                require(not used.intersection(passed), "Concurrent stub paths overlap")
                used.update(passed)
    require(manifest["localDramBytesPerWorker"] <= 384 * 1024 * 1024, "Local DRAM budget exceeded")
    return True


class MockBatcher:
    """Atomic host installation and stream-drained Calendar wave lifecycle."""

    def __init__(self, artifact):
        self.artifact = Path(artifact)
        self.bundle = None
        self.inflight = False
        self.active_slot = None
        self.launch_open = False
        self.completed = set()
        self.launch_id = 0
        self.epoch = 0
        self.trace = []

    def install(self, bundle):
        require(not self.inflight, "Cannot install while Calendar traffic is in flight")
        validate_bundle(self.artifact, bundle)
        self.bundle = json.loads(json.dumps(bundle))
        self.launch_open = False
        self.trace.append({"event": "install_ack", "seal": bundle["seal"]})

    def begin_launch(self):
        require(self.bundle is not None and not self.inflight, "Launch requires installed, drained domain")
        validate_bundle(self.artifact, self.bundle)
        self.launch_id += 1
        self.completed.clear()
        self.launch_open = True
        return self.launch_id

    def activate(self, slot):
        require(not self.inflight, "Slot replacement requires global drain")
        require(self.launch_open, "Activate requires begin_launch")
        require(slot == len(self.completed), "Collective dependency order violation")
        require(self.bundle is not None and 0 <= slot < len(self.bundle["pairs"]), "Unknown plan slot")
        self.epoch = self.epoch % 255 + 1
        pair = self.bundle["pairs"][slot]
        self.inflight = True
        self.active_slot = slot
        self.trace.append(
            {
                "event": "activate",
                "planSlot": slot,
                "epoch": self.epoch,
                "launch": self.launch_id,
                "bindingId": pair["bindingId"],
            }
        )
        return [pair["bindingId"], slot, 1, 1, self.epoch, self.launch_id]

    def drain(self, slot, synchronize):
        require(self.inflight and self.active_slot == slot, "No matching active Calendar wave")
        synchronize()
        self.inflight = False
        self.active_slot = None
        self.completed.add(slot)
        if len(self.completed) == len(self.bundle["pairs"]):
            self.launch_open = False
        self.trace.append({"event": "global_drain_ack", "planSlot": slot, "launch": self.launch_id})
