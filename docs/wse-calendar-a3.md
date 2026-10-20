# WSE Calendar A3 simulation runtime

`simpler.wse` provides a host-side mock Batcher, deployment validation, and a
native A3 kernel runner for the experimental `pypto.wse` frontend. It does not
modify or claim support in the existing `Worker`/`ChipWorker` scheduler.
PyTorch NPU tensors provide allocation, copies, and streams; GEMM, activation,
casts, and the two gathers execute in generated PTO-ISA kernels.
The builder compiles each emitted global kernel in a separate translation unit,
adds a host launch shim, and links one shared library. A successful build receipt
binds that library to its graph, manifest, and binding schema. The hardware
runner rejects metadata or binary changes after compilation/loading.

## Deployment boundary

`prepare_deployment(artifact, MockPG(...), policy=...)` consumes the compiler's
manifest and produces a bundle for the existing `kernels.so`. PG explicitly
distinguishes bad compute cells, bad routers, and bad links.

The mapping stub chooses an intact contiguous 4x8 footprint for 32 logical
workers. `identity` and `reverse` preserve logical ownership while changing
physical placement. It rejects layouts the stub cannot route; it is not a
general placement optimizer.

One `NoCStub.compile(topology=..., groups=..., semantics=...)` invocation
receives all B/C groups and the B-to-C dependency. Shapes, strides, capacities,
and byte counts are not passed to this solver. It returns one route/slot pair
per plan slot, using collinear induced trees and rank-based TDM. Groups within
one collective have disjoint concurrent footprints. No bandwidth, slot-cycle,
background-flow, RG fairness, or congestion model is claimed.

The deployment binder fills expected receive counts after planning. Each
slot image includes the paired route image hash. The bundle binds PG, mapping,
routes, slots, binary, graph, manifest, and binding schema with SHA-256 checksums.
Checksums detect mismatched artifacts; they are not authentication signatures.
Validation also checks induced-tree connectivity/acyclicity, failed resources,
landing sets, member completeness, logical offsets, inactive entries, slot
period/conflicts, private DRAM accounting, and scratch limits.

## Installation and execution

`MockBatcher.install` validates before publishing an immutable copy. Installation
and slot replacement require a drained domain. `begin_launch` assigns a new
launch generation. `activate` advances the nonzero 8-bit epoch and enforces
collective dependency order. `drain` synchronizes the device stream before
recording completion. Failed synchronization leaves the wave in flight.
The launch generation and global drain prevent old waves from overlapping an
epoch wrap. The CPU receiver model separately tests stale-packet rejection.

`A3Runtime.run` allocates separate 2 KiB-aligned, page-padded local backing
slots for every tensor and physical node. Compute helpers access only their
own slot. Only the mock transport adapter reads peer slots; these hold the
completed source tensor of the preceding stage. This avoids waiting for a
logical worker that A3 has not scheduled yet. A3 can therefore time-multiplex
32 logical workers without pretending they are 32 simultaneous physical cores.

Each Calendar receiver arms its context before mock deliveries. Native
`TLOAD`/`TINSERT`/`TSTORE` operations move the data. Device code checks logical
member offsets, landing bits, epoch, installation, launch identity, and committed
remote bytes. Local copies do not contribute to the remote-byte counter.
The host requires every worker's six status fields to match the manifest.
Status records occupy independent 64-byte cache lines to prevent A3 scalar
GM writeback from corrupting adjacent workers' records.

The default validation captures every intermediate, checks inactive/padding
storage for writes, and reconstructs final column shards on the CPU. CPU output
assembly is an export operation, not another device collective. Golden tensors
are calculated separately and never passed into device computation.

## Running and testing

Build the modified PyPTO/PTOAS/PTO-ISA checkouts and make their Python sources
available together with this checkout. Then use
`examples/wse/build_ffn.py` and `examples/wse/validate_a3.py`, following the
[cross-repository run guide](wse-mock/README.md).
The runner must receive the queue-selected device, for example
`--device "$TASK_DEVICE"`, inside `task-submit --device auto --run '...'`.

`tests/wse/test_deployment.py` checks mapping, one-shot NoC planning, both ABI
profiles, corrupted bundles, atomic installation, dependency ordering, failed
drain, and epoch wrap. It uses a fake binary only as a checksum fixture;
hardware validation requires the real bisheng-built library.

The A3 adapter validates functional data flow and modeled hardware constraints.
It does not validate physical WSE PG reads, Batcher registers, CCE instruction
encodings, NoC performance, or the real fabric's retry/epoch protocol.
