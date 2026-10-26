# WSE Calendar adapter on A3

`python/simpler/wse/` provides the deployment and launch support for the
[`wse-sim-1026` PyPTO FFN](https://github.com/chenshengxin2026/pypto/blob/wse-sim-1026/docs/en/dev/wse-a3.md).
Use that guide to download and build all four matching branches, then run:

```bash
bash pypto/examples/models/run_wse_a3.sh "$(command -v python)"
```

The checkouts must be siblings. All device work in the script uses
`task-submit`; the script compiles the PyPTO example and native kernels before
submitting three seeds with forward and reversed physical launch orders.

`Deployment.stub(members, reverse=...)` represents the mapping and route/slot
generator as a black box. Logical member order determines AllGather output
order; physical launch order can change independently. Every deployment gets
a distinct synchronization domain. `install()` uploads paired tables once,
and `prepare()` reserves a shared epoch and launch-owned transport buffers.
`CalendarLaunch.check()` verifies each member's completion and remote bytes.

`launch.py` compiles the generated PyPTO `kernel_entry` wrappers unchanged.
Its thin launch shim dispatches one argument row per resident participant.
`descriptor.cpp` packs the existing runtime's actual Tensor ABI. Compiler
metadata validates argument count, shape and dtype before dispatch, including
the extra Calendar context binding. The runner synchronizes the stream before
releasing descriptors, tables or mailboxes. It needs no simpler native extension.

The FFN schedule in `tests/st/a2a3/wse/run_ffn.py` is static and synchronous.
GEMM, activation, casts and collectives execute in generated A3 kernels;
PyTorch handles allocation/copies and a CPU numerical oracle. The A3 mock in
the matching PTO-ISA branch uses actual L1/GM DMA. This does not implement
ChipWorker orchestration, native WSE routing, timing or a performance model.
The repository's default `pto_isa.pin` is unchanged: this adapter always takes
an explicit `--isa-root` pointing to the matching WSE checkout.

CPU-only deployment checks:

```bash
PYTHONPATH=python python -m pytest \
  --confcutdir=tests/ut/py/wse tests/ut/py/wse -q
```
