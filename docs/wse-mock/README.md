# WSE mock: deployment and A3 FFN runner

This branch adds `simpler.wse`, a host mock runtime for the WSE Calendar
simulation. The upstream baseline is
`97f8b6c92d9cbd6b8270dfa918272c964a7c25e1`. The PR targets the fork's `wse-mock`
branch; review code is on `wse-mock-calendar-a3`.

## Changes and execution boundary

| File | Role |
| ---- | ---- |
| [deployment.py](../../python/simpler/wse/deployment.py) | MockPG, mapping/NoC stubs, binding validation, atomic mock installation and drain |
| [runtime.py](../../python/simpler/wse/runtime.py) | Native library build, receipt checks, private tensor slots, A3 kernel launch and status validation |
| [build_ffn.py](../../examples/wse/build_ffn.py) | PyPTO graph → native PTOAS → PTO-ISA/bisheng → `kernels.so` |
| [validate_a3.py](../../examples/wse/validate_a3.py) | Independent CPU golden, intermediate/gather checks, mapping/repeat invariance, JSON/NPZ evidence |
| [test_deployment.py](../../tests/wse/test_deployment.py) | Binding rejection, route/slot geometry, lifecycle and epoch-wrap tests |

The new Python runner leaves the existing native `Worker/ChipWorker` scheduler
unchanged. PyTorch NPU supplies storage, copies, and streams. GEMM, activation,
cast, and collective data movement execute in native A3 PTO kernels.

Mapping chooses a routable 4×8 footprint. One NoC stub invocation plans all
B/C groups and dependencies without receiving tensor shapes or byte counts.
The binder checks routes, landing sets, slots, offsets, capacities, hashes,
and failed resources. `MockBatcher` installs only on a drained domain and
orders activation. The [runtime reference](../wse-calendar-a3.md) specifies
the contract and mock boundaries.

## Prepare four matching checkouts

While the PRs are open, clone `wse-mock-calendar-a3` from each fork. After
merging all four PRs, `wse-mock` becomes the matching integration branch.

```bash
mkdir -p wse-workspace/repos
cd wse-workspace
for repo in pypto simpler PTOAS pto-isa; do
  git clone --branch wse-mock-calendar-a3 \
    "https://github.com/chenshengxin2026/$repo.git" "repos/$repo"
done
git -C repos/pypto submodule update --init --recursive
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Use Python 3.10+, a C++ compiler, CMake/Ninja, nanobind, pytest/xdist, NumPy,
CPU PyTorch with matching `torch_npu`, and an A3 CANN environment containing
`bisheng`. The validated environment used Python 3.10.19, NumPy 2.2.6,
torch/torch_npu 2.9.0, nanobind 2.11.0, and CANN 9.0.0 on aarch64.
Install missing Python dependencies only into this venv.

Build PyPTO's native extension and the modified native PTOAS from their
checkouts, following their root build guides. PTOAS requires its compatible
LLVM/MLIR 19 toolchain and CANN CMake dependency. No native simpler rebuild
is needed for this isolated Python host runner.

```bash
export WSE_WORKSPACE="$PWD"
export PYTHONPATH="$WSE_WORKSPACE/repos/pypto/python:$WSE_WORKSPACE/repos/simpler/python:$WSE_WORKSPACE/repos/pypto/examples/models${PYTHONPATH:+:$PYTHONPATH}"
export WSE_PTOAS="$WSE_WORKSPACE/repos/PTOAS/build/tools/ptoas/ptoas"
export WSE_PTO_ISA="$WSE_WORKSPACE/repos/pto-isa"
```

Use these modified checkouts explicitly; an installed upstream PTOAS cannot
parse the new WSE operations. Per-layer interfaces and baselines:
[PyPTO](https://github.com/chenshengxin2026/pypto/blob/wse-mock-calendar-a3/docs/en/dev/wse-mock/README.md),
[PTOAS](https://github.com/chenshengxin2026/PTOAS/blob/wse-mock-calendar-a3/docs/wse-mock/README.md),
[PTO-ISA](https://github.com/chenshengxin2026/pto-isa/blob/wse-mock-calendar-a3/docs/wse-mock/README.md).

## Build and validate

From the workspace directory, with the venv and CANN environment active:

```bash
mkdir -p artifacts
(
  cd repos/pypto
  source .claude/skills/testing/load-env.sh
  cd "$WSE_WORKSPACE"
  python -m pytest -q --noconftest -c /dev/null -n "$PYPTO_TEST_JOBS" \
    repos/pypto/tests/ut/wse/test_frontend.py \
    repos/simpler/tests/wse/test_deployment.py \
    repos/PTOAS/test/wse/test_calendar.py
)
g++ -std=c++20 -O2 -Irepos/pto-isa/include \
  repos/pto-isa/tests/cpu/wse_calendar_contract.cpp -o artifacts/wse-contract
artifacts/wse-contract
python repos/simpler/examples/wse/build_ffn.py \
  --case original --nodes 48 --ptoas "$WSE_PTOAS" --pto-isa "$WSE_PTO_ISA" \
  --output artifacts/original-48
```

Use `--case small` for `[T,H,I]=[2,512,512]`; `--nodes 40` explicitly selects
the legacy route profile. Each output directory contains generated PTO/C++,
compiler logs, metadata, `kernels.so`, and a successful build receipt.

On a shared host with `task-submit`, reserve a device and pass its ID explicitly:

```bash
repos/simpler/.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device auto --timeout 600 --max-time 600 --run \
  'cd "$WSE_WORKSPACE" &&
   export ASCEND_PROCESS_LOG_PATH="$WSE_WORKSPACE/artifacts/device-logs/$TASKQUEUE_TASK_ID" &&
   mkdir -p "$ASCEND_PROCESS_LOG_PATH" &&
   .venv/bin/python repos/simpler/examples/wse/validate_a3.py \
     --artifact artifacts/original-48 --device "$TASK_DEVICE" --seed 1014 --repeat 2'
```

Repeat with `--seed 2026` for the second original-case seed. Each invocation
runs identity, reverse, and a PG with its first compute row disabled, reusing
the same binary. `--report-dir` can preserve previous evidence separately.
On a dedicated A3 without this queue, invoke the same Python command directly
with the explicitly reserved device ID; that invocation has no queue isolation.

The report records input/binary hashes, shape, seed, device, output errors,
per-worker Calendar status, mappings, and installation/drain traces. Golden
values never enter device computation. Output assembly concatenates column
shards on the host for export; it adds no third device AllGather.

## Recorded validation and limits

On 2026-10-15, Ascend910_9392 passed 21 FFN executions:

| Case | Coverage | Result |
| ---- | -------- | ------ |
| `8,7168,3072`, 48 nodes | 2 seeds × 3 deployments × 2 repeats | 12/12 |
| `2,512,512`, 48 nodes | 1 seed × 3 deployments × 2 repeats | 6/6 |
| `2,512,512`, 40 nodes | 1 seed × 3 deployments | 3/3 |

The 46 targeted CPU/PTOAS tests and separate C++ receiver contract also passed.
Original FFN maximum absolute error was `0.015625`; maximum relative RMS
was `8.2343e-5`. Thresholds were fixed at pointwise `atol=0.002, rtol=0.01`
and relative RMS `<0.002`. Both gathers were bit-exact against device source
shards for every worker; final output was bit-exact across mappings/repeats
for a fixed seed. Original B/C remote committed bytes totaled `1,523,712`
per run across 32 workers.

On 2026-10-20, the portable scripts in this PR rebuilt the small 48-node case
and passed all three deployments on A3 with seed 1014. The 46 targeted
CPU/PTOAS tests and separate C++ receiver contract passed again.

Supported execution is static 2D FFN with 32 logical workers. Private local
DRAM has 2 KiB pages and a 384 MiB per-worker budget; scratch respects the
smaller of WSE and A3 capacities. Status records use 64-byte cache-line
isolation. Default ABI 3 has 48 nodes/96 route bits; ABI 2 has 40 nodes/80 bits.
Both use 16-byte route entries and cannot be mixed.

PG acquisition, Batcher registers, and new CCE instructions are mocked.
Mapping and NoC planning are stubs, with unsupported topologies rejected.
The A3 transport uses native PTO copies between private backing slots and
host stage drains; this is a functional model without WSE fabric timing,
congestion, retry generation, or performance claims.
