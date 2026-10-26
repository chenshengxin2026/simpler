# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 execution adapter for synchronous WSE Calendar kernels.

Mapping and table construction are deliberate stubs. Payload movement and
completion take place on the device in pto-isa's CCE adapter.
"""

from dataclasses import dataclass
from itertools import count

_group_ids = count(1)
_epochs = count(1)


@dataclass(frozen=True)
class Deployment:
    """Logical group order and a separately chosen physical launch order."""

    members: tuple[int, ...]
    launch_order: tuple[int, ...]
    sync_group: int
    routes: tuple[int, ...]
    slots: tuple[int, ...]

    @classmethod
    def stub(cls, members: tuple[int, ...], *, reverse: bool = False) -> "Deployment":
        if not isinstance(members, tuple) or not 1 <= len(members) <= 24:
            raise ValueError("A3 resident groups require 1..24 members")
        if any(type(bid) is not int or not 0 <= bid < 2**31 for bid in members):
            raise ValueError("Members must be nonnegative int32 logical IDs")
        if len(set(members)) != len(members):
            raise ValueError("Members must be distinct")
        group = next(_group_ids)
        # These tokens are opaque to the compiler. Only the CCE mock interprets
        # route masks; a hardware mapper can replace this paired table builder.
        routes = tuple(((1 << len(members)) - 1) ^ (1 << rank) for rank in range(len(members)))
        slots = tuple(group * 32 + rank for rank in range(len(members)))
        return cls(members, members[::-1] if reverse else members, group, routes, slots)

    def install(self, device: str) -> "InstalledDeployment":
        """Upload the paired tables once, outside the launch path."""
        return InstalledDeployment(self, device)


class InstalledDeployment:
    def __init__(self, deployment: Deployment, device: str):
        import torch  # noqa: PLC0415 - device dependencies load only at execution

        self.deployment = deployment
        self.device = device
        self.routes = torch.tensor(deployment.routes, dtype=torch.int32).to(device)
        self.slots = torch.tensor(deployment.slots, dtype=torch.int32).to(device)

    def prepare(self, shape: tuple[int, int]) -> "CalendarLaunch":
        return CalendarLaunch(self, shape, next(_epochs))


class CalendarLaunch:
    """Own all allocations until the stream completes and status is checked."""

    def __init__(self, installed: InstalledDeployment, shape: tuple[int, int], epoch: int):
        import torch  # noqa: PLC0415 - device dependencies load only at execution

        deployment, device = installed.deployment, installed.device
        rows, cols = shape
        size = len(deployment.members)
        if rows <= 0 or cols <= 0 or cols % 16 or rows * cols * 2 * size > 128 * 1024:
            raise ValueError("Calendar requires packed FP16 rows and at most 128 KiB receive storage")
        self.deployment = deployment
        self.epoch = epoch
        self.send_bytes = rows * cols * 2
        self.routes, self.slots = installed.routes, installed.slots
        self.mailboxes = torch.empty((size, self.send_bytes), dtype=torch.uint8, device=device)
        self.signals = torch.zeros((size, 8), dtype=torch.int64).to(device)
        self.status = torch.full((size, 8), -1, dtype=torch.int64).to(device)
        contexts = []
        for bid in deployment.launch_order:
            rank = deployment.members.index(bid)
            contexts.append(
                [
                    bid,
                    rank,
                    deployment.sync_group,
                    epoch,
                    self.routes.data_ptr(),
                    self.slots.data_ptr(),
                    self.mailboxes.data_ptr(),
                    self.signals.data_ptr(),
                    self.status.data_ptr(),
                    size,
                    self.send_bytes,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        self.contexts = torch.tensor(contexts, dtype=torch.int64).to(device)

    def context_address(self, physical_index: int) -> int:
        return self.contexts.data_ptr() + physical_index * 128

    def check(self) -> list[list[int]]:
        rows = self.status.cpu().tolist()
        expected = (len(self.deployment.members) - 1) * self.send_bytes
        for rank, row in enumerate(rows):
            want = [0, expected, self.epoch, self.deployment.members[rank], self.deployment.sync_group]
            if row[:5] != want:
                raise RuntimeError(f"Calendar rank {rank}: status {row[:5]}, expected {want}")
        return [row[:5] for row in rows]


__all__ = ["Deployment", "InstalledDeployment", "CalendarLaunch"]
