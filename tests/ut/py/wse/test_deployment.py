# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
from simpler.wse import Deployment


def test_mapping_does_not_change_logical_rows():
    direct = Deployment.stub((9, 2, 17))
    reverse = Deployment.stub((9, 2, 17), reverse=True)
    assert direct.members == reverse.members == (9, 2, 17)
    assert reverse.launch_order == (17, 2, 9)
    assert direct.routes == reverse.routes == (6, 5, 3)
    assert direct.slots != reverse.slots
    assert direct.sync_group != reverse.sync_group


@pytest.mark.parametrize("members", [(), (1, 1), (-1,), (True,), tuple(range(25)), (2**31,)])
def test_invalid_group(members):
    with pytest.raises(ValueError):
        Deployment.stub(members)


def test_singleton_has_no_remote_route():
    assert Deployment.stub((42,)).routes == (0,)
