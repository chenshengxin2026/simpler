# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build the PyPTO -> native PTOAS -> PTO-ISA -> bisheng FFN artifacts."""

import argparse
import json
from pathlib import Path

from simpler.wse import build_device_library
from wse_calendar_ffn import build_ffn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("small", "original"), default="original")
    parser.add_argument("--nodes", type=int, choices=(40, 48), default=48)
    parser.add_argument("--ptoas", type=Path, required=True, help="Modified native PTOAS executable")
    parser.add_argument("--pto-isa", type=Path, required=True, help="Modified PTO-ISA checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="bisheng")
    args = parser.parse_args()
    shape = (2, 512, 512) if args.case == "small" else (8, 7168, 3072)
    out = args.output.resolve()
    program = build_ffn(*shape)
    program.compile(out, ptoas=args.ptoas.resolve(), nodes=args.nodes)
    library = build_device_library(out, args.pto_isa, compiler=args.compiler)
    print(json.dumps({"artifact": str(out), "library": str(library), "shape": shape}))


if __name__ == "__main__":
    main()
