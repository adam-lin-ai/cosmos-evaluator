# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LCOV to Cobertura XML format."""

import argparse
from os.path import abspath
from pathlib import Path
import sys

from lcov_cobertura import LcovCobertura


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LCOV to Cobertura XML format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the LCOV coverage report file (i.e. _coverage_report.dat)",
    )
    parser.add_argument(
        "--base_dir",
        "-b",
        default=".",
        type=str,
        help="Root path for all source files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="coverage.xml",
        type=str,
        help="Path to the output file for Cobertura XML",
    )

    args = parser.parse_args()

    # Check if the input file exists
    if not Path(args.input).expanduser().resolve().exists():
        print(f"Error: LCOV input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting LCOV input from {abspath(args.input)} with base directory {abspath(args.base_dir)}")

    with open(args.input, "r", encoding="utf-8") as dat_file:
        lcov_contents = dat_file.read()
    cobertura_xml = LcovCobertura(lcov_data=lcov_contents, base_dir=args.base_dir).convert()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(cobertura_xml)

    print(f"Cobertura XML written to {args.output}")


if __name__ == "__main__":
    main()
