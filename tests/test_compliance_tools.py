# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pathlib
import subprocess

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
ALLOWED = {"Apache-2.0", "MIT", "MIT_WITH_LLVM-exception"}

_spec = importlib.util.spec_from_file_location("check_spdx", TOOLS / "check_spdx.py")
check_spdx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_spdx)


@pytest.mark.parametrize(
    "expression",
    [
        "Apache-2.0",
        "Apache-2.0 AND MIT",
        "(Apache-2.0 OR MIT)",
        "Apache-2.0 AND (MIT OR Apache-2.0)",
        "MIT WITH LLVM-exception",
        "Apache-2.0 */",
        "Apache-2.0 -->",
        "Apache-2.0 #}",
    ],
)
def test_valid_expressions_pass(expression: str) -> None:
    check_spdx.check_expression(expression, ALLOWED)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "()",
        "MIT\tAND GPL-3.0-only",
        "MIT/GPL-3.0-only",
        "GPL-3.0-only",
        "Apache-2.0 and MIT",
        "Apache-2.0 AND",
        "AND Apache-2.0",
        "(Apache-2.0",
        "Apache-2.0)",
        "Apache-2.0 MIT",
        "Apache-2.0 WITH LLVM-exception",
        "MIT WITH",
        "Apache-2.0 (see LICENSE)",
    ],
)
def test_invalid_expressions_fail(expression: str) -> None:
    with pytest.raises(check_spdx.ExpressionError):
        check_spdx.check_expression(expression, ALLOWED)


def test_missing_header_fails(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "x.py"
    source.write_text("print(1)\n")
    assert check_spdx.check_file(str(source), ALLOWED) == [f"missing SPDX header: {source}"]


def test_every_identifier_line_is_checked(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "x.cuh"
    source.write_text("// SPDX-License-Identifier: MIT\n// SPDX-License-Identifier: GPL-3.0-only\n")
    assert len(check_spdx.check_file(str(source), ALLOWED)) == 1


def test_header_past_line_15_is_missing(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "x.py"
    source.write_text("\n" * 15 + "# SPDX-License-Identifier: Apache-2.0\n")
    assert check_spdx.check_file(str(source), ALLOWED)


def _signoff(tmp_path: pathlib.Path, message: str) -> int:
    message_file = tmp_path / "msg"
    message_file.write_text(message)
    result = subprocess.run(
        ["bash", str(TOOLS / "check_signoff.sh"), str(message_file)],
        capture_output=True,
        check=False,
    )
    return result.returncode


@pytest.mark.parametrize(
    ("trailer", "expected"),
    [
        ("Signed-off-by: Dan Gil <dagil@nvidia.com>", 0),
        ("Signed-off-by: ", 1),
        ("Signed-off-by: Dan Gil", 1),
        ("Signed-off-by: <dagil@nvidia.com>", 1),
        ("signed-off-by: Dan Gil <dagil@nvidia.com>", 1),
    ],
)
def test_signoff_trailer(tmp_path: pathlib.Path, trailer: str, expected: int) -> None:
    assert _signoff(tmp_path, f"subject\n\n{trailer}\n") == expected


def test_signoff_rejects_unresolvable_range() -> None:
    result = subprocess.run(
        ["bash", str(TOOLS / "check_signoff.sh"), "--range", "0" * 39 + "1", "HEAD"],
        capture_output=True,
        check=False,
        cwd=TOOLS.parent,
    )
    assert result.returncode == 1
