#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail any file without a valid SPDX-License-Identifier in its first 15 lines.

Each identifier line is parsed as a full SPDX expression: license IDs from
SPDX_ALLOWED (space separated), the operators AND, OR and WITH, and parentheses.
A "<id> WITH <exception>" term passes only when SPDX_ALLOWED lists
"<id>_WITH_<exception>". Anything else, including an empty expression, fails.
"""

from __future__ import annotations

import os
import re
import sys

HEADER_LINES = 15
MARKER = "SPDX-License-Identifier:"
COMMENT_CLOSERS = re.compile(r"\s*(\*/|-->|-?#\}|-?%\}|\]\])\s*$")
TOKEN = re.compile(r"\(|\)|[^\s()]+")


class ExpressionError(ValueError):
    pass


class _Parser:
    def __init__(self, tokens: list[str], allowed: set[str]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.allowed = allowed

    def parse(self) -> None:
        if not self.tokens:
            raise ExpressionError("empty expression")
        self._expression()
        if self.pos != len(self.tokens):
            raise ExpressionError(f"unexpected token {self.tokens[self.pos]!r}")

    def _next(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._next()
        if token is None:
            raise ExpressionError("expression ends early")
        self.pos += 1
        return token

    def _expression(self) -> None:
        self._term()
        while self._next() in ("AND", "OR"):
            self.pos += 1
            self._term()

    def _term(self) -> None:
        token = self._take()
        if token == "(":
            self._expression()
            if self._take() != ")":
                raise ExpressionError("missing closing parenthesis")
            return
        if token in ("AND", "OR", "WITH", ")"):
            raise ExpressionError(f"unexpected token {token!r}")
        license_id = token
        if self._next() == "WITH":
            self.pos += 1
            license_id = f"{token}_WITH_{self._take()}"
        if license_id not in self.allowed:
            raise ExpressionError(f"license {license_id} is not in SPDX_ALLOWED")


def check_expression(expression: str, allowed: set[str]) -> None:
    expression = COMMENT_CLOSERS.sub("", expression)
    _Parser(TOKEN.findall(expression), allowed).parse()


def check_file(path: str, allowed: set[str]) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        head = [line for _, line in zip(range(HEADER_LINES), handle)]
    expressions = [line.split(MARKER, 1)[1] for line in head if MARKER in line]
    if not expressions:
        return [f"missing SPDX header: {path}"]
    errors = []
    for expression in expressions:
        try:
            check_expression(expression, allowed)
        except ExpressionError as error:
            errors.append(f"invalid SPDX expression ({error}): {path}")
    return errors


def main(paths: list[str]) -> int:
    allowed = set(os.environ.get("SPDX_ALLOWED", "Apache-2.0").split())
    errors = [error for path in paths for error in check_file(path, allowed)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
