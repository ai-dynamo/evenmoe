# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton backend for logical-expert counting."""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor

_BLOCK = 256


@triton.jit
def _count_logical_experts_kernel(
    logical_ids_ptr,
    counts_ptr,
    n,
    num_logical_experts,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    logical_id = tl.load(logical_ids_ptr + offsets, mask=mask, other=-1)
    valid = mask & (logical_id >= 0) & (logical_id < num_logical_experts)
    tl.atomic_add(counts_ptr + logical_id, 1.0, mask=valid)


def count_logical_experts_cuda(logical_ids: Tensor, counts: Tensor) -> None:
    n = logical_ids.numel()
    if n == 0:
        return
    _count_logical_experts_kernel[(triton.cdiv(n, _BLOCK),)](
        logical_ids,
        counts,
        n,
        counts.numel(),
        BLOCK=_BLOCK,
    )
