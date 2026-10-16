# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton backend for physical-expert rank counting."""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor

_BLOCK = 256


@triton.jit
def _count_routed_per_rank_kernel(
    physical_ids_ptr,
    rank_map_ptr,
    counts_ptr,
    n,
    experts_per_rank,
    WORLD_SIZE: tl.constexpr,
    USE_RANK_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    physical = tl.load(physical_ids_ptr + offsets, mask=mask, other=-1)
    valid = mask & (physical >= 0)
    safe_physical = tl.maximum(physical, 0)
    if USE_RANK_MAP:
        rank = tl.load(rank_map_ptr + safe_physical, mask=valid, other=0)
    else:
        rank = safe_physical // experts_per_rank

    for target_rank in range(WORLD_SIZE):
        block_count = tl.sum((valid & (rank == target_rank)).to(tl.int64), axis=0)
        if block_count > 0:
            tl.atomic_add(counts_ptr + target_rank, block_count)


def count_routed_per_rank_cuda(
    physical_ids: Tensor,
    counts: Tensor,
    *,
    world_size: int,
    experts_per_rank: int,
    physical_to_rank_map: Tensor | None,
) -> None:
    n = physical_ids.numel()
    if n == 0:
        return
    rank_map = physical_ids if physical_to_rank_map is None else physical_to_rank_map
    _count_routed_per_rank_kernel[(triton.cdiv(n, _BLOCK),)](
        physical_ids,
        rank_map,
        counts,
        n,
        experts_per_rank,
        WORLD_SIZE=world_size,
        USE_RANK_MAP=physical_to_rank_map is not None,
        BLOCK=_BLOCK,
    )
