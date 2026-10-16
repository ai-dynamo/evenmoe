# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-expert and physical-rank counting tensor operators."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def count_logical_experts(
    logical_topk_ids: object,
    num_logical_experts: int,
    dtype: Any = None,
) -> Tensor:
    """Count valid logical TopK slots without counting ``-1`` padding.

    Counts accumulate in float32 on both backends; ``dtype`` only casts the
    result. The Triton backend uses floating-point atomic increments.
    """

    if num_logical_experts <= 0:
        raise ValueError("num_logical_experts must be positive.")
    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Logical expert counting requires a torch.Tensor.")

    counts = torch.zeros(
        num_logical_experts,
        dtype=torch.float32,
        device=logical_topk_ids.device,
    )
    if logical_topk_ids.is_cuda:
        from evenmoe.kernels.ops._logical_count_triton import (
            count_logical_experts_cuda,
        )

        count_logical_experts_cuda(logical_topk_ids, counts)
        return counts if dtype is None else counts.to(dtype)

    valid = (logical_topk_ids >= 0) & (logical_topk_ids < num_logical_experts)
    if valid.any():
        counts.copy_(
            torch.bincount(
                logical_topk_ids[valid].to(torch.int64),
                minlength=num_logical_experts,
            ).to(torch.float32)
        )
    return counts if dtype is None else counts.to(dtype)


def count_routed_per_rank(
    physical_topk_ids: object,
    *,
    world_size: int,
    experts_per_rank: int,
    physical_to_rank_map: object | None = None,
    out: object | None = None,
) -> Tensor:
    """Count valid physical routed slots per expert-parallel rank.

    This operator deliberately accepts only tensors and scalar geometry. The
    caller owns translation from placement or policy objects into these raw
    compute inputs.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if experts_per_rank <= 0:
        raise ValueError("experts_per_rank must be positive.")
    if not isinstance(physical_topk_ids, Tensor):
        raise TypeError("Rank-load counting requires torch.Tensor physical_topk_ids.")
    if physical_to_rank_map is not None and not isinstance(physical_to_rank_map, Tensor):
        raise TypeError("physical_to_rank_map must be a torch.Tensor or None.")

    rank_map = physical_to_rank_map
    if rank_map is not None and rank_map.device != physical_topk_ids.device:
        raise ValueError("physical_to_rank_map must be on the input device.")

    if out is None:
        counts = torch.zeros(
            world_size,
            dtype=torch.int64,
            device=physical_topk_ids.device,
        )
    else:
        if not isinstance(out, Tensor):
            raise TypeError("Rank-load output buffer must be a torch.Tensor.")
        if out.shape != (world_size,) or out.dtype != torch.int64:
            raise ValueError("Rank-load output buffer has incompatible shape or dtype.")
        if out.device != physical_topk_ids.device:
            raise ValueError("Rank-load output buffer must be on the input device.")
        counts = out
        counts.zero_()

    if physical_topk_ids.is_cuda:
        from evenmoe.kernels.ops._rank_count_triton import (
            count_routed_per_rank_cuda,
        )

        count_routed_per_rank_cuda(
            physical_topk_ids,
            counts,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
            physical_to_rank_map=rank_map,
        )
        return counts

    valid = physical_topk_ids >= 0
    safe_ids = physical_topk_ids.clamp(min=0).to(torch.int64)
    ranks = rank_map[safe_ids] if rank_map is not None else safe_ids // experts_per_rank
    flat_ranks = torch.where(valid, ranks, 0).reshape(-1).to(torch.int64)
    contributions = valid.reshape(-1).to(torch.int64)
    return counts.scatter_add_(0, flat_ranks, contributions)


__all__ = ["count_logical_experts", "count_routed_per_rank"]
