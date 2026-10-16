# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton materialization of a shared-expert decision into DeepEP layout."""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor

_BLOCK_SIZE = 256


@triton.jit
def _materialize_shared_expert_kernel(
    physical_ids_ptr,
    weights_ptr,
    chosen_rank_ptr,
    expanded_ids_ptr,
    expanded_weights_ptr,
    num_tokens,
    old_experts_per_rank,
    new_experts_per_rank,
    shared_weight,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_mask = token_idx < num_tokens
    has_valid = tl.zeros((BLOCK_SIZE,), dtype=tl.int1)

    for slot in range(TOPK):
        offset = token_idx * TOPK + slot
        physical_id = tl.load(physical_ids_ptr + offset, mask=token_mask, other=-1)
        weight = tl.load(weights_ptr + offset, mask=token_mask, other=0.0)
        valid = physical_id >= 0
        has_valid |= valid
        interleaved_id = physical_id + physical_id // old_experts_per_rank
        output_offset = token_idx * (TOPK + 1) + slot
        tl.store(
            expanded_ids_ptr + output_offset,
            tl.where(valid, interleaved_id, physical_id),
            mask=token_mask,
        )
        tl.store(
            expanded_weights_ptr + output_offset,
            tl.where(valid, weight, 0.0),
            mask=token_mask,
        )

    chosen_rank = tl.load(chosen_rank_ptr + token_idx, mask=token_mask, other=-1)
    shared_id = chosen_rank * new_experts_per_rank + old_experts_per_rank
    shared_valid = has_valid & (chosen_rank >= 0)
    shared_offset = token_idx * (TOPK + 1) + TOPK
    tl.store(
        expanded_ids_ptr + shared_offset,
        tl.where(shared_valid, shared_id, -1),
        mask=token_mask,
    )
    tl.store(
        expanded_weights_ptr + shared_offset,
        tl.where(has_valid, shared_weight, 0.0),
        mask=token_mask,
    )


def materialize_shared_expert_cuda(
    physical_ids: Tensor,
    weights: Tensor,
    chosen_rank: Tensor,
    expanded_ids: Tensor,
    expanded_weights: Tensor,
    old_experts_per_rank: int,
    new_experts_per_rank: int,
    shared_weight: float,
) -> None:
    num_tokens, topk = physical_ids.shape
    rank_tensor = chosen_rank.to(
        device=physical_ids.device,
        dtype=physical_ids.dtype,
    ).contiguous()
    grid = (triton.cdiv(num_tokens, _BLOCK_SIZE),)
    _materialize_shared_expert_kernel[grid](
        physical_ids,
        weights,
        rank_tensor,
        expanded_ids,
        expanded_weights,
        num_tokens,
        old_experts_per_rank,
        new_experts_per_rank,
        shared_weight,
        TOPK=topk,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
