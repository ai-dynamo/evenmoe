# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton operator for Waterfill shared-expert rank selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_SIZE = 128
_LOCAL_PREFERENCE_DENOMINATOR = 10


def waterfill_select(
    *,
    rank_load: Tensor,
    routed_load: Tensor,
    topk_ids: Tensor,
    allow_all_ranks: bool,
    source_rank: int,
    local_preference: float,
    world_size: int,
    experts_per_rank: int,
    out: Tensor | None = None,
) -> Tensor:
    """Return the shared-expert destination rank for each token.

    ``target_total`` is derived on-device from ``routed_load`` and
    ``rank_load``. The tensors alias in static mode and differ in dynamic
    mode, where ``rank_load`` also includes active shared-expert traffic.
    """

    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=topk_ids.device)

    chosen_rank = (
        torch.empty(num_tokens, dtype=torch.int64, device=topk_ids.device)
        if out is None
        else out
    )
    if chosen_rank.shape != (num_tokens,) or chosen_rank.dtype != torch.int64:
        raise ValueError("Waterfill output buffer has incompatible shape or dtype.")
    if chosen_rank.device != topk_ids.device:
        raise ValueError("Waterfill output buffer must be on the input device.")

    grid = (triton.cdiv(num_tokens, _BLOCK_SIZE),)
    preference_numerator = max(
        round(local_preference * _LOCAL_PREFERENCE_DENOMINATOR),
        1,
    )

    _waterfill_select_kernel[grid](
        topk_ids,
        rank_load,
        routed_load,
        chosen_rank,
        num_tokens,
        topk,
        experts_per_rank,
        world_size,
        source_rank,
        preference_numerator,
        _LOCAL_PREFERENCE_DENOMINATOR,
        ALLOW_ALL_RANKS=allow_all_ranks,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return chosen_rank


@triton.jit
def _waterfill_select_kernel(
    topk_ids_ptr,
    rank_load_ptr,
    routed_load_ptr,
    chosen_rank_ptr,
    num_tokens,
    topk: tl.constexpr,
    experts_per_rank,
    world_size: tl.constexpr,
    source_rank,
    local_pref_numer,
    local_pref_denom,
    ALLOW_ALL_RANKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Select one shared-expert rank per token without a host sync."""

    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    r_idx = tl.arange(0, world_size)
    rank_load_vec = tl.load(
        rank_load_ptr + r_idx,
        mask=r_idx < world_size,
        other=0,
    ).to(tl.int64)
    routed_load_vec = tl.load(
        routed_load_ptr + r_idx,
        mask=r_idx < world_size,
        other=0,
    ).to(tl.int64)
    total_effective_k = tl.sum(rank_load_vec)
    total_routed_k = tl.sum(routed_load_vec)
    total_tokens_global_k = total_routed_k // topk
    target_total = (
        total_effective_k + total_tokens_global_k + world_size - 1
    ) // world_size

    source_count = tl.load(rank_load_ptr + source_rank)
    best_count = tl.where(mask, source_count, 2**30)
    best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    src_rank_i32 = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)

    if ALLOW_ALL_RANKS:
        candidate_mask = tl.full(
            [BLOCK_SIZE],
            (1 << world_size) - 1,
            dtype=tl.int32,
        )
        for rank in range(world_size):
            target_count = tl.load(rank_load_ptr + rank).to(tl.int64)
            better = (
                target_count * local_pref_numer < best_count * local_pref_denom
            ) & mask
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(
                better,
                tl.full([BLOCK_SIZE], rank, dtype=tl.int64),
                best_rank,
            )
    else:
        candidate_mask = (
            tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << src_rank_i32
        ).to(tl.int32)

    for slot in range(topk):
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + slot,
            mask=mask,
            other=-1,
        ).to(tl.int64)
        valid = expert_id >= 0
        has_valid = has_valid | valid

        if not ALLOW_ALL_RANKS:
            target_rank = expert_id // experts_per_rank
            target_rank = tl.minimum(
                tl.maximum(target_rank, 0),
                world_size - 1,
            )
            target_rank_i32 = target_rank.to(tl.int32)
            shift_amount = tl.where(valid, target_rank_i32, 0)
            bit = tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << shift_amount
            candidate_mask = tl.where(
                valid & mask,
                candidate_mask | bit,
                candidate_mask,
            )

            target_count = tl.load(
                rank_load_ptr + target_rank,
                mask=mask & valid,
                other=2**30,
            )
            better = (
                target_count * local_pref_numer < best_count * local_pref_denom
            ) & valid & mask
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)

    total_weight = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for rank in range(world_size):
        present = ((candidate_mask >> rank) & 1) == 1
        rank_load = tl.load(rank_load_ptr + rank).to(tl.int64)
        weight = tl.where(
            target_total > rank_load,
            target_total - rank_load,
            0,
        ).to(tl.int32)
        weight_vec = tl.full([BLOCK_SIZE], weight, dtype=tl.int32)
        weight_vec = tl.where(
            src_rank_i32 == rank,
            weight_vec,
            (weight_vec * local_pref_denom) // local_pref_numer,
        )
        total_weight += tl.where(present, weight_vec, 0)

    token_seed = token_idx.to(tl.uint32) ^ (
        src_rank_i32.to(tl.uint32)
        * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full(
        [BLOCK_SIZE], 1664525, dtype=tl.uint32
    ) + tl.full([BLOCK_SIZE], 1013904223, dtype=tl.uint32)
    safe_total_weight = tl.maximum(total_weight, 1)
    sample = tl.where(
        total_weight > 0,
        token_seed % safe_total_weight.to(tl.uint32),
        0,
    ).to(tl.int32)

    chosen = src_rank_i32
    cumulative_weight = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for rank in range(world_size):
        present = ((candidate_mask >> rank) & 1) == 1
        rank_load = tl.load(rank_load_ptr + rank).to(tl.int64)
        weight = tl.where(
            target_total > rank_load,
            target_total - rank_load,
            0,
        ).to(tl.int32)
        weight_vec = tl.full([BLOCK_SIZE], weight, dtype=tl.int32)
        weight_vec = tl.where(
            src_rank_i32 == rank,
            weight_vec,
            (weight_vec * local_pref_denom) // local_pref_numer,
        )
        weight_vec = tl.where(present, weight_vec, 0)
        pick = (
            (total_weight > 0)
            & present
            & (sample >= cumulative_weight)
            & (sample < cumulative_weight + weight_vec)
        )
        chosen = tl.where(pick, rank, chosen)
        cumulative_weight += weight_vec

    final_rank = tl.where(total_weight > 0, chosen.to(tl.int64), best_rank)
    final_rank = tl.where(
        has_valid,
        final_rank,
        tl.full([BLOCK_SIZE], -1, dtype=tl.int64),
    )
    tl.store(chosen_rank_ptr + token_idx, final_rank, mask=mask)
