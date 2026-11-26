# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA kernel and placement metadata for Rank-Local Replica Balancing."""

from __future__ import annotations

from pathlib import Path

from evenmoe.kernels.jit.utils import cache_once, load_jit, make_cpp_args

EP_SIZES = (16, 32)
BATCH_SIZES = (128, 256)
NUM_LOGICAL_EXPERTS = 128
NUM_PHYSICAL_EXPERTS = 160
TOP_K = 4


def pack_placement(physical_to_logical: list[int], ep_size: int) -> list[int]:
    """Pack the fixed-size placement tables consumed by the CUDA kernel."""
    logical = NUM_LOGICAL_EXPERTS
    physical = NUM_PHYSICAL_EXPERTS
    if ep_size not in EP_SIZES or len(physical_to_logical) != physical:
        raise ValueError("RLRB requires 160 physical experts on EP16 or EP32.")
    if set(physical_to_logical) != set(range(logical)):
        raise ValueError("RLRB requires a dense placement covering all 128 logical experts.")

    slots_per_rank = physical // ep_size
    candidates: list[list[int]] = [[] for _ in range(logical)]
    for slot, expert in enumerate(physical_to_logical):
        candidates[expert].append(slot)

    slot_group = [0] * physical
    group_rank: list[int] = []
    group_expert: list[int] = []
    group_copies: list[int] = []
    group_offsets = [0]
    candidate_offsets = [0]
    replicated = []
    for expert, slots in enumerate(candidates):
        by_rank: dict[int, list[int]] = {}
        for slot in slots:
            by_rank.setdefault(slot // slots_per_rank, []).append(slot)
        if len(by_rank) > 1:
            replicated.append(expert)
        for rank, rank_slots in by_rank.items():
            for slot in rank_slots:
                slot_group[slot] = len(group_rank)
            group_rank.append(rank)
            group_expert.append(expert)
            group_copies.append(len(rank_slots))
        group_offsets.append(len(group_rank))
        candidate_offsets.append(candidate_offsets[-1] + len(slots))

    rank_groups = []
    for rank in range(ep_size):
        start = rank * slots_per_rank
        groups = sorted(set(slot_group[start : start + slots_per_rank]))
        rank_groups.extend(groups + [-1] * (slots_per_rank - len(groups)))

    max_replicated = physical - logical
    # Keep this order in sync with the offsets in csrc/rlrb.cuh.
    return (
        [slot for slots in candidates for slot in slots]
        + slot_group
        + group_rank
        + [0] * (physical - len(group_rank))
        + group_expert
        + [0] * (physical - len(group_expert))
        + group_copies
        + [0] * (physical - len(group_copies))
        + group_offsets
        + [len(slots) for slots in candidates]
        + replicated
        + [-1] * (max_replicated - len(replicated))
        + rank_groups
        + candidate_offsets
        + [len(group_rank), len(replicated)]
    )


@cache_once
def load_kernel(ep_size: int, batch_size: int):
    args = make_cpp_args(ep_size, batch_size)
    return load_jit(
        "rlrb",
        *args,
        cuda_files=["rlrb.cuh"],
        cuda_wrappers=[("route", f"run_rlrb<{args}>")],
        source_root=Path(__file__).parent,
    )
