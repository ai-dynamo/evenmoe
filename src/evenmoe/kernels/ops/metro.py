# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""METRO replica assignment and routing tensor operators.

METRO (arXiv:2512.09277, Algorithm 1) picks one physical copy per active
logical expert so that every EP rank reads as few distinct expert weights as
possible. Given the EP-summed per-logical token count:

1. ``active[l] = count[l] > 0``; every rank sees the same active set.
2. Active experts held by a single rank add a fixed load of one there.
3. Active experts held by two or more ranks are visited in ascending logical
   id; each goes to the eligible rank with the lowest load (ties go to the
   lowest rank), which then gains one.
4. Every routed ``(token, slot)`` of logical ``l`` maps to the chosen copy.

The operators take the per-layer tables as plain tensors; the caller owns
deriving them from a placement. With ``L`` logical experts, ``G`` EP ranks,
``C`` replicated-expert capacity and ``W = ceil(G / 32)``:

- ``default_physical`` ``[L]``: the copy used when ``l`` is not reassigned.
- ``single_rank`` ``[L]``: owning rank if exactly one rank holds ``l``, else -1.
- ``rep_logical`` ``[C]``: logical ids held on two or more ranks, ascending,
  in the first ``num_replicated`` entries.
- ``rep_mask`` ``[C, W]``: bitmask of the ranks holding ``rep_logical``, 32
  ranks per int32 word.
- ``rep_phys`` ``[C, G]``: copy of ``rep_logical`` on each rank, or -1.
- ``num_replicated`` ``[1]``: live entry count of the ``rep_*`` tables, read
  on device so the tables can be rewritten in place under CUDA graphs.

All tables are int32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_RANKS = 128
"""The greedy packs ``(load, rank)`` into one int32 key with 7 rank bits."""

_ROUTE_BLOCK = 256


def metro_mask_words(num_ranks: int) -> int:
    """int32 words per rank bitmask."""

    return (num_ranks + 31) // 32


def metro_workspace_numel(num_logical: int, num_ranks: int, capacity: int) -> int:
    """int32 elements of the scratch :func:`metro_route` needs."""

    return num_logical + (metro_mask_words(num_ranks) + 1) * _r_pad(capacity)


def metro_workspace(
    num_logical: int, num_ranks: int, capacity: int, device: torch.device | str
) -> Tensor:
    """int32 scratch for :func:`metro_route`; keep one per layer under CUDA graphs."""

    return torch.empty(
        metro_workspace_numel(num_logical, num_ranks, capacity), dtype=torch.int32, device=device
    )


def metro_assign_reference(
    counts: Tensor,
    default_physical: Tensor,
    single_rank: Tensor,
    rep_logical: Tensor,
    rep_phys: Tensor,
    num_replicated: Tensor,
) -> Tensor:
    """Host oracle: ``[L]`` int64 chosen physical id per logical expert.

    It reads the holding ranks off ``rep_phys`` rather than ``rep_mask``, so it
    also checks the masks the kernel uses.
    """

    num_ranks = int(rep_phys.shape[1])
    active = (counts.detach().cpu() > 0).tolist()
    single = single_rank.cpu().tolist()
    chosen = default_physical.cpu().tolist()
    count = int(num_replicated.item())
    logicals = rep_logical[:count].cpu().tolist()
    phys = rep_phys[:count].cpu().tolist()
    load = [0] * num_ranks
    for logical, rank in enumerate(single):
        if active[logical] and rank >= 0:
            load[rank] += 1
    for logical, row in zip(logicals, phys):
        if not active[logical]:
            continue
        eligible = [g for g in range(num_ranks) if row[g] >= 0]
        best = min(eligible, key=lambda g: (load[g], g))
        load[best] += 1
        chosen[logical] = row[best]
    return torch.tensor(chosen, dtype=torch.int64)


@triton.jit
def _metro_greedy(
    load,
    place,
    rs,
    mask0,
    rep_mask_ptr,
    mask_by_pos_ptr,
    rank_by_pos_ptr,
    lanes,
    lane_ok,
    WORDS: tl.constexpr,
):
    """Greedy over the ``place``-d replicated experts in ascending logical id.

    Returns the winning rank per expert (-1 where not placed). The placed
    experts are compacted first so the serial chain only visits them; their
    rank masks are staged in scratch so each step is one cached load. ``mask0``
    is every row's first mask word, loaded by the caller ahead of the counts it
    does not depend on. The argmin key packs ``(load, rank)`` so ties go to the
    lowest rank and the winner is the key's low rank bits.
    """

    pos = tl.where(place, tl.cumsum(place.to(tl.int32), 0) - 1, -1)
    n_place = tl.sum(place.to(tl.int32), axis=0)
    for w in tl.static_range(WORDS):
        if w == 0:
            word = mask0
        else:
            word = tl.load(rep_mask_ptr + rs * WORDS + w, mask=place, other=0)
        tl.store(mask_by_pos_ptr + tl.where(place, pos, 0) * WORDS + w, word, mask=place)
    tl.debug_barrier()
    lane_word = tl.where(lane_ok, lanes // 32, 0)
    lane_bit = lanes % 32
    for k in range(n_place):
        if WORDS == 1:
            # One word: a single broadcast load per step.
            mask = tl.load(mask_by_pos_ptr + k)
        else:
            mask = tl.load(mask_by_pos_ptr + k * WORDS + lane_word, mask=lane_ok, other=0)
        eligible = lane_ok & (((mask >> lane_bit) & 1) != 0)
        key = tl.where(eligible, (load << 7) | lanes, 0x7FFFFFFF)
        best = tl.min(key, axis=0)
        load += tl.where(lanes == (best & 127), 1, 0)
        tl.store(rank_by_pos_ptr + k, best & 127)
    tl.debug_barrier()
    rank = tl.load(rank_by_pos_ptr + tl.where(place, pos, 0), mask=place, other=-1)
    tl.debug_barrier()
    return rank


@triton.jit
def _metro_route_kernel(
    out_ptr,
    topk_ids_ptr,
    counts_ptr,
    workspace_ptr,
    default_physical_ptr,
    single_rank_ptr,
    rep_logical_ptr,
    rep_mask_ptr,
    rep_phys_ptr,
    num_replicated_ptr,
    num_token_non_padded_ptr,
    n_total,
    TOPK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    NUM_LOGICAL: tl.constexpr,
    L_PAD: tl.constexpr,
    NUM_RANKS: tl.constexpr,
    G_PAD: tl.constexpr,
    WORDS: tl.constexpr,
    R_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chosen_phys_ptr = workspace_ptr
    mask_by_pos_ptr = workspace_ptr + NUM_LOGICAL
    rank_by_pos_ptr = workspace_ptr + NUM_LOGICAL + WORDS * R_PAD

    ls = tl.arange(0, L_PAD)
    l_ok = ls < NUM_LOGICAL
    active = tl.load(counts_ptr + ls, mask=l_ok, other=0.0) > 0.0
    rs = tl.arange(0, R_PAD)
    r_ok = rs < tl.load(num_replicated_ptr)
    rep_logical = tl.load(rep_logical_ptr + rs, mask=r_ok, other=0)
    mask0 = tl.load(rep_mask_ptr + rs * WORDS, mask=r_ok, other=0)
    rep_active = r_ok & (tl.load(counts_ptr + rep_logical, mask=r_ok, other=0.0) > 0.0)
    lanes = tl.arange(0, G_PAD)
    lane_ok = lanes < NUM_RANKS

    # Fixed load: active experts held by exactly one rank.
    single = tl.load(single_rank_ptr + ls, mask=l_ok, other=-1)
    load = tl.histogram(single, G_PAD, mask=active & (single >= 0))
    tl.store(
        chosen_phys_ptr + ls,
        tl.load(default_physical_ptr + ls, mask=l_ok, other=-1),
        mask=l_ok,
    )
    rank = _metro_greedy(
        load,
        rep_active,
        rs,
        mask0,
        rep_mask_ptr,
        mask_by_pos_ptr,
        rank_by_pos_ptr,
        lanes,
        lane_ok,
        WORDS,
    )
    placed = rep_active & (rank >= 0)
    physical = tl.load(
        rep_phys_ptr + rs * NUM_RANKS + tl.where(placed, rank, 0), mask=placed, other=-1
    )
    tl.store(chosen_phys_ptr + rep_logical, physical, mask=placed)
    tl.debug_barrier()

    if HAS_GATE:
        limit = tl.minimum(tl.load(num_token_non_padded_ptr).to(tl.int32) * TOPK, n_total)
    else:
        limit = n_total
    for start in range(0, n_total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        in_bounds = offs < n_total
        ids = tl.load(topk_ids_ptr + offs, mask=offs < limit, other=-1)
        valid = in_bounds & (ids >= 0) & (ids < NUM_LOGICAL)
        mapped = tl.load(chosen_phys_ptr + tl.where(valid, ids, 0), mask=valid, other=-1)
        tl.store(out_ptr + offs, mapped, mask=in_bounds)


def _r_pad(capacity: int) -> int:
    return max(triton.next_power_of_2(capacity), 2)


def metro_route(
    logical_topk_ids: Tensor,
    counts: Tensor,
    *,
    default_physical: Tensor,
    single_rank: Tensor,
    rep_logical: Tensor,
    rep_mask: Tensor,
    rep_phys: Tensor,
    num_replicated: Tensor,
    num_token_non_padded: Tensor | None = None,
    workspace: Tensor | None = None,
    out_dtype: torch.dtype = torch.int64,
) -> Tensor:
    """Map logical TopK ids to the METRO-chosen physical copies.

    ``counts`` is the ``[L]`` EP-summed token count; an expert is active iff
    its count is positive. Ids outside ``[0, L)`` and rows at or past
    ``num_token_non_padded`` (a one-element int tensor) come back as -1. On
    CUDA the assignment and the mapping run in one launch. Table shapes are
    described in the module docstring; ``L``, ``G`` and ``C`` are read off them.
    """

    num_logical = int(default_physical.numel())
    capacity, num_ranks = (int(n) for n in rep_phys.shape)
    if not 0 < num_ranks <= MAX_RANKS:
        raise ValueError(f"METRO supports 1..{MAX_RANKS} EP ranks; got {num_ranks}.")
    if rep_mask.shape != (capacity, metro_mask_words(num_ranks)):
        raise ValueError(
            f"rep_mask must have shape [{capacity}, {metro_mask_words(num_ranks)}], "
            f"got {tuple(rep_mask.shape)}."
        )
    if counts.shape != (num_logical,):
        raise ValueError(f"counts must have shape [{num_logical}], got {tuple(counts.shape)}.")
    if workspace is not None:
        numel = metro_workspace_numel(num_logical, num_ranks, capacity)
        if (
            workspace.shape != (numel,)
            or workspace.dtype != torch.int32
            or workspace.device != logical_topk_ids.device
        ):
            raise ValueError(
                f"METRO workspace must be a [{numel}] int32 tensor on "
                f"{logical_topk_ids.device}; got {tuple(workspace.shape)} "
                f"{workspace.dtype} on {workspace.device}."
            )
    if not logical_topk_ids.is_cuda:
        chosen = metro_assign_reference(
            counts, default_physical, single_rank, rep_logical, rep_phys, num_replicated
        )
        return _route_host(logical_topk_ids, chosen, num_token_non_padded, out_dtype)

    flat = logical_topk_ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    flat = flat.contiguous()
    if counts.dtype != torch.float32:
        counts = counts.to(torch.float32)
    if workspace is None:
        workspace = metro_workspace(num_logical, num_ranks, capacity, flat.device)
    out = torch.empty(flat.numel(), dtype=out_dtype, device=flat.device)
    has_gate = num_token_non_padded is not None
    _metro_route_kernel[(1,)](
        out,
        flat,
        counts,
        workspace,
        default_physical,
        single_rank,
        rep_logical,
        rep_mask,
        rep_phys,
        num_replicated,
        num_token_non_padded if has_gate else num_replicated,
        flat.numel(),
        TOPK=int(logical_topk_ids.shape[-1]) if logical_topk_ids.dim() > 1 else 1,
        HAS_GATE=has_gate,
        NUM_LOGICAL=num_logical,
        L_PAD=triton.next_power_of_2(num_logical),
        NUM_RANKS=num_ranks,
        G_PAD=max(triton.next_power_of_2(num_ranks), 2),
        WORDS=metro_mask_words(num_ranks),
        R_PAD=_r_pad(capacity),
        BLOCK=_ROUTE_BLOCK,
        num_warps=1,
    )
    return out.view(logical_topk_ids.shape)


def _route_host(
    logical_topk_ids: Tensor,
    chosen: Tensor,
    num_token_non_padded: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor:
    num_logical = int(chosen.numel())
    chosen = chosen.to(logical_topk_ids.device)
    ids = logical_topk_ids.to(torch.int64)
    valid = (ids >= 0) & (ids < num_logical)
    if num_token_non_padded is not None and ids.dim() > 1:
        rows = torch.arange(ids.shape[0], device=ids.device).view(-1, *([1] * (ids.dim() - 1)))
        valid &= rows < int(num_token_non_padded.reshape(-1)[0])
    physical = torch.where(valid, chosen[ids.clamp(0, num_logical - 1)], -1)
    return physical.to(out_dtype)
