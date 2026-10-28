# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
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

Replication is rank-level: several copies of one logical expert on the same
rank collapse to the first one in candidate column order, since only one copy
per rank can lower that rank's load.

The replicated-expert tables are padded to a fixed capacity and the live
replicated count is read on device, so a placement update rewrites them in
place and keeps captured CUDA graphs valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_RANKS = 32
"""The greedy packs ``(load, rank)`` into one int32 key with 5 rank bits."""

_ROUTE_BLOCK = 256


@dataclass(frozen=True)
class MetroTables:
    """Per-layer tables derived from one committed placement.

    Shapes (``L`` logical experts, ``G`` EP ranks, ``C`` replicated capacity):

    - ``physical_by_rank`` ``[L, G]``: first copy of ``l`` on rank ``g``, or -1.
    - ``default_physical`` ``[L]``: copy on the lowest holding rank.
    - ``single_rank`` ``[L]``: owning rank if exactly one rank holds ``l``, else -1.
    - ``rep_logical`` ``[C]``: logical ids held on two or more ranks, ascending,
      in the first ``num_replicated`` entries.
    - ``rep_mask`` ``[C]``: bitmask of the ranks holding ``rep_logical``.
    - ``rep_phys`` ``[C, G]``: ``physical_by_rank[rep_logical]``.
    - ``num_replicated`` ``[1]``: live entry count of the ``rep_*`` tables.
    """

    num_logical: int
    num_ranks: int
    capacity: int
    physical_by_rank: Tensor
    default_physical: Tensor
    single_rank: Tensor
    rep_logical: Tensor
    rep_mask: Tensor
    rep_phys: Tensor
    num_replicated: Tensor

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.physical_by_rank,
            self.default_physical,
            self.single_rank,
            self.rep_logical,
            self.rep_mask,
            self.rep_phys,
            self.num_replicated,
        )


def metro_capacity(num_logical: int, num_physical: int) -> int:
    """Upper bound on logical experts held by two or more ranks."""

    return max(min(num_logical, num_physical - num_logical), 1)


def build_metro_tables(
    logical_to_physical_candidates: Tensor,
    num_physical_experts: int,
    num_ranks: int,
) -> MetroTables:
    """Derive :class:`MetroTables` from a ``[L, max_replicas]`` candidate table.

    Candidates are padded with -1. Physical slots are rank-contiguous:
    ``rank(p) = p // (num_physical_experts // num_ranks)``.
    """

    if not 0 < num_ranks <= MAX_RANKS:
        raise ValueError(f"METRO supports 1..{MAX_RANKS} EP ranks; got {num_ranks}.")
    if num_physical_experts % num_ranks != 0:
        raise ValueError(
            f"num_physical_experts={num_physical_experts} is not divisible by "
            f"num_ranks={num_ranks}."
        )
    candidates = logical_to_physical_candidates.to(torch.int64)
    device = candidates.device
    num_logical = int(candidates.shape[0])
    num_local = num_physical_experts // num_ranks
    valid = candidates >= 0
    candidate_rank = torch.div(candidates.clamp(min=0), num_local, rounding_mode="floor")
    columns = []
    for rank in range(num_ranks):
        on_rank = valid & (candidate_rank == rank)
        first = on_rank.to(torch.int32).argmax(dim=1, keepdim=True)
        columns.append(
            candidates.gather(1, first).squeeze(1).masked_fill(~on_rank.any(dim=1), -1)
        )
    physical_by_rank = torch.stack(columns, dim=1).to(torch.int32)
    has_copy = physical_by_rank >= 0
    if bool((~has_copy).all(dim=1).any()):
        raise ValueError("Every logical expert must have at least one physical copy.")

    rank_bits = torch.ones(num_ranks, dtype=torch.int64, device=device) << torch.arange(
        num_ranks, device=device
    )
    rank_mask = (has_copy.to(torch.int64) * rank_bits).sum(dim=1).to(torch.int32)
    first_rank = has_copy.to(torch.int32).argmax(dim=1)
    default_physical = physical_by_rank.gather(1, first_rank.view(-1, 1)).view(-1)
    num_holding = has_copy.sum(dim=1)
    single_rank = torch.where(num_holding == 1, first_rank, torch.full_like(first_rank, -1))

    replicated = torch.nonzero(num_holding >= 2).flatten()
    capacity = metro_capacity(num_logical, num_physical_experts)
    count = int(replicated.numel())
    if count > capacity:
        raise ValueError(f"{count} replicated experts exceed the capacity {capacity}.")
    rep_logical = torch.zeros(capacity, dtype=torch.int32, device=device)
    rep_mask = torch.zeros(capacity, dtype=torch.int32, device=device)
    rep_phys = torch.full((capacity, num_ranks), -1, dtype=torch.int32, device=device)
    rep_logical[:count] = replicated.to(torch.int32)
    rep_mask[:count] = rank_mask[replicated]
    rep_phys[:count] = physical_by_rank[replicated]
    return MetroTables(
        num_logical=num_logical,
        num_ranks=num_ranks,
        capacity=capacity,
        physical_by_rank=physical_by_rank.contiguous(),
        default_physical=default_physical.to(torch.int32).contiguous(),
        single_rank=single_rank.to(torch.int32).contiguous(),
        rep_logical=rep_logical,
        rep_mask=rep_mask,
        rep_phys=rep_phys,
        num_replicated=torch.tensor([count], dtype=torch.int32, device=device),
    )


def metro_assign_reference(counts: Tensor, tables: MetroTables) -> Tensor:
    """Host oracle: ``[L]`` int64 chosen physical id per logical expert."""

    active = (counts.detach().cpu() > 0).tolist()
    single = tables.single_rank.cpu().tolist()
    chosen = tables.default_physical.cpu().tolist()
    count = int(tables.num_replicated.item())
    rep_logical = tables.rep_logical[:count].cpu().tolist()
    rep_mask = tables.rep_mask[:count].cpu().tolist()
    rep_phys = tables.rep_phys[:count].cpu().tolist()
    load = [0] * tables.num_ranks
    for logical, rank in enumerate(single):
        if active[logical] and rank >= 0:
            load[rank] += 1
    for i, logical in enumerate(rep_logical):
        if not active[logical]:
            continue
        eligible = [g for g in range(tables.num_ranks) if (rep_mask[i] >> g) & 1]
        best = min(eligible, key=lambda g: (load[g], g))
        load[best] += 1
        chosen[logical] = rep_phys[i][best]
    return torch.tensor(chosen, dtype=torch.int64)


@triton.jit
def _metro_greedy(load, place, rep_mask, mask_by_pos_ptr, rank_by_pos_ptr, lanes, lane_ok):
    """Greedy over the ``place``-d replicated experts in ascending logical id.

    Returns the winning rank per expert (-1 where not placed). The placed
    experts are compacted first so the serial chain only visits them; their
    rank masks are staged in scratch so each step is one cached load. The
    argmin key packs ``(load, rank)`` so ties go to the lowest rank and the
    winner is the key's low five bits.
    """

    pos = tl.where(place, tl.cumsum(place.to(tl.int32), 0) - 1, -1)
    n_place = tl.sum(place.to(tl.int32), axis=0)
    tl.store(mask_by_pos_ptr + tl.where(place, pos, 0), rep_mask, mask=place)
    tl.debug_barrier()
    for k in range(n_place):
        mask = tl.load(mask_by_pos_ptr + k)
        # Clamp the shift: lanes past the rank count would shift out of range.
        eligible = lane_ok & (((mask >> tl.where(lane_ok, lanes, 0)) & 1) != 0)
        key = tl.where(eligible, (load << 5) | lanes, 0x7FFFFFFF)
        best = tl.min(key, axis=0)
        load += tl.where(lanes == (best & 31), 1, 0)
        tl.store(rank_by_pos_ptr + k, best & 31)
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
    R_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chosen_phys_ptr = workspace_ptr
    mask_by_pos_ptr = workspace_ptr + NUM_LOGICAL
    rank_by_pos_ptr = workspace_ptr + NUM_LOGICAL + R_PAD

    ls = tl.arange(0, L_PAD)
    l_ok = ls < NUM_LOGICAL
    active = tl.load(counts_ptr + ls, mask=l_ok, other=0.0) > 0.0
    rs = tl.arange(0, R_PAD)
    r_ok = rs < tl.load(num_replicated_ptr)
    rep_logical = tl.load(rep_logical_ptr + rs, mask=r_ok, other=0)
    rep_active = r_ok & (tl.load(counts_ptr + rep_logical, mask=r_ok, other=0.0) > 0.0)
    rep_mask = tl.load(rep_mask_ptr + rs, mask=r_ok, other=0)
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
    rank = _metro_greedy(load, rep_active, rep_mask, mask_by_pos_ptr, rank_by_pos_ptr, lanes, lane_ok)
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


def _r_pad(tables: MetroTables) -> int:
    return max(triton.next_power_of_2(tables.capacity), 2)


def metro_workspace(tables: MetroTables) -> Tensor:
    """int32 scratch for :func:`metro_route`; keep one per layer under CUDA graphs."""

    return torch.empty(
        tables.num_logical + 2 * _r_pad(tables),
        dtype=torch.int32,
        device=tables.default_physical.device,
    )


def metro_route(
    logical_topk_ids: Tensor,
    counts: Tensor,
    tables: MetroTables,
    *,
    num_token_non_padded: Tensor | None = None,
    workspace: Tensor | None = None,
    out_dtype: torch.dtype = torch.int64,
) -> Tensor:
    """Map logical TopK ids to the METRO-chosen physical copies.

    ``counts`` is the ``[L]`` EP-summed token count; an expert is active iff
    its count is positive. Ids outside ``[0, L)`` and rows at or past
    ``num_token_non_padded`` (a one-element int tensor) come back as -1. On
    CUDA the assignment and the mapping run in one launch.
    """

    if counts.shape != (tables.num_logical,):
        raise ValueError(
            f"counts must have shape [{tables.num_logical}], got {tuple(counts.shape)}."
        )
    if not logical_topk_ids.is_cuda:
        return _metro_route_host(logical_topk_ids, counts, tables, num_token_non_padded, out_dtype)

    flat = logical_topk_ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    flat = flat.contiguous()
    if counts.dtype != torch.float32:
        counts = counts.to(torch.float32)
    if workspace is None:
        workspace = metro_workspace(tables)
    out = torch.empty(flat.numel(), dtype=out_dtype, device=flat.device)
    has_gate = num_token_non_padded is not None
    _metro_route_kernel[(1,)](
        out,
        flat,
        counts,
        workspace,
        tables.default_physical,
        tables.single_rank,
        tables.rep_logical,
        tables.rep_mask,
        tables.rep_phys,
        tables.num_replicated,
        num_token_non_padded if has_gate else tables.num_replicated,
        flat.numel(),
        TOPK=int(logical_topk_ids.shape[-1]) if logical_topk_ids.dim() > 1 else 1,
        HAS_GATE=has_gate,
        NUM_LOGICAL=tables.num_logical,
        L_PAD=triton.next_power_of_2(tables.num_logical),
        NUM_RANKS=tables.num_ranks,
        G_PAD=max(triton.next_power_of_2(tables.num_ranks), 2),
        R_PAD=_r_pad(tables),
        BLOCK=_ROUTE_BLOCK,
        num_warps=1,
    )
    return out.view(logical_topk_ids.shape)


def _metro_route_host(
    logical_topk_ids: Tensor,
    counts: Tensor,
    tables: MetroTables,
    num_token_non_padded: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor:
    chosen = metro_assign_reference(counts, tables).to(logical_topk_ids.device)
    ids = logical_topk_ids.to(torch.int64)
    valid = (ids >= 0) & (ids < tables.num_logical)
    if num_token_non_padded is not None and ids.dim() > 1:
        rows = torch.arange(ids.shape[0], device=ids.device).view(-1, *([1] * (ids.dim() - 1)))
        valid &= rows < int(num_token_non_padded.reshape(-1)[0])
    physical = torch.where(valid, chosen[ids.clamp(0, tables.num_logical - 1)], -1)
    return physical.to(out_dtype)
