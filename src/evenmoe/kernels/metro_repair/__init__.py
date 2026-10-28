# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""METRO-repair: METRO's replica assignment plus a bounded repair search.

METRO (:mod:`evenmoe.kernels.ops.metro`) picks one physical copy per active
logical expert greedily, in ascending logical id. Greedy can leave one rank
with more distinct active experts than necessary even though a replica of one
of its experts sits on an underloaded rank -- most often at small decode
batches, where per-rank activation (not token count) sets the GEMM time.

METRO-repair starts from exactly METRO's assignment and then runs at most
``moves`` repair steps. Each step moves load from the most-loaded rank that
can shed it to an underloaded rank (gap >= 2), either directly or through one
intermediate rank (``src -> mid -> dst``, two expert moves, mid's load
unchanged). It stops early once the rank maximum reaches the lower bound
``max(ceil(active / ranks), max forced single-copy load)``. With
``moves=0`` it returns METRO's assignment bit for bit.

Every EP rank evaluates the same deterministic function of the same inputs
(EP-summed count, or an EP-global TopK) and of a rank-invariant placement
table, so all ranks agree without further communication.

Layout:

- ``csrc/route.cuh``  the fused kernel (active set, METRO, repair, remap) and
                      its tvm-ffi entry point; compiled per (id dtype, output
                      dtype, rank count) through :func:`load_jit`, so the
                      rank loops are fully unrolled.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evenmoe.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from evenmoe.kernels.ops.metro import MetroTables

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HERE = pathlib.Path(__file__).parent.resolve()

MAX_LOGICAL = 256
"""Experts tracked in the kernel's 256-bit active / edge masks."""

MAX_RANKS = 32
"""Ranks addressed by one warp lane each."""


@dataclass(frozen=True)
class MetroRepairTables:
    """METRO's tables plus the per-expert holder bitmask the repair walks.

    ``holder_mask[l]`` has bit ``g`` set when rank ``g`` holds a copy of ``l``
    (int32; bit 31 reinterpreted as unsigned in CUDA).
    """

    metro: MetroTables
    holder_mask: Tensor

    def tensors(self) -> tuple[Tensor, ...]:
        return (*self.metro.tensors(), self.holder_mask)


def build_metro_repair_tables(metro: MetroTables) -> MetroRepairTables:
    """Derive the holder bitmask from METRO's ``physical_by_rank`` table."""

    has_copy = metro.physical_by_rank >= 0
    bits = torch.ones(metro.num_ranks, dtype=torch.int64, device=has_copy.device) << torch.arange(
        metro.num_ranks, device=has_copy.device
    )
    mask = (has_copy.to(torch.int64) * bits).sum(dim=1)
    mask = torch.where(mask >= 2**31, mask - 2**32, mask).to(torch.int32)
    return MetroRepairTables(metro=metro, holder_mask=mask.contiguous())


def supports(num_logical: int, num_ranks: int) -> bool:
    """Whether the fused kernel covers this geometry."""

    return 0 < num_logical <= MAX_LOGICAL and 0 < num_ranks <= MAX_RANKS


@cache_once
def _route_module(in_dtype: torch.dtype, out_dtype: torch.dtype, num_ranks: int) -> Module:
    args = make_cpp_args(in_dtype, out_dtype, num_ranks)
    return load_jit(
        "metro_repair_route",
        *args,
        cuda_files=["route.cuh"],
        cuda_wrappers=[("metro_repair_route", f"metro_repair_route<{args}>")],
        source_root=_HERE,
    )


def metro_repair_route(
    logical_topk_ids: Tensor,
    tables: MetroRepairTables,
    *,
    counts: Tensor | None = None,
    num_token_non_padded: Tensor | None = None,
    moves: int = 32,
    early_exit: bool = True,
    out_dtype: torch.dtype = torch.int64,
    rank_load: Tensor | None = None,
) -> Tensor:
    """Map logical TopK ids ``[T, K]`` to METRO-repair's physical copies.

    ``counts`` is the ``[L]`` EP-summed per-logical count (active iff > 0).
    When it is ``None`` the active set is taken from ``logical_topk_ids``
    itself, which is only rank-consistent when every rank holds the same
    EP-global TopK (all-gather MoE). ``num_token_non_padded`` is an optional
    one-element tensor; rows at or past it, and ids outside ``[0, L)``, come
    back as -1. ``rank_load`` (``[G]`` int32), if given, receives each rank's
    number of distinct active experts after routing. One launch on the current
    stream; no host synchronization, so it is CUDA-graph safe.
    """

    metro = tables.metro
    if not logical_topk_ids.is_cuda:
        raise ValueError("metro_repair_route runs on CUDA tensors; use the reference on CPU.")
    if not supports(metro.num_logical, metro.num_ranks):
        raise ValueError(
            f"METRO-repair supports <= {MAX_LOGICAL} experts and <= {MAX_RANKS} ranks; "
            f"got {metro.num_logical} and {metro.num_ranks}."
        )
    if logical_topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"logical_topk_ids must be int32 or int64, got {logical_topk_ids.dtype}.")
    if out_dtype not in (torch.int32, torch.int64):
        raise TypeError(f"out_dtype must be int32 or int64, got {out_dtype}.")
    if moves < 0:
        raise ValueError("moves must be non-negative.")
    ids = logical_topk_ids if logical_topk_ids.dim() == 2 else logical_topk_ids.reshape(-1, 1)
    ids = ids.contiguous()
    device = ids.device
    if counts is None:
        counts_arg = torch.empty(0, dtype=torch.float32, device=device)
    else:
        if counts.shape != (metro.num_logical,):
            raise ValueError(
                f"counts must have shape [{metro.num_logical}], got {tuple(counts.shape)}."
            )
        counts_arg = counts if counts.dtype == torch.float32 else counts.to(torch.float32)
        counts_arg = counts_arg.contiguous()
    if num_token_non_padded is None:
        gate = torch.empty(0, dtype=torch.int32, device=device)
    else:
        gate = num_token_non_padded.reshape(-1)[:1]
        gate = gate if gate.dtype == torch.int32 else gate.to(torch.int32)
    if rank_load is None:
        rank_load = torch.empty(metro.num_ranks, dtype=torch.int32, device=device)
    out = torch.empty(ids.shape, dtype=out_dtype, device=device)
    _route_module(ids.dtype, out_dtype, metro.num_ranks).metro_repair_route(
        ids,
        out,
        counts_arg,
        gate,
        tables.holder_mask,
        metro.physical_by_rank,
        metro.default_physical,
        rank_load,
        int(moves),
        bool(early_exit),
    )
    return out.view(logical_topk_ids.shape)


def metro_repair_assign_reference(
    active: Tensor, tables: MetroRepairTables, *, moves: int = 32, early_exit: bool = False
) -> tuple[Tensor, Tensor]:
    """Host oracle: ``([G] load, [L] owner rank or -1)`` for a ``[L]`` bool active set.

    Starts from METRO's greedy (same order and tie-break as
    ``metro_assign_reference``) and applies the repair moves in the kernel's
    order. ``early_exit=False`` runs every improving move up to ``moves``.
    """

    metro = tables.metro
    ranks = metro.num_ranks
    act = [bool(x) for x in active.detach().cpu().tolist()]
    masks = [int(x) & 0xFFFFFFFF for x in tables.holder_mask.detach().cpu().tolist()]
    holders = [[g for g in range(ranks) if (mask >> g) & 1] for mask in masks]
    owner = [-1] * len(act)
    load = [0] * ranks
    for e, live in enumerate(act):
        if live and len(holders[e]) == 1:
            owner[e] = holders[e][0]
            load[owner[e]] += 1
    forced = max(load) if load else 0
    for e, live in enumerate(act):
        if live and owner[e] < 0:
            owner[e] = min(holders[e], key=lambda g: (load[g], g))
            load[owner[e]] += 1
    lower_bound = max(-(-sum(act) // ranks), forced)
    for _ in range(moves):
        if early_exit and max(load) <= lower_bound:
            break
        edge: dict[tuple[int, int], int] = {}
        for e, live in enumerate(act):
            if live:
                for dst in holders[e]:
                    if dst != owner[e]:
                        edge.setdefault((owner[e], dst), e)
        paths = []
        for src in range(ranks):
            for dst in range(ranks):
                if load[src] - load[dst] < 2:
                    continue
                mid = 32 if (src, dst) in edge else -1
                if mid < 0:
                    mid = next(
                        (m for m in range(ranks) if (src, m) in edge and (m, dst) in edge), -1
                    )
                if mid >= 0:
                    paths.append((-load[src], load[dst], src, dst, mid))
        if not paths:
            break
        _, _, src, dst, mid = min(paths)
        if mid == 32:
            owner[edge[src, dst]] = dst
        else:
            first, second = edge[src, mid], edge[mid, dst]
            owner[first] = mid
            owner[second] = dst
        load[src] -= 1
        load[dst] += 1
    return torch.tensor(load, dtype=torch.int32), torch.tensor(owner, dtype=torch.int32)


__all__ = [
    "MAX_LOGICAL",
    "MAX_RANKS",
    "MetroRepairTables",
    "build_metro_repair_tables",
    "metro_repair_assign_reference",
    "metro_repair_route",
    "supports",
]
