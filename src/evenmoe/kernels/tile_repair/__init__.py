# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tile_repair: fixed-first + repair routing in GEMM tiles of ``block_m`` rows.

A grouped GEMM processes each expert's tokens in tiles of ``block_m`` rows, so
in decode a rank's MoE time follows its number of tiles
``sum_e ceil(m_e / block_m)``, not its token count. tile_repair starts every
logical expert on its primary copy and then moves single partial tiles of
replicated experts off the busiest rank while that strictly lowers the busiest
rank's tile count (see ``csrc/route.cuh`` for the exact rule).

Every EP rank evaluates the same deterministic function of the EP-wide
per-source-rank logical counts ``counts[R, L]`` and of a rank-invariant
placement table, so all ranks agree on the plan without further
communication, and the per-copy token totals equal the plan exactly.

``block_m == 0`` is adaptive: each call picks 8/16/32/48/64 rows from the
expected rows per physical slot (:func:`adaptive_block_m`).

Layout:

- ``csrc/route.cuh``  the fused kernels (totals, repair, exact-quota rewrite)
                      and their tvm-ffi entry points; compiled per (id dtype,
                      output dtype). ``block_m``, ``max_moves`` and the source
                      rank are runtime arguments. ``tile_repair_route`` reads
                      gathered counts; ``tile_repair_route_p2p``
                      (:mod:`.p2p`) gathers them itself over NVLink.
- :func:`tile_repair_plan_reference` / :func:`tile_repair_route_reference`
  are bit-exact host mirrors of the kernel used as the test oracle and as the
  CPU path.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evenmoe.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HERE = pathlib.Path(__file__).parent.resolve()

MAX_LOGICAL = 512
"""Logical experts per layer covered by the kernel's shared-memory tables."""

MAX_RANKS = 32
"""EP ranks, one warp lane each in the repair loop."""

MAX_COPIES = 4
"""Distinct-rank copies per logical expert the router can use. Copies on
further ranks (lowest local index first) are never routed to."""

NUM_STATS = 6
"""Leading entries of the optional stats output: moves, lower bound,
primary-only peak tiles, final peak tiles, the ``block_m`` used, and the P2P
exchange failure bits (:data:`P2P_FAIL_TIMEOUT` | :data:`P2P_FAIL_OVERWRITTEN`
| :data:`P2P_FAIL_OVERFLOW`; always 0 on the gathered-count path); per-rank
tiles follow."""

P2P_FAIL_TIMEOUT = 1
"""A peer's count did not arrive within the timeout."""
P2P_FAIL_OVERWRITTEN = 2
"""A peer had already moved on to a later call (this call's data is gone)."""
P2P_FAIL_OVERFLOW = 4
"""Some rank had 0xFFFF or more entries of one expert (the 16-bit count field)."""

ADAPTIVE_BLOCK_M = 0
"""``block_m`` value that selects the tile height per call."""


def adaptive_block_m(total_rows: int, num_slots: int) -> int:
    """Tile height for ``total_rows`` routed entries over ``num_slots`` physical slots.

    Mirrors the kernel: expected rows per slot ``em = total_rows / num_slots``
    in exact integer compares; em < 5.5 -> 8, < 12.7 -> 16, < 28.4 -> 32,
    < 80 -> 48, else 64 (the M tiles a small-M autotuned grouped GEMM such as
    Humming switches between).
    """

    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")
    rows, slots = int(total_rows), int(num_slots)
    if 2 * rows < 11 * slots:
        return 8
    if 10 * rows < 127 * slots:
        return 16
    if 10 * rows < 284 * slots:
        return 32
    if rows < 80 * slots:
        return 48
    return 64


@dataclass(frozen=True)
class TileRepairTables:
    """Rank-invariant per-layer copy tables.

    ``copy_physical[l, j]`` / ``copy_rank[l, j]`` list logical ``l``'s copies on
    distinct ranks, primary (lowest local slot index) first, -1 padded to
    :data:`MAX_COPIES`. ``replicated`` lists the logical ids with more than one
    usable copy in ascending order, -1 padded to ``[L]``; ``num_replicated`` is
    their count as a one-element device tensor so a same-shape placement update
    can be copied in place under a captured CUDA graph.
    """

    copy_physical: Tensor
    copy_rank: Tensor
    num_copies: Tensor
    replicated: Tensor
    num_replicated: Tensor
    num_ranks: int

    @property
    def num_logical(self) -> int:
        return int(self.copy_physical.shape[0])

    @property
    def primary_physical(self) -> Tensor:
        return self.copy_physical[:, 0]

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.copy_physical,
            self.copy_rank,
            self.num_copies,
            self.replicated,
            self.num_replicated,
        )


def build_tile_repair_tables(
    physical_to_logical_map: Tensor,
    *,
    num_logical: int,
    num_ranks: int,
    num_local_physical_experts: int | None = None,
    physical_to_rank_map: Tensor | None = None,
    device: torch.device | str | None = None,
) -> TileRepairTables:
    """Build :class:`TileRepairTables` for one layer from its physical->logical map.

    A slot's rank is ``physical_to_rank_map[p]`` when given, else
    ``p // num_local_physical_experts``. Its local index is its position among
    the slots of its rank in ascending physical id. Copies of one logical expert
    are ordered by (local index, physical id); a second copy on a rank that
    already holds one is merged into the first (it adds no flexibility).
    """

    p2l = torch.as_tensor(physical_to_logical_map).reshape(-1).to(torch.int64).cpu()
    num_physical = p2l.numel()
    if device is None:
        device = (
            physical_to_logical_map.device
            if isinstance(physical_to_logical_map, Tensor)
            else torch.device("cpu")
        )
    if num_logical <= 0 or num_ranks <= 0:
        raise ValueError("num_logical and num_ranks must be positive.")
    if bool(((p2l < 0) | (p2l >= num_logical)).any()):
        raise ValueError("physical_to_logical_map holds an out-of-range logical expert.")
    if physical_to_rank_map is not None:
        ranks = torch.as_tensor(physical_to_rank_map).reshape(-1).to(torch.int64).cpu()
        if ranks.numel() != num_physical:
            raise ValueError("physical_to_rank_map must have one entry per physical expert.")
    else:
        if num_local_physical_experts is None or num_local_physical_experts <= 0:
            raise ValueError("num_local_physical_experts is required without a rank map.")
        ranks = torch.arange(num_physical) // num_local_physical_experts
    if bool(((ranks < 0) | (ranks >= num_ranks)).any()):
        raise ValueError("A physical expert maps to a rank outside the EP group.")

    rank_list = ranks.tolist()
    logical_list = p2l.tolist()
    local_index = [0] * num_physical
    next_local = [0] * num_ranks
    for p in range(num_physical):
        local_index[p] = next_local[rank_list[p]]
        next_local[rank_list[p]] += 1

    copies: list[list[int]] = [[] for _ in range(num_logical)]
    for p in sorted(range(num_physical), key=lambda q: (local_index[q], q)):
        copies[logical_list[p]].append(p)

    copy_physical = torch.full((num_logical, MAX_COPIES), -1, dtype=torch.int32)
    copy_rank = torch.full((num_logical, MAX_COPIES), -1, dtype=torch.int32)
    num_copies = torch.zeros(num_logical, dtype=torch.int32)
    replicated: list[int] = []
    for e, slots in enumerate(copies):
        if not slots:
            raise ValueError(f"Logical expert {e} has no physical copy.")
        seen: set[int] = set()
        keep: list[int] = []
        for p in slots:
            if rank_list[p] not in seen:
                seen.add(rank_list[p])
                keep.append(p)
        keep = keep[:MAX_COPIES]
        num_copies[e] = len(keep)
        for j, p in enumerate(keep):
            copy_physical[e, j] = p
            copy_rank[e, j] = rank_list[p]
        if len(keep) > 1:
            replicated.append(e)
    replicated_t = torch.full((num_logical,), -1, dtype=torch.int32)
    replicated_t[: len(replicated)] = torch.tensor(replicated, dtype=torch.int32)
    return TileRepairTables(
        copy_physical=copy_physical.to(device),
        copy_rank=copy_rank.to(device),
        num_copies=num_copies.to(device),
        replicated=replicated_t.to(device),
        num_replicated=torch.tensor([len(replicated)], dtype=torch.int32, device=device),
        num_ranks=num_ranks,
    )


def supports(num_logical: int, num_ranks: int) -> bool:
    """Whether the fused kernel covers this geometry."""

    return 0 < num_logical <= MAX_LOGICAL and 0 < num_ranks <= MAX_RANKS


def _ceil(m: int, block_m: int) -> int:
    return -(-m // block_m)


@dataclass(frozen=True)
class TileRepairPlan:
    """Host result of :func:`tile_repair_plan_reference`."""

    copy_tokens: Tensor
    """``[L, MAX_COPIES]`` tokens each copy receives (0 for padding)."""
    rank_tiles: Tensor
    """``[R]`` tiles per rank after repair."""
    moves: int
    lower_bound: int
    primary_peak: int
    block_m: int
    """Tile height used (resolved when the call was adaptive)."""

    @property
    def peak(self) -> int:
        return int(self.rank_tiles.max().item()) if self.rank_tiles.numel() else 0


def tile_repair_plan_reference(
    counts: Tensor,
    tables: TileRepairTables,
    *,
    block_m: int,
    max_moves: int,
    num_slots: int | None = None,
) -> TileRepairPlan:
    """Host oracle of the kernel's plan for EP-wide per-source counts ``[R, L]``.

    ``block_m == 0`` is adaptive and needs ``num_slots`` (physical experts).
    """

    if block_m < 0:
        raise ValueError("block_m must be non-negative (0 = adaptive).")
    if block_m == 0 and not num_slots:
        raise ValueError("Adaptive block_m needs num_slots.")
    ranks = tables.num_ranks
    num_logical = tables.num_logical
    counts = torch.as_tensor(counts).detach().cpu()
    if counts.shape != (ranks, num_logical):
        raise ValueError(f"counts must have shape [{ranks}, {num_logical}], got {tuple(counts.shape)}.")
    totals = counts.to(torch.float64).round().to(torch.int64).sum(0).tolist()
    if block_m == 0:
        assert num_slots is not None
        block_m = adaptive_block_m(sum(totals), num_slots)
    copy_rank = tables.copy_rank.cpu().tolist()
    num_copies = tables.num_copies.cpu().tolist()
    num_rep = int(tables.num_replicated.cpu()[0])
    reps = tables.replicated.cpu().tolist()[:num_rep]

    tiles = [0] * ranks
    tokens = [0] * ranks
    tile_sum = 0
    for e, g in enumerate(totals):
        if g > 0:
            r = copy_rank[e][0]
            t = _ceil(g, block_m)
            tiles[r] += t
            tokens[r] += g
            tile_sum += t
    rm = [[totals[e] if j == 0 else 0 for j in range(MAX_COPIES)] for e in reps]
    rk = [[copy_rank[e][j] if j < num_copies[e] else -1 for j in range(MAX_COPIES)] for e in reps]
    lower_bound = _ceil(tile_sum, ranks)
    primary_peak = max(tiles)
    moves = 0
    while reps and moves < max_moves:
        busiest = max(range(ranks), key=lambda r: (tiles[r], tokens[r], -r))
        peak = tiles[busiest]
        if peak <= lower_bound:
            break
        best = None
        for i in range(len(reps)):
            src = -1
            for j in range(MAX_COPIES):
                if rk[i][j] == busiest and rm[i][j] > 0:
                    src = j
            if src < 0:
                continue
            m = rm[i][src]
            chunk = m - block_m * (_ceil(m, block_m) - 1)
            for j in range(MAX_COPIES):
                dst = rk[i][j]
                if j == src or dst < 0:
                    continue
                add = _ceil(rm[i][j] + chunk, block_m) - _ceil(rm[i][j], block_m)
                if tiles[dst] + add >= peak:
                    continue
                key = (tiles[dst] + add, tokens[dst], block_m - chunk, i * MAX_COPIES + j)
                if best is None or key < best[0]:
                    best = (key, i, src, j, chunk, add)
        if best is None:
            break
        _, i, src, j, chunk, add = best
        dst = rk[i][j]
        rm[i][src] -= chunk
        rm[i][j] += chunk
        tiles[busiest] -= 1
        tiles[dst] += add
        tokens[busiest] -= chunk
        tokens[dst] += chunk
        moves += 1

    copy_tokens = torch.zeros(num_logical, MAX_COPIES, dtype=torch.int64)
    for e in range(num_logical):
        copy_tokens[e, 0] = totals[e]
    for i, e in enumerate(reps):
        copy_tokens[e] = torch.tensor(rm[i], dtype=torch.int64)
    return TileRepairPlan(
        copy_tokens=copy_tokens,
        rank_tiles=torch.tensor(tiles, dtype=torch.int64),
        moves=moves,
        lower_bound=lower_bound,
        primary_peak=primary_peak,
        block_m=block_m,
    )


def tile_repair_route_reference(
    logical_topk_ids: Tensor,
    counts: Tensor,
    tables: TileRepairTables,
    *,
    source_rank: int,
    block_m: int,
    max_moves: int,
    plan: TileRepairPlan | None = None,
    num_slots: int | None = None,
) -> Tensor:
    """Host mirror of :func:`tile_repair_route` (bit-exact, same output dtype as the ids)."""

    if plan is None:
        plan = tile_repair_plan_reference(
            counts, tables, block_m=block_m, max_moves=max_moves, num_slots=num_slots
        )
    counts_h = torch.as_tensor(counts).detach().cpu().to(torch.float64).round().to(torch.int64)
    prefix = counts_h[:source_rank].sum(0).tolist()
    num_logical = tables.num_logical
    copy_physical = tables.copy_physical.cpu().tolist()
    num_copies = tables.num_copies.cpu().tolist()
    num_rep = int(tables.num_replicated.cpu()[0])
    replicated = set(tables.replicated.cpu().tolist()[:num_rep])
    quota = plan.copy_tokens.tolist()
    flat = logical_topk_ids.detach().cpu().reshape(-1).to(torch.int64).tolist()
    seen = [0] * num_logical
    out = []
    for e in flat:
        if e < 0 or e >= num_logical:
            out.append(-1)
            continue
        if e not in replicated:
            out.append(copy_physical[e][0])
            continue
        position = prefix[e] + seen[e]
        seen[e] += 1
        chosen, end = 0, 0
        for j in range(num_copies[e]):
            end += quota[e][j]
            if position < end:
                chosen = j
                break
        out.append(copy_physical[e][chosen])
    result = torch.tensor(out, dtype=torch.int64).view(logical_topk_ids.shape)
    return result.to(device=logical_topk_ids.device, dtype=logical_topk_ids.dtype)


@cache_once
def _route_module(in_dtype: torch.dtype, out_dtype: torch.dtype) -> Module:
    args = make_cpp_args(in_dtype, out_dtype)
    return load_jit(
        "tile_repair_route",
        *args,
        cuda_files=["route.cuh"],
        cuda_wrappers=[
            ("tile_repair_route", f"tile_repair_route<{args}>"),
            ("tile_repair_route_p2p", f"tile_repair_route_p2p<{args}>"),
        ],
        source_root=_HERE,
    )


def _check_route_args(
    logical_topk_ids: Tensor,
    tables: TileRepairTables,
    *,
    block_m: int,
    max_moves: int,
    num_slots: int | None,
    out_dtype: torch.dtype,
) -> None:
    if not logical_topk_ids.is_cuda:
        raise ValueError("tile_repair_route runs on CUDA tensors; use the reference on CPU.")
    num_logical = tables.num_logical
    ranks = tables.num_ranks
    if not supports(num_logical, ranks):
        raise ValueError(
            f"tile_repair supports <= {MAX_LOGICAL} experts and <= {MAX_RANKS} ranks; "
            f"got {num_logical} and {ranks}."
        )
    if logical_topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"logical_topk_ids must be int32 or int64, got {logical_topk_ids.dtype}.")
    if out_dtype not in (torch.int32, torch.int64):
        raise TypeError(f"out_dtype must be int32 or int64, got {out_dtype}.")
    if block_m < 0 or max_moves < 0:
        raise ValueError("block_m (0 = adaptive) and max_moves must be non-negative.")
    if block_m == 0 and not num_slots:
        raise ValueError("Adaptive block_m (0) needs num_slots, the number of physical experts.")


def tile_repair_route(
    logical_topk_ids: Tensor,
    counts: Tensor,
    tables: TileRepairTables,
    *,
    source_rank: int,
    block_m: int,
    max_moves: int = 32,
    out_dtype: torch.dtype = torch.int64,
    plan_out: Tensor | None = None,
    stats_out: Tensor | None = None,
    num_slots: int | None = None,
) -> Tensor:
    """Map logical TopK ids ``[T, K]`` to physical copies with tile_repair.

    ``counts`` is the EP-wide ``[R, L]`` per-source-rank logical count (any
    float or integer dtype; it is passed to the kernel as float32, exact up
    to 2**24 per entry). ``plan_out`` (``[L * MAX_COPIES]`` int32) and
    ``stats_out`` (``[NUM_STATS + R]`` int32) optionally receive the plan.
    ``block_m == 0`` is adaptive and needs ``num_slots`` (physical experts).
    One launch on the current stream, no host synchronisation, CUDA-graph safe.
    """

    _check_route_args(
        logical_topk_ids,
        tables,
        block_m=block_m,
        max_moves=max_moves,
        num_slots=num_slots,
        out_dtype=out_dtype,
    )
    num_logical = tables.num_logical
    ranks = tables.num_ranks
    if counts.shape != (ranks, num_logical):
        raise ValueError(f"counts must have shape [{ranks}, {num_logical}], got {tuple(counts.shape)}.")
    ids = logical_topk_ids if logical_topk_ids.dim() == 2 else logical_topk_ids.reshape(-1, 1)
    ids = ids.contiguous()
    device = ids.device
    counts_arg = counts if counts.dtype == torch.float32 else counts.to(torch.float32)
    counts_arg = counts_arg.contiguous()
    empty = torch.empty(0, dtype=torch.int32, device=device)
    out = torch.empty(ids.shape, dtype=out_dtype, device=device)
    _route_module(ids.dtype, out_dtype).tile_repair_route(
        ids,
        out,
        counts_arg,
        tables.copy_physical,
        tables.copy_rank,
        tables.num_copies,
        tables.replicated,
        tables.num_replicated,
        empty if plan_out is None else plan_out,
        empty if stats_out is None else stats_out,
        int(source_rank),
        int(block_m),
        int(num_slots or 0),
        int(max_moves),
    )
    return out.view(logical_topk_ids.shape)


__all__ = [
    "ADAPTIVE_BLOCK_M",
    "MAX_COPIES",
    "MAX_LOGICAL",
    "MAX_RANKS",
    "NUM_STATS",
    "P2P_FAIL_OVERFLOW",
    "P2P_FAIL_OVERWRITTEN",
    "P2P_FAIL_TIMEOUT",
    "TileRepairPlan",
    "TileRepairTables",
    "adaptive_block_m",
    "build_tile_repair_tables",
    "supports",
    "tile_repair_plan_reference",
    "tile_repair_route",
    "tile_repair_route_reference",
]
