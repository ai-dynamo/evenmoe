# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile-aware L1 placement (``tile``), the placement half of ``tile_repair``.

In decode an expert receives a handful of tokens, and a grouped GEMM pays for
whole tiles of ``block_m`` rows, so the token imbalance EPLB balances is hidden
inside tile padding. This planner balances *expected tiles* instead:

1. Turn the recorded per-layer logical counts into per-forward mean counts
   ``mu_e = tokens_per_pass * share_e`` and expected tiles
   ``E[ceil(K / block_m)]`` with ``K ~ Poisson(mu_e)``.
2. Pack every logical expert once (no redundancy) with DeepSeek EPLB's
   balanced packing on those weights, ``L / R`` primaries per rank.
3. Fill ``X = (P - L) / R`` extra slots per rank with copies of the experts
   with the highest expected tiles, each on the rank that does not hold it yet
   and has the lowest expected tile load.

Each rank's slots are ``[primaries | extras]``, so the primary of every
logical expert is its lowest-local-index copy -- the copy ``tile_repair``
starts from. Tunables (``PlacementPolicyConfig.metadata``, then environment):
``block_m`` (``EVENMOE_TILE_BLOCK_M``, default 64; 0 = adaptive, picked once
from ``tokens_per_pass / num_physical``) and ``tokens_per_pass``
(``EVENMOE_TILE_TOKENS_PER_PASS``: expected routed (token, expert) pairs per
forward on the whole EP group; default ``128 * ep_size``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from evenmoe.core.l1 import PlacementPlanner
from evenmoe.core.types import PlacementPlan, PlacementPolicyConfig, PlacementRequest
from evenmoe.kernels.tile_repair import adaptive_block_m
from evenmoe.policies.l1.eplb.algorithms.deepseek import balanced_packing
from evenmoe.policies.tile_common import (
    expected_tiles,
    resolve_block_m,
    resolve_tokens_per_pass,
)


def _layer_counts(logical_count: object) -> Tensor:
    counts = logical_count if isinstance(logical_count, Tensor) else torch.as_tensor(logical_count)
    counts = counts.detach().to(torch.float64).cpu()
    if counts.dim() == 3:  # [observations or source ranks, layers, logical]
        counts = counts.sum(0)
    elif counts.dim() == 1:
        counts = counts.unsqueeze(0)
    if counts.dim() != 2:
        raise ValueError("logical_count must have shape [layers, experts] or [samples, layers, experts].")
    return counts


def tile_weights(logical_count: object, *, block_m: int, tokens_per_pass: float) -> Tensor:
    """``[layers, experts]`` expected tiles per forward (float64)."""

    counts = _layer_counts(logical_count).clamp(min=0.0)
    totals = counts.sum(-1, keepdim=True)
    num_logical = counts.shape[-1]
    uniform = torch.full_like(counts, 1.0 / num_logical)
    shares = torch.where(totals > 0, counts / totals.clamp(min=1e-300), uniform)
    return expected_tiles(shares * tokens_per_pass, block_m)


def _place_extras(
    weight: list[float], primary_rank: list[int], num_ranks: int, extras_per_rank: int
) -> list[list[int]]:
    """Extra copies per rank for one layer, in placement order."""

    load = [0.0] * num_ranks
    holders: list[set[int]] = [{r} for r in primary_rank]
    for e, r in enumerate(primary_rank):
        load[r] += weight[e]
    extras: list[list[int]] = [[] for _ in range(num_ranks)]
    remaining = extras_per_rank * num_ranks
    order = sorted(range(len(weight)), key=lambda e: (-weight[e], e))
    while remaining > 0:
        placed = 0
        for e in order:
            if remaining == 0:
                break
            candidates = [
                r for r in range(num_ranks) if r not in holders[e] and len(extras[r]) < extras_per_rank
            ]
            if not candidates:
                continue
            r = min(candidates, key=lambda q: (load[q], q))
            extras[r].append(e)
            holders[e].add(r)
            load[r] += weight[e] / len(holders[e])
            remaining -= 1
            placed += 1
        if placed == 0:
            # Every expert is already on every rank with a free slot (for
            # example a single rank); a same-rank duplicate is a valid, if
            # useless, slot filler.
            k = 0
            for r in range(num_ranks):
                while len(extras[r]) < extras_per_rank:
                    extras[r].append(order[k % len(order)])
                    k += 1
            remaining = 0
    return extras


class TilePlacementPlanner(PlacementPlanner):
    """Expected-tile EPLB without redundancy plus per-rank extra copies."""

    name = "tile"

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.name != self.name:
            raise ValueError(f"TilePlacementPlanner received policy {policy.name!r}.")
        if request.stats.logical_count is None:
            raise ValueError("TilePlacementPlanner requires request.stats.logical_count.")

        topology = request.topology
        num_physical = topology.num_physical_experts
        per_rank = topology.num_local_physical_experts
        num_ranks = num_physical // per_rank
        counts = _layer_counts(request.stats.logical_count)
        num_layers, num_logical = counts.shape
        if num_logical % num_ranks != 0:
            raise ValueError(
                f"tile placement needs the {num_logical} logical experts to divide over "
                f"{num_ranks} ranks."
            )
        if num_physical < num_logical or (num_physical - num_logical) % num_ranks != 0:
            raise ValueError(
                f"tile placement needs (num_physical - num_logical) = "
                f"{num_physical - num_logical} to be a non-negative multiple of {num_ranks}."
            )
        primaries_per_rank = num_logical // num_ranks
        extras_per_rank = (num_physical - num_logical) // num_ranks

        block_m = resolve_block_m(policy.metadata)
        tokens_per_pass = resolve_tokens_per_pass(policy.metadata, num_ranks)
        if block_m == 0:
            block_m = adaptive_block_m(round(tokens_per_pass), num_physical)
        weight = tile_weights(counts, block_m=block_m, tokens_per_pass=tokens_per_pass)
        pack_index, rank_in_pack = balanced_packing(weight, num_ranks)
        pack_index = pack_index.cpu()
        rank_in_pack = rank_in_pack.cpu()

        phy2log = torch.empty(num_layers, num_physical, dtype=torch.int64)
        log2phy_lists: list[list[list[int]]] = []
        for layer in range(num_layers):
            ranks = pack_index[layer].tolist()
            slots = rank_in_pack[layer].tolist()
            copies: list[list[int]] = [[] for _ in range(num_logical)]
            for e in range(num_logical):
                p = ranks[e] * per_rank + slots[e]
                phy2log[layer, p] = e
                copies[e].append(p)
            extras = _place_extras(weight[layer].tolist(), ranks, num_ranks, extras_per_rank)
            for r, experts in enumerate(extras):
                for k, e in enumerate(experts):
                    p = r * per_rank + primaries_per_rank + k
                    phy2log[layer, p] = e
                    copies[e].append(p)
            log2phy_lists.append(copies)

        max_copies = max(len(c) for layer in log2phy_lists for c in layer)
        log2phy = torch.full((num_layers, num_logical, max_copies), -1, dtype=torch.int64)
        logcnt = torch.zeros(num_layers, num_logical, dtype=torch.int64)
        for layer, copies in enumerate(log2phy_lists):
            for e, slots in enumerate(copies):
                # Primary first, then extras by (local index, physical id): the
                # copy order tile_repair's routing tables use.
                ordered = slots[:1] + sorted(slots[1:], key=lambda p: (p % per_rank, p))
                log2phy[layer, e, : len(ordered)] = torch.tensor(ordered, dtype=torch.int64)
                logcnt[layer, e] = len(ordered)

        return PlacementPlan(
            physical_to_logical_map=phy2log,
            logical_to_all_physical_map=log2phy,
            logical_to_physical_count=logcnt,
            metadata={
                "requested_policy": self.name,
                "resolved_policy": self.name,
                "block_m": block_m,
                "tokens_per_pass": tokens_per_pass,
                "extra_slots_per_rank": extras_per_rank,
                "expected_tiles": weight,
            },
        )


__all__ = ["TilePlacementPlanner", "tile_weights"]
