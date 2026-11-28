# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer adaptive placement: re-plan only the layers whose placement drifted.

A recorded EPLB placement is right for the traffic it was recorded on and
wrong, layer by layer, for traffic from another domain. Which layers drift
depends on the new traffic, so it cannot be decided when the placement is
recorded. This planner decides it online, per layer, every time the framework
asks for a rebalance:

* it plans a fresh EPLB placement from the window's counts;
* for every layer it compares the window's rank imbalance (max / mean of the
  routed rows per EP rank, a logical expert's rows split evenly over its
  replicas) under the committed placement with the fresh one;
* a layer keeps its committed placement unless that imbalance exceeds
  ``threshold`` **and** a fresh placement lowers it by at least ``min_gain``
  on samples it was not fitted on (fit on half of the window, score on the
  other half, both ways), so a re-plan that would only fit the window's noise
  is not made; stable layers therefore never move a byte;
* a re-planned layer is relabelled to the committed one before it is
  returned: the fresh plan's ranks are matched to the committed ranks that
  already hold most of their experts, and within a rank an expert stays in
  the slot it already occupies, so the framework moves only what changed.

Without a committed placement in ``PlacementRequest.current`` (for example
the startup placement computed from a recorded distribution) it returns the
fresh EPLB plan unchanged, identical to the ``auto`` EPLB policy.

A layer is judged only when the window holds enough routed rows for its
imbalance to mean something (``min_rows_per_slot`` per physical slot on
average); a window of a few decode steps leaves every layer as it is.

Tunables (environment, read on first use): ``EVENMOE_ADAPTIVE_THRESHOLD``
(default 1.10), ``EVENMOE_ADAPTIVE_MIN_GAIN`` (default 0.03),
``EVENMOE_ADAPTIVE_MIN_ROWS_PER_SLOT`` (default 200) and
``EVENMOE_ADAPTIVE_BASE`` (EPLB variant for the fresh plan, default ``auto``).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping

import torch
from torch import Tensor

from evenmoe.core.l1 import PlacementPlanner
from evenmoe.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
)
from evenmoe.policies.l1.eplb.algorithms import compute_algorithm, rebalance_experts

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 1.10
_DEFAULT_MIN_GAIN = 0.03
_DEFAULT_MIN_ROWS_PER_SLOT = 200.0


def rank_imbalance(physical_to_logical: Tensor, count: Tensor, ep_size: int) -> Tensor:
    """Max / mean of the routed rows per EP rank, ``[layers]``.

    ``physical_to_logical`` is ``[layers, physical]``; ``count`` is
    ``[layers, logical]``. Each logical expert's rows split evenly over its
    replicas (the dynamic replica rule's expectation). Layers without rows
    return 1.
    """

    p2l = physical_to_logical.to(torch.int64)
    count = count.to(torch.float64)
    num_layers, num_physical = p2l.shape
    replicas = torch.zeros_like(count)
    replicas.scatter_add_(1, p2l, torch.ones_like(p2l, dtype=count.dtype))
    per_slot = torch.gather(count / replicas.clamp_min(1), 1, p2l)
    per_rank = per_slot.view(num_layers, ep_size, num_physical // ep_size).sum(-1)
    mean = per_rank.mean(-1)
    out = per_rank.max(-1).values / mean.clamp_min(1e-12)
    return torch.where(mean > 0, out, torch.ones_like(out))


def _assignment(weight: list[list[int]]) -> list[int]:
    """Maximum-weight perfect matching, rows to columns (Hungarian, O(n^3))."""

    n = len(weight)
    big = max((max(row) for row in weight), default=0)
    cost = [[big - w for w in row] for row in weight]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    match = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            match[p[j] - 1] = j - 1
    return match


def relabel_to_committed(
    fresh: list[int], committed: list[int], *, ep_size: int, num_nodes: int = 1
) -> list[int]:
    """Re-express one layer's fresh placement with minimal weight movement.

    Ranks are interchangeable within a node, and slots within a rank, so the
    fresh plan's rank groups are matched (inside each node) to the committed
    ranks holding most of their experts, and an expert already present on its
    new rank keeps its slot. The result has exactly the fresh plan's per-rank
    expert multisets, only on different ranks and slots.
    """

    num_physical = len(fresh)
    local = num_physical // ep_size
    ranks_per_node = ep_size // num_nodes
    groups_new = [fresh[r * local : (r + 1) * local] for r in range(ep_size)]
    groups_old = [committed[r * local : (r + 1) * local] for r in range(ep_size)]

    def overlap(a: list[int], b: list[int]) -> int:
        left: dict[int, int] = {}
        for e in a:
            left[e] = left.get(e, 0) + 1
        hit = 0
        for e in b:
            if left.get(e, 0) > 0:
                left[e] -= 1
                hit += 1
        return hit

    target = [0] * ep_size
    for node in range(num_nodes):
        ranks = list(range(node * ranks_per_node, (node + 1) * ranks_per_node))
        weight = [[overlap(groups_new[a], groups_old[b]) for b in ranks] for a in ranks]
        for i, j in enumerate(_assignment(weight)):
            target[ranks[i]] = ranks[j]

    out = [0] * num_physical
    for new_rank, rank in enumerate(target):
        old = groups_old[rank]
        pending: dict[int, int] = {}
        for e in groups_new[new_rank]:
            pending[e] = pending.get(e, 0) + 1
        slots: list[int | None] = [None] * local
        for s, e in enumerate(old):
            if pending.get(e, 0) > 0:
                slots[s] = e
                pending[e] -= 1
        rest = [e for e, n in sorted(pending.items()) for _ in range(n)]
        it = iter(rest)
        for s in range(local):
            if slots[s] is None:
                slots[s] = next(it)
        out[rank * local : (rank + 1) * local] = slots  # type: ignore[assignment]
    return out


def maps_from_physical_to_logical(
    physical_to_logical: Tensor, num_logical: int
) -> tuple[Tensor, Tensor]:
    """``(logical_to_all_physical_map, logical_to_physical_count)`` of a placement.

    Replicas are listed in ascending physical id and padded with -1 to the
    largest replica count, the layout of the EPLB algorithms' output.
    """

    p2l = physical_to_logical.to(torch.int64).cpu()
    num_layers, num_physical = p2l.shape
    count = torch.zeros(num_layers, num_logical, dtype=torch.int64)
    count.scatter_add_(1, p2l, torch.ones_like(p2l))
    width = max(int(count.max()), 1)
    # Sort slots by (logical, physical); a slot's rank among its logical
    # expert's replicas is its position minus the expert's first position.
    phys = torch.arange(num_physical).expand(num_layers, num_physical)
    order = torch.argsort(p2l * num_physical + phys, dim=1)
    logical_sorted = torch.gather(p2l, 1, order)
    start = torch.cumsum(count, dim=1) - count
    slot_rank = torch.arange(num_physical).expand(num_layers, num_physical) - torch.gather(
        start, 1, logical_sorted
    )
    l2p = torch.full((num_layers, num_logical, width), -1, dtype=torch.int64)
    layer_idx = torch.arange(num_layers).unsqueeze(1).expand(num_layers, num_physical)
    l2p[layer_idx, logical_sorted, slot_rank] = order
    return l2p, count


def held_out_gain(
    sub_count: Tensor,
    committed: Tensor,
    fresh_plan,
    ep_size: int,
    *,
    fallback: Tensor,
) -> Tensor:
    """Imbalance a re-plan removes on samples it was not fitted on, ``[layers]``.

    A plan fitted on a window always beats the committed placement on that
    same window, if only by fitting its noise. Fit on the even samples and
    score on the odd ones, then the other way round, and average: a layer whose
    traffic really drifted gains on held-out samples too, a layer that only
    fluctuates does not. With fewer than two samples the in-window gain
    (``fallback``) is used.
    """

    if sub_count.shape[0] < 2:
        return fallback
    halves = (sub_count[0::2], sub_count[1::2])
    gains = []
    for fit, score in (halves, halves[::-1]):
        plan, _, _ = fresh_plan(fit)
        held = score.to(torch.float64).sum(0).cpu()
        gains.append(
            rank_imbalance(committed, held, ep_size)
            - rank_imbalance(plan.to("cpu", torch.int64), held, ep_size)
        )
    return (gains[0] + gains[1]) / 2


class AdaptivePlacementPlanner(PlacementPlanner):
    """Keep each layer's committed placement unless its traffic has drifted."""

    name = "adaptive"
    logical_count = "global"

    def __init__(
        self,
        *,
        threshold: float | None = None,
        min_gain: float | None = None,
        min_rows_per_slot: float | None = None,
        base: str | None = None,
    ) -> None:
        self._threshold = threshold
        self._min_gain = min_gain
        self._min_rows_per_slot = min_rows_per_slot
        self._base = base

    def _tunables(self) -> tuple[float, float, float, str]:
        threshold = self._threshold
        if threshold is None:
            threshold = float(os.environ.get("EVENMOE_ADAPTIVE_THRESHOLD", _DEFAULT_THRESHOLD))
        min_gain = self._min_gain
        if min_gain is None:
            min_gain = float(os.environ.get("EVENMOE_ADAPTIVE_MIN_GAIN", _DEFAULT_MIN_GAIN))
        min_rows = self._min_rows_per_slot
        if min_rows is None:
            min_rows = float(
                os.environ.get("EVENMOE_ADAPTIVE_MIN_ROWS_PER_SLOT", _DEFAULT_MIN_ROWS_PER_SLOT)
            )
        base = self._base or os.environ.get("EVENMOE_ADAPTIVE_BASE", "auto")
        if threshold < 1.0 or min_gain < 0.0 or min_rows < 0.0:
            raise ValueError(
                "Adaptive placement needs threshold >= 1, min_gain >= 0 and min_rows_per_slot >= 0."
            )
        return threshold, min_gain, min_rows, base

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.name != self.name:
            raise ValueError(f"AdaptivePlacementPlanner received policy {policy.name!r}.")
        threshold, min_gain, min_rows, base = self._tunables()
        started = time.perf_counter()

        logical_count = request.stats.logical_count
        if logical_count is None:
            raise ValueError("Adaptive placement requires request.stats.logical_count.")
        if not isinstance(logical_count, Tensor):
            logical_count = torch.as_tensor(logical_count)
        if logical_count.dim() == 2:
            logical_count = logical_count.unsqueeze(0)
        if logical_count.dim() != 3:
            raise ValueError("logical_count must have shape [samples, layers, experts].")

        topology = request.topology
        ep_size = topology.num_physical_experts // topology.num_local_physical_experts
        algorithm = compute_algorithm(
            raw_algorithm=base, num_groups=topology.num_groups, num_nodes=topology.num_nodes
        )
        metadata = {"requested_policy": self.name, "resolved_policy": algorithm.name}

        def fresh_plan(count: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            # EPLB plans every layer independently, so a subset of layers can be planned alone.
            return rebalance_experts(
                tokens_per_expert=count,
                num_physical_experts=topology.num_physical_experts,
                num_local_physical_experts=topology.num_local_physical_experts,
                num_groups=topology.num_groups,
                num_nodes=topology.num_nodes,
                algorithm=algorithm,
                active_ranks=topology.active_ranks,
            )

        current: Mapping[int, PlacementSnapshot] | None = request.current
        num_layers, num_logical = logical_count.shape[1:]
        if not current or any(layer not in current for layer in range(num_layers)):
            # No committed placement to keep (startup): plain EPLB.
            logger.info(
                "adaptive placement: no committed placement, planning every layer with EPLB"
            )
            p2l, l2p, count = fresh_plan(logical_count)
            return PlacementPlan(
                physical_to_logical_map=p2l,
                logical_to_all_physical_map=l2p,
                logical_to_physical_count=count,
                metadata={**metadata, "adaptive": "no committed placement"},
            )

        committed = torch.stack(
            [
                torch.as_tensor(current[layer].physical_to_logical_map).to("cpu", torch.int64)
                for layer in range(num_layers)
            ]
        )
        window = logical_count.to(torch.float64).sum(0).cpu()
        before = rank_imbalance(committed, window, ep_size)
        after = before.clone()
        # Only layers already over the threshold, with enough rows for that to
        # be more than sampling noise, are worth a fresh plan.
        enough = window.sum(-1) >= min_rows * topology.num_physical_experts
        candidates = ((before > threshold) & enough).nonzero().flatten()
        out = committed.clone()
        replanned: list[int] = []
        moved = 0
        if candidates.numel() > 0:
            sub = logical_count[:, candidates.to(logical_count.device), :]
            fresh, _, _ = fresh_plan(sub)
            fresh = fresh.to("cpu", torch.int64)
            after[candidates] = rank_imbalance(fresh, window[candidates], ep_size)
            gain = held_out_gain(
                sub,
                committed[candidates],
                fresh_plan,
                ep_size,
                fallback=before[candidates] - after[candidates],
            )
            for i, layer in enumerate(candidates.tolist()):
                if float(gain[i]) < min_gain:
                    continue
                relabelled = relabel_to_committed(
                    fresh[i].tolist(),
                    committed[layer].tolist(),
                    ep_size=ep_size,
                    num_nodes=topology.num_nodes,
                )
                out[layer] = torch.tensor(relabelled, dtype=torch.int64)
                moved += int((out[layer] != committed[layer]).sum())
                replanned.append(layer)

        logger.info(
            "adaptive placement: %d/%d layers with enough rows, %d over %.3f, re-planned %d (%s), "
            "moved %d slots; max imbalance %.3f -> %.3f; %.1f ms",
            int(enough.sum()),
            num_layers,
            int(candidates.numel()),
            threshold,
            len(replanned),
            ",".join(map(str, replanned)),
            moved,
            float(before.max()),
            float(
                torch.where(
                    torch.isin(
                        torch.arange(num_layers), torch.tensor(replanned, dtype=torch.int64)
                    ),
                    after,
                    before,
                ).max()
            ),
            1e3 * (time.perf_counter() - started),
        )
        l2p, count = maps_from_physical_to_logical(out, num_logical)
        if replanned != list(range(num_layers)):
            # Kept layers keep the committed replica order too (it decides which
            # replica a token's hash picks), so nothing about them changes.
            kept = [layer for layer in range(num_layers) if layer not in set(replanned)]
            live = [
                torch.as_tensor(current[layer].logical_to_physical_candidates) for layer in kept
            ]
            width = max([l2p.shape[-1]] + [int(t.shape[-1]) for t in live])
            if width > l2p.shape[-1]:
                l2p = torch.nn.functional.pad(l2p, (0, width - l2p.shape[-1]), value=-1)
            for layer, cand in zip(kept, live):
                cand = cand.to("cpu", torch.int64)
                n = torch.as_tensor(current[layer].logical_to_physical_count).to("cpu", torch.int64)
                row = torch.full((num_logical, width), -1, dtype=torch.int64)
                row[:, : cand.shape[-1]] = cand
                row[torch.arange(width).expand(num_logical, width) >= n.unsqueeze(1)] = -1
                l2p[layer] = row
        device = logical_count.device
        return PlacementPlan(
            physical_to_logical_map=out.to(device),
            logical_to_all_physical_map=l2p.to(device),
            logical_to_physical_count=count.to(device),
            metadata={
                **metadata,
                "adaptive_threshold": threshold,
                "adaptive_min_gain": min_gain,
                "replanned_layers": tuple(replanned),
                "moved_slots": moved,
                "imbalance_before": before.tolist(),
                "imbalance_fresh": after.tolist(),
            },
        )
