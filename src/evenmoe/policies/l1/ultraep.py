# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UltraEP placement algorithm behind EvenMoE's existing planner interface.

UltraEP solves placement and routing together: this L1 half publishes the
per-replica token quota that its L2 half (``policies.l2.ultraep``) reads, so
the two are selected as a pair rather than composed with an unrelated router
the way ``deepseek`` / ``lplb`` / ``waterfill`` compose freely.

Placement only: moving expert weight to match a solved ``PlacementPlan`` is
the framework adapter's job, done with whatever mover that framework already
owns. ``RefreshGate`` paces how often the framework asks for a new solve.
"""

from __future__ import annotations

import torch

from evenmoe.core.l1 import PlacementPlanner
from evenmoe.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
)
from evenmoe.kernels.ultraep import (
    UltraEPKernels,
    UltraEPPlacementBackend,
    UltraEPPlacementTuning,
    load_placement_tuning_from_env,
)


class UltraEPPlacementPolicy(PlacementPlanner):
    """Compute replica placement from per-rank logical expert loads."""

    name = "ultraep"

    def __init__(
        self,
        *,
        backend: UltraEPKernels | None = None,
        tuning: UltraEPPlacementTuning | None = None,
    ) -> None:
        self._backend: UltraEPKernels = backend or UltraEPPlacementBackend()
        self._tuning = tuning

    @property
    def _resolved_tuning(self) -> UltraEPPlacementTuning:
        """Env-derived tuning, read on first use.

        ``load_placement_tuning_from_env`` raises on a malformed ``ULTRA_EP_*``
        value, and the orchestrator constructs every L1 policy up front, so
        reading here keeps that failure on the first solve that consumes the
        tuning rather than on the construction of a balancer that never selects
        this policy.
        """
        if self._tuning is None:
            self._tuning = load_placement_tuning_from_env()
        return self._tuning

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.name != self.name:
            raise ValueError(f"UltraEPPlacementPolicy received policy {policy.name!r}.")

        logical_count = request.stats.logical_count
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.as_tensor(logical_count)
        if logical_count.ndim == 2:
            logical_count = logical_count.unsqueeze(0)
        if logical_count.ndim != 3:
            raise ValueError("logical_count must have shape [samples, layers, experts].")

        topology = request.topology
        ep_size = topology.num_physical_experts // topology.num_local_physical_experts
        num_layers, num_logical_experts = logical_count.shape[1:]
        if ep_size <= 1 or num_logical_experts % ep_size != 0:
            raise ValueError("UltraEP requires divisible experts and ep_size > 1.")
        num_local_master_experts = num_logical_experts // ep_size
        num_local_redundant_experts = topology.num_local_physical_experts - num_local_master_experts
        if num_local_redundant_experts <= 0:
            raise ValueError("UltraEP requires rank-local redundant expert slots.")

        rank = int(policy.metadata.get("rank", 0))
        num_nvl_ranks = int(policy.metadata.get("num_nvl_ranks", ep_size))
        per_rank_loads = logical_count.to(torch.int32).contiguous()
        global_loads = per_rank_loads.sum(dim=0, dtype=torch.int32).contiguous()
        max_replicas = ep_size
        # The kernel resets all four before it solves (the two maps to -1, the
        # rest to 0); zeros only avoid handing it uninitialised memory.
        physical_to_logical = torch.zeros(
            (num_layers, topology.num_physical_experts),
            dtype=torch.int32,
            device=global_loads.device,
        )
        logical_to_physical = torch.zeros(
            (num_layers, num_logical_experts, max_replicas),
            dtype=torch.int32,
            device=global_loads.device,
        )
        replica_counts = torch.zeros(
            (num_layers, num_logical_experts),
            dtype=torch.int32,
            device=global_loads.device,
        )
        rank_quota_prefix = torch.zeros_like(logical_to_physical)

        for layer_id in range(num_layers):
            # Non-contiguous whenever num_layers > 1: this slices the middle
            # dimension of a [ep_size, num_layers, num_logical_experts]
            # tensor, and the kernel reads it by raw pointer.
            per_rank = per_rank_loads[:, layer_id].contiguous()
            if per_rank.shape[0] != ep_size:
                raise ValueError("UltraEP placement requires one logical-load sample per EP rank.")
            logical_quota = torch.empty_like(logical_to_physical[layer_id])
            logical_quota_prefix = torch.empty_like(logical_quota)
            self._backend.solve(
                expert_loads=global_loads[layer_id],
                expert_loads_per_rank=per_rank,
                physical_to_logical_map=physical_to_logical[layer_id],
                logical_to_physical_map=logical_to_physical[layer_id],
                logical_replica_counts=replica_counts[layer_id],
                logical_instance_quota=logical_quota,
                logical_instance_quota_prefix=logical_quota_prefix,
                rank_quota_prefix=rank_quota_prefix[layer_id],
                num_ranks=ep_size,
                num_local_master_experts=num_local_master_experts,
                num_local_redundant_experts=num_local_redundant_experts,
                num_nvl_ranks=num_nvl_ranks,
                rank=rank,
                tuning=self._resolved_tuning,
            )

        # A redundant slot the solve declined (its replica would fall below the
        # tuning's token floor) is left at -1. Every slot in a PlacementPlan
        # must name a real logical expert, so such slots take this rank's own
        # master experts round-robin, as the initial layout does. Nothing
        # routes to them: logical_to_all_physical_map, which the solve owns, is
        # left untouched.
        # Fill unassigned slots with their rank-local master unconditionally:
        # `if unassigned.any()` pulls a bool off the device and stalls the
        # stream once per solve, which under UltraEP's fast refresh is once per
        # layer per refresh. The where() is a no-op when nothing is unassigned.
        unassigned = physical_to_logical < 0
        physical_ids = torch.arange(
            topology.num_physical_experts, device=physical_to_logical.device
        )
        local_idx = physical_ids % topology.num_local_physical_experts
        rank_of_slot = physical_ids // topology.num_local_physical_experts
        fallback = (
            rank_of_slot * num_local_master_experts + local_idx % num_local_master_experts
        ).to(torch.int32)
        physical_to_logical = torch.where(unassigned, fallback, physical_to_logical)

        return PlacementPlan(
            physical_to_logical_map=physical_to_logical,
            logical_to_all_physical_map=logical_to_physical,
            logical_to_physical_count=replica_counts,
            metadata={
                "requested_policy": self.name,
                "resolved_policy": self.name,
                "rank_quota_prefix": rank_quota_prefix,
            },
        )


class RefreshGate:
    """Per-layer "N representative batches since the last refresh" gate.

    UltraEP ties placement freshness to how much traffic has been observed,
    not to wall-clock time (the reference SGLang integration's
    ``moe_balance_refresh_interval``, default 64 prefill batches). Which
    forward passes count as representative -- big enough, the right execution
    stage -- is the framework's call, made before ``should_refresh``.

    Split into two calls so a caller whose ``representative`` check is itself
    expensive (e.g. it needs a GPU-CPU sync to read a token count off a
    device tensor) only pays for it when a refresh could actually happen:
    call ``is_due(layer_id)`` every time (cheap, pure Python bookkeeping, no
    GPU access) and only compute ``representative`` + call ``should_refresh``
    when it returns True -- when it returns False, ``should_refresh`` would
    decline regardless of what ``representative`` turns out to be, so
    computing it is pure waste.
    """

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError("RefreshGate requires a positive interval.")
        self._interval = interval
        self._batches_since_refresh: dict[int, int] = {}

    def is_due(self, layer_id: int) -> bool:
        """Advance ``layer_id``'s call counter by one and report whether a
        refresh could happen this call.

        The counter lives here, not in ``should_refresh``, precisely so a
        caller can call this on every single pass without ever computing
        ``representative``: skipping this call on a "not yet due" pass would
        leave the counter permanently unadvanced, and the layer would never
        become due through natural progression.
        """
        if layer_id not in self._batches_since_refresh:
            return True
        batches = min(self._batches_since_refresh[layer_id] + 1, self._interval)
        self._batches_since_refresh[layer_id] = batches
        return batches >= self._interval

    def should_refresh(self, layer_id: int, *, representative: bool) -> bool:
        """Whether layer ``layer_id`` should be re-solved on this call.

        Call only immediately after ``is_due(layer_id)`` returned True for
        the same ``layer_id`` on this same pass -- this method trusts that
        the counter is already current and does not advance it itself.

        The first call for any ``layer_id`` always refreshes -- there is no
        placement for it yet. If that bootstrap call was not representative
        (too small a batch, warmup, ...), the layer is due again on its very
        next call rather than leaving a placement solved from thin data in
        place for a full interval. Otherwise a due call only actually
        refreshes when it is also representative; a due-but-not-
        representative call refreshes nothing and stays due next time.
        """
        if layer_id not in self._batches_since_refresh:
            self._batches_since_refresh[layer_id] = 0 if representative else self._interval
            return True
        if representative:
            self._batches_since_refresh[layer_id] = 0
            return True
        return False
