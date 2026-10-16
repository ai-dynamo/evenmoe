# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UltraEP quota rerouting in EvenMoE's existing L2 pipeline."""

from __future__ import annotations

import torch

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import RoutingPolicySpec
from evenmoe.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.kernels.ultraep import UltraEPKernels, UltraEPPlacementBackend


class UltraEPL2Router(TokenRouter):
    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, *, backend: UltraEPKernels | None = None) -> None:
        self._backend: UltraEPKernels = backend or UltraEPPlacementBackend()
        self._counters: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._placements: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        ] = {}

    def prepare_layer(self, placement) -> None:
        self._placements[placement.layer_id] = (
            placement.logical_to_physical_candidates,
            placement.logical_to_physical_count,
            placement.metadata.get("rank_quota_prefix"),
        )

    def route_prepared_tokens(
        self,
        layer_id: int,
        logical_topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        try:
            logical_to_physical, replica_counts, rank_quota_prefix = self._placements[
                layer_id
            ]
        except KeyError as exc:
            raise RuntimeError(f"layer {layer_id} routing was not prepared") from exc

        if rank_quota_prefix is None:
            valid = logical_topk_ids >= 0
            physical_ids = logical_to_physical[
                logical_topk_ids.clamp_min(0).to(torch.int64), 0
            ]
            return torch.where(valid, physical_ids, logical_topk_ids)

        logical_ids = logical_topk_ids.contiguous()
        physical_ids = torch.empty_like(logical_ids, dtype=torch.int64)
        num_logical_experts = replica_counts.numel()
        key = (logical_ids.device, num_logical_experts)
        counters = self._counters.get(key)
        if counters is None:
            counters = torch.empty(
                num_logical_experts,
                dtype=torch.int32,
                device=logical_ids.device,
            )
            self._counters[key] = counters
        # reroute_sparse's CUDA binding requires int32 for these three. The
        # framework's own placement tensors follow its own convention
        # (typically int64) and stay that way in self._placements -- the
        # fallback branch above indexes with them directly and must keep
        # seeing whatever dtype the framework gave it. EvenMoE's own quota is
        # already int32 by construction; cast defensively rather than trust
        # that forever. .to() is a no-op when the dtype already matches.
        self._backend.reroute_sparse(
            logical_topk_ids=logical_ids,
            routed_topk_ids=physical_ids,
            logical_to_physical_map=logical_to_physical.to(torch.int32),
            logical_replica_counts=replica_counts.to(torch.int32),
            rank_quota_prefix=rank_quota_prefix.to(torch.int32),
            counters=counters,
        )
        return physical_ids

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        placement = request.placement
        if placement is None:
            raise ValueError("UltraEPL2Router requires a committed placement.")
        self.prepare_layer(placement)
        physical_ids = self.route_prepared_tokens(
            request.layer_id,
            request.logical_topk_ids,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


ROUTING_POLICY = RoutingPolicySpec(
    "ultraep",
    UltraEPL2Router.phase,
    requires_placement_state=True,
    requires_deepep=True,
    supports_concurrent_microbatches=False,
    graph_stability="realloc_on_placement_change",
    required_placement="ultraep",
    requires_redundancy=True,
    requires_placement_metadata=True,
    build_router=lambda context: UltraEPL2Router(),
)
