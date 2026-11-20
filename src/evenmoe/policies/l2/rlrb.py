# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-Local Replica Balancing (RLRB) for decode with a committed placement."""

from __future__ import annotations

import torch
from torch import Tensor

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import RouterBuildContext, RoutingPolicySpec
from evenmoe.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from evenmoe.kernels.rlrb import (
    BATCH_SIZES,
    EP_SIZES,
    NUM_LOGICAL_EXPERTS,
    NUM_PHYSICAL_EXPERTS,
    TOP_K,
    load_kernel,
    pack_placement,
)
from evenmoe.policies.l2.replica import route_dynamic_replicas


class RankLocalReplicaRouter(TokenRouter):
    """Project one update from local Top-K counts and sample a physical replica."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, *, ep_size: int, source_rank: int) -> None:
        if not 0 <= source_rank < ep_size:
            raise ValueError("RLRB source rank must belong to its EP group.")
        self.ep_size = ep_size
        self.source_rank = source_rank
        self._metadata: dict[int, Tensor] = {}

    def prepare_layer(self, placement: PlacementSnapshot) -> None:
        """Pack a committed placement before capture and update it in place."""
        p2l = placement.physical_to_logical_map
        metadata = self._metadata.get(placement.layer_id)
        if (
            placement.ep_size != self.ep_size
            or self.ep_size not in EP_SIZES
            or placement.num_logical_experts != NUM_LOGICAL_EXPERTS
            or placement.num_physical_experts != NUM_PHYSICAL_EXPERTS
            or placement.num_local_physical_experts * self.ep_size != NUM_PHYSICAL_EXPERTS
            or not isinstance(p2l, Tensor)
            or not p2l.is_cuda
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(p2l.device) not in ((10, 0), (10, 3))
        ):
            if metadata is not None:
                raise ValueError("An active RLRB layer cannot change topology without recapture.")
            return

        packed = pack_placement(p2l.detach().cpu().tolist(), self.ep_size)
        if metadata is not None:
            if metadata.device != p2l.device:
                raise ValueError("An active RLRB layer cannot change device without recapture.")
            metadata.copy_(torch.tensor(packed, dtype=torch.int32, device="cpu"))
            return

        with torch.cuda.device(p2l.device):
            for size in BATCH_SIZES:
                load_kernel(self.ep_size, size)
        self._metadata[placement.layer_id] = torch.tensor(
            packed, dtype=torch.int32, device=p2l.device
        )

    def route_prepared_tokens(
        self,
        layer_id: int,
        logical_topk_ids: Tensor,
        *,
        token_count: Tensor | None = None,
        expert_load_view: Tensor | None = None,
        record_enabled: Tensor | None = None,
    ) -> Tensor | None:
        """Return physical IDs, or None when the caller should route natively."""
        metadata = self._metadata.get(layer_id)
        if (
            metadata is None
            or logical_topk_ids.ndim != 2
            or logical_topk_ids.shape[0] not in BATCH_SIZES
            or logical_topk_ids.shape[1] != TOP_K
            or logical_topk_ids.dtype != torch.int32
            or logical_topk_ids.device != metadata.device
            or not logical_topk_ids.is_contiguous()
        ):
            return None

        if (expert_load_view is None) != (record_enabled is None):
            raise ValueError("RLRB expert_load_view and record_enabled must be supplied together.")
        if record_enabled is not None and record_enabled.dtype not in (torch.bool, torch.int32):
            raise TypeError("RLRB record_enabled must have bool or int32 dtype.")

        empty = metadata[:0] if expert_load_view is None or token_count is None else None
        output = torch.empty_like(logical_topk_ids)
        load_kernel(self.ep_size, logical_topk_ids.shape[0]).route(
            logical_topk_ids,
            metadata,
            output,
            empty if expert_load_view is None else expert_load_view,
            empty if record_enabled is None else record_enabled.view(1),
            empty if token_count is None else token_count.view(1),
            self.source_rank,
        )
        return output

    def route_tokens(self, request: RoutingRequest, policy: RoutingPolicyConfig) -> RoutingDecision:
        placement = request.placement
        if placement is None:
            raise ValueError("RLRB requires a committed placement.")

        physical_ids = None
        if request.stage is RoutingStage.DECODE:
            physical_ids = self.route_prepared_tokens(
                request.layer_id,
                request.logical_topk_ids,
                token_count=request.token_count,
            )
        if physical_ids is None:
            physical_ids = route_dynamic_replicas(
                request.logical_topk_ids,
                placement.logical_to_physical_candidates,
                placement.logical_to_physical_count,
            )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


def _build_router(context: RouterBuildContext) -> RankLocalReplicaRouter:
    return RankLocalReplicaRouter(ep_size=context.ep_size, source_rank=context.source_rank)


ROUTING_POLICY = RoutingPolicySpec(
    "rlrb",
    RankLocalReplicaRouter.phase,
    requires_placement_state=True,
    requires_redundancy=True,
    fuses_replica_mapping_and_recording=True,
    incompatible_with=frozenset({"waterfill"}),
    build_router=_build_router,
)
