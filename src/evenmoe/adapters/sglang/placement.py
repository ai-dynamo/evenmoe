# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evenmoe.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
    PlacementStats,
    PlacementTopology,
    TensorLike,
)

if TYPE_CHECKING:
    from sglang.srt.runtime_context import RuntimeContext

    from evenmoe.core.orchestrator import MoELoadBalancer


def _expert_layout(context: RuntimeContext):
    # Model facts are available before committed placement metadata exists.
    model_info = context.resources.mlb_model_info
    if model_info is None:
        raise ValueError("EvenMoE routing requires model expert information in context.")
    moe = context.config_bag("exec").moe
    parallel = context.parallel
    base_num_physical_experts = (
        model_info["num_logical_experts"] + moe.ep_num_redundant_experts
    )
    ep_size = parallel.ep_size
    initial_ep_size = parallel.elastic_ep_initial_size
    if initial_ep_size is not None:
        if moe.ep_join_mode == "scale":
            ep_size = max(ep_size, parallel.ep_join_rank_offset + parallel.tp_size)
        assert base_num_physical_experts % initial_ep_size == 0
        num_local_physical_experts = base_num_physical_experts // initial_ep_size
        num_physical_experts = num_local_physical_experts * ep_size
    else:
        assert base_num_physical_experts % ep_size == 0
        num_physical_experts = base_num_physical_experts
        num_local_physical_experts = num_physical_experts // ep_size
    return {
        "ep_size": ep_size,
        "num_physical_experts": num_physical_experts,
        "num_local_physical_experts": num_local_physical_experts,
        "num_groups": model_info["num_groups"],
    }


def _placement_policy_name(context: RuntimeContext) -> str:
    """The L1 policy SGLang runs: the one the routing pipeline requires
    (UltraEP's own placement), otherwise ``--eplb-algorithm``."""
    from evenmoe.core.routing_pipeline import RoutingPipeline

    moe = context.config_bag("exec").moe
    pipeline = RoutingPipeline.from_value(moe.moe_load_balancer_algorithm)
    return pipeline.placement_policy or moe.eplb_algorithm


def _placement(context: RuntimeContext):
    metadata = context.resources.expert_location_metadata
    if metadata is None:
        raise RuntimeError("EvenMoE routing requires committed expert metadata.")
    return metadata


def commit_placement(
    mlb: MoELoadBalancer, context: RuntimeContext, layer_ids: Iterable[int]
) -> None:
    """Notify EvenMoE after SGLang has committed the placement and weights."""
    if not mlb.routing_capabilities.requires_placement_state:
        return
    metadata = _placement(context)
    for layer_id in layer_ids:
        mlb.on_placement_committed(to_placement_snapshot(metadata, layer_id))


@dataclass(frozen=True)
class SGLangPlacementMaps:
    """Plain candidate maps returned to SGLang Glue.

    The map tensors and ``routing_metadata`` are indexed by the layers of the
    plan they came from, which are model layer ids for a whole-model plan.
    ``to_sglang_maps(plan, layer_id=...)`` re-keys a single-layer plan's
    ``routing_metadata`` to the model layer id.
    """

    physical_to_logical_map: TensorLike
    logical_to_all_physical_map: TensorLike
    logical_to_physical_count: TensorLike
    routing_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)


def to_placement_request(
    logical_count: TensorLike | None = None,
    *,
    context: RuntimeContext | None = None,
    num_physical_experts: int | None = None,
    num_local_physical_experts: int | None = None,
    num_groups: int | None = None,
    num_nodes: int | None = None,
    algorithm: str | None = None,
    active_ranks: TensorLike | None = None,
    policy_metadata: dict[str, Any] | None = None,
) -> PlacementRequest:
    """Translate SGLang-completed EPLB statistics into an EvenMoE request.

    SGLang must perform recording, dump, physical-to-logical aggregation,
    EP aggregation and threshold gating before calling this function. A policy
    whose ``mlb.placement_logical_count()`` is ``"per_source_rank"`` needs one
    logical-count sample per source EP rank.

    For a per-layer refresh, omit ``logical_count`` and hand the one-layer
    request to :meth:`MoELoadBalancer.plan_layer_placement`, which combines
    this rank's counts over the EP group and supplies the statistics.

    Context supplies omitted layout and policy values. Explicit arguments,
    such as ``num_nodes=1`` for flat topology, take precedence.

    ``policy_metadata`` is opaque, per-policy configuration. ``ultraep``
    reads ``rank`` to build the current source rank's quota and
    ``num_nvl_ranks`` to describe its locality domain. Other policies leave
    it unset.
    """

    if context is not None:
        layout = _expert_layout(context)
        if num_physical_experts is None:
            num_physical_experts = layout["num_physical_experts"]
        if num_local_physical_experts is None:
            num_local_physical_experts = layout["num_local_physical_experts"]
        if num_groups is None:
            num_groups = layout["num_groups"]
        if num_nodes is None:
            num_nodes = context.parallel.nnodes
        if algorithm is None:
            algorithm = _placement_policy_name(context)
        if policy_metadata is None and algorithm == "ultraep":
            if layout["ep_size"] % context.parallel.nnodes != 0:
                raise ValueError(
                    "UltraEP requires the EP group to be evenly distributed "
                    "across nodes."
                )
            policy_metadata = {
                "rank": context.parallel.moe_ep_rank,
                "num_nvl_ranks": layout["ep_size"] // context.parallel.nnodes,
            }
    if num_physical_experts is None or num_local_physical_experts is None or num_nodes is None:
        raise ValueError("Placement requires a context or explicit expert and node counts.")

    # A planner that decides per layer whether to move (``adaptive``) needs the
    # placement SGLang has committed. None at startup, before the first commit.
    current = None
    if context is not None and algorithm == "adaptive":
        live = getattr(context.resources, "expert_location_metadata", None)
        if live is not None:
            current = _committed_placement(live)

    return PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            active_ranks=active_ranks,
        ),
        policy=PlacementPolicyConfig(
            name="auto" if algorithm is None else algorithm, metadata=policy_metadata or {}
        ),
        current=current,
    )


def _committed_placement(live_metadata: Any) -> dict[int, PlacementSnapshot]:
    """SGLang's committed placement per layer, read from its host copies.

    The adaptive planner works on the host, so host copies keep a periodic
    check from synchronizing with the GPU. Layers whose routing metadata
    carries its own candidates fall back to the live snapshot.
    """

    import torch

    num_layers = int(live_metadata.physical_to_logical_map.shape[0])
    p2l = getattr(live_metadata, "physical_to_logical_map_cpu", None)
    l2p = getattr(live_metadata, "logical_to_all_physical_map_cpu", None)
    routing = getattr(live_metadata, "mlb_routing_metadata", None) or {}
    if p2l is None or l2p is None:
        return {layer: to_placement_snapshot(live_metadata, layer) for layer in range(num_layers)}
    num_physical = int(live_metadata.num_physical_experts)
    num_local = int(live_metadata.num_local_physical_experts)
    num_logical = int(live_metadata.num_logical_experts)
    p2l = p2l.to(torch.int64)
    count = torch.zeros((num_layers, num_logical), dtype=torch.int64).scatter_add_(
        1, p2l, torch.ones_like(p2l)
    )
    # SGLang pads the candidate table to the physical width; keep the used part.
    l2p = l2p[..., : int(count.max())]
    return {
        layer: (
            to_placement_snapshot(live_metadata, layer)
            if "logical_to_physical_candidates" in routing.get(layer, {})
            else PlacementSnapshot(
                layer_id=layer,
                num_logical_experts=num_logical,
                num_physical_experts=num_physical,
                ep_size=num_physical // num_local,
                num_local_physical_experts=num_local,
                physical_to_logical_map=p2l[layer],
                logical_to_physical_candidates=l2p[layer],
                logical_to_physical_count=count[layer],
            )
        )
        for layer in range(num_layers)
    }


def to_sglang_maps(plan: PlacementPlan, *, layer_id: int | None = None) -> SGLangPlacementMaps:
    """Translate an uncommitted EvenMoE plan into plain SGLang mapping tensors.

    ``layer_id`` marks a single-layer plan, such as
    :meth:`MoELoadBalancer.plan_layer_placement` returns, as that model layer:
    the map tensors keep their leading dimension of 1, while
    ``routing_metadata`` is keyed by ``layer_id`` so it can be published as is.
    """

    if layer_id is not None and plan.physical_to_logical_map.shape[0] != 1:
        raise ValueError("layer_id applies only to a single-layer plan.")
    quota = plan.metadata.get("rank_quota_prefix")
    routing_metadata = (
        {
            layer if layer_id is None else layer_id: {
                "logical_to_physical_candidates": plan.logical_to_all_physical_map[layer],
                "logical_to_physical_count": plan.logical_to_physical_count[layer],
                "rank_quota_prefix": quota[layer],
            }
            for layer in range(quota.shape[0])
        }
        if quota is not None
        else {}
    )
    return SGLangPlacementMaps(
        physical_to_logical_map=plan.physical_to_logical_map,
        logical_to_all_physical_map=plan.logical_to_all_physical_map,
        logical_to_physical_count=plan.logical_to_physical_count,
        routing_metadata=routing_metadata,
    )


def plan_initial_placement(
    mlb: MoELoadBalancer, context: RuntimeContext, *, num_layers: int
) -> SGLangPlacementMaps | None:
    """The configured placement policy's layout for SGLang's startup placement.

    None means the policy has no layout of its own, and SGLang keeps its default
    initial placement. SGLang still loads the weights and commits the maps.
    """

    layout = _expert_layout(context)
    plan = mlb.plan_initial_placement(
        PlacementTopology(
            num_physical_experts=layout["num_physical_experts"],
            num_local_physical_experts=layout["num_local_physical_experts"],
            num_groups=layout["num_groups"],
            num_nodes=context.parallel.nnodes,
        ),
        num_logical_experts=context.resources.mlb_model_info["num_logical_experts"],
        num_layers=num_layers,
    )
    return None if plan is None else to_sglang_maps(plan)


def to_placement_snapshot(
    live_metadata: Any,
    layer_id: int,
) -> PlacementSnapshot:
    """Translate SGLang's current committed placement into an L2 snapshot.

    The function stores nothing and returns tensor views from ``live_metadata``.
    Calling it again after SGLang commits a rebalance observes the new mapping.
    """

    if live_metadata is None:
        raise ValueError("live_metadata must be SGLang's committed placement metadata.")
    num_physical = int(live_metadata.num_physical_experts)
    num_local = int(live_metadata.num_local_physical_experts)
    if num_physical % num_local != 0:
        raise ValueError(
            "SGLang placement requires num_physical_experts to be divisible by "
            "num_local_physical_experts."
        )
    ep_size = num_physical // num_local

    rank_dispatch = getattr(
        live_metadata,
        "logical_to_rank_dispatch_physical_map",
        None,
    )
    explicit_rank_map = getattr(live_metadata, "physical_to_rank_map", None)
    if explicit_rank_map is not None and getattr(explicit_rank_map, "ndim", 1) > 1:
        explicit_rank_map = explicit_rank_map[layer_id]
    routing_metadata = getattr(live_metadata, "mlb_routing_metadata", {}).get(
        layer_id, {}
    )

    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=int(live_metadata.num_logical_experts),
        num_physical_experts=num_physical,
        ep_size=ep_size,
        num_local_physical_experts=num_local,
        physical_to_logical_map=live_metadata.physical_to_logical_map[layer_id],
        logical_to_physical_candidates=routing_metadata.get(
            "logical_to_physical_candidates",
            live_metadata.logical_to_all_physical_map[layer_id],
        ),
        logical_to_physical_count=routing_metadata.get(
            "logical_to_physical_count",
            live_metadata.logical_to_all_physical_map_num_valid[layer_id],
        ),
        default_physical_for_logical=(
            rank_dispatch[layer_id] if rank_dispatch is not None else None
        ),
        physical_to_rank_map=explicit_rank_map,
        metadata=routing_metadata,
    )
