# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evenmoe.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
    TensorLike,
)


@dataclass(frozen=True)
class VllmPlacementMaps:
    """Plain candidate maps returned to vLLM Glue.

    vLLM only consumes ``physical_to_logical_map``; it derives its own
    ``logical_to_physical_map`` / ``logical_replica_count`` from that map via
    ``vllm.distributed.eplb.eplb_state.compute_logical_maps``.  The other two
    fields are carried so that Glue can assert agreement during bring-up.
    """

    physical_to_logical_map: TensorLike
    logical_to_all_physical_map: TensorLike
    logical_to_physical_count: TensorLike


def to_placement_request(
    logical_count: TensorLike,
    *,
    num_replicas: int,
    num_ranks: int,
    num_groups: int | None,
    num_nodes: int,
    algorithm: str = "auto",
    old_physical_to_logical_map: TensorLike | None = None,
    active_ranks: TensorLike | None = None,
    ep_rank: int | None = None,
) -> PlacementRequest:
    """Translate vLLM's ``AbstractEplbPolicy.rebalance_experts`` arguments.

    vLLM completes recording, physical-to-logical aggregation, window
    summation and the EP all-reduce before invoking its placement policy, so
    ``logical_count`` arrives already in the shape EvenMoE wants:
    ``[num_moe_layers, num_logical_experts]`` -- except for ``ultraep``, whose
    placement kernel solves once per rank and so needs the pre-reduction
    per-rank breakdown instead:
    ``[num_ranks, num_moe_layers, num_logical_experts]``.  The caller selects
    which of the two to pass by policy name; EvenMoE declares no capability for
    this today.

    ``num_replicas`` is vLLM's name for the total physical expert slot count;
    ``num_ranks`` is the EP world size.  EvenMoE's topology is expressed in terms
    of slots per rank, so the two are combined here rather than in Glue.

    ``old_physical_to_logical_map`` is vLLM's ``old_global_expert_indices``.
    No EvenMoE L1 policy consumes a previous placement today, so it is carried in
    ``metadata`` rather than being converted into ``PlacementRequest.current``:
    building per-layer ``PlacementSnapshot`` objects that nothing reads would
    cost real time on a 58-layer model.  Glue is responsible for slot
    preservation until an EvenMoE policy claims this input.

    ``ep_rank`` is the caller's own EP rank -- vLLM runs one process per rank,
    so this call happens independently on each of them.  Only ``ultraep``
    reads it (as ``policy.metadata["rank"]``: its placement kernel solves
    once per rank rather than once globally, and rejects an out-of-range
    value).  Every other EvenMoE policy ignores it, so it is safe to leave unset.
    """

    if num_ranks <= 0:
        raise ValueError("vLLM placement requires a positive num_ranks.")
    if num_replicas % num_ranks != 0:
        raise ValueError(
            "vLLM placement requires num_replicas to be divisible by num_ranks "
            f"(got num_replicas={num_replicas}, num_ranks={num_ranks})."
        )

    metadata: dict[str, Any] = {"framework": "vllm"}
    if old_physical_to_logical_map is not None:
        metadata["old_physical_to_logical_map"] = old_physical_to_logical_map

    policy_metadata: dict[str, Any] = {}
    if ep_rank is not None:
        policy_metadata["rank"] = ep_rank

    return PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_replicas,
            num_local_physical_experts=num_replicas // num_ranks,
            num_groups=num_groups,
            num_nodes=num_nodes,
            active_ranks=active_ranks,
        ),
        policy=PlacementPolicyConfig(name=algorithm, metadata=policy_metadata),
        metadata=metadata,
    )


def to_vllm_maps(plan: PlacementPlan) -> VllmPlacementMaps:
    """Translate an uncommitted EvenMoE plan into plain vLLM mapping tensors."""

    return VllmPlacementMaps(
        physical_to_logical_map=plan.physical_to_logical_map,
        logical_to_all_physical_map=plan.logical_to_all_physical_map,
        logical_to_physical_count=plan.logical_to_physical_count,
    )


def to_vllm_physical_to_logical(plan: PlacementPlan) -> TensorLike:
    """Return the single tensor vLLM's placement policy contract expects."""

    return plan.physical_to_logical_map
