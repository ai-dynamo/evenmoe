# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral L1 planner for the EPLB placement-policy family."""

from __future__ import annotations

import torch
from torch import Tensor

from evenmoe.core.l1 import PlacementPlanner
from evenmoe.core.types import PlacementPlan, PlacementPolicyConfig, PlacementRequest
from evenmoe.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    rebalance_experts,
)

SUPPORTED_PLACEMENT_POLICIES: tuple[str, ...] = (
    "auto",
    EplbAlgorithm.deepseek.name,
    EplbAlgorithm.deepseek_hierarchical.name,
    EplbAlgorithm.deepseek_vec.name,
    EplbAlgorithm.deepseek_vec_hierarchical.name,
    EplbAlgorithm.elasticity_aware.name,
    EplbAlgorithm.elasticity_aware_hierarchical.name,
)


class EPLBPlacementPlanner(PlacementPlanner):
    """Planner bound to one directly selectable L1 placement policy.

    The Orchestrator registry maps each implemented policy name (for example
    ``deepseek``) to a planner instance bound to that name. Runtime
    geometry comes from ``request.topology``; the policy name is never hidden
    inside metadata.
    """

    def __init__(self, policy_name: str):
        if policy_name not in SUPPORTED_PLACEMENT_POLICIES:
            raise ValueError(
                f"Unsupported placement policy {policy_name!r}; expected one of "
                f"{SUPPORTED_PLACEMENT_POLICIES}."
            )
        self._policy_name = policy_name

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.is_disabled:
            raise ValueError("EPLBPlacementPlanner cannot run a disabled policy.")
        if policy.name != self._policy_name:
            raise ValueError(f"Planner for {self._policy_name!r} received policy {policy.name!r}.")

        logical_count = request.stats.logical_count
        if logical_count is None:
            raise ValueError("EPLBPlacementPlanner requires request.stats.logical_count.")
        if not isinstance(logical_count, Tensor):
            logical_count = torch.tensor(logical_count)
        if logical_count.dim() == 2:
            # Promote [layers, logical] to [observations, layers, logical].
            logical_count = logical_count.unsqueeze(0)

        topology = request.topology
        algorithm = compute_algorithm(
            raw_algorithm=self._policy_name,
            num_groups=topology.num_groups,
            num_nodes=topology.num_nodes,
        )

        phy2log, log2allphy, expert_count = rebalance_experts(
            tokens_per_expert=logical_count,
            num_physical_experts=topology.num_physical_experts,
            num_local_physical_experts=topology.num_local_physical_experts,
            num_groups=topology.num_groups,
            num_nodes=topology.num_nodes,
            algorithm=algorithm,
            active_ranks=topology.active_ranks,
        )

        return PlacementPlan(
            physical_to_logical_map=phy2log,
            logical_to_all_physical_map=log2allphy,
            logical_to_physical_count=expert_count,
            metadata={
                "requested_policy": self._policy_name,
                "resolved_policy": algorithm.name,
            },
        )
