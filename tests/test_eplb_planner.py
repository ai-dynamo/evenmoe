# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe import (
    MoELoadBalancer,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)
from evenmoe.adapters.sglang import to_placement_request
from evenmoe.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    rebalance_experts,
)
from evenmoe.policies.l1.eplb.planner import SUPPORTED_PLACEMENT_POLICIES


@pytest.mark.unit
def test_orchestrator_eplb_matches_raw_deepseek_algorithm():
    logical_count = torch.tensor([[20, 3, 9, 4], [5, 16, 2, 7]])
    request = to_placement_request(
        logical_count,
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )

    plan = MoELoadBalancer().plan_placement(request)
    expected = rebalance_experts(
        tokens_per_expert=logical_count.unsqueeze(0),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm=EplbAlgorithm.deepseek,
    )

    assert torch.equal(plan.physical_to_logical_map, expected[0])
    assert torch.equal(plan.logical_to_all_physical_map, expected[1])
    assert torch.equal(plan.logical_to_physical_count, expected[2])
    assert not hasattr(plan, "snapshots")


@pytest.mark.unit
def test_cold_start_and_periodic_paths_use_same_request_and_policy_entry():
    logical_count = torch.tensor([[10, 7, 4, 1]])
    kwargs = {
        "num_physical_experts": 8,
        "num_local_physical_experts": 4,
        "num_groups": 1,
        "num_nodes": 1,
        "algorithm": "deepseek",
    }
    mlb = MoELoadBalancer()

    cold_start = mlb.plan_placement(to_placement_request(logical_count, **kwargs))
    periodic = mlb.plan_placement(to_placement_request(logical_count, **kwargs))

    assert torch.equal(cold_start.physical_to_logical_map, periodic.physical_to_logical_map)
    assert torch.equal(
        cold_start.logical_to_all_physical_map,
        periodic.logical_to_all_physical_map,
    )


@pytest.mark.unit
def test_framework_must_not_call_disabled_l1_policy():
    request = PlacementRequest(
        stats=PlacementStats(logical_count=torch.ones((1, 2))),
        topology=PlacementTopology(
            num_physical_experts=2,
            num_local_physical_experts=1,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(),
    )
    with pytest.raises(ValueError, match="framework must skip"):
        MoELoadBalancer().plan_placement(request)


@pytest.mark.unit
def test_l1_policy_name_directly_selects_algorithm():
    request = to_placement_request(
        torch.tensor([[10, 7, 4, 1]]),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )

    assert request.policy.name == "deepseek"
    assert "algorithm" not in request.policy.metadata
    assert request.topology.num_physical_experts == 8
    plan = MoELoadBalancer().plan_placement(request)
    assert plan.metadata["requested_policy"] == "deepseek"
    assert plan.metadata["resolved_policy"] == "deepseek"


@pytest.mark.unit
def test_auto_is_a_policy_and_reports_its_resolved_algorithm():
    request = to_placement_request(
        torch.tensor([[10, 7, 4, 1]]),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="auto",
    )

    plan = MoELoadBalancer().plan_placement(request)

    assert request.policy.name == "auto"
    assert plan.metadata["requested_policy"] == "auto"
    assert plan.metadata["resolved_policy"] == "deepseek_hierarchical"


@pytest.mark.unit
def test_default_registry_advertises_all_migrated_eplb_policies():
    assert SUPPORTED_PLACEMENT_POLICIES == (
        "auto",
        "deepseek",
        "deepseek_hierarchical",
        "deepseek_vec",
        "deepseek_vec_hierarchical",
        "elasticity_aware",
        "elasticity_aware_hierarchical",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "algorithm",
    [
        "deepseek",
        "deepseek_hierarchical",
        "deepseek_vec",
        "deepseek_vec_hierarchical",
        "elasticity_aware",
        "elasticity_aware_hierarchical",
    ],
)
def test_all_migrated_eplb_policies_run_through_default_registry(algorithm):
    active_ranks = torch.tensor([True, True]) if algorithm.startswith("elasticity_aware") else None
    request = to_placement_request(
        torch.tensor(
            [
                [[20, 3, 9, 4], [5, 16, 2, 7]],
                [[18, 4, 11, 2], [6, 12, 3, 8]],
            ]
        ),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm=algorithm,
        active_ranks=active_ranks,
    )
    plan = MoELoadBalancer().plan_placement(request)

    assert plan.physical_to_logical_map.shape == (2, 8)
    assert plan.logical_to_all_physical_map.shape[:2] == (2, 4)
    assert plan.logical_to_physical_count.shape == (2, 4)
    assert torch.all(plan.logical_to_physical_count >= 1)
    assert plan.metadata["resolved_policy"] == algorithm
