# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe import PlacementPlan, RoutingDecision, RoutingPolicyConfig
from evenmoe.adapters.sglang import (
    count_logical_experts,
    to_placement_request,
    to_placement_snapshot,
    to_routing_request,
    to_sglang_maps,
    to_sglang_routing_output,
)


def test_logical_count_ignores_dp_attention_padding():
    counts = count_logical_experts(
        torch.tensor([[0, 2, -1], [2, 3, -1]], dtype=torch.int32),
        4,
    )

    assert counts.dtype == torch.float32
    assert torch.equal(counts, torch.tensor([1.0, 0.0, 2.0, 1.0]))


class _Metadata:
    num_logical_experts = 2
    num_physical_experts = 4
    num_local_physical_experts = 2

    def __init__(self):
        self.physical_to_logical_map = torch.tensor([[0, 1, 0, 1]])
        self.logical_to_all_physical_map = torch.tensor([[[0, 2], [1, 3]]])
        self.logical_to_all_physical_map_num_valid = torch.tensor([[2, 2]])
        self.logical_to_rank_dispatch_physical_map = torch.tensor([[0, 1]])


@pytest.mark.unit
def test_l1_pre_and_post_adapters_are_plain_translations():
    logical_count = torch.tensor([[10, 20]])
    request = to_placement_request(
        logical_count,
        num_physical_experts=4,
        num_local_physical_experts=2,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )
    assert request.stats.logical_count is logical_count
    assert request.policy.name == "deepseek"
    assert request.policy.metadata == {}
    assert request.topology.num_physical_experts == 4
    assert request.topology.num_local_physical_experts == 2
    assert request.topology.num_groups == 1
    assert request.topology.num_nodes == 1

    plan = PlacementPlan(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]]),
        logical_to_physical_count=torch.tensor([[2, 2]]),
    )
    maps = to_sglang_maps(plan)
    assert maps.physical_to_logical_map is plan.physical_to_logical_map
    assert maps.logical_to_all_physical_map is plan.logical_to_all_physical_map
    assert maps.logical_to_physical_count is plan.logical_to_physical_count


@pytest.mark.unit
def test_snapshot_reads_live_committed_metadata_without_state():
    metadata = _Metadata()
    before = to_placement_snapshot(metadata, 0)
    assert before.ep_size == 2
    assert before.num_local_physical_experts == 2
    assert before.physical_to_rank_map is None

    metadata.logical_to_all_physical_map = torch.tensor([[[2, 0], [3, 1]]])
    after = to_placement_snapshot(metadata, 0)
    assert torch.equal(before.logical_to_physical_candidates, torch.tensor([[0, 2], [1, 3]]))
    assert torch.equal(after.logical_to_physical_candidates, torch.tensor([[2, 0], [3, 1]]))


@pytest.mark.unit
def test_routing_pre_adapter_only_packages_prepared_values():
    metadata = _Metadata()
    snapshot = to_placement_snapshot(metadata, 0)
    request = to_routing_request(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.tensor([[0.7, 0.3]]),
        policies=(
            RoutingPolicyConfig(name="lplb"),
            RoutingPolicyConfig(name="waterfill", metadata={"experts_per_rank": 2}),
        ),
        placement=snapshot,
        stage="PREFILL",
    )
    assert request.stage.value == "prefill"
    assert request.global_logical_count is None
    assert tuple(policy.name for policy in request.policies) == ("lplb", "waterfill")


@pytest.mark.unit
def test_post_adapter_keeps_recorder_ids_separate_from_shared_expert():
    physical_ids = torch.tensor([[0, 3]], dtype=torch.int64)
    decision = RoutingDecision(
        routed_physical_topk_ids=physical_ids,
        topk_weights=torch.tensor([[0.7, 0.3]]),
        shared_expert_rank=torch.tensor([1]),
    )

    output = to_sglang_routing_output(
        decision,
        router_logits=torch.tensor([[2.0, 1.0]]),
        num_routed_experts=4,
        world_size=2,
        routed_scaling_factor=1.0,
    )

    assert output.recorded_physical_topk_ids is physical_ids
    # old experts/rank=2, new experts/rank=3: physical 3 -> interleaved 4,
    # and rank-1 shared expert is slot 5.
    assert torch.equal(output.topk_ids, torch.tensor([[0, 4, 5]]))
    assert torch.allclose(output.topk_weights, torch.tensor([[0.7, 0.3, 1.0]]))
    assert torch.equal(output.router_logits, torch.tensor([[2.0, 1.0]]))


@pytest.mark.unit
def test_post_adapter_does_not_append_when_waterfill_did_not_run():
    ids = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.6, 0.4]])
    output = to_sglang_routing_output(
        RoutingDecision(
            routed_physical_topk_ids=ids,
            topk_weights=weights,
        )
    )
    assert output.topk_ids is ids
    assert output.topk_weights is weights
