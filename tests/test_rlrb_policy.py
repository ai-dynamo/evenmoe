# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from evenmoe import PlacementSnapshot, RoutingPolicyConfig, RoutingRequest, RoutingStage
from evenmoe.kernels.rlrb import pack_placement
from evenmoe.policies.l2.rlrb import RankLocalReplicaRouter


def test_placement_keeps_cross_rank_replicas_separate_from_same_rank_copies():
    layout = list(range(128)) + list(range(32))
    layout[1], layout[128] = layout[128], layout[1]
    layout[159] = 0

    metadata = pack_placement(layout, ep_size=16)
    assert metadata[:3] == [0, 1, 159]
    slot_group = metadata[160:320]
    assert slot_group[0] == slot_group[1] != slot_group[159]
    group_rank = metadata[320:480]
    group_copies = metadata[640:800]
    assert (group_rank[slot_group[0]], group_copies[slot_group[0]]) == (0, 2)
    assert (group_rank[slot_group[159]], group_copies[slot_group[159]]) == (15, 1)


def test_unsupported_cpu_batch_routes_to_valid_replicas_without_changing_weights():
    layout = torch.tensor([0, 1, 0, 1])
    placement = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=layout,
        logical_to_physical_candidates=torch.tensor([[0, 2], [1, 3]]),
        logical_to_physical_count=torch.tensor([2, 2]),
    )
    router = RankLocalReplicaRouter(ep_size=2, source_rank=0)
    router.prepare_layer(placement)
    ids = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    decision = router.route_tokens(
        RoutingRequest(
            layer_id=0,
            logical_topk_ids=ids,
            topk_weights=weights,
            placement=placement,
            stage=RoutingStage.DECODE,
        ),
        RoutingPolicyConfig(name="rlrb"),
    )

    output = decision.routed_physical_topk_ids
    valid = ids >= 0
    torch.testing.assert_close(layout[output[valid]], ids[valid].long())
    assert output[1, 1].item() == -1
    assert decision.topk_weights is weights
