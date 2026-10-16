# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from evenmoe import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.policies.l2.ultraep import UltraEPL2Router


class _Backend:
    def __init__(self):
        self.calls = []

    def reroute_sparse(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["routed_topk_ids"].copy_(kwargs["logical_topk_ids"] + 10)


def _placement(*, quota=True):
    candidates = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=torch.tensor([2, 2], dtype=torch.int32),
        metadata=(
            {"rank_quota_prefix": torch.tensor([[1, 2], [1, 2]], dtype=torch.int32)}
            if quota
            else {}
        ),
    )


def _request(placement, ids=None):
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=(torch.tensor([[0, 1]], dtype=torch.int32) if ids is None else ids),
        topk_weights=torch.tensor([[0.6, 0.4]]),
        policies=("ultraep",),
        placement=placement,
    )


@pytest.mark.unit
def test_ultraep_l2_uses_committed_snapshot():
    backend = _Backend()
    router = UltraEPL2Router(backend=backend)
    placement = _placement()
    decision = router.route_tokens(_request(placement), RoutingPolicyConfig(name="ultraep"))

    assert router.phase is RoutingPhase.PHYSICAL_ROUTING
    assert decision.routed_physical_topk_ids.tolist() == [[10, 11]]
    assert backend.calls[0]["logical_to_physical_map"] is (placement.logical_to_physical_candidates)
    assert backend.calls[0]["rank_quota_prefix"] is placement.metadata["rank_quota_prefix"]


@pytest.mark.unit
def test_ultraep_l2_prepared_fast_path():
    backend = _Backend()
    router = UltraEPL2Router(backend=backend)
    load_balancer = MoELoadBalancer(
        placement_planners={},
        routing_policies={"ultraep": router},
    )
    placement = _placement()

    load_balancer.on_placement_committed(placement)
    physical_ids = load_balancer.route_prepared_tokens(
        "ultraep",
        placement.layer_id,
        torch.tensor([[0, 1]], dtype=torch.int32),
    )

    assert physical_ids.tolist() == [[10, 11]]
    assert backend.calls[0]["logical_to_physical_map"] is (
        placement.logical_to_physical_candidates
    )


@pytest.mark.unit
def test_ultraep_l2_bootstrap_routes_to_fixed_masters():
    backend = _Backend()
    router = UltraEPL2Router(backend=backend)
    decision = router.route_tokens(
        _request(_placement(quota=False), torch.tensor([[0, 1, -1]])),
        RoutingPolicyConfig(name="ultraep"),
    )

    assert decision.routed_physical_topk_ids.tolist() == [[0, 1, -1]]
    assert backend.calls == []


@pytest.mark.unit
def test_ultraep_l2_requires_placement():
    router = UltraEPL2Router(backend=_Backend())
    with pytest.raises(ValueError, match="committed placement"):
        router.route_tokens(_request(None), RoutingPolicyConfig(name="ultraep"))


@pytest.mark.unit
def test_ultraep_is_built_by_the_unified_algorithm_factory():
    mlb = MoELoadBalancer.from_algorithm(
        "ultraep",
        ep_size=2,
        source_rank=0,
        experts_per_rank=2,
    )

    decision = mlb.route_tokens(_request(_placement(quota=False)))

    assert mlb.placement_policy == "ultraep"
    assert decision.routed_physical_topk_ids.tolist() == [[0, 1]]
