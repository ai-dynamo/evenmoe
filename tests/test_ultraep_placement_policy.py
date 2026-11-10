# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from evenmoe import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)
from evenmoe.kernels.ultraep.config import UltraEPPlacementTuning
from evenmoe.policies.l1.ultraep import UltraEPPlacementPolicy


class _Backend:
    def __init__(self):
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["physical_to_logical_map"].copy_(torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.int32))
        kwargs["logical_to_physical_map"].fill_(-1)
        kwargs["logical_to_physical_map"][:, 0].copy_(torch.tensor([0, 1, 3, 4], dtype=torch.int32))
        kwargs["logical_replica_counts"].fill_(1)
        kwargs["rank_quota_prefix"].zero_()


def _request():
    return PlacementRequest(
        stats=PlacementStats(
            logical_count=torch.tensor(
                [
                    [[4, 2, 6, 0], [2, 2, 0, 4]],
                    [[2, 0, 2, 0], [0, 2, 2, 0]],
                ],
                dtype=torch.float32,
            )
        ),
        topology=PlacementTopology(
            num_physical_experts=6,
            num_local_physical_experts=3,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(name="ultraep", metadata={"rank": 0, "num_nvl_ranks": 2}),
    )


@pytest.mark.unit
def test_ultraep_placement_policy_uses_existing_placement_plan_contract():
    backend = _Backend()
    plan = UltraEPPlacementPolicy(backend=backend).plan(_request())

    assert len(backend.calls) == 2
    assert backend.calls[0]["expert_loads"].tolist() == [6, 2, 8, 0]
    assert backend.calls[0]["expert_loads_per_rank"].tolist() == [
        [4, 2, 6, 0],
        [2, 0, 2, 0],
    ]
    assert backend.calls[0]["expert_loads"].dtype == torch.int32
    assert backend.calls[0]["expert_loads_per_rank"].dtype == torch.int32
    assert plan.physical_to_logical_map.shape == (2, 6)
    assert plan.logical_to_all_physical_map.shape == (2, 4, 2)
    assert plan.metadata["rank_quota_prefix"].shape == (2, 4, 2)


@pytest.mark.integration
def test_ultraep_placement_policy_handles_multiple_layers():
    """plan() slices per_rank_loads[:, layer_id] -- the middle dimension of a
    [ep_size, num_layers, num_logical_experts] tensor -- which is only
    non-contiguous once num_layers > 1. The mock-backend tests above never
    exercise this against the real kernel, which is the only thing that
    enforces contiguity; a mock happily accepts any stride.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")

    from evenmoe.kernels.ultraep import UltraEPPlacementBackend

    ep_size, local_masters, local_redundant = 8, 8, 2
    num_logical = ep_size * local_masters
    num_physical = ep_size * (local_masters + local_redundant)
    num_layers = 2
    device = torch.device("cuda")

    generator = torch.Generator().manual_seed(0)
    # The two layers get independently-shuffled loads so a layer-1 solve that
    # silently reused layer-0's (or garbage) memory would very likely produce
    # a different, and likely invalid, placement instead of coincidentally
    # matching this one.
    logical_count = torch.stack(
        [
            torch.randint(1, 50, (ep_size, num_logical), generator=generator)
            for _ in range(num_layers)
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)

    request = PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_physical,
            num_local_physical_experts=local_masters + local_redundant,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(
            name="ultraep", metadata={"rank": 0, "num_nvl_ranks": ep_size}
        ),
    )
    plan = UltraEPPlacementPolicy(backend=UltraEPPlacementBackend()).plan(request)

    assert plan.physical_to_logical_map.shape == (num_layers, num_physical)
    assert int(plan.physical_to_logical_map.min()) >= 0
    assert int(plan.physical_to_logical_map.max()) < num_logical
    for layer_id in range(num_layers):
        # Every logical expert must own at least one physical slot in every
        # layer, independently -- what the contiguous() fix protects.
        assert (
            len(torch.unique(plan.physical_to_logical_map[layer_id]))
            == num_logical
        )
    assert not torch.equal(
        plan.physical_to_logical_map[0], plan.physical_to_logical_map[1]
    )


@pytest.mark.unit
def test_constructing_the_policy_does_not_read_the_environment(monkeypatch):
    """A malformed ULTRA_EP_* value must not fail construction.

    The orchestrator registers every L1 policy by name up front, so this
    constructor runs for every MoELoadBalancer regardless of the selected
    algorithm.  Reading the environment here would let a stray variable break
    a deployment that never selects ultraep.
    """
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")  # only 0 or 1 are valid
    UltraEPPlacementPolicy(backend=_Backend())


@pytest.mark.unit
def test_env_is_still_validated_on_the_first_solve(monkeypatch):
    """Deferring the read must not discard the validation."""
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")
    policy = UltraEPPlacementPolicy(backend=_Backend())
    with pytest.raises(ValueError, match="ULTRA_EP_QUOTA_KERNEL_STAGE"):
        policy.plan(_request())


@pytest.mark.unit
def test_explicit_tuning_still_bypasses_the_environment(monkeypatch):
    """An explicitly supplied tuning must win over ULTRA_EP_* entirely."""
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")
    tuning = UltraEPPlacementTuning(
        balance_threshold=1.0,
        locality_aware=True,
        min_tokens_per_replica=1024,
        allow_zero_master_quota=False,
        oracle_eps=0.01,
        kernel_stage=1,
    )
    backend = _Backend()
    UltraEPPlacementPolicy(backend=backend, tuning=tuning).plan(_request())
    assert backend.calls[0]["tuning"] is tuning


def _topology(num_physical=12, num_local=6):
    return PlacementTopology(
        num_physical_experts=num_physical,
        num_local_physical_experts=num_local,
        num_groups=1,
        num_nodes=1,
    )


@pytest.mark.unit
def test_initial_plan_holds_masters_then_rank_local_redundant_slots():
    """Two ranks, eight experts, two redundant slots each: rank r holds masters
    4r..4r+3 and repeats its own first masters in its redundant slots."""
    plan = UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
        _topology(), num_logical_experts=8, num_layers=3
    )

    layout = [0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5]
    assert plan.physical_to_logical_map.shape == (3, 12)
    assert plan.physical_to_logical_map.dtype == torch.int32
    for layer in range(3):
        assert plan.physical_to_logical_map[layer].tolist() == layout
    assert plan.logical_to_physical_count[0].tolist() == [2, 2, 1, 1, 2, 2, 1, 1]
    assert plan.logical_to_all_physical_map[0].tolist() == [
        [0, 4],
        [1, 5],
        [2, -1],
        [3, -1],
        [6, 10],
        [7, 11],
        [8, -1],
        [9, -1],
    ]
    assert "rank_quota_prefix" not in plan.metadata


@pytest.mark.unit
def test_initial_plan_maps_agree_with_each_other():
    plan = UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
        _topology(num_physical=24, num_local=6), num_logical_experts=16, num_layers=1
    )
    physical_to_logical = plan.physical_to_logical_map[0]
    candidates = plan.logical_to_all_physical_map[0]
    counts = plan.logical_to_physical_count[0]
    # As wide as plan()'s: one column per EP rank.
    assert candidates.shape == (16, 4)
    for logical in range(16):
        replicas = candidates[logical][: counts[logical]].tolist()
        assert replicas == sorted(replicas)
        assert (candidates[logical][counts[logical] :] == -1).all()
        assert sorted(replicas) == (physical_to_logical == logical).nonzero().flatten().tolist()
        # The master comes first, on the rank that owns the expert.
        assert replicas[0] // 6 == logical // 4


@pytest.mark.unit
def test_solve_fills_declined_slots_with_the_initial_layout():
    """A slot the solve leaves unassigned takes the same expert the initial
    layout puts there, so the two rules cannot drift apart."""

    class _Declines(_Backend):
        def solve(self, **kwargs):
            super().solve(**kwargs)
            kwargs["physical_to_logical_map"].fill_(-1)

    request = _request()
    plan = UltraEPPlacementPolicy(backend=_Declines()).plan(request)
    initial = UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
        request.topology, num_logical_experts=4, num_layers=2
    )
    assert torch.equal(plan.physical_to_logical_map, initial.physical_to_logical_map)
    for name in ("logical_to_all_physical_map", "logical_to_physical_count"):
        assert getattr(initial, name).shape == getattr(plan, name).shape
        assert getattr(initial, name).dtype == getattr(plan, name).dtype


@pytest.mark.unit
def test_initial_plan_lists_at_most_ep_size_replicas():
    """One master per rank and three redundant slots: round-robin gives each
    expert four slots on a two-rank group. Like plan(), only ep_size of them
    are listed, master first; the others hold the expert unrouted."""
    plan = UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
        _topology(num_physical=8, num_local=4), num_logical_experts=2, num_layers=1
    )
    assert plan.physical_to_logical_map[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert plan.logical_to_all_physical_map[0].tolist() == [[0, 1], [4, 5]]
    assert plan.logical_to_physical_count[0].tolist() == [2, 2]


@pytest.mark.unit
@pytest.mark.parametrize(
    "topology, num_logical, message",
    [
        (_topology(num_physical=6, num_local=6), 4, "ep_size > 1"),
        (_topology(), 7, "divisible experts"),
        (_topology(num_physical=8, num_local=4), 8, "redundant"),
    ],
)
def test_initial_plan_rejects_layouts_ultraep_cannot_use(topology, num_logical, message):
    with pytest.raises(ValueError, match=message):
        UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
            topology, num_logical_experts=num_logical, num_layers=1
        )


@pytest.mark.unit
def test_load_balancer_initial_placement_comes_from_the_configured_policy():
    from evenmoe import MoELoadBalancer

    ultraep = MoELoadBalancer(routing_pipeline="ultraep")
    plan = ultraep.plan_initial_placement(_topology(), num_logical_experts=8, num_layers=1)
    assert plan is not None
    assert plan.physical_to_logical_map[0].tolist() == [0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5]

    # EPLB has no layout of its own, and no policy at all means no opinion.
    eplb = MoELoadBalancer(placement_policy="deepseek")
    assert eplb.plan_initial_placement(_topology(), num_logical_experts=8, num_layers=1) is None
    assert (
        MoELoadBalancer().plan_initial_placement(_topology(), num_logical_experts=8, num_layers=1)
        is None
    )


@pytest.mark.unit
def test_per_source_rank_statistics_follow_the_placement_policy():
    """Declared by the L1 policy, so it holds however the routing pipeline is
    configured, including vLLM's UltraEP placement with static routing."""
    from evenmoe import MoELoadBalancer

    assert MoELoadBalancer(routing_pipeline="ultraep").requires_per_source_rank_statistics()
    assert MoELoadBalancer(
        routing_pipeline="ultraep+waterfill"
    ).requires_per_source_rank_statistics()
    assert MoELoadBalancer(placement_policy="ultraep").requires_per_source_rank_statistics()
    assert not MoELoadBalancer(placement_policy="deepseek").requires_per_source_rank_statistics()
    assert not MoELoadBalancer().requires_per_source_rank_statistics()
    # A framework that picks its L1 policy separately can name it.
    assert MoELoadBalancer().requires_per_source_rank_statistics("ultraep")
    assert not MoELoadBalancer().requires_per_source_rank_statistics("deepseek")
    with pytest.raises(ValueError, match="Unknown placement policy"):
        MoELoadBalancer().requires_per_source_rank_statistics("nope")


@pytest.mark.unit
def test_initial_plan_routes_every_expert_to_its_master():
    """Committed as is, the quota-free initial plan sends each logical expert
    to its master slot through UltraEP's router."""
    from evenmoe import PlacementSnapshot, RoutingPolicyConfig, RoutingRequest
    from evenmoe.policies.l2.ultraep import UltraEPL2Router

    plan = UltraEPPlacementPolicy(backend=_Backend()).initial_plan(
        _topology(), num_logical_experts=8, num_layers=1
    )
    snapshot = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=8,
        num_physical_experts=12,
        ep_size=2,
        num_local_physical_experts=6,
        physical_to_logical_map=plan.physical_to_logical_map[0],
        logical_to_physical_candidates=plan.logical_to_all_physical_map[0],
        logical_to_physical_count=plan.logical_to_physical_count[0],
        metadata=plan.metadata,
    )
    logical_ids = torch.tensor([[0, 3, 4], [7, -1, 5]])
    decision = UltraEPL2Router(backend=_Backend()).route_tokens(
        RoutingRequest(
            layer_id=0,
            logical_topk_ids=logical_ids,
            topk_weights=torch.ones((2, 3)),
            placement=snapshot,
        ),
        RoutingPolicyConfig("ultraep"),
    )

    # Rank r's masters 4r..4r+3 sit in its first four of six slots.
    assert decision.routed_physical_topk_ids.tolist() == [[0, 3, 6], [9, -1, 7]]


class _Collectives:
    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        return payload + 1


@pytest.mark.unit
def test_aggregate_logical_count_follows_the_placement_policy():
    from evenmoe import MoELoadBalancer

    local = torch.tensor([[2, 0, 1, 0]], dtype=torch.int32)
    collectives = _Collectives(rank=1)
    mlb = MoELoadBalancer(routing_pipeline="ultraep", collectives=collectives)

    per_rank = mlb.aggregate_logical_count(local, ep_size=2)
    # One flat payload; this rank fills only its own row.
    assert collectives.payloads[-1].tolist() == [0, 0, 0, 0, 2, 0, 1, 0]
    assert per_rank.tolist() == [[[1, 1, 1, 1]], [[3, 1, 2, 1]]]

    summed = mlb.aggregate_logical_count(local, ep_size=2, placement_policy="deepseek")
    assert collectives.payloads[-1].tolist() == [2, 0, 1, 0]
    assert summed.tolist() == [[[3, 1, 2, 1]]]


@pytest.mark.unit
def test_aggregate_logical_count_checks_its_collectives():
    from evenmoe import MoELoadBalancer

    local = torch.tensor([[1, 2]], dtype=torch.int32)
    # A single rank needs no collective.
    assert MoELoadBalancer(routing_pipeline="ultraep").aggregate_logical_count(
        local, ep_size=1
    ).tolist() == [[[1, 2]]]
    with pytest.raises(RuntimeError, match="RoutingCollectives"):
        MoELoadBalancer(routing_pipeline="ultraep").aggregate_logical_count(local, ep_size=2)
    with pytest.raises(ValueError, match="world_size"):
        MoELoadBalancer(
            routing_pipeline="ultraep", collectives=_Collectives(world_size=4)
        ).aggregate_logical_count(local, ep_size=2)
    with pytest.raises(ValueError, match="layers, logical_experts"):
        MoELoadBalancer(routing_pipeline="ultraep").aggregate_logical_count(
            local[0], ep_size=1
        )
