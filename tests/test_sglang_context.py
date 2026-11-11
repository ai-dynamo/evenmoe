# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from evenmoe import MoELoadBalancer, PlacementPlan, RoutingDecision, RoutingPipeline
from evenmoe.adapters.sglang import (
    placement,
    plan_initial_placement,
    plan_layer_from_topk,
    to_deployment_config,
    to_load_balancer_kwargs,
    to_placement_request,
    to_placement_snapshot,
    to_routing_request,
    to_sglang_routing_output,
)

pytestmark = pytest.mark.unit


def metadata(mapping=(0, 1)):
    return SimpleNamespace(
        ep_size=2,
        num_logical_experts=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]]),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2]]),
        logical_to_rank_dispatch_physical_map=torch.tensor([mapping]),
    )


@pytest.fixture
def engine():
    config = SimpleNamespace(
        moe=SimpleNamespace(
            moe_load_balancer_algorithm="static",
            eplb_algorithm="deepseek",
            ep_num_redundant_experts=2,
            ep_join_mode=None,
        )
    )
    ctx = SimpleNamespace(
        config_bag=lambda name: config,
        parallel=SimpleNamespace(
            moe_ep_rank=0,
            nnodes=2,
            moe_ep_group=Mock(),
            ep_size=2,
            elastic_ep_initial_size=None,
            ep_join_rank_offset=0,
            tp_size=2,
        ),
        resources=SimpleNamespace(
            mlb_model_info={"num_logical_experts": 2, "num_groups": 1},
            expert_location_metadata=metadata(),
            expert_distribution_recorder=Mock(),
            experts_capturer=Mock(),
        ),
    )
    mlb = MoELoadBalancer.from_algorithm(**to_load_balancer_kwargs(ctx))
    assert type(mlb) is MoELoadBalancer
    return mlb, ctx


@pytest.mark.parametrize(
    "ep_size, initial_ep_size, join_mode, offset, expected",
    [
        (2, None, None, 0, (2, 4, 2)),
        (4, 2, None, 0, (4, 8, 2)),
        (2, 2, "scale", 4, (6, 12, 2)),
        (8, 2, "scale", 4, (8, 16, 2)),
        (4, 4, None, 0, (4, 4, 1)),
    ],
)
def test_layout_from_context(engine, ep_size, initial_ep_size, join_mode, offset, expected):
    _, ctx = engine
    ctx.parallel.ep_size = ep_size
    ctx.parallel.elastic_ep_initial_size = initial_ep_size
    ctx.parallel.ep_join_rank_offset = offset
    ctx.config_bag("exec").moe.ep_join_mode = join_mode
    layout = placement._expert_layout(ctx)
    assert (
        layout["ep_size"],
        layout["num_physical_experts"],
        layout["num_local_physical_experts"],
    ) == expected
    assert layout["num_groups"] == 1


def test_bootstrap_and_planning_do_not_import_sglang(engine, monkeypatch):
    import builtins

    _, ctx = engine
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sglang" or name.startswith("sglang."):
            raise AssertionError(f"Unexpected SGLang import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ctx.resources.expert_location_metadata = None
    kwargs = to_load_balancer_kwargs(ctx)
    assert kwargs["ep_size"] == 2 and kwargs["experts_per_rank"] == 2
    request = to_placement_request(torch.ones((1, 2)), context=ctx)
    assert request.topology.num_groups == 1


def test_layout_requires_published_model_info(engine):
    _, ctx = engine
    ctx.resources.mlb_model_info = None
    with pytest.raises(ValueError, match="model expert information"):
        to_load_balancer_kwargs(ctx)


def test_layout_rejects_indivisible_experts(engine):
    _, ctx = engine
    ctx.resources.mlb_model_info["num_logical_experts"] = 3
    with pytest.raises(AssertionError):
        placement._expert_layout(ctx)
    ctx.parallel.elastic_ep_initial_size = 2
    with pytest.raises(AssertionError):
        placement._expert_layout(ctx)


def route(engine, ids=None):
    mlb, ctx = engine
    ids = torch.tensor([[0, 1]]) if ids is None else ids
    request = to_routing_request(
        context=ctx,
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones_like(ids, dtype=torch.float32),
        token_count=len(ids),
        stage="prefill",
        routed_scaling_factor=1.0,
    )
    decision = mlb.route_tokens(request)
    return to_sglang_routing_output(decision, context=ctx)


def test_routing_reads_in_place_changes_and_replaced_metadata(engine):
    _mlb, ctx = engine
    assert route(engine).topk_ids.tolist() == [[0, 1]]
    ctx.resources.expert_location_metadata.logical_to_rank_dispatch_physical_map.copy_(
        torch.tensor([[2, 3]])
    )
    assert route(engine).topk_ids.tolist() == [[2, 3]]
    ctx.resources.expert_location_metadata = metadata((0, 3))
    assert route(engine).topk_ids.tolist() == [[0, 3]]
    ctx.resources = SimpleNamespace(expert_location_metadata=metadata((2, 1)))
    assert route(engine).topk_ids.tolist() == [[2, 1]]


def test_empty_routing_still_reaches_pipeline(engine):
    mlb, ctx = engine
    mlb.route_tokens = Mock(wraps=mlb.route_tokens)
    output = route(engine, torch.empty((0, 2), dtype=torch.int64))
    assert output.topk_ids.shape == (0, 2)
    mlb.route_tokens.assert_called_once()
    ctx.resources.expert_distribution_recorder.on_select_experts.assert_not_called()


def test_missing_committed_metadata_fails_at_routing_not_construction(engine):
    _mlb, ctx = engine
    ctx.resources.expert_location_metadata = None
    new_mlb = MoELoadBalancer.from_algorithm(**to_load_balancer_kwargs(ctx))
    with pytest.raises(RuntimeError, match="committed expert metadata"):
        route((new_mlb, ctx))


def test_snapshot_reads_replaced_metadata_without_copying(engine):
    _, ctx = engine
    ctx.resources.expert_location_metadata = metadata((2, 3))
    snapshot = to_placement_snapshot(ctx.resources.expert_location_metadata, 0)
    assert snapshot.default_physical_for_logical.tolist() == [2, 3]
    assert (
        snapshot.physical_to_logical_map.data_ptr()
        == ctx.resources.expert_location_metadata.physical_to_logical_map.data_ptr()
    )


def test_collective_transport_reads_current_group(engine):
    _mlb, ctx = engine
    transport = to_load_balancer_kwargs(ctx)["collectives"]
    replacement = Mock(rank_in_group=1, world_size=2)
    ctx.parallel.moe_ep_group = replacement
    payload = torch.ones(2)
    assert transport.rank == 1 and transport.world_size == 2
    assert transport.all_reduce_sum(payload) is replacement.all_reduce.return_value
    replacement.all_reduce.assert_called_once_with(payload)


def test_collective_transport_gathers_into_one_flat_tensor(engine):
    _mlb, ctx = engine
    transport = to_load_balancer_kwargs(ctx)["collectives"]
    group = Mock(rank_in_group=1, world_size=2)
    ctx.parallel.moe_ep_group = group
    payload = torch.tensor([3, 1, 4], dtype=torch.int32)
    gathered = transport.all_gather(payload)
    output, sent = group.all_gather_into_tensor.call_args.args
    assert output is gathered and sent is payload
    assert gathered.shape == (6,) and gathered.dtype == torch.int32


def test_planning_reads_context_and_keeps_per_call_inputs_explicit(engine):
    _mlb, ctx = engine
    counts = torch.tensor([[[10, 20]]])
    ctx.config_bag("exec").moe.eplb_algorithm = "deepseek_vec"
    active = torch.tensor([True, True])
    request = to_placement_request(counts, context=ctx, active_ranks=active, num_nodes=1)
    assert request.stats.logical_count is counts
    assert request.topology.active_ranks is active
    assert request.topology.num_nodes == 1
    assert request.policy.name == "deepseek_vec"
    assert request.topology.num_physical_experts == 4
    assert request.topology.num_local_physical_experts == 2
    ctx.parallel.nnodes = 3
    assert to_placement_request(counts, context=ctx).topology.num_nodes == 3


def test_two_contexts_do_not_share_placement_or_mutate_topk(engine):
    mlb, first = engine
    second = SimpleNamespace(resources=SimpleNamespace(expert_location_metadata=metadata((2, 3))))
    ids = torch.tensor([[0, 1]])
    assert route((mlb, first), ids).topk_ids.tolist() == [[0, 1]]
    assert route((mlb, second), ids).topk_ids.tolist() == [[2, 3]]
    assert route((mlb, first), ids).topk_ids.tolist() == [[0, 1]]
    assert ids.tolist() == [[0, 1]]


def test_shared_output_reads_current_context_geometry(engine):
    _, ctx = engine
    ids = torch.tensor([[0, 3]])
    decision = RoutingDecision(
        routed_physical_topk_ids=ids,
        topk_weights=torch.tensor([[0.7, 0.3]]),
        shared_expert_rank=torch.tensor([1]),
    )
    output = to_sglang_routing_output(decision, context=ctx)
    assert output.topk_ids.tolist() == [[0, 4, 5]]
    assert output.recorded_physical_topk_ids is ids

    ctx.resources.expert_location_metadata = SimpleNamespace(num_physical_experts=6, ep_size=2)
    output = to_sglang_routing_output(decision, context=ctx)
    assert output.topk_ids.tolist() == [[0, 4, 7]]
    assert output.recorded_physical_topk_ids is ids


def test_explicit_placement_values_override_context_defaults(engine):
    _, ctx = engine
    request = to_placement_request(
        torch.ones((1, 2)),
        context=ctx,
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=2,
        num_nodes=1,
        algorithm="deepseek_vec",
    )
    assert request.topology.num_physical_experts == 8
    assert request.topology.num_local_physical_experts == 4
    assert request.topology.num_groups == 2
    assert request.topology.num_nodes == 1
    assert request.policy.name == "deepseek_vec"
    assert request.policy.metadata == {}


def test_coupled_placement_uses_rank_and_preserves_quota(engine):
    from evenmoe import PlacementPlan
    from evenmoe.adapters.sglang import to_sglang_maps

    _, ctx = engine
    ctx.config_bag("exec").moe.moe_load_balancer_algorithm = "ultraep"
    ctx.parallel.moe_ep_rank = 1
    request = to_placement_request(torch.ones((2, 1, 2)), context=ctx)
    assert request.policy.name == "ultraep"
    assert request.policy.metadata == {"rank": 1, "num_nvl_ranks": 1}
    candidates = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32)
    replica_counts = torch.tensor([[2, 2]], dtype=torch.int32)
    quota = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.int32)
    live = ctx.resources.expert_location_metadata
    plan = PlacementPlan(
        physical_to_logical_map=live.physical_to_logical_map,
        logical_to_all_physical_map=candidates,
        logical_to_physical_count=replica_counts,
        metadata={"rank_quota_prefix": quota},
    )
    maps = to_sglang_maps(plan)
    live.logical_to_all_physical_map = torch.nn.functional.pad(
        candidates, (0, live.num_physical_experts - candidates.shape[-1]), value=-1
    )
    live.mlb_routing_metadata = maps.routing_metadata
    snapshot = to_placement_snapshot(live, 0)
    assert live.logical_to_all_physical_map.shape[-1] == live.num_physical_experts
    assert snapshot.logical_to_physical_candidates.data_ptr() == candidates.data_ptr()
    assert snapshot.logical_to_physical_count.data_ptr() == replica_counts.data_ptr()
    assert snapshot.metadata["rank_quota_prefix"].data_ptr() == quota.data_ptr()


def test_coupled_placement_rejects_uneven_node_distribution(engine):
    _, ctx = engine
    ctx.config_bag("exec").moe.moe_load_balancer_algorithm = "ultraep"
    ctx.parallel.ep_size = 4
    ctx.parallel.nnodes = 3
    with pytest.raises(ValueError, match="evenly distributed"):
        to_placement_request(torch.ones((4, 1, 2)), context=ctx)


def test_initial_placement_comes_from_the_coupled_policy(engine):
    _, ctx = engine
    ctx.config_bag("exec").moe.moe_load_balancer_algorithm = "ultraep"
    ctx.resources.mlb_model_info["num_logical_experts"] = 4
    ctx.config_bag("exec").moe.ep_num_redundant_experts = 2
    ctx.parallel.nnodes = 1
    mlb = MoELoadBalancer.from_algorithm(**to_load_balancer_kwargs(ctx))
    maps = plan_initial_placement(mlb, ctx, num_layers=2)
    # Two ranks of three slots: masters 2r, 2r+1, then the rank's first master.
    assert maps.physical_to_logical_map.tolist() == [[0, 1, 0, 2, 3, 2]] * 2
    assert maps.logical_to_physical_count.tolist() == [[2, 1, 2, 1]] * 2
    assert maps.routing_metadata == {}


def test_initial_placement_is_left_to_sglang_without_a_coupled_policy(engine):
    mlb, ctx = engine
    assert plan_initial_placement(mlb, ctx, num_layers=2) is None


class _TwoRankCollectives:
    """One rank of a two-rank EP group whose peer routed one token to expert 1."""

    world_size = 2

    def __init__(self, rank=0):
        self.rank = rank
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        peer_count = torch.tensor([0, 1], dtype=payload.dtype)
        if payload.numel() == peer_count.numel():
            return payload + peer_count
        peer = torch.zeros_like(payload)
        peer.view(self.world_size, -1)[1 - self.rank] = peer_count
        return payload + peer


class _RecordingPlanner:
    def __init__(self, plan, per_rank):
        self.result = plan
        self.requires_per_source_rank_statistics = per_rank
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return self.result


def _layer_mlb(plan, per_rank=True):
    """A balancer whose ``deepseek`` planner (the context's eplb_algorithm)
    records its request, with a two-rank collective."""
    planner = _RecordingPlanner(plan, per_rank)
    collectives = _TwoRankCollectives()
    mlb = MoELoadBalancer(placement_planners={"deepseek": planner}, collectives=collectives)
    return mlb, planner, collectives


def test_layer_plan_gathers_every_rank_and_keys_routing_state_by_layer(engine):
    _, ctx = engine
    quota = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.int32)
    plan = PlacementPlan(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32),
        logical_to_physical_count=torch.tensor([[2, 2]], dtype=torch.int32),
        metadata={"rank_quota_prefix": quota},
    )
    mlb, planner, collectives = _layer_mlb(plan)

    maps = plan_layer_from_topk(mlb, ctx, 5, torch.tensor([[0, 0], [1, -1]]))

    # This rank contributes only its own row, flattened so a framework
    # all-reduce cannot take it for [tokens, hidden]; padding (-1) is not counted.
    assert collectives.payloads[0].tolist() == [2, 1, 0, 0]
    (request,) = planner.requests
    assert request.policy.name == "deepseek"
    assert request.stats.logical_count.tolist() == [[[2, 1]], [[0, 1]]]
    assert list(maps.routing_metadata) == [5]
    assert maps.routing_metadata[5]["rank_quota_prefix"].data_ptr() == quota[0].data_ptr()
    assert maps.physical_to_logical_map is plan.physical_to_logical_map


def test_layer_plan_without_routing_state_publishes_none(engine):
    _, ctx = engine
    mlb, _, _ = _layer_mlb(_SINGLE_LAYER_PLAN)
    maps = plan_layer_from_topk(mlb, ctx, 3, torch.tensor([[0, 1]]))
    assert maps.routing_metadata == {}


_SINGLE_LAYER_PLAN = PlacementPlan(
    physical_to_logical_map=torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
    logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32),
    logical_to_physical_count=torch.tensor([[2, 2]], dtype=torch.int32),
)


def test_layer_plan_sums_ranks_for_a_policy_without_per_rank_statistics(engine):
    _, ctx = engine
    mlb, planner, collectives = _layer_mlb(_SINGLE_LAYER_PLAN, per_rank=False)

    plan_layer_from_topk(mlb, ctx, 3, torch.tensor([[0, 0], [1, -1]]))

    # Only this rank's [experts] vector goes over the wire, not a [ranks, experts] matrix.
    assert [p.tolist() for p in collectives.payloads] == [[2, 1]]
    (request,) = planner.requests
    assert request.stats.logical_count.tolist() == [[[2, 2]]]


class _UltraEPBackend:
    """Records what the UltraEP kernel would be given; solves to the master layout."""

    def __init__(self):
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["physical_to_logical_map"].copy_(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
        kwargs["logical_to_physical_map"].fill_(-1)
        kwargs["logical_to_physical_map"][:, 0].copy_(torch.tensor([0, 2], dtype=torch.int32))
        kwargs["logical_replica_counts"].fill_(1)
        kwargs["rank_quota_prefix"].fill_(7)


@pytest.mark.parametrize(
    "routing, eplb_algorithm, configured",
    [
        ("ultraep", "deepseek", "ultraep"),
        # UltraEP placement under static routing: the balancer has no placement
        # policy configured, and the context's eplb_algorithm still picks UltraEP.
        ("static", "ultraep", None),
    ],
)
def test_layer_plan_hands_ultraep_its_own_rank_and_every_rank_row(
    engine, routing, eplb_algorithm, configured
):
    """Through the real UltraEP L1 policy: the rank SGLang names in the policy
    metadata and the row this rank fills in the gather are the same rank."""
    from evenmoe.policies.l1.ultraep import UltraEPPlacementPolicy

    _, ctx = engine
    ctx.config_bag("exec").moe.moe_load_balancer_algorithm = routing
    ctx.config_bag("exec").moe.eplb_algorithm = eplb_algorithm
    ctx.parallel.moe_ep_rank = 1
    backend = _UltraEPBackend()
    mlb = MoELoadBalancer(
        placement_planners={"ultraep": UltraEPPlacementPolicy(backend=backend)},
        placement_policy=configured,
        collectives=_TwoRankCollectives(rank=1),
    )

    maps = plan_layer_from_topk(mlb, ctx, 7, torch.tensor([[0, 0], [0, -1]]))

    (call,) = backend.calls
    assert call["rank"] == 1
    assert call["expert_loads_per_rank"].tolist() == [[0, 1], [3, 0]]
    assert call["expert_loads"].tolist() == [3, 1]
    assert maps.physical_to_logical_map.tolist() == [[0, 0, 1, 1]]
    assert list(maps.routing_metadata) == [7]
    assert maps.routing_metadata[7]["rank_quota_prefix"].unique().tolist() == [7]


def test_deployment_config_lets_ultraep_report_its_layout_at_startup(engine):
    _, ctx = engine
    deployment = to_deployment_config(ctx, routes_shared_expert=False)
    assert deployment.ep_size == 2
    assert deployment.num_logical_experts == 2
    assert deployment.num_redundant_experts == 2
    assert deployment.carries_placement_metadata
    assert RoutingPipeline.from_value("ultraep").is_applicable(deployment) is None

    ctx.config_bag("exec").moe.ep_num_redundant_experts = 3
    reason = RoutingPipeline.from_value("ultraep").is_applicable(
        to_deployment_config(ctx, routes_shared_expert=False)
    )
    assert reason is not None and "split evenly" in reason


def test_deployment_config_before_model_info_skips_only_the_expert_count(engine):
    _, ctx = engine
    ctx.resources.mlb_model_info = None
    deployment = to_deployment_config(ctx, routes_shared_expert=True)
    assert deployment.num_logical_experts is None
    assert deployment.ep_size == 2
    assert deployment.routes_shared_expert
