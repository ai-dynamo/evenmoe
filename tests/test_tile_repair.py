# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from evenmoe import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPipeline,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from evenmoe.core.policy import ExpertDeploymentConfig
from evenmoe.kernels.tile_repair import (
    MAX_COPIES,
    NUM_STATS,
    adaptive_block_m,
    build_tile_repair_tables,
    tile_repair_plan_reference,
    tile_repair_route,
    tile_repair_route_reference,
)
from evenmoe.policies.l2.tile_repair import TileRepairL2Router, tables_from_snapshot


def _ceil(m, b):
    return -(-m // b)


def _random_p2l(num_logical, ranks, extras_per_rank, *, seed, layout_seed=None):
    """Rank-contiguous ``[primaries | extras]`` with random primaries and extras."""

    rng = random.Random(seed if layout_seed is None else layout_seed)
    perm = list(range(num_logical))
    rng.shuffle(perm)
    per_primary = num_logical // ranks
    p2l = []
    for r in range(ranks):
        primaries = perm[r * per_primary : (r + 1) * per_primary]
        others = [e for e in range(num_logical) if e not in primaries]
        p2l += primaries + rng.sample(others, extras_per_rank)
    return p2l


def _snapshot(p2l, *, num_logical, ranks, layer_id=0, device="cpu"):
    p2l_t = torch.tensor(p2l, dtype=torch.int64, device=device)
    num_physical = len(p2l)
    candidates = torch.full((num_logical, ranks + 1), -1, dtype=torch.int64)
    count = torch.zeros(num_logical, dtype=torch.int64)
    for p, e in enumerate(p2l):
        candidates[e, count[e]] = p
        count[e] += 1
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=num_logical,
        num_physical_experts=num_physical,
        ep_size=ranks,
        num_local_physical_experts=num_physical // ranks,
        physical_to_logical_map=p2l_t,
        logical_to_physical_candidates=candidates.to(device),
        logical_to_physical_count=count.to(device),
        default_physical_for_logical=candidates[:, 0].to(device),
    )


def _skewed_ids(tokens, top_k, num_logical, *, seed, alpha=1.2, pad=0):
    gen = torch.Generator().manual_seed(seed)
    weight = 1.0 / torch.arange(1, num_logical + 1, dtype=torch.float64) ** alpha
    weight = weight[torch.randperm(num_logical, generator=gen)]
    if tokens == 0:
        ids = torch.empty(0, top_k, dtype=torch.int64)
    else:
        ids = torch.multinomial(weight.expand(tokens, -1), top_k, generator=gen)
    if pad:
        ids = torch.cat([ids, torch.full((pad, top_k), -1, dtype=torch.int64)])
    return ids


def _histogram(ids, num_logical):
    valid = ids[(ids >= 0) & (ids < num_logical)]
    return torch.bincount(valid.reshape(-1), minlength=num_logical).to(torch.float32)


def _ep_batch(ranks, num_logical, *, seed, tokens=None, top_k=8, pad=0):
    rng = random.Random(seed)
    ids = []
    for r in range(ranks):
        t = rng.randint(0, 24) if tokens is None else tokens[r]
        ids.append(_skewed_ids(t, top_k, num_logical, seed=seed * 131 + r, pad=pad))
    counts = torch.stack([_histogram(x, num_logical) for x in ids])
    return ids, counts


def _check_ep_routing(ids, counts, tables, p2l, *, block_m, max_moves, outputs=None):
    """Invariants of one EP-wide routing step; returns the plan."""

    plan = tile_repair_plan_reference(counts, tables, block_m=block_m, max_moves=max_moves)
    num_logical = tables.num_logical
    ranks = tables.num_ranks
    p2l_t = torch.tensor(p2l)
    per_rank = len(p2l) // ranks
    slot_total = torch.zeros(len(p2l), dtype=torch.int64)
    for r in range(ranks):
        out = (
            tile_repair_route_reference(
                ids[r], counts, tables, source_rank=r, block_m=block_m, max_moves=max_moves
            )
            if outputs is None
            else outputs[r].cpu().to(torch.int64)
        )
        valid = (ids[r] >= 0) & (ids[r] < num_logical)
        assert torch.equal(out[~valid], torch.full_like(out[~valid], -1))
        # Every output slot is a copy of the input logical expert.
        assert torch.equal(p2l_t[out[valid]], ids[r][valid].to(torch.int64))
        slot_total += torch.bincount(out[valid], minlength=len(p2l))
    copy_physical = tables.copy_physical.tolist()
    for e in range(num_logical):
        for j in range(int(tables.num_copies[e])):
            assert int(slot_total[copy_physical[e][j]]) == int(plan.copy_tokens[e, j])
    # Realised per-rank tiles equal the plan, never above primary-only.
    tiles = [0] * ranks
    for p in range(len(p2l)):
        tiles[p // per_rank] += _ceil(int(slot_total[p]), block_m)
    assert tiles == plan.rank_tiles.tolist()
    assert plan.peak <= plan.primary_peak
    assert plan.moves <= max_moves
    return plan


def test_tables_put_the_primary_first_and_merge_same_rank_copies():
    # 4 logical, 2 ranks x 4 slots: rank 0 = [0, 1, 2, 2], rank 1 = [3, 2, 0, 1].
    p2l = [0, 1, 2, 2, 3, 2, 0, 1]
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=4, num_ranks=2, num_local_physical_experts=4
    )
    assert tables.copy_physical.tolist() == [
        [0, 6, -1, -1],
        [1, 7, -1, -1],
        [5, 2, -1, -1],  # local index 1 on rank 1 beats local index 2 on rank 0
        [4, -1, -1, -1],
    ]
    assert tables.copy_rank.tolist()[2] == [1, 0, -1, -1]
    assert tables.num_copies.tolist() == [2, 2, 2, 1]
    assert tables.replicated.tolist() == [0, 1, 2, -1]
    assert tables.num_replicated.tolist() == [3]


def test_tables_honour_an_explicit_rank_map():
    p2l = [0, 1, 0, 1]
    tables = build_tile_repair_tables(
        torch.tensor(p2l),
        num_logical=2,
        num_ranks=2,
        physical_to_rank_map=torch.tensor([0, 1, 1, 0]),
    )
    assert tables.copy_physical.tolist() == [[0, 2, -1, -1], [1, 3, -1, -1]]
    assert tables.copy_rank.tolist() == [[0, 1, -1, -1], [1, 0, -1, -1]]


def test_tables_reject_a_missing_expert():
    with pytest.raises(ValueError, match="no physical copy"):
        build_tile_repair_tables(
            torch.tensor([0, 0]), num_logical=2, num_ranks=1, num_local_physical_experts=2
        )


def test_repair_moves_the_partial_tile_of_a_replicated_expert():
    # Rank 0 holds 0, 1 (+ extra copy of 2); rank 1 holds 2, 3 (+ extra copy of 0).
    p2l = [0, 1, 2, 2, 3, 0]
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=4, num_ranks=2, num_local_physical_experts=3
    )
    # Expert 0 has 5 tokens (two tiles of 4), expert 1 has 3 (one tile), all on
    # rank 0; experts 2 and 3 have nothing. Moving expert 0's partial tile
    # (1 token) to its copy on rank 1 reaches the lower bound of 2 tiles.
    counts = torch.tensor([[5, 3, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    plan = tile_repair_plan_reference(counts, tables, block_m=4, max_moves=8)
    assert plan.primary_peak == 3
    assert plan.lower_bound == 2
    assert plan.copy_tokens[0].tolist() == [4, 1, 0, 0]
    assert plan.rank_tiles.tolist() == [2, 1]
    assert plan.moves == 1
    ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, -1], [0, -1]])
    out = tile_repair_route_reference(
        ids, counts, tables, source_rank=0, block_m=4, max_moves=8
    )
    assert out.tolist() == [[0, 1], [0, 1], [0, 1], [0, -1], [5, -1]]


def test_zero_moves_is_primary_only():
    p2l = _random_p2l(32, 4, 2, seed=1)
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=32, num_ranks=4, num_local_physical_experts=10
    )
    ids, counts = _ep_batch(4, 32, seed=3)
    for r in range(4):
        out = tile_repair_route_reference(
            ids[r], counts, tables, source_rank=r, block_m=16, max_moves=0
        )
        primary = tables.copy_physical[:, 0]
        expected = torch.where(ids[r] >= 0, primary[ids[r].clamp(min=0)], -1)
        assert torch.equal(out, expected)


@pytest.mark.parametrize("block_m", [1, 16, 32, 64, 128])
@pytest.mark.parametrize("ranks,num_logical,extras", [(2, 16, 2), (4, 64, 4), (8, 256, 4)])
def test_reference_invariants(block_m, ranks, num_logical, extras):
    for seed in range(6):
        p2l = _random_p2l(num_logical, ranks, extras, seed=seed)
        tables = build_tile_repair_tables(
            torch.tensor(p2l),
            num_logical=num_logical,
            num_ranks=ranks,
            num_local_physical_experts=len(p2l) // ranks,
        )
        tokens = None if seed % 2 else [64] * ranks
        ids, counts = _ep_batch(ranks, num_logical, seed=seed, tokens=tokens, pad=seed % 3)
        plan = _check_ep_routing(ids, counts, tables, p2l, block_m=block_m, max_moves=32)
        if plan.moves < 32:
            # Stopped at the lower bound or because no move strictly helps:
            # a larger budget changes nothing.
            more = tile_repair_plan_reference(
                counts, tables, block_m=block_m, max_moves=plan.moves + 5
            )
            assert more.moves == plan.moves
            assert torch.equal(more.copy_tokens, plan.copy_tokens)


def test_repair_reaches_the_lower_bound_when_copies_allow_it():
    # Two ranks, every expert on both: the partial-tile moves level the tiles.
    p2l = [0, 1, 2, 3, 0, 1, 2, 3]
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=4, num_ranks=2, num_local_physical_experts=4
    )
    counts = torch.tensor([[3, 2, 0, 0], [0, 0, 2, 1]], dtype=torch.float32)
    plan = tile_repair_plan_reference(counts, tables, block_m=8, max_moves=32)
    assert plan.lower_bound == 2
    assert plan.peak == 2


def _routing_request(ids, placement, stage=RoutingStage.DECODE, policies=()):
    return RoutingRequest(
        layer_id=placement.layer_id,
        logical_topk_ids=ids,
        topk_weights=torch.ones(ids.shape),
        placement=placement,
        stage=stage,
        policies=policies,
    )


class _PeerCollectives:
    """Rank ``rank`` of an EP group whose peers contributed ``counts``' other rows."""

    def __init__(self, rank, counts):
        self.rank = rank
        self.world_size = counts.shape[0]
        self.counts = counts
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        peers = self.counts.clone()
        peers[self.rank] = 0
        return payload + peers.to(payload)


def _ep_balancers(ranks, counts, algorithm="tile_repair", **options):
    out = []
    for r in range(ranks):
        collectives = _PeerCollectives(r, counts)
        mlb = MoELoadBalancer.from_algorithm(
            algorithm,
            ep_size=ranks,
            source_rank=r,
            experts_per_rank=1,
            collectives=collectives,
            **options,
        )
        out.append((mlb, collectives))
    return out


def test_orchestrator_gathers_per_rank_counts_and_routes_every_rank():
    ranks, num_logical = 4, 32
    p2l = _random_p2l(num_logical, ranks, 2, seed=7)
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=11, tokens=[6, 0, 13, 9], pad=1)
    outputs = []
    for r, (mlb, collectives) in enumerate(_ep_balancers(ranks, counts)):
        mlb.on_placement_committed(placement)
        decision = mlb.route_tokens(_routing_request(ids[r], placement))
        assert decision.metadata["executed_policies"] == ("tile_repair",)
        assert decision.metadata["tile_repair"] is True
        (payload,) = collectives.payloads
        assert payload.shape == (ranks, num_logical)
        assert torch.equal(payload[r], counts[r])
        others = torch.cat([payload[:r], payload[r + 1 :]])
        assert not others.any()
        outputs.append(decision.routed_physical_topk_ids)
    tables = tables_from_snapshot(placement)
    _check_ep_routing(ids, counts, tables, p2l, block_m=64, max_moves=32, outputs=outputs)


@pytest.mark.parametrize("stage", [RoutingStage.PREFILL, RoutingStage.MIXED])
def test_prefill_routes_primary_only_but_joins_the_collective(stage):
    ranks, num_logical = 2, 8
    p2l = [0, 1, 2, 3, 4, 7, 4, 5, 6, 7, 0, 1]
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=5, tokens=[8, 8])
    (mlb, collectives), _ = _ep_balancers(ranks, counts)
    decision = mlb.route_tokens(_routing_request(ids[0], placement, stage=stage))
    assert len(collectives.payloads) == 1
    assert decision.metadata["tile_repair"] is False
    primary = tables_from_snapshot(placement).copy_physical[:, 0]
    assert torch.equal(decision.routed_physical_topk_ids, primary[ids[0]])


def test_stages_all_repairs_prefill_too(monkeypatch):
    monkeypatch.setenv("EVENMOE_TILE_REPAIR_STAGES", "all")
    ranks, num_logical = 2, 8
    p2l = [0, 1, 2, 3, 4, 7, 4, 5, 6, 7, 0, 1]
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=5, tokens=[8, 8])
    (mlb, _), _ = _ep_balancers(ranks, counts)
    decision = mlb.route_tokens(_routing_request(ids[0], placement, stage=RoutingStage.PREFILL))
    assert decision.metadata["tile_repair"] is True


def test_layer_without_replicas_skips_the_collective():
    placement = _snapshot([0, 1, 2, 3], num_logical=4, ranks=2)
    ids = torch.tensor([[0, 3], [1, 2]])
    (mlb, collectives), _ = _ep_balancers(2, torch.zeros(2, 4))
    decision = mlb.route_tokens(_routing_request(ids, placement))
    assert collectives.payloads == []
    assert decision.routed_physical_topk_ids.tolist() == [[0, 3], [1, 2]]


def test_count_logical_experts_into_a_payload_row():
    from evenmoe.kernels.ops.counting import count_logical_experts

    payload = torch.zeros(3, 4)
    ids = torch.tensor([[0, 3], [3, -1], [2, 3]])
    count_logical_experts(ids, 4, out=payload[1])
    assert payload.tolist() == [[0, 0, 0, 0], [1, 0, 1, 3], [0, 0, 0, 0]]
    with pytest.raises(ValueError, match="contiguous float32"):
        count_logical_experts(ids, 4, out=torch.zeros(4, 3)[:, 0])


def test_ep_summed_count_is_rejected():
    p2l = [0, 1, 2, 3, 4, 7, 4, 5, 6, 7, 0, 1]
    placement = _snapshot(p2l, num_logical=8, ranks=2)
    router = TileRepairL2Router(num_ranks=2, source_rank=0)
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.ones(1, 2),
        placement=placement,
        stage=RoutingStage.DECODE,
        global_logical_count=torch.ones(8),
    )
    with pytest.raises(ValueError, match="per-source-rank"):
        router.route_tokens(request, RoutingPolicyConfig(name="tile_repair"))


def test_policy_metadata_overrides_block_m(monkeypatch):
    monkeypatch.setenv("EVENMOE_TILE_BLOCK_M", "64")
    p2l = [0, 1, 2, 3, 0, 1, 2, 3]
    placement = _snapshot(p2l, num_logical=4, ranks=2)
    counts = torch.tensor([[3, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    ids = torch.tensor([[0, 1], [0, 1], [0, -1]])
    router = TileRepairL2Router(num_ranks=2, source_rank=0)
    assert router.block_m == 64

    def route(policy):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=ids,
            topk_weights=torch.ones(ids.shape),
            placement=placement,
            stage=RoutingStage.DECODE,
            global_logical_count=counts,
        )
        return router.route_tokens(request, policy).routed_physical_topk_ids.tolist()

    # block_m=64: one tile per expert, rank 0 has 2 tiles against a lower bound
    # of 1; the larger chunk (expert 0, 3 tokens) moves to rank 1.
    assert route(RoutingPolicyConfig(name="tile_repair")) == [[4, 1], [4, 1], [4, -1]]
    # block_m=2: expert 0 is two tiles, expert 1 one; rank 0 has 3 against a lower
    # bound of 2. Expert 1's full tile (chunk 2) beats expert 0's partial tile.
    small = route(RoutingPolicyConfig(name="tile_repair", metadata={"block_m": 2}))
    assert small == [[0, 5], [0, 5], [0, -1]]


def test_pipeline_declarations():
    pipeline = RoutingPipeline.from_value("tile_repair")
    capabilities = pipeline.capabilities
    assert capabilities.consumes_per_rank_logical_count
    assert capabilities.consumes_global_logical_count
    assert capabilities.requires_placement_state
    assert capabilities.graph_stability == "stable"
    assert capabilities.supports_concurrent_microbatches
    assert not capabilities.dispatch_fixed_by_placement
    reason = pipeline.is_applicable(ExpertDeploymentConfig(num_redundant_experts=0))
    assert reason is not None and "redundant" in reason
    assert pipeline.is_applicable(ExpertDeploymentConfig(num_redundant_experts=32)) is None
    combined = RoutingPipeline.from_value("tile_repair+waterfill")
    assert combined.policies == ("tile_repair", "waterfill")
    assert combined.capabilities.supports_concurrent_microbatches


def test_routers_on_every_rank_build_the_same_tables():
    p2l = _random_p2l(64, 8, 4, seed=2)
    placement = _snapshot(p2l, num_logical=64, ranks=8)
    reference = tables_from_snapshot(placement)
    for r in range(8):
        router = TileRepairL2Router(num_ranks=8, source_rank=r)
        router.prepare_layer(placement)
        tables = router._layer_state[0].tables
        for a, b in zip(tables.tensors(), reference.tensors()):
            assert torch.equal(a, b)


# --------------------------------------------------------------------------- CUDA

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _cuda_tables(p2l, num_logical, ranks):
    return build_tile_repair_tables(
        torch.tensor(p2l),
        num_logical=num_logical,
        num_ranks=ranks,
        num_local_physical_experts=len(p2l) // ranks,
        device="cuda",
    )


@cuda
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "ranks,num_logical,extras",
    [(2, 16, 2), (4, 64, 4), (8, 256, 4), (8, 256, 32), (16, 512, 8), (32, 256, 2)],
)
def test_cuda_kernel_matches_reference(block_m, ranks, num_logical, extras):
    for seed in range(3):
        p2l = _random_p2l(num_logical, ranks, extras, seed=seed + 100)
        tables = _cuda_tables(p2l, num_logical, ranks)
        if seed == 0:
            tokens = [0 if r == 1 else 48 for r in range(ranks)]  # an empty rank
        elif seed == 1:
            tokens = [block_m * 3] * ranks  # several tiles per hot expert
        else:
            tokens = None
        ids, counts = _ep_batch(ranks, num_logical, seed=seed, tokens=tokens, pad=2 * (seed % 2))
        plan = tile_repair_plan_reference(counts, tables, block_m=block_m, max_moves=32)
        outputs = []
        plans = []
        for r in range(ranks):
            plan_out = torch.full((num_logical * MAX_COPIES,), -7, dtype=torch.int32, device="cuda")
            stats = torch.full((NUM_STATS + ranks,), -7, dtype=torch.int32, device="cuda")
            out = tile_repair_route(
                ids[r].cuda(),
                counts.cuda(),
                tables,
                source_rank=r,
                block_m=block_m,
                max_moves=32,
                plan_out=plan_out,
                stats_out=stats,
            )
            expected = tile_repair_route_reference(
                ids[r], counts, tables, source_rank=r, block_m=block_m, max_moves=32, plan=plan
            )
            assert torch.equal(out.cpu(), expected)
            assert plan_out.view(num_logical, MAX_COPIES).cpu().tolist() == plan.copy_tokens.tolist()
            assert stats.cpu().tolist() == [
                plan.moves,
                plan.lower_bound,
                plan.primary_peak,
                plan.peak,
                block_m,
                *plan.rank_tiles.tolist(),
            ]
            outputs.append(out)
            plans.append(plan_out.cpu())
        # Identical plan on every rank, and the realised routing honours it.
        assert all(torch.equal(p, plans[0]) for p in plans)
        _check_ep_routing(ids, counts, tables, p2l, block_m=block_m, max_moves=32, outputs=outputs)


@cuda
@pytest.mark.parametrize("in_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_cuda_dtypes_padding_and_empty_batches(in_dtype, out_dtype):
    ranks, num_logical = 8, 256
    p2l = _random_p2l(num_logical, ranks, 4, seed=9)
    tables = _cuda_tables(p2l, num_logical, ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=4, tokens=[0, 3, 8, 0, 16, 1, 2, 9], pad=3)
    for r in range(ranks):
        out = tile_repair_route(
            ids[r].to(in_dtype).cuda(),
            counts.cuda(),
            tables,
            source_rank=r,
            block_m=64,
            out_dtype=out_dtype,
        )
        assert out.dtype == out_dtype and out.shape == ids[r].shape
        expected = tile_repair_route_reference(
            ids[r], counts, tables, source_rank=r, block_m=64, max_moves=32
        )
        assert torch.equal(out.cpu().to(torch.int64), expected)
    empty = torch.empty(0, 8, dtype=in_dtype, device="cuda")
    out = tile_repair_route(empty, counts.cuda(), tables, source_rank=0, block_m=64)
    assert out.shape == (0, 8)


@cuda
def test_cuda_large_batch_matches_reference():
    # More entries than threads in the CTA, many tiles per expert.
    ranks, num_logical = 8, 256
    p2l = _random_p2l(num_logical, ranks, 4, seed=21)
    tables = _cuda_tables(p2l, num_logical, ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=8, tokens=[700] * ranks)
    for r in (0, 5):
        out = tile_repair_route(ids[r].cuda(), counts.cuda(), tables, source_rank=r, block_m=32)
        expected = tile_repair_route_reference(
            ids[r], counts, tables, source_rank=r, block_m=32, max_moves=32
        )
        assert torch.equal(out.cpu(), expected)


@cuda
def test_cuda_router_through_the_balancer():
    ranks, num_logical = 4, 64
    p2l = _random_p2l(num_logical, ranks, 4, seed=13)
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks, device="cuda")
    ids, counts = _ep_batch(ranks, num_logical, seed=17, tokens=[16, 5, 0, 32])
    outputs = []
    for r, (mlb, _) in enumerate(_ep_balancers(ranks, counts.cuda())):
        mlb.on_placement_committed(placement)
        decision = mlb.route_tokens(_routing_request(ids[r].cuda(), placement))
        outputs.append(decision.routed_physical_topk_ids)
    tables = build_tile_repair_tables(
        torch.tensor(p2l),
        num_logical=num_logical,
        num_ranks=ranks,
        num_local_physical_experts=len(p2l) // ranks,
    )
    _check_ep_routing(ids, counts, tables, p2l, block_m=64, max_moves=32, outputs=outputs)


@cuda
def test_cuda_graph_replays_an_updated_placement_and_batch():
    ranks, num_logical = 8, 256
    first = _random_p2l(num_logical, ranks, 4, seed=31)
    second = _random_p2l(num_logical, ranks, 4, seed=32)
    router = TileRepairL2Router(num_ranks=ranks, source_rank=3)
    router.prepare_layer(_snapshot(first, num_logical=num_logical, ranks=ranks, device="cuda"))
    ids, counts = _ep_batch(ranks, num_logical, seed=1, tokens=[16] * ranks)
    static_ids = ids[3].cuda()
    static_counts = counts.cuda()
    placement = _snapshot(first, num_logical=num_logical, ranks=ranks, device="cuda")
    policy = RoutingPolicyConfig(name="tile_repair")

    def route():
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=static_ids,
            topk_weights=torch.ones(static_ids.shape, device="cuda"),
            placement=placement,
            stage=RoutingStage.DECODE,
            global_logical_count=static_counts,
        )
        return router.route_tokens(request, policy).routed_physical_topk_ids

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route()  # compile and warm up outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route()
    graph.replay()
    torch.cuda.synchronize()
    tables = build_tile_repair_tables(
        torch.tensor(first), num_logical=num_logical, num_ranks=ranks, num_local_physical_experts=36
    )
    expected = tile_repair_route_reference(
        ids[3], counts, tables, source_rank=3, block_m=64, max_moves=32
    )
    assert torch.equal(captured.cpu(), expected)

    # New batch and a new same-shape placement, both updated in place.
    ids2, counts2 = _ep_batch(ranks, num_logical, seed=2, tokens=[16] * ranks)
    static_ids.copy_(ids2[3].cuda())
    static_counts.copy_(counts2.cuda())
    router.prepare_layer(_snapshot(second, num_logical=num_logical, ranks=ranks, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    tables2 = build_tile_repair_tables(
        torch.tensor(second), num_logical=num_logical, num_ranks=ranks, num_local_physical_experts=36
    )
    expected2 = tile_repair_route_reference(
        ids2[3], counts2, tables2, source_rank=3, block_m=64, max_moves=32
    )
    assert torch.equal(captured.cpu(), expected2)


@cuda
@pytest.mark.parametrize("start_with_remote_copies", [False, True])
def test_cuda_graph_keeps_collective_when_copy_ranks_change(start_with_remote_copies):
    local_copies = [0, 1, 0, 2, 3, 2]
    remote_copies = [0, 1, 2, 2, 3, 0]
    layouts = [local_copies, remote_copies]
    if start_with_remote_copies:
        layouts.reverse()
    counts = torch.tensor([[128, 64, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    (mlb, collectives), _ = _ep_balancers(2, counts.cuda())
    first = _snapshot(layouts[0], num_logical=4, ranks=2, device="cuda")
    mlb.on_placement_committed(first)
    ids = torch.tensor([0] * 128 + [1] * 64, device="cuda").reshape(-1, 1)
    request = _routing_request(ids, first)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mlb.route_tokens(request)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = mlb.route_tokens(request).routed_physical_topk_ids
    # Both layouts must capture the collective, including local-only copies.
    assert len(collectives.payloads) == 2
    for p2l in (layouts[1], layouts[0]):
        placement = _snapshot(p2l, num_logical=4, ranks=2, device="cuda")
        mlb.on_placement_committed(placement)
        graph.replay()
        torch.cuda.synchronize()
        eager = mlb.route_tokens(_routing_request(ids, placement)).routed_physical_topk_ids
        expected = tile_repair_route_reference(
            ids.cpu(), counts, tables_from_snapshot(placement), source_rank=0,
            block_m=64, max_moves=32,
        )
        assert torch.equal(captured.cpu(), expected)
        assert torch.equal(eager.cpu(), expected)


@cuda
@pytest.mark.parametrize("logical,ranks", [(1024, 2), (256, 64)])
def test_unsupported_geometry_skips_exchange_and_replays_primary_mapping(logical, ranks):
    # No collectives: asking for a count is itself a regression on this path.
    mlb = MoELoadBalancer.from_algorithm(
        "tile_repair", ep_size=ranks, source_rank=0, experts_per_rank=logical // ranks + 1,
    )
    ids = torch.tensor([[0, logical - 1], [7, -1]], device="cuda")
    first = _snapshot(
        _random_p2l(logical, ranks, 1, seed=32), num_logical=logical, ranks=ranks, device="cuda"
    )
    mlb.on_placement_committed(first)
    request = _routing_request(ids, first)
    mlb.route_tokens(request)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mlb.route_tokens(request).routed_physical_topk_ids
    for placement in (first, _snapshot(
        _random_p2l(logical, ranks, 1, seed=33), num_logical=logical, ranks=ranks, device="cuda"
    )):
        mlb.on_placement_committed(placement)
        graph.replay()
        torch.cuda.synchronize()
        primary = tables_from_snapshot(placement).primary_physical.long()
        expected = torch.where(ids >= 0, primary[ids.clamp(min=0)], -1)
        decision = mlb.route_tokens(_routing_request(ids, placement))
        assert not decision.metadata["tile_repair"]
        assert torch.equal(decision.routed_physical_topk_ids, expected)
        assert torch.equal(captured, expected)


@pytest.mark.parametrize(
    "rows,slots,expected",
    [(0, 288, 8), (1583, 288, 8), (1584, 288, 16), (3657, 288, 16), (3658, 288, 32),
     (8179, 288, 32), (8180, 288, 48), (23039, 288, 48), (23040, 288, 64), (10**6, 288, 64)],
)
def test_adaptive_block_m_buckets(rows, slots, expected):
    # em = rows / slots against 5.5 / 12.7 / 28.4 / 80 in exact integer compares.
    assert adaptive_block_m(rows, slots) == expected


def test_adaptive_plan_reference_resolves_block_m():
    ranks, num_logical = 8, 256
    p2l = _random_p2l(num_logical, ranks, 4, seed=3)
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=num_logical, num_ranks=ranks, num_local_physical_experts=36
    )
    for tokens, seed in ((4, 1), (16, 2), (64, 3), (512, 4)):
        _, counts = _ep_batch(ranks, num_logical, seed=seed, tokens=[tokens] * ranks)
        bm = adaptive_block_m(int(counts.sum()), len(p2l))
        adaptive = tile_repair_plan_reference(
            counts, tables, block_m=0, max_moves=32, num_slots=len(p2l)
        )
        fixed = tile_repair_plan_reference(counts, tables, block_m=bm, max_moves=32)
        assert adaptive.block_m == bm
        assert torch.equal(adaptive.copy_tokens, fixed.copy_tokens)
    with pytest.raises(ValueError, match="num_slots"):
        tile_repair_plan_reference(counts, tables, block_m=0, max_moves=32)


@cuda
@pytest.mark.parametrize("block_m", [0, 64])
def test_router_owns_scratch_per_concurrent_microbatch(block_m):
    ranks, logical = 4, 64
    p2l = _random_p2l(logical, ranks, 4, seed=17)
    placement = _snapshot(p2l, num_logical=logical, ranks=ranks, device="cuda")
    router = TileRepairL2Router(num_ranks=ranks, source_rank=0, block_m=block_m)
    router.prepare_layer(placement)
    policy = RoutingPolicyConfig(name="tile_repair")
    tables = tables_from_snapshot(placement)
    pending = []
    for seed in range(2):
        ids, counts = _ep_batch(ranks, logical, seed=seed, tokens=[16 + seed * 64] * ranks)
        expected = tile_repair_route_reference(
            ids[0], counts, tables, source_rank=0, block_m=block_m, max_moves=32,
            num_slots=len(p2l),
        )
        request = replace(
            _routing_request(ids[0].cuda(), placement), global_logical_count=counts.cuda()
        )
        router.route_tokens(request, policy)  # compile before capturing
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = router.route_tokens(request, policy).routed_physical_topk_ids
        pending.append((stream, graph, request, output, expected))
    torch.cuda.synchronize()
    for _ in range(4):
        for stream, graph, _, _, _ in pending:
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for _, _, _, output, expected in pending:
        assert torch.equal(output.cpu(), expected)


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two or more CUDA devices",
)
def test_collective_graph_placement_updates_across_processes(tmp_path):
    world = min(torch.cuda.device_count(), 8)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "src"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
         os.path.join(os.path.dirname(__file__), "multigpu_tile_repair.py"), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=900, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    results = sorted(tmp_path.glob("rank*.ok"))
    assert len(results) == world, proc.stdout[-2000:] + proc.stderr[-2000:]
