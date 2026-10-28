# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch

from evenmoe import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPipeline,
    RoutingRequest,
    RoutingStage,
)
from evenmoe.kernels.metro_repair import (
    build_metro_repair_tables,
    metro_repair_assign_reference,
    metro_repair_route,
)
from evenmoe.kernels.ops.metro import build_metro_tables, metro_assign_reference, metro_route
from evenmoe.policies.l2.metro_repair import MetroRepairL2Router, rank_invariant_candidates

# Six logical experts on three ranks of three slots: rank 0 holds 0, 2, 3; rank 1
# holds 1, 2, 4; rank 2 holds 0, 1, 5. With 0, 1, 2 active METRO's greedy
# (ascending id, ties to the lowest rank) sends 0 -> rank 0, 1 -> rank 1 and
# 2 -> rank 0, i.e. loads [2, 1, 0]; moving logical 0 to its copy on rank 2
# reaches [1, 1, 1].
_P2L = [0, 2, 3, 1, 2, 4, 0, 1, 5]
_CANDIDATES = [[0, 6], [3, 7], [1, 4], [2, -1], [5, -1], [8, -1]]
_STATIC = [0, 3, 1, 2, 5, 8]


class _RecordingCollectives:
    rank = 0
    world_size = 3

    def __init__(self, peer_contribution):
        self.peer_contribution = peer_contribution
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        return payload + self.peer_contribution.to(payload)


def _placement(p2l=_P2L, candidates=_CANDIDATES, *, ranks=3, layer_id=0, device="cpu"):
    candidates = torch.tensor(candidates, dtype=torch.int32, device=device)
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=candidates.shape[0],
        num_physical_experts=len(p2l),
        ep_size=ranks,
        num_local_physical_experts=len(p2l) // ranks,
        physical_to_logical_map=torch.tensor(p2l, dtype=torch.int32, device=device),
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=(candidates >= 0).sum(dim=1).to(torch.int32),
        default_physical_for_logical=torch.tensor(
            _STATIC[: candidates.shape[0]], dtype=torch.int32
        ),
    )


def _request(ids, *, placement=None, stage=RoutingStage.DECODE, token_count=None):
    ids = torch.tensor(ids, dtype=torch.int32)
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones(ids.shape),
        placement=_placement() if placement is None else placement,
        stage=stage,
        token_count=token_count,
    )


def _balancer(peer_contribution, algorithm="metro_repair"):
    collectives = _RecordingCollectives(torch.tensor(peer_contribution, dtype=torch.float32))
    balancer = MoELoadBalancer.from_algorithm(
        algorithm, ep_size=3, source_rank=0, experts_per_rank=3, collectives=collectives
    )
    return balancer, collectives


def _tables(p2l=_P2L, candidates=_CANDIDATES, ranks=3):
    return build_metro_repair_tables(build_metro_tables(torch.tensor(candidates), len(p2l), ranks))


def _random_placement(num_logical, num_ranks, redundant, generator):
    """Home experts rank-contiguous, redundant slots filled with skewed random experts."""

    home = -(-num_logical // num_ranks)
    extra = redundant // num_ranks
    weights = torch.rand(num_logical, generator=generator) ** 3 + 0.05
    slots = []
    for rank in range(num_ranks):
        own = [e for e in range(rank * home, min((rank + 1) * home, num_logical))]
        own += [e for e in range(num_logical)][: max(0, home - len(own))]
        slots += own[:home]
        slots += torch.multinomial(weights, extra, replacement=True, generator=generator).tolist()
    p2l = torch.tensor(slots)
    width = int(torch.bincount(p2l, minlength=num_logical).max())
    candidates = torch.full((num_logical, width), -1, dtype=torch.int32)
    for logical in range(num_logical):
        held = torch.nonzero(p2l == logical).flatten()
        candidates[logical, : held.numel()] = held.to(torch.int32)
    return p2l, candidates


def _peak(load):
    return int(load.max())


# ----------------------------------------------------------------------------- host oracle


@pytest.mark.unit
def test_holder_mask():
    tables = _tables()
    assert tables.holder_mask.tolist() == [5, 6, 3, 1, 2, 4]


@pytest.mark.unit
def test_zero_moves_is_metro():
    generator = torch.Generator().manual_seed(1)
    for ranks in (2, 3, 8, 16):
        p2l, candidates = _random_placement(64, ranks, 3 * ranks, generator)
        metro = build_metro_tables(candidates, p2l.numel(), ranks)
        tables = build_metro_repair_tables(metro)
        for _ in range(20):
            active = torch.rand(64, generator=generator) < torch.rand(1, generator=generator)
            _, owner = metro_repair_assign_reference(active, tables, moves=0)
            chosen = metro_assign_reference(active.float(), metro)
            live = torch.nonzero(active).flatten()
            got = metro.physical_by_rank[live, owner[live].long()]
            assert torch.equal(got.long(), chosen[live])


@pytest.mark.unit
def test_repair_finds_the_reachable_balance():
    tables = _tables()
    active = torch.tensor([True, True, True, False, False, False])

    metro_load, metro_owner = metro_repair_assign_reference(active, tables, moves=0)
    load, owner = metro_repair_assign_reference(active, tables, moves=32)

    assert metro_load.tolist() == [2, 1, 0]
    assert metro_owner.tolist() == [0, 1, 0, -1, -1, -1]
    assert load.tolist() == [1, 1, 1]
    # The lowest movable logical id on the (0 -> 2) edge moves.
    assert owner.tolist() == [2, 1, 0, -1, -1, -1]


@pytest.mark.unit
def test_two_hop_repair():
    # Rank 0 holds 1, 2, 3; rank 1 holds 0, 1, 2; rank 2 holds 0, 4, 5. METRO sends
    # 0 -> rank 1, 1 -> rank 0, 2 -> rank 0 ([2, 1, 0]). No expert on rank 0 has a
    # copy on rank 2, but 0 -> 1 (logical 1) and 1 -> 2 (logical 0) form a path.
    p2l = [1, 2, 3, 0, 1, 2, 0, 4, 5]
    candidates = [[3, 6], [0, 4], [1, 5], [2, -1], [7, -1], [8, -1]]
    tables = _tables(p2l, candidates)
    active = torch.tensor([True, True, True, False, False, False])

    metro_load, _ = metro_repair_assign_reference(active, tables, moves=0)
    load, owner = metro_repair_assign_reference(active, tables, moves=32)

    assert metro_load.tolist() == [2, 1, 0]
    assert load.tolist() == [1, 1, 1]
    assert owner.tolist() == [2, 1, 0, -1, -1, -1]


@pytest.mark.unit
def test_repair_never_raises_the_peak_and_early_exit_keeps_it():
    generator = torch.Generator().manual_seed(2)
    for ranks in (2, 4, 8):
        p2l, candidates = _random_placement(48, ranks, 2 * ranks, generator)
        tables = build_metro_repair_tables(build_metro_tables(candidates, p2l.numel(), ranks))
        for _ in range(30):
            active = torch.rand(48, generator=generator) < 0.3
            base, _ = metro_repair_assign_reference(active, tables, moves=0)
            full, _ = metro_repair_assign_reference(active, tables, moves=64)
            early, _ = metro_repair_assign_reference(active, tables, moves=64, early_exit=True)
            assert _peak(full) <= _peak(base)
            assert _peak(early) == _peak(full)
            assert int(full.sum()) == int(active.sum())


# ----------------------------------------------------------------------------- tables


@pytest.mark.unit
def test_rank_invariant_candidates_ignore_per_rank_pruning():
    # SGLang with an a2a backend keeps only the local copy where one exists.
    pruned_rank2 = [[6, -1], [7, -1], [1, 4], [2, -1], [5, -1], [8, -1]]
    full = rank_invariant_candidates(_placement())
    pruned = rank_invariant_candidates(_placement(candidates=pruned_rank2))

    assert torch.equal(full, pruned)
    assert full.tolist() == _CANDIDATES


@pytest.mark.unit
def test_inconsistent_candidates_are_rejected():
    with pytest.raises(ValueError):
        rank_invariant_candidates(
            _placement(candidates=[[1, 6], [3, 7], [1, 4], [2, -1], [5, -1], [8, -1]])
        )


@pytest.mark.unit
def test_routers_on_every_rank_build_the_same_tables():
    tables = []
    for local in range(3):
        pruned = []
        for copies in _CANDIDATES:
            mine = [p for p in copies if p >= 0 and p // 3 == local]
            pruned.append((mine[:1] + [-1]) if mine else copies)
        router = MetroRepairL2Router(num_gpus=3)
        router.prepare_layer(_placement(candidates=pruned))
        tables.append(router._layer_state[0].tables)
    for other in tables[1:]:
        assert all(torch.equal(a, b) for a, b in zip(tables[0].tensors(), other.tensors()))


# ----------------------------------------------------------------------------- router


@pytest.mark.unit
def test_decode_repairs_the_metro_assignment():
    balancer, collectives = _balancer([0.0] * 6)
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]]))

    assert collectives.payloads[0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert decision.routed_physical_topk_ids.dtype == torch.int64
    # METRO would route [[0, 3], [1, -1]]; logical 0 moves to its copy on rank 2.
    assert decision.routed_physical_topk_ids.tolist() == [[6, 3], [1, -1]]
    assert decision.metadata["metro_assignment"] is True
    assert decision.metadata["executed_policies"] == ("metro_repair",)


@pytest.mark.unit
def test_zero_moves_matches_metro():
    metro, _ = _balancer([0.0] * 6, algorithm="metro")
    metro.on_placement_committed(_placement())
    request = _request([[0, 1], [2, -1]])
    want = metro.route_tokens(request).routed_physical_topk_ids

    router = MetroRepairL2Router(num_gpus=3, moves=0)
    router.prepare_layer(_placement())
    got = router.route_tokens(
        replace(
            request,
            global_logical_count=torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            policies=("metro_repair",),
        )
    ).routed_physical_topk_ids

    assert got.tolist() == want.tolist() == [[0, 3], [1, -1]]


@pytest.mark.unit
def test_peer_counts_join_the_active_set():
    # The peer's logical 3 (single copy on rank 0) is fixed load: METRO then gives
    # [2, 1, 1], already at the bound ceil(4 / 3) = 2, so nothing moves.
    balancer, _ = _balancer([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]]))

    assert decision.routed_physical_topk_ids.tolist() == [[6, 3], [1, -1]]


@pytest.mark.unit
def test_padding_rows_are_dropped():
    balancer, _ = _balancer([0.0] * 6)
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(
        _request([[0, 1], [2, 3]], token_count=torch.tensor(1, dtype=torch.int32))
    )

    assert decision.routed_physical_topk_ids.tolist() == [[6, 3], [-1, -1]]


@pytest.mark.unit
@pytest.mark.parametrize("stage", [RoutingStage.PREFILL, RoutingStage.MIXED])
def test_prefill_uses_the_static_map_but_joins_the_collective(stage):
    balancer, collectives = _balancer([0.0] * 6)
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]], stage=stage))

    assert len(collectives.payloads) == 1
    assert decision.routed_physical_topk_ids.tolist() == [[0, 3], [1, -1]]
    assert decision.metadata["metro_assignment"] is False


@pytest.mark.unit
def test_layer_without_replicas_skips_the_collective():
    balancer, collectives = _balancer([0.0] * 6)
    single = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=6,
        num_physical_experts=6,
        ep_size=3,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.arange(6, dtype=torch.int32),
        logical_to_physical_candidates=torch.arange(6, dtype=torch.int32).view(6, 1),
        logical_to_physical_count=torch.ones(6, dtype=torch.int32),
    )
    balancer.on_placement_committed(single)

    decision = balancer.route_tokens(_request([[0, 5]], placement=single))

    assert collectives.payloads == []
    assert decision.routed_physical_topk_ids.tolist() == [[0, 5]]


@pytest.mark.unit
def test_pipeline_declarations():
    capabilities = RoutingPipeline.from_value("metro_repair").capabilities

    assert capabilities.consumes_global_logical_count
    assert capabilities.requires_rank_dispatch_map
    assert capabilities.requires_deepep
    assert capabilities.supports_concurrent_microbatches
    assert not capabilities.dispatch_fixed_by_placement
    assert capabilities.graph_stability == "stable"
    with pytest.raises(ValueError):
        RoutingPipeline.from_value("metro_repair+lplb")


# ----------------------------------------------------------------------------- CUDA


def _want(ids, counts, tables, moves, early_exit, gate=None):
    _, owner = metro_repair_assign_reference(counts > 0, tables, moves=moves, early_exit=early_exit)
    metro = tables.metro
    by_rank = metro.physical_by_rank.cpu().long()
    logical = torch.arange(metro.num_logical)
    owner = owner.long()
    chosen = torch.where(
        owner >= 0, by_rank[logical, owner.clamp(min=0)], metro.default_physical.cpu().long()
    )
    ids = ids.cpu().long()
    valid = (ids >= 0) & (ids < metro.num_logical)
    if gate is not None:
        valid &= torch.arange(ids.shape[0]).view(-1, 1) < gate
    return torch.where(valid, chosen[ids.clamp(0, metro.num_logical - 1)], -1)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("ranks", [2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize("num_logical", [32, 160, 256])
def test_cuda_kernel_matches_reference(ranks, num_logical):
    generator = torch.Generator().manual_seed(ranks * 1000 + num_logical)
    p2l, candidates = _random_placement(num_logical, ranks, max(ranks, num_logical // 2), generator)
    tables = build_metro_repair_tables(build_metro_tables(candidates.cuda(), p2l.numel(), ranks))
    host_tables = build_metro_repair_tables(build_metro_tables(candidates, p2l.numel(), ranks))
    for num_tokens in (0, 1, 3, 8, 32, 128):
        skew = torch.rand(num_logical, generator=generator) ** 4 + 1e-3
        ids = torch.multinomial(
            skew.expand(max(num_tokens, 1), -1), min(8, num_logical), generator=generator
        )
        ids = ids[:num_tokens].to(torch.int32)
        if num_tokens > 2:
            ids[-1, 0] = -1
        counts = torch.bincount(ids[ids >= 0].long(), minlength=num_logical).float()
        for moves, early_exit in ((0, False), (32, False), (32, True)):
            want = _want(ids, counts, host_tables, moves, early_exit)
            want_load, _ = metro_repair_assign_reference(
                counts > 0, host_tables, moves=moves, early_exit=early_exit
            )
            for in_dtype, out_dtype in (
                (torch.int32, torch.int64),
                (torch.int64, torch.int64),
                (torch.int32, torch.int32),
            ):
                load = torch.empty(ranks, dtype=torch.int32, device="cuda")
                got = metro_repair_route(
                    ids.to(in_dtype).cuda(),
                    tables,
                    counts=counts.cuda(),
                    moves=moves,
                    early_exit=early_exit,
                    out_dtype=out_dtype,
                    rank_load=load,
                )
                assert got.dtype == out_dtype
                assert torch.equal(got.cpu().long(), want)
                assert torch.equal(load.cpu(), want_load)
            # EP-global TopK path: the active set comes from the ids themselves.
            got = metro_repair_route(ids.cuda(), tables, moves=moves, early_exit=early_exit)
            assert torch.equal(got.cpu(), want)
        if num_tokens > 1:
            gate = torch.tensor([num_tokens // 2], dtype=torch.int32, device="cuda")
            got = metro_repair_route(
                ids.cuda(), tables, counts=counts.cuda(), num_token_non_padded=gate
            )
            assert torch.equal(
                got.cpu(), _want(ids, counts, host_tables, 32, True, num_tokens // 2)
            )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("redundant", [8, 32, 64, 96])
def test_cuda_zero_moves_is_bitwise_metro(redundant):
    generator = torch.Generator().manual_seed(redundant)
    p2l, candidates = _random_placement(256, 8, redundant, generator)
    metro = build_metro_tables(candidates.cuda(), p2l.numel(), 8)
    tables = build_metro_repair_tables(metro)
    for num_tokens in (1, 4, 64, 300):
        ids = torch.randint(-1, 256, (num_tokens, 8), generator=generator, dtype=torch.int32).cuda()
        counts = torch.bincount(ids[ids >= 0].long(), minlength=256).float()
        assert torch.equal(
            metro_repair_route(ids, tables, counts=counts, moves=0),
            metro_route(ids, counts, metro),
        )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_graph_replays_an_updated_placement():
    generator = torch.Generator().manual_seed(0)
    first_p2l, first = _random_placement(256, 8, 64, generator)
    second_p2l, second = _random_placement(256, 8, 64, generator)

    def placement(p2l, candidates):
        return PlacementSnapshot(
            layer_id=0,
            num_logical_experts=256,
            num_physical_experts=p2l.numel(),
            ep_size=8,
            num_local_physical_experts=p2l.numel() // 8,
            physical_to_logical_map=p2l.to(torch.int32).cuda(),
            logical_to_physical_candidates=candidates.cuda(),
            logical_to_physical_count=(candidates >= 0).sum(dim=1).to(torch.int32).cuda(),
        )

    router = MetroRepairL2Router(num_gpus=8)
    router.prepare_layer(placement(first_p2l, first))
    ids = torch.randint(0, 256, (16, 8), generator=generator, dtype=torch.int32).cuda()
    counts = torch.bincount(ids.flatten().long(), minlength=256).float()
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones(ids.shape, device="cuda"),
        policies=("metro_repair",),
        placement=placement(first_p2l, first),
        stage=RoutingStage.DECODE,
        global_logical_count=counts,
    )
    router.route_tokens(request)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = router.route_tokens(request).routed_physical_topk_ids

    router.prepare_layer(placement(second_p2l, second))
    graph.replay()
    tables = build_metro_repair_tables(build_metro_tables(second, second_p2l.numel(), 8))
    assert torch.equal(out.cpu(), _want(ids, counts.cpu(), tables, 32, True))
