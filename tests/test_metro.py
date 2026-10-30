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
from evenmoe.kernels.ops.metro import (
    metro_assign_reference,
    metro_route,
    metro_workspace,
)
from evenmoe.policies.l2.metro import MetroL2Router, MetroTables, build_metro_tables

# Four logical experts on two ranks of three slots: logical 0 and 1 are held by
# both ranks, logical 2 only by rank 0 and logical 3 only by rank 1.
_P2L = [0, 1, 2, 3, 0, 1]
_CANDIDATES = [[0, 4], [1, 5], [2, -1], [3, -1]]
_SWAPPED_P2L = [0, 1, 2, 3, 1, 0]
_SWAPPED_CANDIDATES = [[0, 5], [1, 4], [2, -1], [3, -1]]


class _RecordingCollectives:
    rank = 0
    world_size = 2

    def __init__(self, peer_contribution):
        self.peer_contribution = peer_contribution
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        return payload + self.peer_contribution.to(payload)


def _placement(p2l=_P2L, candidates=_CANDIDATES, *, layer_id=0, device="cpu", default=True):
    candidates = torch.tensor(candidates, dtype=torch.int32, device=device)
    num_physical = len(p2l)
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=candidates.shape[0],
        num_physical_experts=num_physical,
        ep_size=2,
        num_local_physical_experts=num_physical // 2,
        physical_to_logical_map=torch.tensor(p2l, dtype=torch.int32, device=device),
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=(candidates >= 0).sum(dim=1).to(torch.int32),
        default_physical_for_logical=(
            torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device) if default else None
        ),
    )


def _reference(counts, tables: MetroTables):
    return metro_assign_reference(
        counts,
        tables.default_physical,
        tables.single_rank,
        tables.rep_logical,
        tables.rep_phys,
        tables.num_replicated,
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


def _balancer(peer_contribution):
    collectives = _RecordingCollectives(torch.tensor(peer_contribution, dtype=torch.float32))
    balancer = MoELoadBalancer.from_algorithm(
        "metro", ep_size=2, source_rank=0, experts_per_rank=3, collectives=collectives
    )
    return balancer, collectives


@pytest.mark.unit
def test_tables_split_single_and_replicated_experts():
    tables = build_metro_tables(torch.tensor(_P2L), 4, 2)

    assert tables.default_physical.tolist() == [0, 1, 2, 3]
    assert tables.single_rank.tolist() == [-1, -1, 0, 1]
    assert tables.capacity == 2
    assert tables.num_replicated.tolist() == [2]
    assert tables.rep_logical.tolist() == [0, 1]
    assert tables.rep_mask.tolist() == [[3], [3]]
    assert tables.rep_phys.tolist() == [[0, 4], [1, 5]]


@pytest.mark.unit
def test_same_rank_copies_collapse_to_the_lowest_slot():
    # Rank 0 holds slots 0-2 = [1, 0, 0], rank 1 holds slots 3-5 = [2, 1, 1].
    tables = build_metro_tables(torch.tensor([1, 0, 0, 2, 1, 1]), 3, 2)

    assert tables.default_physical.tolist() == [1, 0, 3]
    assert tables.single_rank.tolist() == [0, -1, 1]
    assert tables.num_replicated.tolist() == [1]
    assert tables.rep_logical[:1].tolist() == [1]
    assert tables.rep_phys[:1].tolist() == [[0, 4]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("counts", "chosen"),
    [
        # Fixed load [1, 1]; logical 0 ties to rank 0, logical 1 then goes to rank 1.
        ([1.0, 1.0, 1.0, 1.0], [0, 5, 2, 3]),
        # Fixed load [1, 0]; logical 0 goes to rank 1, logical 1 ties to rank 0.
        ([1.0, 2.0, 3.0, 0.0], [4, 1, 2, 3]),
    ],
)
def test_reference_greedy(counts, chosen):
    tables = build_metro_tables(torch.tensor(_P2L), 4, 2)

    assert _reference(torch.tensor(counts), tables).tolist() == chosen


@pytest.mark.unit
def test_every_rank_builds_the_same_tables_from_its_own_candidate_view():
    """Frameworks may narrow each logical expert's candidates to the copy on
    their own rank; the tables, and so the assignment, must not change."""

    generator = torch.Generator().manual_seed(0)
    p2l, candidates = _random_placement(32, 4, 16, generator, "cpu")
    num_physical = p2l.numel()
    num_local = num_physical // 4
    counts = (torch.rand(32, generator=generator) > 0.3).float()
    ids = torch.randint(0, 32, (8, 4), generator=generator, dtype=torch.int32)
    views, tables, routed = [], [], []
    for rank in range(4):
        view = candidates.clone()
        for logical in range(32):
            held = view[logical][view[logical] >= 0]
            local = held[torch.div(held, num_local, rounding_mode="floor") == rank]
            if local.numel():
                view[logical] = -1
                view[logical, 0] = local[0]
        views.append(view)
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=32,
            num_physical_experts=num_physical,
            ep_size=4,
            num_local_physical_experts=num_local,
            physical_to_logical_map=p2l,
            logical_to_physical_candidates=view,
            logical_to_physical_count=(view >= 0).sum(dim=1).to(torch.int32),
            default_physical_for_logical=view[:, 0],
        )
        router = MetroL2Router(num_gpus=4)
        router.prepare_layer(placement)
        tables.append(router._layer_state[0].tables)
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=ids,
            topk_weights=torch.ones(ids.shape),
            policies=("metro",),
            placement=placement,
            stage=RoutingStage.DECODE,
            global_logical_count=counts,
        )
        routed.append(router.route_tokens(request).routed_physical_topk_ids)

    assert any(not torch.equal(views[0], view) for view in views[1:])
    for other in tables[1:]:
        assert all(torch.equal(a, b) for a, b in zip(tables[0].tensors(), other.tensors()))
    assert all(torch.equal(routed[0], other) for other in routed[1:])


@pytest.mark.unit
def test_candidates_that_disagree_with_the_physical_map_are_rejected():
    router = MetroL2Router(num_gpus=2)

    with pytest.raises(ValueError, match="disagrees"):
        router.prepare_layer(_placement(_P2L, _SWAPPED_CANDIDATES))


@pytest.mark.unit
@pytest.mark.parametrize("num_ranks", [33, 72, 128])
def test_rank_masks_span_several_words(num_ranks):
    generator = torch.Generator().manual_seed(num_ranks)
    p2l, _ = _random_placement(256, num_ranks, 2 * num_ranks, generator, "cpu")
    tables = build_metro_tables(p2l, 256, num_ranks)

    count = int(tables.num_replicated.item())
    assert count > 0
    assert tables.rep_mask.shape == (tables.capacity, (num_ranks + 31) // 32)
    for words, row in zip(tables.rep_mask[:count].tolist(), tables.rep_phys[:count].tolist()):
        held = [g for g in range(num_ranks) if (words[g // 32] >> (g % 32)) & 1]
        assert held == [g for g in range(num_ranks) if row[g] >= 0]


@pytest.mark.unit
def test_more_than_128_ranks_are_rejected():
    with pytest.raises(ValueError, match="1..128"):
        build_metro_tables(torch.arange(129) % 64, 64, 129)


@pytest.mark.unit
def test_a_mis_sized_workspace_is_rejected():
    tables = build_metro_tables(torch.tensor(_P2L), 4, 2)
    workspace = metro_workspace(4, 2, tables.capacity, "cpu")

    with pytest.raises(ValueError, match="workspace"):
        metro_route(
            torch.tensor([[0, 1]]),
            torch.ones(4),
            **tables.operands(),
            workspace=workspace[1:],
        )
    with pytest.raises(ValueError, match="workspace"):
        metro_route(
            torch.tensor([[0, 1]]),
            torch.ones(4),
            **tables.operands(),
            workspace=workspace.to(torch.int64),
        )


@pytest.mark.unit
def test_decode_routes_from_the_global_count():
    balancer, collectives = _balancer([0.0, 0.0, 0.0, 1.0])
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]]))

    assert collectives.payloads[0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert decision.routed_physical_topk_ids.dtype == torch.int64
    assert decision.routed_physical_topk_ids.tolist() == [[0, 5], [2, -1]]
    assert decision.metadata["metro_assignment"] is True
    assert decision.metadata["executed_policies"] == ("metro",)


@pytest.mark.unit
def test_inactive_experts_do_not_load_their_rank():
    balancer, _ = _balancer([0.0, 0.0, 0.0, 0.0])
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]]))

    assert decision.routed_physical_topk_ids.tolist() == [[4, 1], [2, -1]]


@pytest.mark.unit
def test_padding_rows_are_dropped():
    balancer, _ = _balancer([0.0, 0.0, 0.0, 1.0])
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(
        _request([[0, 1], [2, 3]], token_count=torch.tensor(1, dtype=torch.int32))
    )

    assert decision.routed_physical_topk_ids.tolist() == [[0, 5], [-1, -1]]


@pytest.mark.unit
@pytest.mark.parametrize("stage", [RoutingStage.PREFILL, RoutingStage.MIXED])
def test_prefill_uses_the_static_map_but_joins_the_collective(stage):
    balancer, collectives = _balancer([0.0, 0.0, 0.0, 1.0])
    balancer.on_placement_committed(_placement())

    decision = balancer.route_tokens(_request([[0, 1], [2, -1]], stage=stage))

    assert len(collectives.payloads) == 1
    assert decision.routed_physical_topk_ids.tolist() == [[0, 1], [2, -1]]
    assert decision.metadata["metro_assignment"] is False


@pytest.mark.unit
def test_static_steps_require_the_framework_default_map():
    balancer, _ = _balancer([0.0, 0.0, 0.0, 1.0])
    placement = _placement(default=False)
    balancer.on_placement_committed(placement)

    with pytest.raises(ValueError, match="default_physical_for_logical"):
        balancer.route_tokens(_request([[0, 1]], placement=placement, stage=RoutingStage.PREFILL))


@pytest.mark.unit
def test_layer_without_replicas_skips_the_collective():
    balancer, collectives = _balancer([0.0, 0.0, 0.0, 0.0])
    placement = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.arange(4, dtype=torch.int32),
        logical_to_physical_candidates=torch.arange(4, dtype=torch.int32).view(4, 1),
        logical_to_physical_count=torch.ones(4, dtype=torch.int32),
        default_physical_for_logical=torch.arange(4, dtype=torch.int32),
    )
    balancer.on_placement_committed(placement)

    decision = balancer.route_tokens(_request([[0, 3]], placement=placement))

    assert collectives.payloads == []
    assert decision.routed_physical_topk_ids.tolist() == [[0, 3]]


@pytest.mark.unit
def test_recommit_updates_tables_in_place():
    router = MetroL2Router(num_gpus=2)
    router.prepare_layer(_placement())
    before = [t.data_ptr() for t in router._layer_state[0].tables.tensors()]

    router.prepare_layer(_placement(_SWAPPED_P2L, _SWAPPED_CANDIDATES))
    request = replace(
        _request([[0, 1]], placement=_placement(_SWAPPED_P2L, _SWAPPED_CANDIDATES)),
        global_logical_count=torch.ones(4),
        policies=("metro",),
    )

    assert [t.data_ptr() for t in router._layer_state[0].tables.tensors()] == before
    assert router.route_tokens(request).routed_physical_topk_ids.tolist() == [[0, 4]]


@pytest.mark.unit
def test_pipeline_declarations():
    capabilities = RoutingPipeline.from_value("metro").capabilities

    assert capabilities.consumes_global_logical_count
    assert capabilities.requires_rank_dispatch_map
    assert capabilities.requires_deepep
    assert not capabilities.supports_concurrent_microbatches
    assert not capabilities.dispatch_fixed_by_placement
    assert capabilities.graph_stability == "stable"
    with pytest.raises(ValueError):
        RoutingPipeline.from_value("metro+lplb")


def _random_placement(num_logical, num_ranks, redundant, generator, device):
    """Each expert once round-robin over the ranks, the rest of the slots random.

    Returns the ``[P]`` physical-to-logical map and the full candidate table.
    """

    per_rank = -(-(num_logical + redundant) // num_ranks)
    homes = [list(range(rank, num_logical, num_ranks)) for rank in range(num_ranks)]
    slots = []
    for home in homes:
        extra = per_rank - len(home)
        slots += home + torch.randint(0, num_logical, (extra,), generator=generator).tolist()
    p2l = torch.tensor(slots, dtype=torch.int32)
    width = int(torch.bincount(p2l.long(), minlength=num_logical).max())
    candidates = torch.full((num_logical, width), -1, dtype=torch.int32)
    for logical in range(num_logical):
        held = torch.nonzero(p2l == logical).flatten()
        candidates[logical, : held.numel()] = held.to(torch.int32)
    return p2l.to(device), candidates.to(device)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    ("num_ranks", "redundant"),
    [(8, 8), (8, 32), (8, 64), (8, 96), (64, 64), (72, 144), (128, 128)],
)
def test_cuda_kernel_matches_reference(num_ranks, redundant):
    generator = torch.Generator().manual_seed(num_ranks * 1000 + redundant)
    p2l, _ = _random_placement(256, num_ranks, redundant, generator, "cuda")
    tables = build_metro_tables(p2l, 256, num_ranks)
    for num_tokens in (0, 1, 4, 64, 300):
        ids = torch.randint(-1, 256, (num_tokens, 8), generator=generator, dtype=torch.int32)
        counts = torch.bincount(ids[ids >= 0].long(), minlength=256).float()
        counts *= torch.rand(256, generator=generator) > 0.3
        expected = _reference(counts, tables)
        valid = ids >= 0
        want = torch.where(valid, expected[ids.clamp(min=0).long()], -1)
        for out_dtype in (torch.int64, torch.int32):
            got = metro_route(ids.cuda(), counts.cuda(), **tables.operands(), out_dtype=out_dtype)
            assert got.dtype == out_dtype
            assert torch.equal(got.cpu().long(), want)
        if num_tokens:
            gate = torch.tensor(num_tokens // 2, dtype=torch.int32, device="cuda")
            got = metro_route(
                ids.cuda(), counts.cuda(), **tables.operands(), num_token_non_padded=gate
            )
            want_gated = want.clone()
            want_gated[num_tokens // 2 :] = -1
            assert torch.equal(got.cpu(), want_gated)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_graph_replays_an_updated_placement():
    generator = torch.Generator().manual_seed(0)
    first, first_candidates = _random_placement(256, 8, 64, generator, "cuda")
    second, second_candidates = _random_placement(256, 8, 64, generator, "cuda")
    num_physical = first.numel()

    def placement(p2l, candidates):
        return PlacementSnapshot(
            layer_id=0,
            num_logical_experts=256,
            num_physical_experts=num_physical,
            ep_size=8,
            num_local_physical_experts=num_physical // 8,
            physical_to_logical_map=p2l,
            logical_to_physical_candidates=candidates,
            logical_to_physical_count=(candidates >= 0).sum(dim=1).to(torch.int32),
        )

    router = MetroL2Router(num_gpus=8)
    router.prepare_layer(placement(first, first_candidates))
    ids = torch.randint(0, 256, (16, 8), generator=generator, dtype=torch.int32).cuda()
    counts = torch.bincount(ids.flatten().long(), minlength=256).float()
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones(ids.shape, device="cuda"),
        policies=("metro",),
        placement=placement(first, first_candidates),
        stage=RoutingStage.DECODE,
        global_logical_count=counts,
    )
    router.route_tokens(request)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = router.route_tokens(request).routed_physical_topk_ids

    router.prepare_layer(placement(second, second_candidates))
    graph.replay()
    want = _reference(counts, build_metro_tables(second, 256, 8))[ids.long().cpu()]
    assert torch.equal(out.cpu(), want)
