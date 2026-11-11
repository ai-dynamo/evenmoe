# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waterfill's tile mode: the plan's and the split's properties, kernels against the host mirror, policy end to end.

Single-GPU P2P tests emulate an EP group in one process (every rank's receive
buffer on one device, each rank's launch on its own stream). The multi-GPU
test is ``tests/multigpu_waterfill_tile.py`` (torchrun).
"""

from dataclasses import replace

import _waterfill_tile_reference as ref
import pytest
import torch

from evenmoe import MoELoadBalancer, RoutingRequest
from evenmoe.core.routing_pipeline import RoutingPipeline
from evenmoe.core.types import PlacementSnapshot, RoutingPolicyConfig
from evenmoe.policies.l2.waterfill import ROUTING_POLICY, TileConfig, WaterfillL2Router

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

R, PER, K = 8, 32, 8
P = R * PER
TILES = (1, 8, 16, 24, 32, 64, 128)
NV = [8, 16, 3, 32, 0, 12, 64, 5]  # rank 4 idle, several partial tiles


def _batch(seed, n, experts=P):
    g = torch.Generator().manual_seed(seed)
    if n == 0:
        return torch.zeros(0, K, dtype=torch.int64)
    hot = torch.rand(experts, generator=g) ** 3
    return torch.stack([torch.multinomial(hot, K, generator=g) for _ in range(n)])


def _gathered(batches, nv, tiles, experts=P):
    return torch.stack([ref.count_rows(x, experts, n, t) for x, n, t in zip(batches, nv, tiles)])


def _plan(gathered, rank=0, **kw):
    return ref.plan(gathered, num_physical=P, num_local_physical=PER, rank=rank, **kw)


# ------------------------------------------------------------ the plan (CPU)


@pytest.mark.parametrize("tile", TILES)
def test_plan_is_whole_tiles_within_the_routed_maximum(tile):
    gathered = _gathered([_batch(100 + r, n) for r, n in enumerate(NV)], NV, [tile] * R)
    plan = _plan(gathered)
    quota, routed_tiles, routed_rows = plan[0].tolist(), plan[2].tolist(), plan[3].tolist()
    assert sum(quota) == sum(NV) and plan[4].tolist() == [tile] * R
    assert sum(q % tile != 0 for q in quota) <= 1  # at most one partial tile
    # While a rank has room below the routed maximum, no rank is pushed above it.
    shared_tiles = [-(-q // tile) for q in quota]
    if sum(shared_tiles) <= sum(max(routed_tiles) - t for t in routed_tiles) - R:
        assert all(t + s <= max(routed_tiles) for t, s in zip(routed_tiles, shared_tiles))
    assert len(routed_rows) == R


@pytest.mark.parametrize("tile", TILES)
def test_assignment_is_the_proportional_split_and_exact(tile):
    plan = _plan(_gathered([_batch(7 + r, n) for r, n in enumerate(NV)], NV, [tile] * R))
    quota = plan[0].tolist()
    # Independent restatement: interleave all rows round-robin by source, cut into the quotas in rank order.
    order = [(s, j) for j in range(max(NV)) for s in range(R) if j < NV[s]]
    bounds = [sum(quota[:r]) for r in range(R + 1)]
    want = {key: next(r for r in range(R) if bounds[r] <= pos < bounds[r + 1]) for pos, key in enumerate(order)}
    received, pairs = [0] * R, torch.zeros(R, R, dtype=torch.int64)
    for s in range(R):
        dest = ref.assign(plan, s, NV[s] + 3)  # 3 padding rows
        assert dest[NV[s]:].tolist() == [-1] * 3
        for j, d in enumerate(dest[: NV[s]].tolist()):
            assert d == want[(s, j)]
            received[d] += 1
            pairs[s, d] += 1
    assert received == quota and pairs.sum(1).tolist() == NV


def test_ranks_agree_on_the_largest_tile_of_ranks_with_tokens():
    tiles = [16, 24, 16, 8, 128, 16, 16, 24]  # the idle rank's 128 does not count
    gathered = _gathered([_batch(3 + r, n) for r, n in enumerate(NV)], NV, tiles)
    plans = [_plan(gathered, rank=r) for r in range(R)]
    assert all(torch.equal(p, plans[0]) for p in plans) and plans[0][4].tolist() == [24] * R
    idle = _gathered([torch.zeros(0, K, dtype=torch.int64)] * R, [0] * R, tiles)
    assert _plan(idle, rank=4)[4].tolist() == [128] * R


def test_failed_exchange_keeps_rows_local():
    plan = _plan(_gathered([_batch(r, n) for r, n in enumerate(NV)], NV, [64] * R), rank=2, status=1)
    assert plan[0].tolist() == NV and ref.assign(plan, 2, NV[2], local_only=True).tolist() == [2] * NV[2]


# --------------------------------------------------- spec, knobs, composition


def test_tile_rows_one_is_todays_waterfill(monkeypatch):
    monkeypatch.delenv("EVENMOE_WATERFILL_TILE_ROWS", raising=False)
    base = replace(ROUTING_POLICY, configure=None)
    assert ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill")) == base
    assert ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill", metadata={"tile_rows": 1})) == base
    caps = RoutingPipeline.from_value("waterfill").capabilities
    assert caps.supported_a2a_backends == ("deepep", "megamoe") and caps.supports_concurrent_microbatches
    lb = MoELoadBalancer.from_algorithm("waterfill", ep_size=R, source_rank=0, experts_per_rank=PER)
    assert lb._routing_policies["waterfill"].tile_mode is None


@cuda
@pytest.mark.parametrize("dynamic", [False, True])
def test_tile_rows_one_routes_exactly_like_the_row_kernel(dynamic):
    from evenmoe.kernels.ops.waterfill import waterfill_select

    g = torch.Generator().manual_seed(5)
    for trial in range(10):
        n = int(torch.randint(1, 200, (1,), generator=g))
        ids = torch.randint(-1, P, (n, K), generator=g).cuda()
        routed = torch.bincount(ids[ids >= 0] // PER, minlength=R).to(torch.int64)
        active = torch.randint(0, 64, (R,), generator=g).cuda() if dynamic else None
        router = WaterfillL2Router(source_rank=trial % R, world_size=R, experts_per_rank=PER, tile=TileConfig(1))
        req = RoutingRequest(
            layer_id=0,
            logical_topk_ids=ids,
            topk_weights=torch.rand(n, K, generator=g).cuda(),
            routed_physical_topk_ids=ids,
            routed_rank_load=routed,
            active_rank_token_count=active,
            metadata={"tile_rows": 64},  # ignored without tile mode
        )
        dec = router.route_tokens(req, RoutingPolicyConfig("waterfill"))
        want = waterfill_select(
            rank_load=routed if active is None else routed + active,
            routed_load=routed,
            topk_ids=ids,
            allow_all_ranks=active is None,
            source_rank=trial % R,
            local_preference=1.1,
            world_size=R,
            experts_per_rank=PER,
        )
        assert torch.equal(dec.shared_expert_rank, want)


def test_tile_mode_capabilities_knobs_and_composition(monkeypatch):
    monkeypatch.setenv("EVENMOE_WATERFILL_TILE_ROWS", "24")
    monkeypatch.setenv("EVENMOE_WATERFILL_TILE_RHO", "1.25")
    spec = ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill"))
    assert spec.rank_load is None and spec.requires_placement_state  # so the framework calls prepare_layer
    assert RoutingPipeline.from_value("static+waterfill").capabilities.requires_placement_state
    assert spec.supported_a2a_backends == ("deepep", "megamoe") and not spec.supports_concurrent_microbatches
    for algorithm in ("static+waterfill", "lplb+waterfill"):
        RoutingPipeline.from_value(algorithm)
    with pytest.raises(ValueError, match="cannot be combined with"):
        RoutingPipeline.from_value("ultraep+waterfill")
    lb = MoELoadBalancer.from_algorithm("waterfill", ep_size=R, source_rank=1, experts_per_rank=PER)
    assert lb._routing_policies["waterfill"].tile_mode.config == TileConfig(24, 1.25)
    assert ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill", metadata={"tile_rows": 1})).rank_load == "local"


def test_tile_mode_config_errors(monkeypatch):
    tiled = {"tile_rows": 64}
    monkeypatch.setenv("WATERFILL_USE_GLOBAL_LOAD", "1")
    with pytest.raises(ValueError, match="WATERFILL_USE_GLOBAL_LOAD"):
        ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill", metadata=tiled))
    monkeypatch.delenv("WATERFILL_USE_GLOBAL_LOAD")
    with pytest.raises(ValueError, match="stages"):
        ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill", enabled_stages=frozenset({"decode"}), metadata=tiled))
    for bad in (0, -8, "x", 2.5, True):
        with pytest.raises(ValueError):
            ROUTING_POLICY.resolve(RoutingPolicyConfig("waterfill", metadata={"tile_rows": bad}))
    router = WaterfillL2Router(source_rank=0, world_size=R, experts_per_rank=PER, tile=TileConfig(64))
    for bad in (0, 2.0, True):
        req = RoutingRequest(layer_id=0, logical_topk_ids=None, topk_weights=None, routed_physical_topk_ids=torch.zeros(1, K, dtype=torch.int32), metadata={"tile_rows": bad})
        with pytest.raises(ValueError, match="tile_rows"):
            router.route_tokens(req, RoutingPolicyConfig("waterfill"))


# ------------------------------------------------------------- CUDA kernels


@cuda
@pytest.mark.parametrize("tile", TILES)
def test_kernels_match_the_host_mirror(tile):
    from evenmoe.kernels.tile_waterfill import count_and_exchange, plan_and_assign

    for n, id_dtype in ((0, torch.int32), (37, torch.int64), (1100, torch.int32)):
        ids = torch.randint(-1, P + 3, (n, K), generator=torch.Generator().manual_seed(n)).to(id_dtype)  # padding, out of range
        for token_count in (None, max(n - 3, 0), torch.tensor([max(n - 5, 0)], dtype=torch.int32)):
            tc = token_count.cuda() if isinstance(token_count, torch.Tensor) else token_count
            row, status = count_and_exchange(ids.cuda(), P, tc, tile)
            assert status is None and torch.equal(row.cpu(), ref.count_rows(ids, P, token_count, tile))
    for seed, ranks in ((0, 8), (1, 8), (2, 4)):  # seed 0: all ranks idle but one
        per = PER if ranks == 8 else 16
        g = torch.Generator().manual_seed(seed)
        nv = [0] * (ranks - 1) + [37] if seed == 0 else torch.randint(0, 200, (ranks,), generator=g).tolist()
        tiles = [tile if r % 2 else max(1, tile // 2) for r in range(ranks)]  # ranks may disagree
        gathered = _gathered([_batch(seed * 31 + r, nv[r], ranks * per) for r in range(ranks)], nv, tiles, ranks * per)
        for flag in (0, 1):
            status = torch.tensor([flag], dtype=torch.int32, device="cuda")
            for rank in range(ranks):
                want = ref.plan(gathered, num_physical=ranks * per, num_local_physical=per, rank=rank, status=flag)
                shared = plan_and_assign(gathered.cuda(), num_local_physical=per, rank=rank, num_tokens=nv[rank] + 2, rho=1.5, status=status)
                assert torch.equal(shared.cpu(), ref.assign(want, rank, nv[rank] + 2, local_only=flag != 0))


@cuda
@pytest.mark.parametrize("tile", TILES)
def test_p2p_emulated_group_gives_the_quotas_on_every_rank(tile):
    from evenmoe.kernels.tile_waterfill import count_and_exchange, payload_width, plan_and_assign

    exchanges = ref.local_peer_exchanges(R, payload_width(P))
    streams = [torch.cuda.Stream() for _ in range(R)]
    for it in range(2):
        xs = [_batch(10 * it + r, NV[r]).to(torch.int32).cuda() for r in range(R)]
        outs = [None] * R
        torch.cuda.synchronize()
        for r in range(R):
            with torch.cuda.stream(streams[r]):
                outs[r] = count_and_exchange(xs[r], P, NV[r], tile, exchanges[r])
        torch.cuda.synchronize()
        want = _gathered([x.cpu() for x in xs], NV, [tile] * R)
        received = torch.zeros(R, dtype=torch.int64)
        for r in range(R):
            rows, status = outs[r]
            assert torch.equal(rows.cpu(), want) and int(status.item()) == 0
            shared = plan_and_assign(rows, num_local_physical=PER, rank=r, num_tokens=NV[r], rho=1.5, status=status)
            received += torch.bincount(shared.cpu(), minlength=R)
        assert received.tolist() == _plan(want)[0].tolist()  # every rank's split adds up to the same quotas


@cuda
def test_a_rank_that_keeps_skipping_calls_cannot_stall_the_others():
    from evenmoe.kernels.tile_waterfill import count_and_exchange, payload_width

    exchanges = ref.local_peer_exchanges(R, payload_width(P), timeout_ms=5.0)
    x = _batch(1, 4).to(torch.int32).cuda()
    for call in range(9):  # rank 0 never calls
        statuses = []
        for r in range(1, R):
            _, status = count_and_exchange(x, P, 4, 64, exchanges[r])
            statuses.append(int(status.item()))
        if call < 8:
            assert all(st & 1 for st in statuses)  # timed out, rows kept local
        else:
            assert statuses == [8] * (R - 1)  # stopped exchanging: no more waiting


class _Coll:
    """All-reduce stand-in: every rank's zero-padded row was posted already."""

    world_size = R

    def __init__(self, rank, full):
        self.rank, self.full, self.calls = rank, full, 0

    def all_reduce_sum(self, x):
        assert torch.equal(x[self.rank].cpu(), self.full[self.rank])
        self.calls += 1
        return self.full.clone().to(x.device)


def _snapshot(device):
    p2l = torch.randperm(P, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(p2l).to(torch.int32)
    return PlacementSnapshot(
        layer_id=0, num_logical_experts=P, num_physical_experts=P, ep_size=R, num_local_physical_experts=PER,
        physical_to_logical_map=p2l.to(device), logical_to_physical_candidates=inv[:, None].to(torch.int64).to(device),
        logical_to_physical_count=torch.ones(P, dtype=torch.int64, device=device), default_physical_for_logical=inv.to(device),
    )


@cuda
def test_policy_through_the_orchestrator_and_the_adapter_including_an_idle_rank():
    from evenmoe.adapters.sglang.routing import to_routing_request, to_sglang_routing_output

    dev, tile = torch.device("cuda"), 24
    snap = _snapshot(dev)
    inv = snap.default_physical_for_logical.cpu().to(torch.int64)
    logical = [_batch(11 + r, n) for r, n in enumerate(NV)]
    full = _gathered([inv[x] for x in logical], NV, [tile] * R).to(torch.float32)
    config = RoutingPolicyConfig("waterfill", metadata={"tile_rows": 64})
    for s in range(R):  # rank 4 routes an empty batch and still joins the exchange
        coll = _Coll(s, full)
        lb = MoELoadBalancer.from_algorithm(RoutingPipeline.from_value(["static", config]), ep_size=R, source_rank=s, experts_per_rank=PER, collectives=coll, ep_collectives=coll)
        x = torch.cat([logical[s], torch.full((2, K), -1, dtype=torch.int64)]).to(torch.int32).to(dev)  # 2 padded rows
        req = to_routing_request(
            layer_id=0, logical_topk_ids=x, topk_weights=torch.rand(x.shape[0], K, device=dev), placement=snap,
            token_count=torch.tensor([NV[s]], dtype=torch.int32, device=dev), routed_scaling_factor=2.5,
        )
        req = replace(req, metadata={**req.metadata, "tile_rows": tile})  # the framework's kernel tile for this call
        dec = lb.route_tokens(req)
        want = ref.assign(_plan(full.to(torch.int32), rank=s), s, x.shape[0])
        assert coll.calls == 1 and torch.equal(dec.shared_expert_rank.cpu().to(torch.int64), want)
        out = to_sglang_routing_output(dec, num_routed_experts=P, world_size=R, routed_scaling_factor=2.5)
        assert torch.equal(out.topk_ids[: NV[s], K].cpu().to(torch.int64), want[: NV[s]] * (PER + 1) + PER)
