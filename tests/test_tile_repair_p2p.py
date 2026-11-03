# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tile_repair's fused NVLink P2P count exchange.

Single-GPU tests emulate an EP group in one process: every rank's receive
buffer lives on one device (:func:`local_peer_exchanges`) and each rank's
launch runs on its own stream, so the spinning one-CTA launches of one layer
run concurrently. ``tests/multigpu_tile_repair_p2p.py`` runs the real
symmetric-memory path across processes.
"""

import os
import subprocess
import sys

# The emulated ranks' launches run on separate streams and must not share a
# hardware work queue (a shared queue serialises them and the exchange would
# time out). Read when the CUDA context is created, after collection.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import pytest  # noqa: E402
import torch
from test_tile_repair import (
    _check_ep_routing,
    _ep_batch,
    _random_p2l,
    _snapshot,
)

from evenmoe import MoELoadBalancer, RoutingRequest, RoutingStage
from evenmoe.core.policy import RouterBuildContext
from evenmoe.kernels.tile_repair import (
    MAX_COPIES,
    NUM_STATS,
    P2P_FAIL_OVERFLOW,
    P2P_FAIL_OVERWRITTEN,
    P2P_FAIL_TIMEOUT,
    adaptive_block_m,
    build_tile_repair_tables,
    tile_repair_plan_reference,
    tile_repair_route,
    tile_repair_route_reference,
)
from evenmoe.kernels.tile_repair import p2p as p2p_module
from evenmoe.kernels.tile_repair.p2p import (
    local_peer_exchanges,
    resolve_timeout_ms,
    tile_repair_route_p2p,
)
from evenmoe.policies.l2 import tile_repair as tile_repair_policy
from evenmoe.policies.l2.tile_repair import TileRepairL2Router
from evenmoe.policies.tile_common import resolve_block_m, resolve_p2p

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


# --------------------------------------------------------------------------- host


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


def test_p2p_and_block_m_knobs(monkeypatch):
    monkeypatch.delenv("EVENMOE_TILE_P2P", raising=False)
    assert resolve_p2p() is False
    monkeypatch.setenv("EVENMOE_TILE_P2P", "1")
    assert resolve_p2p() is True
    assert resolve_p2p({"p2p": False}) is False
    monkeypatch.setenv("EVENMOE_TILE_P2P", "maybe")
    with pytest.raises(ValueError):
        resolve_p2p()
    monkeypatch.setenv("EVENMOE_TILE_BLOCK_M", "0")
    assert resolve_block_m() == 0
    monkeypatch.setenv("EVENMOE_TILE_BLOCK_M", "-1")
    with pytest.raises(ValueError):
        resolve_block_m()
    monkeypatch.setenv("EVENMOE_TILE_P2P_TIMEOUT_MS", "50")
    assert resolve_timeout_ms() == 50.0
    with pytest.raises(ValueError):
        resolve_timeout_ms(0)


class _GroupCollectives:
    def __init__(self, rank, world_size, group):
        self.rank = rank
        self.world_size = world_size
        self.device_group = group

    def all_reduce_sum(self, payload):  # pragma: no cover - not reached
        raise AssertionError("the P2P path must not use the all-reduce")


def test_build_reads_the_group_from_collectives_and_falls_back(monkeypatch):
    seen = []

    def fake_create(group, **kwargs):
        seen.append(group)
        return None  # e.g. multi-node: every rank keeps the all-reduce path

    monkeypatch.setattr(tile_repair_policy, "create_peer_exchange", fake_create)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ranks, num_logical = 4, 32
    p2l = _random_p2l(num_logical, ranks, 2, seed=5)
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks)
    sentinel = object()
    mlb = MoELoadBalancer.from_algorithm(
        "tile_repair",
        ep_size=ranks,
        source_rank=1,
        experts_per_rank=1,
        collectives=_GroupCollectives(1, ranks, sentinel),
        tile_p2p=True,
    )
    mlb.on_placement_committed(placement)
    router = mlb._routing_policies["tile_repair"]
    assert seen == [sentinel]
    assert router.p2p_exchange is None
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.zeros(2, 8, dtype=torch.int64),
        topk_weights=torch.ones(2, 8),
        placement=placement,
        stage=RoutingStage.DECODE,
    )
    assert router.needs_global_logical_count(request) is True  # all-reduce path
    router.prepare_layer(placement)
    assert seen == [sentinel]  # set up once


def test_p2p_without_a_group_keeps_the_all_reduce_path():
    context = RouterBuildContext(ep_size=4, source_rank=0, experts_per_rank=1, options={"tile_p2p": True})
    router = tile_repair_policy._build_router(context)
    assert router.p2p_requested
    p2l = _random_p2l(32, 4, 2, seed=6)
    router.prepare_layer(_snapshot(p2l, num_logical=32, ranks=4))
    assert router.p2p_exchange is None


# --------------------------------------------------------------------------- CUDA


def _cuda_tables(p2l, num_logical, ranks):
    return build_tile_repair_tables(
        torch.tensor(p2l),
        num_logical=num_logical,
        num_ranks=ranks,
        num_local_physical_experts=len(p2l) // ranks,
        device="cuda",
    )


def _launch_all(exchanges, ids, tables, *, block_m, max_moves=32, num_slots=None, ranks_to_run=None,
                stall_rank=None, stall_cycles=0, with_plan=True, streams=None):
    """One layer: every (selected) rank's P2P launch on its own stream."""

    ranks = len(exchanges)
    ranks_to_run = range(ranks) if ranks_to_run is None else ranks_to_run
    streams = streams or [torch.cuda.Stream() for _ in range(ranks)]
    outputs, plans, stats = {}, {}, {}
    num_logical = tables.num_logical
    inputs = {r: ids[r].cuda() for r in ranks_to_run}
    torch.cuda.synchronize()
    for r in ranks_to_run:
        with torch.cuda.stream(streams[r]):
            if r == stall_rank:
                torch.cuda._sleep(stall_cycles)
            plans[r] = torch.full((num_logical * MAX_COPIES,), -7, dtype=torch.int32, device="cuda")
            stats[r] = torch.full((NUM_STATS + ranks,), -7, dtype=torch.int32, device="cuda")
            outputs[r] = tile_repair_route_p2p(
                inputs[r],
                tables,
                exchanges[r],
                block_m=block_m,
                max_moves=max_moves,
                num_slots=num_slots,
                plan_out=plans[r] if with_plan else None,
                stats_out=stats[r],
            )
    torch.cuda.synchronize()
    return outputs, plans, stats


def _primary_only(ids, tables):
    primary = tables.copy_physical[:, 0].cpu().to(torch.int64)
    valid = (ids >= 0) & (ids < tables.num_logical)
    out = torch.full_like(ids, -1)
    out[valid] = primary[ids[valid]]
    return out


@cuda
@pytest.mark.parametrize("block_m", [16, 64, 0])
@pytest.mark.parametrize(
    "ranks,num_logical,extras", [(2, 16, 2), (4, 64, 4), (8, 256, 4), (8, 256, 32), (16, 512, 8)]
)
def test_p2p_matches_the_reference(block_m, ranks, num_logical, extras):
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    for seed in range(3):
        p2l = _random_p2l(num_logical, ranks, extras, seed=seed + 200)
        tables = _cuda_tables(p2l, num_logical, ranks)
        tokens = [0 if r == 1 else 48 for r in range(ranks)] if seed == 0 else None
        ids, counts = _ep_batch(ranks, num_logical, seed=seed + 40, tokens=tokens, pad=seed % 2)
        num_slots = len(p2l)
        plan = tile_repair_plan_reference(
            counts, tables, block_m=block_m, max_moves=32, num_slots=num_slots
        )
        outputs, plans, stats = _launch_all(exchanges, ids, tables, block_m=block_m, num_slots=num_slots)
        for r in range(ranks):
            expected = tile_repair_route_reference(
                ids[r], counts, tables, source_rank=r, block_m=block_m, max_moves=32, plan=plan
            )
            assert torch.equal(outputs[r].cpu(), expected)
            assert plans[r].view(num_logical, MAX_COPIES).cpu().tolist() == plan.copy_tokens.tolist()
            assert stats[r].cpu().tolist() == [
                plan.moves, plan.lower_bound, plan.primary_peak, plan.peak, plan.block_m, 0,
                *plan.rank_tiles.tolist(),
            ]
            # Identical to the gathered-count kernel.
            gathered = tile_repair_route(
                ids[r].cuda(), counts.cuda(), tables, source_rank=r, block_m=block_m,
                num_slots=num_slots,
            )
            assert torch.equal(outputs[r], gathered)
        _check_ep_routing(
            ids, counts, tables, p2l, block_m=plan.block_m, max_moves=32,
            outputs=[outputs[r] for r in range(ranks)],
        )
    assert all(int(x.epoch.item()) == 3 for x in exchanges)
    assert all(x.failure_count() == 0 for x in exchanges)


@cuda
@pytest.mark.parametrize("in_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_p2p_dtypes_and_large_batches(in_dtype, out_dtype):
    ranks, num_logical = 4, 256
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    p2l = _random_p2l(num_logical, ranks, 8, seed=12)
    tables = _cuda_tables(p2l, num_logical, ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=5, tokens=[700, 0, 3, 1100], pad=2)
    streams = [torch.cuda.Stream() for _ in range(ranks)]
    outs = []
    for r in range(ranks):
        with torch.cuda.stream(streams[r]):
            outs.append(
                tile_repair_route_p2p(
                    ids[r].to(in_dtype).cuda(), tables, exchanges[r], block_m=32, out_dtype=out_dtype
                )
            )
    torch.cuda.synchronize()
    for r in range(ranks):
        assert outs[r].dtype == out_dtype and outs[r].shape == ids[r].shape
        expected = tile_repair_route_reference(
            ids[r], counts, tables, source_rank=r, block_m=32, max_moves=32
        )
        assert torch.equal(outs[r].cpu().to(torch.int64), expected)


@cuda
def test_p2p_zero_moves_routes_primary_only_but_exchanges():
    ranks, num_logical = 4, 64
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    p2l = _random_p2l(num_logical, ranks, 4, seed=8)
    tables = _cuda_tables(p2l, num_logical, ranks)
    ids, _ = _ep_batch(ranks, num_logical, seed=9, tokens=[40] * ranks)
    outputs, _, stats = _launch_all(exchanges, ids, tables, block_m=16, max_moves=0)
    for r in range(ranks):
        assert torch.equal(outputs[r].cpu(), _primary_only(ids[r], tables))
        assert int(stats[r][5]) == 0
    assert all(int(x.epoch.item()) == 1 for x in exchanges)


@cuda
def test_p2p_epoch_wraps():
    ranks, num_logical = 4, 64
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    # Fast-forward to just before the 16-bit wrap: counters at 65531 and the
    # receive slots holding the words of epochs 65530 (parity 0) and 65531.
    buffers = exchanges[0]._keepalive[0].view(ranks, 2, ranks * num_logical)
    for parity, epoch in ((0, 65530), (1, 65531)):
        buffers[:, parity] = (epoch << 16) - (1 << 32)  # as int32
    for x in exchanges:
        x.epoch.fill_(65531)
    p2l = _random_p2l(num_logical, ranks, 4, seed=10)
    tables = _cuda_tables(p2l, num_logical, ranks)
    for step in range(9):  # epochs 65532..65535, 0, 1, ... across the 16-bit wrap
        ids, counts = _ep_batch(ranks, num_logical, seed=300 + step)
        outputs, _, stats = _launch_all(exchanges, ids, tables, block_m=8, with_plan=False)
        for r in range(ranks):
            expected = tile_repair_route_reference(
                ids[r], counts, tables, source_rank=r, block_m=8, max_moves=32
            )
            assert torch.equal(outputs[r].cpu(), expected), step
            assert int(stats[r][5]) == 0
    assert [int(x.epoch.item()) for x in exchanges] == [(65531 + 9) & 0xFFFF] * ranks


@cuda
def test_p2p_late_peer_times_out_to_primary_only_and_recovers():
    ranks, num_logical = 4, 64
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical, timeout_ms=20)
    p2l = _random_p2l(num_logical, ranks, 4, seed=14)
    tables = _cuda_tables(p2l, num_logical, ranks)
    ids, counts = _ep_batch(ranks, num_logical, seed=15, tokens=[32] * ranks)
    # Rank 3 stalls ~200 ms before its launch (e.g. a first-use JIT): the others
    # give up after 20 ms and route primary-only; rank 3 then finds every
    # peer's word of this call and plans normally.
    outputs, _, stats = _launch_all(
        exchanges, ids, tables, block_m=16, stall_rank=3, stall_cycles=400_000_000
    )
    for r in range(3):
        assert torch.equal(outputs[r].cpu(), _primary_only(ids[r], tables))
        assert int(stats[r][5]) == P2P_FAIL_TIMEOUT
        assert exchanges[r].failure_count() == 1
    expected = tile_repair_route_reference(ids[3], counts, tables, source_rank=3, block_m=16, max_moves=32)
    assert torch.equal(outputs[3].cpu(), expected)
    assert int(stats[3][5]) == 0
    # Epochs stayed in step: the next call is exact on every rank.
    ids, counts = _ep_batch(ranks, num_logical, seed=16, tokens=[32] * ranks)
    outputs, _, stats = _launch_all(exchanges, ids, tables, block_m=16)
    for r in range(ranks):
        expected = tile_repair_route_reference(
            ids[r], counts, tables, source_rank=r, block_m=16, max_moves=32
        )
        assert torch.equal(outputs[r].cpu(), expected)
        assert int(stats[r][5]) == 0


@cuda
def test_p2p_missing_peer_and_overwritten_words_fall_back():
    ranks, num_logical = 4, 64
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical, timeout_ms=10)
    p2l = _random_p2l(num_logical, ranks, 4, seed=17)
    tables = _cuda_tables(p2l, num_logical, ranks)
    batches = [_ep_batch(ranks, num_logical, seed=400 + k, tokens=[24] * ranks) for k in range(4)]
    # Ranks 0-2 run three calls while rank 3 is missing: all time out.
    for k in range(3):
        outputs, _, stats = _launch_all(exchanges, batches[k][0], tables, block_m=16, ranks_to_run=[0, 1, 2])
        for r in range(3):
            assert torch.equal(outputs[r].cpu(), _primary_only(batches[k][0][r], tables))
            assert int(stats[r][5]) == P2P_FAIL_TIMEOUT
    # Rank 3 catches up. Call 1's words were overwritten by call 3 (same
    # parity): it notices without waiting. Calls 2 and 3 find their words.
    for k in range(3):
        outputs, _, stats = _launch_all(exchanges, batches[k][0], tables, block_m=16, ranks_to_run=[3])
        if k == 0:
            assert int(stats[3][5]) == P2P_FAIL_OVERWRITTEN
            assert torch.equal(outputs[3].cpu(), _primary_only(batches[k][0][3], tables))
        else:
            assert int(stats[3][5]) == 0
            expected = tile_repair_route_reference(
                batches[k][0][3], batches[k][1], tables, source_rank=3, block_m=16, max_moves=32
            )
            assert torch.equal(outputs[3].cpu(), expected)
    assert [x.failure_count() for x in exchanges] == [3, 3, 3, 1]
    ids, counts = batches[3]
    outputs, _, stats = _launch_all(exchanges, ids, tables, block_m=16)
    for r in range(ranks):
        expected = tile_repair_route_reference(ids[r], counts, tables, source_rank=r, block_m=16, max_moves=32)
        assert torch.equal(outputs[r].cpu(), expected)


@cuda
def test_p2p_count_overflow_is_primary_only_on_every_rank():
    ranks, num_logical = 2, 16
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    p2l = _random_p2l(num_logical, ranks, 2, seed=18)
    tables = _cuda_tables(p2l, num_logical, ranks)
    hot = int(tables.replicated[0])
    ids = [torch.full((0xFFFF, 1), hot, dtype=torch.int64), torch.tensor([[hot], [3]])]
    outputs, _, stats = _launch_all(exchanges, ids, tables, block_m=64)
    for r in range(ranks):
        assert int(stats[r][5]) == P2P_FAIL_OVERFLOW
        assert torch.equal(outputs[r].cpu(), _primary_only(ids[r], tables))


@cuda
def test_p2p_cuda_graph_replay():
    ranks, num_logical = 8, 256
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    p2l = _random_p2l(num_logical, ranks, 4, seed=19)
    tables = _cuda_tables(p2l, num_logical, ranks)
    tokens = [16] * ranks
    ids, _ = _ep_batch(ranks, num_logical, seed=20, tokens=tokens)
    static_ids = [x.cuda() for x in ids]
    streams = [torch.cuda.Stream() for _ in range(ranks)]
    # Warm up (compile) with a real, complete call on every rank.
    _launch_all(exchanges, ids, tables, block_m=0, num_slots=len(p2l), streams=streams)
    graphs, captured = [], []
    for r in range(ranks):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(streams[r]):
            with torch.cuda.graph(graph, stream=streams[r]):
                # Two layers per graph: the epoch counter advances inside the graph.
                first = tile_repair_route_p2p(
                    static_ids[r], tables, exchanges[r], block_m=0, num_slots=len(p2l)
                )
                second = tile_repair_route_p2p(
                    static_ids[r], tables, exchanges[r], block_m=0, num_slots=len(p2l)
                )
        graphs.append(graph)
        captured.append((first, second))
    for step in range(4):
        ids, counts = _ep_batch(ranks, num_logical, seed=500 + step, tokens=tokens)
        for r in range(ranks):
            static_ids[r].copy_(ids[r].cuda())
        torch.cuda.synchronize()
        for r in range(ranks):
            with torch.cuda.stream(streams[r]):
                graphs[r].replay()
        torch.cuda.synchronize()
        for r in range(ranks):
            expected = tile_repair_route_reference(
                ids[r], counts, tables, source_rank=r, block_m=0, max_moves=32, num_slots=len(p2l)
            )
            assert torch.equal(captured[r][0].cpu(), expected)
            assert torch.equal(captured[r][1].cpu(), expected)
    assert all(int(x.epoch.item()) == 1 + 2 * 4 for x in exchanges)
    assert all(x.failure_count() == 0 for x in exchanges)


@cuda
def test_p2p_router_through_the_balancer_skips_the_collective():
    ranks, num_logical = 4, 64
    exchanges = local_peer_exchanges(ranks, max_logical=num_logical)
    p2l = _random_p2l(num_logical, ranks, 4, seed=21)
    placement = _snapshot(p2l, num_logical=num_logical, ranks=ranks, device="cuda")
    ids, counts = _ep_batch(ranks, num_logical, seed=22, tokens=[16, 5, 0, 32])
    streams = [torch.cuda.Stream() for _ in range(ranks)]
    outputs = []
    for r in range(ranks):
        mlb = MoELoadBalancer.from_algorithm(
            "tile_repair",
            ep_size=ranks,
            source_rank=r,
            experts_per_rank=1,
            collectives=_GroupCollectives(r, ranks, None),
            tile_p2p_exchange=exchanges[r],
        )
        mlb.on_placement_committed(placement)
        with torch.cuda.stream(streams[r]):
            request = RoutingRequest(
                layer_id=0,
                logical_topk_ids=ids[r].cuda(),
                topk_weights=torch.ones(ids[r].shape, device="cuda"),
                placement=placement,
                stage=RoutingStage.DECODE,
            )
            decision = mlb.route_tokens(request)
        assert decision.metadata["exchange"] == "p2p"
        outputs.append(decision.routed_physical_topk_ids)
    torch.cuda.synchronize()
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=num_logical, num_ranks=ranks,
        num_local_physical_experts=len(p2l) // ranks,
    )
    _check_ep_routing(ids, counts, tables, p2l, block_m=64, max_moves=32, outputs=outputs)


# ---------------------------------------------------------------- multi-process

_MULTI = os.path.join(os.path.dirname(__file__), "multigpu_tile_repair_p2p.py")


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two or more CUDA devices",
)
def test_p2p_symmetric_memory_across_processes(tmp_path):
    world = min(torch.cuda.device_count(), 8)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "src"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
         _MULTI, str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    results = sorted(tmp_path.glob("rank*.ok"))
    assert len(results) == world, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_module_exports():
    assert p2p_module.TIMEOUT_ENV == "EVENMOE_TILE_P2P_TIMEOUT_MS"
    assert isinstance(TileRepairL2Router(num_ranks=2, source_rank=0).p2p_requested, bool)
