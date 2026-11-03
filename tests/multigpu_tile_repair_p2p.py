# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-process check of tile_repair's P2P exchange over torch symmetric memory.

Run with ``python -m torch.distributed.run --standalone --nproc-per-node=N
tests/multigpu_tile_repair_p2p.py OUT_DIR`` (driven by
``test_tile_repair_p2p.py::test_p2p_symmetric_memory_across_processes``).
Every rank checks its output bit-exactly against the host reference over many
calls, under CUDA-graph replay, through ``MoELoadBalancer``, and with a late
peer (timeout -> primary-only, then back in step); then writes
``OUT_DIR/rank<r>.ok``.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch.distributed as dist
from test_tile_repair import _ep_batch, _random_p2l, _snapshot

from evenmoe import MoELoadBalancer, RoutingRequest, RoutingStage
from evenmoe.kernels.tile_repair import (
    NUM_STATS,
    P2P_FAIL_TIMEOUT,
    build_tile_repair_tables,
    tile_repair_route_reference,
)
from evenmoe.kernels.tile_repair.p2p import create_peer_exchange, tile_repair_route_p2p


class _Collectives:
    def __init__(self, rank: int, world: int, group: object):
        self.rank = rank
        self.world_size = world
        self.device_group = group

    def all_reduce_sum(self, payload):
        raise AssertionError("the P2P path must not use the all-reduce")


def _primary_only(ids, tables):
    primary = tables.copy_physical[:, 0].cpu().to(torch.int64)
    valid = (ids >= 0) & (ids < tables.num_logical)
    out = torch.full_like(ids, -1)
    out[valid] = primary[ids[valid]]
    return out


def main(out_dir: pathlib.Path) -> None:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    num_logical, extras = 256, 4
    p2l = _random_p2l(num_logical, world, extras, seed=1234)
    num_slots = len(p2l)
    tables = build_tile_repair_tables(
        torch.tensor(p2l), num_logical=num_logical, num_ranks=world,
        num_local_physical_experts=num_slots // world, device="cuda",
    )

    exchange = create_peer_exchange(group)
    assert exchange is not None and exchange.rank == rank and exchange.world_size == world

    # 1. Many eager calls, fixed and adaptive block_m.
    for step in range(24):
        block_m = (16, 64, 0)[step % 3]
        tokens = [(step * 7 + 3 * r) % 40 for r in range(world)]
        ids, counts = _ep_batch(world, num_logical, seed=1000 + step, tokens=tokens, pad=step % 2)
        stats = torch.empty(NUM_STATS + world, dtype=torch.int32, device="cuda")
        out = tile_repair_route_p2p(
            ids[rank].cuda(), tables, exchange, block_m=block_m, num_slots=num_slots, stats_out=stats
        )
        expected = tile_repair_route_reference(
            ids[rank], counts, tables, source_rank=rank, block_m=block_m, max_moves=32,
            num_slots=num_slots,
        )
        assert torch.equal(out.cpu(), expected), (rank, step)
        assert int(stats[5]) == 0

    # 2. CUDA graph of three layers, replayed with new batches.
    static = torch.zeros(16, 8, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            outs = [
                tile_repair_route_p2p(static, tables, exchange, block_m=64, num_slots=num_slots)
                for _ in range(3)
            ]
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(6):
        ids, counts = _ep_batch(world, num_logical, seed=2000 + step, tokens=[16] * world)
        static.copy_(ids[rank].cuda())
        graph.replay()
        torch.cuda.synchronize()
        expected = tile_repair_route_reference(
            ids[rank], counts, tables, source_rank=rank, block_m=64, max_moves=32
        )
        for o in outs:
            assert torch.equal(o.cpu(), expected), (rank, step)

    # 3. Through MoELoadBalancer: the group comes from RoutingCollectives.device_group.
    placement = _snapshot(p2l, num_logical=num_logical, ranks=world, device="cuda")
    mlb = MoELoadBalancer.from_algorithm(
        "tile_repair", ep_size=world, source_rank=rank, experts_per_rank=num_slots // world,
        collectives=_Collectives(rank, world, group), tile_p2p=True,
    )
    mlb.on_placement_committed(placement)
    router = mlb._routing_policies["tile_repair"]
    assert router.p2p_exchange is not None
    for step in range(4):
        ids, counts = _ep_batch(world, num_logical, seed=3000 + step)
        request = RoutingRequest(
            layer_id=0, logical_topk_ids=ids[rank].cuda(),
            topk_weights=torch.ones(ids[rank].shape, device="cuda"),
            placement=placement, stage=RoutingStage.DECODE,
        )
        decision = mlb.route_tokens(request)
        assert decision.metadata["exchange"] == "p2p"
        expected = tile_repair_route_reference(
            ids[rank], counts, tables, source_rank=rank, block_m=64, max_moves=32
        )
        assert torch.equal(decision.routed_physical_topk_ids.cpu(), expected)

    # 4. A late peer: rank world-1 stalls ~300 ms; the others time out after 50 ms.
    late = create_peer_exchange(group, timeout_ms=50)
    assert late is not None
    ids, counts = _ep_batch(world, num_logical, seed=4000, tokens=[24] * world)
    dist.barrier()
    torch.cuda.synchronize()
    if rank == world - 1:
        torch.cuda._sleep(600_000_000)
    stats = torch.empty(NUM_STATS + world, dtype=torch.int32, device="cuda")
    out = tile_repair_route_p2p(ids[rank].cuda(), tables, late, block_m=16, stats_out=stats)
    torch.cuda.synchronize()
    if rank == world - 1:
        expected = tile_repair_route_reference(
            ids[rank], counts, tables, source_rank=rank, block_m=16, max_moves=32
        )
        assert torch.equal(out.cpu(), expected)
        assert int(stats[5]) == 0
    else:
        assert int(stats[5]) == P2P_FAIL_TIMEOUT, stats.tolist()
        assert torch.equal(out.cpu(), _primary_only(ids[rank], tables))
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    ids, counts = _ep_batch(world, num_logical, seed=4001, tokens=[24] * world)
    out = tile_repair_route_p2p(ids[rank].cuda(), tables, late, block_m=16, stats_out=stats)
    expected = tile_repair_route_reference(
        ids[rank], counts, tables, source_rank=rank, block_m=16, max_moves=32
    )
    assert torch.equal(out.cpu(), expected)
    assert int(stats[5]) == 0

    dist.barrier()
    (out_dir / f"rank{rank}.ok").write_text("ok\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
