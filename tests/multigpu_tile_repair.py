# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run under torchrun: mixed eager/graph ranks with live placement updates."""

import os
import pathlib
import sys

import torch
import torch.distributed as dist
from test_tile_repair import _snapshot

from evenmoe import MoELoadBalancer, RoutingRequest, RoutingStage
from evenmoe.kernels.tile_repair import build_tile_repair_tables, tile_repair_route_reference


class _Collectives:
    def __init__(self, rank, world, group):
        self.rank, self.world_size, self.group = rank, world, group

    def all_reduce_sum(self, payload):
        dist.all_reduce(payload, group=self.group)
        return payload


def _check_graph_placement_updates(group, *, start_remote: bool) -> None:
    """Mix graph/eager peers after local-only <-> cross-rank copy updates."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    logical = 2 * world
    local_copies, remote_copies = [], []
    for r in range(world):
        local_copies.extend([2 * r, 2 * r + 1, 2 * r])
        remote_copies.extend([2 * r, 2 * r + 1, 2 * ((r + 1) % world)])
    layouts = [remote_copies, local_copies] if start_remote else [local_copies, remote_copies]
    snapshots = [_snapshot(p, num_logical=logical, ranks=world, device="cuda") for p in layouts]
    mlb = MoELoadBalancer.from_algorithm(
        "tile_repair", ep_size=world, source_rank=rank, experts_per_rank=3,
        collectives=_Collectives(rank, world, group),
    )
    mlb.on_placement_committed(snapshots[0])
    ids = torch.zeros(128, 1, dtype=torch.int64, device="cuda")
    weights = torch.ones_like(ids, dtype=torch.float32)

    def route(snapshot):
        return mlb.route_tokens(RoutingRequest(
            layer_id=0, logical_topk_ids=ids, topk_weights=weights,
            placement=snapshot, stage=RoutingStage.DECODE,
        )).routed_physical_topk_ids

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route(snapshots[0])
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = route(snapshots[0])
    torch.cuda.current_stream().wait_stream(stream)
    counts = torch.zeros(world, logical)
    counts[:, 0] = ids.numel()
    for index in (1, 0, 1):
        snapshot = snapshots[index]
        mlb.on_placement_committed(snapshot)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        if rank % 2 == 0:
            graph.replay()
            out = captured
        else:
            out = route(snapshot)
        tables = build_tile_repair_tables(
            torch.tensor(layouts[index]), num_logical=logical, num_ranks=world,
            num_local_physical_experts=3,
        )
        expected = tile_repair_route_reference(
            ids.cpu(), counts, tables, source_rank=rank, block_m=64, max_moves=32,
        )
        assert torch.equal(out.cpu(), expected), (rank, start_remote, index)
    torch.cuda.synchronize()
    dist.barrier(group=group)


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    for start_remote in (False, True):
        _check_graph_placement_updates(group, start_remote=start_remote)
    (pathlib.Path(sys.argv[1]) / f"rank{dist.get_rank()}.ok").write_text("PASS\n")
    dist.destroy_process_group()
