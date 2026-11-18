# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU check of Waterfill's tile mode over real NVLink P2P (not collected by pytest).

    torchrun --nproc-per-node 8 tests/multigpu_waterfill_tile.py

A) K1's P2P exchange against ``dist.all_gather`` of the host-mirror payloads and
   K2 against the host mirror, eager and CUDA-graph replay, tiles 1..128, idle
   ranks, ranks reporting different tiles; a late peer makes the others keep
   every row local and the next call is back in step.
B) The whole policy (``static+waterfill`` with ``tile_rows``, through
   ``MoELoadBalancer.from_algorithm``) and the SGLang adapter, with the P2P
   exchange and with the framework all-reduce (collectives without a
   ``device_group``), each swept tile passed per call in ``metadata['tile_rows']``; CUDA-graph replay.

Prints one PASS/FAIL line per section on rank 0 and exits non-zero on a failure.
"""

import os
import sys
import time
import traceback
from dataclasses import replace

import _waterfill_tile_reference as ref
import torch
import torch.distributed as dist

from evenmoe import MoELoadBalancer
from evenmoe.adapters.sglang.routing import to_routing_request, to_sglang_routing_output
from evenmoe.core.routing_pipeline import RoutingPipeline
from evenmoe.core.types import PlacementSnapshot, RoutingPolicyConfig
from evenmoe.kernels.p2p import create_peer_exchange
from evenmoe.kernels.tile_waterfill import count_and_exchange, payload_width, plan_and_assign

K, PER = 8, 32
TILES = (1, 8, 16, 24, 32, 64, 128)
RSF = 2.5


def log(*args):
    if dist.get_rank() == 0:
        print(*args, flush=True)


class AllReduceOnly:
    """The framework's collectives without a ``device_group``: tile mode uses the all-reduce."""

    def __init__(self, group):
        self.group = group
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)

    def all_reduce_sum(self, x):
        dist.all_reduce(x, group=self.group)
        return x


class WithGroup(AllReduceOnly):
    @property
    def device_group(self):
        return self.group


def batch(seed, n, experts, device):
    if n == 0:
        return torch.zeros(0, K, dtype=torch.int32, device=device)
    g = torch.Generator().manual_seed(seed)
    hot = torch.rand(experts, generator=g) ** 3
    return torch.stack([torch.multinomial(hot, K, generator=g) for _ in range(n)]).to(torch.int32).to(device)


def all_gather_rows(row):
    out = [torch.empty_like(row) for _ in range(dist.get_world_size())]
    dist.all_gather(out, row)
    return torch.stack(out)


def nv_for(step, rank, ranks):
    if step % 3 == 0 and rank == ranks - 1:
        return 0  # an idle rank
    return (7 * step + 13 * rank) % 97 + (1 if step % 2 else 40)


def check_k2(gathered, status, want, me, p, n):
    shared = plan_and_assign(gathered, num_local_physical=PER, rank=me, num_tokens=n, rho=1.5, status=status)
    want_plan = ref.plan(want.cpu(), num_physical=p, num_local_physical=PER, rank=me)
    assert torch.equal(shared.cpu(), ref.assign(want_plan, me, n))
    received = torch.bincount(shared[shared >= 0].to(torch.int64), minlength=dist.get_world_size())
    dist.all_reduce(received)
    assert received.cpu().tolist() == want_plan[0].tolist(), "the ranks' splits must add up to the same quotas"


def section_a(me, ranks, dev, group):
    p = ranks * PER
    width = payload_width(p)
    ex = create_peer_exchange(group, width=width)
    assert ex is not None, "P2P exchange unavailable"
    checked = 0
    for tile in TILES:
        for step in range(6):
            n = nv_for(step, me, ranks)
            x = batch(1000 * tile + 10 * step + me, n + 3, p, dev)  # 3 rows past nv
            my_tile = tile if step % 2 else max(1, tile // 2 + me % 3)  # ranks may disagree
            rows, status = count_and_exchange(x, p, n, my_tile, ex)
            want = all_gather_rows(ref.count_rows(x, p, n, my_tile).to(dev))
            assert int(status.item()) == 0 and torch.equal(rows, want)
            check_k2(rows, status, want, me, p, x.shape[0])
            checked += 1
    # CUDA graph: K1 + K2 captured once, replayed with new inputs.
    x_buf = torch.zeros(160, K, dtype=torch.int32, device=dev)
    nv_buf = torch.zeros(1, dtype=torch.int32, device=dev)

    def run():
        rows, status = count_and_exchange(x_buf, p, nv_buf, 24, ex)
        return rows, status, plan_and_assign(rows, num_local_physical=PER, rank=me, num_tokens=160, rho=1.5, status=status)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_rows, g_status, g_shared = run()
    dist.barrier()
    for step in range(10):
        n = nv_for(step, me, ranks) % 150
        x = batch(77 * step + me, 160, p, dev)
        x[n:] = -1
        x_buf.copy_(x)
        nv_buf.fill_(n)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        want = all_gather_rows(ref.count_rows(x, p, n, 24).to(dev))
        assert int(g_status.item()) == 0 and torch.equal(g_rows, want)
        assert torch.equal(g_shared.cpu(), ref.assign(ref.plan(want.cpu(), num_physical=p, num_local_physical=PER, rank=me), me, 160))
        checked += 1
    # Late peer: rank 3 arrives ~300 ms late with a 50 ms timeout.
    late = create_peer_exchange(group, width=width, timeout_ms=50)
    x = batch(5, 20, p, dev)
    for _ in range(2):  # compile and settle first
        count_and_exchange(x, p, 20, 16, late)
        torch.cuda.synchronize()
        dist.barrier()
    if me == 3 % ranks:
        time.sleep(0.3)
    rows, status = count_and_exchange(x, p, 20, 16, late)
    shared = plan_and_assign(rows, num_local_physical=PER, rank=me, num_tokens=20, rho=1.5, status=status)
    torch.cuda.synchronize()
    if me != 3 % ranks:
        assert int(status.item()) != 0, "a late peer must fail the waiting ranks' call"
        assert shared.tolist() == [me] * 20, "a failed call keeps every row local"
    dist.barrier()
    rows, status = count_and_exchange(x, p, 20, 16, late)
    torch.cuda.synchronize()
    want = all_gather_rows(ref.count_rows(x, p, 20, 16).to(dev))
    assert int(status.item()) == 0 and torch.equal(rows, want), "the next call is back in step"
    return checked


def snapshot(ranks, device):
    p = ranks * PER
    p2l = torch.randperm(p, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(p2l).to(torch.int32)
    return PlacementSnapshot(
        layer_id=0, num_logical_experts=p, num_physical_experts=p, ep_size=ranks, num_local_physical_experts=PER,
        physical_to_logical_map=p2l.to(device), logical_to_physical_candidates=inv[:, None].to(torch.int64).to(device),
        logical_to_physical_count=torch.ones(p, dtype=torch.int64, device=device), default_physical_for_logical=inv.to(device),
    )


def section_b(me, ranks, dev, group):
    p = ranks * PER
    snap = snapshot(ranks, dev)
    inv = snap.default_physical_for_logical.cpu().to(torch.int64)
    config = RoutingPolicyConfig("waterfill", metadata={"tile_rows": 64})
    lbs = {}
    for name, coll in (("p2p", WithGroup(group)), ("all_reduce", AllReduceOnly(group))):
        lb = MoELoadBalancer.from_algorithm(
            RoutingPipeline.from_value(["static", config]), ep_size=ranks, source_rank=me, experts_per_rank=PER, collectives=coll, ep_collectives=coll
        )
        lb.on_placement_committed(snap)
        lbs[name] = lb
    assert lbs["p2p"]._routing_policies["waterfill"].tile_mode.peer_exchange is not None, "P2P inside one node"
    assert lbs["all_reduce"]._routing_policies["waterfill"].tile_mode.peer_exchange is None
    checked = 0
    for tile in TILES:
        for step in range(4):
            n = nv_for(step, me, ranks)
            x = batch(500 * tile + 10 * step + me, n + 2, p, dev)
            w = torch.rand(x.shape[0], K, device=dev)
            row = ref.count_rows(inv[x.cpu().to(torch.int64)], p, n, tile)
            want = ref.assign(ref.plan(all_gather_rows(row.to(dev)).cpu(), num_physical=p, num_local_physical=PER, rank=me), me, x.shape[0])
            outs = []
            for lb in lbs.values():
                req = to_routing_request(
                    layer_id=0, logical_topk_ids=x, topk_weights=w, placement=snap,
                    token_count=torch.tensor([n], dtype=torch.int32, device=dev), routed_scaling_factor=RSF,
                )
                req = replace(req, metadata={**req.metadata, "tile_rows": tile})
                dec = lb.route_tokens(req)
                assert torch.equal(dec.shared_expert_rank.cpu().to(torch.int64), want)
                outs.append(to_sglang_routing_output(dec, num_routed_experts=p, world_size=ranks, routed_scaling_factor=RSF))
            assert torch.equal(outs[0].topk_ids, outs[1].topk_ids) and torch.equal(outs[0].topk_weights, outs[1].topk_weights)
            checked += 1
    # CUDA graph of the whole policy + adapter, replayed with new inputs.
    lb = lbs["p2p"]
    x_buf = torch.full((96, K), -1, dtype=torch.int32, device=dev)
    w_buf = torch.rand(96, K, device=dev)
    nv_buf = torch.zeros(1, dtype=torch.int32, device=dev)

    def run():
        req = to_routing_request(
            layer_id=0, logical_topk_ids=x_buf, topk_weights=w_buf, placement=snap, token_count=nv_buf,
            routed_scaling_factor=RSF,
        )
        req = replace(req, metadata={**req.metadata, "tile_rows": 16})
        dec = lb.route_tokens(req)
        return dec, to_sglang_routing_output(dec, num_routed_experts=p, world_size=ranks, routed_scaling_factor=RSF)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_dec, g_out = run()
    dist.barrier()
    for step in range(10):
        n = nv_for(step, me, ranks) % 90
        x = batch(31 * step + me, 96, p, dev)
        x[n:] = -1
        x_buf.copy_(x)
        nv_buf.fill_(n)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        row = ref.count_rows(inv[x.cpu().to(torch.int64).clamp(min=0)].masked_fill(x.cpu() < 0, -1), p, n, 16)
        want = ref.assign(ref.plan(all_gather_rows(row.to(dev)).cpu(), num_physical=p, num_local_physical=PER, rank=me), me, 96)
        assert torch.equal(g_dec.shared_expert_rank.cpu().to(torch.int64), want)
        assert torch.equal(g_out.topk_ids[:n, K].cpu().to(torch.int64), want[:n] * (PER + 1) + PER)
        checked += 1
    return checked


def main():
    dist.init_process_group("nccl")
    me, ranks = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", me)))
    dev = torch.device("cuda", torch.cuda.current_device())
    failed = False
    for name, section in (("A exchange", section_a), ("B policy", section_b)):
        try:
            checked = section(me, ranks, dev, dist.group.WORLD)
            ok = torch.tensor([1], device=dev)
        except Exception:  # noqa: BLE001 -- report every rank's failure; the others may hang in a collective
            print(f"rank {me}: {name} failed:\n{traceback.format_exc()}", flush=True)
            checked, ok = 0, torch.tensor([0], device=dev)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        failed |= not bool(ok.item())
        log(f"{name}: {'PASS' if ok.item() else 'FAIL'} ({checked} cases per rank, {ranks} ranks, tiles {TILES})")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
