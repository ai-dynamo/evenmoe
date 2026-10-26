# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end L1 placement -> commit -> L2 routing on one GPU.

Every other GPU test drives one kernel or one policy in isolation. This one
runs the path a serving framework takes, at DeepSeek-V3 scale, through the
public API and the real CUDA kernels:

  1. record a skewed logical load and hand it to the L1 planner
  2. commit the plan as the per-layer snapshots the framework would publish
  3. route a fresh batch from every EP source rank through the L2 pipeline

EP is simulated in one process: each source rank routes its own slice of the
batch against the same committed placement, and the EP-wide logical count the
orchestrator would all-reduce is passed in precomputed, as vLLM does.

Two claims are checked. Routing is correct: every token lands on a physical
slot that holds its logical expert. Routing is useful: the busiest EP rank
receives less traffic than it would with no redundancy and no rebalancing.
"""

from __future__ import annotations

import pytest
import torch

from evenmoe import (
    MoELoadBalancer,
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
    RoutingRequest,
    RoutingStage,
)
from evenmoe.adapters.vllm import (
    build_physical_to_rank_map,
    nearest_replica_table,
    to_placement_snapshot,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]

# DeepSeek-V3 routed experts on one 8-GPU node with 32 redundant slots, the
# configuration the README's performance table uses.
NUM_LOGICAL = 256
TOP_K = 8
EP_SIZE = 8
NUM_REDUNDANT = 32
NUM_PHYSICAL = NUM_LOGICAL + NUM_REDUNDANT
NUM_LOCAL_PHYSICAL = NUM_PHYSICAL // EP_SIZE
NUM_LAYERS = 2
TOKENS_PER_RANK = 512

# (L1 placement policy, L2 routing pipeline)
PIPELINES = [
    ("deepseek", "static"),
    ("deepseek", "dynamic"),
    ("deepseek", "lplb"),
    ("ultraep", "ultraep"),
]


def _popularity(seed: int) -> torch.Tensor:
    """Per-layer expert popularity with a heavy head, like real MoE routers."""

    generator = torch.Generator().manual_seed(seed)
    return torch.rand(NUM_LAYERS, NUM_LOGICAL, generator=generator).pow(4) * 8


def _sample_topk(popularity: torch.Tensor, seed: int) -> torch.Tensor:
    """[layers, EP ranks, tokens per rank, top_k] logical TopK ids."""

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        NUM_LAYERS, EP_SIZE, TOKENS_PER_RANK, NUM_LOGICAL, generator=generator
    )
    logits = popularity[:, None, None, :] + noise
    return logits.topk(TOP_K, dim=-1).indices.to(torch.int32)


def _logical_count(topk: torch.Tensor) -> torch.Tensor:
    """[EP ranks, layers, logical] per-source-rank token counts."""

    one_hot = torch.zeros(NUM_LAYERS, EP_SIZE, NUM_LOGICAL)
    one_hot.scatter_add_(
        -1,
        topk.flatten(2).to(torch.int64),
        torch.ones(NUM_LAYERS, EP_SIZE, TOKENS_PER_RANK * TOP_K),
    )
    return one_hot.transpose(0, 1).contiguous()


def _plan(policy: str, logical_count: torch.Tensor, rank: int) -> PlacementPlan:
    if policy == "ultraep":
        # UltraEP solves once per source rank from the per-rank breakdown.
        stats = logical_count.cuda()
        metadata = {"rank": rank, "num_nvl_ranks": EP_SIZE}
    else:
        # EPLB plans from the EP-wide sum.
        stats = logical_count.sum(dim=0)
        metadata = {}
    request = PlacementRequest(
        stats=PlacementStats(logical_count=stats),
        topology=PlacementTopology(
            num_physical_experts=NUM_PHYSICAL,
            num_local_physical_experts=NUM_LOCAL_PHYSICAL,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(name=policy, metadata=metadata),
    )
    return MoELoadBalancer().plan_placement(request)


def _commit(plan: PlacementPlan, layer_id: int, rank: int):
    """The snapshot a framework publishes once the plan's weights are loaded."""

    device = torch.device("cuda")
    candidates = plan.logical_to_all_physical_map[layer_id].to(device, torch.int64)
    counts = plan.logical_to_physical_count[layer_id].to(device, torch.int64)
    quota = plan.metadata.get("rank_quota_prefix")
    return to_placement_snapshot(
        None,
        layer_id,
        physical_to_logical_map=plan.physical_to_logical_map[layer_id].to(device, torch.int64),
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        ep_size=EP_SIZE,
        candidates=candidates,
        counts=counts,
        default_physical_for_logical=nearest_replica_table(
            candidates,
            counts,
            ep_rank=rank,
            num_local_physical_experts=NUM_LOCAL_PHYSICAL,
            ep_size=EP_SIZE,
        ),
        physical_to_rank_map=build_physical_to_rank_map(NUM_PHYSICAL, EP_SIZE, device),
        rank_quota_prefix=None if quota is None else quota[layer_id].to(device),
    )


def _imbalance(rank_load: torch.Tensor) -> float:
    return rank_load.max().item() / rank_load.float().mean().item()


def _baseline_rank_load(topk: torch.Tensor) -> torch.Tensor:
    """Per-layer EP rank load with one copy per expert in contiguous blocks."""

    owner = topk.to(torch.int64) // (NUM_LOGICAL // EP_SIZE)
    return torch.stack(
        [torch.bincount(owner[layer].flatten(), minlength=EP_SIZE) for layer in range(NUM_LAYERS)]
    )


@pytest.fixture(scope="module")
def workload():
    popularity = _popularity(seed=0)
    # L1 plans from a recorded window; L2 routes a later batch from the same
    # distribution, as it does between two rebalances.
    recorded = _sample_topk(popularity, seed=1)
    live = _sample_topk(popularity, seed=2)
    return _logical_count(recorded), live


@pytest.mark.parametrize(("placement_policy", "routing_pipeline"), PIPELINES)
def test_placement_then_routing_end_to_end(workload, placement_policy, routing_pipeline):
    logical_count, live = workload
    routed_rank_load = torch.zeros(NUM_LAYERS, EP_SIZE, dtype=torch.int64)
    reference_maps = None

    for rank in range(EP_SIZE):
        plan = _plan(placement_policy, logical_count, rank)
        maps = plan.physical_to_logical_map.to(torch.int64).cpu()
        assert maps.shape == (NUM_LAYERS, NUM_PHYSICAL)
        assert int(maps.min()) >= 0 and int(maps.max()) < NUM_LOGICAL
        for layer_id in range(NUM_LAYERS):
            assert len(torch.unique(maps[layer_id])) == NUM_LOGICAL, (
                f"layer {layer_id}: some logical expert has no physical slot"
            )
        # Every rank must commit the same placement, or the ranks would
        # disagree on where an expert's weights live.
        if reference_maps is None:
            reference_maps = maps
        else:
            assert torch.equal(maps, reference_maps), f"rank {rank} planned a different placement"

        balancer = MoELoadBalancer.from_algorithm(
            routing_pipeline,
            ep_size=EP_SIZE,
            source_rank=rank,
            experts_per_rank=NUM_LOCAL_PHYSICAL,
        )
        assert balancer.routing_pipeline.policies == (routing_pipeline,)

        for layer_id in range(NUM_LAYERS):
            snapshot = _commit(plan, layer_id, rank)
            balancer.on_placement_committed(snapshot)

            logical_ids = live[layer_id, rank].cuda()
            global_count = torch.bincount(
                live[layer_id].flatten().to(torch.int64), minlength=NUM_LOGICAL
            ).to(device="cuda", dtype=torch.float32)
            decision = balancer.route_tokens(
                RoutingRequest(
                    layer_id=layer_id,
                    logical_topk_ids=logical_ids,
                    topk_weights=torch.full(
                        logical_ids.shape, 1.0 / TOP_K, device="cuda"
                    ),
                    placement=snapshot,
                    stage=RoutingStage.DECODE,
                    global_logical_count=global_count,
                )
            )
            torch.cuda.synchronize()

            assert decision.metadata["executed_policies"] == (routing_pipeline,)
            physical_ids = decision.routed_physical_topk_ids.to(torch.int64)
            assert physical_ids.shape == logical_ids.shape
            assert bool(((physical_ids >= 0) & (physical_ids < NUM_PHYSICAL)).all()), (
                f"rank {rank} layer {layer_id}: out-of-range physical id"
            )
            landed_on = snapshot.physical_to_logical_map[physical_ids]
            assert torch.equal(landed_on, logical_ids.to(torch.int64)), (
                f"rank {rank} layer {layer_id}: a token reached a slot holding another expert"
            )
            routed_rank_load[layer_id] += torch.bincount(
                (physical_ids // NUM_LOCAL_PHYSICAL).flatten(), minlength=EP_SIZE
            ).cpu()

    baseline_rank_load = _baseline_rank_load(live)
    for layer_id in range(NUM_LAYERS):
        routed = _imbalance(routed_rank_load[layer_id])
        baseline = _imbalance(baseline_rank_load[layer_id])
        print(
            f"{placement_policy}+{routing_pipeline} layer {layer_id}: "
            f"max/mean rank load {routed:.3f} (no balancing {baseline:.3f})"
        )
        assert routed < baseline, (
            f"layer {layer_id}: busiest rank got {routed:.3f}x the mean, "
            f"no better than {baseline:.3f}x without balancing"
        )
