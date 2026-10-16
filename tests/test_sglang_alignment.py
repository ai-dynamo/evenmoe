# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do our L2 policies decide the same thing SGLang's do?

`static`, `dynamic` and `lplb` exist on both runtimes under the same names, and
the point of EvenMoE is that they are the same algorithm either side of the
adapter. A policy reimplemented with a different tie-break is a different policy
wearing a borrowed name, and every cross-runtime comparison of that row becomes
meaningless -- which is exactly what had happened to `static`, whose map was
built by a "nearest, else rotate by rank" shortcut that has neither SGLang's
same-node tier nor its even share among ranks with no local replica.

The reference implementations are vendored in `_sglang_reference.py` rather than
imported, so these tests fail on drift instead of skipping.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sglang_reference as ref

from evenmoe.adapters.vllm import nearest_replica_table
from evenmoe.policies.l2.replica import (
    route_dynamic_random_replicas,
    route_static_replicas,
)

SHAPES = [
    # (num_logical, num_redundant, ep_size, nnodes)
    (64, 16, 8, 1),
    (64, 16, 8, 2),
    (128, 32, 8, 2),
    (256, 32, 16, 2),
]


def placement(num_logical, num_redundant, max_copies=4):
    num_phy = num_logical + num_redundant
    cand = torch.full((num_logical, max_copies), -1, dtype=torch.int64)
    cand[:, 0] = torch.arange(num_logical, dtype=torch.int64)
    cand[:num_redundant, 1] = torch.arange(num_logical, num_phy, dtype=torch.int64)
    counts = torch.ones(num_logical, dtype=torch.int64)
    counts[:num_redundant] = 2
    return cand, counts, num_phy


@pytest.mark.parametrize("num_logical,num_redundant,ep_size,nnodes", SHAPES)
def test_static_map_matches_sglang(num_logical, num_redundant, ep_size, nnodes):
    """The per-rank dispatch map has to be identical, for every rank.

    Tier 3 (same node) decides whether traffic crosses the inter-node link, and
    tier 4's even share is what stops the ranks with no local replica from all
    piling onto whichever candidate a rotation happens to land on. Both were
    missing before.
    """
    cand, counts, num_phy = placement(num_logical, num_redundant)
    args = types.SimpleNamespace(ep_size=ep_size, nnodes=nnodes)

    for rank in range(ep_size):
        theirs = ref.compute_logical_to_rank_dispatch_physical_map(
            args, cand.unsqueeze(0).clone(), ep_size=ep_size,
            num_physical_experts=num_phy, ep_rank=rank,
        )[0].to(torch.int64)
        ours = nearest_replica_table(
            cand, counts, ep_rank=rank,
            num_local_physical_experts=num_phy // ep_size,
            num_gpus_per_node=ep_size // nnodes,
        ).to(torch.int64)
        assert torch.equal(theirs, ours), (
            f"rank {rank}: {int((theirs != ours).sum())}/{num_logical} entries differ"
        )


@pytest.mark.parametrize("num_logical,num_redundant,ep_size,nnodes", SHAPES)
def test_static_routing_matches_sglang(num_logical, num_redundant, ep_size, nnodes):
    """And routing through that map has to agree token for token."""
    cand, counts, num_phy = placement(num_logical, num_redundant)
    args = types.SimpleNamespace(ep_size=ep_size, nnodes=nnodes)
    torch.manual_seed(0)
    topk = torch.randint(0, num_logical, (512, 8), dtype=torch.int32)

    for rank in (0, ep_size // 2, ep_size - 1):
        their_map = ref.compute_logical_to_rank_dispatch_physical_map(
            args, cand.unsqueeze(0).clone(), ep_size=ep_size,
            num_physical_experts=num_phy, ep_rank=rank,
        )[0]
        theirs = ref.topk_ids_logical_to_physical_static(topk.to(torch.int64), their_map)
        our_map = nearest_replica_table(
            cand, counts, ep_rank=rank,
            num_local_physical_experts=num_phy // ep_size,
            num_gpus_per_node=ep_size // nnodes,
        )
        ours = route_static_replicas(topk, our_map)
        assert torch.equal(theirs.to(torch.int64), ours.to(torch.int64))


def test_dynamic_random_draws_the_way_sglang_does():
    """`dynamic_random` reproduces SGLang's RNG draw, not a hash.

    Same seed, same shapes, same result. `dynamic` deliberately differs -- it
    hashes the token index so tensor-parallel peers agree without talking --
    which is why both exist rather than one replacing the other.
    """
    cand, counts, _ = placement(64, 16)
    topk = torch.randint(0, 64, (256, 8), dtype=torch.int32)

    torch.manual_seed(7)
    theirs = ref.topk_ids_logical_to_physical_dynamic(
        topk.to(torch.int64), cand, counts)
    torch.manual_seed(7)
    ours = route_dynamic_random_replicas(topk, cand, counts)

    assert torch.equal(theirs.to(torch.int64), ours.to(torch.int64))
    # And it must actually be choosing, not trivially agreeing.
    contested = int((counts[topk.reshape(-1).to(torch.int64)] > 1).sum())
    assert contested > 0


def test_dynamic_stays_deterministic_and_dynamic_random_does_not():
    """The reason both exist, asserted rather than left in a comment."""
    from evenmoe.policies.l2.replica import route_dynamic_replicas

    cand, counts, _ = placement(64, 16)
    topk = torch.randint(0, 64, (256, 8), dtype=torch.int32)

    a = route_dynamic_replicas(topk, cand, counts)
    b = route_dynamic_replicas(topk, cand, counts)
    assert torch.equal(a, b), "dynamic must not depend on RNG state"

    torch.manual_seed(1)
    c = route_dynamic_random_replicas(topk, cand, counts)
    torch.manual_seed(2)
    d = route_dynamic_random_replicas(topk, cand, counts)
    assert not torch.equal(c, d), (
        "dynamic_random should follow the RNG; if it does not it is no longer "
        "reproducing SGLang's draw"
    )


def test_l1_and_lplb_are_the_same_code_on_both_runtimes():
    """EPLB placement and LPLB are shared, not reimplemented per adapter.

    Unlike `static`, these two were never at risk of drifting: SGLang's
    `ExpertLocationMetadata.init_by_eplb` calls `plan_placement` on the same
    balancer object, and its `lp` dispatch branch calls the same
    `dispatch_probability` kernel. Both adapters only translate types.

    This pins that down so a future adapter cannot quietly grow its own copy --
    the failure it guards against is a second implementation appearing, which a
    numerical test would not catch until the two had already diverged.
    """
    import inspect

    from evenmoe.adapters.sglang import placement as sgl_placement
    from evenmoe.adapters.vllm import placement as vllm_placement
    from evenmoe.policies.l2 import lplb

    # L1: both adapters build a neutral request and hand it to one planner.
    def code_only(src):
        # Strip comments and docstrings: an adapter naming the framework's
        # interface in prose is fine, calling a planner of its own is not.
        import ast

        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(
                        body[0].value, ast.Constant) and isinstance(
                        body[0].value.value, str):
                    body[0].value.value = ""
        return ast.unparse(tree)

    for mod in (sgl_placement, vllm_placement):
        src = code_only(inspect.getsource(mod))
        assert "PlacementRequest" in src, f"{mod.__name__} bypasses the L1 contract"
        for forbidden in ("EplbAlgorithm", "rebalance_experts", "compute_balanced"):
            assert forbidden not in src, (
                f"{mod.__name__} references {forbidden} in code, which suggests "
                "it plans placement itself instead of translating a request for "
                "the shared planner"
            )

    # L2/LPLB: the LP solution is realised by the shared dispatch kernel, so
    # neither runtime can be applying the share table its own way.
    lplb_src = inspect.getsource(lplb.LPLBL2Router.route_tokens)
    assert "dispatch_probability" in lplb_src


@pytest.mark.parametrize("num_nodes,num_groups", [(1, 8), (2, 8), (2, None)])
def test_l1_placement_is_identical_through_both_adapters(num_nodes, num_groups):
    """Same load, same topology, same placement -- whichever adapter asked.

    The structural test above only shows neither adapter reimplements planning.
    It cannot show they *ask the same question*: `num_nodes` and `num_groups`
    feed the `auto` rule that picks hierarchical versus flat deepseek, so an
    adapter passing them differently gets a different placement out of the same
    planner, and every cross-runtime row would then be comparing two layouts.
    """
    from evenmoe.adapters.sglang import to_placement_request as sgl_req
    from evenmoe.adapters.vllm import to_placement_request as vllm_req
    from evenmoe.core.orchestrator import MoELoadBalancer

    torch.manual_seed(0)
    num_logical, num_redundant, ep_size = 64, 16, 8
    num_physical = num_logical + num_redundant
    # A skewed count, so the planner has something to actually decide.
    counts = torch.randint(1, 1000, (2, num_logical), dtype=torch.int64)
    counts[:, :8] *= 20

    # The two adapters name these differently -- vLLM counts ranks, SGLang
    # counts experts per rank -- which is exactly the kind of mismatch that
    # would produce two layouts from one planner without anything erroring.
    common = {"num_groups": num_groups, "num_nodes": num_nodes, "algorithm": "auto"}
    a = vllm_req(
        logical_count=counts, num_replicas=num_physical,
        num_ranks=ep_size, **common)
    b = sgl_req(
        logical_count=counts, num_physical_experts=num_physical,
        num_local_physical_experts=num_physical // ep_size, **common)

    for field in ("num_physical_experts", "num_local_physical_experts",
                  "num_groups", "num_nodes"):
        assert getattr(a.topology, field) == getattr(b.topology, field), (
            f"{field} differs: vLLM {getattr(a.topology, field)} vs "
            f"SGLang {getattr(b.topology, field)}"
        )
    assert str(a.policy) == str(b.policy), "resolved algorithm differs"

    # L1 only; no replica policy is involved in planning.
    balancer = MoELoadBalancer()
    pa = balancer.plan_placement(a)
    pb = balancer.plan_placement(b)
    assert torch.equal(pa.physical_to_logical_map, pb.physical_to_logical_map)
    assert torch.equal(pa.logical_to_physical_count, pb.logical_to_physical_count)
