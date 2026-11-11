# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe import MoELoadBalancer, RoutingPolicyConfig, RoutingRequest
from evenmoe.policies.l2 import waterfill
from evenmoe.policies.l2.waterfill import WaterfillL2Router

pytestmark = pytest.mark.unit

_TOPK = torch.tensor([[0, 5], [2, 3], [7, -1]])  # ranks (W=4, 2 experts/rank): {0,2} {1} {3}
_ROUTED = torch.tensor([1, 2, 1, 1])
_ACTIVE = torch.tensor([3, 3, 2, 4])


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("WATERFILL_LOAD_VIEW", "WATERFILL_CANDIDATES"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_select(**kwargs):
        seen.update(kwargs)
        return torch.zeros(kwargs["topk_ids"].shape[0], dtype=torch.int64)

    monkeypatch.setattr(waterfill, "waterfill_select", fake_select)
    return seen


def _router(**kwargs) -> WaterfillL2Router:
    return WaterfillL2Router(source_rank=1, world_size=4, experts_per_rank=2, **kwargs)


def _request(active=None) -> RoutingRequest:
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=_TOPK,
        routed_physical_topk_ids=_TOPK,
        topk_weights=torch.ones(_TOPK.shape),
        routed_rank_load=_ROUTED,
        active_rank_token_count=active,
    )


@pytest.mark.parametrize(
    ("global_view", "candidates", "rank_load", "routed_load", "allow_all"),
    [
        (False, "all", _ROUTED, _ROUTED, True),
        # local*W + T_local (3 rows) over local*W.
        (False, "routed", _ROUTED * 4 + 3, _ROUTED * 4, False),
        (True, "routed", _ROUTED + _ACTIVE, _ROUTED, False),
        (True, "all", _ROUTED + _ACTIVE, _ROUTED, True),
    ],
)
def test_each_combination_feeds_the_kernel_its_loads(
    captured, global_view, candidates, rank_load, routed_load, allow_all
):
    decision = _router(candidates=candidates).route_tokens(
        _request(_ACTIVE if global_view else None), RoutingPolicyConfig(name="waterfill")
    )

    assert torch.equal(captured["rank_load"], rank_load)
    assert torch.equal(captured["routed_load"], routed_load)
    assert captured["allow_all_ranks"] is allow_all
    assert decision.metadata["load_view"] == ("global" if global_view else "local")
    assert decision.metadata["allow_all_ranks"] is allow_all


def test_local_routed_weights_are_the_local_weights_scaled_by_world_size(captured):
    """The uniform T_local term cancels in target_total - rank_load."""
    _router(candidates="routed").route_tokens(_request(), RoutingPolicyConfig(name="waterfill"))
    topk = _TOPK.shape[1]

    def weights(rank_load, routed_load):
        total = int(rank_load.sum()) + int(routed_load.sum()) // topk
        target = -(-total // 4)
        return (target - rank_load).clamp(min=0)

    assert torch.equal(
        weights(captured["rank_load"], captured["routed_load"]), weights(_ROUTED, _ROUTED) * 4
    )


def test_the_default_candidate_set_is_routed(captured):
    _router().route_tokens(_request(), RoutingPolicyConfig(name="waterfill"))

    assert captured["allow_all_ranks"] is False


def test_a_selection_overrides_the_router_candidate_set(captured):
    router = _router(candidates="routed")

    router.route_tokens(
        _request(), RoutingPolicyConfig(name="waterfill", metadata={"candidates": "all"})
    )

    assert captured["allow_all_ranks"] is True


@pytest.mark.parametrize(("env", "allow_all"), [(None, False), ("routed", False), ("all", True)])
def test_built_router_reads_the_experiment_override(monkeypatch, captured, env, allow_all):
    if env is not None:
        monkeypatch.setenv("WATERFILL_CANDIDATES", env)
    mlb = MoELoadBalancer.from_algorithm("waterfill", ep_size=4, source_rank=1, experts_per_rank=2)
    router = mlb._routing_policies["waterfill"]

    router.route_tokens(_request(), RoutingPolicyConfig(name="waterfill"))

    assert captured["allow_all_ranks"] is allow_all


def test_router_rejects_an_unknown_candidate_set():
    with pytest.raises(ValueError, match="candidates"):
        _router(candidates="some")


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _cuda_request(global_view: bool, num_tokens: int = 512) -> RoutingRequest:
    gen = torch.Generator().manual_seed(0)
    topk = torch.randint(0, 8, (num_tokens, 2), generator=gen).cuda()
    routed = torch.bincount((topk // 2).flatten(), minlength=4)
    active = torch.tensor([num_tokens, 300, 200, 100]).cuda() if global_view else None
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=topk,
        routed_physical_topk_ids=topk,
        topk_weights=torch.ones(topk.shape, device="cuda"),
        routed_rank_load=routed,
        active_rank_token_count=active,
    )


@_cuda
@pytest.mark.parametrize("global_view", [False, True])
def test_routed_candidates_stay_on_the_source_or_a_routed_rank(global_view):
    request = _cuda_request(global_view)
    config = RoutingPolicyConfig(name="waterfill")

    chosen = _router(candidates="routed").route_tokens(request, config).shared_expert_rank
    routed_ranks = request.routed_physical_topk_ids // 2

    allowed = (routed_ranks == chosen[:, None]).any(dim=1) | (chosen == 1)
    assert bool(allowed.all())
    spread = _router(candidates="all").route_tokens(request, config).shared_expert_rank
    assert int(spread.min()) >= 0 and int(spread.max()) < 4


@_cuda
@pytest.mark.parametrize("global_view", [False, True])
@pytest.mark.parametrize("candidates", ["routed", "all"])
def test_every_combination_captures_in_a_cuda_graph(global_view, candidates):
    """Graph capture fails on a host sync, so this proves each path is sync-free."""
    request = _cuda_request(global_view, num_tokens=64)
    router = _router(candidates=candidates)
    config = RoutingPolicyConfig(name="waterfill")
    eager = router.route_tokens(request, config).shared_expert_rank

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = router.route_tokens(request, config).shared_expert_rank
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured, eager)
