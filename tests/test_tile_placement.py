# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from evenmoe import MoELoadBalancer
from evenmoe.core.types import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)
from evenmoe.kernels.tile_repair import build_tile_repair_tables
from evenmoe.policies.l1.tile import TilePlacementPlanner, tile_weights
from evenmoe.policies.tile_common import expected_tiles, resolve_stages


def _brute_expected_tiles(mu, block_m, kmax=4000):
    total = 0.0
    log_p = -mu
    for k in range(kmax):
        if k > 0:
            log_p += math.log(mu) - math.log(k)
        total += math.exp(log_p) * math.ceil(k / block_m)
    return total


@pytest.mark.parametrize("block_m", [1, 8, 16, 64, 128])
def test_expected_tiles_matches_the_poisson_sum(block_m):
    mus = [0.0, 0.05, 0.7, 3.0, 17.5, 64.0, 250.0, 900.0]
    got = expected_tiles(torch.tensor(mus), block_m).tolist()
    for mu, value in zip(mus, got):
        expected = 0.0 if mu == 0.0 else _brute_expected_tiles(mu, block_m)
        assert value == pytest.approx(expected, rel=1e-7, abs=1e-12)
    if block_m == 1:
        assert got == pytest.approx(mus, rel=1e-7, abs=1e-12)  # E[K] = mu


def _counts(layers=3, num_logical=32, seed=0, alpha=1.1):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(layers):
        weight = 1.0 / torch.arange(1, num_logical + 1, dtype=torch.float64) ** alpha
        out.append(weight[torch.randperm(num_logical, generator=gen)] * 1000)
    return torch.stack(out).round()


def _request(counts, *, num_physical, per_rank, metadata=None):
    return PlacementRequest(
        stats=PlacementStats(logical_count=counts),
        topology=PlacementTopology(
            num_physical_experts=num_physical,
            num_local_physical_experts=per_rank,
            num_groups=None,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(name="tile", metadata=metadata or {}),
    )


def _check_plan(plan, *, num_logical, ranks, extras):
    p2l = plan.physical_to_logical_map
    layers, num_physical = p2l.shape
    per_rank = num_physical // ranks
    primaries = num_logical // ranks
    assert per_rank == primaries + extras
    for layer in range(layers):
        row = p2l[layer].tolist()
        # Rank-contiguous [primaries | extras]: the primaries of all ranks are
        # each logical expert exactly once.
        prim = [row[r * per_rank + k] for r in range(ranks) for k in range(primaries)]
        assert sorted(prim) == list(range(num_logical))
        for r in range(ranks):
            block = row[r * per_rank : (r + 1) * per_rank]
            if ranks > 1:
                # Extras sit on a rank that holds no other copy of the expert.
                assert len(set(block)) == len(block)
        # Every live log2phy entry is a copy of its expert, primary first.
        log2phy = plan.logical_to_all_physical_map[layer]
        count = plan.logical_to_physical_count[layer]
        for e in range(num_logical):
            slots = log2phy[e, : count[e]].tolist()
            assert sorted(slots) == [p for p, x in enumerate(row) if x == e]
            assert slots[0] % per_rank < primaries
            assert (log2phy[e, count[e] :] == -1).all()
        # The routing tables pick the planner's primary as copy 0.
        tables = build_tile_repair_tables(
            p2l[layer], num_logical=num_logical, num_ranks=ranks, num_local_physical_experts=per_rank
        )
        assert tables.copy_physical[:, 0].tolist() == log2phy[:, 0].tolist()
        assert int(tables.num_replicated[0]) == (count > 1).sum().item()


@pytest.mark.parametrize("ranks,extras", [(2, 0), (4, 2), (8, 4), (8, 1)])
def test_tile_placement_is_valid(ranks, extras):
    num_logical = 64
    counts = _counts(num_logical=num_logical)
    plan = TilePlacementPlanner().plan(
        _request(
            counts,
            num_physical=num_logical + extras * ranks,
            per_rank=num_logical // ranks + extras,
        )
    )
    _check_plan(plan, num_logical=num_logical, ranks=ranks, extras=extras)
    assert plan.metadata["extra_slots_per_rank"] == extras
    assert plan.metadata["block_m"] == 64
    assert plan.metadata["tokens_per_pass"] == 128.0 * ranks
    if extras:
        # The extras copy the experts with the highest expected tiles.
        weight = plan.metadata["expected_tiles"]
        for layer in range(counts.shape[0]):
            copies = plan.logical_to_physical_count[layer]
            assert copies[int(weight[layer].argmax())] == 2
            assert (copies > 1).sum() <= extras * ranks
            assert copies.sum() == num_logical + extras * ranks


def test_tile_placement_is_deterministic_and_accepts_samples():
    counts = _counts(num_logical=32)
    request = _request(counts, num_physical=48, per_rank=12)
    first = TilePlacementPlanner().plan(request)
    second = TilePlacementPlanner().plan(request)
    assert torch.equal(first.physical_to_logical_map, second.physical_to_logical_map)
    # [samples, layers, experts] is summed over samples.
    stacked = TilePlacementPlanner().plan(
        _request(torch.stack([counts / 2, counts / 2]), num_physical=48, per_rank=12)
    )
    assert torch.equal(first.physical_to_logical_map, stacked.physical_to_logical_map)


def test_tile_weights_follow_block_m_and_tokens_per_pass():
    counts = _counts(layers=1, num_logical=16)
    small = tile_weights(counts, block_m=64, tokens_per_pass=16)
    large = tile_weights(counts, block_m=64, tokens_per_pass=1_000_000)
    # Few tokens: every active expert costs about one tile, so the weights are
    # much flatter than the token shares.
    shares = counts / counts.sum()
    assert small.max() / small.min() < shares.max() / shares.min()
    # Many tokens: tiles are proportional to tokens.
    assert torch.allclose(large / large.sum(), shares, rtol=0.05)
    zero = tile_weights(torch.zeros(1, 16), block_m=64, tokens_per_pass=16)
    assert torch.allclose(zero, zero[0, 0].expand_as(zero))


def test_tile_placement_env_and_metadata(monkeypatch):
    counts = _counts(num_logical=32)
    monkeypatch.setenv("EVENMOE_TILE_BLOCK_M", "16")
    monkeypatch.setenv("EVENMOE_TILE_TOKENS_PER_PASS", "512")
    plan = TilePlacementPlanner().plan(_request(counts, num_physical=40, per_rank=10))
    assert plan.metadata["block_m"] == 16
    assert plan.metadata["tokens_per_pass"] == 512.0
    plan = TilePlacementPlanner().plan(
        _request(counts, num_physical=40, per_rank=10, metadata={"block_m": 128})
    )
    assert plan.metadata["block_m"] == 128


def test_single_rank_fills_extras_with_duplicates():
    counts = _counts(layers=1, num_logical=8)
    plan = TilePlacementPlanner().plan(_request(counts, num_physical=10, per_rank=10))
    row = plan.physical_to_logical_map[0].tolist()
    assert sorted(row[:8]) == list(range(8))
    assert len(row) == 10


def test_tile_placement_rejects_indivisible_layouts():
    counts = _counts(num_logical=32)
    with pytest.raises(ValueError, match="divide"):
        TilePlacementPlanner().plan(_request(counts, num_physical=33, per_rank=11))
    with pytest.raises(ValueError, match="multiple"):
        TilePlacementPlanner().plan(_request(counts, num_physical=16, per_rank=4))


def test_registry_selects_the_tile_planner():
    counts = _counts(num_logical=32)
    plan = MoELoadBalancer().plan_placement(_request(counts, num_physical=40, per_rank=10))
    assert plan.metadata["resolved_policy"] == "tile"
    _check_plan(plan, num_logical=32, ranks=4, extras=2)


def test_stage_parsing():
    assert resolve_stages({"stages": "all"}) == frozenset(
        resolve_stages({"stages": "decode,prefill,mixed,idle,speculative"})
    )
    assert {s.value for s in resolve_stages({"stages": "decode+idle"})} == {"decode", "idle"}
