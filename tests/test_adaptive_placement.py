# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from evenmoe import MoELoadBalancer
from evenmoe.adapters.sglang import to_load_balancer_kwargs, to_placement_request
from evenmoe.adapters.sglang.placement import _committed_placement
from evenmoe.core.types import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
    PlacementStats,
    PlacementTopology,
)
from evenmoe.policies.l1.adaptive import (
    AdaptivePlacementPlanner,
    maps_from_physical_to_logical,
    rank_imbalance,
    relabel_to_committed,
)
from evenmoe.policies.l1.eplb import EPLBPlacementPlanner

pytestmark = pytest.mark.unit

EP, LOCAL, LOGICAL, LAYERS = 4, 5, 16, 6
PHYSICAL = EP * LOCAL
TOPOLOGY = PlacementTopology(
    num_physical_experts=PHYSICAL, num_local_physical_experts=LOCAL, num_groups=1, num_nodes=1
)


def skewed_counts(seed: int, hot: list[int], samples: int = 4) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    base = torch.randint(50, 100, (samples, LAYERS, LOGICAL), generator=gen).float()
    for e in hot:
        base[..., e] *= 8
    return base


def eplb_plan(counts: torch.Tensor):
    return EPLBPlacementPlanner("auto").plan(
        PlacementRequest(
            stats=PlacementStats(logical_count=counts), topology=TOPOLOGY, policy="auto"
        )
    )


def snapshots(p2l: torch.Tensor, l2p=None) -> dict[int, PlacementSnapshot]:
    built, count = maps_from_physical_to_logical(p2l, LOGICAL)
    l2p = built if l2p is None else l2p
    return {
        layer: PlacementSnapshot(
            layer_id=layer,
            num_logical_experts=LOGICAL,
            num_physical_experts=PHYSICAL,
            ep_size=EP,
            num_local_physical_experts=LOCAL,
            physical_to_logical_map=p2l[layer],
            logical_to_physical_candidates=l2p[layer],
            logical_to_physical_count=count[layer],
        )
        for layer in range(p2l.shape[0])
    }


def adaptive_plan(counts, current=None, **kw):
    kw.setdefault("min_rows_per_slot", 0.0)
    return AdaptivePlacementPlanner(**kw).plan(
        PlacementRequest(
            stats=PlacementStats(logical_count=counts),
            topology=TOPOLOGY,
            policy=PlacementPolicyConfig(name="adaptive"),
            current=current,
        )
    )


def rank_groups(p2l_layer: torch.Tensor) -> list[Counter]:
    rows = p2l_layer.tolist()
    return [Counter(rows[r * LOCAL : (r + 1) * LOCAL]) for r in range(EP)]


def test_without_committed_placement_it_is_plain_eplb():
    counts = skewed_counts(0, [1, 2])
    plan = adaptive_plan(counts)
    assert torch.equal(plan.physical_to_logical_map, eplb_plan(counts).physical_to_logical_map)


def test_layers_whose_traffic_still_fits_never_move():
    recorded = skewed_counts(0, [1, 2])
    first = eplb_plan(recorded)
    committed = first.physical_to_logical_map
    # same traffic mix at another volume: a fresh plan cannot do better
    plan = adaptive_plan(
        recorded * 3,
        snapshots(committed, first.logical_to_all_physical_map),
        threshold=1.08,
        min_gain=0.03,
    )
    assert plan.metadata["replanned_layers"] == ()
    assert plan.metadata["moved_slots"] == 0
    assert torch.equal(plan.physical_to_logical_map, committed)
    # kept layers keep the committed replica lists, order included
    committed_l2p = eplb_plan(recorded).logical_to_all_physical_map
    width = committed_l2p.shape[-1]
    assert torch.equal(plan.logical_to_all_physical_map[..., :width], committed_l2p)
    assert bool((plan.logical_to_all_physical_map[..., width:] == -1).all())


def test_only_drifted_layers_are_replanned_and_end_up_balanced():
    recorded = skewed_counts(0, [1, 2])
    committed = eplb_plan(recorded).physical_to_logical_map
    window = recorded.clone()
    drifted = [0, 3, 4]
    window[:, drifted] = skewed_counts(3, [9, 10, 11])[:, drifted]
    plan = adaptive_plan(window, snapshots(committed), threshold=1.08, min_gain=0.03)
    assert set(plan.metadata["replanned_layers"]) == set(drifted)
    out = plan.physical_to_logical_map
    for layer in set(range(LAYERS)) - set(drifted):
        assert torch.equal(out[layer], committed[layer])
    total = window.sum(0)
    fresh = eplb_plan(window).physical_to_logical_map
    before = rank_imbalance(committed, total, EP)
    after = rank_imbalance(out, total, EP)
    for layer in drifted:
        assert after[layer] < before[layer] - 0.03
        # relabelling keeps the fresh plan's rank groups, only on other ranks
        assert sorted(map(sorted, map(Counter.elements, rank_groups(out[layer])))) == sorted(
            map(sorted, map(Counter.elements, rank_groups(fresh[layer])))
        )
        assert torch.isclose(after[layer], rank_imbalance(fresh, total, EP)[layer])
    # maps agree with the placement and every logical expert keeps a replica
    l2p, count = plan.logical_to_all_physical_map, plan.logical_to_physical_count
    assert torch.equal(count.sum(-1), torch.full((LAYERS,), PHYSICAL))
    assert bool((count >= 1).all())
    for layer in range(LAYERS):
        for logical in range(LOGICAL):
            phys = [p for p in l2p[layer, logical].tolist() if p >= 0]
            assert len(phys) == int(count[layer, logical])
            assert all(int(out[layer, p]) == logical for p in phys)


def test_relabelling_a_permuted_copy_moves_nothing():
    committed = eplb_plan(skewed_counts(0, [1, 2])).physical_to_logical_map[0]
    groups = [committed[r * LOCAL : (r + 1) * LOCAL].tolist() for r in range(EP)]
    shuffled = []
    for g in reversed(groups):
        shuffled += list(reversed(g))
    assert relabel_to_committed(shuffled, committed.tolist(), ep_size=EP) == committed.tolist()


def test_relabelling_moves_at_most_what_the_rank_groups_differ_by():
    committed = eplb_plan(skewed_counts(0, [1, 2])).physical_to_logical_map[1].tolist()
    fresh = eplb_plan(skewed_counts(5, [7, 8, 12])).physical_to_logical_map[1].tolist()
    out = relabel_to_committed(fresh, committed, ep_size=EP)
    moved = sum(a != b for a, b in zip(out, committed))
    naive = sum(a != b for a, b in zip(fresh, committed))
    assert moved <= naive
    assert sorted(Counter(out).items()) == sorted(Counter(fresh).items())


def test_threshold_and_gain_gate_the_move():
    recorded = skewed_counts(0, [1, 2])
    committed = eplb_plan(recorded).physical_to_logical_map
    window = skewed_counts(3, [9, 10, 11])
    loose = adaptive_plan(window, snapshots(committed), threshold=1.0, min_gain=0.0)
    strict = adaptive_plan(window, snapshots(committed), threshold=100.0, min_gain=0.0)
    assert len(loose.metadata["replanned_layers"]) > 0
    assert strict.metadata["replanned_layers"] == ()


def test_sglang_adapter_hands_the_committed_placement_to_adaptive_only():
    live = SimpleNamespace(
        ep_size=2,
        num_logical_experts=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]], [[1, 3], [0, 2]]]),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2], [2, 2]]),
        logical_to_rank_dispatch_physical_map=None,
    )
    config = SimpleNamespace(
        moe=SimpleNamespace(
            moe_load_balancer_algorithm="dynamic",
            eplb_algorithm="adaptive",
            ep_num_redundant_experts=2,
            ep_join_mode=None,
        )
    )
    ctx = SimpleNamespace(
        config_bag=lambda name: config,
        parallel=SimpleNamespace(
            moe_ep_rank=0,
            nnodes=1,
            moe_ep_group=Mock(),
            ep_size=2,
            elastic_ep_initial_size=None,
            ep_join_rank_offset=0,
            tp_size=2,
        ),
        resources=SimpleNamespace(
            mlb_model_info={"num_logical_experts": 2, "num_groups": 1},
            expert_location_metadata=live,
            expert_distribution_recorder=Mock(),
            experts_capturer=Mock(),
        ),
    )
    counts = torch.tensor([[[10.0, 10.0], [10.0, 10.0]]])
    request = to_placement_request(counts, context=ctx)
    assert request.policy.name == "adaptive"
    assert sorted(request.current) == [0, 1]
    assert torch.equal(request.current[1].physical_to_logical_map, live.physical_to_logical_map[1])
    # balanced traffic: the committed placement stays as it is
    mlb = MoELoadBalancer.from_algorithm(**to_load_balancer_kwargs(ctx))
    plan = mlb.plan_placement(request)
    assert torch.equal(plan.physical_to_logical_map, live.physical_to_logical_map)
    # startup (nothing committed yet) and other policies get no committed placement
    ctx.resources.expert_location_metadata = None
    assert to_placement_request(counts, context=ctx).current is None
    ctx.resources.expert_location_metadata = live
    config.moe.eplb_algorithm = "deepseek"
    assert to_placement_request(counts, context=ctx).current is None


def test_windows_too_small_to_judge_move_nothing():
    recorded = skewed_counts(0, [1, 2])
    committed = eplb_plan(recorded).physical_to_logical_map
    drifted = skewed_counts(3, [9, 10, 11])
    rows = drifted.sum(0).sum(-1)  # rows per layer in the window
    per_slot = float(rows.min()) / PHYSICAL
    judged = adaptive_plan(drifted, snapshots(committed), min_rows_per_slot=per_slot * 0.5)
    skipped = adaptive_plan(drifted, snapshots(committed), min_rows_per_slot=per_slot * 1e3)
    assert len(judged.metadata["replanned_layers"]) > 0
    assert skipped.metadata["replanned_layers"] == ()
    assert torch.equal(skipped.physical_to_logical_map, committed)


def test_held_out_gain_separates_drift_from_noise():
    from evenmoe.policies.l1.adaptive import held_out_gain

    gen = torch.Generator().manual_seed(11)
    recorded = skewed_counts(0, [1, 2], samples=8)
    committed = eplb_plan(recorded).physical_to_logical_map

    def plan_fn(count):
        p = eplb_plan(count)
        return p.physical_to_logical_map, p.logical_to_all_physical_map, p.logical_to_physical_count

    # noise: every sample re-draws which experts are hot, the mix averages to the recorded one
    noisy = recorded.clone()
    for i in range(noisy.shape[0]):
        hot = torch.randperm(LOGICAL, generator=gen)[:2]
        noisy[i, :, hot] *= 3
    drifted = skewed_counts(4, [9, 10, 11], samples=8)
    in_window_noise = rank_imbalance(committed, noisy.sum(0), EP) - rank_imbalance(
        eplb_plan(noisy).physical_to_logical_map, noisy.sum(0), EP
    )
    g_noise = held_out_gain(noisy, committed, plan_fn, EP, fallback=in_window_noise)
    g_drift = held_out_gain(drifted, committed, plan_fn, EP, fallback=in_window_noise)
    assert float(g_drift.min()) > 0.05
    assert float(g_noise.mean()) < float(in_window_noise.mean())
    assert float(g_noise.mean()) < float(g_drift.mean()) / 3


def test_sglang_adapter_reads_the_committed_placement_from_host_copies():
    # Device tensors on "meta" have no data: reading them would fail, so the
    # snapshots must come from SGLang's host copies.
    p2l = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    l2p = torch.tensor([[[0, 2, -1, -1], [1, 3, -1, -1]], [[1, 3, -1, -1], [0, 2, -1, -1]]])
    live = SimpleNamespace(
        num_logical_experts=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
        physical_to_logical_map=p2l.to("meta"),
        physical_to_logical_map_cpu=p2l,
        logical_to_all_physical_map=l2p.to("meta"),
        logical_to_all_physical_map_cpu=l2p,
        logical_to_all_physical_map_num_valid=torch.full((2, 2), 2, device="meta"),
        mlb_routing_metadata={},
    )
    current = _committed_placement(live)
    assert sorted(current) == [0, 1]
    for layer in (0, 1):
        snap = current[layer]
        assert snap.physical_to_logical_map.device.type == "cpu"
        assert torch.equal(snap.physical_to_logical_map, p2l[layer])
        # only the used part of SGLang's padded candidate table is kept
        assert torch.equal(snap.logical_to_physical_candidates, l2p[layer][:, :2])
        assert snap.logical_to_physical_count.tolist() == [2, 2]
        assert snap.ep_size == 2
