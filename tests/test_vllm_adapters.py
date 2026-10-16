# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe import PlacementPlan, RoutingDecision, RoutingStage
from evenmoe.adapters.vllm import (
    collapse_candidates_to_local,
    nearest_replica_table,
    stage_from_vllm,
    to_placement_request,
    to_placement_snapshot,
    to_routing_request,
    to_vllm_maps,
    to_vllm_physical_to_logical,
    to_vllm_topk_ids,
)


class _LayerState:
    """Stand-in for vLLM's per-layer ``EplbLayerState``.

    vLLM slices the model-level maps by layer in ``set_layer_state``, so the
    adapter receives already-per-layer tensors.
    """

    def __init__(self, candidates: torch.Tensor, counts: torch.Tensor):
        self.logical_to_physical_map = candidates
        self.logical_replica_count = counts


@pytest.mark.unit
def test_l1_request_is_a_plain_translation():
    # vLLM hands its policy the globally reduced logical load directly.
    weight = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    request = to_placement_request(
        weight,
        num_replicas=8,
        num_ranks=2,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )

    assert request.stats.logical_count is weight
    assert request.topology.num_physical_experts == 8
    # vLLM speaks in ranks; EvenMoE's topology is per-rank slots.
    assert request.topology.num_local_physical_experts == 4
    assert request.policy.name == "deepseek"
    assert request.metadata["framework"] == "vllm"


@pytest.mark.unit
def test_l1_request_carries_caller_rank_for_ultraep():
    # ultraep's placement kernel solves once per EP rank and validates the
    # value is in range, so the request must carry which rank is calling --
    # every other EvenMoE policy ignores it.
    request = to_placement_request(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        num_replicas=8,
        num_ranks=2,
        num_groups=1,
        num_nodes=1,
        algorithm="ultraep",
        ep_rank=1,
    )
    assert request.policy.metadata["rank"] == 1


@pytest.mark.unit
def test_l1_request_omits_rank_metadata_when_not_given():
    request = to_placement_request(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        num_replicas=8,
        num_ranks=2,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )
    assert "rank" not in request.policy.metadata


@pytest.mark.unit
def test_l1_request_carries_old_placement_for_glue():
    old = torch.tensor([[0, 1, 2, 3]])
    request = to_placement_request(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        num_replicas=4,
        num_ranks=2,
        num_groups=1,
        num_nodes=1,
        old_physical_to_logical_map=old,
    )
    # No L1 policy consumes a previous placement yet, so it is carried in
    # metadata rather than being turned into snapshots nothing reads.
    assert request.metadata["old_physical_to_logical_map"] is old


@pytest.mark.unit
def test_l1_rejects_geometry_vllm_cannot_produce():
    with pytest.raises(ValueError, match="divisible"):
        to_placement_request(
            torch.zeros(1, 4),
            num_replicas=7,
            num_ranks=2,
            num_groups=1,
            num_nodes=1,
        )
    with pytest.raises(ValueError, match="positive num_ranks"):
        to_placement_request(
            torch.zeros(1, 4),
            num_replicas=8,
            num_ranks=0,
            num_groups=1,
            num_nodes=1,
        )


@pytest.mark.unit
def test_l1_post_adapter_returns_the_single_tensor_vllm_wants():
    plan = PlacementPlan(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]]),
        logical_to_physical_count=torch.tensor([[2, 2]]),
    )
    maps = to_vllm_maps(plan)
    assert maps.physical_to_logical_map is plan.physical_to_logical_map
    # vLLM derives its own logical maps, so the policy contract is one tensor.
    assert to_vllm_physical_to_logical(plan) is plan.physical_to_logical_map


@pytest.mark.unit
def test_placement_snapshot_from_vllm_layer_state():
    candidates = torch.tensor([[0, 2], [1, -1]])
    counts = torch.tensor([2, 1])
    snapshot = to_placement_snapshot(
        _LayerState(candidates, counts),
        layer_id=3,
        physical_to_logical_map=torch.tensor([0, 1, 0, -1]),
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
    )

    assert snapshot.layer_id == 3
    assert snapshot.num_local_physical_experts == 2
    assert snapshot.logical_to_physical_candidates is candidates
    assert snapshot.logical_to_physical_count is counts
    # vLLM keeps no precomputed per-rank dispatch table.
    assert snapshot.default_physical_for_logical is None


@pytest.mark.unit
def test_placement_snapshot_requires_eplb_state():
    with pytest.raises(ValueError, match="EPLB is probably not enabled"):
        to_placement_snapshot(
            _LayerState(None, None),
            layer_id=0,
            physical_to_logical_map=torch.tensor([0, 1]),
            num_logical_experts=2,
            num_physical_experts=2,
            ep_size=1,
        )


@pytest.mark.unit
def test_routing_request_normalizes_stage():
    request = to_routing_request(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.tensor([[0.5, 0.5]]),
        stage="decode",
    )
    assert request.stage is RoutingStage.DECODE
    assert stage_from_vllm("prefill") is RoutingStage.PREFILL
    assert stage_from_vllm(None) is None
    # vLLM has no dedicated idle routing mode; unknown labels stay unset
    # rather than being guessed at.
    assert stage_from_vllm("something_else") is None


@pytest.mark.unit
def test_routing_output_is_the_physical_ids():
    ids = torch.tensor([[0, 3]])
    decision = RoutingDecision(
        routed_physical_topk_ids=ids,
        topk_weights=torch.tensor([[0.5, 0.5]]),
    )
    assert to_vllm_topk_ids(decision) is ids


@pytest.mark.unit
def test_shared_expert_decision_is_handed_back_not_dropped():
    """A shared-expert rank reaches the framework through
    `to_vllm_shared_expert_rank`; `to_vllm_topk_ids` answers only about
    physical ids, and a decision without a rank yields None."""
    from evenmoe.adapters.vllm import to_vllm_shared_expert_rank

    rank = torch.tensor([0, 3], dtype=torch.int64)
    decision = RoutingDecision(
        routed_physical_topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        topk_weights=torch.zeros(2, 2),
        shared_expert_rank=rank,
    )

    ids = to_vllm_topk_ids(decision)
    assert ids.shape == (2, 2)

    got = to_vllm_shared_expert_rank(decision)
    assert got is not None, "shared-expert rank was dropped by the adapter"
    assert torch.equal(got, rank)

    # A decision without one answers None rather than inventing a rank.
    plain = RoutingDecision(
        routed_physical_topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        topk_weights=torch.zeros(2, 2),
    )
    assert to_vllm_shared_expert_rank(plain) is None


@pytest.mark.unit
def test_nearest_replica_table_prefers_local_then_first_valid():
    # 4 ranks x 2 slots. Logical 0 lives on rank 0 and rank 2; logical 1 only
    # on rank 1; logical 2 only on rank 3.
    candidates = torch.tensor([[0, 4, -1], [2, -1, -1], [6, -1, -1]])
    counts = torch.tensor([2, 1, 1])

    # From rank 2, logical 0 must resolve to its local replica (slot 4).
    table = nearest_replica_table(candidates, counts, ep_rank=2, num_local_physical_experts=2)
    assert int(table[0]) == 4
    # Experts with no local replica fall back to the first valid one.
    assert int(table[1]) == 2
    assert int(table[2]) == 6

    # From rank 0 the same expert resolves to the other replica.
    table0 = nearest_replica_table(candidates, counts, ep_rank=0, num_local_physical_experts=2)
    assert int(table0[0]) == 0


@pytest.mark.unit
def test_collapse_keeps_one_local_replica_and_leaves_remote_lists_whole():
    # 4 ranks x 2 slots. Logical 0 is replicated on rank 0 and rank 2;
    # logical 1 lives only on rank 1.
    candidates = torch.tensor([[0, 4, -1], [2, -1, -1]])
    counts = torch.tensor([2, 1])

    got, got_counts = collapse_candidates_to_local(
        candidates, counts, ep_rank=2, num_local_physical_experts=2
    )
    # Rank 2 holds a copy of logical 0, so it keeps only that one.
    assert got[0].tolist() == [4, -1, -1]
    assert int(got_counts[0]) == 1
    # Logical 1 is remote from here, so its list survives untouched.
    assert got[1].tolist() == [2, -1, -1]
    assert int(got_counts[1]) == 1

    # A rank holding neither replica keeps the full choice.
    got3, counts3 = collapse_candidates_to_local(
        candidates, counts, ep_rank=3, num_local_physical_experts=2
    )
    assert got3[0].tolist() == [0, 4, -1]
    assert int(counts3[0]) == 2


@pytest.mark.unit
def test_collapse_does_not_synchronize_on_candidate_values():
    """The sentinel must come from the dtype, not from ``candidates.max()``.

    Rank-dispatch callers run this per layer inside a forward pass, where
    reading a value off the device stalls the pipeline every call. Guard it by
    construction: a tensor that raises when read cannot be read.
    """

    class _NoItem(torch.Tensor):
        @staticmethod
        def __new__(cls, data):
            return torch.Tensor._make_subclass(cls, data)

        def item(self):  # pragma: no cover - must never be reached
            raise AssertionError("collapse_candidates_to_local synced the device")

    candidates = _NoItem(torch.tensor([[0, 4, -1], [2, -1, -1]]))
    counts = torch.tensor([2, 1])
    got, _ = collapse_candidates_to_local(
        candidates, counts, ep_rank=2, num_local_physical_experts=2
    )
    # `got` inherits the subclass, so compare without going back through it.
    assert torch.equal(
        got.as_subclass(torch.Tensor)[0], torch.tensor([4, -1, -1])
    )


@pytest.mark.unit
def test_nearest_replica_table_ignores_padding_beyond_count():
    # Padded columns must never be selected even when they hold stale ids.
    candidates = torch.tensor([[5, 1]])
    counts = torch.tensor([1])
    table = nearest_replica_table(candidates, counts, ep_rank=0, num_local_physical_experts=2)
    # Slot 1 would be "local" for rank 0, but it is past the valid count.
    assert int(table[0]) == 5


def test_single_node_fallback_shares_evenly_as_sglang_does():
    """On one node the ranks without a local copy split the replicas evenly.

    SGLang's same-node tier applies only when it narrows the candidate set
    (v0.5.16). On a single node every candidate is on the node, so those ranks
    fall through to `_fair_choices`: with 8 ranks and 2 replicas, the six
    non-owners land three on each copy. The alignment tests pin the same rule
    against the vendored reference rank for rank.
    """
    # Logical 0 is replicated on slot 0 (rank 0) and slot 8 (rank 4).
    candidates = torch.tensor([[0, 8, -1]])
    counts = torch.tensor([2])
    picks = [
        int(nearest_replica_table(
            candidates, counts, ep_rank=r, num_local_physical_experts=2, ep_size=8
        )[0])
        for r in range(8)
    ]
    # Owners still take their own copy.
    assert picks[0] == 0
    assert picks[4] == 8
    # Everyone else is shared evenly between the two copies.
    assert sorted(picks[r] for r in (1, 2, 3, 5, 6, 7)) == [0, 0, 0, 8, 8, 8]
