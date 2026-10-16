# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    rebalance_experts,
)
from evenmoe.policies.l1.eplb.algorithms.deepseek import balanced_packing

_COUNTS = torch.tensor(
    [
        [[20, 3, 9, 4, 15, 2, 8, 6], [5, 16, 2, 7, 3, 14, 1, 10]],
        [[18, 4, 11, 2, 13, 5, 7, 9], [6, 12, 3, 8, 4, 15, 2, 9]],
    ],
    dtype=torch.float32,
)
_ACTIVE_RANKS = torch.tensor([True, False, True, True])

# Golden physical maps captured from SGLang before its duplicate algorithm
# package was removed. Full tensor parity against that implementation was
# also checked for physical/logical maps and replica counts during migration.
_SGLANG_GOLDEN_PHYSICAL = {
    EplbAlgorithm.deepseek: [
        [6, 0, 3, 7, 2, 2, 4, 0, 1, 4, 0, 5],
        [5, 7, 2, 5, 3, 3, 1, 0, 6, 1, 7, 4],
    ],
    EplbAlgorithm.deepseek_hierarchical: [
        [0, 2, 1, 0, 2, 3, 7, 6, 6, 4, 4, 5],
        [1, 0, 2, 1, 3, 3, 5, 7, 4, 5, 7, 6],
    ],
    EplbAlgorithm.deepseek_vec: [
        [6, 2, 2, 7, 0, 1, 4, 0, 5, 4, 0, 3],
        [3, 1, 1, 5, 1, 4, 5, 7, 2, 0, 7, 6],
    ],
    EplbAlgorithm.deepseek_vec_hierarchical: [
        [0, 2, 1, 0, 2, 3, 6, 7, 7, 4, 4, 5],
        [1, 3, 3, 1, 0, 2, 5, 7, 4, 5, 7, 6],
    ],
    EplbAlgorithm.elasticity_aware: [
        [4, 7, 3, 0, 0, 0, 2, 6, 1, 0, 0, 5],
        [1, 0, 6, 0, 0, 0, 7, 5, 2, 3, 5, 4],
    ],
    EplbAlgorithm.elasticity_aware_hierarchical: [
        [4, 7, 3, 0, 0, 0, 2, 6, 1, 0, 0, 5],
        [1, 0, 6, 0, 0, 0, 7, 5, 2, 3, 5, 4],
    ],
}


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", list(EplbAlgorithm))
def test_migrated_algorithms_match_sglang_golden(algorithm):
    physical_to_logical, logical_to_physical, logical_count = rebalance_experts(
        tokens_per_expert=_COUNTS,
        num_physical_experts=12,
        num_local_physical_experts=3,
        num_groups=4,
        num_nodes=2,
        algorithm=algorithm,
        active_ranks=_ACTIVE_RANKS,
    )

    assert torch.equal(
        physical_to_logical,
        torch.tensor(_SGLANG_GOLDEN_PHYSICAL[algorithm], dtype=physical_to_logical.dtype),
    )
    assert logical_to_physical.shape[:2] == (2, 8)
    assert logical_count.shape == (2, 8)
    assert torch.all(logical_count >= 1)

    for layer_id in range(logical_count.shape[0]):
        for logical_id in range(logical_count.shape[1]):
            count = int(logical_count[layer_id, logical_id])
            physical_ids = logical_to_physical[layer_id, logical_id, :count]
            assert torch.all(physical_ids >= 0)
            assert len(physical_ids.unique()) == count


@pytest.mark.unit
def test_balanced_packing_uses_every_slot_once_and_balances_heavy_items():
    weight = torch.tensor([[9.0, 1.0, 1.0, 1.0], [8.0, 7.0, 2.0, 1.0]])
    pack_index, rank_in_pack = balanced_packing(weight, num_packs=2)

    assert pack_index.shape == weight.shape
    assert rank_in_pack.shape == weight.shape
    for layer_id in range(weight.shape[0]):
        slots = set(zip(pack_index[layer_id].tolist(), rank_in_pack[layer_id].tolist()))
        assert slots == {(0, 0), (0, 1), (1, 0), (1, 1)}

    pack_weights = torch.zeros(2)
    for item, pack in enumerate(pack_index[0].tolist()):
        pack_weights[pack] += weight[0, item]
    assert pack_weights.max().item() == 10.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("num_layers", "num_items", "num_packs", "uniform"),
    [
        (3, 8, 4, False),
        (2, 6, 3, False),
        (1, 4, 2, False),
        (16, 8, 4, False),
        (3, 8, 4, True),
        (2, 6, 6, False),
    ],
)
def test_balanced_packing_preserves_all_assignment_invariants(
    num_layers, num_items, num_packs, uniform
):
    torch.manual_seed(7)
    weight = torch.ones(num_layers, num_items) if uniform else torch.rand(num_layers, num_items)
    pack_index, rank_in_pack = balanced_packing(weight, num_packs=num_packs)
    items_per_pack = num_items // num_packs

    assert pack_index.shape == weight.shape
    assert rank_in_pack.shape == weight.shape
    assert torch.all((pack_index >= 0) & (pack_index < num_packs))
    assert torch.all((rank_in_pack >= 0) & (rank_in_pack < items_per_pack))
    for layer_id in range(num_layers):
        assert torch.equal(
            torch.bincount(pack_index[layer_id], minlength=num_packs),
            torch.full((num_packs,), items_per_pack),
        )
        slots = set(zip(pack_index[layer_id].tolist(), rank_in_pack[layer_id].tolist()))
        assert len(slots) == num_items


@pytest.mark.unit
def test_balanced_packing_is_deterministic_for_uniform_weights():
    weight = torch.ones(3, 8)
    first = balanced_packing(weight, num_packs=4)
    second = balanced_packing(weight, num_packs=4)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.unit
def test_auto_resolution_stays_in_mlb():
    assert (
        compute_algorithm("auto", num_groups=8, num_nodes=2) is EplbAlgorithm.deepseek_hierarchical
    )
    assert compute_algorithm("auto", num_groups=None, num_nodes=2) is EplbAlgorithm.deepseek
    assert compute_algorithm("deepseek_vec", 8, 2) is EplbAlgorithm.deepseek_vec


@pytest.mark.unit
def test_elasticity_aware_requires_framework_supplied_active_ranks():
    with pytest.raises(ValueError, match="require active_ranks"):
        rebalance_experts(
            tokens_per_expert=_COUNTS,
            num_physical_experts=12,
            num_local_physical_experts=3,
            num_groups=4,
            num_nodes=2,
            algorithm=EplbAlgorithm.elasticity_aware,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "algorithm",
    [
        EplbAlgorithm.deepseek_hierarchical,
        EplbAlgorithm.deepseek_vec_hierarchical,
        EplbAlgorithm.elasticity_aware_hierarchical,
    ],
)
def test_hierarchical_algorithms_require_group_topology(algorithm):
    with pytest.raises(ValueError, match="requires num_groups"):
        rebalance_experts(
            tokens_per_expert=_COUNTS,
            num_physical_experts=12,
            num_local_physical_experts=3,
            num_groups=None,
            num_nodes=2,
            algorithm=algorithm,
            active_ranks=_ACTIVE_RANKS,
        )
