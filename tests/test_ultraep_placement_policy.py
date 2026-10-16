# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from evenmoe import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)
from evenmoe.kernels.ultraep.config import UltraEPPlacementTuning
from evenmoe.policies.l1.ultraep import UltraEPPlacementPolicy


class _Backend:
    def __init__(self):
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["physical_to_logical_map"].copy_(torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.int32))
        kwargs["logical_to_physical_map"].fill_(-1)
        kwargs["logical_to_physical_map"][:, 0].copy_(torch.tensor([0, 1, 3, 4], dtype=torch.int32))
        kwargs["logical_replica_counts"].fill_(1)
        kwargs["rank_quota_prefix"].zero_()


def _request():
    return PlacementRequest(
        stats=PlacementStats(
            logical_count=torch.tensor(
                [
                    [[4, 2, 6, 0], [2, 2, 0, 4]],
                    [[2, 0, 2, 0], [0, 2, 2, 0]],
                ],
                dtype=torch.float32,
            )
        ),
        topology=PlacementTopology(
            num_physical_experts=6,
            num_local_physical_experts=3,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(name="ultraep", metadata={"rank": 0, "num_nvl_ranks": 2}),
    )


@pytest.mark.unit
def test_ultraep_placement_policy_uses_existing_placement_plan_contract():
    backend = _Backend()
    plan = UltraEPPlacementPolicy(backend=backend).plan(_request())

    assert len(backend.calls) == 2
    assert backend.calls[0]["expert_loads"].tolist() == [6, 2, 8, 0]
    assert backend.calls[0]["expert_loads_per_rank"].tolist() == [
        [4, 2, 6, 0],
        [2, 0, 2, 0],
    ]
    assert backend.calls[0]["expert_loads"].dtype == torch.int32
    assert backend.calls[0]["expert_loads_per_rank"].dtype == torch.int32
    assert plan.physical_to_logical_map.shape == (2, 6)
    assert plan.logical_to_all_physical_map.shape == (2, 4, 2)
    assert plan.metadata["rank_quota_prefix"].shape == (2, 4, 2)


@pytest.mark.integration
def test_ultraep_placement_policy_handles_multiple_layers():
    """plan() slices per_rank_loads[:, layer_id] -- the middle dimension of a
    [ep_size, num_layers, num_logical_experts] tensor -- which is only
    non-contiguous once num_layers > 1. The mock-backend tests above never
    exercise this against the real kernel, which is the only thing that
    enforces contiguity; a mock happily accepts any stride.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")

    from evenmoe.kernels.ultraep import UltraEPPlacementBackend

    ep_size, local_masters, local_redundant = 8, 8, 2
    num_logical = ep_size * local_masters
    num_physical = ep_size * (local_masters + local_redundant)
    num_layers = 2
    device = torch.device("cuda")

    generator = torch.Generator().manual_seed(0)
    # The two layers get independently-shuffled loads so a layer-1 solve that
    # silently reused layer-0's (or garbage) memory would very likely produce
    # a different, and likely invalid, placement instead of coincidentally
    # matching this one.
    logical_count = torch.stack(
        [
            torch.randint(1, 50, (ep_size, num_logical), generator=generator)
            for _ in range(num_layers)
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)

    request = PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_physical,
            num_local_physical_experts=local_masters + local_redundant,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(
            name="ultraep", metadata={"rank": 0, "num_nvl_ranks": ep_size}
        ),
    )
    plan = UltraEPPlacementPolicy(backend=UltraEPPlacementBackend()).plan(request)

    assert plan.physical_to_logical_map.shape == (num_layers, num_physical)
    assert int(plan.physical_to_logical_map.min()) >= 0
    assert int(plan.physical_to_logical_map.max()) < num_logical
    for layer_id in range(num_layers):
        # Every logical expert must own at least one physical slot in every
        # layer, independently -- what the contiguous() fix protects.
        assert (
            len(torch.unique(plan.physical_to_logical_map[layer_id]))
            == num_logical
        )
    assert not torch.equal(
        plan.physical_to_logical_map[0], plan.physical_to_logical_map[1]
    )


@pytest.mark.unit
def test_constructing_the_policy_does_not_read_the_environment(monkeypatch):
    """A malformed ULTRA_EP_* value must not fail construction.

    The orchestrator registers every L1 policy by name up front, so this
    constructor runs for every MoELoadBalancer regardless of the selected
    algorithm.  Reading the environment here would let a stray variable break
    a deployment that never selects ultraep.
    """
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")  # only 0 or 1 are valid
    UltraEPPlacementPolicy(backend=_Backend())


@pytest.mark.unit
def test_env_is_still_validated_on_the_first_solve(monkeypatch):
    """Deferring the read must not discard the validation."""
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")
    policy = UltraEPPlacementPolicy(backend=_Backend())
    with pytest.raises(ValueError, match="ULTRA_EP_QUOTA_KERNEL_STAGE"):
        policy.plan(_request())


@pytest.mark.unit
def test_explicit_tuning_still_bypasses_the_environment(monkeypatch):
    """An explicitly supplied tuning must win over ULTRA_EP_* entirely."""
    monkeypatch.setenv("ULTRA_EP_QUOTA_KERNEL_STAGE", "2")
    tuning = UltraEPPlacementTuning(
        balance_threshold=1.0,
        locality_aware=True,
        min_tokens_per_replica=1024,
        allow_zero_master_quota=False,
        oracle_eps=0.01,
        kernel_stage=1,
    )
    backend = _Backend()
    UltraEPPlacementPolicy(backend=backend, tuning=tuning).plan(_request())
    assert backend.calls[0]["tuning"] is tuning
