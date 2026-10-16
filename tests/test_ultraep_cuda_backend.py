# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU checks for the JIT-built UltraEP kernels.

The placement solve is held to upstream UltraEP's own extension, bit for bit,
when that extension is importable; the sparse reroute is checked on a
non-default stream, which is where a wrong stream resolution would show.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def test_placement_backend_matches_ultraep_reference_extension():
    ultraep_extension = pytest.importorskip("ultra_ep._C")

    from evenmoe.kernels.ultraep import (
        UltraEPPlacementBackend,
        UltraEPPlacementTuning,
    )

    num_ranks = 8
    local_masters = 8
    local_redundant = 2
    num_nvl_ranks = 8
    logical = num_ranks * local_masters
    physical = num_ranks * (local_masters + local_redundant)
    device = torch.device("cuda")

    loads_per_rank = torch.zeros((num_ranks, logical), dtype=torch.int32, device=device)
    loads_per_rank[:, :16] = torch.arange(1, 17, dtype=torch.int32, device=device)
    loads_per_rank[0, 0] = 4096
    loads_per_rank[3, 7] = 2048
    global_loads = loads_per_rank.sum(dim=0, dtype=torch.int32)
    options = {"dtype": torch.int32, "device": device}
    outputs = (
        torch.empty(physical, **options),
        torch.empty((logical, num_ranks), **options),
        torch.empty(logical, **options),
        torch.empty((logical, num_ranks), **options),
        torch.empty((logical, num_ranks), **options),
        torch.empty((logical, num_ranks), **options),
    )
    tuning = UltraEPPlacementTuning(
        balance_threshold=1.0,
        locality_aware=True,
        min_tokens_per_replica=1024,
        allow_zero_master_quota=False,
        oracle_eps=0.01,
        kernel_stage=1,
    )
    backend = UltraEPPlacementBackend()
    stream = torch.cuda.Stream(device=device)

    for rank in (0, 3, 7):
        with torch.cuda.stream(stream):
            backend.solve(
                expert_loads=global_loads,
                expert_loads_per_rank=loads_per_rank,
                physical_to_logical_map=outputs[0],
                logical_to_physical_map=outputs[1],
                logical_replica_counts=outputs[2],
                logical_instance_quota=outputs[3],
                logical_instance_quota_prefix=outputs[4],
                rank_quota_prefix=outputs[5],
                num_ranks=num_ranks,
                num_local_master_experts=local_masters,
                num_local_redundant_experts=local_redundant,
                num_nvl_ranks=num_nvl_ranks,
                rank=rank,
                tuning=tuning,
            )
        stream.synchronize()
        reference = ultraep_extension.solve_placement_for_test(
            global_loads,
            loads_per_rank,
            num_ranks,
            local_masters,
            local_redundant,
            num_nvl_ranks,
            False,
            tuning.balance_threshold,
            tuning.min_tokens_per_replica,
            tuning.allow_zero_master_quota,
            tuning.locality_aware,
            tuning.oracle_eps,
            tuning.kernel_stage,
            rank,
        )
        torch.cuda.synchronize()
        for actual, expected in zip(outputs, reference, strict=True):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("topk_dtype", [torch.int32, torch.int64])
def test_sparse_reroute_quota_on_non_default_stream(topk_dtype):
    from evenmoe.kernels.ultraep import UltraEPPlacementBackend

    device = torch.device("cuda")
    backend = UltraEPPlacementBackend()
    logical_to_physical = torch.tensor(
        [[10, 11, 12], [20, 21, -1]],
        dtype=torch.int32,
        device=device,
    )
    replica_counts = torch.tensor([3, 2], dtype=torch.int32, device=device)
    rank_quota_prefix = torch.tensor(
        [[2, 4, 6], [1, 4, 4]],
        dtype=torch.int32,
        device=device,
    )
    original = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, -1, 9]],
        dtype=topk_dtype,
        device=device,
    )
    routed = torch.empty(original.shape, dtype=torch.int64, device=device)
    counters = torch.full((2,), 123, dtype=torch.int32, device=device)

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        backend.reroute_sparse(
            logical_topk_ids=original,
            routed_topk_ids=routed,
            logical_to_physical_map=logical_to_physical,
            logical_replica_counts=replica_counts,
            rank_quota_prefix=rank_quota_prefix,
            counters=counters,
        )
    stream.synchronize()

    assert sorted(routed[original == 0].tolist()) == [10, 10, 11, 11, 12, 12]
    assert sorted(routed[original == 1].tolist()) == [20, 21, 21, 21]
    assert routed[original == -1].item() == -1
    assert routed[original == 9].item() == 9
    assert counters.tolist() == [6, 4]
    assert original.tolist() == [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, -1, 9]]

    reused = torch.tensor(
        [[0] * 18, [1] * 12 + [-1] * 6],
        dtype=topk_dtype,
        device=device,
    )
    routed_reused = torch.empty(reused.shape, dtype=torch.int64, device=device)
    with torch.cuda.stream(stream):
        backend.reroute_sparse(
            logical_topk_ids=reused,
            routed_topk_ids=routed_reused,
            logical_to_physical_map=logical_to_physical,
            logical_replica_counts=replica_counts,
            rank_quota_prefix=rank_quota_prefix,
            counters=counters,
        )
    stream.synchronize()

    assert sorted(routed_reused[reused == 0].tolist()) == (
        [10] * 6 + [11] * 6 + [12] * 6
    )
    assert sorted(routed_reused[reused == 1].tolist()) == [20] * 3 + [21] * 9

    empty = torch.empty((0, 2), dtype=topk_dtype, device=device)
    routed_empty = torch.empty(empty.shape, dtype=torch.int64, device=device)
    counters.fill_(123)
    with torch.cuda.stream(stream):
        backend.reroute_sparse(
            logical_topk_ids=empty,
            routed_topk_ids=routed_empty,
            logical_to_physical_map=logical_to_physical,
            logical_replica_counts=replica_counts,
            rank_quota_prefix=rank_quota_prefix,
            counters=counters,
        )
    stream.synchronize()

    assert routed_empty.shape == (0, 2)
    assert counters.tolist() == [0, 0]
