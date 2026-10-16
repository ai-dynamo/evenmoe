# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA implementation of the UltraEP kernel contract."""

from __future__ import annotations

import torch

from evenmoe.kernels.ultraep import cuda_placement
from evenmoe.kernels.ultraep.config import UltraEPPlacementTuning


class UltraEPPlacementBackend:
    """Framework-independent implementation of the :class:`UltraEPKernels` contract.

    Stateless. Each kernel is JIT-compiled once per process on its first call
    and launched on the current stream of the tensors' device. Runtime state and
    expert communication intentionally live outside this class.
    """

    def solve(
        self,
        *,
        expert_loads: torch.Tensor,
        expert_loads_per_rank: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_counts: torch.Tensor,
        logical_instance_quota: torch.Tensor,
        logical_instance_quota_prefix: torch.Tensor,
        rank_quota_prefix: torch.Tensor,
        num_ranks: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        num_nvl_ranks: int,
        rank: int,
        tuning: UltraEPPlacementTuning,
    ) -> None:
        cuda_placement.solve_placement(
            expert_loads=expert_loads,
            expert_loads_per_rank=expert_loads_per_rank,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_counts=logical_replica_counts,
            logical_instance_quota=logical_instance_quota,
            logical_instance_quota_prefix=logical_instance_quota_prefix,
            rank_quota_prefix=rank_quota_prefix,
            num_ranks=num_ranks,
            num_local_master_experts=num_local_master_experts,
            num_local_redundant_experts=num_local_redundant_experts,
            num_nvl_ranks=num_nvl_ranks,
            rank=rank,
            tuning=tuning,
        )

    def reroute_sparse(
        self,
        *,
        logical_topk_ids: torch.Tensor,
        routed_topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_counts: torch.Tensor,
        rank_quota_prefix: torch.Tensor,
        counters: torch.Tensor,
    ) -> None:
        cuda_placement.reroute_sparse(
            logical_topk_ids=logical_topk_ids,
            routed_topk_ids=routed_topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_counts=logical_replica_counts,
            rank_quota_prefix=rank_quota_prefix,
            counters=counters,
        )
