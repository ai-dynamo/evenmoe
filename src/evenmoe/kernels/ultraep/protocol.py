# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution contract implemented by an UltraEP kernel backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evenmoe.core.types import TensorLike
from evenmoe.kernels.ultraep.config import UltraEPPlacementTuning


@runtime_checkable
class UltraEPKernels(Protocol):
    """The two kernels the ``ultraep`` policy pair calls.

    The policies own the decisions and every tensor involved: the L1 half
    allocates the placement and quota outputs ``solve`` fills, the L2 half owns
    the per-layer counters ``reroute_sparse`` uses as scratch. An implementation
    only launches; ``UltraEPPlacementBackend`` is the CUDA one.
    """

    def solve(
        self,
        *,
        expert_loads: TensorLike,
        expert_loads_per_rank: TensorLike,
        physical_to_logical_map: TensorLike,
        logical_to_physical_map: TensorLike,
        logical_replica_counts: TensorLike,
        logical_instance_quota: TensorLike,
        logical_instance_quota_prefix: TensorLike,
        rank_quota_prefix: TensorLike,
        num_ranks: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        num_nvl_ranks: int,
        rank: int,
        tuning: UltraEPPlacementTuning,
    ) -> None: ...

    def reroute_sparse(
        self,
        *,
        logical_topk_ids: TensorLike,
        routed_topk_ids: TensorLike,
        logical_to_physical_map: TensorLike,
        logical_replica_counts: TensorLike,
        rank_quota_prefix: TensorLike,
        counters: TensorLike,
    ) -> None: ...
