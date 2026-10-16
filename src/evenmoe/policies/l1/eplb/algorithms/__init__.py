# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek EPLB algorithm family — framework-neutral.

This package owns the algorithm-side formerly implemented in SGLang's
``python/sglang/srt/eplb/eplb_algorithms/``. All DeepSeek, vectorized, and
Elastic-EP-aware variants live here; frameworks provide runtime inputs such
as load statistics, topology, and active ranks through the common request.

The dispatcher and ``compute_algorithm`` ``auto`` rule are bit-for-bit copies
of sglang's behaviour, so a ``PlacementRequest`` going through
:class:`evenmoe.policies.l1.eplb.EPLBPlacementPlanner` produces the
same ``(physical_to_logical_map, logical_to_all_physical_map, expert_count)``
tensors as a direct call into sglang's ``eplb_algorithms.rebalance_experts``.
"""

from __future__ import annotations

from enum import Enum, auto

import torch

from evenmoe.policies.l1.eplb.algorithms import (
    deepseek,
    deepseek_vec,
    elasticity_aware,
)


class EplbAlgorithm(Enum):
    """Framework-neutral EPLB placement algorithms."""

    deepseek = auto()
    deepseek_hierarchical = auto()
    deepseek_vec = auto()
    deepseek_vec_hierarchical = auto()
    elasticity_aware = auto()
    elasticity_aware_hierarchical = auto()


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int | None,
    num_nodes: int,
    algorithm: EplbAlgorithm,
    active_ranks: torch.Tensor | None = None,
):
    """Dispatch to the requested EPLB variant.

    Signature matches ``sglang.srt.eplb.eplb_algorithms.rebalance_experts``
    so the adapter can call through without translation. ``active_ranks`` is
    only consumed by the ``elasticity_aware*`` variants; pass ``None`` for
    the others.

    Returns ``(physical_to_logical_map, logical_to_all_physical_map,
    expert_count)``.
    """

    hierarchical_algorithms = (
        EplbAlgorithm.deepseek_hierarchical,
        EplbAlgorithm.deepseek_vec_hierarchical,
        EplbAlgorithm.elasticity_aware_hierarchical,
    )
    if algorithm in hierarchical_algorithms and num_groups is None:
        raise ValueError(f"{algorithm.name} requires num_groups.")
    effective_num_groups = 1 if num_groups is None else num_groups

    if algorithm in (EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical):
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=effective_num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    if algorithm in (
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ):
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=effective_num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )

    if algorithm in (
        EplbAlgorithm.elasticity_aware,
        EplbAlgorithm.elasticity_aware_hierarchical,
    ):
        if active_ranks is None:
            raise ValueError(
                "elasticity_aware variants require active_ranks; the SGLang "
                "adapter sources this from ElasticEPStateManager."
            )
        return elasticity_aware.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=effective_num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.elasticity_aware_hierarchical,
            active_ranks=active_ranks,
        )

    raise NotImplementedError(f"Unknown EplbAlgorithm: {algorithm!r}")


def compute_algorithm(
    raw_algorithm: str,
    num_groups: int | None,
    num_nodes: int,
) -> EplbAlgorithm:
    """Resolve a string flag (e.g. SGLang ``--eplb-algorithm``) to an enum.

    Matches sglang's ``auto`` rule verbatim: hierarchical iff
    ``num_groups % num_nodes == 0``; otherwise non-hierarchical deepseek.
    """

    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    return EplbAlgorithm.deepseek
