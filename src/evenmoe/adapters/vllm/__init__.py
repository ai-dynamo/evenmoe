# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless translation between vLLM values and EvenMoE framework-neutral types.

This package must not import vLLM.  Glue on the vLLM side owns every framework
call; these helpers only reshape plain values and tensors.
"""

from evenmoe.adapters.vllm.placement import (
    VllmPlacementMaps,
    to_placement_request,
    to_vllm_maps,
    to_vllm_physical_to_logical,
)
from evenmoe.adapters.vllm.routing import (
    build_physical_to_rank_map,
    collapse_candidates_to_local,
    nearest_replica_table,
    stage_from_vllm,
    to_placement_snapshot,
    to_routing_request,
    to_vllm_shared_expert_rank,
    to_vllm_topk_ids,
)

__all__ = [
    "VllmPlacementMaps",
    "build_physical_to_rank_map",
    "collapse_candidates_to_local",
    "nearest_replica_table",
    "stage_from_vllm",
    "to_placement_request",
    "to_placement_snapshot",
    "to_routing_request",
    "to_vllm_maps",
    "to_vllm_physical_to_logical",
    "to_vllm_shared_expert_rank",
    "to_vllm_topk_ids",
]
