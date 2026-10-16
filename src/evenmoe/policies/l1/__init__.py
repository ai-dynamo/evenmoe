# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 placement policy implementations.

Provides the DeepSeek EPLB family and UltraEP's placement solve. Future
planners (CRAFT, MoETuner, replica-budget allocators) implement the same
:class:`evenmoe.core.l1.PlacementPlanner` protocol.
"""

from __future__ import annotations

from evenmoe.policies.l1.eplb import (
    SUPPORTED_PLACEMENT_POLICIES,
    EplbAlgorithm,
    EPLBPlacementPlanner,
    compute_algorithm,
    rebalance_experts,
)
from evenmoe.policies.l1.ultraep import RefreshGate, UltraEPPlacementPolicy

__all__ = [
    "SUPPORTED_PLACEMENT_POLICIES",
    "EPLBPlacementPlanner",
    "EplbAlgorithm",
    "RefreshGate",
    "UltraEPPlacementPolicy",
    "compute_algorithm",
    "rebalance_experts",
]
