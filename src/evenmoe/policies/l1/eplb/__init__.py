# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek EPLB family — L1 placement planner.

Exports a framework-neutral :class:`EPLBPlacementPlanner` plus the vendored
algorithm dispatcher (``rebalance_experts``, ``compute_algorithm``,
``EplbAlgorithm``). SGLang adapter at
:mod:`evenmoe.adapters.sglang.eplb` is the only framework-specific
glue around this package.
"""

from __future__ import annotations

from evenmoe.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    rebalance_experts,
)
from evenmoe.policies.l1.eplb.planner import (
    SUPPORTED_PLACEMENT_POLICIES,
    EPLBPlacementPlanner,
)

__all__ = [
    "SUPPORTED_PLACEMENT_POLICIES",
    "EPLBPlacementPlanner",
    "EplbAlgorithm",
    "compute_algorithm",
    "rebalance_experts",
]
