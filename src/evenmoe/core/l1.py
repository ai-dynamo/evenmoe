# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evenmoe.core.types import PlacementPlan, PlacementRequest


@runtime_checkable
class PlacementPlanner(Protocol):
    """Framework-neutral L1 candidate-placement policy."""

    def plan(self, request: PlacementRequest) -> PlacementPlan: ...
