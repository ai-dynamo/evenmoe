# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evenmoe.core.types import PlacementPlan, PlacementRequest

_L1_DATA_VIEWS = ("global", "per_source_rank")


@runtime_checkable
class PlacementPlanner(Protocol):
    """Framework-neutral L1 candidate-placement policy.

    Two optional extensions let the framework ask the planner instead of
    special-casing it.

    The class attribute ``logical_count`` names the data view of
    ``request.stats.logical_count`` the planner solves from, in the vocabulary
    of :attr:`RoutingPolicySpec.logical_count`: ``"global"`` (the default when
    absent) is the EP-wide sum, ``[1, layers, logical_experts]``;
    ``"per_source_rank"`` is EP-wide but keeps the source-rank axis,
    ``[ep_size, layers, logical_experts]``. ``ultraep`` declares
    ``"per_source_rank"``: its solve also computes each source rank's
    per-replica token quota (``rank_quota_prefix``) for its L2 half, filling
    each replica from its host rank's own tokens first, so it needs to know
    which rank the demand comes from. Frameworks read the view through
    :meth:`MoELoadBalancer.placement_logical_count`.

    The method ``initial_plan(topology, *, num_logical_experts, num_layers) ->
    PlacementPlan`` returns the planner's own placement before any statistics
    exist; :meth:`MoELoadBalancer.plan_initial_placement` calls it. Without it
    the framework keeps its default initial placement.
    """

    def plan(self, request: PlacementRequest) -> PlacementPlan: ...

