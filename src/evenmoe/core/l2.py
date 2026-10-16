# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evenmoe.core.types import (
    RoutingDecision,
    RoutingLoadHistory,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)


@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):
    """Optional source for future policies that consume cross-batch history."""

    def load_history(self, layer_id: int) -> RoutingLoadHistory: ...


@runtime_checkable
class RoutingCollectives(Protocol):
    """Framework-owned collective transport injected into EvenMoE.

    EvenMoE owns the count tensors, payload packing, and the point in the routing
    pipeline at which communication is required.  The serving framework owns
    the process group and the concrete all-reduce implementation.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_reduce_sum(self, payload: object) -> object: ...


@runtime_checkable
class GlobalLogicalCountConsumer(Protocol):
    """A router whose decision depends on the EP-wide per-logical token count.

    Gathering that count costs one EP collective per layer per forward, so the
    orchestrator pays for it only when a router both declares this protocol and
    answers that the *current placement* still needs it. An LP over replicated
    experts, for instance, is degenerate when no logical expert has more than
    one replica; counting for it is pure overhead.

    Declaring the need is a capability, not a name, so the pipeline stays open
    to policies the orchestrator has never heard of.

    This is the orchestrator's per-request check against one router instance.
    The framework-facing, per-pipeline declaration of the same underlying fact
    is :attr:`~evenmoe.core.routing_pipeline.RoutingCapabilities.consumes_global_logical_count`.
    """

    def needs_global_logical_count(self, request: RoutingRequest) -> bool: ...


@runtime_checkable
class TokenRouter(Protocol):
    """One framework-neutral policy in the ordered L2 pipeline."""

    phase: RoutingPhase

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision: ...
