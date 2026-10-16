# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

TensorLike = Any


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


class RoutingStage(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    IDLE = "idle"
    SPECULATIVE = "speculative"

    @classmethod
    def from_str(cls, value: RoutingStage | str) -> RoutingStage:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"RoutingStage.from_str expects a RoutingStage or str, got {type(value).__name__}."
            )
        return cls(value.lower())


class RoutingPhase(str, Enum):
    """Ordered phases in the framework-neutral L2 pipeline."""

    PHYSICAL_ROUTING = "physical_routing"
    SHARED_EXPERT = "shared_expert"


@dataclass(frozen=True)
class RoutingPolicyConfig:
    """Selection and stable configuration for one L2 policy."""

    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    enabled_stages: frozenset[RoutingStage] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if self.enabled_stages is None:
            return
        enabled = frozenset(RoutingStage.from_str(stage) for stage in self.enabled_stages)
        if not enabled:
            raise ValueError("RoutingPolicyConfig.enabled_stages cannot be empty.")
        object.__setattr__(self, "enabled_stages", enabled)

    @classmethod
    def from_value(cls, value: RoutingPolicyConfig | str) -> RoutingPolicyConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError(
            "RoutingPolicyConfig.from_value expects RoutingPolicyConfig or str, "
            f"got {type(value).__name__}."
        )

    @property
    def is_disabled(self) -> bool:
        return self.name == self.DISABLED

    def allows_stage(self, stage: RoutingStage | None) -> bool:
        if self.enabled_stages is None:
            return True
        return stage is not None and stage in self.enabled_stages

    def should_run_for_stage(self, stage: RoutingStage | None) -> bool:
        return not self.is_disabled and self.allows_stage(stage)


@dataclass(frozen=True)
class PlacementSnapshot:
    """Current, committed placement for one layer.

    A snapshot must be derived from the framework's live placement metadata
    after weight loading or migration has completed. It is never constructed
    from an uncommitted :class:`PlacementPlan`.
    """

    layer_id: int
    num_logical_experts: int
    num_physical_experts: int
    ep_size: int
    num_local_physical_experts: int
    physical_to_logical_map: TensorLike
    # The final dimension may be padded beyond the largest live replica count;
    # consumers must honor logical_to_physical_count rather than its width.
    logical_to_physical_candidates: TensorLike
    logical_to_physical_count: TensorLike
    default_physical_for_logical: TensorLike | None = None
    physical_to_rank_map: TensorLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("PlacementSnapshot.layer_id must be non-negative.")
        for name in (
            "num_logical_experts",
            "num_physical_experts",
            "ep_size",
            "num_local_physical_experts",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"PlacementSnapshot.{name} must be positive.")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class PlacementStats:
    """Framework-completed statistics consumed by an L1 policy."""

    logical_count: TensorLike | None = None
    physical_count: TensorLike | None = None
    rank_load: TensorLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class PlacementTopology:
    """Runtime placement geometry supplied by the serving framework."""

    num_physical_experts: int
    num_local_physical_experts: int
    num_groups: int | None
    num_nodes: int
    active_ranks: TensorLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_physical_experts <= 0:
            raise ValueError("PlacementTopology.num_physical_experts must be positive.")
        if self.num_local_physical_experts <= 0:
            raise ValueError("PlacementTopology.num_local_physical_experts must be positive.")
        if self.num_physical_experts % self.num_local_physical_experts != 0:
            raise ValueError(
                "PlacementTopology.num_physical_experts must be divisible by "
                "num_local_physical_experts."
            )
        if self.num_groups is not None and self.num_groups <= 0:
            raise ValueError("PlacementTopology.num_groups must be positive or None.")
        if self.num_nodes <= 0:
            raise ValueError("PlacementTopology.num_nodes must be positive.")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class PlacementPolicyConfig:
    """Selection and stable configuration for one L1 placement policy."""

    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @classmethod
    def from_value(cls, value: PlacementPolicyConfig | str) -> PlacementPolicyConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError(
            "PlacementPolicyConfig.from_value expects PlacementPolicyConfig or str, "
            f"got {type(value).__name__}."
        )

    @property
    def is_disabled(self) -> bool:
        return self.name == self.DISABLED


@dataclass(frozen=True, kw_only=True)
class PlacementRequest:
    """Framework-neutral L1 planning input.

    ``policy`` overrides the balancer's configured placement policy for this
    planning call. Leave it disabled to use the configured default.
    """

    stats: PlacementStats
    topology: PlacementTopology
    policy: PlacementPolicyConfig | str = PlacementPolicyConfig.DISABLED
    current: Mapping[int, PlacementSnapshot] | None = None
    hbm_budget_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", PlacementPolicyConfig.from_value(self.policy))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class PlacementPlan:
    """Uncommitted candidate placement produced by an L1 policy."""

    physical_to_logical_map: TensorLike
    logical_to_all_physical_map: TensorLike
    logical_to_physical_count: TensorLike
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class RoutingLoadHistory:
    """Optional long-lived load view for future L2 policies."""

    live_rank_load: TensorLike | None = None
    historical_rank_load: TensorLike | None = None
    historical_expert_load: TensorLike | None = None
    historical_rank_expert_load: TensorLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True)
class RoutingRequest:
    """Framework-neutral input to the ordered L2 policy pipeline.

    ``logical_topk_ids`` is always the framework's logical TopK result.
    The framework normally supplies logical TopK, weights, the live committed
    placement and token count. The remaining load fields are internal policy
    inputs populated by the EvenMoE orchestrator while it executes the pipeline.
    """

    layer_id: int
    logical_topk_ids: TensorLike
    topk_weights: TensorLike
    policies: tuple[RoutingPolicyConfig | str, ...] = ()
    placement: PlacementSnapshot | None = None
    routed_physical_topk_ids: TensorLike | None = None
    stage: RoutingStage | None = None
    routed_rank_load: TensorLike | None = None
    active_rank_token_count: TensorLike | None = None
    global_logical_count: TensorLike | None = None
    token_count: TensorLike | None = None
    lb_token_mask: TensorLike | None = None
    routed_scaling_factor: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("RoutingRequest.layer_id must be non-negative.")
        normalized = tuple(RoutingPolicyConfig.from_value(value) for value in self.policies)
        object.__setattr__(self, "policies", normalized)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class RoutingDecision:
    """Final physical routing result returned by the L2 orchestrator."""

    routed_physical_topk_ids: TensorLike
    topk_weights: TensorLike
    shared_expert_rank: TensorLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
