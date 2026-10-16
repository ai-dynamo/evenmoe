# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarations a routing policy publishes about itself.

Everything the pipeline, the orchestrator, and framework glue need to know
about one L2 policy lives in its :class:`RoutingPolicySpec`, next to the
policy's own code. Core never branches on a policy name; it reads these
declarations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from evenmoe.core.l1 import PlacementPlanner
from evenmoe.core.l2 import TokenRouter
from evenmoe.core.types import RoutingPhase, RoutingPolicyConfig, TensorLike

_DATA_VIEWS = (None, "local", "global")
_GRAPH_STABILITIES = ("stable", "realloc_on_placement_change")


@dataclass(frozen=True)
class ExpertDeploymentConfig:
    """What a framework can tell EvenMoE about a deployment before routing starts.

    Deliberately small: this is consulted while the framework is still reading
    its configuration, so it may only contain what is known that early.
    """

    num_redundant_experts: int = 0
    routes_shared_expert: bool = False
    """Whether the framework can dispatch the shared expert to a chosen rank.
    False on runtimes that replicate it per rank."""
    carries_placement_metadata: bool = False
    """Whether the framework threads ``PlacementPlan.metadata`` into the
    committed ``PlacementSnapshot.metadata``. An L1 policy may publish state its
    own L2 half needs there (``ultraep`` puts its per-replica quota there); a
    framework that does not carry it silently degrades that policy."""


@dataclass(frozen=True)
class RoutingCapabilities:
    """Framework-visible requirements of one configured L2 pipeline.

    A framework acts on these declarations instead of on assumptions about
    what a policy is. Keying behaviour off a policy name both taxes policies
    that do not need the behaviour and leaves policies that do need it unable
    to ask.
    """

    requires_post_topk_routing: bool = False
    requires_placement_state: bool = False
    requires_rank_dispatch_map: bool = False
    routes_shared_expert: bool = False
    requires_deepep: bool = False
    dispatch_fixed_by_placement: bool = False
    """The per-token answer is a pure function of the committed placement, so a
    framework may materialise it once per placement and skip the per-forward
    routing call. Crossing the routing boundary is not free: on vLLM it measured
    426 us per layer per forward against a policy-free path. A policy setting
    this must publish its answer via ``PlacementSnapshot.default_physical_for_logical``."""
    consumes_global_logical_count: bool = False
    """Some policy reads the EP-wide per-logical token count, which costs one
    collective per layer per forward. Read once by a framework adapter at
    configuration time; the orchestrator's per-request check of the same fact is
    :class:`~evenmoe.core.l2.GlobalLogicalCountConsumer`."""
    supports_concurrent_microbatches: bool = True
    """False when a policy keeps per-layer state that concurrent micro-batches
    would clobber."""
    graph_stability: str = "stable"
    """``stable``, or ``realloc_on_placement_change`` when a placement update can
    replace policy tensors and invalidate a captured graph."""


@dataclass(frozen=True)
class RouterBuildContext:
    """What :meth:`MoELoadBalancer.from_algorithm` knows when it builds routers.

    ``options`` carries policy-specific injections (for example a kernel
    backend for tests) keyed by the caller; each policy reads only what it
    declared it needs.

    TODO: ``ep_size`` and ``experts_per_rank`` duplicate
    ``PlacementSnapshot.ep_size`` / ``num_local_physical_experts``; Waterfill
    and LPLB should read them from ``request.placement`` at route time like
    static/dynamic do. ``source_rank`` is the only fact with no runtime home
    yet -- give it one on ``PlacementSnapshot`` -- after which this context
    reduces to ``options`` and most policies build with a no-arg lambda.
    """

    ep_size: int
    source_rank: int
    experts_per_rank: int
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingPolicySpec:
    """Facts Pipeline and Orchestrator need to use one L2 policy.

    ``logical_count`` / ``rank_load`` name the data view the policy consumes:
    ``None`` (does not read it), ``"local"`` (this rank's own numbers), or
    ``"global"`` (EP-wide, which the orchestrator gathers with one collective).

    A policy whose declaration depends on its :class:`RoutingPolicyConfig` sets
    ``configure``; the pipeline calls it once per selection and stores the
    returned spec, so every later reader sees plain values.

    ``build_router`` constructs the policy's :class:`TokenRouter` for
    :meth:`MoELoadBalancer.from_algorithm`; a spec without it can only be used
    with an explicitly supplied router.
    """

    name: str
    phase: RoutingPhase
    logical_count: str | None = None
    rank_load: str | None = None
    requires_placement_state: bool = False
    requires_rank_dispatch_map: bool = False
    requires_deepep: bool = False
    dispatch_fixed_by_placement: bool = False
    supports_concurrent_microbatches: bool = True
    graph_stability: str = "stable"
    requires_redundancy: bool = False
    requires_placement_metadata: bool = False
    required_placement: str | None = None
    incompatible_with: frozenset[str] = frozenset()
    configure: Callable[[RoutingPolicyConfig], RoutingPolicySpec] | None = None
    build_router: Callable[[RouterBuildContext], TokenRouter] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "incompatible_with", frozenset(self.incompatible_with))
        if self.logical_count not in _DATA_VIEWS:
            raise ValueError("logical_count must be None, 'local', or 'global'.")
        if self.rank_load not in _DATA_VIEWS:
            raise ValueError("rank_load must be None, 'local', or 'global'.")
        if self.rank_load is not None and self.phase is not RoutingPhase.SHARED_EXPERT:
            raise ValueError("Only a shared-expert policy may consume rank load.")
        if self.graph_stability not in _GRAPH_STABILITIES:
            raise ValueError("Unknown graph_stability value.")

    def resolve(self, config: RoutingPolicyConfig) -> RoutingPolicySpec:
        """Return the declaration that applies to one concrete selection."""

        if self.configure is None:
            return self
        resolved = self.configure(config)
        if resolved.name != self.name:
            raise ValueError(f"{self.name}.configure returned a spec named {resolved.name!r}.")
        return replace(resolved, configure=None)

    def is_applicable(self, deployment: ExpertDeploymentConfig) -> str | None:
        """Why this policy cannot act on a deployment, or None if it can."""

        if self.requires_redundancy and deployment.num_redundant_experts == 0:
            return f"{self.name} requires redundant experts, but this deployment has none"
        if self.requires_placement_metadata and not deployment.carries_placement_metadata:
            return f"{self.name} requires plan metadata in the committed PlacementSnapshot"
        if self.phase is RoutingPhase.SHARED_EXPERT and not deployment.routes_shared_expert:
            return f"{self.name} requires the runtime to dispatch the shared expert"
        return None


InputTransform = Callable[
    [TensorLike, TensorLike | None], tuple[TensorLike, TensorLike | None]
]


@dataclass(frozen=True)
class InputTransformSpec:
    """An experimental pre-TopK stimulus and the physical router it implies."""

    name: str
    apply: InputTransform
    physical_routing: str


@dataclass(frozen=True)
class PolicyRegistry:
    """Name-to-declaration catalogue consumed while resolving an L2 selection.

    Routers are grouped by the phase they run in. ``physical_routers`` map a
    logical expert to one physical replica; ``shared_expert_routers`` choose the
    rank for the shared expert. A selection with no physical router gets
    ``default_physical_router`` (or the transform's own default) filled in.

    ``placement_planners`` are the L1 policies, as factories so every
    :class:`MoELoadBalancer` gets its own planner instances.

    ``aliases`` map alternative selection names onto registered routers (for
    example SGLang's ``lp`` spelling of ``lplb``). They are input-only: the
    resolved pipeline always carries the router's canonical ``spec.name``.
    """

    physical_routers: Mapping[str, RoutingPolicySpec] = field(default_factory=dict)
    shared_expert_routers: Mapping[str, RoutingPolicySpec] = field(default_factory=dict)
    default_physical_router: str | None = None
    input_transforms: Mapping[str, InputTransformSpec] = field(default_factory=dict)
    placement_planners: Mapping[str, Callable[[], PlacementPlanner]] = field(
        default_factory=dict
    )
    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = (
            (self.physical_routers, RoutingPhase.PHYSICAL_ROUTING),
            (self.shared_expert_routers, RoutingPhase.SHARED_EXPERT),
        )
        for routers, phase in groups:
            for name, spec in routers.items():
                if name != spec.name:
                    raise ValueError(
                        f"Routing policy {spec.name!r} is registered under the name {name!r}."
                    )
                if spec.phase is not phase:
                    raise ValueError(
                        f"Routing policy {name!r} runs in {spec.phase.name} and cannot be "
                        f"registered as a {phase.name} router."
                    )
                if (
                    spec.required_placement is not None
                    and spec.required_placement not in self.placement_planners
                ):
                    raise ValueError(
                        f"Routing policy {name!r} requires placement policy "
                        f"{spec.required_placement!r}, which is not registered."
                    )
        overlap = set(self.input_transforms) & (
            set(self.physical_routers) | set(self.shared_expert_routers)
        )
        if overlap:
            raise ValueError(f"Names {sorted(overlap)} are both routers and input transforms.")
        if (
            self.default_physical_router is not None
            and self.default_physical_router not in self.physical_routers
        ):
            raise ValueError(
                f"Default physical router {self.default_physical_router!r} is not registered."
            )
        for alias, target in self.aliases.items():
            if alias in self.names:
                raise ValueError(f"Alias {alias!r} is already a registered policy name.")
            if self.physical_routers.get(target) is None and (
                self.shared_expert_routers.get(target) is None
            ):
                raise ValueError(
                    f"Alias {alias!r} points at {target!r}, which is not a registered router."
                )

    def lookup(self, name: str) -> RoutingPolicySpec | None:
        name = self.aliases.get(name, name)
        spec = self.physical_routers.get(name)
        return spec if spec is not None else self.shared_expert_routers.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Every name a selection expression may use, sorted."""

        return tuple(
            sorted(
                set(self.physical_routers)
                | set(self.shared_expert_routers)
                | set(self.input_transforms)
            )
        )
