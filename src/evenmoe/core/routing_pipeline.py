# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse, order, and validate one L2 policy selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from evenmoe.core.policy import (
    ExpertDeploymentConfig,
    InputTransformSpec,
    PolicyRegistry,
    RoutingCapabilities,
    RoutingPolicySpec,
)
from evenmoe.core.types import RoutingPhase, RoutingPolicyConfig

_PHASE_ORDER = {RoutingPhase.PHYSICAL_ROUTING: 0, RoutingPhase.SHARED_EXPERT: 1}
_DISABLED = frozenset({"", "none", "disabled"})


@dataclass(frozen=True)
class _RoutingStep:
    """One selected policy: its resolved declaration and its configuration."""

    spec: RoutingPolicySpec
    config: RoutingPolicyConfig
    implicit: bool = False


@dataclass(frozen=True)
class RoutingPipeline:
    """The selected L2 policies in execution order."""

    steps: tuple[_RoutingStep, ...] = ()
    input_transform: InputTransformSpec | None = None
    capabilities: RoutingCapabilities = field(init=False)

    def __post_init__(self) -> None:
        steps = tuple(sorted(self.steps, key=lambda step: _PHASE_ORDER[step.spec.phase]))
        phases = [step.spec.phase for step in steps]
        if len(phases) != len(set(phases)):
            raise ValueError(
                "At most one physical-routing policy and one shared-expert policy are supported."
            )

        names = {step.spec.name for step in steps}
        for step in steps:
            conflicts = step.spec.incompatible_with & names
            if conflicts:
                raise ValueError(
                    f"{step.spec.name} cannot be combined with {sorted(conflicts)}."
                )
        required_l1 = {step.spec.required_placement for step in steps} - {None}
        if len(required_l1) > 1:
            raise ValueError("Routing policies require incompatible L1 placement policies.")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "capabilities", _merge_capabilities(steps))

    @property
    def name(self) -> str:
        parts = [self.input_transform.name] if self.input_transform else []
        parts += [
            step.spec.name
            for step in self.steps
            if self.input_transform is None or not step.implicit
        ]
        return "+".join(parts) or "disabled"

    @property
    def policies(self) -> tuple[str, ...]:
        return tuple(step.spec.name for step in self.steps)

    @property
    def post_topk_policies(self) -> tuple[RoutingPolicyConfig, ...]:
        """The selected policy configs in execution order.

        TODO: Orchestrator.route_tokens iterates these configs and then looks
        the resolved spec back up through ``spec_for``. Make it iterate
        ``steps`` directly (each step already pairs spec and config) and build
        a request-level override with ``from_value(request.policies)``; then
        this property, ``spec_for``, and ``Orchestrator._policy_spec`` go away.
        The one open question is routers registered under a name with no
        declaration (see the ``"custom"`` router in tests/test_orchestrator.py):
        ``from_value`` rejects unknown names, so that path must either be
        dropped or given a synthesised spec.
        """

        return tuple(step.config for step in self.steps)

    def spec_for(self, name: str) -> RoutingPolicySpec | None:
        """The resolved declaration of one selected policy, if it is selected.

        Transitional; see the TODO on ``post_topk_policies``.
        """

        return next((step.spec for step in self.steps if step.spec.name == name), None)

    @property
    def placement_policy(self) -> str | None:
        return next(
            (
                step.spec.required_placement
                for step in self.steps
                if step.spec.required_placement is not None
            ),
            None,
        )

    def is_applicable(self, deployment: ExpertDeploymentConfig) -> str | None:
        for step in self.steps:
            reason = step.spec.is_applicable(deployment)
            if reason is not None:
                return reason
        return None

    @classmethod
    def default_replica_routing(
        cls, *, registry: PolicyRegistry | None = None
    ) -> RoutingPipeline:
        if registry is None:
            from evenmoe.policies.registry import builtin_registry

            registry = builtin_registry()
        if registry.default_physical_router is None:
            raise ValueError("The policy registry has no default physical router.")
        return cls.from_value(registry.default_physical_router, registry=registry)

    @classmethod
    def from_value(
        cls,
        value: RoutingPipeline | str | Sequence[RoutingPolicyConfig | str] | None,
        *,
        registry: PolicyRegistry | None = None,
    ) -> RoutingPipeline:
        if isinstance(value, cls):
            return value
        if registry is None:
            from evenmoe.policies.registry import builtin_registry

            registry = builtin_registry()
        if value is None or (isinstance(value, str) and value.strip().lower() in _DISABLED):
            return cls()

        if isinstance(value, str):
            raw = tuple(part.strip().lower() for part in value.split("+"))
            if any(not part for part in raw):
                raise ValueError(f"Invalid routing algorithm expression {value!r}.")
            configs = tuple(RoutingPolicyConfig(name=part) for part in raw)
        elif isinstance(value, (tuple, list)):
            configs = tuple(RoutingPolicyConfig.from_value(item) for item in value)
        else:
            raise TypeError(
                "Routing configuration must be a string, a sequence of policy configs, "
                f"or a RoutingPipeline; got {type(value).__name__}."
            )

        transform: InputTransformSpec | None = None
        steps: list[_RoutingStep] = []
        selected: set[str] = set()
        for config in configs:
            name = config.name.strip().lower()
            if name in registry.input_transforms:
                if transform is not None:
                    raise ValueError("At most one input transform is supported.")
                if config.enabled_stages is not None or config.metadata:
                    raise ValueError("Input transforms do not accept stage gates or metadata.")
                transform = registry.input_transforms[name]
                continue

            spec = registry.lookup(name)
            if spec is None:
                raise ValueError(
                    f"Unknown routing algorithm {name!r}; supported algorithms are "
                    f"{list(registry.names)}."
                )
            if spec.name in selected:
                raise ValueError(f"Routing algorithm {spec.name!r} is selected twice.")
            selected.add(spec.name)
            if config.name != spec.name:
                config = replace(config, name=spec.name)
            steps.append(_RoutingStep(spec.resolve(config), config))

        if steps or transform:
            has_physical = any(
                step.spec.phase is RoutingPhase.PHYSICAL_ROUTING for step in steps
            )
            if not has_physical:
                default_name = (
                    transform.physical_routing
                    if transform is not None
                    else registry.default_physical_router
                )
                if default_name is None:
                    raise ValueError("The policy registry has no default physical router.")
                default = registry.physical_routers.get(default_name)
                if default is None:
                    raise ValueError(
                        f"Default physical router {default_name!r} is not registered."
                    )
                config = RoutingPolicyConfig(name=default.name)
                steps.append(_RoutingStep(default.resolve(config), config, implicit=True))

        return cls(steps=tuple(steps), input_transform=transform)


def _merge_capabilities(steps: Sequence[_RoutingStep]) -> RoutingCapabilities:
    specs = [step.spec for step in steps]
    return RoutingCapabilities(
        requires_post_topk_routing=bool(steps),
        requires_placement_state=any(spec.requires_placement_state for spec in specs),
        requires_rank_dispatch_map=any(spec.requires_rank_dispatch_map for spec in specs),
        routes_shared_expert=any(
            step.spec.phase is RoutingPhase.SHARED_EXPERT for step in steps
        ),
        requires_deepep=any(spec.requires_deepep for spec in specs),
        # An input transform only perturbs the pre-TopK stimulus; it does not
        # make the logical->physical answer depend on the batch, so ``fake+static``
        # stays a deterministic, placement-fixed dispatch.
        dispatch_fixed_by_placement=bool(steps)
        and all(spec.dispatch_fixed_by_placement for spec in specs),
        consumes_global_logical_count=any(spec.logical_count == "global" for spec in specs),
        supports_concurrent_microbatches=all(
            spec.supports_concurrent_microbatches for spec in specs
        ),
        graph_stability=(
            "realloc_on_placement_change"
            if any(spec.graph_stability == "realloc_on_placement_change" for spec in specs)
            else "stable"
        ),
    )
