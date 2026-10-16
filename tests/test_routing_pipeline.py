# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from evenmoe import (
    ExpertDeploymentConfig,
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPhase,
    RoutingPipeline,
)
from evenmoe.core.policy import PolicyRegistry, RoutingPolicySpec


def _placement() -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=3,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([0, 1, 0, 1]),
        logical_to_physical_candidates=torch.tensor([[0, 2], [1, 3]]),
        logical_to_physical_count=torch.tensor([2, 2]),
        default_physical_for_logical=torch.tensor([0, 1]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "canonical", "replica", "post_topk"),
    [
        ("static", "static", "static", ("static",)),
        ("dynamic", "dynamic", "dynamic", ("dynamic",)),
        ("fake", "fake", "dynamic", ("dynamic",)),
        ("fake+static", "fake+static", "static", ("static",)),
        ("fake+lplb", "fake+lplb", "lplb", ("lplb",)),
        ("lp", "lplb", "lplb", ("lplb",)),
        ("lp+waterfill", "lplb+waterfill", "lplb", ("lplb", "waterfill")),
        ("waterfill", "static+waterfill", "static", ("static", "waterfill")),
        (
            "dynamic+waterfill",
            "dynamic+waterfill",
            "dynamic",
            ("dynamic", "waterfill"),
        ),
        ("lplb+waterfill", "lplb+waterfill", "lplb", ("lplb", "waterfill")),
    ],
)
def test_routing_pipeline_parses_and_canonicalizes(value, canonical, replica, post_topk):
    pipeline = RoutingPipeline.from_value(value)

    assert pipeline.name == canonical
    assert pipeline.policies[0] == replica
    assert tuple(policy.name for policy in pipeline.post_topk_policies) == post_topk


@pytest.mark.unit
def test_pipeline_capabilities_hide_algorithm_selection_from_framework_glue():
    pipeline = RoutingPipeline.from_value("lplb+waterfill")

    assert pipeline.capabilities.requires_post_topk_routing
    assert pipeline.capabilities.requires_placement_state
    assert pipeline.capabilities.routes_shared_expert
    assert pipeline.capabilities.requires_deepep
    assert not pipeline.capabilities.requires_rank_dispatch_map


@pytest.mark.unit
def test_default_replica_selection_is_owned_by_mlb():
    pipeline = RoutingPipeline.default_replica_routing()

    assert pipeline.name == "static"
    assert pipeline.capabilities.requires_rank_dispatch_map
    assert pipeline.capabilities.requires_post_topk_routing


@pytest.mark.unit
def test_fake_is_a_pre_topk_preset_with_dynamic_default():
    pipeline = RoutingPipeline.from_value("fake+waterfill")

    assert pipeline.name == "fake+waterfill"
    assert pipeline.input_transform.name == "fake"
    assert pipeline.policies[0] == "dynamic"
    assert tuple(policy.name for policy in pipeline.post_topk_policies) == (
        "dynamic",
        "waterfill",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "static+dynamic",
        "waterfill+waterfill",
        "unknown",
        "static+",
    ],
)
def test_routing_pipeline_rejects_invalid_composition(value):
    with pytest.raises(ValueError):
        RoutingPipeline.from_value(value)


class _PreparingRouter:
    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self):
        self.seen = None

    def prepare_layer(self, placement):
        self.seen = placement


@pytest.mark.unit
def test_placement_commit_notifies_active_policy_without_policy_name():
    router = _PreparingRouter()
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": router},
        routing_pipeline="lplb",
    )
    placement = _placement()

    assert mlb.on_placement_committed(placement) is None
    assert router.seen is placement


@pytest.mark.unit
class TestApplicabilityIsThePolicysCall:
    """Which deployments a policy can act on is a property of the policy.

    A framework that decides it by assumption -- "L2 means replica routing, so
    no redundant experts means nothing to do" -- switches off policies that had
    work to do. Waterfill assigns the shared expert to a rank and does not care
    how many replicas a routed expert has.
    """

    def test_lplb_declines_a_deployment_with_no_replicas(self):
        pipeline = RoutingPipeline.from_value("lplb")
        reason = pipeline.is_applicable(ExpertDeploymentConfig(num_redundant_experts=0))
        assert reason is not None and "redundant" in reason

    def test_lplb_applies_once_experts_are_replicated(self):
        pipeline = RoutingPipeline.from_value("lplb")
        assert pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=16)
        ) is None

    def test_waterfill_does_not_depend_on_routed_redundancy(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        assert pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=0, routes_shared_expert=True)
        ) is None

    def test_waterfill_declines_a_runtime_that_cannot_dispatch_it(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        reason = pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=32, routes_shared_expert=False)
        )
        assert reason is not None and "shared expert" in reason

    def test_the_reason_is_returned_not_a_bare_bool(self):
        """A framework that disables a policy has to be able to say why."""
        reason = RoutingPipeline.from_value("lplb").is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=0)
        )
        assert isinstance(reason, str) and len(reason) > 20


@pytest.mark.unit
class TestCapabilitiesDeclareRuntimeRequirements:
    def test_only_the_lp_consumes_the_global_load(self):
        assert RoutingPipeline.from_value("lplb").capabilities \
            .consumes_global_logical_count is True
        for expr in ("static", "dynamic"):
            assert RoutingPipeline.from_value(expr).capabilities \
                .consumes_global_logical_count is False

    def test_concurrency_and_graph_stability_are_per_policy(self):
        lp = RoutingPipeline.from_value("lplb").capabilities
        assert lp.supports_concurrent_microbatches is False
        assert lp.graph_stability == "realloc_on_placement_change"

        st = RoutingPipeline.from_value("static").capabilities
        assert st.supports_concurrent_microbatches is True
        assert st.graph_stability == "stable"


@pytest.mark.unit
def test_ultraep_is_rejected_when_the_framework_drops_placement_metadata():
    """Caught while reading configuration, not silently at the first token.

    ultraep's L2 half reads the quota its own L1 half publishes on
    PlacementPlan.metadata. A framework that does not carry that metadata into
    the committed snapshot leaves the router routing everything to replica 0 --
    plausible-looking but unbalanced, and strictly worse than `static` while
    still paying ultraep's cost.
    """
    reason = RoutingPipeline.from_value("ultraep").is_applicable(
        ExpertDeploymentConfig(num_redundant_experts=4, carries_placement_metadata=False)
    )
    assert reason is not None
    assert "PlacementSnapshot" in reason


@pytest.mark.unit
def test_ultraep_is_applicable_once_the_framework_carries_the_metadata():
    assert (
        RoutingPipeline.from_value("ultraep").is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=4, carries_placement_metadata=True)
        )
        is None
    )


@pytest.mark.unit
def test_other_policies_do_not_need_placement_metadata():
    """The constraint is ultraep's, not a new requirement on every policy."""
    config = ExpertDeploymentConfig(num_redundant_experts=4, carries_placement_metadata=False)
    for algorithm in ("static", "dynamic", "lplb"):
        assert RoutingPipeline.from_value(algorithm).is_applicable(config) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("expression", "ordered"),
    [
        ("waterfill+lplb", ("lplb", "waterfill")),
        ("waterfill+ultraep", ("ultraep", "waterfill")),
    ],
)
def test_pipeline_orders_selected_policies_by_phase(expression, ordered):
    assert RoutingPipeline.from_value(expression).policies == ordered


@pytest.mark.unit
def test_lplb_local_count_mode_is_declared_by_the_policy(monkeypatch):
    monkeypatch.setenv("LPLB_USE_LOCAL_COUNT", "1")

    pipeline = RoutingPipeline.from_value("lplb")

    assert not pipeline.capabilities.consumes_global_logical_count
    assert pipeline.steps[0].spec.logical_count == "local"


@pytest.mark.unit
def test_waterfill_global_load_mode_is_declared_by_the_policy(monkeypatch):
    monkeypatch.setenv("WATERFILL_USE_GLOBAL_LOAD", "1")

    pipeline = RoutingPipeline.from_value("waterfill")
    waterfill = pipeline.steps[1]

    assert waterfill.spec.rank_load == "global"


@pytest.mark.unit
def test_global_waterfill_rejects_lplb_until_collectives_are_uniform(monkeypatch):
    monkeypatch.setenv("WATERFILL_USE_GLOBAL_LOAD", "1")

    with pytest.raises(ValueError, match="waterfill cannot be combined with.*lplb"):
        RoutingPipeline.from_value("lplb+waterfill")


@pytest.mark.unit
def test_composition_disables_fixed_dispatch_capability():
    assert RoutingPipeline.from_value("static").capabilities.dispatch_fixed_by_placement
    # A pre-TopK stimulus does not make static's dispatch batch-dependent.
    assert RoutingPipeline.from_value("fake+static").capabilities.dispatch_fixed_by_placement
    assert not RoutingPipeline.from_value("fake").capabilities.dispatch_fixed_by_placement
    assert not RoutingPipeline.from_value(
        "static+waterfill"
    ).capabilities.dispatch_fixed_by_placement


@pytest.mark.unit
def test_pipeline_reads_phase_and_capabilities_from_registered_policy():
    external = RoutingPolicySpec(
        name="external",
        phase=RoutingPhase.SHARED_EXPERT,
        rank_load="local",
        requires_deepep=True,
    )
    registry = PolicyRegistry(
        physical_routers={"static": RoutingPipeline.from_value("static").steps[0].spec},
        shared_expert_routers={"external": external},
        default_physical_router="static",
    )

    pipeline = RoutingPipeline.from_value("external", registry=registry)

    assert pipeline.policies == ("static", "external")
    assert pipeline.capabilities.routes_shared_expert
    assert pipeline.capabilities.requires_deepep


@pytest.mark.unit
def test_from_algorithm_builds_routers_from_policy_declarations():
    """Core never names a policy: a spec registered from outside is built by its
    own ``build_router`` and receives the deployment context and options."""
    seen = {}

    class _Router:
        phase = RoutingPhase.PHYSICAL_ROUTING

        def route_tokens(self, request, policy):
            raise NotImplementedError

    def _build(context):
        seen["context"] = context
        return _Router()

    registry = PolicyRegistry(
        physical_routers={
            "external": RoutingPolicySpec(
                "external", RoutingPhase.PHYSICAL_ROUTING, build_router=_build
            )
        },
        default_physical_router="external",
    )

    mlb = MoELoadBalancer.from_algorithm(
        "external",
        ep_size=4,
        source_rank=1,
        experts_per_rank=8,
        registry=registry,
        backend="stub",
    )

    assert isinstance(mlb._routing_policies["external"], _Router)
    assert (seen["context"].ep_size, seen["context"].source_rank) == (4, 1)
    assert seen["context"].experts_per_rank == 8
    assert seen["context"].options == {"backend": "stub"}


@pytest.mark.unit
def test_from_algorithm_rejects_policies_without_a_builder():
    registry = PolicyRegistry(
        physical_routers={"opaque": RoutingPolicySpec("opaque", RoutingPhase.PHYSICAL_ROUTING)},
        default_physical_router="opaque",
    )

    with pytest.raises(ValueError, match="does not declare build_router"):
        MoELoadBalancer.from_algorithm(
            "opaque", ep_size=1, source_rank=0, experts_per_rank=1, registry=registry
        )


@pytest.mark.unit
def test_registry_rejects_a_router_registered_under_the_wrong_phase():
    shared = RoutingPolicySpec("shared", RoutingPhase.SHARED_EXPERT)

    with pytest.raises(ValueError, match="SHARED_EXPERT.*PHYSICAL_ROUTING"):
        PolicyRegistry(physical_routers={"shared": shared})


@pytest.mark.unit
def test_registry_rejects_a_default_that_is_not_a_physical_router():
    with pytest.raises(ValueError, match="not registered"):
        PolicyRegistry(default_physical_router="static")


@pytest.mark.unit
def test_registry_alias_must_point_at_a_registered_router():
    physical = RoutingPolicySpec("physical", RoutingPhase.PHYSICAL_ROUTING)

    with pytest.raises(ValueError, match="not a registered router"):
        PolicyRegistry(physical_routers={"physical": physical}, aliases={"p": "missing"})
    with pytest.raises(ValueError, match="already a registered policy name"):
        PolicyRegistry(physical_routers={"physical": physical}, aliases={"physical": "physical"})


@pytest.mark.unit
def test_registry_alias_resolves_to_the_canonical_name():
    physical = RoutingPolicySpec("physical", RoutingPhase.PHYSICAL_ROUTING)
    registry = PolicyRegistry(physical_routers={"physical": physical}, aliases={"p": "physical"})

    assert registry.lookup("p") is physical
    assert RoutingPipeline.from_value("p", registry=registry).name == "physical"


@pytest.mark.unit
def test_registry_rejects_a_required_placement_that_is_not_registered():
    coupled = RoutingPolicySpec(
        "coupled", RoutingPhase.PHYSICAL_ROUTING, required_placement="missing_l1"
    )

    with pytest.raises(ValueError, match="requires placement policy 'missing_l1'"):
        PolicyRegistry(physical_routers={"coupled": coupled})


@pytest.mark.unit
def test_default_placement_planners_come_from_the_registry():
    from evenmoe.policies.registry import builtin_registry

    mlb = MoELoadBalancer()

    assert set(mlb._placement_planners) == set(builtin_registry().placement_planners)
    assert "ultraep" in mlb._placement_planners
    assert "deepseek_hierarchical" in mlb._placement_planners
