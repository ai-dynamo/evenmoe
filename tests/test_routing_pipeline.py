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
from evenmoe.core import policy as policy_module
from evenmoe.core.policy import PolicyRegistry, RoutingPolicySpec
from evenmoe.core.types import RoutingPolicyConfig


@pytest.fixture(autouse=True)
def _waterfill_defaults(monkeypatch):
    """Run every test under Waterfill's defaults, with fresh one-time warnings."""
    for name in ("WATERFILL_LOAD_VIEW", "WATERFILL_CANDIDATES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(policy_module, "_reported_fallbacks", set())


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
    assert pipeline.capabilities.supported_a2a_backends == ("deepep", "megamoe")
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


def _waterfill_spec(pipeline: RoutingPipeline) -> RoutingPolicySpec:
    spec = pipeline.spec_for("waterfill")
    assert spec is not None
    return spec


def _options(config=None):
    from evenmoe.policies.l2.waterfill import waterfill_option

    return waterfill_option(config, "load_view"), waterfill_option(config, "candidates")


@pytest.mark.unit
@pytest.mark.parametrize(
    "algorithm", ["waterfill", "dynamic+waterfill", "lplb+waterfill", "ultraep+waterfill"]
)
def test_waterfill_defaults_to_local_view_with_routed_candidates(algorithm, caplog):
    with caplog.at_level("WARNING", logger="evenmoe"):
        waterfill = _waterfill_spec(RoutingPipeline.from_value(algorithm))

    assert waterfill.rank_load == "local"
    assert not waterfill.reads_global_view
    assert _options() == ("local", "routed")
    assert not caplog.records


@pytest.mark.unit
@pytest.mark.parametrize("load_view", ["local", "global"])
@pytest.mark.parametrize("candidates", ["routed", "all"])
def test_waterfill_options_are_independent(monkeypatch, load_view, candidates):
    config = RoutingPolicyConfig(
        name="waterfill", metadata={"load_view": load_view, "candidates": candidates}
    )

    waterfill = _waterfill_spec(RoutingPipeline.from_value([config]))

    assert waterfill.rank_load == load_view
    assert _options(config) == (load_view, candidates)


@pytest.mark.unit
def test_waterfill_config_options_override_the_environment(monkeypatch):
    monkeypatch.setenv("WATERFILL_LOAD_VIEW", "global")
    monkeypatch.setenv("WATERFILL_CANDIDATES", "all")
    config = RoutingPolicyConfig(
        name="waterfill", metadata={"load_view": "local", "candidates": "routed"}
    )

    assert _options(config) == ("local", "routed")
    pipeline = RoutingPipeline.from_value([config])
    assert _waterfill_spec(pipeline).rank_load == "local"
    assert _options(RoutingPolicyConfig(name="waterfill")) == ("global", "all")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "value", "match"),
    [
        ("WATERFILL_LOAD_VIEW", "auto", "WATERFILL_LOAD_VIEW='auto'"),
        ("WATERFILL_CANDIDATES", "some", "WATERFILL_CANDIDATES='some'"),
    ],
)
def test_waterfill_rejects_unknown_option_values(monkeypatch, env, value, match):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=match):
        _options()


@pytest.mark.unit
def test_waterfill_rejects_an_unknown_config_option_value():
    with pytest.raises(ValueError, match="load_view"):
        RoutingPipeline.from_value(
            [RoutingPolicyConfig(name="waterfill", metadata={"load_view": "sometimes"})]
        )


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["lplb+waterfill", "lp+waterfill", "fake+lplb+waterfill"])
@pytest.mark.parametrize("candidates", ["routed", "all"])
def test_global_waterfill_falls_back_to_local_after_lplb_with_one_warning(
    monkeypatch, caplog, algorithm, candidates
):
    monkeypatch.setenv("WATERFILL_LOAD_VIEW", "global")
    monkeypatch.setenv("WATERFILL_CANDIDATES", candidates)

    with caplog.at_level("WARNING", logger="evenmoe"):
        for _ in range(3):
            waterfill = _waterfill_spec(RoutingPipeline.from_value(algorithm))
            assert waterfill.rank_load == "local"

    # The fallback changes the load view only.
    assert _options()[1] == candidates
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert "waterfill falls back (rank_load 'global' -> 'local')" in warnings[0]
    assert "lplb" in warnings[0]


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["waterfill", "dynamic+waterfill", "ultraep+waterfill"])
def test_global_waterfill_keeps_the_global_view_without_lplb(monkeypatch, caplog, algorithm):
    monkeypatch.setenv("WATERFILL_LOAD_VIEW", "global")

    with caplog.at_level("WARNING", logger="evenmoe"):
        waterfill = _waterfill_spec(RoutingPipeline.from_value(algorithm))

    assert waterfill.rank_load == "global"
    assert waterfill.fallback is not None
    assert waterfill.fallback.rank_load == "local"
    assert not caplog.records


@pytest.mark.unit
def test_conflict_fallback_is_generic_and_errors_without_one():
    """A conflicting policy is swapped for its declared fallback, never by name."""
    local = RoutingPolicySpec("shared", RoutingPhase.SHARED_EXPERT, rank_load="local")
    eager = RoutingPolicySpec(
        "shared",
        RoutingPhase.SHARED_EXPERT,
        rank_load="global",
        incompatible_with=frozenset({"static"}),
        fallback=local,
    )
    static = RoutingPipeline.from_value("static").steps[0].spec

    def registry(shared):
        return PolicyRegistry(
            physical_routers={"static": static},
            shared_expert_routers={"shared": shared},
            default_physical_router="static",
        )

    pipeline = RoutingPipeline.from_value("shared", registry=registry(eager))
    assert pipeline.spec_for("shared") is local

    strict = RoutingPolicySpec(
        "shared",
        RoutingPhase.SHARED_EXPERT,
        rank_load="global",
        incompatible_with=frozenset({"static"}),
    )
    with pytest.raises(ValueError, match="shared cannot be combined with"):
        RoutingPipeline.from_value("shared", registry=registry(strict))
    with pytest.raises(ValueError, match="same policy"):
        RoutingPolicySpec(
            "other",
            RoutingPhase.SHARED_EXPERT,
            fallback=local,
        )


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
        supported_a2a_backends=("deepep", "megamoe"),
    )
    registry = PolicyRegistry(
        physical_routers={"static": RoutingPipeline.from_value("static").steps[0].spec},
        shared_expert_routers={"external": external},
        default_physical_router="static",
    )

    pipeline = RoutingPipeline.from_value("external", registry=registry)

    assert pipeline.policies == ("static", "external")
    assert pipeline.capabilities.routes_shared_expert
    assert pipeline.capabilities.supported_a2a_backends == ("deepep", "megamoe")


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


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["disabled", "static", "dynamic", "fake", "lplb"])
def test_replica_policies_do_not_restrict_a2a_backend(algorithm):
    assert RoutingPipeline.from_value(algorithm).capabilities.supported_a2a_backends is None


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["waterfill", "lplb+waterfill", "dynamic+waterfill"])
def test_waterfill_supports_deepep_and_megamoe(algorithm):
    assert RoutingPipeline.from_value(algorithm).capabilities.supported_a2a_backends == (
        "deepep",
        "megamoe",
    )


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["ultraep", "metro", "ultraep+waterfill", "metro+waterfill"])
def test_deepep_only_policy_keeps_its_restriction(algorithm):
    assert RoutingPipeline.from_value(algorithm).capabilities.supported_a2a_backends == ("deepep",)


@pytest.mark.unit
@pytest.mark.parametrize(
    "physical, shared, expected",
    [
        (None, None, None),
        (("megamoe", "deepep"), None, ("megamoe", "deepep")),
        (None, ("custom",), ("custom",)),
        (("megamoe", "deepep"), ("deepep", "megamoe"), ("megamoe", "deepep")),
        (("megamoe", "deepep"), ("deepep",), ("deepep",)),
        (("megamoe",), ("deepep",), ()),
    ],
)
def test_composition_intersects_backends_and_preserves_preference(physical, shared, expected):
    registry = PolicyRegistry(
        physical_routers={
            "physical": RoutingPolicySpec(
                "physical", RoutingPhase.PHYSICAL_ROUTING, supported_a2a_backends=physical
            )
        },
        shared_expert_routers={
            "shared": RoutingPolicySpec(
                "shared", RoutingPhase.SHARED_EXPERT, supported_a2a_backends=shared
            )
        },
        default_physical_router="physical",
    )
    if expected == ():
        with pytest.raises(ValueError, match="no common supported A2A backend"):
            RoutingPipeline.from_value("shared", registry=registry)
    else:
        assert (
            RoutingPipeline.from_value(
                "shared", registry=registry
            ).capabilities.supported_a2a_backends
            == expected
        )


@pytest.mark.unit
@pytest.mark.parametrize("backends", [(), ("",), (1,), "deepep"])
def test_policy_rejects_invalid_backend_restriction(backends):
    with pytest.raises(ValueError, match="supported_a2a_backends"):
        RoutingPolicySpec(
            "external", RoutingPhase.PHYSICAL_ROUTING, supported_a2a_backends=backends
        )
