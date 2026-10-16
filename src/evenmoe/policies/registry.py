# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in policy registration.

Adding a policy means declaring it next to its implementation and listing it
here; nothing in ``core/`` changes.
"""

from __future__ import annotations

from functools import lru_cache, partial

from evenmoe.core.policy import PolicyRegistry


@lru_cache(maxsize=1)
def builtin_registry() -> PolicyRegistry:
    from evenmoe.policies.l1.eplb import (
        SUPPORTED_PLACEMENT_POLICIES,
        EPLBPlacementPlanner,
    )
    from evenmoe.policies.l1.ultraep import UltraEPPlacementPolicy
    from evenmoe.policies.l2.lplb import ROUTING_POLICY as lplb
    from evenmoe.policies.l2.replica import INPUT_TRANSFORMS as replica_transforms
    from evenmoe.policies.l2.replica import ROUTING_POLICIES as replica
    from evenmoe.policies.l2.ultraep import ROUTING_POLICY as ultraep_l2
    from evenmoe.policies.l2.waterfill import ROUTING_POLICY as waterfill

    return PolicyRegistry(
        physical_routers={spec.name: spec for spec in (*replica, lplb, ultraep_l2)},
        shared_expert_routers={waterfill.name: waterfill},
        default_physical_router="static",
        input_transforms={spec.name: spec for spec in replica_transforms},
        placement_planners={
            **{name: partial(EPLBPlacementPlanner, name) for name in SUPPORTED_PLACEMENT_POLICIES},
            UltraEPPlacementPolicy.name: UltraEPPlacementPolicy,
        },
        # SGLang spells its own LPLB dispatch algorithm ``lp``; accept the same
        # name so a user can switch to EvenMoE without renaming it.
        aliases={"lp": lplb.name},
    )
