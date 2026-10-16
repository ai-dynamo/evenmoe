# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang interoperability for the framework-neutral load balancer.

SGLang owns lifecycle, collective transport, threshold decisions, recording,
weight movement and committed placement metadata. EvenMoE decides when L2 needs a
collective and invokes the injected SGLang transport adapter. This package
contains data translations and stateless context integration functions.
"""

from evenmoe.adapters.sglang.placement import (
    SGLangPlacementMaps,
    commit_placement,
    to_placement_request,
    to_placement_snapshot,
    to_sglang_maps,
)
from evenmoe.adapters.sglang.routing import (
    SGLangRoutingOutput,
    count_logical_experts,
    to_load_balancer_kwargs,
    to_routing_request,
    to_sglang_routing_output,
)

__all__ = [
    "SGLangPlacementMaps",
    "SGLangRoutingOutput",
    "commit_placement",
    "count_logical_experts",
    "to_load_balancer_kwargs",
    "to_placement_request",
    "to_placement_snapshot",
    "to_routing_request",
    "to_sglang_maps",
    "to_sglang_routing_output",
]
