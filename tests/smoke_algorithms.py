# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke run of every L2 algorithm expression through MoELoadBalancer.

Prints one JSON document describing, per expression: the resolved pipeline,
its capabilities, and the routing decision on a fixed request. Run it on two
trees and diff the output to see whether a refactor changed behaviour.

    python tests/smoke_algorithms.py > /tmp/smoke_<tree>.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evenmoe import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)


@dataclass
class _Kernels:
    """Shape-valid LPLB kernel double (same as tests/test_lplb_policy.py)."""

    calls: list = field(default_factory=list)

    def warmup(self, nc, nv, num_iters, device):
        pass

    def warmup_shape(self, **kwargs):
        pass

    def prep_lp_inputs(self, A_full, b, t1, global_counts, *rest):
        self.calls.append("prep")
        A_full[:, -1].fill_(0.0)
        b.fill_(0.0)
        t1.fill_(1.0)

    def solve_ipm(self, A, b, c, num_iters=5, result=None):
        self.calls.append("solve")
        result.fill_(0.5)
        return result

    def extract_log2phy_prob(self, log2phy_prob, x, t1, *rest):
        self.calls.append("extract")
        log2phy_prob.fill_(0.5)

    def dispatch_probability(self, topk_ids, log2phy_prob, log2phy_map, **kwargs):
        # Deterministic stand-in for the CUDA sampler: always take replica 0.
        self.calls.append("dispatch")
        safe = topk_ids.to(torch.int64).clamp(min=0)
        physical = log2phy_map.to(torch.int64)[safe, 0]
        return torch.where(topk_ids >= 0, physical, topk_ids.to(torch.int64)).to(topk_ids.dtype)


class _Collectives:
    rank = 0
    world_size = 2

    def __init__(self):
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone().tolist())
        return payload * 2


def _placement() -> PlacementSnapshot:
    # 4 logical, 6 physical, EP=2, experts 0 and 2 replicated.
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 1, 2, 0, 2, 3]),
        logical_to_physical_candidates=torch.tensor(
            [[0, 3], [1, -1], [2, 4], [5, -1]], dtype=torch.int64
        ),
        logical_to_physical_count=torch.tensor([2, 1, 2, 1], dtype=torch.int64),
        default_physical_for_logical=torch.tensor([0, 1, 2, 5]),
        metadata={
            "rank_quota_prefix": torch.tensor([[2, 4], [2, 4], [2, 4], [2, 4]], dtype=torch.int32)
        },
    )


def _request(policies=()) -> RoutingRequest:
    ids = torch.tensor([[0, 2], [0, 1], [2, 3], [0, 2]], dtype=torch.int64)
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=ids,
        topk_weights=torch.ones(ids.shape, dtype=torch.float32),
        placement=_placement(),
        stage=RoutingStage.PREFILL,
        token_count=4,
        policies=tuple(RoutingPolicyConfig(name=p) for p in policies),
    )


def _tolist(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _run(expression: str, env: dict[str, str]) -> dict:
    for key in ("LPLB_USE_LOCAL_COUNT", "WATERFILL_USE_GLOBAL_LOAD"):
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.manual_seed(0)
    collectives = _Collectives()
    try:
        mlb = MoELoadBalancer.from_algorithm(
            expression,
            ep_size=2,
            source_rank=0,
            experts_per_rank=3,
            collectives=collectives,
            lplb_kernels=_Kernels(),
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    pipeline = mlb.routing_pipeline
    out = {
        "pipeline_name": pipeline.name,
        "policies": list(pipeline.policies),
        "placement_policy": mlb.placement_policy,
        "capabilities": asdict(mlb.routing_capabilities),
        "routers": sorted(type(r).__name__ for r in mlb._routing_policies.values()),
    }
    logits = torch.zeros((4, 4))
    transformed, _ = mlb.transform_topk_inputs(logits.clone(), None)
    out["transform_changes_logits"] = not torch.equal(transformed, logits)

    if not pipeline.policies:
        return out
    try:
        mlb.on_placement_committed(_placement())
        decision = mlb.route_tokens(_request())
        out["decision"] = {
            "physical_ids": _tolist(decision.routed_physical_topk_ids),
            "shared_expert_rank": _tolist(decision.shared_expert_rank),
            "executed": list(decision.metadata.get("executed_policies", ())),
            "metadata_keys": sorted(decision.metadata),
        }
        out["collective_payloads"] = collectives.payloads
    except Exception as exc:  # noqa: BLE001
        out["route_error"] = f"{type(exc).__name__}: {exc}"
    return out


CASES = [
    ("disabled", "none", {}),
    ("static", "static", {}),
    ("dynamic", "dynamic", {}),
    ("dynamic_random", "dynamic_random", {}),
    ("lplb", "lplb", {}),
    ("lplb_local_count", "lplb", {"LPLB_USE_LOCAL_COUNT": "1"}),
    ("waterfill", "waterfill", {}),
    ("static+waterfill", "static+waterfill", {}),
    ("dynamic+waterfill", "dynamic+waterfill", {}),
    ("lplb+waterfill", "lplb+waterfill", {}),
    ("metro", "metro", {}),
    ("fake", "fake", {}),
    ("fake+static", "fake+static", {}),
    ("fake+lplb", "fake+lplb", {}),
    ("fake+waterfill", "fake+waterfill", {}),
    ("ultraep", "ultraep", {}),
    ("ultraep+waterfill", "ultraep+waterfill", {}),
    ("reordered waterfill+lplb", "waterfill+lplb", {}),
    ("invalid static+dynamic", "static+dynamic", {}),
    ("invalid unknown", "unknown", {}),
]

# Global Waterfill was spelled ``waterfill_dynamic`` before the refactor and is
# an environment switch after it; run whichever this tree understands.
try:
    MoELoadBalancer.from_algorithm(
        "dynamic+waterfill_dynamic", ep_size=2, source_rank=0, experts_per_rank=3
    )
    CASES.append(("global waterfill", "dynamic+waterfill_dynamic", {}))
    CASES.append(("global waterfill after lplb (must fail)", "lplb+waterfill_dynamic", {}))
except ValueError:
    CASES.append(("global waterfill", "dynamic+waterfill", {"WATERFILL_USE_GLOBAL_LOAD": "1"}))
    CASES.append(
        (
            "global waterfill after lplb (must fail)",
            "lplb+waterfill",
            {"WATERFILL_USE_GLOBAL_LOAD": "1"},
        )
    )

results = {label: _run(expr, env) for label, expr, env in CASES}
json.dump(results, sys.stdout, indent=1, sort_keys=True)
print()
