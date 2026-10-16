# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline and experimental physical-replica routing policies."""

from __future__ import annotations

import torch
from torch import Tensor

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import InputTransformSpec, RoutingPolicySpec
from evenmoe.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.kernels.ops.replica_draw import replica_index


def transform_fake_routing_inputs(
    router_logits: Tensor,
    correction_bias: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Create the uniform random routing stimulus used by SGLang fake mode."""

    if not isinstance(router_logits, Tensor):
        raise TypeError("Fake routing requires torch.Tensor router_logits.")
    router_logits.uniform_(5, 10)
    if correction_bias is not None:
        if not isinstance(correction_bias, Tensor):
            raise TypeError("Fake routing correction_bias must be a torch.Tensor or None.")
        correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def route_static_replicas(
    logical_topk_ids: Tensor,
    default_physical_for_logical: Tensor,
) -> Tensor:
    """Map every logical expert to the framework-selected default replica."""

    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Replica routing requires torch.Tensor logical_topk_ids.")
    if not isinstance(default_physical_for_logical, Tensor):
        raise TypeError("Static replica routing requires a torch.Tensor default map.")
    safe_ids, valid = _safe_logical_ids(
        logical_topk_ids,
        default_physical_for_logical.shape[0],
    )
    physical_ids = default_physical_for_logical[safe_ids]
    return torch.where(valid, physical_ids, logical_topk_ids)


def route_dynamic_replicas(
    logical_topk_ids: Tensor,
    logical_to_physical_candidates: Tensor,
    logical_to_physical_count: Tensor,
) -> Tensor:
    """Randomly choose one committed physical replica for every logical ID."""

    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Replica routing requires torch.Tensor logical_topk_ids.")
    if not isinstance(logical_to_physical_candidates, Tensor) or not isinstance(
        logical_to_physical_count, Tensor
    ):
        raise TypeError("Dynamic replica routing requires torch.Tensor placement maps.")

    safe_ids, valid = _safe_logical_ids(
        logical_topk_ids,
        logical_to_physical_count.shape[0],
    )
    # From the token index, not an RNG: the choice has to be reproducible and
    # agreed upon by every rank that sees the token. See kernels.ops.replica_draw.
    chosen_dispatch_index = replica_index(
        logical_topk_ids.shape,
        logical_topk_ids.shape[-1],
        logical_to_physical_count[safe_ids],
        logical_topk_ids.device,
    )
    physical_ids = logical_to_physical_candidates[safe_ids, chosen_dispatch_index]
    return torch.where(valid, physical_ids, logical_topk_ids)


def route_replicas(
    policy_name: str | None,
    logical_topk_ids: Tensor,
    *,
    default_physical_for_logical: Tensor | None,
    logical_to_physical_candidates: Tensor,
    logical_to_physical_count: Tensor,
) -> Tensor:
    """Route replica IDs through one EvenMoE-selected inline policy."""

    if policy_name is None:
        return logical_topk_ids
    if policy_name == "static":
        if default_physical_for_logical is None:
            raise ValueError("Static replica routing requires a default physical map.")
        return route_static_replicas(logical_topk_ids, default_physical_for_logical)
    if policy_name == "dynamic":
        return route_dynamic_replicas(
            logical_topk_ids,
            logical_to_physical_candidates,
            logical_to_physical_count,
        )
    raise ValueError(f"Unknown inline replica-routing policy {policy_name!r}.")


def _safe_logical_ids(
    logical_topk_ids: Tensor,
    num_logical_experts: int,
) -> tuple[Tensor, Tensor]:
    if num_logical_experts <= 0:
        raise ValueError("Replica routing requires at least one logical expert.")
    valid = (logical_topk_ids >= 0) & (logical_topk_ids < num_logical_experts)
    safe_ids = logical_topk_ids.clamp(
        min=0,
        max=num_logical_experts - 1,
    ).to(torch.int64)
    return safe_ids, valid


def _fused_replica_dispatch(request, shares, candidates):
    """Route through the shared dispatch kernel.

    All three replica policies differ only in the share table they hand over --
    None for a uniform draw, one-hot for a pinned replica, an LP solution for
    LPLB -- so they can share one launch instead of each assembling ids from
    tensor ops. Measured on H20 at 29.6 us per layer against 151.9 us for the
    tensor-op route.

    Load recording is the framework's job, not this kernel's: it always runs
    the load view it already owns through its own pass afterward, so handing
    it one here would only get counted twice.
    """
    from evenmoe.kernels.lplb import CUDALPLBKernels

    kernels = getattr(request, "_kernels", None) or CUDALPLBKernels()
    return kernels.dispatch_probability(
        request.logical_topk_ids,
        shares,
        candidates,
    )



def route_dynamic_random_replicas(
    logical_topk_ids: Tensor,
    candidates: Tensor,
    counts: Tensor,
) -> Tensor:
    """SGLang's `dynamic`, reproduced verbatim.

    `torch.randint(0, 65536) % live_replicas`, exactly as
    `_topk_ids_logical_to_physical_dynamic` does it. Kept as its own policy
    rather than folded into `dynamic` because the two are not interchangeable
    here: the draw is per-process RNG, and tensor-parallel peers see the same
    tokens while holding different experts, so each peer draws its own numbers
    and a token can be sent to two different replicas of its expert. That is
    why `dynamic` uses a hash of the token index instead.

    This exists so the two runtimes can be compared on the same policy, not so
    it can be deployed under TP. `is_tp_safe` says so, and the orchestrator is
    expected to surface that rather than let it be selected silently.
    """
    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Replica routing requires torch.Tensor logical_topk_ids.")
    shape, dtype = logical_topk_ids.shape, logical_topk_ids.dtype
    flat = logical_topk_ids.reshape(-1).to(torch.int64)
    safe = flat.clamp(0, candidates.shape[0] - 1)
    live = counts.to(torch.int64)[safe].clamp(min=1)
    idx = torch.randint(
        0, 65536, flat.shape, dtype=torch.int32, device=flat.device
    ).to(torch.int64) % live
    picked = candidates.to(torch.int64)[safe, idx]
    return torch.where(flat >= 0, picked, flat).reshape(shape).to(dtype)


class StaticL2Router(TokenRouter):
    """Deterministic routing through the framework-selected replica map.

    Always resolves ids on the tensor route below. A share-table form used to
    exist for runtimes that fuse replica selection with load recording, but
    vLLM -- the only caller that had such a kernel -- now bakes this policy's
    answer into a placement-fixed map instead (see
    ``RoutingCapabilities.dispatch_fixed_by_placement``) and never calls this
    method at all, and routing the share table through that same fused kernel
    had already measured worse than this tensor route in three attempts
    (-3.79% against -2.03%). Nothing else ever asked for the share form.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("Static replica routing requires a committed placement.")
        default_map = request.placement.default_physical_for_logical
        if default_map is None:
            raise ValueError(
                "Static replica routing requires " "PlacementSnapshot.default_physical_for_logical."
            )
        physical_ids = route_static_replicas(request.logical_topk_ids, default_map)
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


class DynamicL2Router(TokenRouter):
    """Random routing among the committed replicas of each logical expert."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("Dynamic replica routing requires a committed placement.")
        candidates = request.placement.logical_to_physical_candidates
        if request.logical_topk_ids.is_cuda:
            # No share table: the kernel then draws the way the framework's own
            # does, hash modulo live replicas, which is what keeps this policy
            # bit-identical to the built-in path it is the control for.
            physical_ids = _fused_replica_dispatch(request, None, candidates)
        else:
            physical_ids = route_dynamic_replicas(
                request.logical_topk_ids,
                candidates,
                request.placement.logical_to_physical_count,
            )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


class DynamicRandomL2Router(TokenRouter):
    """`dynamic` as SGLang draws it: a per-call RNG rather than a token hash.

    Provided for cross-runtime comparison. Not TP-safe -- see
    :func:`route_dynamic_random_replicas`.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING
    is_tp_safe = False

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("Dynamic replica routing requires a committed placement.")
        physical_ids = route_dynamic_random_replicas(
            request.logical_topk_ids,
            request.placement.logical_to_physical_candidates,
            request.placement.logical_to_physical_count,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


ROUTING_POLICIES = (
    RoutingPolicySpec(
        "static",
        StaticL2Router.phase,
        requires_rank_dispatch_map=True,
        dispatch_fixed_by_placement=True,
        build_router=lambda context: StaticL2Router(),
    ),
    RoutingPolicySpec(
        "dynamic",
        DynamicL2Router.phase,
        build_router=lambda context: DynamicL2Router(),
    ),
    RoutingPolicySpec(
        "dynamic_random",
        DynamicRandomL2Router.phase,
        build_router=lambda context: DynamicRandomL2Router(),
    ),
)

# Fake is a pre-TopK workload stimulus; its historical physical routing is
# Dynamic when no explicit replica router is selected.
INPUT_TRANSFORMS = (
    InputTransformSpec("fake", transform_fake_routing_inputs, physical_routing="dynamic"),
)
