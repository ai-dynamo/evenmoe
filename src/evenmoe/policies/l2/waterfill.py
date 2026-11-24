# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert Waterfill L2 router.

The algorithm is a degenerate L2 case: it does not modify the routed top-k
(which already lives in physical-id space after the preceding replica router),
and only decides which EP rank receives each token's shared expert. Two
independent options shape it, set with ``WATERFILL_LOAD_VIEW`` /
``WATERFILL_CANDIDATES`` or, per policy, ``RoutingPolicyConfig.metadata``
(which takes precedence):

- ``load_view``: ``global`` (default) balances against the EP-wide routed
  load plus every rank's token count, which the orchestrator gathers with one
  all-reduce per MoE layer; ``local`` uses this rank's own routed load and no
  collective. ``global`` falls back to ``local`` with a warning where the
  collective is unsafe (after LPLB, without collectives, on MegaMoE).
- ``candidates``: ``routed`` (default) keeps the shared row on the source
  rank or a rank the token's routed experts already go to; ``all`` allows
  any rank.

``local`` + ``routed`` estimates the EP-wide view from this rank alone
(routed load x W, plus this batch's rows on every rank). The uniform term
cancels in the kernel's weights, so they are W times the ``local`` + ``all``
weights; only the candidate set differs. Every path is sync-free.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from torch import Tensor

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import A2A_DEEPEP, A2A_MEGAMOE, RouterBuildContext, RoutingPolicySpec
from evenmoe.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.kernels.ops.waterfill import waterfill_select

# option -> (environment variable, allowed values, the first being the default)
_OPTIONS = {
    "load_view": ("WATERFILL_LOAD_VIEW", ("global", "local")),
    "candidates": ("WATERFILL_CANDIDATES", ("routed", "all")),
}


def waterfill_option(config: RoutingPolicyConfig | None, option: str) -> str:
    """``config.metadata[option]``, else its environment variable, else the default."""

    env, allowed = _OPTIONS[option]
    value = None if config is None else config.metadata.get(option)
    if value is None:
        value = os.environ.get(env) or allowed[0]
    choice = str(value).strip().lower()
    if choice not in allowed:
        raise ValueError(f"Waterfill {option}={value!r} is not one of {', '.join(allowed)}.")
    return choice


class WaterfillL2Router(TokenRouter):
    """Shared-expert Waterfill rank selection router.

    Implements ``core.TokenRouter``. ``route_tokens`` consumes a
    ``RoutingRequest`` whose canonical fields carry per-batch state:

    - ``request.routed_rank_load``: per-rank routed expert assignment count.
    - ``request.active_rank_token_count``: per-rank active token count
      (set only in dynamic mode; ``None`` signals static mode).
    The constructor receives the stable contiguous physical layout geometry;
    per-request tensors remain in ``RoutingRequest``.

    The router does not consume ``request.placement`` (Waterfill operates on
    already-physical IDs produced earlier in the EvenMoE token pipeline).
    """

    phase = RoutingPhase.SHARED_EXPERT

    def __init__(
        self,
        *,
        source_rank: int,
        local_preference: float = 1.1,
        world_size: int,
        experts_per_rank: int,
        candidates: str = "routed",
    ):
        if world_size <= 0:
            raise ValueError("WaterfillL2Router.world_size must be positive.")
        if not (0 <= source_rank < world_size):
            raise ValueError(
                "WaterfillL2Router.source_rank must satisfy 0 <= source_rank < world_size."
            )
        if local_preference <= 0:
            raise ValueError("WaterfillL2Router.local_preference must be positive.")
        if experts_per_rank <= 0:
            raise ValueError("WaterfillL2Router.experts_per_rank must be positive.")
        self.source_rank = source_rank
        self.local_preference = local_preference
        self.world_size = world_size
        self.experts_per_rank = experts_per_rank
        if candidates not in _OPTIONS["candidates"][1]:
            raise ValueError(f"WaterfillL2Router.candidates={candidates!r} is not routed or all.")
        self.candidates = candidates

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        topk_ids = request.routed_physical_topk_ids
        if topk_ids is None:
            raise ValueError("WaterfillL2Router requires routed_physical_topk_ids.")
        if not isinstance(topk_ids, Tensor):
            raise TypeError("WaterfillL2Router requires torch.Tensor topk_ids.")
        routed = request.routed_rank_load
        if routed is None or not isinstance(routed, Tensor):
            raise ValueError("WaterfillL2Router requires routed_rank_load as a torch.Tensor.")

        active = request.active_rank_token_count
        if active is not None and not isinstance(active, Tensor):
            raise TypeError(
                "WaterfillL2Router.active_rank_token_count must be a torch.Tensor or None."
            )

        if topk_ids.numel() == 0:
            return RoutingDecision(
                routed_physical_topk_ids=topk_ids,
                topk_weights=request.topk_weights,
                shared_expert_rank=torch.empty(
                    topk_ids.shape[0],
                    dtype=torch.int64,
                    device=topk_ids.device,
                ),
                metadata={"policy": policy.name, "empty": True},
            )

        candidates = (
            waterfill_option(policy, "candidates")
            if "candidates" in policy.metadata
            else self.candidates
        )
        allow_all_ranks = candidates == "all"
        if active is not None:
            # Global view: rank_load scores, routed_load sets the fill level.
            rank_load = routed + active
            routed_load = routed
        elif allow_all_ranks:
            rank_load = routed
            routed_load = routed
        else:
            # Local view estimate: this rank's load stands in for every rank's.
            routed_load = routed.to(torch.int64) * self.world_size
            rank_load = routed_load + int(topk_ids.shape[0])

        chosen_rank = waterfill_select(
            rank_load=rank_load,
            routed_load=routed_load,
            topk_ids=topk_ids,
            allow_all_ranks=allow_all_ranks,
            source_rank=self.source_rank,
            local_preference=self.local_preference,
            world_size=self.world_size,
            experts_per_rank=self.experts_per_rank,
        )

        return RoutingDecision(
            routed_physical_topk_ids=topk_ids,
            topk_weights=request.topk_weights,
            shared_expert_rank=chosen_rank,
            metadata={
                "policy": policy.name,
                "load_view": "local" if active is None else "global",
                "allow_all_ranks": allow_all_ranks,
            },
        )


def _configure(config: RoutingPolicyConfig) -> RoutingPolicySpec:
    if waterfill_option(config, "load_view") == "local":
        return _LOCAL
    # Not every rank is guaranteed to enter LPLB's collective, so a second
    # one after it could hang.
    return replace(
        _LOCAL, rank_load="global", incompatible_with=frozenset({"lplb"}), fallback=_LOCAL
    )


def _build_router(context: RouterBuildContext) -> WaterfillL2Router:
    return WaterfillL2Router(
        source_rank=context.source_rank,
        world_size=context.ep_size,
        experts_per_rank=context.experts_per_rank,
        candidates=waterfill_option(None, "candidates"),
    )


_LOCAL = RoutingPolicySpec(
    "waterfill",
    WaterfillL2Router.phase,
    rank_load="local",
    supported_a2a_backends=(A2A_DEEPEP, A2A_MEGAMOE),
    build_router=_build_router,
)
ROUTING_POLICY = replace(_LOCAL, configure=_configure)
