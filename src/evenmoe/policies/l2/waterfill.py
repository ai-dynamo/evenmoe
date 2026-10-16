# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert Waterfill L2 router.

The algorithm is a degenerate L2 case: it does not modify the routed top-k
(which already lives in physical-id space after the preceding replica router),
and only decides which EP rank receives each token's shared expert. The two
load views are distinguished by whether
``RoutingRequest.active_rank_token_count`` is provided:

- Local-load Waterfill (``waterfill``, no EP all-reduce): the orchestrator passes
  ``routed_rank_load = local_routed
  counts`` and ``active_rank_token_count = None``. The kernel sees
  ``rank_load == routed_load`` and derives ``target_total`` from their sum.
  ``ALLOW_ALL_RANKS=True`` (caller asserts every rank is a valid shared-expert
  destination because EPLB already balances routed load).

- Collective-backed Waterfill (``WATERFILL_USE_GLOBAL_LOAD=1``): the orchestrator passes
  ``routed_rank_load`` (post-AR
  global routed counts) and ``active_rank_token_count`` (post-AR per-rank
  active token counts). The kernel uses ``rank_load = routed + active`` for
  scoring and ``routed_load = routed`` for ``total_tokens`` derivation. The
  candidate set is always restricted to ``token's routed ranks ∪ source``
  (``ALLOW_ALL_RANKS=False``) — this is conservative (loses the "fully
  balanced → all-ranks" optimization the PR's CPU-sync-bearing path enabled)
  but eliminates the two GPU→CPU syncs per layer.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from torch import Tensor

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import RouterBuildContext, RoutingPolicySpec
from evenmoe.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.kernels.ops.waterfill import waterfill_select


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

        # Both paths are sync-free: routed_load is always pure routed counts
        # (used by the kernel to derive total_tokens internally). rank_load is
        # the effective load used for scoring (= routed in static; routed+active
        # in dynamic — a single elementwise add, no .item()).
        if active is None:
            rank_load = routed
            routed_load = routed
            allow_all_ranks = True
        else:
            rank_load = routed + active
            routed_load = routed
            allow_all_ranks = False

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
                "allow_all_ranks": allow_all_ranks,
            },
        )


def _configure(config: RoutingPolicyConfig) -> RoutingPolicySpec:
    # TODO: Replace WATERFILL_USE_GLOBAL_LOAD with an explicit
    # RoutingPolicyConfig option read from ``config``.
    del config
    if os.environ.get("WATERFILL_USE_GLOBAL_LOAD", "0") == "1":
        # Global Waterfill cannot follow LPLB yet: all ranks are not guaranteed
        # to enter the second collective.
        return replace(ROUTING_POLICY, rank_load="global", incompatible_with=frozenset({"lplb"}))
    return ROUTING_POLICY


def _build_router(context: RouterBuildContext) -> WaterfillL2Router:
    return WaterfillL2Router(
        source_rank=context.source_rank,
        world_size=context.ep_size,
        experts_per_rank=context.experts_per_rank,
    )


ROUTING_POLICY = RoutingPolicySpec(
    "waterfill",
    WaterfillL2Router.phase,
    rank_load="local",
    requires_deepep=True,
    configure=_configure,
    build_router=_build_router,
)
