# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert Waterfill L2 router.

The algorithm is a degenerate L2 case: it does not modify the routed top-k
(which already lives in physical-id space after the preceding replica router),
and only decides which EP rank receives each token's shared expert. Two
independent options shape it.

``load_view`` -- which per-rank load the weights are computed from:

- ``local`` (default): this rank's own routed load. No collective.
- ``global``: the EP-wide routed load plus every rank's active token count,
  gathered by the orchestrator with one int64 all-reduce (2 x EP values) per
  MoE layer. It falls back to ``local`` with one warning when the collective
  cannot run safely: after LPLB (not every rank is guaranteed to enter LPLB's
  collective, so a second one could hang), with no ``RoutingCollectives`` and
  EP size > 1, with no committed placement, or on an A2A backend that may skip
  routing on ranks without tokens (MegaMoE). The fallback keeps ``candidates``.

``candidates`` -- which ranks may receive a token's shared expert:

- ``routed`` (default): the source rank plus the ranks the token's routed
  experts already go to (kernel ``ALLOW_ALL_RANKS=False``), so the shared row
  adds no new dispatch destination.
- ``all``: any rank (``ALLOW_ALL_RANKS=True``).

The kernel scores ``rank_load`` and derives ``target_total`` from
``routed_load``; every weight is ``target_total - rank_load[r]``:

========  ==========  =====================================  ===========
view      candidates  rank_load / routed_load                ALL_RANKS
========  ==========  =====================================  ===========
local     all         local / local                          True
local     routed      local*W + T_local / local*W            False
global    routed      global + active / global               False
global    all         global + active / global               True
========  ==========  =====================================  ===========

``local``+``routed`` scales the local load by ``W`` and adds the uniform term
``T_local`` (this batch's rows) so the restricted kernel path can be used
without a collective. A uniform term cancels in ``target_total - rank_load``,
so the weights are exactly ``W`` times the ``local``+``all`` weights (up to
the ceiling in ``target_total``); only the candidate set differs. Every path
is sync-free.

Deployments set the options with the ``WATERFILL_LOAD_VIEW`` (``local`` /
``global``) and ``WATERFILL_CANDIDATES`` (``routed`` / ``all``) environment
variables, since the framework selectors (SGLang
``--moe-load-balancer-algorithm``, vLLM ``VLLM_EVENMOE_L2_ALGORITHM``) carry
only the algorithm name. Programmatic callers can instead set them per policy
in ``RoutingPolicyConfig(name="waterfill", metadata={"load_view": ...,
"candidates": ...})``, which takes precedence over the environment.
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

LOAD_VIEWS = ("local", "global")
CANDIDATE_SETS = ("routed", "all")
DEFAULT_LOAD_VIEW = "local"
DEFAULT_CANDIDATES = "routed"

# The deployment-wide settings; RoutingPolicyConfig.metadata overrides them.
_LOAD_VIEW_ENV = "WATERFILL_LOAD_VIEW"
_CANDIDATES_ENV = "WATERFILL_CANDIDATES"


def waterfill_option(config: RoutingPolicyConfig | None, option: str) -> str:
    """Resolve ``load_view`` or ``candidates`` for one Waterfill selection.

    Precedence: ``config.metadata[option]``, then the option's environment
    variable (``WATERFILL_LOAD_VIEW`` / ``WATERFILL_CANDIDATES``), then the
    default.
    """

    if option == "load_view":
        env, allowed, default = _LOAD_VIEW_ENV, LOAD_VIEWS, DEFAULT_LOAD_VIEW
    elif option == "candidates":
        env, allowed, default = _CANDIDATES_ENV, CANDIDATE_SETS, DEFAULT_CANDIDATES
    else:
        raise ValueError(f"Unknown Waterfill option {option!r}.")
    value = None if config is None else config.metadata.get(option)
    source = f"RoutingPolicyConfig.metadata[{option!r}]"
    if value is None:
        value = os.environ.get(env, "").strip() or None
        source = env
    if value is None:
        return default
    choice = str(value).strip().lower()
    if choice not in allowed:
        raise ValueError(f"{source}={value!r} is not one of {', '.join(allowed)}.")
    return choice


class WaterfillL2Router(TokenRouter):
    """Shared-expert Waterfill rank selection router.

    Implements ``core.TokenRouter``. ``route_tokens`` consumes a
    ``RoutingRequest`` whose canonical fields carry per-batch state:

    - ``request.routed_rank_load``: per-rank routed expert assignment count.
    - ``request.active_rank_token_count``: per-rank active token count, set
      only for the global load view (``None`` means the local view).

    ``candidates`` is the default candidate set; a selection's
    ``RoutingPolicyConfig.metadata["candidates"]`` overrides it per call.
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
        candidates: str = DEFAULT_CANDIDATES,
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
        if candidates not in CANDIDATE_SETS:
            raise ValueError(f"WaterfillL2Router.candidates must be one of {CANDIDATE_SETS}.")
        self.source_rank = source_rank
        self.local_preference = local_preference
        self.world_size = world_size
        self.experts_per_rank = experts_per_rank
        self.candidates = candidates

    def _candidates_for(self, policy: RoutingPolicyConfig) -> str:
        if policy.metadata.get("candidates") is None:
            return self.candidates
        return waterfill_option(policy, "candidates")

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

        # Every branch is sync-free: elementwise ops on device tensors and a
        # Python-int row count taken from the (static) shape.
        allow_all_ranks = self._candidates_for(policy) == "all"
        load_view = "local" if active is None else "global"
        if active is not None:
            rank_load = routed + active
            routed_load = routed
        elif allow_all_ranks:
            rank_load = routed
            routed_load = routed
        else:
            # Local view on the restricted path: see the module docstring.
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
                "load_view": load_view,
                "allow_all_ranks": allow_all_ranks,
            },
        )


def _configure(config: RoutingPolicyConfig) -> RoutingPolicySpec:
    if waterfill_option(config, "load_view") == "global":
        return _GLOBAL_LOAD
    return _LOCAL_LOAD


def _build_router(context: RouterBuildContext) -> WaterfillL2Router:
    return WaterfillL2Router(
        source_rank=context.source_rank,
        world_size=context.ep_size,
        experts_per_rank=context.experts_per_rank,
        candidates=waterfill_option(None, "candidates"),
    )


_LOCAL_LOAD = RoutingPolicySpec(
    "waterfill",
    WaterfillL2Router.phase,
    rank_load="local",
    supported_a2a_backends=(A2A_DEEPEP, A2A_MEGAMOE),
    build_router=_build_router,
)
# Global Waterfill cannot follow LPLB yet: all ranks are not guaranteed to
# enter the second collective. It falls back to the local view instead.
_GLOBAL_LOAD = replace(
    _LOCAL_LOAD,
    rank_load="global",
    incompatible_with=frozenset({"lplb"}),
    fallback=_LOCAL_LOAD,
)
ROUTING_POLICY = replace(_LOCAL_LOAD, configure=_configure)
