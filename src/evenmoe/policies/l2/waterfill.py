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

Tile mode (``tile_rows`` > 1, from metadata or ``EVENMOE_WATERFILL_TILE_ROWS``;
``tile_rho`` / ``EVENMOE_WATERFILL_TILE_RHO``; a request may carry its kernel's
tile in ``metadata["tile_rows"]``): place the shared rows in whole GEMM tiles of
a plan every rank computes from the exchanged per-expert counts
(:mod:`evenmoe.kernels.tile_waterfill`). Every EP rank must make every routing
call, idle ranks included (with an empty batch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch
from torch import Tensor

from evenmoe.core.l2 import RoutingCollectives, TokenRouter
from evenmoe.core.policy import A2A_DEEPEP, A2A_MEGAMOE, RouterBuildContext, RoutingPolicySpec
from evenmoe.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from evenmoe.kernels.ops.waterfill import waterfill_select


@dataclass(frozen=True)
class TileConfig:
    """Tile-mode knobs: ``tile_rows`` (1 = row-level Waterfill) and ``rho``."""

    tile_rows: int = 1
    rho: float = 1.5

    def __post_init__(self) -> None:
        if type(self.tile_rows) is not int or self.tile_rows < 1:
            raise ValueError(f"waterfill tile_rows must be an int >= 1, got {self.tile_rows!r}.")
        if not self.rho >= 0:
            raise ValueError(f"waterfill tile_rho must be non-negative, got {self.rho!r}.")

    @classmethod
    def from_config(cls, config: RoutingPolicyConfig) -> TileConfig:
        def knob(key: str, env: str, default: str) -> Any:
            value = config.metadata.get(key)
            return value if value is not None else (os.environ.get(env, "").strip() or default)

        rows = knob("tile_rows", "EVENMOE_WATERFILL_TILE_ROWS", "1")
        try:
            tile_rows = int(rows) if isinstance(rows, str) else rows
        except ValueError:
            raise ValueError(f"waterfill tile_rows must be an int >= 1, got {rows!r}.") from None
        return cls(tile_rows, float(knob("tile_rho", "EVENMOE_WATERFILL_TILE_RHO", "1.5")))


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
        tile: TileConfig | None = None,
        process_group: Any = None,
        collectives: RoutingCollectives | None = None,
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
        self.tile_mode = (
            _TileWaterfill(tile, source_rank, world_size, experts_per_rank, process_group, collectives)
            if tile is not None and tile.tile_rows > 1
            else None
        )

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Tile mode: set up the P2P exchange once (a collective, outside graph capture)."""

        if self.tile_mode is not None:
            self.tile_mode.set_up_exchange()

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if self.tile_mode is not None:
            return self.tile_mode.route(request, policy)
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
    tile = TileConfig.from_config(config)
    if tile.tile_rows > 1:
        if os.environ.get("WATERFILL_USE_GLOBAL_LOAD", "0") == "1" or config.enabled_stages is not None:
            raise ValueError(
                "waterfill tile_rows > 1 needs every EP rank in every routing call; it cannot be "
                "combined with WATERFILL_USE_GLOBAL_LOAD=1 or restricted to some stages."
            )
        return replace(
            ROUTING_POLICY,
            rank_load=None,  # counts the routed physical ids itself
            supported_a2a_backends=(A2A_DEEPEP, A2A_MEGAMOE),
            incompatible_with=frozenset({"ultraep"}),  # not validated with tile mode
            supports_concurrent_microbatches=False,  # one P2P epoch per process
            build_router=partial(_build_router, tile=tile),
        )
    if os.environ.get("WATERFILL_USE_GLOBAL_LOAD", "0") == "1":
        # Global Waterfill cannot follow LPLB yet: all ranks are not guaranteed
        # to enter the second collective.
        return replace(ROUTING_POLICY, rank_load="global", incompatible_with=frozenset({"lplb"}))
    return ROUTING_POLICY


def _build_router(context: RouterBuildContext, tile: TileConfig | None = None) -> WaterfillL2Router:
    collectives: Any = context.options.get("ep_collectives")  # the framework's RoutingCollectives
    return WaterfillL2Router(
        source_rank=context.source_rank,
        world_size=context.ep_size,
        experts_per_rank=context.experts_per_rank,
        tile=tile,
        process_group=getattr(collectives, "device_group", None),
        collectives=collectives,
    )


class _TileWaterfill:
    """Tile mode of :class:`WaterfillL2Router`."""

    def __init__(
        self,
        config: TileConfig,
        rank: int,
        num_ranks: int,
        experts_per_rank: int,
        process_group: Any,
        collectives: RoutingCollectives | None,
    ):
        from evenmoe.kernels.tile_waterfill import MAX_RANKS, payload_width

        if num_ranks > MAX_RANKS:
            raise ValueError(f"waterfill tile mode supports at most {MAX_RANKS} EP ranks, got {num_ranks}.")
        self.config = config
        self.rank = rank
        self.num_ranks = num_ranks
        self.experts_per_rank = experts_per_rank
        self.width = payload_width(num_ranks * experts_per_rank)
        self._process_group = process_group
        self._collectives = collectives
        self._set_up = False
        self.peer_exchange: Any = None  # NVLink P2P buffers; None = the framework all-reduce

    def set_up_exchange(self) -> None:
        if self._set_up or self.num_ranks == 1 or self._process_group is None:
            return
        from evenmoe.kernels.p2p import create_peer_exchange

        self.peer_exchange = create_peer_exchange(self._process_group, width=self.width)
        self._set_up = True

    def route(self, request: RoutingRequest, policy: RoutingPolicyConfig) -> RoutingDecision:
        from evenmoe.kernels.tile_waterfill import count_and_exchange, plan_and_assign

        physical = request.routed_physical_topk_ids
        if not isinstance(physical, Tensor):
            raise TypeError("waterfill tile mode requires routed_physical_topk_ids as a torch.Tensor.")
        tile_rows = request.metadata.get("tile_rows", self.config.tile_rows)
        if type(tile_rows) is not int or tile_rows < 1:
            raise ValueError(f"RoutingRequest.metadata['tile_rows'] must be an int >= 1, got {tile_rows!r}.")
        rows, status = count_and_exchange(
            physical, self.num_ranks * self.experts_per_rank, request.token_count, tile_rows, self.peer_exchange
        )
        if status is None and self.num_ranks > 1:
            if self._collectives is None:
                raise RuntimeError("waterfill tile mode needs RoutingCollectives (or P2P) for more than one rank.")
            # All-gather as one sum all-reduce of a zeroed [R, W] payload (exact in fp32 below 2**24).
            padded = torch.zeros((self.num_ranks, self.width), dtype=torch.float32, device=rows.device)
            padded[self.rank].copy_(rows)
            gathered = self._collectives.all_reduce_sum(padded)
            assert isinstance(gathered, Tensor)
            rows = gathered
        shared_rank = plan_and_assign(
            rows.view(self.num_ranks, -1),
            num_local_physical=self.experts_per_rank,
            rank=self.rank,
            num_tokens=physical.shape[0],
            rho=self.config.rho,
            status=status,
            out_dtype=physical.dtype,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
            shared_expert_rank=shared_rank,
            metadata={"policy": policy.name},
        )


ROUTING_POLICY = RoutingPolicySpec(
    "waterfill",
    WaterfillL2Router.phase,
    rank_load="local",
    supported_a2a_backends=(A2A_DEEPEP, A2A_MEGAMOE),
    configure=_configure,
    build_router=_build_router,
)
