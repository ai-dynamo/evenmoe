# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tile_repair: tile-aware fixed-first + repair physical routing.

Per layer per forward, using the EP-wide per-source-rank logical counts, all
tokens of an expert start on its primary copy; then single partial GEMM tiles
of replicated experts move off the busiest rank while that strictly lowers its
tile count (:mod:`evenmoe.kernels.tile_repair`). Tokens are then assigned to
copies with exact quotas, so every rank computes the same plan and the
per-copy totals equal it exactly. The busiest rank's tile count is never
above primary-only static routing.

Every supported layer with extra slots gathers per-rank counts through the
framework's all-reduce, including stages that route primary-only. Participation
stays fixed across same-shape placement updates for CUDA Graph replay. Each
call owns its scratch tensors; committed placement tables are read-only during
routing. Placement updates must be synchronized with in-flight calls.

Pair with the ``tile`` placement policy and the same ``block_m``. Defaults:
64-row tiles, at most 32 moves, repair in decode/idle/speculative stages.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from evenmoe.core.l2 import TokenRouter
from evenmoe.core.policy import RouterBuildContext, RoutingPolicySpec
from evenmoe.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from evenmoe.kernels.tile_repair import (
    TileRepairTables,
    build_tile_repair_tables,
    supports,
    tile_repair_route,
    tile_repair_route_reference,
)
from evenmoe.policies.l2.replica import route_static_replicas
from evenmoe.policies.tile_common import (
    resolve_block_m,
    resolve_max_moves,
    resolve_stages,
)

logger = logging.getLogger(__name__)


@dataclass
class _LayerState:
    tables: TileRepairTables
    num_slots: int

    @property
    def has_extra_slots(self) -> bool:
        # Unlike the current number of distinct-rank copies, capacity cannot
        # change in a same-shape placement update.
        return self.num_slots > self.tables.num_logical


def tables_from_snapshot(snapshot: PlacementSnapshot) -> TileRepairTables:
    """Rank-invariant tile_repair tables for one committed layer placement."""

    p2l = snapshot.physical_to_logical_map
    p2l = p2l if isinstance(p2l, Tensor) else torch.as_tensor(p2l)
    if p2l.numel() != snapshot.num_physical_experts:
        raise ValueError(
            f"physical_to_logical_map has {p2l.numel()} entries for "
            f"{snapshot.num_physical_experts} physical experts."
        )
    rank_map = snapshot.physical_to_rank_map
    if rank_map is not None and not isinstance(rank_map, Tensor):
        rank_map = torch.as_tensor(rank_map)
    return build_tile_repair_tables(
        p2l,
        num_logical=snapshot.num_logical_experts,
        num_ranks=snapshot.ep_size,
        num_local_physical_experts=snapshot.num_local_physical_experts,
        physical_to_rank_map=rank_map,
        device=p2l.device,
    )


class TileRepairL2Router(TokenRouter):
    """tile_repair physical-replica router (see the module docstring)."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(
        self,
        *,
        num_ranks: int,
        source_rank: int,
        block_m: int | None = None,
        max_moves: int | None = None,
        stages: frozenset[RoutingStage] | str | None = None,
        out_dtype: torch.dtype | None = torch.int64,
    ):
        if num_ranks <= 0:
            raise ValueError("TileRepairL2Router.num_ranks must be positive.")
        if not 0 <= source_rank < num_ranks:
            raise ValueError("TileRepairL2Router.source_rank must lie in [0, num_ranks).")
        self.num_ranks = num_ranks
        self.source_rank = source_rank
        self.block_m = resolve_block_m({"block_m": block_m})
        self.max_moves = resolve_max_moves({"max_moves": max_moves})
        self.stages = resolve_stages({"stages": stages})
        self.out_dtype = out_dtype
        self._layer_state: dict[int, _LayerState] = {}
        self._warned_fallback = False

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Install the tables of a newly committed placement for one layer.

        Tables of an unchanged shape are rewritten in place so a captured CUDA
        graph keeps reading the live placement.
        """

        state = self._build_state(snapshot)
        current = self._layer_state.get(snapshot.layer_id)
        if current is not None and _same_layout(current.tables, state.tables):
            for dst, src in zip(current.tables.tensors(), state.tables.tensors()):
                dst.copy_(src)
            current.num_slots = state.num_slots
            return
        self._layer_state[snapshot.layer_id] = state

    def needs_global_logical_count(self, request: RoutingRequest) -> bool:
        """Keep exchange participation fixed across same-shape updates.

        Deliberately independent of the stage: every EP rank must join the
        collective, including ranks whose own stage routes primary-only.
        """

        if request.placement is None:
            return False
        state = self._state(request.layer_id, request.placement)
        return state.has_extra_slots and (
            not _is_cuda(request.logical_topk_ids)
            or supports(state.tables.num_logical, state.tables.num_ranks)
        )

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig | None = None,
    ) -> RoutingDecision:
        if policy is None:
            if len(request.policies) != 1:
                raise ValueError(
                    "Direct TileRepairL2Router calls require exactly one request policy; "
                    "the unified orchestrator always passes the selected policy explicitly."
                )
            policy = RoutingPolicyConfig.from_value(request.policies[0])
        placement = request.placement
        if placement is None:
            raise ValueError("TileRepairL2Router requires RoutingRequest.placement.")
        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("TileRepairL2Router requires torch.Tensor logical_topk_ids.")
        out_dtype = self.out_dtype or topk_ids.dtype
        state = self._state(request.layer_id, placement)
        block_m, max_moves, stages = self._tunables(policy.metadata)
        tables = state.tables
        if topk_ids.is_cuda and not supports(tables.num_logical, tables.num_ranks):
            if not self._warned_fallback:
                logger.warning(
                    "tile_repair covers <= 512 experts and <= 32 ranks; layer %d "
                    "(%d experts, %d ranks) routes to primary copies.",
                    request.layer_id,
                    tables.num_logical,
                    tables.num_ranks,
                )
                self._warned_fallback = True
            return self._primary_only(request, state, policy, out_dtype)
        if not state.has_extra_slots:
            return self._primary_only(request, state, policy, out_dtype)

        repair = (
            (request.stage is None or request.stage in stages)
            and policy.should_run_for_stage(request.stage)
        )
        if not repair:
            return self._primary_only(request, state, policy, out_dtype)

        counts = request.global_logical_count
        if counts is None:
            raise ValueError(
                "TileRepairL2Router requires the per-source-rank logical count on a layer "
                "with replicated experts."
            )
        if not isinstance(counts, Tensor):
            raise TypeError("TileRepairL2Router requires a torch.Tensor logical count.")
        expected = (tables.num_ranks, tables.num_logical)
        if tuple(counts.shape) != expected:
            raise ValueError(
                f"tile_repair needs the per-source-rank logical count of shape {list(expected)} "
                f"(logical_count='per_rank'), got {list(counts.shape)}; an EP-summed count "
                "cannot give every rank its token order."
            )

        if not topk_ids.is_cuda:
            physical = tile_repair_route_reference(
                topk_ids,
                counts,
                tables,
                source_rank=self.source_rank,
                block_m=block_m,
                max_moves=max_moves,
                num_slots=state.num_slots,
            ).to(out_dtype)
        else:
            physical = tile_repair_route(
                topk_ids,
                counts,
                tables,
                source_rank=self.source_rank,
                block_m=block_m,
                max_moves=max_moves,
                out_dtype=out_dtype,
                num_slots=state.num_slots,
            )
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
            metadata={
                "policy": policy.name,
                "tile_repair": True,
                "block_m": block_m,
                "max_moves": max_moves,
                "exchange": "all_reduce",
            },
        )

    def _primary_only(
        self,
        request: RoutingRequest,
        state: _LayerState,
        policy: RoutingPolicyConfig,
        out_dtype: torch.dtype,
    ) -> RoutingDecision:
        topk_ids = request.logical_topk_ids
        primary = state.tables.primary_physical.to(device=topk_ids.device, dtype=topk_ids.dtype)
        physical = route_static_replicas(topk_ids, primary)
        return RoutingDecision(
            routed_physical_topk_ids=physical.to(out_dtype),
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name, "tile_repair": False},
        )

    def _tunables(self, metadata: Mapping[str, Any]) -> tuple[int, int, frozenset[RoutingStage]]:
        if not metadata:
            return self.block_m, self.max_moves, self.stages
        block_m = int(metadata.get("block_m", self.block_m))
        max_moves = int(metadata.get("max_moves", self.max_moves))
        stages = resolve_stages(metadata) if "stages" in metadata else self.stages
        if block_m < 0 or max_moves < 0:
            raise ValueError("tile_repair needs block_m >= 0 (0 = adaptive) and max_moves >= 0.")
        return block_m, max_moves, stages

    def _state(self, layer_id: int, placement: PlacementSnapshot) -> _LayerState:
        state = self._layer_state.get(layer_id)
        if state is None:
            state = self._build_state(placement)
            self._layer_state[layer_id] = state
        return state

    def _build_state(self, snapshot: PlacementSnapshot) -> _LayerState:
        if snapshot.ep_size != self.num_ranks:
            raise ValueError(
                f"Placement ep_size={snapshot.ep_size} does not match the router's "
                f"num_ranks={self.num_ranks}."
            )
        tables = tables_from_snapshot(snapshot)
        return _LayerState(
            tables=tables,
            num_slots=snapshot.num_physical_experts,
        )


def _same_layout(left: TileRepairTables, right: TileRepairTables) -> bool:
    return all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        for a, b in zip(left.tensors(), right.tensors())
    )


def _is_cuda(value: object) -> bool:
    return isinstance(value, Tensor) and value.is_cuda


def _build_router(context: RouterBuildContext) -> TileRepairL2Router:
    # Optional injections; unset ones fall back to the environment and defaults.
    options: Mapping[str, Any] = context.options
    return TileRepairL2Router(
        num_ranks=context.ep_size,
        source_rank=context.source_rank,
        block_m=options.get("tile_block_m"),
        max_moves=options.get("tile_max_moves"),
        stages=options.get("tile_repair_stages"),
    )


ROUTING_POLICY = RoutingPolicySpec(
    "tile_repair",
    TileRepairL2Router.phase,
    logical_count="per_rank",
    requires_placement_state=True,
    # A stage gate set through RoutingPolicyConfig.enabled_stages skips the
    # policy, and the framework's static replica map answers instead.
    requires_rank_dispatch_map=True,
    requires_deepep=True,
    requires_redundancy=True,
    build_router=_build_router,
)
