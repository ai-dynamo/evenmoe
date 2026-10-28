# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""METRO-repair: METRO decode routing plus a bounded repair of its greedy.

Same contract as :mod:`evenmoe.policies.l2.metro`: the policy runs in every
stage (every EP rank joins the count collective); decode, speculative and idle
steps use the METRO-repair assignment; prefill and mixed steps use the
framework's static replica map. The assignment is METRO's greedy followed by
at most ``moves`` repair steps (:mod:`evenmoe.kernels.metro_repair`), so
``moves=0`` reproduces ``metro`` exactly.

Placement tables are built from the rank-invariant ``physical_to_logical_map``
rather than from ``logical_to_physical_candidates``: a framework may prune the
candidate list per source rank (SGLang with an a2a backend keeps only the
local copy when one exists), and per-rank tables would let ranks disagree on
an expert's copy, so one expert would run on several ranks.

Policy metadata ``{"moves": int, "early_exit": bool}`` overrides the router's
defaults (32 moves, early exit at the lower bound).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
from evenmoe.kernels.metro_repair import (
    MetroRepairTables,
    build_metro_repair_tables,
    metro_repair_assign_reference,
    metro_repair_route,
    supports,
)
from evenmoe.kernels.ops.metro import build_metro_tables, metro_route
from evenmoe.policies.l2.replica import route_static_replicas

logger = logging.getLogger(__name__)

_STATIC_STAGES = frozenset({RoutingStage.PREFILL, RoutingStage.MIXED})

DEFAULT_MOVES = 32


@dataclass
class _LayerState:
    tables: MetroRepairTables
    has_redundancy: bool


def rank_invariant_candidates(snapshot: PlacementSnapshot) -> Tensor:
    """``[L, max_copies]`` candidate table rebuilt from the physical->logical map.

    Every live entry of ``snapshot.logical_to_physical_candidates`` must be a
    copy of its logical expert under that map; the rebuilt table lists every
    copy, so it is the same on every rank.
    """

    p2l = snapshot.physical_to_logical_map
    p2l = (p2l if isinstance(p2l, Tensor) else torch.as_tensor(p2l)).reshape(-1).to(torch.int64)
    num_logical = snapshot.num_logical_experts
    if p2l.numel() != snapshot.num_physical_experts:
        raise ValueError(
            f"physical_to_logical_map has {p2l.numel()} entries for "
            f"{snapshot.num_physical_experts} physical experts."
        )
    host = p2l.cpu()
    if bool(((host < 0) | (host >= num_logical)).any()):
        raise ValueError("physical_to_logical_map holds an out-of-range logical expert.")

    listed = snapshot.logical_to_physical_candidates
    listed = (
        (listed if isinstance(listed, Tensor) else torch.as_tensor(listed)).to(torch.int64).cpu()
    )
    count = snapshot.logical_to_physical_count
    if isinstance(count, Tensor) and listed.dim() == 2 and listed.shape[1] > 1:
        column = torch.arange(listed.shape[1]).view(1, -1)
        listed = torch.where(column < count.cpu().view(-1, 1), listed, -1)
    live = listed >= 0
    owner = torch.arange(listed.shape[0]).view(-1, 1).expand_as(listed)
    if bool((host[listed.clamp(0, host.numel() - 1)][live] != owner[live]).any()):
        raise ValueError("logical_to_physical_candidates disagrees with physical_to_logical_map.")

    copies = torch.bincount(host, minlength=num_logical)
    if bool((copies == 0).any()):
        raise ValueError("Every logical expert must have at least one physical copy.")
    order = torch.argsort(
        host * host.numel() + torch.arange(host.numel())
    )  # by logical, then physical
    rank_in_logical = torch.arange(host.numel()) - torch.repeat_interleave(
        torch.cumsum(copies, 0) - copies, copies
    )
    table = torch.full((num_logical, int(copies.max())), -1, dtype=torch.int64)
    table[host[order], rank_in_logical] = order
    return table.to(p2l.device)


class MetroRepairL2Router(TokenRouter):
    """METRO-repair physical-replica router (see the module docstring)."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(
        self,
        *,
        num_gpus: int,
        moves: int = DEFAULT_MOVES,
        early_exit: bool = True,
        out_dtype: torch.dtype | None = torch.int64,
    ):
        if num_gpus <= 0:
            raise ValueError("MetroRepairL2Router.num_gpus must be positive.")
        if moves < 0:
            raise ValueError("MetroRepairL2Router.moves must be non-negative.")
        self.num_gpus = num_gpus
        self.moves = moves
        self.early_exit = early_exit
        self.out_dtype = out_dtype
        self._layer_state: dict[int, _LayerState] = {}
        self._warned_fallback = False

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Install the tables of a newly committed placement for one layer.

        Same-shape tables are rewritten in place so captured CUDA graphs stay valid.
        """

        state = self._build_state(snapshot)
        current = self._layer_state.get(snapshot.layer_id)
        if current is not None and _same_layout(current.tables, state.tables):
            for dst, src in zip(current.tables.tensors(), state.tables.tensors()):
                dst.copy_(src)
            current.has_redundancy = state.has_redundancy
            return
        self._layer_state[snapshot.layer_id] = state

    def needs_global_logical_count(self, request: RoutingRequest) -> bool:
        """Only a layer with a replicated expert has a choice to make."""

        if request.placement is None:
            return False
        return self._state(request.layer_id, request.placement).has_redundancy

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig | None = None,
    ) -> RoutingDecision:
        if policy is None:
            if len(request.policies) != 1:
                raise ValueError(
                    "Direct MetroRepairL2Router calls require exactly one request policy; "
                    "the unified orchestrator always passes the selected policy explicitly."
                )
            policy = RoutingPolicyConfig.from_value(request.policies[0])
        placement = request.placement
        if placement is None:
            raise ValueError("MetroRepairL2Router requires RoutingRequest.placement.")
        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("MetroRepairL2Router requires torch.Tensor logical_topk_ids.")
        out_dtype = self.out_dtype or topk_ids.dtype
        state = self._state(request.layer_id, placement)

        if (
            not state.has_redundancy
            or request.stage in _STATIC_STAGES
            or not policy.should_run_for_stage(request.stage)
        ):
            default = placement.default_physical_for_logical
            if not isinstance(default, Tensor):
                default = state.tables.metro.default_physical
            physical = route_static_replicas(topk_ids, default.to(topk_ids.device))
            return RoutingDecision(
                routed_physical_topk_ids=physical.to(out_dtype),
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "metro_assignment": False},
            )

        counts = request.global_logical_count
        if counts is None:
            raise ValueError(
                "MetroRepairL2Router requires RoutingRequest.global_logical_count on a "
                "layer with replicated experts."
            )
        if not isinstance(counts, Tensor):
            raise TypeError("MetroRepairL2Router requires a torch.Tensor global_logical_count.")
        moves = int(policy.metadata.get("moves", self.moves))
        early_exit = bool(policy.metadata.get("early_exit", self.early_exit))
        token_count = request.token_count
        gate = token_count if isinstance(token_count, Tensor) else None

        metro = state.tables.metro
        if not topk_ids.is_cuda:
            physical = _route_host(topk_ids, counts, state.tables, gate, moves, early_exit)
            physical = physical.to(out_dtype)
        elif supports(metro.num_logical, metro.num_ranks):
            physical = metro_repair_route(
                topk_ids,
                state.tables,
                counts=counts,
                num_token_non_padded=gate,
                moves=moves,
                early_exit=early_exit,
                out_dtype=out_dtype,
            )
        else:
            if not self._warned_fallback:
                logger.warning(
                    "METRO-repair kernel covers <= 256 experts and <= 32 ranks; "
                    "layer %d (%d experts, %d ranks) falls back to plain METRO.",
                    request.layer_id,
                    metro.num_logical,
                    metro.num_ranks,
                )
                self._warned_fallback = True
            physical = metro_route(
                topk_ids,
                counts,
                metro,
                num_token_non_padded=gate,
                out_dtype=out_dtype,
            )
            moves = 0
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name, "metro_assignment": True, "repair_moves": moves},
        )

    def _state(self, layer_id: int, placement: PlacementSnapshot) -> _LayerState:
        state = self._layer_state.get(layer_id)
        if state is None:
            state = self._build_state(placement)
            self._layer_state[layer_id] = state
        return state

    def _build_state(self, snapshot: PlacementSnapshot) -> _LayerState:
        if snapshot.ep_size != self.num_gpus:
            raise ValueError(
                f"Placement ep_size={snapshot.ep_size} does not match the router's "
                f"num_gpus={self.num_gpus}."
            )
        candidates = rank_invariant_candidates(snapshot)
        metro = build_metro_tables(candidates, snapshot.num_physical_experts, snapshot.ep_size)
        tables = build_metro_repair_tables(metro)
        return _LayerState(tables=tables, has_redundancy=int(metro.num_replicated.item()) > 0)


def _route_host(
    ids: Tensor,
    counts: Tensor,
    tables: MetroRepairTables,
    gate: Tensor | None,
    moves: int,
    early_exit: bool,
) -> Tensor:
    metro = tables.metro
    _, owner = metro_repair_assign_reference(
        counts.detach().cpu() > 0, tables, moves=moves, early_exit=early_exit
    )
    by_rank = metro.physical_by_rank.cpu().long()
    default = metro.default_physical.cpu().long()
    logical = torch.arange(metro.num_logical)
    owner = owner.long()
    chosen = torch.where(owner >= 0, by_rank[logical, owner.clamp(min=0)], default)
    host_ids = ids.detach().cpu().long()
    valid = (host_ids >= 0) & (host_ids < metro.num_logical)
    if gate is not None and host_ids.dim() > 1:
        rows = torch.arange(host_ids.shape[0]).view(-1, *([1] * (host_ids.dim() - 1)))
        valid &= rows < int(gate.reshape(-1)[0])
    physical = torch.where(valid, chosen[host_ids.clamp(0, metro.num_logical - 1)], -1)
    return physical.to(ids.device)


def _same_layout(left: MetroRepairTables, right: MetroRepairTables) -> bool:
    return all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        for a, b in zip(left.tensors(), right.tensors())
    )


def _build_router(context: RouterBuildContext) -> MetroRepairL2Router:
    return MetroRepairL2Router(num_gpus=context.ep_size)


ROUTING_POLICY = RoutingPolicySpec(
    "metro_repair",
    MetroRepairL2Router.phase,
    logical_count="global",
    requires_placement_state=True,
    # Prefill and mixed steps fall back to the framework's static replica map.
    requires_rank_dispatch_map=True,
    requires_deepep=True,
    requires_redundancy=True,
    build_router=_build_router,
)
