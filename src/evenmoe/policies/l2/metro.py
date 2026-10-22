# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""METRO: decode replica routing that balances active experts per rank.

In memory-bound decode a rank's routed GEMM cost tracks how many distinct
expert weights it reads, not how many tokens it serves. METRO (arXiv:2512.09277)
uses the EP-wide active set of each layer to send every active replicated
expert to exactly one rank, choosing ranks greedily so the per-rank count of
active experts stays even. Every rank derives the same choice from the same
all-reduced count, so no further coordination is needed. See
:mod:`evenmoe.kernels.ops.metro` for the algorithm.

The policy runs in every stage because the count collective must be joined by
every EP rank. Decode, speculative and idle steps use the METRO assignment;
prefill and mixed steps, whose load is token-bound, use the framework's static
replica map.
"""

from __future__ import annotations

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
from evenmoe.kernels.ops.metro import (
    MetroTables,
    build_metro_tables,
    metro_route,
    metro_workspace,
)
from evenmoe.policies.l2.replica import route_static_replicas

_STATIC_STAGES = frozenset({RoutingStage.PREFILL, RoutingStage.MIXED})


@dataclass
class _MetroLayerState:
    tables: MetroTables
    workspace: Tensor
    has_redundancy: bool


class MetroL2Router(TokenRouter):
    """METRO physical-replica router.

    Per-batch inputs arrive via ``RoutingRequest``:

    - ``placement``: required; prepared through :meth:`prepare_layer` when the
      framework commits it.
    - ``global_logical_count``: the EP-summed per-logical token count the
      orchestrator gathers. Not needed when the layer has no replicated expert.
    - ``token_count``: optional one-element tensor; rows at or past it are
      padding and come back as -1.

    Placement updates rewrite the prepared tables in place, so a captured CUDA
    graph stays valid. Physical ids are returned as ``out_dtype`` (int64 by
    default, the dtype DeepEP dispatch consumes); ``None`` keeps the input dtype.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, *, num_gpus: int, out_dtype: torch.dtype | None = torch.int64):
        if num_gpus <= 0:
            raise ValueError("MetroL2Router.num_gpus must be positive.")
        self.num_gpus = num_gpus
        self.out_dtype = out_dtype
        self._layer_state: dict[int, _MetroLayerState] = {}

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Install the tables of a newly committed placement for one layer."""

        state = self._build_state(snapshot)
        current = self._layer_state.get(snapshot.layer_id)
        if current is not None and _same_layout(current.tables, state.tables):
            for dst, src in zip(current.tables.tensors(), state.tables.tensors()):
                dst.copy_(src)
            current.has_redundancy = state.has_redundancy
            return
        self._layer_state[snapshot.layer_id] = state

    def needs_global_logical_count(self, request: RoutingRequest) -> bool:
        """Only a layer with a replicated expert has a choice to make.

        The answer depends on the placement alone, so every EP rank agrees on
        whether to join the count collective.
        """

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
                    "Direct MetroL2Router calls require exactly one request policy; "
                    "the unified orchestrator always passes the selected policy explicitly."
                )
            policy = RoutingPolicyConfig.from_value(request.policies[0])
        placement = request.placement
        if placement is None:
            raise ValueError("MetroL2Router requires RoutingRequest.placement.")
        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("MetroL2Router requires torch.Tensor logical_topk_ids.")
        out_dtype = self.out_dtype or topk_ids.dtype
        state = self._state(request.layer_id, placement)

        if (
            not state.has_redundancy
            or request.stage in _STATIC_STAGES
            or not policy.should_run_for_stage(request.stage)
        ):
            default = placement.default_physical_for_logical
            if not isinstance(default, Tensor):
                default = state.tables.default_physical
            physical = route_static_replicas(topk_ids, default.to(topk_ids.device))
            return RoutingDecision(
                routed_physical_topk_ids=physical.to(out_dtype),
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "metro_assignment": False},
            )

        counts = request.global_logical_count
        if counts is None:
            raise ValueError(
                "MetroL2Router requires RoutingRequest.global_logical_count on a "
                "layer with replicated experts."
            )
        if not isinstance(counts, Tensor):
            raise TypeError("MetroL2Router requires a torch.Tensor global_logical_count.")
        token_count = request.token_count
        physical = metro_route(
            topk_ids,
            counts,
            state.tables,
            num_token_non_padded=token_count if isinstance(token_count, Tensor) else None,
            workspace=state.workspace,
            out_dtype=out_dtype,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name, "metro_assignment": True},
        )

    def _state(self, layer_id: int, placement: PlacementSnapshot) -> _MetroLayerState:
        state = self._layer_state.get(layer_id)
        if state is None:
            state = self._build_state(placement)
            self._layer_state[layer_id] = state
        return state

    def _build_state(self, snapshot: PlacementSnapshot) -> _MetroLayerState:
        if snapshot.ep_size != self.num_gpus:
            raise ValueError(
                f"Placement ep_size={snapshot.ep_size} does not match the router's "
                f"num_gpus={self.num_gpus}."
            )
        candidates = snapshot.logical_to_physical_candidates
        if not isinstance(candidates, Tensor):
            candidates = torch.as_tensor(candidates)
        count = snapshot.logical_to_physical_count
        if isinstance(count, Tensor) and candidates.shape[1] > 1:
            # Columns past a row's live count are padding.
            column = torch.arange(candidates.shape[1], device=candidates.device)
            candidates = torch.where(
                column.view(1, -1) < count.to(candidates.device).view(-1, 1), candidates, -1
            )
        tables = build_metro_tables(candidates, snapshot.num_physical_experts, snapshot.ep_size)
        return _MetroLayerState(
            tables=tables,
            workspace=metro_workspace(tables),
            has_redundancy=int(tables.num_replicated.item()) > 0,
        )


def _same_layout(left: MetroTables, right: MetroTables) -> bool:
    return all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        for a, b in zip(left.tensors(), right.tensors())
    )


def _build_router(context: RouterBuildContext) -> MetroL2Router:
    return MetroL2Router(num_gpus=context.ep_size)


ROUTING_POLICY = RoutingPolicySpec(
    "metro",
    MetroL2Router.phase,
    logical_count="global",
    requires_placement_state=True,
    # Prefill and mixed steps fall back to the framework's static replica map.
    requires_rank_dispatch_map=True,
    requires_deepep=True,
    # One scratch workspace per layer.
    supports_concurrent_microbatches=False,
    requires_redundancy=True,
    build_router=_build_router,
)
