# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""METRO: decode replica routing that balances active experts per rank.

In memory-bound decode a rank's routed GEMM cost tracks how many distinct
expert weights it reads, not how many tokens it serves. METRO (arXiv:2512.09277)
uses the EP-wide active set of each layer to send every active replicated
expert to exactly one rank, choosing ranks greedily so the per-rank count of
active experts stays even. Every rank derives the same choice from the same
all-reduced count and the same tables, so no further coordination is needed.
See :mod:`evenmoe.kernels.ops.metro` for the algorithm.

The tables come from the physical-to-logical map, which is the same on every
EP rank. A framework may narrow ``logical_to_physical_candidates`` to the
copies on its own rank, and tables built from that would differ per rank.

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
    MAX_RANKS,
    metro_mask_words,
    metro_route,
    metro_workspace,
)
from evenmoe.policies.l2.replica import route_static_replicas

_STATIC_STAGES = frozenset({RoutingStage.PREFILL, RoutingStage.MIXED})


@dataclass(frozen=True)
class MetroTables:
    """Per-layer METRO tables derived from one committed placement.

    The tensor layout is described in :mod:`evenmoe.kernels.ops.metro`. The
    capacity depends on the layout alone, so a placement commit rewrites the
    tables in place and a captured CUDA graph stays valid.
    """

    num_logical: int
    num_ranks: int
    capacity: int
    default_physical: Tensor
    single_rank: Tensor
    rep_logical: Tensor
    rep_mask: Tensor
    rep_phys: Tensor
    num_replicated: Tensor

    def operands(self) -> dict[str, Tensor]:
        """The keyword tables :func:`metro_route` takes."""

        return {
            "default_physical": self.default_physical,
            "single_rank": self.single_rank,
            "rep_logical": self.rep_logical,
            "rep_mask": self.rep_mask,
            "rep_phys": self.rep_phys,
            "num_replicated": self.num_replicated,
        }

    def tensors(self) -> tuple[Tensor, ...]:
        return tuple(self.operands().values())


def metro_capacity(num_logical: int, num_physical: int) -> int:
    """Upper bound on logical experts held by two or more ranks."""

    return max(min(num_logical, num_physical - num_logical), 1)


def build_metro_tables(
    physical_to_logical_map: Tensor,
    num_logical: int,
    num_ranks: int,
) -> MetroTables:
    """Derive :class:`MetroTables` from a ``[P]`` physical-to-logical map.

    Physical slots are rank-contiguous: ``rank(p) = p // (P // num_ranks)``.
    Several copies of one logical expert on one rank collapse to the lowest
    physical id, since only one copy per rank can lower that rank's load.
    """

    if not 0 < num_ranks <= MAX_RANKS:
        raise ValueError(f"METRO supports 1..{MAX_RANKS} EP ranks; got {num_ranks}.")
    p2l = physical_to_logical_map.reshape(-1).to(torch.int64)
    num_physical = int(p2l.numel())
    if num_physical % num_ranks != 0:
        raise ValueError(
            f"num_physical_experts={num_physical} is not divisible by num_ranks={num_ranks}."
        )
    if bool(((p2l < 0) | (p2l >= num_logical)).any()):
        raise ValueError("physical_to_logical_map holds an out-of-range logical expert.")
    device = p2l.device
    physical = torch.arange(num_physical, device=device)
    rank_of = torch.div(physical, num_physical // num_ranks, rounding_mode="floor")
    first = torch.full((num_logical * num_ranks,), num_physical, dtype=torch.int64, device=device)
    first.scatter_reduce_(0, p2l * num_ranks + rank_of, physical, reduce="amin")
    physical_by_rank = torch.where(first < num_physical, first, -1).view(num_logical, num_ranks)
    has_copy = physical_by_rank >= 0
    if bool((~has_copy).all(dim=1).any()):
        raise ValueError("Every logical expert must have at least one physical copy.")

    ranks = torch.arange(num_ranks, device=device)
    words = torch.zeros(num_logical, metro_mask_words(num_ranks), dtype=torch.int64, device=device)
    words.scatter_add_(
        1,
        torch.div(ranks, 32, rounding_mode="floor").expand(num_logical, -1),
        has_copy.to(torch.int64) << (ranks % 32),
    )
    # Reinterpret each 32-bit word as int32.
    rank_mask = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    first_rank = has_copy.to(torch.int32).argmax(dim=1)
    default_physical = physical_by_rank.gather(1, first_rank.view(-1, 1)).view(-1)
    num_holding = has_copy.sum(dim=1)
    single_rank = torch.where(num_holding == 1, first_rank, torch.full_like(first_rank, -1))

    replicated = torch.nonzero(num_holding >= 2).flatten()
    capacity = metro_capacity(num_logical, num_physical)
    count = int(replicated.numel())
    if count > capacity:
        raise ValueError(f"{count} replicated experts exceed the capacity {capacity}.")
    rep_logical = torch.zeros(capacity, dtype=torch.int32, device=device)
    rep_mask = torch.zeros(capacity, metro_mask_words(num_ranks), dtype=torch.int32, device=device)
    rep_phys = torch.full((capacity, num_ranks), -1, dtype=torch.int32, device=device)
    rep_logical[:count] = replicated.to(torch.int32)
    rep_mask[:count] = rank_mask[replicated]
    rep_phys[:count] = physical_by_rank[replicated].to(torch.int32)
    return MetroTables(
        num_logical=num_logical,
        num_ranks=num_ranks,
        capacity=capacity,
        default_physical=default_physical.to(torch.int32).contiguous(),
        single_rank=single_rank.to(torch.int32).contiguous(),
        rep_logical=rep_logical,
        rep_mask=rep_mask,
        rep_phys=rep_phys,
        num_replicated=torch.tensor([count], dtype=torch.int32, device=device),
    )


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
                raise ValueError(
                    "MetroL2Router routes prefill, mixed and unreplicated layers with "
                    "PlacementSnapshot.default_physical_for_logical, which is missing."
                )
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
            **state.tables.operands(),
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
        p2l = snapshot.physical_to_logical_map
        p2l = (p2l if isinstance(p2l, Tensor) else torch.as_tensor(p2l)).reshape(-1)
        if p2l.numel() != snapshot.num_physical_experts:
            raise ValueError(
                f"physical_to_logical_map has {p2l.numel()} entries for "
                f"{snapshot.num_physical_experts} physical experts."
            )
        _check_candidates(snapshot, p2l)
        tables = build_metro_tables(p2l, snapshot.num_logical_experts, snapshot.ep_size)
        return _MetroLayerState(
            tables=tables,
            workspace=metro_workspace(
                tables.num_logical, tables.num_ranks, tables.capacity, p2l.device
            ),
            has_redundancy=int(tables.num_replicated.item()) > 0,
        )


def _check_candidates(snapshot: PlacementSnapshot, p2l: Tensor) -> None:
    """Reject a snapshot whose candidates are not copies under ``p2l``.

    The tables ignore the candidates, but a candidate listing a copy of another
    expert means the two maps describe different placements.
    """

    candidates = snapshot.logical_to_physical_candidates
    if not isinstance(candidates, Tensor):
        candidates = torch.as_tensor(candidates)
    candidates = candidates.to(device=p2l.device, dtype=torch.int64)
    candidates = candidates.reshape(candidates.shape[0], -1)
    count = snapshot.logical_to_physical_count
    if isinstance(count, Tensor) and candidates.shape[1] > 1:
        # Columns past a row's live count are padding.
        column = torch.arange(candidates.shape[1], device=candidates.device)
        candidates = torch.where(
            column.view(1, -1) < count.to(candidates.device).view(-1, 1), candidates, -1
        )
    live = candidates >= 0
    owner = torch.arange(candidates.shape[0], device=candidates.device).view(-1, 1)
    held = p2l.to(torch.int64)[candidates.clamp(0, p2l.numel() - 1)]
    if bool((live & ((candidates >= p2l.numel()) | (held != owner))).any()):
        raise ValueError(
            "PlacementSnapshot.logical_to_physical_candidates disagrees with "
            "physical_to_logical_map."
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
