# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LPLB L2 router — LP-based redirection among physical replicas.

Framework-neutral: operates on :class:`PlacementSnapshot` inputs and
routes the four kernels through an injected :class:`LPLBKernels`
implementation supplied by the framework adapter.

Algorithm overview
------------------

The router solves a small LP per MoE layer per forward pass. The LP picks,
for every replicated logical expert, how to split that expert's *global*
token count across its physical replicas so the **peak per-GPU routed load**
is minimized:

    minimize  M + big_m_penalty * mu
    s.t.       C  t            = global_count_replicated      (token conservation)
              B2 t + I v − 1 M = − B1 t1                       (per-GPU peak with single-copy load fixed)
              t, v, M >= 0,  mu Big-M slack

The LP variables are: ``t`` per replicated physical slot, ``v`` per-GPU
slack, ``M`` peak, and one Big-M auxiliary. Single-copy logicals contribute
a fixed per-GPU baseline (``B1 t1``) that the per-batch ``prep_lp_inputs``
kernel folds into the RHS.

Per-layer state
---------------

The LP constraint matrices (``B1``, ``A_base``, ``c_vec``, ``log2phy``, …)
depend only on the placement — not on per-batch data. The framework calls
``on_placement_committed`` after initial weight loading and after each committed
placement update. EvenMoE then updates the affected layer's derived state; no
placement version or tensor-identity protocol is required.

Collective boundary
-------------------

The LPLB policy consumes ``RoutingRequest.global_logical_count``. The EvenMoE
orchestrator creates that tensor and requests one sum through the injected
framework collective adapter before invoking the policy. The policy itself
contains no framework or process-group calls. Placement-derived matrices are
prepared per committed layer placement. Mutable outputs (``A_full``, ``b``,
``t1``, ``x``, ``log2phy_prob``) are isolated per CUDA graph or eager stream
so overlapping execution cannot overwrite another batch's LP state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Hashable
from dataclasses import dataclass, replace

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
)
from evenmoe.kernels.lplb import LPLBKernels

logger = logging.getLogger(__name__)

# How coarsely the LP's row count is rounded. Every distinct value NVRTC-
# compiles three kernels, and every row of padding is paid by every solve, so
# this is the dial between compile count and solve time. 8 gives at most
# num_redundant/8 shapes and at most 7 rows of slack.
_SHAPE_BUCKET = 8


def _ceil_to(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def bucketed_shape_set(num_logical: int, num_phy: int, num_gpus: int, max_copies: int):
    """Every LP shape this deployment can produce, in closed form.

    Bucketing makes the set small, but small is not the same as absent: the
    shapes it does contain are still NVRTC-compiled on first use, and first use
    is inside whichever rearrangement first produces them. Since the set does
    not depend on the observed load -- only on how many redundant slots exist
    and how coarsely the row count is rounded -- it can be enumerated and
    compiled before serving starts instead.

    Returns dicts shaped for :func:`cuda_solver.warmup_shape`.
    """
    num_redundant = num_phy - num_logical
    if num_redundant <= 0:
        return []
    targets = sorted(
        {min(_ceil_to(k, _SHAPE_BUCKET), num_redundant)
         for k in range(1, num_redundant + 1)}
    )
    shapes = []
    for num_red_log in targets:
        num_red_phy = num_red_log + num_redundant
        shapes.append({
            "nc": num_red_log + num_gpus,
            "nv": num_red_phy + num_gpus + 2,
            "num_single": num_logical - num_red_log,
            "num_red_log": num_red_log,
            "num_gpus": num_gpus,
            "num_logical": num_logical,
            "max_copies": max_copies,
            "num_red_phy": num_red_phy,
        })
    return shapes

# Sentinel preserved through the no-redundancy fallback for the
# "-1 padded slot" convention frameworks use for invalid TopK entries.
_INVALID_ID = -1


@dataclass
class _LPLBLayerState:
    """Pre-built LP constraint matrices + reusable workspace for one layer.

    Constructed lazily by ``LPLBL2Router._get_or_build_layer_state`` on the
    first ``route_tokens`` call for a layer. The constant matrices
    (everything except the workspace tail) are
    derived from ``PlacementSnapshot``; the workspace tensors are
    pre-allocated once and overwritten in place by every routing call.
    """

    # Geometry
    num_logical: int
    num_phy: int
    num_gpus: int
    has_redundancy: bool

    # Logical / physical partition into single-copy and replicated.
    log_single: Tensor  # [num_single] int64
    phy_single: Tensor  # [num_single] int64
    log_replicated: Tensor  # [num_red_log] int64
    phy_replicated: Tensor  # [num_red_phy] int64

    # Constant LP inputs (built once from the placement snapshot).
    B1: Tensor  # [num_gpus, num_single] float32
    A_base: Tensor  # [NC, NV-1] float32 — A_full first NV-1 cols
    A_base_row_sum: Tensor  # [NC] float32 (precomputed for prep kernel)
    c_vec: Tensor  # [NV] float32 — objective coefficients
    log2phy: Tensor  # [num_logical, max_copies] int64 (-1 padded)
    log2phy_map_for_dispatch: Tensor  # [num_logical, max_copies] int32 view

    # Workspace template (cloned per CUDA graph or eager stream).
    A_full: Tensor  # [NC, NV] float32 — first NV-1 cols hold A_base
    b: Tensor  # [NC] float32
    t1: Tensor  # [num_single] float32
    x: Tensor  # [NV] float32
    log2phy_prob: Tensor  # [num_logical, max_copies] float32


_PLACEMENT_STATE_TENSORS = (
    "log_single",
    "phy_single",
    "log_replicated",
    "phy_replicated",
    "B1",
    "A_base",
    "A_base_row_sum",
    "c_vec",
    "log2phy",
    "log2phy_map_for_dispatch",
    "A_full",
)


def _same_state_layout(left: _LPLBLayerState, right: _LPLBLayerState) -> bool:
    if (
        left.num_logical != right.num_logical
        or left.num_phy != right.num_phy
        or left.num_gpus != right.num_gpus
        or left.has_redundancy != right.has_redundancy
    ):
        return False
    for name in _PLACEMENT_STATE_TENSORS:
        left_tensor = getattr(left, name)
        right_tensor = getattr(right, name)
        if (
            left_tensor.shape != right_tensor.shape
            or left_tensor.dtype != right_tensor.dtype
            or left_tensor.device != right_tensor.device
        ):
            return False
    return True


def _copy_placement_state_(destination: _LPLBLayerState, source: _LPLBLayerState) -> None:
    for name in _PLACEMENT_STATE_TENSORS:
        getattr(destination, name).copy_(getattr(source, name))


def _build_layer_state(
    snapshot: PlacementSnapshot,
    num_gpus: int,
    big_m_penalty: float,
) -> _LPLBLayerState:
    """Build LP matrices + workspace for one MoE layer.

    Consumes a framework-neutral :class:`PlacementSnapshot`. Validates the
    contiguous-per-rank ownership layout (``num_phy % num_gpus == 0``).

    Degenerate case: when no logical expert has more than one physical copy
    (``has_redundancy=False``), most of the LP machinery is skipped. The
    returned state still allocates a ``log2phy_prob`` buffer with the same
    shape as the placement so callers downstream of a future has_redundancy
    transition do not need to special-case the shape.
    """

    log2phy_raw = snapshot.logical_to_physical_candidates
    log_counts_raw = snapshot.logical_to_physical_count

    if not isinstance(log2phy_raw, Tensor):
        raise TypeError(
            "LPLBL2Router requires PlacementSnapshot.logical_to_physical_candidates "
            "to be a torch.Tensor."
        )
    if not isinstance(log_counts_raw, Tensor):
        raise TypeError(
            "LPLBL2Router requires PlacementSnapshot.logical_to_physical_count "
            "to be a torch.Tensor."
        )

    num_logical = snapshot.num_logical_experts
    num_phy = snapshot.num_physical_experts
    if num_phy % num_gpus != 0:
        raise ValueError(
            "LPLBL2Router requires PlacementSnapshot.num_physical_experts "
            f"({num_phy}) divisible by num_gpus ({num_gpus}); the LP "
            "constraint matrices assume per-rank contiguous ownership."
        )
    num_phy_per_gpu = num_phy // num_gpus

    device = log2phy_raw.device
    log2phy = log2phy_raw.to(torch.int64).contiguous()
    log_counts = log_counts_raw.to(torch.int64)
    max_copies = log2phy.shape[1]

    # Reconstruct phy2log from log2phy + log_counts.
    # For every (logical i, slot j < log_counts[i]) pair, log2phy[i, j] is a
    # valid physical expert id and we want phy2log[log2phy[i, j]] = i.
    j_idx = torch.arange(max_copies, device=device).unsqueeze(0).expand(num_logical, -1)
    valid_mask = j_idx < log_counts.unsqueeze(1)
    valid_phy = log2phy[valid_mask]
    valid_log = torch.arange(num_logical, device=device).unsqueeze(1).expand_as(j_idx)[valid_mask]

    # Defensive validation: every valid_phy entry must be a real physical
    # slot index, otherwise the scatter below silently OOBs and the next
    # ``log_counts[phy2log]`` lookup triggers an async device-side assert
    # that surfaces minutes later as an opaque CUDA error. Some placements
    # can leave physical slots that no logical claims (see below); the
    # range check here catches genuinely malformed inputs first.
    if valid_phy.numel() > 0:
        vp_min = int(valid_phy.min().item())
        vp_max = int(valid_phy.max().item())
        if vp_min < 0 or vp_max >= num_phy:
            raise RuntimeError(
                f"LPLB _build_layer_state: valid_phy out of [0, {num_phy}) "
                f"(observed min={vp_min}, max={vp_max}). "
                f"log2phy.shape={tuple(log2phy.shape)}, "
                f"log_counts shape={tuple(log_counts.shape)}, "
                f"log_counts min/max="
                f"{int(log_counts.min().item())}/"
                f"{int(log_counts.max().item())}."
            )

    # Initialize phy2log to a sentinel (num_logical) so we can distinguish
    # physical slots that no logical claims through
    # logical_to_physical_candidates. Some placements reserve
    # ``num_physical_experts`` slots that include redundant capacity but
    # only populate the first ``num_logical`` of them — the rest stay
    # unclaimed. That is a legitimate state: those slots simply carry no
    # expert, so the LP must ignore them (they're not in any logical's
    # replica set, so they contribute nothing to peak GPU load). The
    # masked indexing below skips them safely.
    _PHY_UNCOVERED = num_logical  # sentinel: out-of-range for log_counts
    phy2log = torch.full((num_phy,), _PHY_UNCOVERED, dtype=torch.int64, device=device)
    phy2log[valid_phy] = valid_log
    is_covered = phy2log != _PHY_UNCOVERED

    # Partition logicals into single-copy vs replicated.
    safe_phy2log = phy2log.clamp(max=num_logical - 1)
    is_replicated = log_counts > 1
    has_redundancy = bool(is_replicated.any())

    # Pad that partition so the LP has the same shape on every layer.
    #
    # Without this, ``num_red_log`` is however many logical experts a
    # load-aware placement happened to give a second copy, which differs
    # layer to layer -- a hot expert may take three copies while another
    # takes none. Every distinct value is a distinct (nc, nv), and each of
    # those NVRTC-compiles its own IPM, prep and post kernel: ~16 s apiece,
    # inside the rearrangement that triggered the rebuild. Measured on
    # DeepSeek-V3.2 at 32 redundant experts, one rearrangement compiled 17
    # shapes and took 266 s, past vLLM's 300 s execute-model timeout; at 16
    # it took 8 s, past DeepEP's shorter CPU-recv timeout on the peer ranks
    # still waiting in dispatch.
    #
    # The padding promotes single-copy experts into the replicated set until
    # it holds ``num_redundant`` of them. That is an identity, not an
    # approximation: a promoted expert has exactly one replica, so its top
    # row reads ``x_e = load_e`` and pins its own flow. Its load simply
    # arrives at its GPU through ``B2 @ x`` instead of through the
    # ``B1 @ t1`` base term -- same GPU, same amount, same peak. The row
    # holds a variable no other row touches, so the constraint matrix stays
    # full rank.
    num_redundant = num_phy - num_logical
    if has_redundancy and num_redundant > 0:
        # Round up to a bucket rather than all the way to num_redundant.
        #
        # Padding every layer to the maximum does make the shape constant, but
        # the solve is compute-bound in nc: measured on H20 at 32 redundant
        # experts, a layer that needed nc=16 takes 75 us and one that needs
        # nc=40 takes 301 us. Flattening everything to 40 costs +83% on the
        # average solve, or +8.4 ms per forward across 61 layers -- a permanent
        # per-batch tax traded for a one-off compile, on a policy that already
        # loses throughput to its solve.
        #
        # Bucketing keeps both: at most num_redundant/_SHAPE_BUCKET distinct
        # shapes to compile, and at most _SHAPE_BUCKET-1 rows of padding on any
        # layer instead of up to num_redundant - num_red_log.
        target = min(
            _ceil_to(int(is_replicated.sum()), _SHAPE_BUCKET),
            num_redundant,
        )
        shortfall = target - int(is_replicated.sum())
        if shortfall > 0:
            singles = torch.nonzero(~is_replicated).flatten()
            # Lowest ids first: every rank builds this from the same
            # committed placement, so the rule only has to be deterministic.
            is_replicated[singles[:shortfall]] = True

    log_single = torch.nonzero(~is_replicated).flatten().to(torch.int64)
    phy_single = log2phy[log_single, 0].to(torch.int64).contiguous()
    log_replicated = torch.nonzero(is_replicated).flatten().to(torch.int64)
    # phy_replicated: physical slots that (a) are claimed by some logical
    # AND (b) that logical is in the replicated set. Uncovered slots are
    # masked out before the gather so the indexing stays in bounds even when
    # ``_PHY_UNCOVERED == num_logical`` would otherwise OOB.
    phy_replicated = (
        torch.nonzero(is_covered & is_replicated[safe_phy2log])
        .flatten()
        .to(torch.int64)
        .contiguous()
    )

    num_single = int(log_single.numel())
    num_red_log = int(log_replicated.numel())
    num_red_phy = int(phy_replicated.numel())

    # Promotion fixes num_red_phy at 2 * num_redundant only when every
    # physical slot is claimed by some logical. A placement that leaves
    # redundant slots unclaimed cannot reach that target by promoting
    # logicals, so the shape varies again and the kernels recompile. Correct,
    # just slow -- and worth saying out loud rather than rediscovering it as
    # another multi-minute rearrangement.
    # Bucketing fixes num_red_phy only because it is determined by num_red_log:
    # every replicated logical brings its base copy, and the redundant slots
    # bring one extra each. A placement that leaves redundant slots unclaimed
    # breaks that identity, so the shape varies again and the kernels recompile
    # for that layer alone. Correct, just slow -- and worth saying out loud
    # rather than rediscovering it as another multi-minute rearrangement.
    expected_red_phy = num_red_log + num_redundant
    if has_redundancy and num_redundant > 0 and num_red_phy != expected_red_phy:
        logger.warning(
            "LPLB layer %s: %d unclaimed redundant slots leave the LP shape "
            "at num_red_phy=%d instead of %d, so its kernels compile for this "
            "layer alone.",
            snapshot.layer_id,
            expected_red_phy - num_red_phy,
            num_red_phy,
            expected_red_phy,
        )

    log2phy_map_for_dispatch = log2phy.to(torch.int32).contiguous()

    if not has_redundancy:
        # No LP needed — pre-allocate the minimum state so identity dispatch
        # has the buffers it needs.
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        return _LPLBLayerState(
            num_logical=num_logical,
            num_phy=num_phy,
            num_gpus=num_gpus,
            has_redundancy=False,
            log_single=log_single,
            phy_single=phy_single,
            log_replicated=log_replicated,
            phy_replicated=phy_replicated,
            B1=torch.zeros((num_gpus, num_single), dtype=torch.float32, device=device),
            A_base=empty_f32.reshape(0, 0),
            A_base_row_sum=empty_f32,
            c_vec=empty_f32,
            log2phy=log2phy,
            log2phy_map_for_dispatch=log2phy_map_for_dispatch,
            A_full=empty_f32.reshape(0, 0),
            b=empty_f32,
            t1=torch.empty(num_single, dtype=torch.float32, device=device),
            x=empty_f32,
            log2phy_prob=torch.empty(log2phy.shape, dtype=torch.float32, device=device),
        )

    # GPU ownership matrix B_full[i, j] = 1 iff GPU i owns physical slot j.
    B_full = torch.zeros((num_gpus, num_phy), dtype=torch.float32, device=device)
    for i in range(num_gpus):
        B_full[i, i * num_phy_per_gpu : (i + 1) * num_phy_per_gpu] = 1
    B1 = B_full[:, phy_single].contiguous()
    B2 = B_full[:, phy_replicated]

    # Copy-to-logical mapping C[i, j] = 1 iff phy_replicated[j] belongs to
    # log_replicated[i].
    C = torch.zeros((num_red_log, num_red_phy), dtype=torch.float32, device=device)
    phy2log_rep = phy2log[phy_replicated]
    for i in range(num_red_log):
        C[i, phy2log_rep == log_replicated[i]] = 1.0

    # A_base = [[C 0 0], [B2 I -1]] (the Big-M aux column is appended live
    # via A_full[:, -1] from the prep kernel each batch).
    zeros_top_g = torch.zeros((num_red_log, num_gpus), dtype=torch.float32, device=device)
    zeros_top_1 = torch.zeros((num_red_log, 1), dtype=torch.float32, device=device)
    I_g = torch.eye(num_gpus, dtype=torch.float32, device=device)
    neg_ones = torch.full((num_gpus, 1), -1.0, dtype=torch.float32, device=device)
    A_top = torch.hstack([C, zeros_top_g, zeros_top_1])
    A_bottom = torch.hstack([B2, I_g, neg_ones])
    A_base = torch.vstack([A_top, A_bottom]).contiguous()

    nc, nv_minus_1 = A_base.shape
    nv = nv_minus_1 + 1  # +1 for the Big-M auxiliary column

    # Objective: minimize M (second-to-last variable), penalize the Big-M aux.
    c_vec = torch.zeros(nv, dtype=torch.float32, device=device)
    c_vec[-2] = 1.0
    c_vec[-1] = big_m_penalty

    A_base_row_sum = A_base.sum(dim=1).contiguous()

    # Workspace tensors (overwritten per batch by the kernels).
    A_full = torch.empty(nc, nv, dtype=torch.float32, device=device)
    A_full[:, : nv - 1].copy_(A_base)
    b = torch.empty(nc, dtype=torch.float32, device=device)
    t1 = torch.empty(num_single, dtype=torch.float32, device=device)
    x = torch.empty(nv, dtype=torch.float32, device=device)
    log2phy_prob = torch.empty(log2phy.shape, dtype=torch.float32, device=device)

    return _LPLBLayerState(
        num_logical=num_logical,
        num_phy=num_phy,
        num_gpus=num_gpus,
        has_redundancy=True,
        log_single=log_single,
        phy_single=phy_single,
        log_replicated=log_replicated,
        phy_replicated=phy_replicated,
        B1=B1,
        A_base=A_base,
        A_base_row_sum=A_base_row_sum,
        c_vec=c_vec,
        log2phy=log2phy,
        log2phy_map_for_dispatch=log2phy_map_for_dispatch,
        A_full=A_full,
        b=b,
        t1=t1,
        x=x,
        log2phy_prob=log2phy_prob,
    )


def _identity_dispatch(topk_ids: Tensor, snapshot: PlacementSnapshot) -> Tensor:
    """Map logical top-k ids to physical ids without LP redistribution.

    Used when the layer has no replicated logicals (``has_redundancy=False``)
    so the LP would degenerate. Falls back, in order:

    1. ``snapshot.default_physical_for_logical[logical_id]`` if provided
       (this is the framework's precomputed rank-dispatch baseline).
    2. ``snapshot.logical_to_physical_candidates[logical_id, 0]`` otherwise.

    Tokens with ``logical_id < 0`` (padding / invalid slots) are preserved
    as-is so framework TopK padding conventions are respected.
    """
    log2phy = snapshot.logical_to_physical_candidates
    default_map = snapshot.default_physical_for_logical
    valid = topk_ids >= 0
    safe_ids = topk_ids.to(torch.int64).clamp(min=0)

    if default_map is not None:
        physical = default_map.to(torch.int64)[safe_ids]
    else:
        physical = log2phy.to(torch.int64)[safe_ids, 0]

    result = torch.where(valid, physical, topk_ids.to(torch.int64))
    return result.to(topk_ids.dtype)


def _rank_ids_for_physical(
    physical_ids: Tensor,
    snapshot: PlacementSnapshot,
) -> Tensor:
    safe_ids = physical_ids.clamp(min=0).to(torch.int64)
    if snapshot.physical_to_rank_map is not None:
        rank_map = snapshot.physical_to_rank_map
        if not isinstance(rank_map, Tensor):
            rank_map = torch.as_tensor(rank_map, device=physical_ids.device)
        return rank_map[safe_ids].to(torch.int64)
    return safe_ids // snapshot.num_local_physical_experts


def _planned_rank_load_from_mapping(
    global_counts: Tensor,
    physical_for_logical: Tensor,
    snapshot: PlacementSnapshot,
) -> Tensor:
    rank_ids = _rank_ids_for_physical(physical_for_logical, snapshot)
    valid = (physical_for_logical >= 0) & (rank_ids >= 0) & (rank_ids < snapshot.ep_size)
    planned = torch.zeros(
        snapshot.ep_size,
        dtype=torch.float32,
        device=global_counts.device,
    )
    planned.scatter_add_(
        0,
        torch.where(valid, rank_ids, 0),
        torch.where(valid, global_counts.to(torch.float32), 0),
    )
    return planned.round().to(torch.int64)


def _planned_rank_load_from_probability(
    global_counts: Tensor,
    probabilities: Tensor,
    physical_candidates: Tensor,
    snapshot: PlacementSnapshot,
) -> Tensor:
    """Derive Waterfill's global planned load without a second collective."""

    rank_ids = _rank_ids_for_physical(physical_candidates, snapshot)
    valid = (physical_candidates >= 0) & (rank_ids >= 0) & (rank_ids < snapshot.ep_size)
    contributions = global_counts.to(torch.float32)[:, None] * probabilities.to(torch.float32)
    planned = torch.zeros(
        snapshot.ep_size,
        dtype=torch.float32,
        device=global_counts.device,
    )
    planned.scatter_add_(
        0,
        torch.where(valid, rank_ids, 0).reshape(-1),
        torch.where(valid, contributions, 0).reshape(-1),
    )
    return planned.round().to(torch.int64)


class LPLBL2Router(TokenRouter):
    """LP-based per-batch routing among physical replicas.

    Per-batch state arrives via ``RoutingRequest``:

    - ``request.placement``: required. The ``logical_to_physical_candidates``,
      ``logical_to_physical_count``, and ``num_physical_experts`` fields drive
      the LP constraint construction. The framework prepares every committed
      placement before forwarding requests that use it.
    - ``request.global_logical_count``: required. Per-logical-expert token
      count summed across the EP group by the EvenMoE orchestrator through its
      injected collective adapter.
    - ``request.logical_topk_ids``: required. ``[num_tokens, top_k]`` tensor
      of logical expert ids from the router.

    Stage gating is driven by the policy config selected by the unified
    orchestrator. The request carries an ordered policy pipeline rather than
    embedding a second, competing single-policy field.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(
        self,
        *,
        kernels: LPLBKernels,
        num_gpus: int,
        big_m_penalty: float = 1000.0,
        ipm_iters: int = 5,
    ):
        if num_gpus <= 0:
            raise ValueError("LPLBL2Router.num_gpus must be positive.")
        if big_m_penalty <= 0:
            raise ValueError("LPLBL2Router.big_m_penalty must be positive.")
        if ipm_iters <= 0:
            raise ValueError("LPLBL2Router.ipm_iters must be positive.")

        self._kernels = kernels
        self.num_gpus = num_gpus
        self.big_m_penalty = big_m_penalty
        self.ipm_iters = ipm_iters
        # Placement-derived state is authoritative only for the current
        # framework-committed snapshot of each layer.
        self._layer_state: dict[int, _LPLBLayerState] = {}
        # Mutable LP workspaces cannot be shared by overlapping CUDA graphs.
        self._execution_state: dict[tuple[int, Hashable], _LPLBLayerState] = {}
        # The shape set is compiled once per process, on the first layer seen.
        self._warmed_shape_set = False

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Install state for the framework's newly committed placement.

        Shape-compatible updates preserve tensor addresses and copy new
        placement-derived values in place. Shape-changing updates replace the
        prepared state used by subsequent uncaptured execution. Keeping a
        previously captured graph valid across such a shape change requires a
        future fixed-shape LPLB state representation.
        """

        layer_id = snapshot.layer_id
        state = self._build_and_warmup_layer_state(snapshot)
        current = self._layer_state.get(layer_id)
        if current is not None and _same_state_layout(current, state):
            _copy_placement_state_(current, state)
            for key, execution_state in self._execution_state.items():
                if key[0] == layer_id:
                    _copy_placement_state_(execution_state, state)
            return

        self._layer_state[layer_id] = state
        self._execution_state = {
            key: value for key, value in self._execution_state.items() if key[0] != layer_id
        }

    def needs_global_logical_count(self, request: RoutingRequest) -> bool:
        """Report whether this layer's LP actually consumes the count.

        A layer whose logical experts each hold exactly one replica has no
        traffic split to decide: :meth:`route_tokens` returns identity dispatch
        and never reads the count. Gathering it anyway costs one EP collective
        per layer per forward for a result that is discarded -- the whole of the
        overhead seen on zero-redundancy configurations.

        The answer is read off the cached per-layer state, so asking costs
        nothing after the first call for a given placement.
        """

        if request.placement is None:
            return False
        return self._get_or_build_layer_state(
            request.layer_id,
            request.placement,
        ).has_redundancy

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig | None = None,
    ) -> RoutingDecision:
        if policy is None:
            if len(request.policies) != 1:
                raise ValueError(
                    "Direct LPLBL2Router calls require exactly one request policy; "
                    "the unified orchestrator always passes the selected policy explicitly."
                )
            policy = RoutingPolicyConfig.from_value(request.policies[0])
        if not policy.should_run_for_stage(request.stage):
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "skipped": "stage_gated"},
            )

        if request.placement is None:
            raise ValueError(
                "LPLBL2Router requires RoutingRequest.placement to be a "
                "PlacementSnapshot (logical->physical translation is non-optional)."
            )
        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("LPLBL2Router requires torch.Tensor logical_topk_ids.")

        base_state = self._get_or_build_layer_state(
            request.layer_id,
            request.placement,
        )

        # No-redundancy fast path: LP is degenerate; route to the canonical
        # physical slot for each logical. Still passes through empty batches
        # untouched (caller upstream is expected to short-circuit the path
        # when num_tokens == 0, but be tolerant here).
        #
        # This runs before the global_logical_count requirement below, because
        # the count is what this path exists to avoid paying for: the
        # orchestrator asks needs_global_logical_count() first and skips the EP
        # collective when the answer is False, so arriving here without a count
        # is the expected case, not an error.
        if not base_state.has_redundancy:
            metadata: dict[str, object] = {
                "policy": policy.name,
                "lp_skipped": "no_redundancy",
            }
            # The planned load can only be stated when the count was gathered.
            # Without it the orchestrator falls back to counting the physical
            # ids returned here, which for identity dispatch describes the very
            # same placement.
            if request.global_logical_count is not None:
                default_map = request.placement.default_physical_for_logical
                if default_map is None:
                    default_map = request.placement.logical_to_physical_candidates[:, 0]
                metadata["planned_rank_load"] = _planned_rank_load_from_mapping(
                    request.global_logical_count,
                    default_map,
                    request.placement,
                )
            return RoutingDecision(
                routed_physical_topk_ids=_identity_dispatch(topk_ids, request.placement),
                topk_weights=request.topk_weights,
                metadata=metadata,
            )

        if request.global_logical_count is None:
            raise ValueError(
                "LPLBL2Router requires RoutingRequest.global_logical_count "
                "(the EvenMoE orchestrator must count + all-reduce per-logical-expert "
                "token counts before invoking the policy)."
            )

        # Empty-token fast path: the orchestrator has already participated in
        # the EP all-reduce on this rank; avoid invoking the kernels on a
        # degenerate local input.
        if topk_ids.shape[0] == 0:
            return RoutingDecision(
                routed_physical_topk_ids=topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "lp_skipped": "empty_batch"},
            )

        state = self._get_execution_state(
            request.layer_id,
            topk_ids,
            base_state,
        )

        # Cast global_logical_count to the float32 the prep kernel expects.
        global_counts = request.global_logical_count
        if not isinstance(global_counts, Tensor):
            raise TypeError("LPLBL2Router requires torch.Tensor global_logical_count.")
        if global_counts.dtype != torch.float32:
            global_counts = global_counts.to(torch.float32)

        # Four-stage pipeline. Mutable outputs belong to this execution state.
        self._kernels.prep_lp_inputs(
            state.A_full,
            state.b,
            state.t1,
            global_counts,
            state.log_single,
            state.log_replicated,
            state.B1,
            state.A_base_row_sum,
        )
        self._kernels.solve_ipm(
            state.A_full,
            state.b,
            state.c_vec,
            num_iters=self.ipm_iters,
            result=state.x,
        )
        self._kernels.extract_log2phy_prob(
            state.log2phy_prob,
            state.x,
            state.t1,
            state.phy_single,
            state.phy_replicated,
            state.log2phy,
        )
        metadata = {
            "policy": policy.name,
            "planned_rank_load": _planned_rank_load_from_probability(
                global_counts,
                state.log2phy_prob,
                state.log2phy_map_for_dispatch,
                request.placement,
            ),
            # The LP's actual solution, exposed unconditionally rather than
            # only when a caller asks to apply it itself: MLB_DUMP_LP reads
            # this to compute achievable headroom offline, and that diagnostic
            # should not depend on which dispatch mechanism the run happens to
            # use.
            #
            # Unlike planned_rank_load above, this is the live per-layer buffer
            # itself, not a value computed fresh from it -- extract_log2phy_prob
            # overwrites it in place on this layer's next forward, the same way
            # every other CUDA-graph-safe state tensor in this module does. A
            # consumer that needs the value later, not synchronously, must copy
            # it (``.clone()`` / ``.cpu()``) before returning.
            "lp_probability": state.log2phy_prob,
        }

        # Load recording is the framework's job, not this kernel's: it always
        # runs the load view it already owns through its own pass afterward
        # regardless of what this call does, so handing it one here would
        # only get every slot counted twice.
        routed_physical = self._kernels.dispatch_probability(
            topk_ids,
            state.log2phy_prob,
            state.log2phy_map_for_dispatch,
        )

        return RoutingDecision(
            routed_physical_topk_ids=routed_physical,
            topk_weights=request.topk_weights,
            metadata=metadata,
        )

    # -- Internal helpers --------------------------------------------------

    def _get_or_build_layer_state(
        self, layer_id: int, snapshot: PlacementSnapshot
    ) -> _LPLBLayerState:
        """Return prepared state, with lazy construction for direct callers."""

        cached = self._layer_state.get(layer_id)
        if cached is not None:
            return cached

        state = self._build_and_warmup_layer_state(snapshot)
        self._layer_state[layer_id] = state
        return state

    def _warmup_shape_set(self, snapshot: PlacementSnapshot) -> None:
        """Compile the whole shape set once, on the first layer prepared.

        The framework announces its initial placement after weight loading and
        before serving, so this lands off the critical path. Leaving it lazy
        instead puts the compiles inside a rearrangement, where a stall is
        measured against watchdogs: 266 s of compiling once tripped vLLM's
        300 s execute-model timeout and killed every engine core.
        """
        if self._warmed_shape_set:
            return
        self._warmed_shape_set = True
        shapes = bucketed_shape_set(
            snapshot.num_logical_experts,
            snapshot.num_physical_experts,
            self.num_gpus,
            int(snapshot.logical_to_physical_candidates.shape[1]),
        )
        if not shapes:
            return
        # Optional on purpose: precompiling is an optimisation, and a backend
        # that cannot do it should fall back to compiling lazily rather than
        # fail to run at all.
        warm = getattr(self._kernels, "warmup_shape", None)
        if warm is None:
            return
        device = snapshot.logical_to_physical_candidates.device
        for shape in shapes:
            warm(num_iters=self.ipm_iters, device=device, **shape)
        logger.info(
            "LPLB: precompiled %d LP shapes (nc %s) before serving",
            len(shapes),
            ",".join(str(s["nc"]) for s in shapes),
        )

    def _build_and_warmup_layer_state(self, snapshot: PlacementSnapshot) -> _LPLBLayerState:
        self._warmup_shape_set(snapshot)
        state = _build_layer_state(snapshot, self.num_gpus, self.big_m_penalty)

        # Warmup is per-shape; only meaningful when LP is actually used.
        if state.has_redundancy:
            nc, nv = state.A_full.shape
            self._kernels.warmup(
                nc=nc,
                nv=nv,
                num_iters=self.ipm_iters,
                device=state.A_full.device,
            )
        return state

    def _get_execution_state(
        self,
        layer_id: int,
        topk_ids: Tensor,
        base: _LPLBLayerState,
    ) -> _LPLBLayerState:
        """Return scratch buffers private to one CUDA graph or eager stream.

        Placement-derived matrices remain shared. Only buffers overwritten by
        the four-stage LP pipeline are cloned. This prevents a replay for one
        batch shape from corrupting another overlapping replay.
        """

        if not topk_ids.is_cuda:
            return base

        if torch.cuda.is_current_stream_capturing():
            execution_key: Hashable = ("graph", topk_ids.data_ptr())
        else:
            execution_key = (
                "stream",
                torch.cuda.current_stream(topk_ids.device).cuda_stream,
            )
        cache_key = (layer_id, execution_key)
        cached = self._execution_state.get(cache_key)
        if cached is not None:
            return cached

        state = self._clone_execution_state(base)
        self._execution_state[cache_key] = state
        return state

    @staticmethod
    def _clone_execution_state(base: _LPLBLayerState) -> _LPLBLayerState:
        return replace(
            base,
            A_full=base.A_full.clone(),
            b=torch.empty_like(base.b),
            t1=torch.empty_like(base.t1),
            x=torch.empty_like(base.x),
            log2phy_prob=torch.empty_like(base.log2phy_prob),
        )


def _configure(config: RoutingPolicyConfig) -> RoutingPolicySpec:
    # TODO: Replace LPLB_USE_LOCAL_COUNT with an explicit RoutingPolicyConfig
    # option read from ``config``.
    del config
    if os.environ.get("LPLB_USE_LOCAL_COUNT", "0") == "1":
        return replace(ROUTING_POLICY, logical_count="local")
    return ROUTING_POLICY


def _build_router(context: RouterBuildContext) -> LPLBL2Router:
    kernels = context.options.get("lplb_kernels")
    if kernels is None:
        from evenmoe.kernels.lplb import CUDALPLBKernels

        kernels = CUDALPLBKernels()
    elif not isinstance(kernels, LPLBKernels):
        raise TypeError("lplb_kernels must implement LPLBKernels.")
    return LPLBL2Router(kernels=kernels, num_gpus=context.ep_size)


ROUTING_POLICY = RoutingPolicySpec(
    "lplb",
    LPLBL2Router.phase,
    logical_count="global",
    requires_placement_state=True,
    supports_concurrent_microbatches=False,
    graph_stability="realloc_on_placement_change",
    requires_redundancy=True,
    configure=_configure,
    build_router=_build_router,
)
