# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from evenmoe.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TensorLike,
)

if TYPE_CHECKING:
    import torch

# vLLM ForwardMode -> EvenMoE RoutingStage.  vLLM does not distinguish a dedicated
# "idle" routing mode the way SGLang does; dummy/profile runs reuse DECODE.
_VLLM_MODE_TO_STAGE = {
    "prefill": RoutingStage.PREFILL,
    "decode": RoutingStage.DECODE,
    "mixed": RoutingStage.MIXED,
    "spec": RoutingStage.SPECULATIVE,
}


def collapse_candidates_to_local(
    candidates: TensorLike,
    counts: TensorLike,
    *,
    ep_rank: int,
    num_local_physical_experts: int,
) -> tuple[TensorLike, TensorLike]:
    """Collapse the candidate list to the nearest local replica.

    For each logical expert with at least one replica on this EP rank, the
    candidate list is replaced by just that replica; experts with no local
    replica keep their full (remote) list.

    **This is for rank-dispatch policies only** -- ``static`` and anything else
    that reports ``requires_rank_dispatch_map``. Do not apply it to a
    replica-routing policy such as LPLB, whose entire job is to decide how a
    replicated expert's traffic splits: leaving one candidate answers that
    question in advance, and a rank that holds a copy of every expert it sees is
    left with no redundancy at all, so the LP is skipped outright.

    SGLang draws the same line. The collapse there lives in
    ``compute_logical_to_rank_dispatch_physical_map``, built only when the
    pipeline reports ``requires_rank_dispatch_map``, and in the
    ``init_by_mapping`` / ``init_trivial`` startup paths. ``init_by_eplb`` --
    taken on every rebalance, and the configuration its published EPLB numbers
    were measured in -- passes the policy the full global list.

    Args:
        candidates: ``[num_logical, max_replicas]`` physical slot ids, -1 where
            unused.
        counts:     ``[num_logical]`` number of valid replicas per expert.
        ep_rank:    this rank's index in the EP group.
        num_local_physical_experts: number of physical slots owned by each rank.

    Returns:
        (collapsed_candidates, collapsed_counts) with the same shapes.
    """
    import torch

    if not isinstance(candidates, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise TypeError("collapse_candidates_to_local requires torch.Tensor inputs.")

    low = ep_rank * num_local_physical_experts
    high = low + num_local_physical_experts

    is_local = (candidates >= low) & (candidates < high) & (candidates >= 0)
    has_local = is_local.any(dim=-1)  # [num_logical]

    # For experts with a local copy, find the first (lowest-slot) local replica.
    # The sentinel only has to lose a `min`, so take it from the dtype rather
    # than from the data: reading `candidates.max()` would sync the device on
    # every call, and callers may run this per layer inside a forward pass.
    sentinel = torch.iinfo(candidates.dtype).max
    local_only_vals = torch.where(is_local, candidates, torch.full_like(candidates, sentinel))
    first_local = local_only_vals.min(dim=-1).values  # [num_logical]

    # Build collapsed tensor: keep original for non-local experts, replace for local.
    local_row = torch.full_like(candidates, -1)
    local_row[:, 0] = first_local  # [local_id, -1, -1, ...]

    collapsed = torch.where(has_local.unsqueeze(-1), local_row, candidates)
    collapsed_counts = torch.where(has_local, torch.ones_like(counts), counts)
    return collapsed, collapsed_counts


def build_physical_to_rank_map(
    num_physical_experts: int,
    ep_size: int,
    device: torch.device | str,
) -> TensorLike:
    """Build a [num_physical_experts] tensor mapping each slot to its EP rank.

    SGLang passes this as ``PlacementSnapshot.physical_to_rank_map``, letting
    LPLB incorporate cross-GPU transfer cost into its placement decisions.
    """
    import torch

    num_local = num_physical_experts // ep_size
    return torch.arange(num_physical_experts, device=device) // num_local


def nearest_replica_table(
    candidates: TensorLike,
    counts: TensorLike,
    *,
    ep_rank: int,
    num_local_physical_experts: int,
    ep_size: int | None = None,
    num_gpus_per_node: int | None = None,
    seed: int = 42,
) -> TensorLike:
    """Build the per-rank dispatch map `static` routes through.

    This follows SGLang's `compute_logical_to_rank_dispatch_physical_map`
    exactly, because `static` is meant to be the same policy on both runtimes
    and a differently-built map is a different policy wearing the same name.
    Four tiers, per logical expert and per rank:

      1. a single candidate is taken as-is
      2. otherwise a replica on this rank's own GPU, if one exists
      3. otherwise, when the candidates on this rank's node are a strict
         subset of all candidates, the first of those (SGLang >= v0.5.16)
      4. ranks still unassigned share the candidates evenly -- `_fair_choices`
         repeats the candidate list `k // len` times, samples the remainder,
         and shuffles from a fixed seed so every rank derives the same map
         without communicating

    The even share in step 4 is the part that matters and the part a
    "nearest, else rotate" shortcut misses: without it the ranks that have no
    local replica pile onto whichever candidate the rotation lands on.

    ``ep_size`` must be passed when any rank holds no replica of any expert:
    inferring it from the largest physical id present undercounts exactly then,
    and the caller's own rank can fall outside the table. ``num_gpus_per_node``
    defaults to the whole EP group, i.e. a single node, where tier 3 never
    narrows anything and every rank without a local copy joins the even share.
    """
    import random

    import torch

    cand = candidates.to(torch.int64).cpu()
    cnt = counts.to(torch.int64).cpu()
    num_logical = int(cand.shape[0])
    per_gpu = max(1, int(num_local_physical_experts))
    ep_size = int(ep_size) if ep_size else _ep_size_from(cand, per_gpu)
    gpus_per_node = int(num_gpus_per_node) if num_gpus_per_node else ep_size
    per_node = per_gpu * gpus_per_node

    rng = random.Random(seed)
    out = [-1] * num_logical

    for logical in range(num_logical):
        live = int(cnt[logical].item())
        cands = [int(cand[logical, j].item()) for j in range(live)]
        if not cands:
            out[logical] = logical
            continue

        # Every rank is resolved, not just ours: step 4's even share depends on
        # how many ranks came up empty, so the whole column has to be built.
        chosen = [-1] * ep_size
        remaining = []
        for r in range(ep_size):
            if len(cands) == 1:
                chosen[r] = cands[0]
                continue
            same_gpu = [c for c in cands if c // per_gpu == r]
            if same_gpu:
                chosen[r] = same_gpu[0]
                continue
            same_node = [c for c in cands if c // per_node == r // gpus_per_node]
            # The node tier applies only when it narrows the choice (SGLang >= v0.5.16).
            if 0 < len(same_node) < len(cands):
                chosen[r] = same_node[0]
                continue
            remaining.append(r)

        if remaining:
            k = len(remaining)
            quotient, rem = divmod(k, len(cands))
            share = cands * quotient + rng.sample(cands, k=rem)
            rng.shuffle(share)
            for r, c in zip(remaining, share):
                chosen[r] = c

        out[logical] = chosen[ep_rank]

    return torch.tensor(out, dtype=torch.int64, device=candidates.device)


def _ep_size_from(cand, per_gpu) -> int:
    """Largest physical id present tells how many ranks the placement spans."""
    import torch

    valid = cand[cand >= 0]
    if valid.numel() == 0:
        return 1
    return int(torch.div(valid.max(), per_gpu, rounding_mode="floor").item()) + 1

def to_placement_snapshot(
    layer_state: object,
    layer_id: int,
    *,
    physical_to_logical_map: TensorLike,
    num_logical_experts: int,
    num_physical_experts: int,
    ep_size: int,
    candidates: TensorLike | None = None,
    counts: TensorLike | None = None,
    default_physical_for_logical: TensorLike | None = None,
    physical_to_rank_map: TensorLike | None = None,
    rank_quota_prefix: TensorLike | None = None,
) -> PlacementSnapshot:
    """Translate vLLM's per-layer ``EplbLayerState`` into an L2 snapshot.

    ``candidates`` and ``counts`` are the logical-to-physical candidate map and
    replica-count tensor to pass to EvenMoE.  If omitted, they are read from
    ``layer_state.logical_to_physical_map`` and
    ``layer_state.logical_replica_count`` (the raw global-all-replicas lists
    that vLLM keeps).

    Callers that want SGLang-compatible behaviour should pre-collapse the
    candidate list with :func:`collapse_candidates_to_local` and pass the
    result here.  Without collapsing, LPLB sees every global replica and may
    route cross-GPU even when a local copy exists.

    ``physical_to_logical_map`` must be supplied by Glue: vLLM keeps it on the
    model-level ``EplbModelState``, not on the layer state.

    ``physical_to_rank_map`` maps each physical slot to its EP rank; SGLang
    always provides this so LPLB can incorporate cross-GPU transfer cost.
    Build it with :func:`build_physical_to_rank_map`.

    ``rank_quota_prefix`` is the per-replica token quota table UltraEP's L1
    policy publishes on ``PlacementPlan.metadata``; ``UltraEPL2Router`` reads
    it back off this snapshot's own ``metadata``. Omitted for every other
    policy, and by ``UltraEPL2Router`` itself before the first L1 solve.
    """

    if candidates is None:
        candidates = getattr(layer_state, "logical_to_physical_map", None)
    if counts is None:
        counts = getattr(layer_state, "logical_replica_count", None)
    if candidates is None or counts is None:
        raise ValueError(
            "vLLM L2 routing requires a layer state with logical_to_physical_map "
            "and logical_replica_count; EPLB is probably not enabled."
        )
    if num_physical_experts % ep_size != 0:
        raise ValueError("vLLM placement requires num_physical_experts to be divisible by ep_size.")

    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        ep_size=ep_size,
        num_local_physical_experts=num_physical_experts // ep_size,
        physical_to_logical_map=physical_to_logical_map,
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=counts,
        default_physical_for_logical=default_physical_for_logical,
        physical_to_rank_map=physical_to_rank_map,
        metadata=(
            {}
            if rank_quota_prefix is None
            else {"rank_quota_prefix": rank_quota_prefix}
        ),
    )


def to_routing_request(
    *,
    layer_id: int,
    logical_topk_ids: TensorLike,
    topk_weights: TensorLike,
    policies: Sequence[RoutingPolicyConfig | str] = (),
    placement: PlacementSnapshot | None = None,
    stage: RoutingStage | str | None = None,
    token_count: TensorLike | None = None,
    routed_scaling_factor: float = 1.0,
    global_logical_count: TensorLike | None = None,
) -> RoutingRequest:
    """Translate vLLM's logical TopK and live placement into EvenMoE input.

    ``global_logical_count`` is the pre-computed, all-reduced per-logical-expert
    token count for this layer.  When provided, EvenMoE skips its internal
    count_logical_experts + EP all_reduce (which are not CUDA-graph-safe).
    vLLM computes counts inside the graph and all_reduces them outside, then
    passes the result here as a stable tensor so LP solve is fully capturable.
    """
    normalized_stage = None if stage is None else RoutingStage.from_str(stage)
    return RoutingRequest(
        layer_id=layer_id,
        logical_topk_ids=logical_topk_ids,
        topk_weights=topk_weights,
        policies=tuple(policies),
        placement=placement,
        stage=normalized_stage,
        token_count=token_count,
        routed_scaling_factor=routed_scaling_factor,
        global_logical_count=global_logical_count,
    )


def to_vllm_topk_ids(decision: RoutingDecision) -> TensorLike:
    """Translate an EvenMoE decision into the tensor vLLM's router writes back.

    vLLM's routing boundary (``BaseRouter._apply_eplb_mapping``) returns only
    the physical ``topk_ids`` of the routed experts; weights are produced
    upstream and are not rewritten. A shared-expert decision adds a column
    rather than substituting an id, so it travels separately through
    ``to_vllm_shared_expert_rank``.
    """

    return decision.routed_physical_topk_ids


def to_vllm_shared_expert_rank(decision: RoutingDecision) -> TensorLike | None:
    """The per-token EP rank a shared-expert policy chose, or None when the
    pipeline has no shared-expert policy.

    vLLM builds its widened shared-expert layout from the layer's own geometry,
    so it takes the rank and does the id arithmetic itself, unlike SGLang, which
    takes expanded ids from ``to_sglang_routing_output``.
    """

    return decision.shared_expert_rank


def stage_from_vllm(name: str | None) -> RoutingStage | None:
    """Map a vLLM-side stage label onto EvenMoE's stage enum."""

    if name is None:
        return None
    return _VLLM_MODE_TO_STAGE.get(name.lower())
