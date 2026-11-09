# SPDX-License-Identifier: Apache-2.0
"""Experimental V3-only MLB bridge to the native MegaMoE forwards.

Routing/planning use this repository's policies. Packed weights and their global
scales move together with SGLang's existing synchronous expert mover. This is an
external routing adapter, not a claim that the policies are fused into CUDA.
"""

import copy
import logging
import os
from dataclasses import replace

import torch
import torch.distributed as dist

POLICY = os.environ.get("EVENMOE_MEGAMOE_POLICY", "baseline")
ENABLED = POLICY != "baseline"
_TEMP = {}
_MOVER_GROUP = None


def initialize_mover_group(device, rank, world_size):
    """Initialize the native mover's default NCCL group on every rank once.

    SGLang uses a separate EP group for load statistics. Its native mover uses
    P2POp without a group, hence WORLD. A first batch_isend_irecv on a lazy WORLD
    communicator requires every rank, even if the actual transfer is sparse.
    Initialize it collectively before any rank can enter that first transfer.
    """
    global _MOVER_GROUP
    assert dist.is_initialized()
    group = dist.group.WORLD
    if _MOVER_GROUP is group:
        return
    assert dist.get_world_size(group) == world_size and dist.get_rank(group) == rank
    marker = torch.zeros(1, dtype=torch.int32, device=device)
    dist.all_reduce(marker, group=group)
    _MOVER_GROUP = group


def packed_views(experts, routed_local):
    """Expert-contiguous byte rows plus both global scales; exclude shared slot."""
    count = experts.num_local_experts
    if getattr(experts, "_mega_moe_sm90_fp8_weights", False):
        tensors = [*experts.mega_l1_weights, *experts.mega_l2_weights]
    else:
        tensors = [
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            experts.mega_l1_global_scales,
            experts.mega_l2_global_scales,
        ]
    assert all(t.is_contiguous() and t.shape[0] == count for t in tensors)
    return [t.view(count, -1)[:routed_local] for t in tensors]


def move_packed(experts, old_map, new_map, rank, world_size=8):
    from sglang.srt.eplb.expert_location_updater import (
        create_temp_buffers,
        update_expert_weights_single_layer,
    )

    local = len(old_map) // world_size
    assert len(new_map) == len(old_map) and set(old_map) == set(new_map)
    weights = getattr(experts, "_matched_fp8_mlb_weights", None)
    if weights is None:
        weights = packed_views(experts, local)
    assert all(w.shape[0] == local for w in weights)
    initialize_mover_group(weights[0].device, rank, world_size)
    key = tuple((tuple(w.shape), str(w.dtype), str(w.device)) for w in weights)
    if key not in _TEMP:
        _TEMP[key] = create_temp_buffers(weights)
    missing = []
    update_expert_weights_single_layer(
        routed_experts_weights=weights,
        temp_buffers=_TEMP[key],
        old_physical_to_logical_map=old_map,
        new_physical_to_logical_map=new_map,
        num_local_physical_experts=local,
        num_gpu_per_node=world_size,
        rank=rank,
        world_size=world_size,
        missing_logical_experts_info=missing,
    )
    assert not missing, missing


class Collectives:
    def __init__(self, group, rank, world_size):
        self.group, self.rank, self.world_size = group, rank, world_size

    def all_reduce_sum(self, payload):
        dist.all_reduce(payload, group=self.group)
        return payload


def snapshot_from_plan(plan, layer, rank, world_size=8):
    from evenmoe.core.types import PlacementSnapshot

    mapping = plan.physical_to_logical_map[0]
    candidates = plan.logical_to_all_physical_map[0]
    counts = plan.logical_to_physical_count[0]
    meta = {}
    if "rank_quota_prefix" in plan.metadata:
        meta["rank_quota_prefix"] = plan.metadata["rank_quota_prefix"][0]
    return PlacementSnapshot(
        layer_id=layer,
        num_logical_experts=candidates.shape[0],
        num_physical_experts=mapping.numel(),
        ep_size=world_size,
        num_local_physical_experts=mapping.numel() // world_size,
        physical_to_logical_map=mapping,
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=counts,
        metadata=meta,
    )


def full_replica_snapshot(snapshot):
    """Recover all loaded replicas, not SGLang's rank-local dispatch shortlist.

    Native MegaMoE metadata can collapse an expert's candidates to the nearest
    copy. LPLB must see every physical slot identically on every EP rank. Keep
    the native default pick for static routing, but rebuild candidates/counts
    from the authoritative loaded physical-to-logical map once at commit.
    """
    mapping = snapshot.physical_to_logical_map.detach().cpu().tolist()
    assert len(mapping) == snapshot.num_physical_experts
    candidates = [[] for _ in range(snapshot.num_logical_experts)]
    for physical, logical in enumerate(mapping):
        assert 0 <= logical < len(candidates), (physical, logical)
        candidates[logical].append(physical)
    counts = [len(row) for row in candidates]
    assert min(counts) > 0, "Loaded layout is missing a logical expert"
    width = max(counts)
    padded = [row + [-1] * (width - len(row)) for row in candidates]
    device = snapshot.physical_to_logical_map.device
    return replace(
        snapshot,
        logical_to_physical_candidates=torch.tensor(padded, dtype=torch.int64, device=device),
        logical_to_physical_count=torch.tensor(counts, dtype=torch.int64, device=device),
    )


def placement_from_counts(model_config, logical_count):
    """Called at load time in place of SGLang's planner, retaining its metadata."""
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    from sglang.srt.runtime_context import get_parallel

    from evenmoe import MoELoadBalancer
    from evenmoe.adapters.sglang import to_placement_request

    common = ExpertLocationMetadata._init_common(model_config)
    assert common["ep_size"] == 8
    plan = MoELoadBalancer().plan_placement(
        to_placement_request(
            logical_count.to("cuda"),
            num_physical_experts=common["num_physical_experts"],
            num_local_physical_experts=common["num_physical_experts"] // 8,
            num_groups=8,
            num_nodes=1,
            algorithm="auto",
        )
    )
    logging.getLogger(__name__).info("NVFP4_MLB_PLACEMENT policy=%s", dict(plan.metadata))
    return ExpertLocationMetadata._init_raw(
        ep_size=8,
        physical_to_logical_map=plan.physical_to_logical_map.to("cuda"),
        logical_to_all_physical_map=plan.logical_to_all_physical_map.to("cuda"),
        moe_ep_rank=get_parallel().moe_ep_rank,
    )


class State:
    def __init__(self, moe):
        from sglang.srt.configs.shared_experts import SharedExpertsConfig
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        from sglang.srt.runtime_context import get_exec, get_parallel

        from evenmoe import MoELoadBalancer
        from evenmoe.adapters.sglang import to_placement_snapshot
        from evenmoe.policies.l1.ultraep import RefreshGate

        p = get_parallel()
        assert p.moe_ep_size == 8 and moe.config.hidden_size == 7168
        assert getattr(moe.experts, "_mlb_v3_converted", False) or getattr(
            moe.experts, "_mega_moe_sm90_fp8_weights", False
        )
        self.rank = p.moe_ep_rank
        self.group = p.moe_ep_group.device_group
        self.layer = moe.layer_id
        self.physical = 256 + get_exec().moe.ep_num_redundant_experts
        self.shared = moe.num_fused_shared_experts
        assert self.shared == int("waterfill" in POLICY or POLICY == "home_shared")
        self.cfg = copy.copy(moe.topk.topk_config)
        self.cfg.top_k = 8
        self.cfg.shared_experts = SharedExpertsConfig(fusion_disabled=True)
        self.cfg.fused_gate_packed_ids = False
        algorithm = "static" if POLICY == "home_shared" else POLICY
        self.balancer = MoELoadBalancer.from_algorithm(
            algorithm,
            ep_size=8,
            source_rank=self.rank,
            experts_per_rank=self.physical // 8,
            collectives=Collectives(self.group, self.rank, 8),
        )
        live = get_global_expert_location_metadata()
        if live is None:
            from evenmoe.core.types import PlacementSnapshot

            ids = torch.arange(256, device="cuda", dtype=torch.int64)
            self.placement = PlacementSnapshot(
                self.layer, 256, 256, 8, 32, ids, ids[:, None], torch.ones_like(ids), ids
            )
        else:
            self.placement = full_replica_snapshot(to_placement_snapshot(live, self.layer))
        self.balancer.on_placement_committed(self.placement)
        self.refresh = RefreshGate(int(os.environ.get("EVENMOE_MEGAMOE_REFRESH_INTERVAL", "8")))
        self.refreshes = 0
        self.old_map = self.placement.physical_to_logical_map.cpu().tolist()
        logging.getLogger(__name__).warning(
            "NVFP4_MLB_ACTIVE rank=%s layer=%s policy=%s physical=%s topk=%s",
            self.rank,
            self.layer,
            POLICY,
            self.physical + self.shared * 8,
            8 + self.shared,
        )

    def maybe_refresh(self, moe, ids, batch):
        if not POLICY.startswith("ultraep") or batch is None:
            return
        mode = batch.global_forward_mode or batch.forward_mode
        # Match README adapter semantics: representative prefill refreshes;
        # pure decode uses the committed placement and quota router.
        if (mode.is_decode() or mode.is_idle()) and not batch.is_extend_in_batch:
            return
        if not self.refresh.is_due(self.layer):
            return
        counts = batch.original_global_num_tokens_cpu
        if counts is None:
            counts = batch.global_num_tokens_cpu
        total = sum(counts) if counts is not None else batch.global_num_token_non_padded_cpu
        if not self.refresh.should_refresh(self.layer, representative=total >= 512):
            return
        from evenmoe.adapters.sglang import to_placement_request
        from evenmoe.kernels.ops.counting import count_logical_experts

        local = count_logical_experts(ids, 256, dtype=torch.int32)
        gathered = torch.empty((8, 256), device=ids.device, dtype=torch.int32)
        dist.all_gather_into_tensor(gathered, local.contiguous(), group=self.group)
        plan = self.balancer.plan_placement(
            to_placement_request(
                gathered[:, None, :],
                num_physical_experts=self.physical,
                num_local_physical_experts=self.physical // 8,
                num_groups=8,
                num_nodes=1,
                algorithm="ultraep",
                policy_metadata={"rank": self.rank, "num_nvl_ranks": 8},
            )
        )
        new_map = plan.physical_to_logical_map[0].cpu().tolist()
        move_packed(moe.experts, self.old_map, new_map, self.rank)
        # Publish only AFTER all packed tensors and scales have moved.
        self.placement = snapshot_from_plan(plan, self.layer, self.rank)
        self.balancer.on_placement_committed(self.placement)
        self.old_map = new_map
        self.refreshes += 1
        if self.layer == 3:
            logging.getLogger(__name__).warning(
                "NVFP4_MLB_REFRESH rank=%s count=%s", self.rank, self.refreshes
            )


from .decode_guard import is_collective_decode


def forward(moe, hidden_states, forward_batch, input_ids_global, num_tokens):
    from sglang.srt.layers.moe.mega_moe import run_mega_routed_experts
    from sglang.srt.layers.moe.topk import select_experts

    from evenmoe.adapters.sglang import to_sglang_routing_output
    from evenmoe.core.types import RoutingRequest, RoutingStage

    if not hasattr(moe, "_nvfp4_mlb_state"):
        moe._nvfp4_mlb_state = State(moe)
    state = moe._nvfp4_mlb_state
    assert not torch.cuda.is_current_stream_capturing(), (
        "MLB comparison uses matched eager execution"
    )
    token_count, _ = moe.topk.routing_inputs()
    if num_tokens:
        logits = moe.gate(hidden_states, forward_batch=forward_batch)
        top = select_experts(
            hidden_states,
            logits,
            state.cfg,
            layer_id=moe.layer_id,
            num_token_non_padded=token_count,
            expert_location_dispatch_info=None,
        )
        ids, weights = top.topk_ids, top.topk_weights
    else:
        ids = torch.empty((0, 8), device=hidden_states.device, dtype=torch.int32)
        weights = torch.empty((0, 8), device=hidden_states.device, dtype=torch.float32)
    state.maybe_refresh(moe, ids, forward_batch)
    scale = 1.0 if moe.routed_scaling_factor_fused_in_topk else float(moe.routed_scaling_factor)
    stage = (
        RoutingStage.DECODE
        if forward_batch is not None and forward_batch.forward_mode.is_decode()
        else RoutingStage.PREFILL
    )
    use_twf = os.environ.get("EVENMOE_MEGAMOE_TWF", "0") == "1"
    if use_twf and is_collective_decode(forward_batch):
        assert POLICY == "static+waterfill", "Pilot TWF supports static EPLB only"
        from .workbuddy_tilewf import route as tile_route

        routed = tile_route(state, ids, weights, token_count, scale)
    else:
        decision = state.balancer.route_tokens(
            RoutingRequest(
                layer_id=moe.layer_id,
                logical_topk_ids=ids,
                topk_weights=weights,
                placement=state.placement,
                token_count=token_count if token_count is not None else num_tokens,
                stage=stage,
                routed_scaling_factor=scale,
            )
        )
        if POLICY == "home_shared":
            decision = replace(
                decision,
                shared_expert_rank=torch.full(
                    (num_tokens,), state.rank, dtype=torch.int64, device=hidden_states.device
                ),
            )
        routed = to_sglang_routing_output(
            decision, num_routed_experts=state.physical, world_size=8, routed_scaling_factor=scale
        )
    return run_mega_routed_experts(
        moe.experts,
        hidden_states,
        routed.topk_ids,
        routed.topk_weights,
        hidden_size=7168,
        intermediate_size=2048,
        top_k=8 + state.shared,
        num_tokens=num_tokens,
        activation_clamp=None,
        routed_scaling_factor=scale,
        num_shared_experts=0,
        shared_l1_weights=None,
        shared_l2_weights=None,
        routing_in_buffer=False,
    )
