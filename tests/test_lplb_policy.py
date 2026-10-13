# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
#
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
#
# Modifications:
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`LPLBL2Router` and its LP matrix builder.

These tests run on CPU with a mock :class:`LPLBKernels` implementation
because the real SGLang kernels require CUDA + cuBLASDx + Math-DX. Kernel
*correctness* is validated separately by the GPU parity tests in
``tests/test_lplb_runtime_against_sglang.py``; here we validate:

1. The LP constraint matrices match a hand-derived reference for a known
   placement (geometry and sign conventions).
2. Explicit ``prepare_layer`` calls replace placement-derived state, while
   forwards reuse the prepared state without version or identity checks.
3. ``route_tokens`` fast paths (stage gating, no redundancy, empty batch)
   bypass the kernel pipeline cleanly.
4. ``route_tokens`` invokes the kernels in the right order with buffers
   from the prepared layer state (no per-call allocation).
5. Protocol conformance and input validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from moe_load_balancer.kernels.lplb import LPLBKernels
from moe_load_balancer.policies.l2.lplb import (
    LPLBL2Router,
    _build_layer_state,
    _identity_dispatch,
)

# -- Test fixtures: a tiny placement we can reason about by hand -----------
#
# Layout: 4 logical experts -> 6 physical experts -> 2 GPUs (3 physicals each)
#
#   logical 0  (replicated): physicals 0 (GPU 0) and 3 (GPU 1)
#   logical 1  (single):     physical  1 (GPU 0)
#   logical 2  (replicated): physicals 2 (GPU 0) and 4 (GPU 1)
#   logical 3  (single):     physical  5 (GPU 1)
#
# Derived:
#   log_single     = [1, 3]              phy_single     = [1, 5]
#   log_replicated = [0, 2]              phy_replicated = [0, 2, 3, 4]
#
#   B_full = [[1,1,1,0,0,0],
#             [0,0,0,1,1,1]]
#   B1 = B_full[:, [1,5]]                = [[1,0],[0,1]]
#   B2 = B_full[:, [0,2,3,4]]            = [[1,1,0,0],[0,0,1,1]]
#   C  = [[1,0,1,0],                       # logical 0 -> phy_replicated [0,2]
#         [0,1,0,1]]                       # logical 2 -> phy_replicated [1,3]
#   A_base shape = (NC=4, NV-1=7)  ->  A_full has NV=8
#
# -------------------------------------------------------------------------


def _toy_snapshot(layer_id: int = 0) -> PlacementSnapshot:
    log2phy = torch.tensor([[0, 3], [1, -1], [2, 4], [5, -1]], dtype=torch.int64)
    log_counts = torch.tensor([2, 1, 2, 1], dtype=torch.int64)
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 1, 2, 0, 2, 3]),
        logical_to_physical_candidates=log2phy,
        logical_to_physical_count=log_counts,
        default_physical_for_logical=None,
    )


def _same_shape_rebalanced_snapshot(layer_id: int = 0) -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 1, 1, 2, 3, 3]),
        logical_to_physical_candidates=torch.tensor(
            [[0, -1], [1, 2], [3, -1], [4, 5]], dtype=torch.int64
        ),
        logical_to_physical_count=torch.tensor([1, 2, 1, 2], dtype=torch.int64),
    )


def _shape_changing_rebalanced_snapshot(layer_id: int = 0) -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 1, 2, 3, 0, 0]),
        logical_to_physical_candidates=torch.tensor(
            [[0, 4, 5], [1, -1, -1], [2, -1, -1], [3, -1, -1]],
            dtype=torch.int64,
        ),
        logical_to_physical_count=torch.tensor([3, 1, 1, 1], dtype=torch.int64),
    )


def _no_redundancy_snapshot(
    layer_id: int = 0, default_map: Tensor | None = None
) -> PlacementSnapshot:
    """4 logicals each with exactly one physical copy."""
    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([0, 1, 2, 3]),
        logical_to_physical_candidates=torch.tensor(
            [[0, -1], [1, -1], [2, -1], [3, -1]], dtype=torch.int64
        ),
        logical_to_physical_count=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        default_physical_for_logical=default_map,
    )


# -- Mock LPLBKernels for the pipeline tests ------------------------------


@dataclass
class _KernelCall:
    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]


@dataclass
class _RecordingKernels:
    """Mock LPLBKernels that records calls and writes plausible outputs.

    Outputs are not LP-correct; they're just shape-valid so the policy
    can finish the pipeline and return a RoutingDecision. The point of
    these tests is to validate the policy plumbing — kernel correctness
    is covered by the GPU integration tests.
    """

    calls: list[_KernelCall] = field(default_factory=list)
    warmup_shapes: list[tuple[int, int]] = field(default_factory=list)
    precompiled: list[dict] = field(default_factory=list)
    # Identity tracking — verifies buffer reuse across calls.
    last_A_full_id: int | None = None
    last_x_id: int | None = None
    last_log2phy_prob_id: int | None = None

    def _record(self, name: str, *tensors: Tensor) -> None:
        shapes = tuple(tuple(t.shape) for t in tensors)
        dtypes = tuple(str(t.dtype) for t in tensors)
        self.calls.append(_KernelCall(name=name, shapes=shapes, dtypes=dtypes))

    def warmup(self, nc, nv, num_iters, device):
        self.warmup_shapes.append((nc, nv))

    def warmup_shape(self, **kwargs):
        # Deliberately not recorded in warmup_shapes: that list is what the
        # per-layer warmup assertions count, and precompiling the set up front
        # is a different event.
        self.precompiled.append(kwargs)

    def prep_lp_inputs(
        self, A_full, b, t1, global_counts, log_single, log_replicated, B1, A_base_row_sum
    ):
        self._record(
            "prep",
            A_full,
            b,
            t1,
            global_counts,
            log_single,
            log_replicated,
            B1,
            A_base_row_sum,
        )
        self.last_A_full_id = id(A_full)
        # Write plausible values: A_full's last col -> 0, b -> 0, t1 -> ones.
        A_full[:, -1].fill_(0.0)
        b.fill_(0.0)
        t1.fill_(1.0)

    def solve_ipm(self, A, b, c, num_iters=5, result=None):
        self._record("solve", A, b, c)
        self.last_x_id = id(result)
        # Fallback value matches PR #24515's LPLBSolver non-convergence path.
        result.fill_(0.5)
        return result

    def extract_log2phy_prob(self, log2phy_prob, x, t1, phy_single, phy_replicated, log2phy):
        self._record("extract", log2phy_prob, x, t1, phy_single, phy_replicated, log2phy)
        self.last_log2phy_prob_id = id(log2phy_prob)
        # Write a valid distribution: 1.0/count for each valid slot.
        valid = log2phy >= 0
        per_logical_count = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
        log2phy_prob.zero_()
        ones = (1.0 / per_logical_count).expand_as(log2phy_prob).contiguous()
        log2phy_prob[valid] = ones[valid]

    def dispatch_probability(self, topk_ids, log2phy_prob, log2phy_map, **kwargs):
        self._record("dispatch", topk_ids, log2phy_prob, log2phy_map)
        # Always pick column 0 of log2phy_map for each logical id.
        safe = topk_ids.to(torch.int64).clamp(min=0)
        physical = log2phy_map.to(torch.int64)[safe, 0]
        return torch.where(topk_ids >= 0, physical, topk_ids.to(torch.int64)).to(topk_ids.dtype)


def _toy_request(
    *,
    snapshot: PlacementSnapshot,
    topk_ids: Tensor,
    topk_weights: Tensor | None = None,
    stage: RoutingStage | None = RoutingStage.PREFILL,
    global_logical_count: Tensor | None = None,
    enabled_stages: frozenset | None = None,
) -> RoutingRequest:
    if topk_weights is None:
        topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32)
    if global_logical_count is None:
        global_logical_count = torch.ones(snapshot.num_logical_experts, dtype=torch.float32)
    return RoutingRequest(
        layer_id=snapshot.layer_id,
        logical_topk_ids=topk_ids,
        topk_weights=topk_weights,
        placement=snapshot,
        stage=stage,
        global_logical_count=global_logical_count,
        policies=(RoutingPolicyConfig(name="lplb", enabled_stages=enabled_stages),),
    )


# -- 1. LP matrix construction --------------------------------------------


@pytest.mark.unit
class TestBuildLayerState:
    def test_partition_is_correct_for_toy_placement(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)

        assert state.has_redundancy is True
        assert state.log_single.tolist() == [1, 3]
        assert state.phy_single.tolist() == [1, 5]
        assert state.log_replicated.tolist() == [0, 2]
        assert state.phy_replicated.tolist() == [0, 2, 3, 4]

    def test_B1_matches_hand_derivation(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)

        # B1[i, j] = 1 iff GPU i owns phy_single[j].
        # phy_single = [1, 5]. GPU 0 owns phys 0..2 → owns phy 1 (slot 0).
        # GPU 1 owns phys 3..5 → owns phy 5 (slot 1).
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.equal(state.B1, expected)

    def test_A_base_matches_hand_derivation(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)

        # Top block (num_red_log=2 rows): [C | 0 (num_gpus=2 wide) | 0 (1 wide)]
        # C row 0: logical 0 -> phys 0 and 3 -> phy_replicated indices [0, 2]
        # C row 1: logical 2 -> phys 2 and 4 -> phy_replicated indices [1, 3]
        # Bottom block (num_gpus=2 rows): [B2 | I | -1]
        # B2 = B_full[:, [0,2,3,4]] = [[1,1,0,0],[0,0,1,1]]
        expected = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0],
            ]
        )
        assert torch.equal(state.A_base, expected)
        assert state.A_full.shape == (4, 8)  # NC=4, NV = NV-1 + 1 = 8
        # A_full's first NV-1 cols hold A_base.
        assert torch.equal(state.A_full[:, :-1], expected)

    def test_c_vec_minimizes_M_with_big_m_penalty(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=12345.0)
        # c_vec[NV-2] = 1.0  (the peak load M)
        # c_vec[NV-1] = big_m_penalty
        # All other entries = 0
        assert state.c_vec.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 12345.0]

    def test_A_base_row_sum_is_precomputed(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)
        assert torch.equal(state.A_base_row_sum, state.A_base.sum(dim=1))

    def test_workspace_buffers_have_correct_shapes(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)
        nc, nv = state.A_full.shape
        assert state.b.shape == (nc,)
        assert state.t1.shape == (state.log_single.numel(),)
        assert state.x.shape == (nv,)
        assert state.log2phy_prob.shape == (4, 2)

    def test_log2phy_map_for_dispatch_is_int32(self):
        state = _build_layer_state(_toy_snapshot(), num_gpus=2, big_m_penalty=1000.0)
        assert state.log2phy_map_for_dispatch.dtype == torch.int32

    def test_no_redundancy_state_is_degenerate(self):
        snap = _no_redundancy_snapshot()
        state = _build_layer_state(snap, num_gpus=2, big_m_penalty=1000.0)
        assert state.has_redundancy is False
        # No LP matrices; log2phy_prob buffer still allocated for shape stability.
        assert state.log2phy_prob.shape == (4, 2)

    def test_uncovered_physicals_do_not_crash(self):
        """SGLang reserves slots in ``num_physical_experts`` that no logical
        claims (DeepSeek-R1 + --ep-num-redundant-experts under trivial
        placement). The LP must treat those as no-op padding, not as bugs."""
        # 4 logical, 6 physical, but only 4 phys are claimed: log2phy[i, 0]=i,
        # log_counts=[1, 1, 1, 1]. Phys 4 and 5 are reserved but unused.
        log2phy = torch.tensor(
            [
                [0, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1],
                [2, -1, -1, -1, -1, -1],
                [3, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int64,
        )
        log_counts = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        snap = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=4,
            num_physical_experts=6,
            ep_size=2,
            num_local_physical_experts=3,
            physical_to_logical_map=torch.tensor([0, 1, 2, 3, -1, -1]),
            logical_to_physical_candidates=log2phy,
            logical_to_physical_count=log_counts,
        )
        state = _build_layer_state(snap, num_gpus=2, big_m_penalty=1000.0)
        # With every logical at exactly 1 copy and 2 physical slots unclaimed,
        # the LP must degenerate.
        assert state.has_redundancy is False
        # Identity dispatch should also succeed (covers the route_tokens
        # no_redundancy fast path under this snapshot).
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        topk = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        decision = router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))
        assert decision.metadata["lp_skipped"] == "no_redundancy"

    def test_rejects_non_divisible_num_phy(self):
        # 5 physical experts, 2 GPUs → 5 % 2 != 0.
        bad = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=3,
            num_physical_experts=5,
            ep_size=2,
            num_local_physical_experts=3,
            physical_to_logical_map=torch.tensor([0, 0, 1, 2, 2]),
            logical_to_physical_candidates=torch.tensor(
                [[0, 1], [2, -1], [3, 4]], dtype=torch.int64
            ),
            logical_to_physical_count=torch.tensor([2, 1, 2], dtype=torch.int64),
        )
        with pytest.raises(ValueError, match="divisible by num_gpus"):
            _build_layer_state(bad, num_gpus=2, big_m_penalty=1000.0)


# -- 2. Prepared layer state ----------------------------------------------


@pytest.mark.unit
class TestPreparedLayerState:
    def test_forward_reuses_prepared_state(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot(layer_id=0)
        topk = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        router.prepare_layer(snap)
        router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))
        router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        # Warmup fires during prepare only; forwards reuse the installed state.
        assert kernels.warmup_shapes == [(4, 8)]

    def test_shape_compatible_prepare_updates_existing_storage_in_place(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        first = _toy_snapshot()
        second = _same_shape_rebalanced_snapshot()

        router.prepare_layer(first)
        first_state = router._layer_state[0]
        execution_state = router._clone_execution_state(first_state)
        router._execution_state[(0, ("graph", 1))] = execution_state

        router.prepare_layer(second)
        second_state = router._layer_state[0]

        assert second_state is first_state
        assert torch.equal(second_state.log2phy, second.logical_to_physical_candidates)
        # Only the placement-derived columns. The last column is the Big-M
        # auxiliary, which the prep kernel writes per batch and which is
        # torch.empty until it does -- comparing it means comparing
        # uninitialised memory, and torch.equal reports any NaN that lands
        # there as a difference. That made this assertion fail roughly one run
        # in a hundred for a reason that had nothing to do with the state copy.
        assert torch.equal(execution_state.A_full[:, :-1], second_state.A_full[:, :-1])
        assert kernels.warmup_shapes == [(4, 8), (4, 8)]

    def test_shape_changing_prepare_replaces_storage_and_evicts_execution_state(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        router.prepare_layer(_toy_snapshot())
        first_state = router._layer_state[0]
        router._execution_state[(0, ("graph", 1))] = router._clone_execution_state(first_state)

        router.prepare_layer(_shape_changing_rebalanced_snapshot())
        assert router._layer_state[0] is not first_state
        assert router._execution_state == {}

    def test_prepare_keeps_one_state_per_layer(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)

        for _ in range(3):
            router.prepare_layer(_toy_snapshot())

        assert list(router._layer_state.keys()) == [0]

    def test_buffer_reuse_across_calls(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot()
        topk = torch.tensor([[0, 1]], dtype=torch.int32)

        router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))
        first_A_id = kernels.last_A_full_id
        first_x_id = kernels.last_x_id
        first_prob_id = kernels.last_log2phy_prob_id

        router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        assert kernels.last_A_full_id == first_A_id
        assert kernels.last_x_id == first_x_id
        assert kernels.last_log2phy_prob_id == first_prob_id

    def test_execution_state_shares_constants_but_owns_mutable_scratch(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        base = _build_layer_state(
            _toy_snapshot(),
            num_gpus=2,
            big_m_penalty=1000.0,
        )

        execution = router._clone_execution_state(base)

        assert execution.A_base is base.A_base
        assert execution.log2phy is base.log2phy
        assert execution.A_full.data_ptr() != base.A_full.data_ptr()
        assert execution.b.data_ptr() != base.b.data_ptr()
        assert execution.x.data_ptr() != base.x.data_ptr()
        assert execution.log2phy_prob.data_ptr() != base.log2phy_prob.data_ptr()


# -- 3. Fast paths ---------------------------------------------------------


@pytest.mark.unit
class TestRouteFastPaths:
    def test_stage_gating_returns_passthrough(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        topk = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.3, 0.2]])

        # Policy enabled only for PREFILL, request is DECODE.
        request = _toy_request(
            snapshot=_toy_snapshot(),
            topk_ids=topk,
            topk_weights=weights,
            stage=RoutingStage.DECODE,
            enabled_stages=frozenset({RoutingStage.PREFILL}),
        )
        decision = router.route_tokens(request)

        assert torch.equal(decision.routed_physical_topk_ids, topk)
        assert torch.equal(decision.topk_weights, weights)
        assert decision.metadata["skipped"] == "stage_gated"
        # No kernel invocations whatsoever (not even warmup).
        assert kernels.calls == []
        assert kernels.warmup_shapes == []

    def test_no_redundancy_uses_default_map(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        default_map = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        snap = _no_redundancy_snapshot(default_map=default_map)
        topk = torch.tensor([[0, 1, -1], [2, 3, 0]], dtype=torch.int32)

        decision = router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        expected = torch.tensor([[10, 20, -1], [30, 40, 10]], dtype=torch.int32)
        assert torch.equal(decision.routed_physical_topk_ids, expected)
        assert decision.metadata["lp_skipped"] == "no_redundancy"
        assert kernels.calls == []

    def test_no_redundancy_falls_back_to_log2phy_col0(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        # No default_physical_for_logical provided; fall back to log2phy[:, 0].
        snap = _no_redundancy_snapshot(default_map=None)
        topk = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32)

        decision = router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        # log2phy is [[0,-1],[1,-1],[2,-1],[3,-1]], col 0 = [0,1,2,3].
        expected = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32)
        assert torch.equal(decision.routed_physical_topk_ids, expected)

    def test_empty_batch_skips_kernels(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        empty_topk = torch.empty((0, 3), dtype=torch.int32)
        empty_weights = torch.empty((0, 3), dtype=torch.float32)

        decision = router.route_tokens(
            _toy_request(
                snapshot=_toy_snapshot(),
                topk_ids=empty_topk,
                topk_weights=empty_weights,
            )
        )

        assert decision.routed_physical_topk_ids.shape == (0, 3)
        assert decision.metadata["lp_skipped"] == "empty_batch"
        # Warmup fires (layer state is still built), but no per-batch kernels run.
        assert kernels.calls == []
        assert kernels.warmup_shapes == [(4, 8)]


# -- 4. Full kernel pipeline (mocked) -------------------------------------


@pytest.mark.unit
class TestRoutePipeline:
    def test_kernel_call_order(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot()
        topk = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int32)

        router.route_tokens(
            _toy_request(
                snapshot=snap,
                topk_ids=topk,
                global_logical_count=torch.tensor([3.0, 1.0, 2.0, 0.0]),
            )
        )

        assert [c.name for c in kernels.calls] == ["prep", "solve", "extract", "dispatch"]

    def test_kernel_argument_shapes(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot()
        topk = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int32)

        router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        prep = next(c for c in kernels.calls if c.name == "prep")
        # prep_lp_inputs(A_full[NC,NV], b[NC], t1[num_single], global_counts[num_logical],
        #               log_single[num_single], log_replicated[num_red_log],
        #               B1[num_gpus,num_single], A_base_row_sum[NC])
        assert prep.shapes == (
            (4, 8),
            (4,),
            (2,),
            (4,),
            (2,),
            (2,),
            (2, 2),
            (4,),
        )

        solve = next(c for c in kernels.calls if c.name == "solve")
        # solve_ipm(A[NC,NV], b[NC], c[NV])
        assert solve.shapes == ((4, 8), (4,), (8,))

        extract = next(c for c in kernels.calls if c.name == "extract")
        # extract_log2phy_prob(log2phy_prob[NL,MC], x[NV], t1[num_single],
        #                     phy_single[num_single], phy_replicated[num_red_phy],
        #                     log2phy[NL,MC])
        assert extract.shapes == (
            (4, 2),
            (8,),
            (2,),
            (2,),
            (4,),
            (4, 2),
        )

        dispatch = next(c for c in kernels.calls if c.name == "dispatch")
        # dispatch_probability(topk_ids[N,K], log2phy_prob[NL,MC], log2phy_map[NL,MC])
        assert dispatch.shapes == ((2, 3), (4, 2), (4, 2))

    def test_global_logical_count_cast_to_float32(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot()
        topk = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        # Pass an int32 count — the policy should cast to float32 before prep.
        int_counts = torch.tensor([3, 1, 2, 0], dtype=torch.int32)
        router.route_tokens(
            _toy_request(snapshot=snap, topk_ids=topk, global_logical_count=int_counts)
        )

        prep = next(c for c in kernels.calls if c.name == "prep")
        # Index 3 is the global_counts arg.
        assert prep.dtypes[3] == "torch.float32"

    def test_dispatch_output_preserves_topk_shape(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _toy_snapshot()
        topk = torch.tensor([[0, 1, 2], [2, 3, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.int32)

        decision = router.route_tokens(_toy_request(snapshot=snap, topk_ids=topk))

        assert decision.routed_physical_topk_ids.shape == topk.shape
        assert decision.routed_physical_topk_ids.dtype == topk.dtype


# -- 5. Validation --------------------------------------------------------


@pytest.mark.unit
class TestRouteValidation:
    def test_requires_placement(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.ones((1, 2)),
            placement=None,
            stage=RoutingStage.PREFILL,
            global_logical_count=torch.ones(4),
            policies=(RoutingPolicyConfig(name="lplb"),),
        )
        with pytest.raises(ValueError, match="placement"):
            router.route_tokens(request)

    def test_requires_global_logical_count(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.ones((1, 2)),
            placement=_toy_snapshot(),
            stage=RoutingStage.PREFILL,
            global_logical_count=None,
            policies=(RoutingPolicyConfig(name="lplb"),),
        )
        with pytest.raises(ValueError, match="global_logical_count"):
            router.route_tokens(request)

    def test_constructor_rejects_invalid_args(self):
        kernels = _RecordingKernels()
        with pytest.raises(ValueError, match="num_gpus"):
            LPLBL2Router(kernels=kernels, num_gpus=0)
        with pytest.raises(ValueError, match="big_m_penalty"):
            LPLBL2Router(kernels=kernels, num_gpus=2, big_m_penalty=-1.0)
        with pytest.raises(ValueError, match="ipm_iters"):
            LPLBL2Router(kernels=kernels, num_gpus=2, ipm_iters=0)


# -- 6. Identity dispatch + protocol conformance --------------------------


@pytest.mark.unit
class TestIdentityDispatch:
    def test_uses_default_map_when_provided(self):
        snap = _no_redundancy_snapshot(
            default_map=torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        )
        topk = torch.tensor([[0, 1, -1, 3]], dtype=torch.int32)
        result = _identity_dispatch(topk, snap)
        assert result.dtype == torch.int32
        assert result.tolist() == [[10, 20, -1, 40]]

    def test_falls_back_to_log2phy_col_zero(self):
        snap = _no_redundancy_snapshot(default_map=None)
        topk = torch.tensor([[3, 2, -1, 0]], dtype=torch.int32)
        result = _identity_dispatch(topk, snap)
        assert result.tolist() == [[3, 2, -1, 0]]


@pytest.mark.unit
class TestProtocolConformance:
    def test_lplb_l2_router_satisfies_token_router(self):
        assert isinstance(LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2), TokenRouter)

    def test_recording_kernels_satisfies_lplb_kernels(self):
        # The mock is the reference for "minimal valid LPLBKernels impl".
        assert isinstance(_RecordingKernels(), LPLBKernels)


class TestNoRedundancyNeedsNoCount:
    """The zero-redundancy path exists to avoid paying for the count.

    The orchestrator asks before gathering it (``GlobalLogicalCountConsumer``),
    so LPLB has to answer honestly *and* stay callable without one -- otherwise
    the collective it declines is still issued, or declining it raises.
    """

    def test_a_layer_without_replicas_declines_the_count(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        snap = _no_redundancy_snapshot()
        request = _toy_request(
            snapshot=snap, topk_ids=torch.tensor([[0, 1, -1]], dtype=torch.int32)
        )

        assert router.needs_global_logical_count(request) is False

    def test_a_layer_with_replicas_still_asks_for_the_count(self):
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        request = _toy_request(
            snapshot=_toy_snapshot(), topk_ids=torch.tensor([[0, 1, -1]], dtype=torch.int32)
        )

        assert router.needs_global_logical_count(request) is True

    def test_routing_without_a_count_still_dispatches_by_the_default_map(self):
        kernels = _RecordingKernels()
        router = LPLBL2Router(kernels=kernels, num_gpus=2)
        snap = _no_redundancy_snapshot(
            default_map=torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        )
        topk = torch.tensor([[0, 1, -1], [2, 3, 0]], dtype=torch.int32)
        request = RoutingRequest(
            layer_id=snap.layer_id,
            logical_topk_ids=topk,
            topk_weights=torch.ones(topk.shape, dtype=torch.float32),
            placement=snap,
            stage=RoutingStage.PREFILL,
            global_logical_count=None,
            policies=(RoutingPolicyConfig(name="lplb"),),
        )

        decision = router.route_tokens(request)

        assert torch.equal(
            decision.routed_physical_topk_ids,
            torch.tensor([[10, 20, -1], [30, 40, 10]], dtype=torch.int32),
        )
        assert decision.metadata["lp_skipped"] == "no_redundancy"
        # Nothing to plan from without a count; the orchestrator falls back to
        # counting the physical ids returned here, which for identity dispatch
        # describes the same placement.
        assert "planned_rank_load" not in decision.metadata
        assert kernels.calls == []

    def test_a_supplied_count_is_still_reported_as_planned_load(self):
        """A caller that has the count anyway loses no diagnostics."""
        router = LPLBL2Router(kernels=_RecordingKernels(), num_gpus=2)
        snap = _no_redundancy_snapshot(
            default_map=torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        )
        decision = router.route_tokens(
            _toy_request(snapshot=snap, topk_ids=torch.tensor([[0, 1]], dtype=torch.int32))
        )

        assert decision.metadata["planned_rank_load"] is not None


class TestTheDrawIsPartOfTheDecision:
    """Which token lands on which replica must not depend on who applies it.

    The share table is the policy's decision; turning it into per-token ids is
    the mechanical step that realises it. If each runtime draws that step its
    own way, one algorithm produces different routing on different frameworks --
    and an RNG additionally makes it unreproducible and leaves tensor-parallel
    peers, whose router output is replicated, free to send one token to two
    different replicas of its expert.
    """

    def test_the_draw_is_a_pure_function_of_the_token_index(self):
        from moe_load_balancer.kernels.ops.replica_draw import uniform_draw

        a = uniform_draw(16, 4, "cpu")
        b = uniform_draw(16, 4, "cpu")
        assert torch.equal(a, b)

    def test_slots_of_one_token_share_a_draw(self):
        """The choice is per token, not per slot: a token's k routed experts are
        all decided from the same draw, as they are wherever this is applied
        inside a fused kernel."""
        from moe_load_balancer.kernels.ops.replica_draw import uniform_draw

        topk = 4
        draw = uniform_draw(3 * topk, topk, "cpu")
        for token in range(3):
            row = draw[token * topk : (token + 1) * topk]
            assert torch.equal(row, row[0].expand_as(row))

    def test_it_matches_the_hash_a_fused_runtime_uses(self):
        """Same constant, same arithmetic, same result -- so the decision
        realises identically whether the policy applies it or the framework
        does."""
        from moe_load_balancer.kernels.ops.replica_draw import (
            KNUTH_MULTIPLIER,
            uniform_draw,
        )

        topk, tokens = 2, 8
        draw = uniform_draw(tokens * topk, topk, "cpu")
        for token in range(tokens):
            expected = ((token * KNUTH_MULTIPLIER) & 0xFFFFFFFF) / 4294967296.0
            assert abs(float(draw[token * topk]) - expected) < 1e-6
