# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JIT-compiled UltraEP placement and quota-reroute kernels.

Framework-neutral. The two kernels the ``ultraep`` policy pair calls, built
through :func:`evenmoe.kernels.jit.utils.load_jit` on first use for
the current GPU.

Layout:

- ``csrc/placement.cuh``       ``solve_placement``: the quota placement solve,
                                one call per MoE layer.
- ``csrc/sparse_reroute.cuh``  ``reroute_sparse<IdType>``: quota-aware replica
                                choice per (token, slot); one module per id
                                dtype.
- ``csrc/{launch,ptx,exception}.cuh``  UltraEP's own launch, PTX and assert
                                helpers.
- ``csrc/entry.cuh``           EvenMoE glue shared by the two entry points.

The kernels are UltraEP v1.0.0's (see each file's header and
``LICENSE.ultraep``); the tvm-ffi entry point at the end of ``placement.cuh``
and ``sparse_reroute.cuh`` is EvenMoE's.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch

from evenmoe.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from evenmoe.kernels.ultraep.config import UltraEPPlacementTuning

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HERE = pathlib.Path(__file__).parent.resolve()

# The register-usage hint UltraEP builds its own extension with.
_CUDA_CFLAGS = ["--ptxas-options=--register-usage-level=10"]


@cache_once
def _placement_module() -> Module:
    return load_jit(
        "ultraep_placement",
        cuda_files=["placement.cuh"],
        cuda_wrappers=[("solve_placement", "solve_placement")],
        extra_cuda_cflags=_CUDA_CFLAGS,
        source_root=_HERE,
    )


@cache_once
def _reroute_module(id_dtype: torch.dtype) -> Module:
    args = make_cpp_args(id_dtype)
    return load_jit(
        "ultraep_reroute_sparse",
        *args,
        cuda_files=["sparse_reroute.cuh"],
        cuda_wrappers=[("reroute_sparse", f"reroute_sparse<{args}>")],
        extra_cuda_cflags=_CUDA_CFLAGS,
        source_root=_HERE,
    )


def solve_placement(
    *,
    expert_loads: torch.Tensor,
    expert_loads_per_rank: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_counts: torch.Tensor,
    logical_instance_quota: torch.Tensor,
    logical_instance_quota_prefix: torch.Tensor,
    rank_quota_prefix: torch.Tensor,
    num_ranks: int,
    num_local_master_experts: int,
    num_local_redundant_experts: int,
    num_nvl_ranks: int,
    rank: int,
    tuning: UltraEPPlacementTuning,
) -> None:
    """Solve one layer's replica placement and per-replica token quota in place.

    Every tensor is int32, contiguous, on one CUDA device. ``expert_loads`` is
    ``[num_logical]`` and ``expert_loads_per_rank`` ``[num_ranks, num_logical]``
    (inputs); ``physical_to_logical_map`` is ``[num_physical]``,
    ``logical_replica_counts`` ``[num_logical]`` and the other four
    ``[num_logical, num_ranks]`` (outputs the kernel fills). Shapes and dtypes
    are checked in the entry point, which raises on a mismatch. The launch goes
    to the current stream of the tensors' device.
    """
    _placement_module().solve_placement(
        expert_loads,
        expert_loads_per_rank,
        physical_to_logical_map,
        logical_to_physical_map,
        logical_replica_counts,
        logical_instance_quota,
        logical_instance_quota_prefix,
        rank_quota_prefix,
        num_ranks,
        num_local_master_experts,
        num_local_redundant_experts,
        num_nvl_ranks,
        tuning.balance_threshold,
        tuning.min_tokens_per_replica,
        tuning.allow_zero_master_quota,
        tuning.locality_aware,
        tuning.oracle_eps,
        tuning.kernel_stage,
        rank,
    )


def reroute_sparse(
    *,
    logical_topk_ids: torch.Tensor,
    routed_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_counts: torch.Tensor,
    rank_quota_prefix: torch.Tensor,
    counters: torch.Tensor,
) -> None:
    """Pick a physical replica for every (token, slot) against the solved quota.

    ``logical_topk_ids`` is ``[num_tokens, top_k]`` in int32 or int64;
    ``routed_topk_ids`` is int64 of the same shape and must not alias it.
    ``counters`` (int32, one per logical expert) is the kernel's scratch and is
    zeroed on entry. The other three are the int32 placement tensors as the
    solve left them. Ids outside ``[0, num_logical)`` pass through unchanged.
    """
    if logical_topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"logical_topk_ids must use torch.int32 or torch.int64, got {logical_topk_ids.dtype}"
        )
    _reroute_module(logical_topk_ids.dtype).reroute_sparse(
        logical_topk_ids,
        routed_topk_ids,
        logical_to_physical_map,
        logical_replica_counts,
        rank_quota_prefix,
        counters,
    )
