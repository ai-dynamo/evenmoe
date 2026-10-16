# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution contract implemented by an LPLB kernel backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evenmoe.core.types import TensorLike


@runtime_checkable
class LPLBKernels(Protocol):
    """Tensor-kernel operations required by the framework-neutral LPLB policy.

    The policy owns LP and placement decisions. Implementations own only the
    tensor kernels used by the four execution stages. Mutable output buffers
    are supplied by the policy so implementations can isolate them per CUDA
    graph or eager stream.
    """

    def warmup(
        self,
        nc: int,
        nv: int,
        num_iters: int,
        device: TensorLike,
    ) -> None: ...

    # Optional. Compiles every kernel a shape needs, so a runtime can pay
    # that cost before serving instead of inside a rearrangement. Backends
    # that omit it simply compile lazily.
    def warmup_shape(self, **kwargs) -> None: ...

    def prep_lp_inputs(
        self,
        a_full: TensorLike,
        b: TensorLike,
        t1: TensorLike,
        global_counts: TensorLike,
        log_single: TensorLike,
        log_replicated: TensorLike,
        b1: TensorLike,
        a_base_row_sum: TensorLike,
    ) -> None: ...

    def solve_ipm(
        self,
        a: TensorLike,
        b: TensorLike,
        c: TensorLike,
        num_iters: int = 5,
        result: TensorLike | None = None,
    ) -> TensorLike: ...

    def extract_log2phy_prob(
        self,
        log2phy_prob: TensorLike,
        x: TensorLike,
        t1: TensorLike,
        phy_single: TensorLike,
        phy_replicated: TensorLike,
        log2phy: TensorLike,
    ) -> None: ...

    # ``expert_load_view`` / ``record_enabled`` / ``num_unpadded_tokens`` are
    # optional. Supplying them folds the framework's load accounting into this
    # launch instead of leaving it to a second pass over the same elements --
    # measured at ~45 us per layer, more than the entire fused kernel this one
    # replaced. Omitting them keeps the previous behaviour for a framework that
    # records its own way.
    def dispatch_probability(
        self,
        topk_ids: TensorLike,
        log2phy_prob: TensorLike,
        log2phy_map: TensorLike,
        **kwargs,
    ) -> TensorLike: ...
