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
"""Lazy execution binding for the SDK-owned LPLB CUDA pipeline."""

from __future__ import annotations

from typing import Any


class CUDALPLBKernels:
    """Framework-independent implementation of the :class:`LPLBKernels` contract."""

    def __init__(self) -> None:
        try:
            from evenmoe.kernels.lplb import cuda_solver
        except ImportError as exc:  # pragma: no cover - non-runtime envs only
            raise ImportError(
                "CUDALPLBKernels requires the LPLB CUDA solver and its runtime "
                "dependencies (torch + tvm-ffi). JIT compilation additionally "
                "requires NVIDIA Math-DX headers: set MATHDX_HOME or install the "
                "nvidia-mathdx package."
            ) from exc
        self._cuda = cuda_solver

    def warmup(self, nc: int, nv: int, num_iters: int, device: Any) -> None:
        self._cuda.warmup(nc=nc, nv=nv, num_iters=num_iters, device=device)

    def warmup_shape(self, **kwargs) -> None:
        self._cuda.warmup_shape(**kwargs)

    def prep_lp_inputs(
        self,
        a_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        b1,
        a_base_row_sum,
    ) -> None:
        self._cuda.prep_lp_inputs(
            a_full,
            b,
            t1,
            global_counts,
            log_single,
            log_replicated,
            b1,
            a_base_row_sum,
        )

    def solve_ipm(self, a, b, c, num_iters: int = 5, result: Any | None = None):
        return self._cuda.solve_ipm(a, b, c, num_iters=num_iters, result=result)

    def extract_log2phy_prob(
        self,
        log2phy_prob,
        x,
        t1,
        phy_single,
        phy_replicated,
        log2phy,
    ) -> None:
        self._cuda.extract_log2phy_prob(
            log2phy_prob,
            x,
            t1,
            phy_single,
            phy_replicated,
            log2phy,
        )

    def dispatch_probability(self, topk_ids, log2phy_prob, log2phy_map, **kwargs):
        return self._cuda.dispatch_probability(
            topk_ids, log2phy_prob, log2phy_map, **kwargs
        )
