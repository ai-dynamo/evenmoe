# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LPLB CUDA kernel pipeline — framework-neutral.

This package owns the LPLB Linear-Programming load-balancer kernels:

- ``cuda_solver``           four JIT-compiled CUDA modules (IPM + LP prep /
                             post / dispatch). Build via
                             ``evenmoe.kernels.jit.utils``.
- ``protocol``              execution contract consumed by the LPLB policy.
- ``csrc/``                  CUDA kernel sources (header-only ``.cuh``). The
                             ``evenmoe_jit/*`` helper headers they include
                             (TensorView validation, ``LaunchKernel`` wrapper;
                             adapted from SGLang) live in
                             ``kernels/jit/include/`` and reach every build
                             through ``load_jit``.

The pipeline has no runtime or build-time dependency on any inference
framework. Framework-specific implementations of ``LPLBKernels`` live
separately under ``evenmoe.adapters.<framework>``.
"""

from evenmoe.kernels.lplb.backend import CUDALPLBKernels
from evenmoe.kernels.lplb.protocol import LPLBKernels

__all__ = ["CUDALPLBKernels", "LPLBKernels"]
