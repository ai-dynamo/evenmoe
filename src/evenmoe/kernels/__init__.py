# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral tensor operators and accelerated kernel backends.

Callable tensor operators live in :mod:`evenmoe.kernels.ops`.
Algorithm-specific backends live in subpackages such as ``lplb`` and
``ultraep``. Shared build and loader infrastructure lives in
:mod:`evenmoe.kernels.jit`.

Nothing in this package imports an inference framework. Vendored C++ /
CUDA sources are self-contained: framework-specific glue (Protocol
implementations, runtime hooks) lives in separate adapter packages under
:mod:`evenmoe.adapters`.
"""
