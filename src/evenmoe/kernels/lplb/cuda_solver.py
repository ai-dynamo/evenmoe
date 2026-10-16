# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JIT-compiled CUDA Interior Point Method LP solver.

Framework-neutral; framework-specific glue (a ``LPLBKernels`` Protocol
impl + the runtime hook) lives separately under
``evenmoe.adapters.<framework>.lplb``.

Layout:

- ``csrc/``   CUDA kernel source files (``.cuh``). The ``evenmoe_jit/*`` helper
              headers they include (TensorView validation chain, stream-aware
              ``cudaLaunchKernelEx`` wrapper, etc.; adapted from SGLang's
              jit_kernel) live in ``kernels/jit/include/`` and reach every
              build through ``load_jit``.

NVIDIA Math-DX headers are resolved from MATHDX_HOME or nvidia-mathdx.

Per-call CPU overhead is dominated by the pybind11 dispatch + four
``data_ptr()`` calls (~5–10 µs total).
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import torch

from evenmoe.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
    make_mathdx_include_paths_resolver,
    register_dependency,
)
from evenmoe.kernels.ops.replica_draw import uniform_draw

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_DIM = 256
# Per-element kernels (post-LP dispatch) saturate easily — also 256.
DISPATCH_BLOCK_DIM = 256
DEFAULT_NUM_ITERS = 5

# Kernel-source root. ``load_jit`` resolves ``cuda_files`` paths against
# ``_HERE / "csrc" / <file>`` (the ``.cuh`` files live in a flat ``csrc/``
# layout) and supplies the shared ``evenmoe_jit/*`` headers itself, so the only
# external C++ header dependencies are tvm-ffi and Math-DX.
_HERE = pathlib.Path(__file__).parent.resolve()

# Resolve Math-DX from MATHDX_HOME or the installed nvidia-mathdx package.
register_dependency("mathdx")(make_mathdx_include_paths_resolver())


def _sm_ver() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 100 + arch.minor * 10


@cache_once
def _ipm_module(nc: int, nv: int, block_dim: int, num_iters: int, sm_ver: int) -> Module:
    """JIT-compile the IPM kernel for one shape. Cached for the process lifetime."""
    args = make_cpp_args(nc, nv, block_dim, sm_ver, num_iters)
    # The kernel uses cuBLASDx (header-only) for the GEMMs and a hand-written
    # block-level Cholesky for the POSV. No -rdc=true / static-lib linkage
    # required, so the SDK's tvm-ffi load_jit handles the build with the
    # default flags.
    return load_jit(
        "lplb_ipm",
        *args,
        cuda_files=["ipm.cuh"],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{args}>")],
        extra_dependencies=["mathdx"],
        source_root=_HERE,
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:
    """JIT-compile the kernel for ``(nc, nv)`` so the first real solve isn't
    paying the compile cost. Raises on compile or launch failure.
    """
    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    # Trigger any first-call lazy initialization.
    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    result = torch.empty(nv, dtype=torch.float32, device=device)
    module.ipm_solve(A, b, c, result)
    logger.info(f"LPLB CUDA IPM solver: warmed up for (NC={nc}, NV={nv})")


def warmup_shape(
    *,
    nc: int,
    nv: int,
    num_single: int,
    num_red_log: int,
    num_gpus: int,
    num_logical: int,
    max_copies: int,
    num_red_phy: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:
    """Compile all three kernels for one LP shape.

    ``warmup`` covers the IPM alone, but a shape needs three modules and the
    other two are compiled lazily on first use -- inside a forward, on the
    serving path. The prep module is in fact the most prolific of the three
    (53 cached shapes against the IPM's 41 on this machine), so warming only
    the solver leaves most of the cost exactly where it hurts.
    """
    warmup(nc=nc, nv=nv, num_iters=num_iters, device=device)
    # @cache_once, so building the module is the compile; nothing to launch.
    _prep_module(nc, nv, num_single, num_red_log, num_gpus, DEFAULT_BLOCK_DIM)
    _post_module(num_logical, max_copies, num_single, num_red_phy, DEFAULT_BLOCK_DIM)


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused single-SM IPM kernel.

    cuBLASDx GEMMs + hand-written block Cholesky, dispatched per the
    module docstring.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on CUDA.
        b: RHS vector, shape ``(NC,)``, float32, on CUDA.
        c: Objective coefficients, shape ``(NV,)``, float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).
        result: Optional pre-allocated ``(NV,)`` float32 CUDA buffer to write
            into. When omitted the kernel allocates a fresh result tensor
            (~20 µs of CPU overhead). Passing in a long-lived buffer skips
            that alloc on every solve.

    Returns:
        x: Solution vector, shape ``(NV,)``, float32. The kernel writes 0.5
        for every entry on non-convergence (matches the prior Numba behavior).
    """
    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    module.ipm_solve(A, b, c, result)
    return result


@cache_once
def _prep_module(
    nc: int,
    nv: int,
    num_single: int,
    num_red_log: int,
    num_gpus: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(nc, nv, num_single, num_red_log, num_gpus, block_dim)
    return load_jit(
        "lplb_lp_prep",
        *args,
        cuda_files=["lp_prep.cuh"],
        cuda_wrappers=[("lp_prep", f"lp_prep<{args}>")],
        source_root=_HERE,
    )


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:
    """Replace the 8 torch ops that built the IPM inputs with one CUDA kernel.

    Writes into the caller-provided ``A_full`` (last column only), ``b``,
    and ``t1`` buffers. ``A_full``'s first ``NV-1`` columns must already
    hold ``A_base.copy_()`` from solver init — this kernel does not touch
    them.
    """
    nc, nv = A_full.shape
    num_single = log_single.shape[0]
    num_red_log = log_replicated.shape[0]
    num_gpus, _ns = B1.shape
    module = _prep_module(nc, nv, num_single, num_red_log, num_gpus, DEFAULT_BLOCK_DIM)
    module.lp_prep(A_full, b, t1, global_counts, log_single, log_replicated, B1, A_base_row_sum)


@cache_once
def _post_module(
    num_logical: int,
    max_copies: int,
    num_single: int,
    num_red_phy: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(num_logical, max_copies, num_single, num_red_phy, block_dim)
    return load_jit(
        "lplb_lp_post",
        *args,
        cuda_files=["lp_post.cuh"],
        cuda_wrappers=[("lp_post", f"lp_post<{args}>")],
        source_root=_HERE,
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:
    """Replace the 5 torch ops that turned the IPM output into log2phy_prob
    with one CUDA kernel. Writes into the caller-provided ``log2phy_prob``
    buffer of shape ``(num_logical, max_copies)``.
    """
    num_logical, max_copies = log2phy_prob.shape
    num_single = phy_single.shape[0]
    num_red_phy = phy_replicated.shape[0]
    module = _post_module(num_logical, max_copies, num_single, num_red_phy, DEFAULT_BLOCK_DIM)
    module.lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy)


@cache_once
def _dispatch_module(max_copies: int, block_dim: int) -> Module:
    args = make_cpp_args(max_copies, block_dim)
    return load_jit(
        "lplb_dispatch_probability",
        *args,
        cuda_files=["dispatch_probability.cuh"],
        cuda_wrappers=[("dispatch_probability", f"dispatch_probability<{args}>")],
        source_root=_HERE,
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    expert_load_view: torch.Tensor | None = None,
    record_enabled: torch.Tensor | None = None,
    num_unpadded_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace the 7 torch ops in `_topk_ids_logical_to_physical_probability`
    with a single per-token-per-slot CUDA kernel.

    Args:
        topk_ids: (num_tokens, topk) int32, on CUDA. Logical expert ids from
            the router.
        log2phy_prob: (num_logical, max_copies) float32. LP solver output.
        log2phy_map: (num_logical, max_copies) int32. -1 entries are
            unused replicas; treated as 0-weight in the multinomial.

    Returns:
        Physical topk ids tensor with the same shape as ``topk_ids``.
    """
    original_shape = topk_ids.shape
    topk = original_shape[-1]
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    n = flat_ids.shape[0]
    # A None share table selects the uniform/modulo mode, which is how a
    # policy that just wants "one replica, evenly" gets the framework's own
    # choice out of the same kernel instead of building the table for it.
    if log2phy_prob is None:
        num_logical, max_copies = log2phy_map.shape
    else:
        num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    if n == 0:
        return torch.empty_like(topk_ids)
    map32 = log2phy_map.contiguous().to(torch.int32)

    out = torch.empty(n, dtype=torch.int32, device=topk_ids.device)
    random_vals = uniform_draw(n, original_shape[-1], topk_ids.device)
    module = _dispatch_module(max_copies, DISPATCH_BLOCK_DIM)
    empty = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=topk_ids.device)

    module.dispatch_probability(

        out, flat_ids,
        log2phy_prob if log2phy_prob is not None else empty_f32,
        map32, random_vals,

        expert_load_view if expert_load_view is not None else empty,

        record_enabled if record_enabled is not None else empty,

        num_unpadded_tokens if num_unpadded_tokens is not None else empty,

        topk,

    )
    return out.view(original_shape).to(topk_ids.dtype)
