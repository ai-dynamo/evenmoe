# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waterfill tile-mode kernels (the algorithm is in ``csrc/tile_waterfill.cuh``):
K1 counts and exchanges the per-expert rows, K2 plans and assigns."""

from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evenmoe.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from evenmoe.kernels.p2p import PeerExchange

_HERE = pathlib.Path(__file__).parent.resolve()

MAX_RANKS = 32  # one warp lane per rank in the plan


def payload_width(num_physical: int) -> int:
    """Words of one count row: P counts, nv, tile rows, rounded up to 4."""
    return (int(num_physical) + 2 + 3) // 4 * 4


@functools.cache
def _pdl(device_index: int) -> int:
    return int(torch.cuda.get_device_capability(device_index)[0] >= 9)


@cache_once
def _count_exchange_module(id_dtype: torch.dtype) -> Module:
    args = make_cpp_args(id_dtype)
    return load_jit(
        "tile_waterfill_k1",
        *args,
        cuda_files=["tile_waterfill.cuh"],
        cuda_wrappers=[("twf_count_exchange", f"twf_count_exchange<{args}>")],
        source_root=_HERE,
    )


@cache_once
def _plan_assign_module(out_dtype: torch.dtype) -> Module:
    args = make_cpp_args(out_dtype)
    return load_jit(
        "tile_waterfill_k2",
        *args,
        cuda_files=["tile_waterfill.cuh"],
        cuda_wrappers=[("twf_plan_assign", f"twf_plan_assign<{args}>")],
        source_root=_HERE,
    )


def count_and_exchange(
    physical_ids: Tensor,
    num_physical: int,
    token_count: Tensor | int | None,
    tile_rows: int,
    exchange: PeerExchange | None = None,
) -> tuple[Tensor, Tensor | None]:
    """K1: ``(gathered [R, W], status [1])`` with a P2P exchange, else ``(this rank's row [W], None)``.

    ``physical_ids`` is the ``[num_tokens, top_k]`` routed TopK; ``token_count``
    (rows that are not padding) is a device tensor, an int, or None (all).
    """

    device = physical_ids.device
    ids = physical_ids.contiguous() if physical_ids.numel() else torch.empty_like(physical_ids, memory_format=torch.contiguous_format)
    if isinstance(token_count, Tensor):
        nv_tensor, nv_value = token_count.reshape(-1)[:1].to(device=device, dtype=torch.int32), 0
    else:
        nv_tensor = torch.empty(0, dtype=torch.int32, device=device)
        nv_value = ids.shape[0] if token_count is None else int(token_count)
    width = payload_width(num_physical)
    none32 = torch.empty(0, dtype=torch.int32, device=device)
    if exchange is not None:
        out = torch.empty((exchange.world_size, width), dtype=torch.int32, device=device)
        status = torch.empty(1, dtype=torch.int32, device=device)
        tensors = (none32, out, exchange.peer_buffers, exchange.state, status)
        scalars = (exchange.world_size, exchange.rank, exchange.buffer_words, exchange.timeout_ns)
    else:
        out, status = torch.empty(width, dtype=torch.int32, device=device), None
        tensors = (out, none32.view(0, 1), torch.empty(0, dtype=torch.int64, device=device), none32, none32)
        scalars = (1, 0, 0, 1)
    _count_exchange_module(ids.dtype).twf_count_exchange(
        ids, nv_tensor, *tensors, nv_value, int(num_physical), int(tile_rows), *scalars, _pdl(device.index)
    )
    return out, status


def plan_and_assign(
    gathered: Tensor,
    *,
    num_local_physical: int,
    rank: int,
    num_tokens: int,
    rho: float,
    status: Tensor | None = None,
    out_dtype: torch.dtype = torch.int64,
) -> Tensor:
    """K2: the shared-expert destination rank of each of this rank's ``num_tokens`` rows (-1 past nv)."""

    device = gathered.device
    shared_rank = torch.empty(num_tokens, dtype=out_dtype, device=device)
    _plan_assign_module(out_dtype).twf_plan_assign(
        gathered.to(torch.int32).contiguous(),
        torch.empty(0, dtype=torch.int32, device=device) if status is None else status,
        shared_rank,
        gathered.shape[0] * int(num_local_physical),
        int(num_local_physical),
        round(4 * rho),
        int(rank),
        _pdl(device.index),
    )
    return shared_rank
