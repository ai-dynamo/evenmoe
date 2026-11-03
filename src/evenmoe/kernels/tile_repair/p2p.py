# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused NVLink P2P count exchange for tile_repair (one NVLink domain).

The gathered-count path costs one framework all-reduce of an ``[R, L]``
payload per MoE layer (~15 us on 4-8 GPUs with NCCL, plus the count and the
route launches). Within one NVLink domain the same counts can be exchanged
inside the routing launch itself: :func:`tile_repair_route_p2p` histograms
this rank's ids, stores one self-validating word ``(epoch16 << 16) | count16``
per expert into every peer's receive buffer, polls its own buffer until all
peers' words of the current call have landed, then plans and rewrites exactly
like :func:`~evenmoe.kernels.tile_repair.tile_repair_route` (bit-identical
output for the same counts).

Requirements and guarantees:

- Every rank must call it once per layer with replicated experts, in the same
  order (the epoch is a per-process call counter shared by all layers). Ranks
  with no tokens call it with an empty ``[0, K]`` batch.
- The receive buffers live in torch symmetric memory on the EP process group
  (:func:`create_peer_exchange`, a one-time collective at model setup, never
  inside CUDA-graph capture). Multi-node groups or a missing symmetric-memory
  backend make every rank get ``None`` (callers keep the all-reduce path).
- A late peer (timeout, ``EVENMOE_TILE_P2P_TIMEOUT_MS``, default 2000), a peer
  already past this call, or a per-expert count of 0xFFFF or more makes this
  rank route the call primary-only (valid physical ids, no balancing); the
  epoch still advances so the next call is in step. ``exchange.failures``
  counts such calls on device.
"""

from __future__ import annotations

import logging
import os
import socket
import zlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from evenmoe.kernels.tile_repair import (
    MAX_LOGICAL,
    MAX_RANKS,
    TileRepairTables,
    _check_route_args,
    _route_module,
)

logger = logging.getLogger(__name__)

TIMEOUT_ENV = "EVENMOE_TILE_P2P_TIMEOUT_MS"
DEFAULT_TIMEOUT_MS = 2000.0


def resolve_timeout_ms(value: float | str | None = None) -> float:
    """Exchange timeout in milliseconds: argument, then environment, then 2000."""

    if value is None:
        env = os.environ.get(TIMEOUT_ENV, "").strip()
        value = env if env else DEFAULT_TIMEOUT_MS
    timeout = float(value)
    if not timeout > 0:
        raise ValueError(f"P2P exchange timeout must be positive, got {timeout} ms.")
    return timeout


@dataclass
class TileRepairPeerExchange:
    """One rank's view of the P2P count exchange.

    ``peer_buffers[p]`` is the device address (in this process) of rank ``p``'s
    receive buffer of ``buffer_words`` int32; ``epoch`` / ``failures`` are this
    rank's device scalars. Keep the object alive as long as kernels may run.
    """

    rank: int
    world_size: int
    peer_buffers: Tensor
    buffer_words: int
    epoch: Tensor
    failures: Tensor
    timeout_ns: int
    _keepalive: tuple[Any, ...] = field(default=(), repr=False)

    @property
    def device(self) -> torch.device:
        return self.peer_buffers.device

    def failure_count(self) -> int:
        """Calls this rank routed primary-only after a failed exchange (host sync)."""

        return int(self.failures.item())


def _new_exchange(
    *,
    rank: int,
    world_size: int,
    pointers: list[int],
    buffer_words: int,
    device: torch.device,
    timeout_ms: float,
    keepalive: tuple[Any, ...],
) -> TileRepairPeerExchange:
    return TileRepairPeerExchange(
        rank=rank,
        world_size=world_size,
        peer_buffers=torch.tensor(pointers, dtype=torch.int64, device=device),
        buffer_words=buffer_words,
        epoch=torch.zeros(1, dtype=torch.int32, device=device),
        failures=torch.zeros(1, dtype=torch.int32, device=device),
        timeout_ns=int(timeout_ms * 1e6),
        _keepalive=keepalive,
    )


def local_peer_exchanges(
    num_ranks: int,
    *,
    device: torch.device | str = "cuda",
    max_logical: int = MAX_LOGICAL,
    timeout_ms: float | None = None,
) -> list[TileRepairPeerExchange]:
    """Single-process stand-in: every rank's receive buffer on one device.

    For tests and microbenchmarks: launch rank ``r``'s call with exchange
    ``r`` on its own stream; the calls of one layer must be able to run
    concurrently (one CTA each).
    """

    if not 0 < num_ranks <= MAX_RANKS:
        raise ValueError(f"num_ranks must lie in [1, {MAX_RANKS}].")
    device = torch.device(device)
    words = 2 * num_ranks * max_logical
    buffers = torch.zeros(num_ranks, words, dtype=torch.int32, device=device)
    pointers = [int(buffers[r].data_ptr()) for r in range(num_ranks)]
    timeout = resolve_timeout_ms(timeout_ms)
    return [
        _new_exchange(
            rank=r,
            world_size=num_ranks,
            pointers=pointers,
            buffer_words=words,
            device=device,
            timeout_ms=timeout,
            keepalive=(buffers,),
        )
        for r in range(num_ranks)
    ]


def _host_key() -> int:
    key = socket.gethostname()
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as handle:
            key += "/" + handle.read().strip()
    except OSError:
        pass
    return zlib.crc32(key.encode())


def _all_ranks_agree(flag: bool, group: Any, device: torch.device) -> bool:
    import torch.distributed as dist

    value = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    return bool(value.item())


def create_peer_exchange(
    group: Any,
    *,
    device: torch.device | str | None = None,
    max_logical: int = MAX_LOGICAL,
    timeout_ms: float | None = None,
) -> TileRepairPeerExchange | None:
    """Allocate and rendezvous the receive buffers on the EP process group.

    Collective over ``group`` (a ``torch.distributed`` process group whose
    rank order is the EP rank order): call it on every rank, eagerly, outside
    CUDA-graph capture. Returns ``None`` on every rank when the P2P path is not
    usable -- ranks on more than one host, more than :data:`MAX_RANKS` ranks,
    or torch symmetric memory unavailable -- so all ranks make the same choice.
    """

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    backend = str(dist.get_backend(group)).lower()
    collective_device = device if "nccl" in backend else torch.device("cpu")
    timeout = resolve_timeout_ms(timeout_ms)

    reason = ""
    hosts = torch.empty(world_size, dtype=torch.int64, device=collective_device)
    dist.all_gather_into_tensor(
        hosts, torch.tensor([_host_key()], dtype=torch.int64, device=collective_device), group=group
    )
    if bool((hosts != hosts[0]).any()):
        reason = "the EP group spans more than one host"
    elif not 0 < world_size <= MAX_RANKS:
        reason = f"{world_size} ranks exceed {MAX_RANKS}"

    words = 2 * world_size * max_logical
    buffer: Tensor | None = None
    symm_mem: Any = None
    if not reason:
        try:
            import torch.distributed._symmetric_memory as symm_mem

            buffer = symm_mem.empty(words, dtype=torch.int32, device=device)
        except Exception as exc:  # backend missing, old torch, ...
            reason = f"symmetric memory unavailable ({exc})"
    if not _all_ranks_agree(not reason, group, collective_device):
        logger.info("tile_repair P2P exchange disabled: %s", reason or "a peer cannot use it")
        return None

    pointers: list[int] = []
    handle: Any = None
    try:
        group_name = getattr(group, "group_name", None)
        enable = getattr(symm_mem, "enable_symm_mem_for_group", None)
        if enable is not None and group_name is not None:
            try:
                enable(group_name)
            except Exception:  # newer torch enables every group already
                pass
        handle = symm_mem.rendezvous(buffer, group)
        pointers = [int(p) for p in handle.buffer_ptrs]
        if len(pointers) != world_size or not all(pointers):
            reason = "symmetric-memory rendezvous returned unusable peer pointers"
    except Exception as exc:
        reason = f"symmetric-memory rendezvous failed ({exc})"
    if not _all_ranks_agree(not reason, group, collective_device):
        logger.info("tile_repair P2P exchange disabled: %s", reason or "a peer cannot use it")
        return None

    assert buffer is not None
    # Stale words could carry a matching tag: zero every buffer before any
    # peer can write into it.
    buffer.zero_()
    torch.cuda.synchronize(device)
    _all_ranks_agree(True, group, collective_device)
    logger.info(
        "tile_repair P2P exchange ready: rank %d/%d, %d words per buffer, timeout %.0f ms",
        rank,
        world_size,
        words,
        timeout,
    )
    return _new_exchange(
        rank=rank,
        world_size=world_size,
        pointers=pointers,
        buffer_words=words,
        device=device,
        timeout_ms=timeout,
        keepalive=(buffer, handle),
    )


def tile_repair_route_p2p(
    logical_topk_ids: Tensor,
    tables: TileRepairTables,
    exchange: TileRepairPeerExchange,
    *,
    block_m: int,
    max_moves: int = 32,
    num_slots: int | None = None,
    out_dtype: torch.dtype = torch.int64,
    plan_out: Tensor | None = None,
    stats_out: Tensor | None = None,
) -> Tensor:
    """Count, exchange over NVLink, plan and rewrite in one launch.

    Same output as :func:`~evenmoe.kernels.tile_repair.tile_repair_route` with
    the EP-wide per-source counts of this call (or primary-only after a failed
    exchange; ``stats_out[5]`` then holds the failure bits). ``max_moves=0``
    still exchanges (keeps peers in step) and routes primary-only.
    """

    _check_route_args(
        logical_topk_ids,
        tables,
        block_m=block_m,
        max_moves=max_moves,
        num_slots=num_slots,
        out_dtype=out_dtype,
    )
    if exchange.world_size != tables.num_ranks:
        raise ValueError(
            f"Exchange spans {exchange.world_size} ranks, tables {tables.num_ranks}."
        )
    if exchange.buffer_words < 2 * tables.num_ranks * tables.num_logical:
        raise ValueError("P2P receive buffers are too small for this layer.")
    ids = logical_topk_ids if logical_topk_ids.dim() == 2 else logical_topk_ids.reshape(-1, 1)
    ids = ids.contiguous()
    device = ids.device
    empty = torch.empty(0, dtype=torch.int32, device=device)
    out = torch.empty(ids.shape, dtype=out_dtype, device=device)
    _route_module(ids.dtype, out_dtype).tile_repair_route_p2p(
        ids,
        out,
        exchange.peer_buffers,
        exchange.epoch,
        exchange.failures,
        tables.copy_physical,
        tables.copy_rank,
        tables.num_copies,
        tables.replicated,
        tables.num_replicated,
        empty if plan_out is None else plan_out,
        empty if stats_out is None else stats_out,
        int(exchange.rank),
        int(exchange.buffer_words),
        int(block_m),
        int(num_slots or 0),
        int(max_moves),
        int(exchange.timeout_ns),
    )
    return out.view(logical_topk_ids.shape)


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "TIMEOUT_ENV",
    "TileRepairPeerExchange",
    "create_peer_exchange",
    "local_peer_exchanges",
    "resolve_timeout_ms",
    "tile_repair_route_p2p",
]
