# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVLink peer-exchange buffers for an all-gather inside a routing kernel (one NVLink domain).

Host side only: one receive buffer per EP rank in torch symmetric memory,
allocated and rendezvoused once on the EP process group. The kernel side
(``kernels/tile_waterfill``) implements the protocol: every int32 word is
``(epoch16 << 16) | value16``; a buffer holds ``2 * world_size * width`` words
(two halves selected by the parity of ``epoch``, a ``width``-word slot per
source rank), so a fast peer writing call ``e + 1`` never overwrites call
``e``. A late peer (``timeout_ns``), a peer already past the call or a value of
0xFFFF or more fail the call on that rank; the kernel then takes its
always-valid fallback and still advances ``epoch``. Every EP rank must make
every call, in the same order, and calls must not overlap. A rank that skips a
call cannot hang the others (they wait at most ``timeout_ns``), but the
counters stay out of step, so after a few consecutive failures a process
stops exchanging and keeps its rows local.
"""

from __future__ import annotations

import logging
import socket
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 2000.0
# What a missing or unusable symmetric-memory backend raises across torch versions.
_SETUP_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError)


@dataclass
class PeerExchange:
    """One rank's view: ``peer_buffers[p]`` is rank ``p``'s receive buffer (device address)."""

    rank: int
    world_size: int
    peer_buffers: Tensor
    buffer_words: int
    state: Tensor  # int32 [2]: epoch of the last call, consecutive failed calls
    timeout_ns: int
    _keepalive: tuple[Any, ...] = field(default=(), repr=False)

    @classmethod
    def create(cls, rank: int, world_size: int, pointers: list[int], words: int, device, timeout_ms: float, keepalive):
        return cls(
            rank=rank,
            world_size=world_size,
            peer_buffers=torch.tensor(pointers, dtype=torch.int64, device=device),
            buffer_words=words,
            state=torch.zeros(2, dtype=torch.int32, device=device),
            timeout_ns=int(timeout_ms * 1e6),
            _keepalive=keepalive,
        )


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


def create_peer_exchange(group: Any, *, width: int, timeout_ms: float = DEFAULT_TIMEOUT_MS) -> PeerExchange | None:
    """Collective over the EP ``group`` (eagerly, outside graph capture): buffers for ``width``-word rows.

    Returns None on every rank when P2P is not usable -- ranks on more than one
    host or no torch symmetric memory -- so all ranks fall back together.
    """

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    collective_device = device if "nccl" in str(dist.get_backend(group)).lower() else torch.device("cpu")
    hosts = torch.empty(world_size, dtype=torch.int64, device=collective_device)
    dist.all_gather_into_tensor(hosts, torch.tensor([_host_key()], dtype=torch.int64, device=collective_device), group=group)
    reason = "the EP group spans more than one host" if bool((hosts != hosts[0]).any()) else ""
    words = 2 * world_size * int(width)
    buffer: Tensor | None = None
    handle: Any = None
    symm_mem: Any = None
    if not reason:
        try:
            import torch.distributed._symmetric_memory as symm_mem

            buffer = symm_mem.empty(words, dtype=torch.int32, device=device)
        except _SETUP_ERRORS as exc:
            reason = f"symmetric memory unavailable ({exc})"
    # Agree before the rendezvous (a collective) and again after it.
    for phase in ("setup", "rendezvous"):
        if not _all_ranks_agree(not reason, group, collective_device):
            logger.info("P2P peer exchange disabled: %s", reason or "a peer cannot use it")
            return None
        if phase == "rendezvous":
            break
        try:
            enable = getattr(symm_mem, "enable_symm_mem_for_group", None)
            group_name = getattr(group, "group_name", None)
            if enable is not None and group_name is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)  # deprecated no-op on newer torch
                    try:
                        enable(group_name)
                    except _SETUP_ERRORS as exc:
                        logger.debug("enable_symm_mem_for_group(%s): %s", group_name, exc)
            handle = symm_mem.rendezvous(buffer, group)
            if len(handle.buffer_ptrs) != world_size or not all(handle.buffer_ptrs):
                reason = "symmetric-memory rendezvous returned unusable peer pointers"
        except _SETUP_ERRORS as exc:
            reason = f"symmetric-memory rendezvous failed ({exc})"
    assert buffer is not None and handle is not None
    buffer.zero_()  # stale words could carry a matching tag
    torch.cuda.synchronize(device)
    _all_ranks_agree(True, group, collective_device)
    return PeerExchange.create(
        rank, world_size, [int(p) for p in handle.buffer_ptrs], words, device, float(timeout_ms), (buffer, handle)
    )
