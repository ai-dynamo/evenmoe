# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor operators that turn a replica split into per-token assignments.

A policy decides how an expert's traffic divides across its replicas; which
token lands on which one is the step that realises that decision. Drawing it
from an RNG makes the realisation neither reproducible nor agreed upon: ranks
that hold different experts but see the same tokens -- tensor-parallel peers,
whose router output is replicated -- each draw their own numbers and can send
one token to two different replicas of its expert.

A hash of the token index has neither problem and costs the same. It is also
the draw a fused runtime uses when it applies a share table inside its own
kernel, so a decision realises identically wherever it is executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# floor(2^32 / phi) -- the classic Knuth multiplicative hash constant. Shared
# with the framework-side kernels on purpose: the value is part of the
# agreement, not an implementation detail either side may pick.
KNUTH_MULTIPLIER = 2654435769
_UINT32 = 4294967296.0

_CACHE: dict = {}


def token_hash(token_idx):
    """32-bit hash of a token index, as an integer tensor."""
    return (token_idx.to("cpu" if token_idx.device.type == "meta" else token_idx.device)
            * KNUTH_MULTIPLIER) & 0xFFFFFFFF


def uniform_draw(n: int, topk: int, device) -> torch.Tensor:
    """``n`` draws in ``[0, 1)``, one per routed slot, constant within a token.

    Cached by shape: it depends on nothing else, and rebuilding it every
    forward costs more than the dispatch it feeds.
    """
    import torch

    key = (n, topk, str(device))
    draw = _CACHE.get(key)
    if draw is None:
        slot = torch.arange(n, device=device, dtype=torch.int64)
        hashed = (slot // max(topk, 1) * KNUTH_MULTIPLIER) & 0xFFFFFFFF
        draw = (hashed.to(torch.float32) / _UINT32).contiguous()
        _CACHE[key] = draw
    return draw


def replica_index(shape, topk: int, counts, device) -> torch.Tensor:
    """Which replica each routed slot takes, from the token index alone.

    ``counts`` is the live replica count per slot, so an expert with one copy
    always yields 0 and the caller needs no separate guard.
    """
    import torch

    n = 1
    for d in shape:
        n *= d
    slot = torch.arange(n, device=device, dtype=torch.int64).reshape(shape)
    hashed = (slot // max(topk, 1) * KNUTH_MULTIPLIER) & 0xFFFFFFFF
    return (hashed % counts.clamp(min=1).to(torch.int64)).to(torch.int32)
