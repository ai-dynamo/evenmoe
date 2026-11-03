# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tunables and tile math of the ``tile`` placement and ``tile_repair`` routing.

Both halves must use the same GEMM tile height ``block_m``: placement packs
experts by *expected* tiles and routing balances *actual* tiles. Every tunable
resolves as policy metadata, then environment variable, then default, so a
framework that only forwards the algorithm name (SGLang's
``--moe-load-balancer-algorithm tile_repair``) is configured with one
environment variable for both halves.

Recommended ``block_m`` (the M tile of the MoE grouped GEMM that runs on the
rank):

==========================================  =========
Backend                                     block_m
==========================================  =========
DeepGEMM SM90 FP8 masked grouped GEMM       64
DeepGEMM SM100 grouped GEMM                 128
swap-AB / small-M grouped GEMM kernels      8, 16, 32 (the kernel's M tile)
Humming / autotuned kernels                 0 (adaptive) or the tile the tuner picks
==========================================  =========

``block_m = 0`` is adaptive: routing picks 8/16/32/48/64 rows per call from
the expected rows per physical slot (sum of the call's routed entries over the
number of physical experts; :func:`evenmoe.kernels.tile_repair.adaptive_block_m`),
and placement picks it once from ``tokens_per_pass``. Use it with GEMMs whose M
tile follows the expected rows (Humming-style small-M tuning); DeepGEMM SM90
always runs 64-row tiles, so keep 64 there.

``p2p`` (``EVENMOE_TILE_P2P``, default off) makes ``tile_repair`` exchange the
per-rank counts over NVLink inside its routing launch instead of through the
framework all-reduce (one NVLink domain; see
:mod:`evenmoe.kernels.tile_repair.p2p`).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from evenmoe.core.types import RoutingStage

BLOCK_M_ENV = "EVENMOE_TILE_BLOCK_M"
MAX_MOVES_ENV = "EVENMOE_TILE_MAX_MOVES"
STAGES_ENV = "EVENMOE_TILE_REPAIR_STAGES"
TOKENS_PER_PASS_ENV = "EVENMOE_TILE_TOKENS_PER_PASS"
P2P_ENV = "EVENMOE_TILE_P2P"

DEFAULT_BLOCK_M = 64
DEFAULT_MAX_MOVES = 32
DEFAULT_PAIRS_PER_RANK = 128
"""Default routed (token, expert) pairs per rank per forward for placement:
16 decode tokens per rank x top-8."""
DEFAULT_STAGES = frozenset(
    {RoutingStage.DECODE, RoutingStage.IDLE, RoutingStage.SPECULATIVE}
)


def _lookup(metadata: Mapping[str, Any] | None, key: str, env: str) -> Any:
    if metadata is not None and metadata.get(key) is not None:
        return metadata[key]
    value = os.environ.get(env)
    return None if value is None or value.strip() == "" else value


def resolve_block_m(metadata: Mapping[str, Any] | None = None) -> int:
    """GEMM tile rows; 0 means adaptive (picked per call / per plan)."""

    value = _lookup(metadata, "block_m", BLOCK_M_ENV)
    block_m = DEFAULT_BLOCK_M if value is None else int(value)
    if block_m < 0:
        raise ValueError(f"block_m must be non-negative (0 = adaptive), got {block_m}.")
    return block_m


def resolve_p2p(metadata: Mapping[str, Any] | None = None) -> bool:
    """Whether tile_repair should try the fused NVLink P2P count exchange."""

    value = _lookup(metadata, "p2p", P2P_ENV)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"p2p must be a boolean, got {value!r}.")


def resolve_max_moves(metadata: Mapping[str, Any] | None = None) -> int:
    value = _lookup(metadata, "max_moves", MAX_MOVES_ENV)
    moves = DEFAULT_MAX_MOVES if value is None else int(value)
    if moves < 0:
        raise ValueError(f"max_moves must be non-negative, got {moves}.")
    return moves


def resolve_tokens_per_pass(metadata: Mapping[str, Any] | None, ep_size: int) -> float:
    value = _lookup(metadata, "tokens_per_pass", TOKENS_PER_PASS_ENV)
    tokens = float(DEFAULT_PAIRS_PER_RANK * ep_size) if value is None else float(value)
    if not tokens > 0:
        raise ValueError(f"tokens_per_pass must be positive, got {tokens}.")
    return tokens


def resolve_stages(metadata: Mapping[str, Any] | None = None) -> frozenset[RoutingStage]:
    """Stages that run the repair; ``"all"`` selects every stage."""

    value = _lookup(metadata, "stages", STAGES_ENV)
    if value is None:
        return DEFAULT_STAGES
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return frozenset(RoutingStage)
        parts = [part for part in value.replace("+", ",").split(",") if part.strip()]
    else:
        parts = list(value)
    stages = frozenset(RoutingStage.from_str(str(part).strip()) for part in parts)
    if not stages:
        raise ValueError("tile_repair stages cannot be empty.")
    return stages


def expected_tiles(mean_count: Tensor, block_m: int) -> Tensor:
    """``E[ceil(K / block_m)]`` for ``K ~ Poisson(mean_count)``, elementwise (float64).

    Uses ``E[ceil(K / b)] = sum_{j >= 0} P(K > j b)`` and
    ``P(K > n) = P(n + 1, mu)`` (the regularised lower incomplete gamma), so
    the cost is ``O(mu / b)`` terms instead of a sum over every ``k``.
    """

    if block_m <= 0:
        raise ValueError("block_m must be positive.")
    mu = torch.as_tensor(mean_count, dtype=torch.float64).clamp(min=0.0)
    out = torch.zeros_like(mu)
    if mu.numel() == 0:
        return out
    peak = float(mu.max())
    if peak <= 0.0:
        return out
    terms = math.ceil((peak + 12.0 * math.sqrt(peak) + 12.0) / block_m) + 1
    for j in range(terms):
        out += torch.special.gammainc(torch.full_like(mu, float(j * block_m + 1)), mu)
    return out


__all__ = [
    "BLOCK_M_ENV",
    "DEFAULT_BLOCK_M",
    "DEFAULT_MAX_MOVES",
    "DEFAULT_PAIRS_PER_RANK",
    "DEFAULT_STAGES",
    "MAX_MOVES_ENV",
    "P2P_ENV",
    "STAGES_ENV",
    "TOKENS_PER_PASS_ENV",
    "expected_tiles",
    "resolve_block_m",
    "resolve_max_moves",
    "resolve_p2p",
    "resolve_stages",
    "resolve_tokens_per_pass",
]
