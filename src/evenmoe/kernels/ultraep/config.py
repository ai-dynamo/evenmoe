# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for EvenMoE's UltraEP-derived placement algorithm."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _read_bool_env(name: str, default: bool) -> bool:
    value = _read_env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _read_int_env(name: str, default: int) -> int:
    value = _read_env(name)
    return default if value is None else int(value)


def _read_float_env(name: str, default: float) -> float:
    value = _read_env(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class UltraEPPlacementTuning:
    balance_threshold: float
    locality_aware: bool
    min_tokens_per_replica: int
    allow_zero_master_quota: bool
    oracle_eps: float
    kernel_stage: int


def load_placement_tuning_from_env() -> UltraEPPlacementTuning:
    kernel_stage = _read_int_env("ULTRA_EP_QUOTA_KERNEL_STAGE", 1)
    if kernel_stage not in (0, 1):
        raise ValueError("ULTRA_EP_QUOTA_KERNEL_STAGE supports only 0 or 1")

    min_tokens = _read_int_env("ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA", 1024)
    if min_tokens < 0:
        raise ValueError("ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA must be non-negative")

    return UltraEPPlacementTuning(
        balance_threshold=_read_float_env("ULTRA_EP_BALANCE_THRESHOLD", 1.0),
        locality_aware=_read_bool_env("ULTRA_EP_QUOTA_LOCALITY_AWARE", True),
        min_tokens_per_replica=min_tokens,
        allow_zero_master_quota=_read_bool_env("ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA", False),
        oracle_eps=_read_float_env("ULTRA_EP_QUOTA_ORACLE_EPS", 0.01),
        kernel_stage=kernel_stage,
    )
