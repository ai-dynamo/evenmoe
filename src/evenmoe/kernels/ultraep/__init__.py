# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UltraEP placement and quota-reroute kernels -- framework-neutral.

This package owns the two CUDA kernels behind EvenMoE's ``ultraep`` policy pair:

- ``cuda_placement``   two JIT-compiled CUDA modules (placement solve, sparse
                        quota reroute). Built via
                        ``evenmoe.kernels.jit.utils``.
- ``protocol``         execution contract consumed by the UltraEP policies.
- ``backend``          the CUDA implementation of that contract.
- ``config``           the ``ULTRA_EP_*`` tuning the solve takes.
- ``csrc/``            CUDA kernel sources (header-only ``.cuh``), adapted from
                        UltraEP v1.0.0 under its MIT licence
                        (``LICENSE.ultraep``).

It contains no UltraEP communication runtime, and EvenMoE takes no dependency on
the ``ultra_ep`` package: these kernels only compute. Making memory match a
placement they solved is the serving framework's job, done with the mover it
already has, driven by the before/after maps the solve returns.
"""

from evenmoe.kernels.ultraep.backend import UltraEPPlacementBackend
from evenmoe.kernels.ultraep.config import (
    UltraEPPlacementTuning,
    load_placement_tuning_from_env,
)
from evenmoe.kernels.ultraep.protocol import UltraEPKernels

__all__ = [
    "UltraEPKernels",
    "UltraEPPlacementBackend",
    "UltraEPPlacementTuning",
    "load_placement_tuning_from_env",
]
