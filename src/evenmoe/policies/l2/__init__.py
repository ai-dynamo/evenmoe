# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 routing policy implementations live here.

The actual ``WaterfillL2Router`` import lives at
``evenmoe.policies.l2.waterfill`` and requires ``torch`` and
``triton``. We avoid eagerly importing it here so dev environments without
``triton`` can still consume the framework-neutral core contracts.
"""

__all__: list[str] = []
