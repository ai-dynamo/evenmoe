# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm policy packages.

The L1/L2 sub-packages avoid eager imports so dev environments without the
heavy GPU dependencies (``torch`` / ``triton``) can still consume the
framework-neutral core contracts. Import concrete policies from their
sub-modules directly, e.g.
``from evenmoe.policies.l2.waterfill import WaterfillL2Router``.
"""

__all__: list[str] = []
