# SPDX-License-Identifier: Apache-2.0
"""Collective-safe Decode selection; no framework imports."""


def is_collective_decode(batch):
    """DP attention's global_forward_mode is optional (TBO-only metadata).

    A locally idle rank must join decode collectives. is_extend_in_batch is
    synchronized across DP ranks and keeps *every* rank on the original route
    when any rank is prefilling, including locally decoding/idle ranks.
    """
    if batch is None or batch.is_extend_in_batch:
        return False
    mode = batch.global_forward_mode
    if mode is None:
        mode = batch.forward_mode
    return mode.is_decode_or_idle()
