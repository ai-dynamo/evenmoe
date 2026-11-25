# SPDX-License-Identifier: Apache-2.0
"""Pilot-only adapter of xutizhou tilewf for fixed MegaMoE EPLB placement."""

from types import SimpleNamespace

import torch

from .upstream_tilewf import _State

_STATE = None


def route(state, ids, weights, token_count, scale):
    global _STATE
    if _STATE is None:
        _STATE = _State(group=state.group, me=state.rank, R=8)
        _STATE.ensure_buf(state.physical)
    assert _STATE.group is state.group and state.physical == 288
    n = ids.shape[0]
    tk = ids.to(torch.int32).contiguous()
    tw = weights.float().contiguous()
    mapping = state.placement.default_physical_for_logical.to(torch.int64).contiguous()
    nv = (
        token_count.to(torch.int32).reshape(-1)[:1]
        if isinstance(token_count, torch.Tensor)
        else _STATE.e32
    )
    if isinstance(token_count, int) and token_count != n:
        nv = torch.tensor([token_count], dtype=torch.int32, device=ids.device)
    result_ids = torch.empty((n, 9), dtype=torch.int64, device=ids.device)
    result_w = torch.empty((n, 9), dtype=torch.float32, device=ids.device)
    _STATE.call(tk, tw, nv, mapping, 256, 288, 36, 1.0 / scale, result_ids, result_w)
    return SimpleNamespace(topk_ids=result_ids, topk_weights=result_w)
