# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host mirror of Waterfill's tile-mode kernels (csrc/tile_waterfill.cuh), bit-exact, for the tests."""

import torch

from evenmoe.kernels.p2p import PeerExchange
from evenmoe.kernels.tile_waterfill import payload_width

COST_LIMIT = 1 << 25


def count_rows(physical_ids, num_physical, token_count, tile_rows):
    """Payload row ``[W]`` int32: rows of the first ``nv`` tokens per physical expert, nv, tile rows."""

    ids = physical_ids.detach().cpu().to(torch.int64)
    ids = ids if ids.dim() == 2 else ids.reshape(-1, 1)
    if token_count is None:
        nv = ids.shape[0]
    elif isinstance(token_count, torch.Tensor):
        nv = int(token_count.reshape(-1)[0]) if token_count.numel() else ids.shape[0]
    else:
        nv = int(token_count)
    nv = max(0, min(nv, ids.shape[0]))
    flat = ids[:nv].reshape(-1)
    flat = flat[(flat >= 0) & (flat < num_physical)]
    out = torch.zeros(payload_width(num_physical), dtype=torch.int32)
    out[:num_physical] = torch.bincount(flat, minlength=num_physical).to(torch.int32)
    out[num_physical] = nv
    out[num_physical + 1] = int(tile_rows)
    return out


def plan(gathered, *, num_physical, num_local_physical, rank, rho=1.5, status=0):
    """Plan ``[5, R]`` int32: quota, nv, routed tiles, routed rows, tile rows."""

    g = gathered.detach().cpu().to(torch.int64)
    ranks, per = g.shape[0], int(num_local_physical)
    assert num_physical == ranks * per
    nv = g[:, num_physical].tolist()
    reported = [int(t) for t, n in zip(g[:, num_physical + 1].tolist(), nv) if n > 0 and t > 0]
    tile = max(int(g[rank, num_physical + 1]), 1) if status != 0 or not reported else max(reported)
    totals = g[:, :num_physical].sum(0).reshape(ranks, per)
    tiles = (-(-totals // tile)).sum(1).tolist()
    rows = totals.sum(1).tolist()
    routed_tiles, routed_rows = list(tiles), list(rows)
    max_tiles, max_rows = max(tiles), max(rows)
    num_full, rem = divmod(sum(nv), tile)
    wt, wr = 4 * tile, round(4 * rho)
    inc = wt + tile * wr
    cost = [wt * tiles[r] + wr * rows[r] for r in range(ranks)]
    if status != 0 or any(c + (num_full + 1) * inc >= COST_LIMIT or n < 0 for c, n in zip(cost, nv)):
        quota = list(nv)
    else:
        quota = [0] * ranks
        for q in [tile] * num_full + ([rem] if rem else []):

            def key(r, q=q):
                cls = 0 if tiles[r] + 1 <= max_tiles and rows[r] + q <= max_rows else 1 if tiles[r] + 1 <= max_tiles else 2
                return (cls, cost[r], r)

            r = min(range(ranks), key=key)
            quota[r] += q
            cost[r] += inc
            tiles[r] += 1
            rows[r] += q
    return torch.tensor([quota, nv, routed_tiles, routed_rows, [tile] * ranks], dtype=torch.int32)


def assign(plan_tensor, rank, num_tokens, local_only=False):
    """Destination rank of each of this rank's ``num_tokens`` shared rows (-1 past nv).

    All shared rows of all ranks, interleaved round-robin by source rank, cut
    into the quotas in rank order (``local_only``: a failed exchange).
    """

    p = plan_tensor.detach().cpu().to(torch.int64)
    quota, nv = p[0].tolist(), p[1].tolist()
    out = torch.full((num_tokens,), -1, dtype=torch.int64)
    quota_end = torch.tensor([0, *torch.tensor(quota).cumsum(0).tolist()])
    for i in range(min(nv[rank], num_tokens)):
        position = sum(min(n, i) for n in nv) + sum(1 for t in range(rank) if nv[t] > i)
        out[i] = rank if local_only else int(torch.searchsorted(quota_end, position, right=True)) - 1
    return out


def local_peer_exchanges(num_ranks, width, device="cuda", timeout_ms=2000.0):
    """Single-process stand-in for an EP group: every rank's receive buffer on one device."""

    words = 2 * num_ranks * width
    buffers = torch.zeros(num_ranks, words, dtype=torch.int32, device=device)
    pointers = [int(buffers[r].data_ptr()) for r in range(num_ranks)]
    return [PeerExchange.create(r, num_ranks, pointers, words, device, timeout_ms, (buffers,)) for r in range(num_ranks)]
