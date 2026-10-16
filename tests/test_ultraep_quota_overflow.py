# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quota reroute with more tokens than quota.

`quota_replica_index` follows upstream's `sparse_quota_replica_index` except
for two cases upstream never meets: a zero quota and an ordinal past the end of
the quota. Upstream solves the quota from the microbatch it then reroutes; in
serving the quota describes an earlier batch, so both cases are routine.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration

EP_SIZE = 4
NUM_LOGICAL = 8
MAX_REPLICAS = EP_SIZE


def _reroute(logical_topk_ids, replica_counts, rank_quota_prefix):
    """Run the vendored kernel and return the physical ids it chose."""
    from evenmoe.kernels.ultraep import UltraEPPlacementBackend

    device = logical_topk_ids.device
    # One replica per rank, so a physical id doubles as the rank that holds it
    # and the assertions can read placement straight off the output.
    logical_to_physical = torch.arange(
        NUM_LOGICAL * MAX_REPLICAS, dtype=torch.int32, device=device
    ).reshape(NUM_LOGICAL, MAX_REPLICAS)
    routed = torch.empty_like(logical_topk_ids, dtype=torch.int64)
    counters = torch.zeros(NUM_LOGICAL, dtype=torch.int32, device=device)
    UltraEPPlacementBackend().reroute_sparse(
        logical_topk_ids=logical_topk_ids,
        routed_topk_ids=routed,
        logical_to_physical_map=logical_to_physical,
        logical_replica_counts=replica_counts,
        rank_quota_prefix=rank_quota_prefix,
        counters=counters,
    )
    torch.cuda.synchronize()
    return routed


def _upstream_choice(ordinal: int, prefix: list[int], replica_count: int) -> int:
    """Upstream's formula without the two cases EvenMoE adds; the overflow tests
    assert EvenMoE departs from it."""
    idx = sum(1 for j in range(replica_count) if ordinal >= prefix[j])
    return min(idx, max(replica_count - 1, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tokens_past_the_quota_still_spread_across_replicas():
    """Overflow must not pile onto one replica.

    With 4 replicas quota'd 10 tokens each and 80 tokens arriving, every token
    from the 41st on is past the quota. Upstream's formula sends all of them to
    the last replica; wrapping sends them back through the same proportions.
    """
    device = torch.device("cuda")
    quota_each, replica_count = 10, 4
    num_tokens = 80

    prefix = [quota_each * (j + 1) for j in range(replica_count)]
    rank_quota_prefix = torch.zeros(
        NUM_LOGICAL, MAX_REPLICAS, dtype=torch.int32, device=device
    )
    rank_quota_prefix[:, :replica_count] = torch.tensor(
        prefix, dtype=torch.int32, device=device
    )
    replica_counts = torch.full(
        (NUM_LOGICAL,), replica_count, dtype=torch.int32, device=device
    )

    # Every token asks for logical expert 0, so ordinals run 0..79 and half of
    # them are past the 40-token quota.
    logical_topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)
    routed = _reroute(logical_topk_ids, replica_counts, rank_quota_prefix)

    per_replica = torch.bincount(routed.flatten(), minlength=NUM_LOGICAL * MAX_REPLICAS)
    counts = per_replica[:replica_count].tolist()
    assert sum(counts) == num_tokens
    assert all(c > 0 for c in counts), (
        f"overflow collapsed onto a subset of replicas: {counts}"
    )
    assert max(counts) - min(counts) <= 1, (
        f"overflow did not wrap through the quota's own proportions: {counts}"
    )

    # Without the wrap, upstream's formula sends the 40 tokens past the quota
    # to the last replica.
    assert (
        sum(1 for o in range(num_tokens) if _upstream_choice(o, prefix, replica_count) == replica_count - 1)
        == num_tokens - (replica_count - 1) * quota_each
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_an_unsolved_quota_spreads_evenly_rather_than_pinning_replica_zero():
    """A zero quota means "no solve yet", not "send everything to one rank".

    The cold-start path normally keeps the quota out of the picture entirely,
    but a layer whose expert drew no load in the window it was solved from
    reaches the kernel with a row of zeros.
    """
    device = torch.device("cuda")
    replica_count, num_tokens = 4, 40

    rank_quota_prefix = torch.zeros(
        NUM_LOGICAL, MAX_REPLICAS, dtype=torch.int32, device=device
    )
    replica_counts = torch.full(
        (NUM_LOGICAL,), replica_count, dtype=torch.int32, device=device
    )
    logical_topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)

    routed = _reroute(logical_topk_ids, replica_counts, rank_quota_prefix)
    counts = torch.bincount(
        routed.flatten(), minlength=NUM_LOGICAL * MAX_REPLICAS
    )[:replica_count].tolist()
    assert sum(counts) == num_tokens
    assert max(counts) - min(counts) <= 1, f"zero quota did not spread evenly: {counts}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_within_the_quota_the_kernel_matches_upstreams_formula_exactly():
    """In the regime upstream operates in, the two must agree token for token.

    Both added branches are guarded on conditions upstream never meets, so with
    every ordinal inside the quota the kernel has to reproduce upstream's
    choice exactly -- otherwise the divergence is wider than claimed.
    """
    device = torch.device("cuda")
    replica_count = 4
    prefix = [3, 7, 12, 20]  # uneven, total 20
    num_tokens = prefix[-1]

    rank_quota_prefix = torch.zeros(
        NUM_LOGICAL, MAX_REPLICAS, dtype=torch.int32, device=device
    )
    rank_quota_prefix[:, :replica_count] = torch.tensor(
        prefix, dtype=torch.int32, device=device
    )
    replica_counts = torch.full(
        (NUM_LOGICAL,), replica_count, dtype=torch.int32, device=device
    )
    logical_topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)

    routed = _reroute(logical_topk_ids, replica_counts, rank_quota_prefix).flatten()

    # The kernel hands out ordinals through an atomic counter, so which token
    # gets which ordinal is not fixed -- but the multiset of choices is.
    got = sorted(routed.tolist())
    want = sorted(
        _upstream_choice(o, prefix, replica_count) for o in range(num_tokens)
    )
    assert got == want, f"diverges inside the quota: {got} vs upstream {want}"
