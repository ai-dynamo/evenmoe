# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LP's shape must not depend on how a placement happened to fall.

``_build_layer_state`` turns a committed placement into the LP that LPLB
solves. Its constraint matrix is (num_red_log + num_gpus) by (num_red_phy +
num_gpus + 2), and ``num_red_log`` is however many logical experts the
placement gave a second copy. A load-aware placement gives a hot expert three
copies and its neighbour none, so that count differs layer to layer -- and
each distinct value NVRTC-compiles its own IPM, prep and post kernel.

Measured on DeepSeek-V3.2 / EP8: one rearrangement at 32 redundant experts
compiled 17 shapes and took 266 s against vLLM's 300 s execute-model timeout;
at 16 redundant experts it took 8 s, which is already past the CPU-recv
timeout of the peer ranks sitting in a DeepEP dispatch waiting for it.

So the shape is padded to a constant by promoting single-copy experts into
the replicated set. These tests hold that padding to two claims:

  it works       every placement at a given (num_logical, num_phy, num_gpus)
                 produces one shape, so the kernels compile once
  it is free     a promoted expert has one replica, so its flow is pinned and
                 its load reaches the same GPU by a different term of the same
                 sum -- the LP being solved is unchanged

The second is the one worth testing hard. Padding that quietly changed the
objective would be worse than the stall it removes.
"""

import pytest
import torch

from evenmoe.core.types import PlacementSnapshot
from evenmoe.policies.l2.lplb import _SHAPE_BUCKET, _build_layer_state

NUM_LOGICAL, NUM_GPUS = 64, 8


def make_snapshot(extra_copies, num_redundant, layer_id=0, seed=0):
    """A placement that hands ``extra_copies`` out over the logical experts.

    ``extra_copies`` is a list of (logical_id, how many *extra* copies), which
    is what a load-aware L1 produces: concentrated on the hot experts, not
    spread one-per-expert.
    """
    torch.manual_seed(seed)
    num_phy = NUM_LOGICAL + num_redundant
    assert sum(n for _, n in extra_copies) == num_redundant

    max_copies = 1 + max([n for _, n in extra_copies] + [0])
    l2p = torch.full((NUM_LOGICAL, max_copies), -1, dtype=torch.int64)
    l2p[:, 0] = torch.arange(NUM_LOGICAL, dtype=torch.int64)
    counts = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    phy2log = torch.arange(num_phy, dtype=torch.int64) % NUM_LOGICAL
    phy2log[:NUM_LOGICAL] = torch.arange(NUM_LOGICAL, dtype=torch.int64)

    nxt = NUM_LOGICAL
    for logical, n in extra_copies:
        for k in range(n):
            l2p[logical, 1 + k] = nxt
            phy2log[nxt] = logical
            nxt += 1
        counts[logical] = 1 + n

    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=num_phy,
        ep_size=NUM_GPUS,
        num_local_physical_experts=num_phy // NUM_GPUS,
        physical_to_logical_map=phy2log,
        logical_to_physical_candidates=l2p,
        logical_to_physical_count=counts,
    )


# Four placements of the same 16 redundant slots. Before padding these give
# num_red_log of 16, 8, 4 and 1 -- four shapes, four sets of compiles.
SPREADS = {
    "one each": [(i, 1) for i in range(16)],
    "pairs": [(i, 2) for i in range(8)],
    "quads": [(i, 4) for i in range(4)],
    "all on one": [(0, 16)],
}


def test_shape_count_is_bounded_by_the_bucket():
    """The claim is a bound, not uniqueness.

    Padding every layer to num_redundant would give exactly one shape, but the
    solve is compute-bound in nc -- measured on H20, nc=16 takes 75 us and
    nc=40 takes 301 us -- so flattening to the maximum buys one compile with a
    permanent per-batch tax. Bucketing caps the compiles instead.
    """
    num_redundant = 16
    shapes = {}
    for name, spread in SPREADS.items():
        state = _build_layer_state(make_snapshot(spread, num_redundant), NUM_GPUS, 1e4)
        shapes[name] = (
            tuple(state.A_base.shape),
            int(state.log_single.numel()),
            int(state.log_replicated.numel()),
            int(state.phy_replicated.numel()),
        )
    distinct = {v[0] for v in shapes.values()}
    ceiling = -(-num_redundant // _SHAPE_BUCKET)
    assert len(distinct) <= ceiling, (
        f"at most {ceiling} shapes may survive bucketing; got {shapes}")
    # Without bucketing these four placements give four; the bound has to be
    # doing work, not just being satisfied by a coincidence.
    assert len(distinct) < len(SPREADS), f"bucketing collapsed nothing: {shapes}"

    for name, (shape, num_single, num_red_log, num_red_phy) in shapes.items():
        # Everything else follows from num_red_log, which is what gets bucketed.
        assert num_red_log % _SHAPE_BUCKET == 0, (name, num_red_log)
        assert num_red_phy == num_red_log + num_redundant, name
        assert num_single == NUM_LOGICAL - num_red_log, name
        assert shape[0] == num_red_log + NUM_GPUS, name


def test_padding_per_layer_stays_under_one_bucket():
    """The cost side of the same bound: no layer may carry more slack than a
    bucket's worth, or the solve it pays for every batch drifts back up."""
    for num_redundant in (8, 16, 32):
        for name, spread in list(SPREADS.items()) + [
            ("many", [(i, 1) for i in range(min(num_redundant, 16))]),
        ]:
            total = sum(n for _, n in spread)
            if total != num_redundant:
                continue
            state = _build_layer_state(make_snapshot(spread, num_redundant), NUM_GPUS, 1e4)
            replica_counts = (state.log2phy >= 0).sum(dim=1)
            promoted = int((replica_counts[state.log_replicated] == 1).sum())
            assert promoted < _SHAPE_BUCKET, (
                f"num_redundant={num_redundant} {name}: padded {promoted} rows, "
                f"more than one bucket")


def test_shape_holds_across_redundancy_levels_within_a_level():
    """A different num_redundant is a different deployment, so a different
    shape there is expected -- but within one level it must still be one."""
    for num_redundant in (8, 16, 32):
        spreads = [
            [(i, 1) for i in range(num_redundant)],
            [(i, 2) for i in range(num_redundant // 2)],
            [(0, num_redundant)],
        ]
        shapes = {
            tuple(_build_layer_state(make_snapshot(s, num_redundant), NUM_GPUS, 1e4).A_base.shape)
            for s in spreads
        }
        ceiling = -(-num_redundant // _SHAPE_BUCKET)
        assert len(shapes) <= ceiling, f"num_redundant={num_redundant}: {shapes}"


def gpu_load_from_lp(state, counts_norm):
    """Per-GPU load exactly as the LP accounts for it.

    Two terms, matching lp_prep.cuh: the base load carried by single-copy
    experts (B1 @ t1), and the replicated flow the solver places (B2 @ x).
    With every replicated expert's flow pinned to its own load, x is known
    without solving, which is what lets a promoted expert be checked in
    closed form.
    """
    t1 = counts_norm[state.log_single]
    base = state.B1 @ t1
    # B2 sits in A_base's bottom-left block.
    num_red_log = int(state.log_replicated.numel())
    num_red_phy = int(state.phy_replicated.numel())
    B2 = state.A_base[num_red_log:, :num_red_phy]
    return base, B2


def test_promotion_moves_load_between_terms_not_between_gpus():
    """The claim that makes padding free.

    A promoted expert leaves the B1 @ t1 term and enters B2 @ x. If the two
    terms did not add back to the same per-GPU number, the LP would be
    minimising a different peak and every routing decision downstream of it
    would be answering a different question.
    """
    torch.manual_seed(0)
    counts_norm = torch.rand(NUM_LOGICAL, dtype=torch.float32)
    counts_norm /= counts_norm.sum()

    # Padding bites hardest here: one expert holds all 16 copies, so 15
    # single-copy experts get promoted.
    state = _build_layer_state(make_snapshot(SPREADS["all on one"], 16), NUM_GPUS, 1e4)

    # Guard against the check going vacuous: if promotion ever stops firing,
    # the conservation below still holds trivially and would keep passing
    # while testing nothing about promotion at all.
    replica_counts = (state.log2phy >= 0).sum(dim=1)
    promoted = int((replica_counts[state.log_replicated] == 1).sum())
    assert promoted > 0, "nothing was promoted, so this test proves nothing"

    base, B2 = gpu_load_from_lp(state, counts_norm)

    # Each replicated logical's flow must sum to its load; a promoted one has
    # a single replica, so that pins x for it. Build x by splitting each
    # logical's load evenly over its replicas -- any feasible split works for
    # a conservation check, and for the promoted ones there is only one.
    x = torch.zeros(int(state.phy_replicated.numel()), dtype=torch.float32)
    phy2log_full = state.log2phy  # (num_logical, max_copies)
    for slot_pos, phy in enumerate(state.phy_replicated.tolist()):
        owner = (phy2log_full == phy).nonzero()[0, 0].item()
        n_replicas = int((phy2log_full[owner] >= 0).sum())
        x[slot_pos] = counts_norm[owner] / n_replicas
    lp_view = base + B2 @ x

    # The same per-GPU load, summed directly over physical slots without any
    # notion of single vs replicated.
    per_gpu = int(state.num_phy // NUM_GPUS)
    direct = torch.zeros(NUM_GPUS, dtype=torch.float32)
    for logical in range(NUM_LOGICAL):
        replicas = phy2log_full[logical]
        replicas = replicas[replicas >= 0]
        share = counts_norm[logical] / replicas.numel()
        for phy in replicas.tolist():
            direct[phy // per_gpu] += share

    torch.testing.assert_close(lp_view, direct, rtol=1e-5, atol=1e-6)


def test_promoted_expert_cannot_be_routed_anywhere():
    """A promoted expert must add no freedom, only shape.

    Its row in the top block has to name exactly one column. More than one
    would mean the LP believes it can split that expert's traffic, and it
    would balance by sending tokens to a replica that does not exist.
    """
    state = _build_layer_state(make_snapshot(SPREADS["all on one"], 16), NUM_GPUS, 1e4)
    num_red_log = int(state.log_replicated.numel())
    num_red_phy = int(state.phy_replicated.numel())
    C = state.A_base[:num_red_log, :num_red_phy]

    counts = state.log2phy >= 0
    promoted, genuine = 0, 0
    for row, logical in enumerate(state.log_replicated.tolist()):
        n_replicas = int(counts[logical].sum())
        nonzero = int((C[row] != 0).sum())
        assert nonzero == n_replicas, (
            f"logical {logical} has {n_replicas} replicas but {nonzero} columns"
        )
        if n_replicas == 1:
            promoted += 1
        else:
            genuine += 1
    assert promoted > 0 and genuine == 1, (promoted, genuine)


def test_constraint_matrix_stays_full_rank():
    """Padding with rows that repeat an existing constraint would make A A^T
    singular, and the solver's Cholesky would fail or return garbage. Each
    promoted row names a column no other row touches, so rank is preserved."""
    for name, spread in SPREADS.items():
        state = _build_layer_state(make_snapshot(spread, 16), NUM_GPUS, 1e4)
        A = state.A_base.to(torch.float64)
        assert torch.linalg.matrix_rank(A).item() == A.shape[0], (
            f"{name}: constraint matrix lost rank"
        )


def test_padding_is_deterministic_across_ranks():
    """Every rank builds this from the same committed placement and must pick
    the same experts to promote; disagreement would give the ranks different
    LPs and different splits for the same tokens."""
    a = _build_layer_state(make_snapshot(SPREADS["quads"], 16, seed=1), NUM_GPUS, 1e4)
    b = _build_layer_state(make_snapshot(SPREADS["quads"], 16, seed=2), NUM_GPUS, 1e4)
    assert torch.equal(a.log_replicated, b.log_replicated)
    assert torch.equal(a.log_single, b.log_single)


def test_a_layer_with_no_redundancy_is_left_alone():
    """Padding must not invent an LP for a layer that has no choice to make:
    that would pay a solve and an EP collective per forward for nothing."""
    snap = make_snapshot([], 0)
    state = _build_layer_state(snap, NUM_GPUS, 1e4)
    assert not state.has_redundancy


def test_ordinary_placements_raise_no_warning(caplog):
    """The degradation warning must stay quiet on placements that are fine.

    It exists to make an unclaimed-slot placement visible instead of letting it
    resurface as another multi-minute rearrangement. A warning that also fires
    on ordinary placements teaches everyone to ignore it, which costs exactly
    the visibility it was added for -- and the first version of it did, because
    it still compared against the pad-to-maximum target after the padding moved
    to buckets.
    """
    import logging

    for num_redundant in (8, 16, 32):
        for name, spread in SPREADS.items():
            if sum(n for _, n in spread) != num_redundant:
                continue
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                _build_layer_state(make_snapshot(spread, num_redundant), NUM_GPUS, 1e4)
            assert not caplog.records, (
                f"num_redundant={num_redundant} {name}: {caplog.text}")


def test_enumerated_shape_set_covers_everything_that_can_occur():
    """Precompiling is only worth anything if the enumeration is complete.

    The set is derived in closed form from num_redundant and the bucket size,
    without looking at any load. If a placement can produce a shape outside it,
    that shape compiles on first use -- inside whichever rearrangement first
    produces it, which is the stall the precompile exists to remove. A missed
    shape is therefore not a small loss of coverage; it reproduces the original
    bug for that shape.
    """
    import random

    from evenmoe.policies.l2.lplb import bucketed_shape_set

    rng = random.Random(0)
    for num_redundant in (8, 16, 32):
        declared = {
            (s["nc"], s["nv"], s["num_single"], s["num_red_log"], s["num_red_phy"])
            for s in bucketed_shape_set(NUM_LOGICAL, NUM_LOGICAL + num_redundant,
                                        NUM_GPUS, 2)
        }
        seen = set()
        for _ in range(60):
            # Hand the redundant copies out at random, the way a load-aware L1
            # does: a few hot experts take several, most take none.
            left, extra = num_redundant, []
            while left:
                take = rng.randint(1, min(left, 4))
                extra.append((len(extra), take))
                left -= take
            state = _build_layer_state(
                make_snapshot(extra, num_redundant), NUM_GPUS, 1e4)
            num_red_log = int(state.log_replicated.numel())
            shape = (
                state.A_base.shape[0],
                state.A_base.shape[1] + 1,
                int(state.log_single.numel()),
                num_red_log,
                int(state.phy_replicated.numel()),
            )
            seen.add(shape)
            assert shape in declared, (
                f"num_redundant={num_redundant}: {shape} occurs but was never "
                f"precompiled; declared={sorted(declared)}")
        # And the enumeration must not be bloated: a declared shape that no
        # placement produces costs minutes of NVRTC for a kernel no forward
        # ever launches. The set size is exactly the number of buckets.
        assert seen, "generated no placements"
        assert len(declared) == -(-num_redundant // _SHAPE_BUCKET), (
            f"num_redundant={num_redundant}: declared {len(declared)} shapes, "
            f"expected {-(-num_redundant // _SHAPE_BUCKET)}")
        assert seen <= declared


def test_the_whole_shape_set_is_compiled_before_the_first_solve():
    """Precompiling has to actually happen, and happen once.

    The point is to move NVRTC out of the rearrangement, so it has to fire on
    the first layer prepared -- which the framework does after weight loading
    and before serving -- and not again per layer afterwards.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from test_lplb_policy import _RecordingKernels

    from evenmoe.policies.l2.lplb import (
        LPLBL2Router,
        bucketed_shape_set,
    )

    kernels = _RecordingKernels()
    router = LPLBL2Router(kernels=kernels, num_gpus=NUM_GPUS)
    num_redundant = 16
    expected = bucketed_shape_set(NUM_LOGICAL, NUM_LOGICAL + num_redundant, NUM_GPUS, 2)

    for layer_id in range(4):
        router.prepare_layer(
            make_snapshot(SPREADS["pairs"], num_redundant, layer_id=layer_id))

    assert len(kernels.precompiled) == len(expected), (
        f"expected the set compiled once ({len(expected)} shapes), "
        f"got {len(kernels.precompiled)} calls across 4 layers")
    assert {(s["nc"], s["nv"]) for s in kernels.precompiled} == \
        {(s["nc"], s["nv"]) for s in expected}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_replica_policy_survives_the_fused_path():
    """Each policy must actually run against CUDA tensors, not just host ones.

    The fused branch only fires with CUDA tensors, so the host-only suite
    never enters it -- a wrong argument list there stays invisible until an
    engine fails to start. That is exactly how `static` reached a benchmark
    with a missing positional argument.
    """
    from evenmoe.core.types import RoutingPolicyConfig, RoutingRequest
    from evenmoe.policies.l2.replica import (
        DynamicL2Router,
        StaticL2Router,
    )

    num_redundant = 16
    snap = make_snapshot(SPREADS["pairs"], num_redundant)
    cuda = lambda t: t.cuda() if hasattr(t, "cuda") else t
    snap = type(snap)(**{
        k: (cuda(v) if hasattr(v, "is_cuda") else v)
        for k, v in snap.__dict__.items()
    })
    topk = torch.randint(0, NUM_LOGICAL, (512, 8), dtype=torch.int32, device="cuda")
    num_phy = snap.num_physical_experts

    # `static` routes through the framework-chosen default replica, so the
    # snapshot has to carry one; without it the policy refuses rather than
    # inventing a choice.
    from evenmoe.adapters.vllm import nearest_replica_table

    defaults = nearest_replica_table(
        snap.logical_to_physical_candidates.to(torch.int64),
        snap.logical_to_physical_count.to(torch.int64),
        ep_rank=0,
        num_local_physical_experts=num_phy // NUM_GPUS,
    )
    snap = type(snap)(**{**snap.__dict__, "default_physical_for_logical": defaults})

    for router, name in ((StaticL2Router(), "static"), (DynamicL2Router(), "dynamic")):
        req = RoutingRequest(
            layer_id=0,
            logical_topk_ids=topk,
            topk_weights=torch.ones_like(topk, dtype=torch.float32),
            placement=snap,
        )
        decision = router.route_tokens(req, RoutingPolicyConfig.from_value(name))
        physical_ids = decision.routed_physical_topk_ids
        assert physical_ids.shape == topk.shape, (
            f"{name}: got shape {tuple(physical_ids.shape)}, expected {tuple(topk.shape)}")
        assert bool(((physical_ids >= 0) & (physical_ids < num_phy)).all()), (
            f"{name}: produced an out-of-range physical id")


