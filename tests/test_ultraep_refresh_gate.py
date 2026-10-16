# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from evenmoe.policies.l1 import RefreshGate


def _tick(gate: RefreshGate, layer_id: int, *, representative: bool) -> bool:
    """The correct two-phase calling convention: is_due() first (always),
    should_refresh() only when it says a refresh could happen this call."""
    if not gate.is_due(layer_id):
        return False
    return gate.should_refresh(layer_id, representative=representative)


@pytest.mark.unit
def test_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="positive"):
        RefreshGate(0)


@pytest.mark.unit
def test_first_call_always_refreshes():
    gate = RefreshGate(interval=4)
    assert _tick(gate, 0, representative=False) is True


@pytest.mark.unit
def test_waits_the_full_interval_of_representative_batches():
    gate = RefreshGate(interval=3)
    _tick(gate, 0, representative=True)  # bootstrap, resets counter to 0

    assert _tick(gate, 0, representative=True) is False  # 1
    assert _tick(gate, 0, representative=True) is False  # 2
    assert _tick(gate, 0, representative=True) is True  # 3 -> refresh


@pytest.mark.unit
def test_non_representative_batches_do_not_count():
    gate = RefreshGate(interval=2)
    _tick(gate, 0, representative=True)  # bootstrap

    assert _tick(gate, 0, representative=False) is False
    assert _tick(gate, 0, representative=False) is False
    assert _tick(gate, 0, representative=False) is False
    assert _tick(gate, 0, representative=True) is True


@pytest.mark.unit
def test_a_non_representative_bootstrap_refreshes_again_on_the_next_real_batch():
    """A placement solved from a tiny/unrepresentative first batch is likely
    poor; the next representative batch should fix it immediately rather than
    waiting out a full interval with a known-thin placement in place."""
    gate = RefreshGate(interval=5)
    _tick(gate, 0, representative=False)  # bootstrap, thin data

    assert _tick(gate, 0, representative=False) is False
    assert _tick(gate, 0, representative=True) is True


@pytest.mark.unit
def test_layers_are_tracked_independently():
    gate = RefreshGate(interval=2)
    _tick(gate, 0, representative=True)  # layer 0 bootstrap
    assert _tick(gate, 1, representative=True) is True  # layer 1 bootstrap

    assert _tick(gate, 0, representative=True) is False
    assert _tick(gate, 1, representative=True) is False
    assert _tick(gate, 0, representative=True) is True
    assert _tick(gate, 1, representative=True) is True


@pytest.mark.unit
def test_is_due_is_cheap_to_poll_every_pass_between_real_refreshes():
    """The actual property a caller avoiding an unnecessary sync depends on:
    is_due() takes no `representative` argument at all, so it is always safe
    to call on every pass and only escalate to computing that (and calling
    should_refresh) when it returns True. Interleaved correctly with
    should_refresh (the real calling contract), the interval counts calls,
    not just representative ones -- a due-but-not-representative call must
    leave the layer due again next time, not reset it.
    """
    gate = RefreshGate(interval=2)
    _tick(gate, 0, representative=True)  # bootstrap, counter -> 0

    assert gate.is_due(0) is False  # 1 call since
    assert gate.is_due(0) is True  # 2 -> due
    assert gate.should_refresh(0, representative=False) is False  # declines, stays due

    assert gate.is_due(0) is True  # still due, not reset by the decline
    assert gate.should_refresh(0, representative=True) is True  # finally refreshes
