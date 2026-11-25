# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the pinned MegaMoE example; no server or GPU is started."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evenmoe.core.types import PlacementSnapshot
from examples.megamoe.launch import plan, validate_inputs
from examples.megamoe.runtime.decode_guard import is_collective_decode
from examples.megamoe.runtime.nvfp4_balance_adapter import full_replica_snapshot, packed_views


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode_or_idle(self):
        return self.name in ("decode", "idle")


@pytest.mark.parametrize("local", ["decode", "idle", "extend"])
@pytest.mark.parametrize("global_name", [None, "decode", "extend"])
@pytest.mark.parametrize("has_extend", [False, True])
def test_decode_collective_selection(local, global_name, has_extend):
    batch = SimpleNamespace(
        forward_mode=Mode(local),
        global_forward_mode=Mode(global_name) if global_name else None,
        is_extend_in_batch=has_extend,
    )
    assert is_collective_decode(batch) == (
        not has_extend and (global_name or local) in ("decode", "idle")
    )
    assert not is_collective_decode(None)


def test_twf_is_opt_in_and_requires_supported_policy_and_precision():
    for precision in ("fp8", "nvfp4"):
        config = {"precision": precision, "policy": "baseline", "twf": False}
        result = plan(config, Path("model"), Path("run"))
        assert result["environment"]["EVENMOE_MEGAMOE_TWF"] == "0"
        assert result["environment"]["EVENMOE_MEGAMOE_PRECISION"] == precision
        assert "--disable-shared-experts-fusion" in result["command"]
        with pytest.raises(ValueError):
            plan({**config, "twf": True}, Path("model"), Path("run"))
    config = {"precision": "nvfp4", "policy": "static+waterfill", "twf": True}
    with pytest.raises(ValueError):
        plan(config, Path("model"), Path("run"))
    result = plan(config, Path("model"), Path("run"), Path("placement.json"))
    assert "--enforce-shared-experts-fusion" in result["command"]
    assert result["environment"]["EVENMOE_MEGAMOE_TWF"] == "1"
    with pytest.raises(ValueError):
        plan({**config, "precision": "fp8"}, Path("model"), Path("run"), Path("p"))


def test_incomplete_placement_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 7168,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 256,
                "num_experts_per_tok": 8,
                "n_shared_experts": 1,
                "num_hidden_layers": 61,
                "quantization_config": {"quant_method": "fp8"},
            }
        )
    )
    p = tmp_path / "placement.json"
    row = list(range(256)) + list(range(32))
    p.write_text(json.dumps({"physical_to_logical_map": [row] * 61}))
    assert "placement" in validate_inputs(tmp_path, p)
    p.write_text(json.dumps({"physical_to_logical_map": [[0] * 288] * 61}))
    with pytest.raises(ValueError, match="complete"):
        validate_inputs(tmp_path, p)


@pytest.mark.parametrize("fp8", [False, True])
def test_mover_views_preserve_shared_slot_and_include_scales(fp8):
    tensors = [torch.arange(12).reshape(3, 4) + i * 100 for i in range(4)]
    experts = SimpleNamespace(num_local_experts=3, _mega_moe_sm90_fp8_weights=fp8)
    if fp8:
        experts.mega_l1_weights, experts.mega_l2_weights = tensors[:2], tensors[2:]
    else:
        (
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            experts.mega_l1_global_scales,
            experts.mega_l2_global_scales,
        ) = tensors
    views = packed_views(experts, 2)
    assert len(views) == 4
    for view, original in zip(views, tensors):
        shared = original[2].clone()
        view.fill_(-1)
        assert torch.equal(original[2], shared)
        assert (original[:2] == -1).all()


def test_snapshot_recovers_all_loaded_replicas():
    p = PlacementSnapshot(
        0,
        2,
        4,
        2,
        2,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([[0], [1]]),
        torch.ones(2, dtype=torch.int64),
        torch.tensor([0, 1]),
    )
    result = full_replica_snapshot(p)
    assert result.logical_to_physical_candidates.tolist() == [[0, 2], [1, 3]]
    assert result.logical_to_physical_count.tolist() == [2, 2]
    assert result.default_physical_for_logical.tolist() == [0, 1]
