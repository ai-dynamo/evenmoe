# SPDX-License-Identifier: Apache-2.0
"""Load-time V3 FP8 -> NVFP4 bridge into the current PR's native forward."""

import json
import logging
import os
import time
from pathlib import Path

import torch

from .v3_quant import convert_experts


def build(experts):
    if getattr(experts, "_mega_moe_weights_built", False):
        return
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight.shape[2] == 7168, "This opt-in bridge is V3-only"
    assert experts.w2_weight.shape[2] == 2048
    assert experts.moe_runner_config.swiglu_limit is None
    started = time.perf_counter()
    from flashinfer.megamoe import transform_nvfp4_weights_for_mega_moe

    a, ga, ea = convert_experts(experts.w13_weight.data, experts.w13_weight_scale_inv.data)
    b, gb, eb = convert_experts(experts.w2_weight.data, experts.w2_weight_scale_inv.data)
    experts.mega_l1_weights, experts.mega_l2_weights = transform_nvfp4_weights_for_mega_moe(a, b)
    experts.mega_l1_global_scales, experts.mega_l2_global_scales = ga, gb
    experts._mlb_v3_converted = True
    experts._mega_moe_sm90_fi_nvfp4 = True
    experts._mega_moe_weights_built = True
    record = {
        "layer": getattr(experts, "layer_id", None),
        "rank": torch.distributed.get_rank(),
        "l1_relative_rms": ea,
        "l2_relative_rms": eb,
        "conversion_seconds": time.perf_counter() - started,
        "activation_clamp": "infinity",
        "source": "stored FP8 weights; no calibration",
        "l1_shape": list(experts.w13_weight.shape),
        "l2_shape": list(experts.w2_weight.shape),
    }
    root = Path(os.environ["EVENMOE_MEGAMOE_CONVERSION_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"rank{record['rank']}.jsonl").open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
    for param in (
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight_scale_inv,
    ):
        param.data = torch.empty(0, dtype=param.dtype, device=ga.device)
    logging.getLogger(__name__).info("V3_LOADTIME_NVFP4 %s", json.dumps(record))
