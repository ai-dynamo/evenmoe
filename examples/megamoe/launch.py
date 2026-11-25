# SPDX-License-Identifier: Apache-2.0
"""Print a reproducible launch plan; --execute starts the local SGLang server."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

POLICIES = ("baseline", "static", "lplb", "static+waterfill", "lplb+waterfill")


def plan(config, model, output, placement=None, port=30000):
    precision, policy, twf = config["precision"], config["policy"], config["twf"]
    if precision not in ("fp8", "nvfp4") or policy not in POLICIES:
        raise ValueError("Unsupported precision or routing policy")
    if not isinstance(twf, bool):
        raise TypeError("twf must be a boolean")
    if twf and (precision != "nvfp4" or policy != "static+waterfill"):
        raise ValueError("This pilot exposes TWF only for NVFP4 static+waterfill")
    redundant = 0 if policy == "baseline" else 32
    if bool(placement) != bool(redundant):
        raise ValueError(
            "Balanced policies require a frozen placement; baseline must not supply one"
        )
    if not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    env = {
        "EVENMOE_MEGAMOE_PRECISION": precision,
        "EVENMOE_MEGAMOE_POLICY": policy,
        "EVENMOE_MEGAMOE_TWF": str(int(twf)),
        "EVENMOE_MEGAMOE_CONVERSION_AUDIT_DIR": str(output / "conversion"),
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "WATERFILL_USE_GLOBAL_LOAD": "0",
        "EVM_TWF_MODE": "0",
        "EVM_TWF_RHO": "1.5",
        "EVM_WF_BUILD": str(output / "cache/twf"),
        "FLASHINFER_WORKSPACE_BASE": str(output / "cache/flashinfer"),
        "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
        "OMP_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--served-model-name",
        "deepseek-v3",
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
        "--enable-dp-attention",
        "--load-balance-method",
        "round_robin",
        "--moe-a2a-backend",
        "megamoe",
        "--chunked-prefill-size",
        "8192",
        "--context-length",
        "16384",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--ep-num-redundant-experts",
        str(redundant),
        "--ep-dispatch-algorithm",
        "static",
        "--max-running-requests",
        "2048",
        "--max-total-tokens",
        "262144",
        "--mem-fraction-static",
        "0.70",
        "--random-seed",
        "20260811",
        "--stream-interval",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--watchdog-timeout",
        "1800",
    ]
    command += [
        "--enforce-shared-experts-fusion"
        if "waterfill" in policy
        else "--disable-shared-experts-fusion"
    ]
    if placement:
        command += ["--init-expert-location", str(placement)]
    return {"environment": env, "command": command, "configuration": config}


def validate_inputs(model, placement):
    cfg = json.loads((model / "config.json").read_text())
    expected = {
        "hidden_size": 7168,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "num_hidden_layers": 61,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("This example is restricted to the DeepSeek-V3 geometry")
    if cfg.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("Both arms must load the same stored FP8 checkpoint")
    hashes = {"model_config": hashlib.sha256((model / "config.json").read_bytes()).hexdigest()}
    if placement:
        mappings = json.loads(placement.read_text())["physical_to_logical_map"]
        if len(mappings) != 61 or any(
            len(row) != 288 or set(row) != set(range(256)) for row in mappings
        ):
            raise ValueError("Expected 61 layers of complete 288-slot routed-expert maps")
        hashes["placement"] = hashlib.sha256(placement.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--placement", type=Path)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = plan(
        config,
        args.model.resolve(),
        args.output.resolve(),
        args.placement.resolve() if args.placement else None,
        args.port,
    )
    if args.execute:
        result["input_hashes"] = validate_inputs(args.model, args.placement)
        import torch

        if torch.cuda.device_count() != 8 or any(
            torch.cuda.get_device_capability(i) != (9, 0) for i in range(8)
        ):
            raise RuntimeError("Expose exactly eight SM90 GPUs in one NVLink domain")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "launch.json").write_text(json.dumps(result, indent=2) + "\n")
        env = dict(os.environ, **result["environment"])
        os.execvpe(result["command"][0], result["command"], env)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
