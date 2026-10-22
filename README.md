# EvenMoE

> **Experimental.** This is an early release. Interfaces and behavior may change between versions.

A framework-independent library for expert placement and token routing in Mixture-of-Experts inference aiming to solve the MoE load imbalance issue and improve the MoE model serving efficiency. EvenMoE brings placement and routing policies behind a unified Python API, with adapters for different inference backend.

![EvenMoE — Expert placement. Balanced routing.](assets/banner-nvidia-green.png)

[Architecture](#architecture) · [Supported Models](#supported-models) · [Algorithms and Integrations](#supported-algorithms-and-integrations) · [Performance](#performance) · [Quick Start](#quick-start) · [Guidance](#guidance) · [Roadmap](#roadmap)

## Architecture

EvenMoE balances work at two levels:

- **L1 — Expert placement:** chooses which logical experts occupy physical expert slots across GPUs, including redundant replicas.
- **L2 — Token routing:** selects a physical replica for each logical routing expert selected by TopK and, when enabled, a destination GPU for shared-expert computation.

![EvenMoE architecture: framework adapters connect SGLang and vLLM to L1 expert placement and L2 token routing; the framework commits placement before routing uses it.](assets/architecture.png)

The architecture separates framework-independent policy logic from framework-specific execution. Policies consume standardized request objects and return standardized placement or routing results, while adapters convert each serving framework’s native tensors and metadata into these common representations. The serving framework remains responsible for model execution, communication-group management, rebalance scheduling, and committing updated expert placements. EvenMoE is responsible for computing the placement and routing decisions used by the framework. GPU-accelerated policies rely on the packaged CUDA and Triton kernels for efficient execution.

## Supported Models

EvenMoE is validated on the following MoE model families for SGLang serving:

| Family | Models |
|:---|:---|
| DeepSeek | DeepSeek-V3, DeepSeek-V3.2, DeepSeek-V4-Flash, DeepSeek-V4.1-Flash |
| GLM | GLM-5.2, GLM-5.3 |
| Qwen | Qwen3-235B-A22B, Qwen3.5-397B-A17B |
| Moonshot | Kimi-K2.5 |

Shared-expert routing (Waterfill) applies to models with a shared expert (DeepSeek, GLM, Kimi). Qwen MoE models have no shared expert and use EPLB / LPLB / UltraEP only.

## Supported Algorithms and Integrations

✅ Supported &nbsp;&nbsp; 🚧 On-going &nbsp;&nbsp; ❌ Not supported

Support below refers to EvenMoE's `main`-branch adapters; serving requires the corresponding framework-side hooks. Branch-specific launch instructions are provided in [Guidance](#guidance).

| Level | Algorithm | SGLang | vLLM |
|:---|:---|:---:|:---:|
| **L1 · Expert placement** | EPLB | ✅ | ✅ |
| | UltraEP | ✅ | ✅ |
| **L2 · Token routing** | Static (deterministic replica rule) | ✅ | ✅ |
| | LPLB | ✅ | ✅ |
| | Waterfill | ✅ | ✅ |
| | UltraEP | ✅ | ✅ |
| | LPLB + Waterfill | ✅ | ✅ |
| | UltraEP + Waterfill | ✅ | ✅ |

Compose L2 policies with `+`, for example `lplb+waterfill` or `ultraep+waterfill`, subject to the framework support above.

## Performance

SGLang prefill throughput with **DeepSeek-V3 (FP8)** on **one node with 8 × NVIDIA Hopper GPUs** (TP8 / DP8 / EP8, DeepEP). Every EvenMoE arm allocates **32 redundant expert replicas** (`--ep-num-redundant-experts 32`); the baseline uses none. Values are median input throughput in **K tokens/s**; percentages show the change from the baseline on the same dataset.

Placement is recorded once per dataset. **Same-domain** serves the dataset its placement was recorded from; **cross-domain** serves the other dataset. UltraEP is online-adaptive, so it has no same/cross split (both columns repeat its value).

| Algorithm | Same-domain GSM8K | Same-domain MMLU | Cross-domain GSM8K | Cross-domain MMLU |
|:---|---:|---:|---:|---:|
| Baseline | 18.0K | 18.9K | 18.0K | 18.9K |
| EPLB | 20.5K (+13.9%) | 20.6K (+9.1%) | 18.6K (+3.6%) | 19.4K (+2.8%) |
| EPLB + LPLB | 20.6K (+14.8%) | 20.6K (+8.9%) | 18.7K (+4.2%) | 19.5K (+3.4%) |
| EPLB + Waterfill | 20.7K (+14.9%) | 20.7K (+9.8%) | 19.5K (+8.7%) | 20.3K (+7.3%) |
| EPLB + LPLB + Waterfill | 20.7K (+14.9%) | 20.6K (+8.9%) | 19.6K (+8.9%) | 20.2K (+6.9%) |
| UltraEP | 20.8K (+15.6%) | 20.7K (+9.9%) | 20.8K (+15.6%) | 20.7K (+9.9%) |
| UltraEP + Waterfill | 20.9K (+16.6%) | 21.0K (+11.2%) | 20.9K (+16.6%) | 21.0K (+11.2%) |

Each workload used 1,000 prompts with ~1K-token inputs, concurrency 256, and `max_tokens=1` (prefill-only); values are the median of three measured rounds.

## Quick Start

GPU kernels additionally require a compatible NVIDIA GPU and CUDA toolchain.

| Dependency | Required version |
|:---|:---|
| Python | ≥ 3.10 |
| PyTorch (`torch`) | ≥ 2.5 |
| Triton (`triton`) | ≥ 3.0 |
| TVM FFI (`apache-tvm-ffi`) | ≥ 0.1.0 |
| NVIDIA MathDX (`nvidia-mathdx`) | 25.6.0 (exact) |

```bash
git clone https://github.com/ai-dynamo/evenmoe.git
cd evenmoe
python -m pip install -e .
```

If the serving environment already provides compatible dependencies, install with `python -m pip install -e . --no-deps`.

The UltraEP placement and routing kernels are compiled on first use through TVM FFI, so the runtime environment needs `nvcc` from a CUDA toolkit that matches the installed PyTorch (SM90 and SM100 are tested). No separate build step or build-time flag is required.

Generate an expert placement from per-expert token counts:

```python
import torch

from evenmoe import MoELoadBalancer
from evenmoe.adapters.sglang import to_placement_request

request = to_placement_request(
    torch.tensor([[20, 3, 9, 4], [5, 16, 2, 7]]),
    num_physical_experts=8,
    num_local_physical_experts=4,
    num_groups=1,
    num_nodes=1,
    algorithm="auto",
)

plan = MoELoadBalancer().plan_placement(request)
print(plan.physical_to_logical_map)
```

The returned plan assigns eight physical slots per layer to four logical experts. The serving framework loads or moves the weights and commits the placement before using it for L2 routing.

## Guidance

Install EvenMoE and DeepEP into the serving framework's Python environment. The examples below target the linked integration branches, not stock framework releases. Replace the model path with a compatible MoE checkpoint that fits the available GPU memory.

### SGLang

Use the [SGLang `sglang-moe-lb-waterfill-dev`](https://github.com/xutizhou/sglang-moe-lb-waterfill-dev) fork (its `main` branch) with EvenMoE installed as shown in Quick Start. Server arguments expose L1 placement (`--eplb-algorithm` / `--init-expert-location`) and L2 routing (`--moe-load-balancer-algorithm`) as separate selectors; see [Supported Algorithms](#supported-algorithms-and-integrations) for what SGLang supports.

Start **EPLB + LPLB + Waterfill** on one node with eight GPUs (Waterfill requires a shared-expert model):

```bash
python -m sglang.launch_server \
  --model-path /path/to/moe-checkpoint \
  --tp 8 --dp 8 --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-eplb \
  --eplb-algorithm auto \
  --ep-num-redundant-experts 32 \
  --init-expert-location /path/to/logical_count.pt \
  --moe-load-balancer-algorithm lplb+waterfill
```

Keep the launch flags and change the two selectors to switch algorithms:

| Algorithm | `--eplb-algorithm` | `--moe-load-balancer-algorithm` |
|:---|:---|:---|
| EPLB with SGLang's default replica routing | `auto` | *(omit)* |
| Static EPLB (deterministic replica rule) | `auto` | `static` |
| EPLB + LPLB | `auto` | `lplb` |
| EPLB + Waterfill | `auto` | `waterfill` |
| EPLB + LPLB + Waterfill | `auto` | `lplb+waterfill` |
| UltraEP | `auto` | `ultraep` |
| UltraEP + Waterfill | `auto` | `ultraep+waterfill` |

LPLB requires redundant replicas (`--ep-num-redundant-experts > 0`) and a recorded placement passed via `--init-expert-location` (produced once with `--enable-eplb`). Waterfill requires a shared-expert model. To run Waterfill without periodic EPLB, drop `--enable-eplb`/`--eplb-algorithm`, select `waterfill`, and retain DeepEP.

### vLLM

Use [vLLM `feat/vllm_mlb`](https://github.com/yunyang1999/vllm/tree/feat/vllm_mlb) with EvenMoE `main`. Install the fork as you would upstream vLLM, then EvenMoE into the same environment:

```bash
git clone https://github.com/ai-dynamo/evenmoe.git
python -m pip install -e ./evenmoe --no-deps
```

Start **EPLB + LPLB** on one node with eight GPUs (TP2 / DP4 / EP8):

```bash
VLLM_EVENMOE_L1_ALGORITHM=deepseek_hierarchical \
VLLM_EVENMOE_L2_ALGORITHM=lplb \
vllm serve /path/to/moe-checkpoint \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --enable-eplb \
  --eplb-config '{"policy":"evenmoe","use_async":false,"num_redundant_experts":32}' \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --numa-bind
```

Keep the launch flags and change the environment variables to select an algorithm:

| Algorithm | `VLLM_EVENMOE_L1_ALGORITHM` | `VLLM_EVENMOE_L2_ALGORITHM` |
|:---|:---|:---|
| EPLB with vLLM's native uniform replica routing | `deepseek_hierarchical` | Empty (`""`) |
| Static EPLB (deterministic replica rule) | `deepseek_hierarchical` | `static` |
| EPLB + LPLB | `deepseek_hierarchical` | `lplb` |
| EPLB + Waterfill | `deepseek_hierarchical` | `waterfill` |
| EPLB + LPLB + Waterfill | `deepseek_hierarchical` | `lplb+waterfill` |
| UltraEP | `ultraep` | `ultraep` |
| UltraEP + Waterfill | `ultraep` | `ultraep+waterfill` |
| UltraEP placement with static routing | `ultraep` | `static` |

Waterfill requires a supported shared-expert model and an evenly divisible physical-expert layout; the branch enables shared-expert fusion when Waterfill is selected. LPLB and UltraEP require redundant experts. Keep `use_async` set to `false` and `--enforce-eager`. `--numa-bind` pins each engine to its GPU's CPU socket and is recommended on two-socket nodes. For a periodic re-fit (`"step_interval"` below 1000) also set `"window_size"` to the same value.

Start **UltraEP** on the same topology, refreshing every 8 representative prefill batches:

```bash
VLLM_EVENMOE_L1_ALGORITHM=ultraep \
VLLM_EVENMOE_L2_ALGORITHM=ultraep \
EVENMOE_ULTRAEP_REFRESH_INTERVAL=8 \
vllm serve /path/to/moe-checkpoint \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --enable-eplb \
  --eplb-config '{"policy":"evenmoe","use_async":false,"num_redundant_experts":32}' \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --numa-bind
```

UltraEP re-solves placement and replica quotas every `EVENMOE_ULTRAEP_REFRESH_INTERVAL` representative prefill batches (default 64), moving only the replica slots that changed. `EVENMOE_ULTRAEP_MOVER` selects the weight transfer: `direct` (default, pairwise send/recv), `symm` (one-sided puts through torch symmetric memory, for an EP group inside one NVLink domain), or `auto`. `EVENMOE_ULTRAEP_LAGGED_APPLY=1` applies each refresh on the layer's next forward, trading one step of staleness for the per-layer device sync; off by default. The kernels compile on first use on Hopper and Blackwell.

## Roadmap

- [x] Extend prefill-phase algorithm support to DeepSeek-V4, GLM, and Qwen models (see [Supported Models](#supported-models)).
- [ ] Add decode-phase support with the Metro algorithm.
- [ ] Add more prefill-phase algorithms, including MoonEP.
- [ ] Support MCore training.
- [ ] Integrate with TensorRT-LLM (TRT-LLM).

## Contributing

Bug reports, algorithm ideas, and integration improvements are welcome through [Issues](https://github.com/ai-dynamo/evenmoe/issues) and pull requests.

## Acknowledgments

We thank DeepSeek for [EPLB](https://github.com/deepseek-ai/EPLB) and [LPLB](https://github.com/deepseek-ai/LPLB), RedNote for [UltraEP](https://github.com/Dots-Infra/UltraEP), and the [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm) teams for their inference frameworks. We appreciate these teams and their communities for sharing the algorithms, code, and infrastructure that make this work possible.

## License

Licensed under [Apache-2.0](https://github.com/ai-dynamo/evenmoe/blob/main/LICENSE). See [Third-Party Notices](https://github.com/ai-dynamo/evenmoe/blob/main/THIRD_PARTY_NOTICES.md) for components adapted from DeepSeek, SGLang, and UltraEP.

The UltraEP kernels under `src/evenmoe/kernels/ultraep/` are adapted from [UltraEP](https://github.com/Dots-Infra/UltraEP). The upstream code stays under the MIT license in [LICENSE.ultraep](https://github.com/ai-dynamo/evenmoe/blob/main/src/evenmoe/kernels/ultraep/LICENSE.ultraep), and NVIDIA's modifications are under Apache-2.0. Each adapted file carries both notices in its header.

## Notice and Disclaimer

Installation retrieves NVIDIA MathDX through the declared `nvidia-mathdx` pip dependency.

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.
