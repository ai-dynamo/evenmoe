# DeepSeek-V3 MegaMoE FP8 / NVFP4 with optional Decode TWF

This experimental, source-checkout example packages two existing SM90 MegaMoE
compute paths and an opt-in Tile-aware Waterfill (TWF) Decode route in one
integration. It uses EvenMoE's existing placement and routing APIs; model execution,
weight loading and communication stay in SGLang. It does not add a new public
policy name or replace EvenMoE's framework-independent core.

**Status: draft integration; GPU revalidation of this packaged revision is pending.**
Historical experiments exercised these native backends and NVFP4 TWF. That does
not establish correctness or performance of the newly packaged installation.
See [VALIDATION.md](VALIDATION.md) for the exact checks and remaining gates.

## Scope

| Configuration | Native compute | Routing |
| --- | --- | --- |
| `fp8.json` | DeepGEMM SM90 FP8 MegaMoE | Baseline; no redundant experts |
| `nvfp4.json` | FlashInfer SM90 NVFP4 MegaMoE | Baseline; no redundant experts |
| `nvfp4-waterfill.json` | FlashInfer SM90 NVFP4 MegaMoE | Frozen EPLB + original Waterfill |
| `nvfp4-twf.json` | Same NVFP4 compute | Frozen EPLB + TWF on globally pure Decode |

All configurations load the **same stored FP8 DeepSeek-V3 checkpoint**. NVFP4
converts routed expert weights at load time without data-based quantization
calibration. It is not a calibrated NVFP4 checkpoint. Conversion error is recorded
per layer/rank. Baseline shared experts remain on their original path; Waterfill
merges a shared slot into MegaMoE, changing its quantization path as well.

The example is restricted to one node, eight SM90 GPUs in one NVLink domain,
TP8 / attention-DP8 / EP8, hidden size 7168, intermediate size 2048, 256 logical
routed experts, top8, and eager execution. Balanced configurations require 288
physical routed slots; Waterfill adds eight shared slots, giving 296 slots/top9.
This is not a claim of portability to arbitrary models or GPU generations.

TWF is off by default and exposed only for NVFP4 `static+waterfill`. Every rank,
including idle ranks, participates in the same collective. Any Prefill rank keeps
all ranks on the original route. Missing TBO `global_forward_mode` is handled
using synchronized `is_extend_in_batch` plus local Decode/idle mode. TWF uses
64-row tiles, caps local input at 4096 rows, and traps on exchange timeout.
It is not enabled for LPLB, UltraEP, Prefill, or mixed batches.

## Build from public dependencies

From the repository root:

```bash
bash examples/megamoe/build.sh evenmoe-megamoe:dev
```

The Dockerfile pins the public base image by digest. `dependencies.lock.json`
pins the SGLang and FlashInfer source revisions; DeepGEMM comes from the pinned
image and its package version is checked. Build tools may require network access
and Rust compilation. The resulting image digest and `pip freeze` must be recorded
for each experiment; this is not a bit-for-bit reproducible wheel-build claim.
No private images, model weights or benchmark datasets are bundled.

`install_overlay.py` checks all source hashes and patch output hashes before
writing. It refuses unknown versions and conflicting installed overlay files.
Without `--apply` it only verifies. Run it inside a fresh build environment,
never on an active server. The patch set includes the V3 weight-loading bridge,
unclamped SwiGLU compatibility, routing call site and fail-closed MegaMoE capacity
checks. Optional buffer-alignment experiments and GC-MoE are not included.

## Launch

Print a plan without starting a server (Python standard library only):

```bash
python examples/megamoe/launch.py \
  --config examples/megamoe/configs/fp8.json \
  --model /path/to/DeepSeek-V3 --output /path/to/new-run
```

Start one configuration with a **new** run directory, replacing host paths:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v /path/to/DeepSeek-V3:/model:ro \
  -v /path/to/runs:/runs \
  evenmoe-megamoe:dev \
  --config /opt/evenmoe/examples/megamoe/configs/nvfp4.json \
  --model /model --output /runs/nvfp4-baseline --execute
```

The service binds to loopback port 30000; `--port` overrides it. `launch.json`
records effective command, explicit environment and model/placement hashes.
The launcher rejects unsupported model geometry, precision/policy combinations,
and reuse of an existing output directory. Inspect GPU ownership before launch.

For balanced runs, collect logical expert counts from a disjoint calibration set,
aggregate across EP ranks, and export a JSON array `[61, 256]`. Do not use
measurement requests to create the placement. In the built runtime:

```bash
python examples/megamoe/make_placement.py \
  --counts calibration-counts.json --output placement.json
```

Mount the same frozen `placement.json` into every comparison container and add
`--placement /path/in/container/placement.json`. Choose `nvfp4-waterfill.json`
or `nvfp4-twf.json`; both use exactly that placement. Map dimensions and coverage
are checked before server launch. A zero-count dense layer still needs a complete
mapping; never export a partially populated expert map.

For existing replica-router experiments, copy a JSON config and choose `static`,
`lplb`, or `lplb+waterfill` with TWF disabled and a frozen placement. These are
experimental ports, not newly validated performance claims. Online UltraEP
refresh is intentionally not exposed by this launch interface.

## Validation and benchmark protocol

```bash
python -m ruff check .
python -m pytest tests/test_megamoe_example.py tests/test_megamoe_quantization.py
```

Before accepting GPU performance, compare numerical outputs, complete real-server
warmup, and trace every rank: `fp8_mega_moe` for FP8, `nvfp4_mega_moe` for NVFP4,
and `tilewf_kernel` only in eligible TWF batches. Include empty ranks, sparse
traffic, mixed stages, tile boundaries and long Decode. New package correctness
cannot be inferred from conversion unit tests.

For WorkBuddy, use deterministic seeds 8/28/2026 and identical request files,
placement, model and execution settings. Keep JIT, warmup, profile and measurement
separate. Report K input tokens/s for Prefill and K output tokens/s for Decode;
include collective/routing overhead, completed GPU work and request admission/tail
boundaries. Report median and sample SD with the seed/session/round hierarchy.
The separately submitted WorkBuddy skill is optional; this example does not
require that branch or reference its unmerged local files.

## Source attribution

`runtime/upstream_tilewf.py` and `runtime/tilewf.cu` derive from the SGLang Waterfill
demo at `a85e2a43307d436477d096647066cd4899b64f87`. Local changes include timeout
failure, formatting, and the fixed-placement wrapper/Decode-idle guard. The loader
and quantization bridge came from the historical V3 experiments and now use
portable environment settings. File hashes are in `sources.sha256.json`.
SGLang's Apache-2.0 license is preserved in `licenses/SGLang-LICENSE`.
External source ownership and exact revisions are recorded in
`dependencies.lock.json` and the repository's `THIRD_PARTY_NOTICES.md`.
