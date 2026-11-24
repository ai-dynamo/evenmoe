# WorkBuddy V4: target non-hash layer imbalance

Historical diagnostic evidence, recomputed from saved per-step/layer rank counts; no new GPU run. DeepSeek-V4-Flash-0731 on 8 H20, native MXFP4 Marlin, TP8/EP8, seed 8, 512 generated tokens/request. Only target layers **3–42 (zero-based)** are included. Hash layers, draft layers and the shared expert are excluded from every metric below. These are Decode diagnostics, not the November 23 Prefill throughput run.

For each layer-step, assignment imbalance is `max(rank token–expert assignments) / mean(rank token–expert assignments)` over all eight ranks. Unique-token imbalance instead counts a token once per receiving rank, even when it selects multiple local experts. Both metrics are dimensionless; 1 means equal rank loads. Means below weight layer-steps equally, rather than taking a ratio of pooled totals.

Static EPLB uses one frozen, nonredundant placement calibrated on 128 independent WorkBuddy requests (512 output tokens each, no test prompt overlap), shared across both modes and concurrency levels. MTP uses DSpark; target verify includes rejected candidate computation.

## Full-batch comparison

Full-batch means the actual request batch equals configured global concurrency C; target verify can process multiple candidate tokens per request. All-step statistics, including drain steps, follow below so filtering remains visible.

| Mode / target phase | Global C | Assignment mean: baseline → static EPLB | Unique-token mean: baseline → static EPLB | Full-batch layer-steps: baseline / EPLB |
|---|---:|---:|---:|---:|
| Decode / DECODE | 128 | 1.4276 → 1.1748 | 1.2839 → 1.1396 | 20440 / 20440 |
| Decode / DECODE | 512 | 1.4132 → 1.1237 | 1.2724 → 1.0997 | 20440 / 20440 |
| Decode + MTP / TARGET_VERIFY | 128 | 1.4439 → 1.1396 | 1.2897 → 1.1133 | 3480 / 3480 |
| Decode + MTP / TARGET_VERIFY | 512 | 1.4401 → 1.1280 | 1.2883 → 1.1024 | 3480 / 3480 |

## All captured target non-hash layer-steps

| Mode | C | Placement | Steps | Layer-steps | Assignment mean | Assignment P95 | Assignment max | Unique-token mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| decode | 128 | baseline | 511 | 20440 | 1.4276 | 1.7604 | 3.2083 | 1.2839 |
| decode | 512 | baseline | 511 | 20440 | 1.4132 | 1.7268 | 3.0651 | 1.2724 |
| dspark | 128 | baseline | 200 | 8000 | 1.5411 | 2.2222 | 5.1111 | 1.3518 |
| dspark | 512 | baseline | 203 | 8120 | 1.4612 | 1.8889 | 4.2222 | 1.3021 |
| decode | 128 | static_eplb | 511 | 20440 | 1.1748 | 1.3021 | 2.8854 | 1.1396 |
| decode | 512 | static_eplb | 511 | 20440 | 1.1237 | 1.2330 | 2.9941 | 1.0997 |
| dspark | 128 | static_eplb | 231 | 9240 | 1.3201 | 2.0000 | 4.4444 | 1.2254 |
| dspark | 512 | static_eplb | 232 | 9280 | 1.2756 | 2.0000 | 4.0000 | 1.1969 |

## Interpretation and provenance

Static EPLB reduces mean rank-load imbalance in these captured runs. This is not evidence of throughput speedup or of multi-seed stability: synchronous routing capture affects timing, and this is one seed. Baseline and EPLB outputs are not token-identical: identical complete output sequences are 3/128 and 6/512 for Decode, and 1/128 and 9/512 for DSpark. Changed generated sequences can change routing, so this is not a replay of identical token routes under two placements.

Router outputs sometimes differ across TP ranks. These statistics use each owner rank's actual local expert selections and the recorded physical placement, rather than extrapolating rank 0 or summing replicated full routing tensors. Non-hash layer-step disagreement counts are preserved in the JSON.

[Machine-readable metrics and source SHA256](workbuddy-v4-target-imbalance.json) contain only target non-hash rows. Source campaign names and relative CSV paths identify the retained experiment artifacts; large raw traces and private machine paths are not bundled. Per-rank counts were recomputed into the ratios and checked against the original reports.
