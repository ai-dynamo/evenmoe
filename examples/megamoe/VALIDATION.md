# Validation record

## Checks completed for this packaged source

- Full repository Ruff check: passed, including the new example and tests.
- CPU unit suite in an isolated existing SGLang runtime, no GPU devices exposed:
  **356 passed, 3 skipped, 21 integration tests deselected**.
- Included in that suite: 31 new tests/cases for quantization, Decode/idle and
  mixed-stage selection, launch restrictions, complete placements, shared-slot
  preservation, replica recovery, and installer preflight/idempotence.
- Six patches verified against the pristine pinned SGLang/FlashInfer files:
  all base and output hashes matched; installation and a second idempotent
  installation passed. No live service files were modified.
- Python 3.10 syntax compilation and shell syntax check passed.
- Placement CLI checked on synthetic counts: 61 complete 288-slot mappings.
  Synthetic counts test the interface, not calibration quality.

CPU tests do not establish GPU numerical correctness, collective liveness,
serving quality, or speedup. Historical experiments used a different installed
runtime and the old package namespace. Their throughput numbers are not carried
forward as results for this PR.

## Required before declaring this integration ready

1. Complete a clean Docker build from the public pinned base and external source
   revisions, record image digest and installed dependency versions, and verify
   SGLang's effective command-line arguments.
2. Compare FP8/NVFP4 operator outputs against the reference, including masks,
   shared-expert scaling and frozen expert placement; record precision error.
3. Exercise all eight ranks, zero-token and sparse ranks, 63/64/65-row tile
   boundaries, mixed Prefill/Decode transitions, and long Decode. Verify identical
   collective order and real server warmup after every configuration change.
4. Trace every rank and confirm the requested native backend; TWF must appear
   only in globally pure Decode. Check that baseline has no TWF or redundant slots.
5. Run matched Prefill/Decode smoke workloads and then the fixed WorkBuddy matrix.
   Separate backend precision comparisons from original Waterfill versus TWF;
   preserve raw timing, request counts, failures and independent sessions.

These GPU and clean-build gates remain pending. Keep the PR in draft until they
are satisfied; do not interpret the example's existence as a general support claim.
