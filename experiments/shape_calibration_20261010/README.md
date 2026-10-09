# Batch-shape finish calibration

Start from the native D2H/allocator-fixed GLM 5.3 TP8 prototype at
`44fa24c`. Keep H2D 8 MiB/depth 2, D2H 1 GiB/depth 2, private CPU cache
at 2,130,300,354,560 aggregate bytes, 96 GiB GPU KV per rank, original
model/weights/selector, optimized preparation and load-only readiness.

Fit five ridge coefficients to raw finish-prediction error using mean
context, request count, total context and raw prediction. Forecast age
is excluded. Initial training uses 76 logged compute predictions from
the preceding four paired 16K/60K episodes. Group validation holds out
entire repeat IDs across both workloads. The exploratory screen is not
a substitute for independent serving validation.

The fitted layer replaces the fixed correction inside its support.
Outside request/context/suffix bounds it falls back to the original
90 ms correction. Old checkpoints are unchanged. The new artifact is
frozen and round-trips with all learned base weights and history intact.
Raw prediction, applied correction and fallback status remain observable.

First comparison: control (90 ms correction, 100 ms lead), shape100
(frozen shape correction, 100 ms lead), shape50 (same correction, 50 ms
lead). Two matched 75-request episodes per arm/workload at concurrency
32, one output token. Workloads are 16K, unseen 32K, and 60K, with
prefix/suffix lengths 15552/832, 30720/2048 and 58368/3072. Seeds 9501/2
are independent of training seeds 9401-4. Arm order reverses on the
second repeat. All priming and per-arm 32-request warmups are excluded.

The harness also supports continuous 300-request episodes using 75
cached prefixes with distinct suffixes and one initial reset. Prefix
reuse comes from the previously validated native-lead harness. No
intermediate GPU reset occurs inside those episodes; serving checks
must still establish exact CPU restoration and zero GPU prefix hits.

Each serving cell must pass frozen-state, CPU restoration, no GPU hit,
no preemption, one-token output and all-eight-rank preparation checks.
No DMA audit, replay, stable top-k, or model execution changes are used.
CPU mechanism gate: 244 passed, 9 skipped before serving. Initialization,
priming and warmup are not serving throughput measurements.

Code/data are under `/mnt/localnvme/jeongyoon`. Branch:
`experiment/shape-calibration-20261010`. First allocation root: `sc1`.
The total exploration starts 2026-10-09 22:19:54 UTC and is bounded at
2026-10-10 05:19:54 UTC, reserving time to drain and archive. New
mechanisms require independent branches and evidence from prior runs.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/shape_calibration_20261010/fit.py ROWS FRESH_FIT

PYTHONPATH="$PWD" .venv/bin/python \
  experiments/shape_calibration_20261010/run.py \
  --output /mnt/localnvme/jeongyoon/sc1 \
  --fit /mnt/localnvme/jeongyoon/shape_calibration_20261010/frozen_fit.json
```
