# Transfer-only startup calibration

Branch: `experiment/h2d-startup-calibration-20261009`.
Checkout: `/mnt/localnvme/vllm_h2d_calibration_20261009`.
Artifacts: `/mnt/localnvme/h2d_calibration_20261009`.
Parent: `6e04097`; serving source and native copy implementation are unchanged.
AI assistance: Codex.

## Scope and method

Calibrate chunk size and pipeline depth on the current eight B300 GPUs
(reported as L20D) with concurrent H2D on all ranks. No model compute runs.
Use the production native async submitter, one ordered copy stream per rank,
the existing 4,096-descriptor limit, and all-rank readiness timestamps.

Reconstruct GLM 5.3 TP8 FP8 KV geometry: 78 tensors with 36,864-byte pages and
21 indexer tensors with 8,448-byte pages, totaling 3,052,800 bytes per CPU block
per rank. Prefixes contain 64, 243, or 912 blocks (4,096, 15,552, or 58,368
tokens), matching prior serving restoration volumes. This is reconstructed
page geometry, not a replay of captured serving descriptors.

CPU pages use the existing private mmap initialization and CUDA registration.
Each rank allocates 2,048 CPU blocks and 1,024 GPU blocks. Source pages are
strided by the CPU block width; destination pages are packed per tensor.
Randomized block mappings are fixed within the experiment. NUMA placement and
disjoint CPU affinity match the serving allocation; process-local THP is
disabled. No global machine settings change.

The screening matrix covers chunks 8, 16, 32, 64, 128, 256, and 512 MiB and
depths 1, 2, 4, and 8. Five randomized repetitions over three prefix sizes give
420 all-rank cells, each queuing four FIFO loads per rank. Two warmup loads per
configuration/profile are excluded. Descriptor construction, population,
registration, allocation, and validation are outside measurement boundaries.

That matrix describes the completed historical screen. Future calibration
sweeps accept only depths 1 and 2, including explicit `--configs` selections.
Historical results and plots retain the original matrix. This restriction
does not change native pipeline support or the serving configuration.

Effective bandwidth uses total bytes divided by elapsed time from the earliest
rank submission start to the latest rank's final native completion. Reported
per-GPU bandwidth divides aggregate bandwidth by eight. First-load readiness
and start skew are also measured. Native completion order and statuses are
audited. Sample validation checks first/middle/last blocks and the first/last
32 bytes of each tensor page after copies drain; it is not full-byte validation.

## Selection and confirmation

The screen scores configurations using the equal-weight geometric mean of
bandwidth ratios across the three sizes. Select the shallowest, then smallest,
configuration within 95% of the best score, while keeping every size at least
97% of the baseline's bandwidth. This is an explicit calibration preference,
not an estimate of serving throughput for an arbitrary request mix.

The screen selected 16 MiB/depth 2 as the plateau candidate and 16 MiB/depth 8
as the best aggregate score. A separate eight-repeat confirmation measures
those settings, the 128 MiB/depth 2 control, the 8 MiB/depth 4 small-transfer
winner, and the 64 MiB/depth 2 neighbor. Nominees are fixed before confirmation.
Paired intervals use repeat log bandwidth ratios; comparisons are nominal and
unadjusted for multiple testing. Independent allocations separate screening
from confirmation.

The initial confirmation validated after rank-local completion. The final
`confirmation_strict` repeats the same five nominees with an additional
all-rank barrier after timing, before any validation kernel. Final findings
use this strict confirmation; preliminary evidence is preserved separately.

```bash
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python benchmarks/kernels/calibrate_async_h2d.py \
  --output /mnt/localnvme/h2d_calibration_20261009/new_sweep

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/h2d_calibration_20261009/analyze.py \
  /mnt/localnvme/h2d_calibration_20261009/new_sweep --plot
```

Use a fresh output directory. The isolated calibration records results;
automatic application during server startup is not added in this experiment.
Transfer-only gains do not establish gains under concurrent model compute.

## Full result plots

Generate figures from the existing audited measurements without GPU work:

```bash
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/h2d_calibration_20261009/plot_results.py \
  /mnt/localnvme/h2d_calibration_20261009
```

The `plots/` directory contains PNG, PDF, and SVG exports:

- `full_sweep`: all 28 configurations, grouped by prefix size; bandwidth and
  first-load all-rank readiness, with the current 128 MiB/depth 2 marked.
- `confirmation_results`: all five final nominees, eight repeats each;
  bandwidth, readiness, and paired bandwidth gains with nominal 95% intervals.
- `sweep.csv` and `confirmation_strict.csv`: complete aggregate numerical data.

Copies are archived under `results/plots/`. Screening and strict confirmation
remain separate because only the latter prevents validation work from starting
until all ranks have finished their timed copies.
