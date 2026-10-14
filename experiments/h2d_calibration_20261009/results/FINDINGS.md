# Transfer-only startup calibration: final findings

## Conclusion

Calibration identifies a real transfer-size dependence on the current hardware.
Depth 2 captures most of the pipeline benefit. Smaller chunks improve small
restoration jobs substantially; the existing 128 MiB/depth 2 setting remains
near the plateau for long-prefix jobs. Select a startup setting for the expected
transfer-size mix.

The predeclared equal-profile rule selects 16 MiB/depth 2 as the shallowest
near-plateau compromise. Its strict held-out confirmation improves the
equal-weight geometric bandwidth score by 7.2%. This is a calibration score, not
a serving throughput estimate. For the existing long-prefix workload, retain 128
MiB/depth 2 pending a future serving comparison.

## Strict held-out confirmation

Eight GPUs transfer concurrently. Decimal GB/s per GPU uses all-rank completion
time, including submission and copy gaps. Each trial queues four FIFO loads per
rank. Eight matched randomized repetitions use a fresh allocation separate from
the screen. An all-rank barrier after timing ensures validation runs only after
every measured copy has drained.

<!-- markdownlint-disable MD013 -->

| Chunk MiB | Depth | 4K GB/s/GPU | 16K GB/s/GPU | 60K GB/s/GPU |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2 | 33.94 | 44.86 | 45.48 |
| 64 | 2 | 36.91 | 44.48 | 45.24 |
| 16 | 2 | 42.77 | 44.57 | 44.71 |
| 16 | 8 | 42.71 | 44.98 | 44.58 |
| 8 | 4 | 44.15 | 44.34 | 43.79 |

| Chunk MiB / depth | Prefix restoration | Paired bandwidth gain vs 128/2 | Nominal 95% interval |
| --- | --- | ---: | --- |
| 16 / 2 | 4k | +26.02% | [+25.31%, +26.73%] |
| 16 / 2 | 16k | -0.66% | [-1.75%, +0.43%] |
| 16 / 2 | 60k | -1.70% | [-2.23%, -1.17%] |
| 16 / 8 | 4k | +25.82% | [+24.27%, +27.40%] |
| 16 / 8 | 16k | +0.26% | [-1.13%, +1.67%] |
| 16 / 8 | 60k | -1.98% | [-3.47%, -0.46%] |
| 8 / 4 | 4k | +30.08% | [+29.02%, +31.14%] |
| 8 / 4 | 16k | -1.17% | [-2.27%, -0.07%] |
| 8 / 4 | 60k | -3.69% | [-5.48%, -1.87%] |
| 64 / 2 | 4k | +8.75% | [+7.95%, +9.57%] |
| 64 / 2 | 16k | -0.86% | [-2.54%, +0.85%] |
| 64 / 2 | 60k | -0.54% | [-1.27%, +0.20%] |

| Setting | First 4K load ready on all ranks ms | First 16K load ready ms | First 60K load ready ms |
| --- | ---: | ---: | ---: |
| 128 MiB / 2 | 5.847 | 16.625 | 61.475 |
| 16 MiB / 2 | 4.674 | 16.757 | 62.378 |
| 8 MiB / 4 | 4.565 | 16.765 | 63.226 |

<!-- markdownlint-enable MD013 -->

On 4K jobs, 8 MiB/depth 4 reduces first all-rank readiness from 5.847 ms to
4.565 ms (21.9%). On 60K jobs the same setting loses 3.7% sustained bandwidth.
The 16 MiB/depth 2 compromise gains 26.0% on 4K and loses 1.7% on 60K; its 16K
difference does not establish a repeatable gain or loss.

## Sweep, confirmation, and selection

The 28-configuration screen covers chunks 8–512 MiB and depths 1/2/4/8. Five
repetitions over three sizes give 420 all-rank trials. Screening selected 16
MiB/depth 8 as the best equal-profile score and 16 MiB/depth 2 as the
compromise. Confirmation nominees were fixed from the screen: those two
settings, the baseline, the 8 MiB/depth 4 short-job winner, and the 64 MiB/depth
2 neighbor.

An initial 120-trial confirmation reproduced the size dependence. Validation
originally followed rank-local completion, so other ranks could still be
copying. A final 120-trial confirmation added an all-rank post-timing barrier to
exclude all validation work from concurrent measurement. The final tables and
startup profile use this strict confirmation. Initial evidence and the
three-cell baseline smoke run are retained separately. No model compute ran in
any stage.

At nominal 128 MiB, depth 1 measured about 26.7 GB/s/GPU. Depth 2 measured 34.1
on 4K, 45.5 on 16K, and 46.0 on 60K in the screen. Depth 1 synchronizes after
every chunk; depth 2 submits another chunk before waiting for an older one.
Depths 4 and 8 add little on larger jobs.

The unchanged 4,096-descriptor limit also bounds actual chunk bytes. Nominal 128
MiB produces 2, 7, and 25 chunks per 4K/16K/60K job; 16 MiB produces 12, 45, and
166. The 256/512 MiB settings can encounter descriptor limits before byte
limits. Chunk plans are recorded per trial.

The selection rule gives equal weight to the three size-specific bandwidth
ratios, then chooses the shallowest and smallest setting within 95% of the best
score, with every size at least 97% of baseline. This preference permits small
long-job regressions in exchange for small-job gains. It should be changed to
reflect the intended workload mix when used during startup. The compromise
remains 16 MiB/depth 2 in the strict confirmation. The highest equal-profile
score among the confirmed nominees is 8 MiB/depth 4, narrowly ahead of 16
MiB/depth 8; this does not establish a unique optimum.

Confidence intervals use paired Student-t intervals on log bandwidth ratios,
conditional on the allocation, and are unadjusted for multiple comparisons. The
full screen took 84.8 seconds including worker initialization/cleanup. Strict
confirmation duration is recorded in `confirmation_strict/summary.json`. A
production startup probe can use fewer candidate settings and already-allocated
buffers.

## Startup use and scope

The saved `startup_profile.json` records hardware identity, KV geometry,
selection rule, baseline, compromise candidate, and strict per-size
measurements. A startup calibrator can select a fixed setting using
representative restoration volumes and the expected request mix. A
startup-calibrated size lookup is another possibility if one fixed setting is
insufficient; per-job chunk selection would require a native API change.
Automatic server startup calibration is not added here, and no server/default
configuration changed.

These results establish transfer-only calibration feasibility. Serving
throughput and model-compute/copy interference were intentionally excluded. This
profile covers the current hardware with reconstructed GLM FP8 KV layout and
cannot be assumed optimal for another model or hardware.

## Geometry, boundaries, and verification

Reconstruct GLM FP8 geometry: 78 × 36,864-byte tensor pages and 21 × 8,448-byte
indexer pages, totaling 3,052,800 bytes per CPU block per rank. Restored
prefixes contain 4,096, 15,552, and 58,368 tokens. Per-load bytes per GPU are
195,379,200, 741,830,400, and 2,784,153,600 respectively.

Each rank allocates 2,048 private CPU blocks (6.25 GB) and 1,024 GPU blocks
(3.13 GB). CPU source rows are strided, GPU pages packed per tensor, and block
mappings randomized and fixed. This is reconstructed geometry, not replay of
actual serving descriptors or allocation of the full 2.13 TB CPU cache.
NUMA-local placement, disjoint CPU affinity, and process-local THP disabling
match serving placement; no global settings change.

Bandwidth excludes descriptor construction, warmups, allocation, population,
registration, and validation. Completion, status, FIFO order, byte counts,
configuration, and chunk plans are audited. Sample checks cover
first/middle/last blocks and first/last 32 bytes in every tensor page after
completion; they are not full-byte validation. All checks passed. The existing
native FIFO/lifetime suite passed 9 tests. All owned worker processes and GPU
contexts were released.

Across screening and both confirmations there were 660 all-rank trials and
21,120 measured rank-local loads. The baseline smoke run is separate.

Artifacts: `/mnt/localnvme/h2d_calibration_20261009`. Source branch:
`experiment/h2d-startup-calibration-20261009`. See `sweep/calibration.png`, each
stage report/comparison, and raw `events.jsonl` for full evidence.
