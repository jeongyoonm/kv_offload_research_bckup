# CPU predictor model and architecture comparison

GLM5.3 TP8; existing cached-prefill traces only. All fitting and inference
ran on CPU. No serving predictor was changed and no GPU cells were run.

## Findings

1. With the normal wide-base data, the refitted current architecture
   achieves 13.63 ms MAE. The direct MLP achieves
   16.24 ms; the two-stage MLP achieves
   14.68 ms. These support the existing
   design under this bounded configuration search.
2. An MLP base with tree correction is competitive at
   13.88 ms. Its paired difference versus
   current trees is +0.25 ms, with a 95%
   interval [-0.21, +0.73].
   The interval includes zero, supporting both designs in this
   episode evaluation.
3. Additional base-training batches account for a substantial part of
   the advantage: current-tree MAE changes from
   15.22 ms on matched data to
   13.63 ms on wide data; macro MAE changes from
   28.25 to
   15.90 ms. Matched-data contrasts
   separate this data benefit from the structure of two-stage models.
4. Unseen-workload macro MAE is
   38.79 ms for current trees,
   77.43 ms for two-stage MLP, and
   45.93 ms for direct MLP.
   The per-workload table shows whether one difficult case dominates.
   Sample-weighted held-workload MAE favors direct MLP instead:
   43.44 ms versus
   51.74 ms for current trees, driven by
   improvements on both large 60k workload holdouts. Results favor
   different designs depending on workload and error metric.
5. CPU costs are measured independently on historical-replay checkpoints
   selected in that split, rather than assigning one split's timing to
   another split's fitted model. Both tree and MLP deployments reuse
   frozen parameters and need no inference-time training library.

Keep the current serving architecture while interpreting the full
accuracy, robustness, and timing results together. The study provides
no new evidence of serving throughput or admission gains from changing
the regression architecture.

## Episode-held-out comparison

Three disjoint outer episode folds predict each of 3,759 forecasts once.
Entire test episodes are absent from base and correction training. Each
fold tunes on its training episodes only; corrections use cross-fitted
base predictions. The wide base can use additional compute-only batches.

<!-- markdownlint-disable MD013 -->
| Design | MAE (ms) | MAPE | p95 error (ms) | Macro MAE (ms) |
| --- | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | 13.63 | 3.78% | 44.69 | 15.90 |
| Tree base + tree correction, tuned | 13.10 | 3.66% | 44.23 | 15.06 |
| Tree base only | 17.34 | 4.86% | 60.13 | 16.46 |
| MLP base only | 18.98 | 5.57% | 60.74 | 19.22 |
| Tree base + MLP correction | 14.59 | 4.27% | 47.96 | 17.83 |
| MLP base + tree correction | 13.88 | 3.94% | 46.31 | 18.23 |
| MLP base + MLP correction | 14.68 | 4.28% | 48.63 | 20.53 |
| Single MLP | 16.24 | 4.60% | 52.24 | 30.77 |
| Single boosted-tree model | 16.40 | 4.41% | 51.81 | 39.46 |

<!-- markdownlint-enable MD013 -->

## Matched-data architecture control

The base receives exactly the forecast training rows available to the
direct models. This removes the extra-base-data advantage.

<!-- markdownlint-disable MD013 -->
| Design | MAE (ms) | MAPE | p95 error (ms) | Macro MAE (ms) |
| --- | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | 15.22 | 4.18% | 51.14 | 28.25 |
| Tree base + tree correction, tuned | 15.73 | 4.36% | 53.91 | 29.21 |
| Tree base only | 18.40 | 5.07% | 63.20 | 31.37 |
| MLP base only | 20.82 | 6.49% | 68.69 | 26.74 |
| Tree base + MLP correction | 15.18 | 4.34% | 53.96 | 28.80 |
| MLP base + tree correction | 15.78 | 4.97% | 48.52 | 23.95 |
| MLP base + MLP correction | 15.90 | 4.98% | 52.22 | 25.85 |
| Single MLP | 16.24 | 4.60% | 52.24 | 30.77 |
| Single boosted-tree model | 16.40 | 4.41% | 51.81 | 39.46 |

<!-- markdownlint-enable MD013 -->

### Effect of additional base-training batches

Same held-out forecasts and correction/direct training rows; only the
base-training corpus changes. Each regime tunes on training data.

<!-- markdownlint-disable MD013 -->
| Design | Wide MAE (ms) | Matched MAE (ms) | Wide minus matched, 95% CI |
| --- | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | 13.63 | 15.22 | -1.60 [-2.98, -0.60] |
| Tree base + tree correction, tuned | 13.10 | 15.73 | -2.63 [-4.12, -1.57] |
| Tree base only | 17.34 | 18.40 | -1.06 [-2.88, +0.15] |
| MLP base only | 18.98 | 20.82 | -1.84 [-2.65, -1.12] |
| Tree base + MLP correction | 14.59 | 15.18 | -0.59 [-1.86, +0.34] |
| MLP base + tree correction | 13.88 | 15.78 | -1.90 [-2.61, -1.28] |
| MLP base + MLP correction | 14.68 | 15.90 | -1.22 [-2.09, -0.49] |
| Single MLP | 16.24 | 16.24 | +0.00 [+0.00, +0.00] |
| Single boosted-tree model | 16.40 | 16.40 | +0.00 [+0.00, +0.00] |

<!-- markdownlint-enable MD013 -->

## Workload-held-out comparison

Eight workloads with at least 60 forecasts are each excluded from both
training stages. For homogeneous tests, mixed-workload training batches
are excluded too. Macro MAE treats the eight test workloads equally.

<!-- markdownlint-disable MD013 -->
| Design | MAE (ms) | MAPE | p95 error (ms) | Macro MAE (ms) |
| --- | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | 51.74 | 12.22% | 147.19 | 38.79 |
| Tree base + tree correction, tuned | 53.11 | 12.40% | 148.65 | 39.10 |
| Tree base only | 51.71 | 11.42% | 156.29 | 42.00 |
| MLP base only | 68.12 | 20.02% | 150.06 | 102.32 |
| Tree base + MLP correction | 44.46 | 12.18% | 123.65 | 37.56 |
| MLP base + tree correction | 61.31 | 17.07% | 146.78 | 97.29 |
| MLP base + MLP correction | 54.51 | 16.14% | 136.46 | 77.43 |
| Single MLP | 43.44 | 13.61% | 107.11 | 45.93 |
| Single boosted-tree model | 71.49 | 14.57% | 204.71 | 51.79 |

<!-- markdownlint-enable MD013 -->

### Per-workload MAE (ms)

<!-- markdownlint-disable MD013 -->
| Held-out workload | Samples | Current trees | Tuned trees | Tree + MLP | MLP + tree | MLP + MLP | Direct MLP | Direct tree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed_h100 | 68 | 24.05 | 23.07 | 28.94 | 437.46 | 337.52 | 92.75 | 36.05 |
| p16384_f95_h100 | 143 | 16.35 | 16.61 | 16.99 | 27.66 | 27.32 | 21.99 | 19.67 |
| p24576_f90_h100 | 114 | 13.75 | 12.88 | 24.37 | 10.04 | 16.63 | 17.59 | 36.44 |
| p32768_f90_h100 | 198 | 16.02 | 15.01 | 17.21 | 47.07 | 32.55 | 26.52 | 39.17 |
| p49152_f98_h100 | 269 | 38.14 | 40.72 | 46.65 | 56.13 | 48.44 | 48.05 | 56.74 |
| p61440_f95_h100 | 1357 | 78.87 | 81.93 | 59.36 | 56.78 | 54.60 | 41.43 | 125.46 |
| p61440_f99_h100 | 344 | 98.56 | 97.22 | 80.52 | 96.58 | 51.51 | 80.15 | 72.76 |
| p8192_f95_h100 | 1138 | 24.56 | 25.39 | 26.39 | 46.64 | 50.87 | 38.95 | 28.00 |

<!-- markdownlint-enable MD013 -->

## Replay of the previous evaluation

These 670 forecasts were previously inspected during predictor
development. This section reproduces the earlier comparison, but
is not a new blind test.

<!-- markdownlint-disable MD013 -->
| Design | MAE (ms) | MAPE | p95 error (ms) | Macro MAE (ms) |
| --- | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | 19.03 | 4.92% | 59.35 | 17.44 |
| Tree base + tree correction, tuned | 19.64 | 4.93% | 63.04 | 18.13 |
| Tree base only | 24.26 | 6.25% | 80.94 | 20.84 |
| MLP base only | 76.92 | 14.04% | 578.10 | 85.65 |
| Tree base + MLP correction | 19.97 | 5.02% | 59.48 | 19.40 |
| MLP base + tree correction | 72.80 | 12.65% | 576.49 | 82.88 |
| MLP base + MLP correction | 56.65 | 10.74% | 379.68 | 62.36 |
| Single MLP | 31.16 | 7.51% | 99.71 | 31.97 |
| Single boosted-tree model | 27.61 | 6.76% | 101.10 | 29.71 |

<!-- markdownlint-enable MD013 -->

### Archived checkpoint reproduction

The unchanged archived serving checkpoint gives 19.62443
ms MAE and 5.03686% MAPE, reproducing the previous
report. Refitting with the legacy arithmetic matches that checkpoint
within 0.00000252 ms.

The comparison runner uses NumPy summation for weight normalization;
the old trainer used Python summation. The normalized weights differ
by at most 9.41e-14, but this changes some
tied Extra Trees splits and the freshly refitted historical-replay MAE.
Features and labels are identical. Every newly fitted architecture in
this comparison uses the same NumPy-normalized weights. The archived
checkpoint is kept separate; it would be contaminated as a reference
for pooled episode/workload holdouts because it saw their training rows.
This sensitivity also cautions against interpreting tiny ranking gaps.

## Paired uncertainty

2,000 paired bootstrap draws resample whole episodes. Differences
below zero favor the alternative over the current fixed tree recipe.
These intervals describe episode sampling, conditional on these
fitted models; they do not include model refitting or tuning variance.

<!-- markdownlint-disable MD013 -->
| Design | Episode MAE difference (ms), 95% interval | Workload-held-out MAE difference (ms), 95% interval |
| --- | ---: | ---: |
| Tree base + tree correction, tuned | -0.53 [-0.79, -0.28] | +1.38 [+0.98, +1.75] |
| Tree base only | +3.71 [+3.04, +4.40] | -0.03 [-1.22, +1.21] |
| MLP base only | +5.35 [+4.62, +6.11] | +16.38 [+10.71, +22.59] |
| Tree base + MLP correction | +0.97 [+0.51, +1.40] | -7.28 [-8.95, -5.77] |
| MLP base + tree correction | +0.25 [-0.21, +0.73] | +9.57 [+4.26, +15.45] |
| MLP base + MLP correction | +1.05 [+0.48, +1.66] | +2.77 [-2.21, +8.04] |
| Single MLP | +2.61 [+1.56, +3.90] | -8.29 [-11.74, -4.92] |
| Single boosted-tree model | +2.77 [+1.11, +5.44] | +19.75 [+16.82, +22.88] |

<!-- markdownlint-enable MD013 -->

### Architecture contrasts on matched data

Each pair uses the same available training rows. Values are the MAE
of the first design minus the second, with episode-bootstrap intervals.
A negative difference favors the first design.

<!-- markdownlint-disable MD013 -->
| Designs | MAE difference (ms), 95% interval |
| --- | ---: |
| Tree base + tree correction, tuned vs Single boosted-tree model | -0.67 [-2.00, +0.23] |
| Tree base + tree correction, tuned vs Tree base only | -2.67 [-3.38, -2.03] |
| Tree base + MLP correction vs Tree base only | -3.21 [-4.29, -2.30] |
| MLP base + MLP correction vs Single MLP | -0.34 [-1.40, +0.45] |
| MLP base + MLP correction vs MLP base only | -4.92 [-5.81, -3.99] |
| MLP base + tree correction vs MLP base only | -5.04 [-5.79, -4.30] |

<!-- markdownlint-enable MD013 -->

## Frozen CPU inference

Historical-wide checkpoints; 3,350 randomized single-row calls per
entry, 200 warmups, one BLAS/OpenMP thread, after training finishes.
Native entries use the existing CPU tree walker. MLP uses frozen
NumPy dense arrays. Mixed models combine those paths. Feature
extraction, JSON loading, and live-engine contention are excluded.
Initialization measures object construction from decoded state.

<!-- markdownlint-disable MD013 -->
| Design | Backend | Mean (µs) | p95 (µs) | Init (ms) | JSON (KiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | numpy | 376.62 | 418.03 | 447.18 | 6307.7 |
| Current tree architecture, refitted recipe | native_tree | 31.53 | 39.03 | 409.39 | 6307.7 |
| Tree base + tree correction, tuned | numpy | 397.03 | 421.12 | 418.75 | 6383.6 |
| Tree base + tree correction, tuned | native_tree | 34.37 | 41.99 | 328.80 | 6383.6 |
| Tree base only | numpy | 183.16 | 190.73 | 396.82 | 6130.0 |
| Tree base only | native_tree | 20.49 | 26.53 | 332.12 | 6130.0 |
| MLP base only | numpy | 8.56 | 8.86 | 9.64 | 384.7 |
| Tree base + MLP correction | numpy | 218.28 | 243.82 | 448.83 | 6551.2 |
| Tree base + MLP correction | native_tree | 37.54 | 46.05 | 414.29 | 6551.2 |
| MLP base + tree correction | numpy | 97.81 | 120.65 | 9.48 | 436.4 |
| MLP base + tree correction | native_tree | 17.52 | 22.27 | 4.37 | 436.4 |
| MLP base + MLP correction | numpy | 16.83 | 22.55 | 0.60 | 403.3 |
| Single MLP | numpy | 7.55 | 10.22 | 0.15 | 124.5 |
| Single boosted-tree model | numpy | 261.51 | 303.67 | 25.24 | 371.0 |
| Single boosted-tree model | native_tree | 12.76 | 14.85 | 24.23 | 371.0 |

<!-- markdownlint-enable MD013 -->

## Error direction

Episode-held-out predictions. Early predictions can miss admissions;
late predictions can expose waits. These are prediction errors, not
measured scheduling effects.

<!-- markdownlint-disable MD013 -->
| Design | Mean bias (ms) | Under by >5 ms | Over by >5 ms | p95 under (ms) | p95 over (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current tree architecture, refitted recipe | +0.87 | 28.0% | 37.2% | 34.70 | 31.71 |
| Tree base + tree correction, tuned | +0.49 | 28.2% | 34.6% | 33.79 | 30.21 |
| Tree base only | -1.65 | 34.5% | 33.7% | 48.92 | 36.89 |
| MLP base only | -0.32 | 33.7% | 38.2% | 49.03 | 43.66 |
| Tree base + MLP correction | +0.61 | 30.0% | 39.3% | 36.96 | 33.13 |
| MLP base + tree correction | -0.68 | 33.2% | 34.2% | 37.16 | 30.07 |
| MLP base + MLP correction | -0.66 | 35.2% | 33.9% | 36.81 | 32.61 |
| Single MLP | -3.69 | 44.2% | 26.9% | 40.77 | 32.15 |
| Single boosted-tree model | -0.30 | 30.7% | 36.8% | 39.48 | 35.27 |

<!-- markdownlint-enable MD013 -->

## CPU tuning cost

Historical-wide split. Each component searches 12 configurations.
Shared bases are tuned once and reused across correction designs.
Times include fitting only. Selected-fit times and warnings describe
the inner tuning fit, not the later full-training refit. Final-refit
optimizer warnings were not retained by this runner.

<!-- markdownlint-disable MD013 -->
| Component | Search (s) | Selected fit (s) | Iterations | Warning |
| --- | ---: | ---: | ---: | --- |
| base_tree | 4.70 | 0.53 | — | No |
| correction_tree_tree | 3.99 | 0.36 | — | No |
| correction_tree_mlp | 50.81 | 5.43 | 402 | No |
| base_mlp | 87.46 | 19.53 | 869 | No |
| correction_mlp_tree | 4.01 | 0.26 | — | No |
| correction_mlp_mlp | 44.73 | 1.11 | 526 | No |
| direct_mlp | 44.59 | 4.34 | 722 | No |
| direct_tree | 3.48 | 0.38 | — | No |

<!-- markdownlint-enable MD013 -->

## Training and verification

- 16 completed outer tasks; 1536 tuning fits (12 candidates per component).
- 240 inner/full cross-fit splits have zero episode overlap.
- All exported predictions match fitted predictions: maximum difference
  0.00000199 ms; gate 0.0001 ms.
- Summed fold wall time: 5088.3 s; tasks overlap on four CPU workers, so this is
  not elapsed study time.
- Sum of recorded tuning-fit time: 3638.8 s.
- Optimizer convergence warnings in 332 tuning trials; 17 selected trials.
- Selected configurations, iterations, warnings, and component fit
  times are in `summary.json` and each fold's `tuning.json`.

## Interpretation limits

The study compares nine model/architecture combinations, including
the full base/correction family crossing and base-only ablations.
Base-only models receive 11 batch features; direct models receive
all 23 features; corrections also receive the base prediction.
Inputs and target scaling are fitted on training rows only.

Each family gets 12 configurations, with stagewise tuning. This is
equal candidate count rather than equal training time, parameter
count, or an exhaustive neural architecture search. A single inner
validation split and one initialization seed limit tuning stability.
Two-stage designs have separate base and correction searches; a direct
model has only one search. The current fixed recipe is not retuned.
Convergence warnings indicate a fixed optimizer budget may leave
some MLP candidates short of their optimum.

All underlying data were available during earlier engineering.
Grouped retraining prevents split leakage, but this is retrospective
evidence on one model and TP setting. It cannot establish generality
to other LLMs or predict throughput, TTFT, or recovered admissions
without new serving runs. No architecture is promoted here.

## Artifacts

Data and checkpoints:
`/mnt/localnvme/glm53_predictor_architecture_cpu_20261007`.
Plot: `architecture_comparison.png` and `.pdf`.
Machine-readable results: `summary.json`, `accuracy.csv`, `runtime.csv`.
Code and protocol: `experiments/predictor_architecture_cpu_20261007`.
