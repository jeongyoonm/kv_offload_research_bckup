# Expanded GLM 5.3 TP8 predictor exploration

Artifacts: `/mnt/localnvme/glm53_predictor_expanded_20261007`.
Implementation branch: `experiment/glm53-expanded-architecture-20261007`.

Start after the dataset coverage evaluation and its live comparisons finish.
Reuse the expanded corpus with GLM 5.3, TP8, and the tested 64K context limit.
Keep the existing four-hour TP8 allocation cap; this follow-up uses CPU only.
Every candidate uses frozen model inference. No inference-time fitting is timed.

## Architecture comparison

Repeat all nine designs: fixed and tuned Extra Trees plus boosting, tree and MLP
bases alone, tree/MLP crossed stages, two MLP stages, and direct tree/MLP models.
Keep the previous twelve configurations per tuned component and training-only
normalization. Add three training seeds. Compare wide-base data against a matched
control in which base, correction, and direct models see the same forecast rows.

Pool original training, previously examined historical replay, fresh training,
and fresh development records. Exclude the current final-test set from this pool.
Use source-mass-matched workload weights across all alternatives. Reconstruct
features from actual execution geometry and causal runtime snapshots; CPU cache
hit rate, episode labels, workload labels, and end markers are not input features.

Use three disjoint outer episode folds, with an inner 20% episode validation split
for tuning and three additional episode folds for correction training. Add four
range tests that exclude long contexts (at least 48K), long individual suffixes
(at least 8K), large batches (at least eight requests), and heavy recorded transfer
activity (at least four loads). Purge matching base-only rows as well as forecasts
from each range's training data. Keep priming and target rows in one episode.

Select architecture using the completed outer validation results. Then score the
locked winner on the current final-test corpus. This final corpus is a reused
benchmark after the coverage study; it is not a new independently collected test.
Report episode-bootstrap intervals, workload-macro MAE, MAE/MAPE, p95 error,
directional deadline errors, convergence, training time, and inference latency.
Benchmark native trees and exported NumPy MLP arrays sequentially after fitting.

## Further improvements

After architecture results, use their error breakdowns to select the next step.
Prioritized candidates are MAE-aligned tree losses, pooling/weighting sensitivity,
and cross-fitted calibration or specialization using available geometry/load
features. Each change needs a separate branch and matched validation controls.
Log reasons for trying or rejecting each step. Inspect repeated-geometry variation
as a diagnostic, without presenting it as a proven irreducible noise floor.
Do not tune on final benchmark outcomes or infer throughput gains from replay.

The supervisor records its deadline, stops owned CPU processes at that deadline,
persists completed work, and writes an explicit report of unfinished comparisons.

The verified pool has 8,486 base rows and 4,994 causal forecasts, including 1,235
fresh forecasts. Its 192-row reused benchmark has no training episode overlap.
The architecture study runs 42 tasks (seven split types, two data controls, three
seeds). Follow-up changes reuse each task's training-only selected configuration:
legacy source weighting, full-architecture OOF global/group calibration, an
equal blend with the fixed-tree control when the architectures differ, doubled
fresh weight when fresh validation MAE exceeds historical MAE by 50%, and
absolute-error boosting when the selected architecture contains boosting.

Freeze `final_selection.json` before `benchmark.py` opens benchmark labels.
`branches.py` exports three frozen checkpoints per candidate and records each on
an individual branch. `final_report.py` writes standalone PNG/PDF comparisons,
convergence data, range tests, paired intervals, and export evaluation timings.
Run `verify.py` for the cheapest calibration-leakage and signed-export checks.
