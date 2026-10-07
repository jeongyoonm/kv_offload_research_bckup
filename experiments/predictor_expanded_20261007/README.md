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

If a fresh held geometry-range MAE (context, suffix, or batch size) exceeds twice
fresh episode-CV MAE and exceeds 10 ms,
also test a positive ridge geometry trend with squared/absolute tree residuals.
It uses existing causal features and training-only scaling/alpha selection.
This tests continuous extrapolation where tree leaves remain bounded. The linear
and MLP exports are CPU research artifacts requiring serving integration before
deployment. Benchmark labels cannot trigger or tune this additional experiment.

When the selected architecture has an Extra Trees base, also replace leaf means
with weighted training medians while retaining its exact partitions. Compare
this alone and, where available, together with absolute-error boosting. This
aligns the point prediction with MAE without changing inference layout or cost.
If an MLP or crossed-stage design leads, retain a tuned-tree alternative with
ordinary means/squared loss, median leaves alone, absolute residual loss alone,
and their combination. This tests whether the training objective changes the
architecture conclusion; capacities still come from inner training splits.

Expanded-data validation found that an MLP base can produce a negative raw
duration estimate, which is still a valid intermediate input to correction.
The log-input MLP uses `sign(x) * log1p(abs(x))` consistently in fitting and frozen
evaluation. This exactly preserves the previous transformation for nonnegative
inputs and avoids NaNs for signed intermediate estimates. The abandoned initial
tasks had no completed checkpoints; their attempt log remains preserved.

Training-only diagnostics found matching two-request shapes with 69–92 ms of
CUDA marker enqueue lead and 72–78 ms subsequent GPU span, compared with roughly
zero lead and 125 ms span in fresh samples. The native record's enqueue word is
set immediately before recording the forward-start CUDA event. Test marker-lead
min/max/median/std plus rank-start skew; remove skew and remove lead separately
as feature controls. These inputs use only immutable enqueue/start words already
observable by forecast time. They exclude completion, elapsed-duration, and end
words. Trigger after ordinary comparisons when historical CV MAE remains above
5 ms and at least thirty pooled forecasts have lead at least 50 ms. Keep selected
training-only capacities unchanged. This association is consistent with hiding
CPU kernel submission under GPU backlog; CPU replay cannot confirm that mechanism.
The added feature schemas require serving integration before deployment.

Finally, repeat all nine architectures with fresh fitting rows removed from each
episode fold. Keep the same held episodes and capacities selected on the expanded
training folds; recompute normalization and workload weights on historical rows.
This control isolates fitting-data coverage and gives historical-only models
access to expanded-training hyperparameters. It does not recreate independently
tuned historical deployment. Use only fresh episode-validation labels for this
comparison; do not change the frozen selection from these final reporting runs.
