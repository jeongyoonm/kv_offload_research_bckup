# Preparation training-path mismatch

Branch: `experiment/glm53-training-path-mismatch-20261006`.
Artifacts: `/mnt/localnvme/glm53_training_path_mismatch_20261006`.
Serving code and defaults are unchanged.

## Question and measurement

Does training without cached references, pinned packing, and compact tokens
increase prediction error when serving with these input preparation options?

Use existing real GLM5.3 TP8 execution traces from the three alternating paired
80-request repeats in `/mnt/localnvme/glm53_compact_prep_20261006`. The six cells
contain 42 compute batches per preparation path. Within each paired repeat,
these three flags are the only policy differences. Workload manifests and KV
bytes are identical; all cells passed the earlier serving audit.

Train the existing estimator on those recorded execution labels in CPU code.
Evaluate on the separate 104 compute batches from the latest calibration study,
covering 600 requests. No held-out execution labels are used for fitting or
calibration. This measures forecast error, without another serving benchmark.

Two controls address selection effects:

- Balance training within repeat and exact request-shape stratum. Retain
  36 batches per path with identical feature sequences and observation order.
- Repeat balanced selection and ordering with 128 deterministic seeds. Also
  leave one entire paired repeat out of fitting and evaluate its execution
  records on both preparation paths.

The original saved checkpoint is an additional reference. Its retained
calibration history contains 19 cached observations among 128 total records;
the new balanced fitting histories contain 36 cached observations each.

## Timing boundary

In the V2 model runner, `timing.prepared()` occurs after input preparation and
`kv_connector.pre_forward()`, immediately before model forward submission.
`WorkerTiming.prepared()` records the start CUDA event. The final event joins
required output-copy work on a separate stream.

Consequently, the preparation wall interval preceding the marker is excluded
from the execution duration used as the fitting label. Ordering is verified on
all 1,504 rank/compute observations in this analysis. Concurrent preparation of
the next batch can still affect host submission during this interval. Changed
GPU start time can also change transfer overlap. This boundary does not exclude
all indirect effects of preparation policy.

## Results

Primary results use the recent **shape calibration** mode, with the same
algorithm for both training paths. The balanced rows average prediction-error
metrics over the 128 training selections; they are not an ensemble predictor.

| Training data | Held-out MAE ms | Error stddev ms | Signed bias ms |
| --- | ---: | ---: | ---: |
| Original reused checkpoint | 50.26 | 63.21 | +20.51 |
| Warmed earlier preparation path, balanced | 39.70 | 51.71 | -1.79 |
| Warmed optimized preparation path, balanced | 40.20 | 52.07 | +8.29 |

Define mismatch penalty as MAE when trained on the earlier path minus MAE when
trained on the optimized path, evaluating on the same 104 records. Its average
is **-0.50 ms**, so matching preparation flags provides no consistent error
reduction. The 5th–95th percentile across selection/order seeds is
**-4.01 to +2.86 ms**. This describes sensitivity, not a confidence interval.
The fixed seed-zero comparison favors matching by 0.68 ms, illustrating why
the sign of a single selection should not determine the conclusion.

Using all 42 batches per path without balancing gives shape MAE 41.45 ms and
43.31 ms, respectively. Seed-zero leave-one-repeat-out evaluation on optimized
cells also does not favor optimized-path training: mean MAEs are 47.95 ms for
earlier-path training and 54.78 ms for optimized-path training. These small
samples are sensitive to the observed execution noise.

Both warmed training populations improve MAE by roughly 10 ms compared with
the original checkpoint. More representative cached-workload fitting and
calibration appears more useful here than matching the three preparation flags.
This comparison changes fitting population and history composition as well as
the number of cached samples; it does not isolate their individual effects.

The JSON also retains legacy/raw results, input hashes, exact batch-shape label
comparisons, and each held-out fold. Legacy correction is substantially more
sensitive to training ordering than the recent shape correction.

## Interpretation and limits

This measurement does not support preparation-path mismatch as a major source
of the current calibrated prediction error. Direct preparation time is outside
the fitting interval, and fitting the optimized traces produces no consistent
accuracy gain beyond earlier-path fitting with matched coverage.

This is a retrospective estimate, not an exact causal percentage. Batch shapes
and training feature sequences are matched, but batch membership, transfer
overlap, and submission state are not forced identical. The later validation
run uses a 5 ms compute sealing deadline absent from the training cells. There
are only three independent preparation comparison pairs, and the 128 seeds do
not create additional independent GPU experiments. No throughput improvement
can be inferred from these forecast-error comparisons.

The next useful training experiment should cover the relevant cached batch
sizes and overlap states; matching input preparation alone has low priority
based on this evidence.

## Reproduction

```bash
.venv/bin/python experiments/training_path_mismatch_20261006/analyze.py \
  --training-root /mnt/localnvme/glm53_compact_prep_20261006 \
  --validation-root /mnt/localnvme/glm53_seal_calibration_20261006 \
  --output /mnt/localnvme/glm53_training_path_mismatch_20261006
```

Assertions verify accepted source cells, matching workload/KV bytes, exact
policy differences, all-rank event validity and timing-marker ordering, suffix
token accounting, disjoint execution records, and identical balanced feature
sequences. The analysis completes successfully. Compact evidence is under
`results/`; full paths and hashes remain in `comparison.json`.
