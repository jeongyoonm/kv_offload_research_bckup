# Explicit geometry calibration

Branch: `experiment/shape-basis-calibration-20261010`.
Parent: `e9ed0e3`, including the optional joint-support guard.
Keep GLM 5.3 TP8 and every selected worker, cache and model setting.

The five-feature correction failed crossed context/suffix warmups and
measured short-context/long-suffix batches, despite marginal support.
Its short-context/long-suffix mean error reached -300 ms. The original
90 ms correction was too long by 99 ms on long-context/short-suffix
batches. These opposite errors motivate explicit geometry features
and broader representative cached serving labels.

Schema 3 selects only known causal shape features: request count,
suffix-token sum/mean/maximum, context sum/mean, prefix-index work,
and the frozen raw prediction. Elapsed execution time is forbidden.
Fitting and inference share the query builder. Missing required work
features, unsupported joint geometry, or unsupported marginal shape
retain the original 90 ms fallback. Schemas 1 and 2 keep their behavior.

Fit small ridge candidates for direct duration and residual correction.
Hold every arm/workload of an allocation repeat out together. Select
by workload-macro development MAE. A direct model is encoded as
`raw - direct` bias, leaving the base forest weights/history intact.
The artifact is frozen before independent serving confirmation; no
training occurs during inference. Record ranks to detect deficient
coverage and retain every candidate score, not just the winner.

Use completed `ns1`, `yc1` and `yc2` adjacent-compute labels for the
broader development fit. The initial `yc1` results remain valid for
their originally frozen model; those labels become development data
only for subsequent fits. Do not claim the refit's development MAE
as independent serving accuracy. New confirmation seeds must differ.

The matched harness supports `--arm-fit ARM=PATH` for distinct frozen
calibrations. Control retains the original unmodified checkpoint.
Only deadline lead can differ in policy; unchanged base weights and
round-tripped states are asserted for every arm before launch.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" .venv/bin/python \
  experiments/shape_basis_20261010/fit.py FRESH_FIT \
  --rows ORIGINAL_ROWS LIVE_ROWS_GZ CROSSED_ROWS_GZ
```

Runtime tests cover suffix sensitivity at equal context/count, missing
attention-work fallback, forbidden elapsed features, and prior guard
behavior. Run the CPU mechanism gate before any schema 3 serving run.
No new checkpoint is loaded into the ongoing `yc2` allocation.
