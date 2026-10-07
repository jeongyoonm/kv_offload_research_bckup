# Launch-envelope predictor experiment

GLM 5.3 / TP8 / FP8 KV, CPU-only analysis. All source data and artifacts reside
under `/mnt/localnvme`. No additional GPU collection is authorized by this study.

## Motivation and protocol

Prior measurements showed identical request geometry taking approximately 72–78
ms when execution was queued ahead and 125 ms when it was submitted without
that lead. These observations motivate, but do not establish, this model:

`duration = max(positive workload estimate, host envelope estimate - queue lead)`.

Both affine estimates are latent parameters fitted to total observed execution
latency; neither is a separately measured GPU/CPU cost. Nonnegative geometry
coefficients do not guarantee extrapolation accuracy. Runtime transfer features
are measured at the forecast, without future completion timestamps.

All-rank start minus the latest CPU start-event enqueue timestamp is known at
forecast time. This approximate global lead differs from the earlier per-rank
features. Every timestamp must precede the forecast. GPU end timestamps are
excluded.

Use the expanded study's 42 episode/range folds and three seeds. Forecast rows
alone train both components. Wide and matched regimes consequently share fitting
rows, but inherit different correction capacities from the previous training-only
architecture tuning. Four envelope regularization values are selected by an inner
20% episode split. Three purged episode folds produce correction targets.

Compare an uncorrected envelope and corrections using squared-loss boosting,
absolute-loss boosting, and MLP. Freeze the candidate using fresh episode macro
MAE, subject to at most 10% historical MAE regression against fixed trees. Evaluate
192 final rows only after selection. They are a **reused benchmark**, not a new
blind test; benchmark labels never select or tune a model. No throughput or TTFT
benefit can be inferred from this CPU experiment.

## Reproduction

Use the existing evaluation venv with BLAS/OpenMP thread counts set to one:

```sh
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_launch_envelope_20261007/envelope.py --workers 16
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_launch_envelope_20261007/envelope_evaluate.py
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_launch_envelope_20261007/envelope_evaluate.py --benchmark
```

Artifacts: `/mnt/localnvme/glm53_launch_envelope_20261007`.
Inference requires exported coefficients and optional tree/MLP weights only;
there is no fitting at inference time. Serving integration remains separate work.

## Suffix-support fallback follow-up

A separate branch evaluates two training-bound fallback rules after the envelope
validation revealed long-suffix extrapolation errors. Suffix-only fallback checks
individual maximum suffix length; geometry fallback checks the same eleven
geometry/causal-I/O coordinates as the earlier conservative guard. Out-of-range
rows use the same fold's fixed trees. No new fitting or held-out-label thresholds
are introduced. Selection uses episode macro MAE and historical regression gates,
with exact ties broken by mean withheld-range MAE normalized to fixed trees.
The benchmark remains reused and exploratory.

Run `envelope_guard.py` before `envelope_guard.py --benchmark`. Artifacts are under
`/mnt/localnvme/glm53_envelope_guard_20261007`.
