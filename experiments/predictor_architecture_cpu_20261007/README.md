# CPU architecture comparison

## Protocol fixed before evaluating alternatives

Use existing GLM5.3 TP8 cached-prefill traces only. No GPU execution, new serving
cells, or changes to the deployed predictor. Results and training checkpoints live
under `/mnt/localnvme/glm53_predictor_architecture_cpu_20261007`.

Compare the existing fixed tree recipe, tuned tree base plus tree correction,
tree and MLP bases alone, both crossed combinations of tree/MLP base and
correction, a two-stage MLP, and direct boosted-tree/MLP regressors. Base-only
models use the same 11 workload inputs; complete predictors also receive 12
causal runtime inputs. Two-stage
corrections additionally receive an out-of-fold base prediction. Actual durations,
GPU end markers, phase names, episode identity, and workload labels never enter
the feature matrix. Workload labels are used only for sampling weights and splits.

The target is physical completion duration after the latest rank GPU start:
`max(start_upper + CUDA_span) - max(start_upper)`. This is an initial duration
forecast, not a GPU scheduling counterfactual. Predictions are floored at 1 ms,
matching the existing serving model.

### Data and validation

- Historical training: 6,581 compute batches, 3,089 initial causal forecasts.
- Historical replay: the 670 forecasts behind the previous study. These are
  previously inspected development/evaluation data, not a new blind test.
- Pool the two sources for three disjoint outer episode folds. Entire episodes
  are excluded from both base and correction training in each outer fold.
- Leave each workload with at least 60 forecasts out of both training stages.
  For homogeneous workload tests also exclude mixed-workload rows from training
  to avoid indirect exposure to the held workload through mixed batches.
- Within each outer training set, use a fixed 20% episode validation split for
  hyperparameter selection. No outer test labels are used in selection.
- Corrections train on base predictions from three additional episode folds.
  Base-only compute batches from each held episode are excluded too.
- Score both a wide-base regime and a matched-data control on the historical
  replay and outer episode folds. The matched control gives the base exactly
  the same batch rows available to a direct predictor. This distinguishes the
  staged architecture from the benefit of additional base-training examples.

### Tuning and measurements

Each model family has 12 predeclared configurations. Extra Trees varies tree
count, leaf size, and depth; histogram boosting varies stage count, leaf size,
and regularization; MLP varies width/depth, Adam/L-BFGS, and raw/log input
scaling. MLP input and target normalization uses training data only. Automatic
random-row early stopping is disabled. Maximum optimizer iterations: 1,000.
Selection minimizes workload-macro MAE; sample weighting matches the existing
recipe (`1/max(50, workload_samples)`, normalized to mean one). This gives equal
candidate counts, not equal CPU training seconds. Stagewise tuning is explicit;
it is not exhaustive joint optimization of base/correction configurations.

Report frame MAE/MAPE, workload-macro and worst-workload MAE, p95 absolute error,
under/overprediction beyond 5 ms, paired episode-bootstrap uncertainty, optimizer
convergence, training time, initialization cost, and serialized size. Runtime
benchmarks use one CPU BLAS/OpenMP thread and one input at a time, outside an
active engine. Benchmark both NumPy-only export and the existing native CPU tree
walker; MLP uses exported dense arrays with no training-library call. Verify
exported predictions against the fitted models before accepting a result.

Frozen models are reused for every prediction. No inference retraining is timed.
Offline prediction time excludes feature extraction and engine contention.
Throughput, TTFT, recovered admissions, and new batch execution timelines cannot
be inferred reliably from these traces.

## Run

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
experiments/predictor_architecture_cpu_20261007/study.py \
--output /mnt/localnvme/glm53_predictor_architecture_cpu_20261007 --workers 4
```

The runner persists per-fold audits, tuning records, paired predictions, and
checkpoints, and resumes completed folds. Re-run the CPU latency benchmark
sequentially after all fitting processes exit.

Generate the audited report and figures after the runner completes:

```bash
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
experiments/predictor_architecture_cpu_20261007/reproduce.py \
--output /mnt/localnvme/glm53_predictor_architecture_cpu_20261007

/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
experiments/predictor_architecture_cpu_20261007/report.py \
--output /mnt/localnvme/glm53_predictor_architecture_cpu_20261007
```

Three CPU contract checks in `verify.py` cover frozen MLP inference with
training-only scaling, signed boosted-tree export, and selecting signed
corrections by their complete forecast error.
