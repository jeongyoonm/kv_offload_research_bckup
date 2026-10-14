# Seal calibration and estimator reuse

Branch: `experiment/glm53-seal-calibration-20261006`.
Serving implementation: `b048e95`.
Artifacts: `/mnt/localnvme/glm53_seal_calibration_20261006`.

## Calibration

The fitted nonnegative linear model and its four features are unchanged. The
`late_seal_calibration` experimental option selects:

- `legacy` (default): existing global correction from past corrected forecasts.
- `raw`: 75th percentile of actual duration divided by the current raw fitted
  duration, recomputed from the bounded training history.
- `shape`: the same raw calibration within the same compute regime (cached,
  cold, mixed, or decode). Use up to eight neighbors within distance 0.5 in
  log-transformed total tokens, context tokens and request count, requiring two
  neighbors. Otherwise use the available samples from that regime. Missing
  samples leave the factor at one.

Raw calibration can adjust estimates upward or downward. Final predictions
retain the existing 1 ms floor. Calibration reads the saved observations; it
does not fit new coefficients or learn from measured episodes when frozen.
Legacy error history is maintained independently of the selected calibration,
so a shared model supplies identical underlying state to every policy.

## Reusing a trained estimator

The loopback-only experimental KV scheduling endpoint accepts a `seal_estimator`
request with `action` equal to `status`, `export`, or `load`. Load includes a
`state` object. Serving requests, ordered batches and connector work must be
fully drained. State is JSON data; the server does not read client file paths.

Snapshots contain coefficients, fitting samples, calibration observations,
legacy errors and preparation histories. They exclude request identities,
GPU buffers, transport generations and runtime freeze/calibration flags. Schema,
feature definition, bounded histories and finite values are validated before
replacing the estimator. Compatibility covers model identity, dtype and
quantization, TP/PP/EP, KV dtype and block size, configured batch limits,
chunked prefill and graph settings, GPU name/capability, Torch version and static
H2D configuration. A mismatch is rejected, without automatically retraining.

The runner supports `--save-trained-model PATH` to train once and atomically save
a checkpoint. Subsequent runs use `--trained-model PATH`, load it before model
correctness probes, freeze it through priming and measurement, and verify full
state equality after every measured episode. This skips estimator training;
model loading, CPU cache initialization and prefix priming still occur.

For this experiment, the initial checkpoint is reconstructed from the preceding
run's saved training observations. It reproduces all 86 recorded frozen
forecasts exactly. Only observations before the training cutoff are used. Later
measured episodes are held out from both coefficient fitting and calibration.
Future runs can use checkpoints exported directly by the live server.

```bash
PYTHONPATH=/mnt/localnvme/vllm_glm53_seal_calibration_20261006 \
.venv/bin/python experiments/seal_calibration_20261006/bootstrap.py \
  /mnt/localnvme/glm53_seal_deadline_20261006/comparison_300 \
  /mnt/localnvme/glm53_seal_calibration_20261006

task_artifact_root=/mnt/localnvme/glm53_seal_calibration_20261006
CACHED_PINNED_ARTIFACT_ROOT="$task_artifact_root" \
PYTHONPATH=/mnt/localnvme/vllm_glm53_seal_calibration_20261006 \
.venv/bin/python experiments/seal_calibration_20261006/run.py \
  --trained-model "$task_artifact_root/trained_model.json" \
  --name gpu --policies legacy5,shape5 --workloads 61440:95:100 \
  --requests 75 --repeats 4 --max-model-len 65536 --distinct-fact-probe

.venv/bin/python experiments/seal_calibration_20261006/analyze.py \
  "$task_artifact_root" "$task_artifact_root/trained_model.json"
```

Use a fresh artifact root for each serving run. To create a new model instead of
reusing one, replace `--trained-model` with `--save-trained-model` and provide a
new output path. The one training episode precedes frozen measured episodes.

## CPU and offline verification

568 CPU tests passed; 22 skipped. Three older failures reproduced on the parent
branch and were excluded from the final verification: an incomplete synthetic
TimingChannel fixture, a Path/string mismatch in the notification helper, and
an unavailable old opportunity-analysis experiment. The complete failing run
and parent reproduction remain in the raw artifacts. All new calibration,
JSON round-trip, corrupt/incompatible snapshot, atomic rejection and endpoint
loopback tests passed. The existing held-A and accounting harness also passed.
Repository hooks pass.

On eight held-out preceding episodes, average cell MAE changes from 61.85 ms to
61.87 ms with global raw calibration and 50.91 ms with shape calibration. Average
cell p95 absolute error changes from 133.31 ms to 133.32 ms and 116.01 ms.
These traces motivated the live comparison; they do not establish throughput
improvement.

## Live evaluation protocol

GLM5.3 NVFP4, TP8, FP8 KV unchanged; 61,440 prompt tokens, 58,368 cached prefix
tokens, 100% CPU hits, concurrency 32, one output token. Both modes enforce the
same 5 ms predicted-completion lead. Compare four alternating paired
75-request episodes, 300 measured requests per policy. Cache resets and priming
are excluded. This is not one continuous 300-request stream.

Audits require identical frozen estimator state, no estimator training directory,
source integrity, equal KV bytes, expected CPU hits, no GPU prefix hits or
preemptions, no measured JIT/runtime errors or foreign GPU processes, and
all-rank timing and compact/pinned coverage. Accuracy is reported both for
issued forecasts and for both predictors evaluated on the same live population
of compute batches. CUDA-event execution spans include submission and connector
work, not only model kernels. Tail TTFT is the mean of episode p95 values.

## Live results

All eight episodes passed the audits. Both policies passed known-answer and
distinct-fact correctness checks (16/16 each). Exact text versus the cold run
matched 13/16 and 15/16 distinct probes respectively; factual correctness is the
gate. The initial load succeeded, and all eight exported estimator states match
the loaded checkpoint exactly. No estimator training directory was created.

| Metric | Existing calibration | Shape calibration | Change |
| --- | ---: | ---: | ---: |
| Throughput, requests/s | 7.08785 | 7.14573 | +0.82% |
| Mean TTFT, ms | 3927.92 | 3886.37 | -1.06% |
| Mean episode p95 TTFT, ms | 5645.58 | 5579.44 | -1.17% |
| Compute rounds | 52 | 52 | 0 |
| Late admissions | 274 | 271 | -3 |
| Recovered opportunities | 250/272 | 250/273 | Same recovered count |
| Gap lower bound, ms/episode | 519.87 | 466.55 | About -10% |
| Gap upper bound, ms/episode | 531.16 | 477.80 | About -10% |
| Non-load late enqueue lower bound, ms/episode | 0 | 26.14 | +26.14 ms |

Gap and late enqueue bounds are episode means. Opportunity recovery rates are
91.91% and 91.58%, a decrease of 0.34 percentage points.

Throughput is total measured requests divided by total measured duration. Paired
throughput changes are +2.45%, -0.74%, +2.22%, and -0.63%. The 95% interval for
the paired log-ratio estimate is **-1.93% to +3.63%**. These four pairs do not
establish a repeatable throughput gain. Compute rounds and recovered counts
did not improve.

The following accuracy comparison evaluates all modes on the same 104 actual
compute batches pooled from both live policies, without refitting:

| Prediction error metric, ms | Existing | Global raw | Shape |
| --- | ---: | ---: | ---: |
| Signed bias (predicted minus actual) | -3.45 | -3.41 | +20.51 |
| Mean absolute error | 68.06 | 68.08 | 50.26 |
| Error standard deviation | 82.14 | 82.15 | 63.21 |
| p95 absolute error | 155.88 | 155.86 | 134.53 |

Shape calibration reduces MAE by 26.15% and error standard deviation by 23.05%.
Global raw calibration is effectively unchanged for this checkpoint. Shape
calibration has a positive bias, so improved average accuracy does not ensure a
safe compute-end deadline.

Six non-load successor enqueues were late in shape runs, all after two-request
cached batches. Their calibrated duration is 458.58 ms versus actual execution
durations of 449.20–459.55 ms. The measured seal-to-enqueue spans are
7.01–21.82 ms, and some progress iterations also overshoot the intended expiry.
Five seals exit on predecessor completion; one fires the compute deadline.
The existing
underestimate supplied considerable early-sealing headroom on these shapes;
calibration removes it. The fixed 5 ms lead cannot consistently cover forecast
error, CPU wake latency, and membership/RPC enqueue work. These traces motivate
separate calibration for a safe execution deadline and for readiness waiting,
plus a measured enqueue reserve, before adopting this calibration by default.
They do not justify adding the old preparation reserve wholesale.

The checkpoint contains 128 fitting samples and 19 cached-regime calibration
samples. This experiment validates reuse and calibration for the tested serving
profile; it does not establish accuracy on different workloads or models.

Audited results are in `results/calibration_comparison.json` and
`results/serving_comparison.json`; the six late enqueue traces are summarized in
`results/late_enqueue_diagnostic.json`. Raw logs remain under the artifact root.
The driver and analysis exited successfully. Normal shutdown completed, with no
GPU compute processes and 0 MiB used on all eight GPUs.
