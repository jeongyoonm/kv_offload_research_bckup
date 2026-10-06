# Compute seal deadline experiment

Branch: `experiment/glm53-seal-deadline-20261006`.
Serving implementation: `bcc4212`.
Raw artifacts: `/mnt/localnvme/glm53_seal_deadline_20261006`.

## Policy

With `late_seal_compute_deadline=True`, arm one fixed deadline at predicted
preceding execution completion minus `late_seal_compute_deadline_lead_us`.
The effective expiry is the earlier of the prediction deadline and the existing
maximum wait. A missing prediction retains the maximum wait. Physical completion
and candidate readiness can still seal sooner. All base acknowledgments remain
required. Load-only batching retains its existing policy.

The option defaults to false and requires predictive readiness mode. Shadow mode
(`late_seal_compute_deadline_shadow=True`) records the deadline without enforcing
it. Shadow and active modes cannot be combined. Notification waits and fine
polling sleeps are clipped to the effective expiry. The deadline stays fixed
once armed, even if membership-dependent preparation estimates change.

## Validation

The CPU harness passed 174 tests with eight GPU checks skipped. The eight GPU
adoption checks then passed separately. The harness covers deadline boundaries,
partial acknowledgments, cancellations, delayed base readiness, missing
predictions, early completion, load-only policy, real notification waits and
execute enqueue while the preceding main RPC remains held.

```bash
.venv/bin/python -m pytest tests/v1/kv_offload/test_late_admission_harness.py -q
```

All five short TP8 policies passed 16/16 known-answer and 16/16 distinct-fact
checks. Exact text can differ while the factual answer matches. The 80-request
canary compared shadow control and leads of 0, 5, 10 and 20 ms on one frozen
predictor. It selected 5 ms provisionally. The canary is selection evidence,
not an independent confirmation of the selected gain.

## Paired comparison

Model: GLM5.3 NVFP4, TP8; FP8 KV unchanged. Prompt length 61,440 tokens, cached
prefix 58,368 tokens, all requests CPU hits, no GPU prefix hits, concurrency 32,
one output token. Control is the complete compact/pinned readiness prototype
with shadow logging, not vanilla vLLM.

Four paired 75-request episodes provide 300 measured requests per policy. Order
alternates between pairs. Cache resets and priming are excluded. This is not a
continuous 300-request stream: 300 unique prompts at this length exceed the
unchanged CPU cache. The predictor is trained once and frozen for all pairs.

```bash
task_artifact_root=/mnt/localnvme/glm53_seal_deadline_20261006
task_artifact_root="$task_artifact_root/comparison_300"
CACHED_PINNED_ARTIFACT_ROOT="$task_artifact_root" \
PYTHONPATH=/mnt/localnvme/vllm_glm53_seal_deadline_20261006 \
.venv/bin/python experiments/seal_deadline_20261006/run.py \
  --name gpu --policies compact_shadow,deadline5 \
  --workloads 61440:95:100 --requests 75 --repeats 4 \
  --max-model-len 65536 --distinct-fact-probe

.venv/bin/python experiments/seal_deadline_20261006/analyze.py \
  /mnt/localnvme/glm53_seal_deadline_20261006/comparison_300
.venv/bin/python experiments/seal_deadline_20261006/compare_pairs.py \
  /mnt/localnvme/glm53_seal_deadline_20261006/comparison_300
```

Throughput is total requests divided by summed measured durations. Mean TTFT
weights episodes equally because request counts match. Tail TTFT is the mean of
four cell p95 values, not a pooled 300-request p95. The paired throughput interval
uses Student t with three degrees of freedom on log ratios. Intercompute gaps
are bounded by instrumented rank timings and exclude startup and tail. Execution
spans include host submission and connector work, not only GPU model kernels.

Completed evidence and measurement summaries are retained under `results/`.

## Results

All eight measured cells were accepted: 600 requests total, equal aggregate KV
bytes per cell, all expected CPU hits, no GPU prefix hits, no preemptions, no
measured JIT or runtime errors, no foreign GPU processes, and all-rank execution
and compact/pinned coverage. Both policies pass all 16 known-answer and 16
distinct-fact checks; 15/16 distinct responses match the cold text exactly for
each policy while all factual answers match. Serving source hashes match the
recorded implementation.

| Metric | Control | 5 ms lead | Change |
| --- | ---: | ---: | ---: |
| Aggregate req/s | 7.107 | 7.193 | +1.21% |
| Mean TTFT ms | 3907 | 3858 | -1.26% |
| Mean cell p95 TTFT ms | 5404 | 5443 | +0.71% |
| Compute rounds, total | 50 | 52 | +2 |
| Late admissions, total | 280 | 276 | -4 |
| Recovered compute opportunities | 254/271 | 261/279 | 93.73% / 93.55% |
| Intercompute gap bounds ms/episode | 519–530 | 383–395 | about -26% |
| Non-load late enqueue lower bound ms/episode | 20.09 | 0.00 | -20.09 |

The paired geometric throughput gain is 1.21%, with a 95% interval of -0.69% to
+3.15%. Individual pair gains are -0.09%, +1.43%, +2.78% and +0.75%. The interval
includes no improvement: the canary's +3.93% selected result did not establish
that gain in the independent repeats. Keep the feature optional and disabled by
default; this workload shows a small throughput point estimate and reduced gaps,
not a reliable throughput win.

The deadline removes observed late enqueue exposure, but batch membership and
subsequent load submission change too. Total instrumented execution spans rise
from 39.307 s to 39.479 s across the four episodes. A larger gap reduction than
the direct enqueue saving cannot all be attributed to the timer. Deadline runs
use two more compute rounds and leave eight prepared candidates ready at worker
entry but unselected, versus three for control. Admissions are not fully
preserved, and a batching cost can offset the timing benefit.

There are eight hard-deadline firings, two per episode. Seal overshoot ranges
from 0.151 to 3.616 ms. In the second episode, requested 2 ms fine waits actually
return after about 7 ms; their delayed wake accounts for the 3.0–3.6 ms expiry
overshoot. Requested waits are clipped correctly, but CPU scheduling is not a
real-time guarantee. The deadline should be treated as a target, not an exact
physical GPU-start guarantee.

The maximum wait remains necessary when prediction is unavailable or wrong.
It shares one effective expiry with the compute deadline; it does not add a
second intentional hold. Readiness and completion still exit sooner, while
load-only batching is unchanged.

The eight hard-deadline seals occur 16.8–164.6 ms before the measured all-rank
preceding execution end; execute enqueue occurs 13.5–158.8 ms before it. Thus
"5 ms lead" describes the prediction, not the actual lead. Prediction error is
large enough to matter to batching, and refining the nominal lead alone will
not control the physical sealing margin. Individual firings and wake durations
are included in the serving audit and `results/deadline_firings.json`.

The driver completed normally. All worker processes exited, no GPU applications
remain, and all eight GPUs returned to zero used memory.
