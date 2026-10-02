# GLM 5.3 TP8 polling and prediction experiment

All code, weights, caches and artifacts are under `/mnt/localnvme`.
Artifacts: `/mnt/localnvme/glm53_poll_prediction_20261003`.

The screen compares fourteen policies on three workloads at concurrency 32,
one output token, ten candidate slots and FP8 KV. The workload tuples are:

| Prompt tokens | Cached fraction in hit requests | Hit requests | Replay requests |
|---:|---:|---:|---:|
| 8192 | 95% | 100% | 300 |
| 61440 | 95% | 100% | 80 |
| 61440 | 99% | 100% | 80 |

The long cohorts fit the existing 1984 GiB CPU cache. Each policy has an
independent CPU/GPU reset, concurrency-eight priming and GPU-only reset before
the replay. Exact individual cache positions, transfer bytes, output counts,
compilation and GPU process ownership are validated by the existing harness.

## Policies

- Worker-only: existing chunked asynchronous H2D, two-chunk pipeline and early
  submission, with late admission disabled. This is not unoptimized vLLM.
- Fixed: 200 ms plus conditional 100 ms extension, with existing readiness and
  load-only early exits.
- Readiness: all published noncancelled candidates during compute, four when
  load-only; 1, 2, 5 or 10 ms polling after all eight base acknowledgments.
- Prediction: existing shape-conditioned preparation reserve and compute
  estimate; the same four post-ack polling intervals.
- Recent reserve: same compute predictor, but preparation samples must share the
  cold/cached/mixed regime and be nearby in batch shape. Use the nearest eight
  of the last thirty-two eligible samples, distance <= 0.75 in log1p shape space,
  with their 75th percentile plus the existing 10 ms margin. No match falls back
  to the 10 ms margin. Compare 1 and 5 ms polling. This fallback is deliberately
  measured and is not assumed to avoid exposed preparation.
- Hybrid: recent reserve with 20 or 50 ms maximum initial sleep intervals, then
  2 ms polling in the last 50 ms before the predicted seal deadline. Every
  iteration checks the ordinary early exits before sleeping. Coarse sleeps are
  bounded by the phase transition and safety deadline; this is not a single
  blind sleep until predicted completion.

Pre-acknowledgment polling stays 25 us for every policy. The prediction deadline,
physical-completion exit, readiness exits and two-second safety cap remain
unchanged. The hybrid is opt-in and default-disabled. Fine sleeps retain the
original behavior; a condition during sleep is detected on the next iteration.
Readiness polling applies during compute and load-only waits. Prediction polling
applies during active compute; its existing fixed/partial-ready load-only path
continues to use the unchanged fixed-policy interval.

## Measurements and limits

Retain throughput, mean and p95 TTFT, compute rounds, candidate admissions,
compute-window opportunity recovery, exposed-hold bounds, GPU execution and
between-compute gaps. Additional transport fields retain polling iterations,
adaptive sleep counts, requested/actual sleep totals, coarse-sleep counts and
thread CPU time spent in iterations that lead to adaptive sleeps. That CPU
counter excludes final, pre-ack and fixed-policy iterations; it is not total
engine CPU usage. Physical completion timestamps distinguish polling delay from
preparation and FIFO waiting. Prediction samples include predicted preparation
reserve and actual successor preparation wall time.

The initial screen has one replay per policy/workload. Any selected follow-up
must retain matched controls, a fresh seed, reversed orders and all screen
regressions. Do not interpret a single replay or admission improvement as a
proven throughput benefit. Predictor observations remain past-only but persist
between policies, as in the previous study; priming is independently repeated.

CPU verification: 67 protocol/predictor/endpoint tests and 19 selected option
tests pass. An exploratory broader suite had 406 passes and two inherited
fixture/artifact failures; these are retained in the process log. Native assets
match the previous evaluation. Missing packaged third-party interfaces were
restored in fresh worktrees after an initial startup failure, before any replay.
