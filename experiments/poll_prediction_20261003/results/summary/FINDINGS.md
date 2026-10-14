# GLM 5.3 TP8: polling and prediction experiments

Completed 78 accepted replay cells. Concurrency 32, one output token, ten
candidate slots, FP8 KV, existing worker optimizations enabled. 8K cells have
300 requests; 60K cells have 80. All replay requests have a CPU prefix hit;
95%/99% denotes the cached fraction of each prompt, rounded to 64-token blocks.
Worker-only retains async H2D, pipelining and early submission with late
admission disabled; it is not unoptimized vLLM.

## Polling sweep: single screening seed

Throughput in requests/s. Each point is one replay. This is not a confidence
interval. Pre-ACK polling remains 25 us; only adaptive post-ACK polling changes.

| Policy | 8K / 95% | 60K / 95% | 60K / 99% |
| --- | --- | --- | --- |
| worker | 32.304 | 6.470 | 11.376 |
| fixed | 41.880 | 7.097 | 11.413 |
| readiness_1ms | 39.731 | 7.063 | 11.431 |
| readiness_2ms | 39.230 | 7.066 | 11.515 |
| readiness_5ms | 40.143 | 6.989 | 11.384 |
| readiness_10ms | 42.354 | 6.989 | 11.348 |
| prediction_1ms | 41.720 | 7.099 | 11.605 |
| prediction_2ms | 41.514 | 6.991 | 11.487 |
| prediction_5ms | 42.671 | 7.066 | 11.454 |
| prediction_10ms | 41.353 | 6.987 | 11.350 |
| recent_1ms | 40.746 | 7.021 | 11.187 |
| recent_5ms | 40.596 | 6.941 | 11.355 |
| hybrid_20ms | 40.527 | 7.194 | 11.361 |
| hybrid_50ms | 40.896 | 7.016 | 11.410 |

## Matched screening and fresh-seed confirmation

Cells show arithmetic mean requests/s / mean TTFT in ms, across seed 55 and seed
56. Confirmation reverses selected policy order. Two seeds do not establish a
statistically reliable optimum.

| Policy | 8K / 95% | 60K / 95% | 60K / 99% |
| --- | --- | --- | --- |
| fixed | 42.066 / 729 | 7.062 / 3999 | 11.486 / 2537 |
| readiness_1ms | 40.821 / 758 | 6.989 / 3982 | 11.343 / 2565 |
| readiness_10ms | 42.418 / 728 | 6.972 / 3994 | 11.384 / 2571 |
| prediction_1ms | 41.856 / 738 | 7.040 / 3992 | 11.451 / 2562 |
| prediction_5ms | 41.731 / 736 | 7.035 / 4063 | 11.460 / 2534 |
| recent_1ms | 40.278 / 760 | 7.052 / 3982 | 11.322 / 2567 |
| hybrid_20ms | 40.532 / 758 | 7.136 / 3927 | 11.304 / 2564 |
| hybrid_50ms | 40.830 / 755 | 7.002 / 4059 | 11.395 / 2569 |

## Predicted initial wait followed by readiness polling

Separate server; matched seed 56. Up to 50 ms coarse sleeps, followed by 2 ms
polling after the first predicted deadline minus 50 ms. The forecast controls
checking frequency and cannot seal this variant. Seal on all active published
candidates ready+acknowledged, preceding physical compute completion, or 2 s
safety cap; load-only target=min(4, active candidates). Every wake refreshes
progress; conditions during a sleep wait for the next wake. No forecast falls
back to fine polling.

| Policy | 8K / 95% | 60K / 95% | 60K / 99% |
| --- | --- | --- | --- |
| fixed | 40.797 / 747 | 6.939 / 4054 | 11.551 / 2525 |
| prediction_1ms | 37.663 / 810 | 6.997 / 4041 | 11.636 / 2543 |
| readiness_2ms | 39.940 / 772 | 7.115 / 3892 | 11.303 / 2554 |
| predicted_readiness_50ms | 39.590 / 772 | 7.118 / 3900 | 11.629 / 2499 |

### Hybrid diagnostics against readiness at 2 ms

Throughput and mean TTFT changes use the control from the same new server and
workload. Positive TTFT reduction means improvement. Paired columns show
readiness/hybrid. Polling CPU is the measured adaptive-iteration counter;
exposed hold is the lower timestamp bound summed over the replay.

| Workload | Δreq/s | ΔTTFT | Rounds | CPU ms | Recov% | Expose ms |
| --- | --- | --- | --- | --- | --- | --- |
| 8K / 95% | -0.88% | -0.09% | 38/40 | 78.1/65.2 | 93.4/94.9 | 96.3/17.8 |
| 60K / 95% | +0.04% | -0.22% | 13/13 | 108.3/52.6 | 91.9/95.9 | 337.4/324.8 |
| 60K / 99% | +2.88% | +2.14% | 13/13 | 123.7/84.6 | 98.6/100.0 | 326.6/333.6 |

## Interpretation limits and verification

- Confirmation readiness_1ms on 8K has one missing rank6 execution timing
  snapshot (step7958). All eight primary completion records and all responses
  are valid. GPU-interval diagnostics omit that interval; nominal token work,
  compute counts, throughput and TTFT retain it. No timing values are imputed.
  All other replay batches have complete execution timing.
- Each replay resets and independently primes the CPU cache, then resets GPU
  prefix state. Raw cache positions, exact transfer bytes, uncached token work,
  output counts, compilation/errors and GPU process ownership are validated. No
  replay may be silently dropped.
- Known-answer and distinct-fact probes are stored in each phase. The new branch
  passes 70 CPU protocol/predictor tests and 12 selected option tests. Earlier
  polling/hybrid gates passed 67 CPU checks and 19 option checks. Two broader
  inherited fixture/artifact failures are retained and reproduced on the
  unchanged old evaluation source.
- Prediction uses past observations only, but training history persists across
  policies and their priming passes. Reverse order helps assess sensitivity; it
  does not give each predictor an identical training state.
- The recent reserve filters context regime and nearby batch shape, then takes
  the nearest 8 of the last 32 eligible samples, 75th percentile plus 10 ms.
  Context regime is not a definitive CPU-cache-origin flag. No nearby sample
  falls back to 10 ms and can underestimate preparation.
- Polling CPU covers thread time in iterations leading to adaptive sleep; it
  excludes final/pre-ACK/fixed iterations and is not total engine CPU. Exposed
  hold is an all-rank timestamp bound, not an additive counterfactual throughput
  estimate. Admission denominators vary with resulting batch timing.
- These are prefill/TTFT experiments with one output token; they do not
  establish performance for long generation or other concurrencies.
- All new policies remain opt-in and disabled by default.

## Artifacts

- [Full
  tables](/mnt/localnvme/glm53_poll_prediction_20261003/summary/report.html)
- [Polling
  curves](/mnt/localnvme/glm53_poll_prediction_20261003/summary/polling_screen.png)
- [Prediction versus
  readiness](/mnt/localnvme/glm53_poll_prediction_20261003/summary/predicted_readiness_comparison.png)
- Process log:
  [/mnt/localnvme/glm53_poll_prediction_20261003/process_log.jsonl](/mnt/localnvme/glm53_poll_prediction_20261003/process_log.jsonl)
- [Branches and native SHA
  manifests](/mnt/localnvme/glm53_poll_prediction_20261003/branches.json)
- [Forecast
  diagnostics](/mnt/localnvme/glm53_poll_prediction_20261003/summary/prediction_diagnostics.csv)
