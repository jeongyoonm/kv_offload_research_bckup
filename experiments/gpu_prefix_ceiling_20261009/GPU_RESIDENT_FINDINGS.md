# Audited resident GPU reference

The allocation completed and stopped. All 620 measured requests hit their
expected full prefixes on GPU, with no CPU offloading loads or preemption.
THP is disabled and audited on all eight workers. Warm priming is excluded.
The original GLM model, sparse selector, FP8 KV, TP8 and 96 GiB per-rank
GPU cache remain unchanged.

| Cohort | Requests | Req/s | Mean TTFT ms | Pooled p95 ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 320 | 7.720 | 3066.1 | 4120.7 |
| short16k | 300 | 25.770 | 1090.9 | 1255.8 |

The long cohort uses ten independently primed 32-request waves; the short
cohort uses four 75-request episodes. Concurrency is 32. CPU controls use
the same cohorts and seeds but reset GPU cache after priming. GPU-only
removes the connector and its dependent scheduling/preparation. It is a
measured reference, not a mathematical upper bound on optimized CPU serving.

The 75-prefix 60K working set does not fit the fixed GPU cache budget. The
separate earlier capacity-limited run is not an all-hit ceiling. The first
GPU reference lacked explicit THP auditing; use this corrected reference.

The raw archive and member hashes are verified in
`artifacts/gpu_thp_resident`. Serving acceptance is not whole-model accuracy.
