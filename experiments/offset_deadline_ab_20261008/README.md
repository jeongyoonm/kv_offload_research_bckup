# Latest predictor deadline comparison

Branch: `experiment/glm53-offset-deadline-ab-20261008`.
Artifacts: `/mnt/localnvme/glm53_offset_deadline_ab_20261008`.

## Completed results

All 32 cells and 2,400 measured requests passed. All 184 forecasts replayed
exactly. The fifteen CPU deadline tests passed, and all owned workers exited;
all eight GPUs were free after cleanup. Serving source is unchanged from
`fd3be1d`.

With the latest frozen predictor, deadline enforcement increased aggregate
throughput by 6.28% for 60K and 4.75% for 16K; mixed traffic was unchanged.
The paired 95% intervals include zero in every workload. Mean 60K TTFT improved
5.77%, with unchanged p95; inter-compute gaps fell 49.6%. Initial batches varied
before the first prediction, and compute rounds increased rather than decreased.

See [the table](results/REPORT.md), [interpretation](results/FINDINGS.md),
[machine-readable results](results/comparison.json), and
[completion record](results/completion.json). Raw traces and per-request results
remain in the artifact directory above. The archived process log records the
allocation, every measured cell, audit, and ownership cleanup.

## Protocol

Both arms use the frozen `matched_offset` checkpoint from the completed
offset predictor study. Only deadline enforcement versus shadow recording
changes. The 5 ms nominal lead, full prediction wait, 50 ms guard, 2 ms fine
polling, readiness and completion exits, safety cap, sixteen candidate slots,
load-only target, compact/pinned preparation, FP8 KV, and H2D configuration
remain identical. No retraining is performed.

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. Main workload: 61,440
prompt tokens, 58,368 CPU-cached tokens, 3,072 uncached tokens. Eight paired
75-request episodes per arm: 600 measured requests each. Secondary 16K and
mixed workloads each have four pairs: 300 measured requests per arm/workload.
Off/on order alternates by repeat; a pair shares prompt seed and token work.
Each episode resets and primes caches independently. Six 32-request warmups
precede measurement. All warmup/priming/startup work is excluded.

This is not a continuous 300-request stream. The current CPU cache cannot
retain 300 unique 60K prefixes simultaneously. Episode variation and initial
batch sequences are retained rather than silently excluding slow cells.

Run from this checkout, with all eight GPUs idle:

```bash
.venv/bin/python experiments/offset_deadline_ab_20261008/run.py
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/offset_deadline_ab_20261008/analyze.py
```

Use a new artifact root for another allocation. The driver has a three-hour
allocation watchdog and saves progress after every measured cell. The server
uses the authorized private CPU-cache initialization layout, unchanged NUMA
placement, and process-local THP disabling. Sparse-attention selection is
unchanged. DMA and selector diagnostics are excluded from measurements.

The analyzer checks identical checkpoint states, policy differences, expected
CPU restoration, no GPU prefix hits/preemptions, equal transfer bytes within
pairs, all-rank preparation coverage, exact per-request uncached-token
accounting, unique ordered completion, measured log errors/compilation, and
native prediction replay. It reports throughput, TTFT, GPU gaps, compute rounds,
admissions and deadline exits. Paired confidence intervals use episodes as
the sampling unit. Final cleanup must confirm all owned GPU workers have exited.
