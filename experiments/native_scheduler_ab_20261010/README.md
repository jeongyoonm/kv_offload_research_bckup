# Scheduler ablation after store fixes

GLM 5.3 NVFP4 TP8, FP8 KV, concurrency 32, one output token. Four paired
75-request reset/primed episodes per arm and workload: 300 measured
requests per arm/workload, 1200 total. Warmups and priming are excluded.
This preserves the earlier episode design; it is not continuous serving.

Both arms keep 8 MiB/depth-2 async H2D, early submission, allocator
avoidance, source-before-overwrite safety and native D2H at depth two,
with a 1 GiB packet cap. CPU cache is private, 2,130,300,354,560 bytes
aggregate; GPU cache is 96 GiB per rank. NUMA and process-local THP policy
remain unchanged. Original model, weights and sparse selector are used.

C disables provisional preparation and scheduler late admission. D enables
cached/pinned preparation and the current hold/seal package, including
readiness exits, 50% load-only target and predictor deadline. Its frozen
predictor uses a 90 ms subtraction and 100 ms setup lead. There is no
online training. Changed policy fields must all belong to `late_*`.

Within each workload, repeats use identical C/D seeds, alternating arm
order. This is a package ablation: it does not isolate preparation from
sealing or the predictor. Serving acceptance checks complete CPU prefix
restoration, zero GPU prefix hits, no preemptions, one output token and
unchanged estimator state. D must show preparation on all eight ranks;
C must have zero late admissions and zero compact/pinned preparation.

The report measures throughput, mean/episode-p95 TTFT, compute rounds,
execution envelopes, exposed phase durations, readiness opportunities and
client bookends. Execution envelopes include host submission and
collective waiting; exposed phases are clipped gap overlaps. Neither is
pure useful GPU compute time. Four pairs are descriptive, not a broad
statistical significance claim or whole-model accuracy evaluation.

Branch: `experiment/native-worker-scheduler-ablation-20261010`.
Code: `/mnt/localnvme/jeongyoon/vllm_native_scheduler_ab_20261010`.
Data: `/mnt/localnvme/jeongyoon/native_scheduler_ab_20261010`.
Raw serving root: `/mnt/localnvme/jeongyoon/ns1`.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/native_scheduler_ab_20261010/run.py \
  --output /mnt/localnvme/jeongyoon/ns1

PYTHONPATH="$PWD" .venv/bin/python \
  experiments/native_scheduler_ab_20261010/report.py \
  /mnt/localnvme/jeongyoon/ns1
```

## Completed results

All 16 measured cells passed acceptance. All owned serving processes were
stopped. Throughput changes use paired geometric ratios across four pairs:

| Workload | Worker only | Worker + scheduler | Gain |
| --- | ---: | ---: | ---: |
| 16K | 25.049 req/s | 26.040 req/s | 3.92% |
| 60K | 7.608 req/s | 7.892 req/s | 3.79% |

The 60K result is sensitive to its final worker-only episode: that cell
had 1002 ms of gaps, versus 90–126 ms previously. All pairs remain in the
primary result. The first three 60K pairs gained 1.97% together. These
runs do not establish a reliable difference between workload gains.

Mean TTFT improved 2.40%/3.45% for 16K/60K. Mean episode p95 worsened
2.42% for 16K and improved 9.64% for 60K. These are package effects;
preparation, sealing and deadline components were not separately ablated.

See [the report](artifacts/notes/REPORT.txt) and
[the final audit](artifacts/notes/FINAL_AUDIT.json). Verified archives in
`artifacts/paired` and `artifacts/logbook` retain raw serving evidence,
configuration, predictor state, validation logs and failed preflight logs.
Environment credential fields are redacted in archives.

## Frozen predictor and deadline audit

Logged adjacent compute predictions show workload-dependent correction:

| Workload | Raw bias ms | Corrected bias ms | Raw MAE ms | Corrected MAE ms |
| --- | ---: | ---: | ---: | ---: |
| 16K, 34 predictions | +33.9 | -56.1 | 34.8 | 65.0 |
| 60K, 42 predictions | +96.4 | +6.4 | 96.4 | 39.2 |

The fixed 90 ms subtraction is useful for 60K but overcorrects 16K. It
does not remove variance or replace calibration across batch shapes.

| Workload | Unrecovered / compute-period KV-ready | Deadline-window ACK-ready |
| --- | ---: | ---: |
| 16K | 121 / 258 (46.9%) | 75 |
| 60K | 31 / 295 (10.5%) | 27 |

Deadline-window candidates were published, not cancelled or activated,
and already acknowledged by all eight ranks at a compute-deadline seal.
Their actual all-rank KV completion occurred after that snapshot but
before preceding compute ended. They also never received late admission
at a later seal. Counts sum unique requests within four episodes.
Another 40 short-workload candidates entered that KV window without
complete preparation acknowledgments at the snapshot. These counts are
retrospective timing opportunities, not guaranteed admissions or gains.

The audit verifies snapshot acknowledgments and reconciles its opportunity
counts with the serving report. Reproduce with `predictor_audit.py ROOT
FRESH_OUTPUT`. Per-candidate timestamps and prediction summaries are in
[the audit JSON](artifacts/notes/predictor_opportunity_audit.json).
