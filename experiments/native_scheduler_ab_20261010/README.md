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
