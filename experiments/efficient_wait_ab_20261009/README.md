# Efficient waiting ablation

Branch: `experiment/glm53-efficient-wait-ablation-20261009`.
Source: `/mnt/localnvme/vllm_glm53_efficient_wait_ab_20261009`.
Artifacts: `/mnt/localnvme/glm53_efficient_wait_ab_20261009`.

The latest prototype's authoritative sealing rules stay fixed: frozen offset
predictor, 5 ms compute deadline lead, candidate readiness/completion early exits,
50% load-only published-cohort target, 16 candidate slots, 2 s safety timeout,
and optimized worker preparation and asynchronous H2D. Serving code is unchanged
from `673ddfa`; parent experiment commit is `e2ec78a`.

| Arm | Initial wait | Early wake mechanism | Fine polling |
| --- | --- | --- | --- |
| full | Prediction minus 50 ms | Producer coalescing | 2 ms |
| bridge | Prediction minus 50 ms | Transport observer polling | 2 ms |
| poll | None | Scheduler polling | 2 ms |

The bridge scans progress every 1 ms while active. It wakes on partial transport
changes; full coalesces producer updates into all-rank readiness notifications.
Load-only periods use the same 2 ms polling in all arms. Deadline prediction
remains enabled in poll: this experiment removes efficient waiting, not the
predictor or delayed sealing. Ordinary model selection and weights are unchanged.

GLM 5.3 NVFP4 TP8, concurrency 32, one output token. Workloads match the previous
study: 60K cached, 16K cached, and heterogeneous cached/cold contexts and suffixes.
Six fresh-seed blocks use every permutation of the three arms, balancing position
and within-block immediate carryover. Each arm/workload has six independently
reset/primed 75-request episodes: 450 requests, not one continuous 450-request run.
There are 54 measured episodes (4,050 requests), plus nine excluded 32-request
warmups. Startup and cache priming are excluded. No optional short TP8 canary.

Measure throughput, mean/p95 TTFT, batching, admissions, physical compute-span
gaps, scheduler progress iterations, coarse/fine wait returns, bridge scans,
thread CPU spent waiting/progressing, deadline overshoot, and wait overlap after
physical predecessor completion. Process CPU and persistent-thread context
switches are sampled only at episode boundaries. Those context switches are a
lower bound excluding transient threads, not scheduler wakeup counts. Wait CPU
counters do not cover every final-loop instruction; EngineCore CPU covers the
whole serving measurement boundary interval.

All arms enable the same existing detailed wait tracing. GPU spans include host
submission stalls/output joins; exposed sealing hold is not total GPU bubble
time. Forecast-conditioned exposure excludes windows without a valid predecessor
forecast. Accuracy populations can differ as trajectories change.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/efficient_wait_ab_20261009/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/efficient_wait_ab_20261009/analyze.py
```

Audit ordered unique retirement, exact uncached token accounting, CPU restoration,
zero GPU prefix hits/preemptions, all-rank preparation/execution timing, frozen
native prediction replay, target/deadline arithmetic, matched transfer volume,
and absence of measured compilation/errors. Paired throughput intervals use
episode log ratios with six blocks; intervals are nominal, unadjusted for multiple
comparisons. AI assistance: Codex.
