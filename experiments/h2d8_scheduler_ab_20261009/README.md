# Scheduler ablation with improved H2D submission

This isolated experiment compares C and D on GLM 5.3 NVFP4 TP8. Both use
8 MiB H2D chunks, pipeline depth 2, asynchronous H2D submission, private
CPU-cache initialization, early KV submission, and cooperative response
progress. No model weights, selector, or production defaults change.

- C disables late admission and its provisional preparation lane. Requests
  become eligible through ordinary scheduling after KV readiness.
- D uses optimized provisional preparation and the latest scheduler policy:
  16 candidate slots, predictive compute deadline with 5 ms lead, readiness
  exits, a 50% load-only cohort target, efficient waiting, and a 2 s safety
  timeout.

Only `late_*` options differ. This measures the scheduler mechanism as a
package, including its preparation support. It does not isolate the
contribution of the predictor from other scheduler components.

Both arms load exactly the same frozen predictor state. Only its checkpoint
chunk-size identity is retargeted from 128 to 8 MiB, using the previously
verified helper; learned parameters do not change. C does not use it for
sealing decisions. Every cell checks that the state remains frozen.

Four paired repeats use alternating C/D order and identical prompt seeds.
Each measured episode serves 75 requests at concurrency 32 with one output
token. Workloads are 16K cached, 60K cached, and heterogeneous prompts with
50% request-level CPU-cache hits. Each episode resets and primes caches;
six 32-request warmups are excluded. These are separate episodes, not
continuous 300-request cells.

Run from this checkout with the existing serving virtual environment:

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.h2d8_scheduler_ab_20261009.run \
  --output /mnt/localnvme/h2d8_scheduler_ab_20261009/paired
```

The output path must be fresh. The driver asserts idle GPUs, owns its server
process group, enforces a three-hour watchdog, and stops the server on exit.
Configuration validation can run without GPU allocation using
`--preflight-only` and a fresh output directory.

Accepted episodes require exact expected CPU-cache restoration counts,
unchanged predictor state, one output token per request, no preemptions,
and no unexpected GPU prefix-cache hits. D additionally requires compact
and pinned preparation coverage on every rank. C requires no late
admissions and no preparation entries.

After completion:

```bash
.venv/bin/python -m experiments.h2d8_scheduler_ab_20261009.report \
  /mnt/localnvme/h2d8_scheduler_ab_20261009/paired
```

The report exports absolute metrics and paired geometric percentage changes.
Tail TTFT is the mean of episode p95 values, not a pooled percentile.
Compute-gap estimates use all-rank execution event envelopes and exclude
initial and final idle time. Hold time outside these envelopes includes
load-only waiting and is not an independently attributable throughput cost.
