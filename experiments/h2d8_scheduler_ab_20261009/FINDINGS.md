# Improved-worker scheduler ablation

## Result

With 8 MiB H2D chunks and depth 2, the scheduler package still helps the
16K cached workload. It offers no aggregate throughput improvement on
60K cached or mixed prompts in this experiment. It is therefore useful
for some workloads, but is not a universally beneficial policy after
the worker improvement.

C disables provisional preparation and late admission. D enables the
latest preparation, hold/seal, readiness-exit, load-only target, and
predictor-deadline package. Both retain the same asynchronous H2D worker,
early KV submission, cooperative progress, and transfer geometry.
This is a package ablation; it does not isolate the predictor's effect.

## Serving measurements

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. Four matched
75-request episodes per arm and workload, with alternating C/D order.
This is 300 measured requests per arm/workload, split into four reset
and primed episodes; it is not a continuous 300-request benchmark.
All 24 measured episodes and six excluded warmups passed validation.

Absolute levels are arithmetic episode means. Percentage changes use
the geometric mean of paired D/C ratios. Tail values average episode
p95 values, rather than pooling requests across episodes.

<!-- markdownlint-disable MD013 -->

| Workload | Arm | Throughput req/s | Mean TTFT ms | p95 TTFT ms |
| --- | --- | ---: | ---: | ---: |
| 16K cached | C | 24.654 | 1111.4 | 1411.0 |
| 16K cached | D | 27.265 | 989.6 | 1400.9 |
| 60K cached | C | 7.721 | 3572.2 | 5060.5 |
| 60K cached | D | 7.582 | 3612.5 | 4810.1 |
| Mixed | C | 1.723 | 15541.8 | 21398.5 |
| Mixed | D | 1.724 | 15580.2 | 21409.9 |

| Workload | Throughput change | Mean TTFT change | p95 TTFT change |
| --- | ---: | ---: | ---: |
| 16K cached | +10.73% | -10.82% | -0.75% |
| 60K cached | -1.81% | +1.12% | -5.18% |
| Mixed | +0.066% | +0.247% | +0.054% |

<!-- markdownlint-enable MD013 -->

Four pairs are descriptive evidence, not a broad statistical equivalence
or significance claim. The 16K throughput changes were +6.12%, +0.66%,
+18.96%, and +18.32%. The 60K changes were -0.19%, -4.35%, -2.93%, and
+0.30%. Mixed changes ranged from -0.004% to +0.103%.

## Batching and timing evidence

All entries below are means per 75-request episode. C admits no requests
through the late path. D retains effective late admission on both cached
workloads despite the faster worker.

<!-- markdownlint-disable MD013 -->

| Workload | C compute rounds | D compute rounds | D late admissions | C inter-compute gaps ms | D inter-compute gaps ms | D hold without compute ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K cached | 14.0 | 12.5 | 73.00 | 153.4 | 137.1 | 64.1 |
| 60K cached | 15.0 | 12.0 | 71.75 | 97.1 | 384.5 | 60.1 |
| Mixed | 24.75 | 24.50 | 1.00 | 1311.2 | 1285.3 | 39.1 |

<!-- markdownlint-enable MD013 -->

On 16K prompts, D reduced the summed execution envelopes from 2800.5 to
2502.8 ms, alongside fewer compute rounds. On 60K prompts, the reduction
was smaller: 9273.8 to 9118.1 ms. Its approximately 155.7 ms saving was
offset by approximately 287.4 ms more inter-compute gap time. This is
consistent with the measured throughput loss, but does not identify the
specific synchronization or submission mechanism responsible.

Execution envelopes come from all-rank CUDA event timing. They can include
submission stalls and collective waiting; they are not pure useful kernel
time. Inter-compute gaps exclude the initial and final idle periods.
Hold without compute is the union of base-ready-to-seal intervals outside
measured compute envelopes, including load-only waiting. It is not all
GPU idle time or an independently isolated causal cost. The 60K gap
increase is larger than this measured exposed hold, so attributing the
entire regression to hold duration would be unsupported.

Using the harness's request-level overlap metric, D recovered 270/270
observed opportunities for 16K and 281/281 for 60K. On mixed prompts
it recovered 4/143. This intersects requests admitted by any measured
seal with requests whose all-rank KV completion falls inside any measured
compute interval; it does not prove optimal admission at each seal.
Candidate eligibility restricts complete residual
prefills to at most 4096 tokens; most mixed requests have an 8192-token
suffix. The denominator covers all observed ready requests during
measured compute, including those outside that eligibility restriction.
The denominator can differ between C and D because their compute
intervals and batching differ.

## Integrity and implementation boundary

- Same model, KV precision, CPU/GPU cache budgets, NUMA placement,
  asynchronous H2D, chunk size, depth, and early submission in both arms.
- Policy differences are restricted to `late_*` fields and recorded in
  the plan. C has zero late admissions and zero compact/pinned preparation
  usage. D has prepared-input coverage on all eight ranks in every cell.
- Expected CPU-cache restoration counts exactly match observed counts.
  No extra GPU prefix hits, preemptions, or output-token count violations.
- Both arms load the same frozen predictor. Only the previously tested
  checkpoint chunk-size identity is retargeted from 128 to 8 MiB; weights,
  histories, and other identity fields are unchanged. No online training.
- Nine CPU regression checks and both option-validator preflights passed.
  No production defaults, native binaries, model weights, or selectors
  changed. Changes are confined to experiment drivers and reporting.

The first allocation stopped after an extra harness assertion incorrectly
expected no worker-entry records in C. Ordinary execution is also traced.
The corrected assertion requires zero compact/pinned prepared inputs and
zero late admissions. That allocation collected no measured episodes;
its logs and warmup are preserved under `paired`, while accepted evidence
is under `paired_v2`. The reporting preflight also corrected reading seal
fields from the nested transport record. Neither correction changes model
execution or measured timings.

The owned server and workers exited. All eight GPUs returned to zero
allocated memory. Shutdown emitted inherited resource-tracker warnings;
the subsequent process and GPU cleanup checks passed.

Original artifacts are under
`/mnt/localnvme/h2d8_scheduler_ab_20261009`.
Implementation branch:
`experiment/h2d8-scheduler-ablation-20261009`.
The CSV and JSON include full-precision levels and individual paired
changes. The raw archive retains measured requests, engine and seal
traces, policy snapshots, and the excluded warmups.
