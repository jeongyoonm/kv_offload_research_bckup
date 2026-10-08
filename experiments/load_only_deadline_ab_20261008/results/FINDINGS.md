# Findings

All 48 serving episodes were accepted: 3,600 measured requests, 384 excluded
warmup requests, and 267 exact native prediction replays. GLM 5.3 NVFP4, TP8,
concurrency 32, one output token. All four arms retain the same frozen predictor
and worker optimizations. Deadline off uses shadow deadline logging; it does not
remove prediction from the normal waiting policy.

## Observed effect of the 50% load-only policy

<!-- markdownlint-disable MD013 -->

| Workload | Deadline | Wait for all req/s | 50% ready req/s | Throughput change | Mean TTFT reduction | Paired throughput 95% interval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| familiar60k | off | 6.983 | 6.818 | -2.36% | -2.86% | [-6.63%, +2.14%] |
| familiar60k | on | 6.577 | 6.849 | +4.13% | +4.42% | [-1.07%, +9.37%] |
| short16k | off | 23.708 | 24.395 | +2.90% | +2.71% | [-8.95%, +16.07%] |
| short16k | on | 24.466 | 25.847 | +5.65% | +5.44% | [+0.06%, +11.56%] |
| heterogeneous | off | 1.720 | 1.720 | -0.01% | -0.10% | [-0.43%, +0.41%] |
| heterogeneous | on | 1.722 | 1.723 | +0.03% | +0.56% | [-0.17%, +0.22%] |

<!-- markdownlint-enable MD013 -->

Positive TTFT reduction means lower latency. Throughput change is the aggregate
request/wall-time ratio; confidence intervals are for paired geometric
throughput ratios. Four paired episodes per comparison provide limited
precision. Only the 16K deadline-on interval narrowly excludes zero; the other
policy effects are not established as repeatable gains. Intervals are nominal
and unadjusted for multiple comparisons.

## Mechanism and tradeoff

The policy demonstrably starts compute before all published candidates become
KV-ready. Across the four 60K episodes, partial cohorts contributed 24 KV
completions during recorded compute spans with the deadline off, and 21 with it
on. Corresponding 16K counts were 27 and 11. These counts describe all-rank
completion observations within instrumented execution spans; they are not an
isolated per-kernel concurrency measurement.

For 60K with the deadline enabled, mean inter-batch gaps decreased from 1,311.1
to 826.2 ms per 75-request episode; the load-only category decreased from
1,029.6 to 483.9 ms. Compute rounds increased from 41 to 44 across the four
episodes, and mean execution span increased from 9,662.9 to 9,782.5 ms. Reduced
gaps outweighed the extra execution time in this sample. Mean TTFT decreased
from 4,197.5 to 4,012.1 ms; p95 decreased from 5,512.3 to 5,110.9 ms.

In the second paired 60K deadline-on episode, throughput rose from 6.015 to
6.549 req/s (+8.87%). Load-only gaps fell from 2,925.9 to 1,935.7 ms. Two
16-candidate cohorts sealed at eight ready candidates, and one ten-candidate
cohort at five. The 21 remaining candidates completed KV readiness during the
ensuing compute spans. Compute rounds increased from eight to nine; the measured
execution span increased by 227.7 ms. Initial batch formations differed, so this
is a complete serving-trajectory comparison rather than an intervention on an
identical batch timeline.

The rule does not guarantee fewer total gaps. In 60K with the deadline off, mean
gaps increased from 438.3 to 859.2 ms even though rounds decreased from 47 to
44. The wait-all control had no inter-batch load-only gaps in these four
episodes, leaving little of that kind of stall to remove. Partial sealing
changed the subsequent batch/load trajectory; the observed throughput regression
remains within the paired uncertainty interval.

The 16K deadline-on gain is primarily associated with shorter recorded execution
spans (2,790.8 to 2,662.3 ms per episode), not a large gap reduction (93.7 to
87.7 ms). Both arms had 43 compute rounds and zero inter-batch load-only gap
time; partial load-only seals occurred before the initial compute. Batch shape,
submission stalls, and initial formation can affect execution spans. These
measurements do not isolate a single kernel-level cause for that gain.

Mixed workload throughput was essentially unchanged. The 50% rule does not
address compute-active waits and cannot improve episodes without useful partial
load-only cohorts. The implementation remains opt-in; this evaluation makes no
deployment changes.

## Scope and verification

Each arm/workload contains 300 requests as four independently reset-and-reprimed
75-request episodes, not one continuous 300-request cell. All startup, priming,
and twelve 32-request warmups are excluded. The frozen percentage denominator is
the published request-level candidate cohort, bounded by 16 slots; it is not the
count of every outstanding transfer.

All episodes passed restoration-count checks, zero GPU prefix hits/preemptions,
exact uncached scheduled token accounting, unique strictly ordered completion,
all-rank preparation coverage and execution timing, frozen state, exact
prediction replay, deadline/target arithmetic, and absence of measured
compilation/runtime errors. Every four-arm block used matching seeds and equal
aggregate transfer volume. Original sparse-attention selection and weights were
preserved; no selector oracle or DMA audit was enabled. These checks do not
constitute a model quality evaluation.

Serving code is unchanged from 673ddfa during the allocation. Earlier CPU
validation is inherited: 22 focused passes; broader suite 570 passed, 32
skipped, three failures reproduced on its unchanged parent. The first setup
preflight found a missing vendored dependency before any GPU allocation; it was
restored from the previous checkout and the failed attempt is retained in
setup_attempt_1.

Raw artifacts: `/mnt/localnvme/glm53_load_only_deadline_ab_20261008`.
