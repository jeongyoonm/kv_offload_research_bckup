# Coalesced notification A/B evaluation

Branch: `experiment/glm53-coalesced-seal-notify-20261004`.
Artifacts: `/mnt/localnvme/glm53_coalesced_notify_20261004`.

## CPU replay

Six alternating paired replays per rank-skew pattern; eight persistent worker
processes, ten candidates, production preparation/seal loop. The preceding
RPC stays held. Candidate completion spacing is 2.5 ms. Both modes use the
same native publisher; only coalesced mode arms its notification masks.
CPU time excludes process startup and includes all worker publication work.

<!-- markdownlint-disable MD013 -->

| Rank skew | Mode | Engine CPU ms | Worker CPU ms | Total CPU ms | Mean ready to seal ms | p95 ready to seal ms | Progress wakes | Bridge scans |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | bridge | 2.470 | 2.176 | 4.646 | 0.618 | 0.806 | 10.000 | 36.667 |
| 0 ms | coalesced | 1.859 | 4.998 | 6.857 | 0.125 | 0.176 | 10.000 | 0.000 |
| 1 ms | bridge | 4.043 | 1.558 | 5.601 | 0.832 | 1.054 | 32.167 | 64.167 |
| 1 ms | coalesced | 1.970 | 4.684 | 6.655 | 0.092 | 0.094 | 10.000 | 0.000 |
| 2 ms | bridge | 4.073 | 1.483 | 5.556 | 0.192 | 0.224 | 36.000 | 74.000 |
| 2 ms | coalesced | 1.905 | 4.370 | 6.275 | 0.090 | 0.091 | 10.000 | 0.000 |

All 36 windows admitted all ten candidates. Coalescing removed bridge scans
and reduced latency, but total engine plus worker CPU increased by 12.9–47.6%.
The replay therefore does **not** demonstrate net CPU savings. Each synthetic
pattern has only six repetitions; tail values are descriptive.

## TP8 canary protocol

GLM 5.3 NVFP4, TP8, FP8 E4M3 KV, concurrency 32, 64 requests per cell.
Prompt lengths 8,192 and 61,440; 95% CPU-cached prefix, 100% CPU hit rate.
Three repeats use AB, BA, AB order. Both modes use the full predicted wait
minus 50 ms and 2 ms fine polling, ten candidate slots, and identical native
libraries. A shared predictor trains on both workloads, then freezes before
the measured cells and their cache priming. Training and correctness probes
are excluded from results.

CPU counters cover the API server and every descendant, including all eight
workers and their threads. Sampling interval is 25 ms; both the full profile
envelope and the replay trace interval are retained. Replay CPU includes a
25 ms response-cleanup margin. These are sampled process CPU estimates, not
instruction-level attribution.

Readiness latency uses native all-rank KV completion timestamps and engine
observation of the final preparation acknowledgment. It can underestimate
publication-to-seal latency when an acknowledgment is observed late. Fine
polling and existing GPU completion observation remain active.

The old bridge uses the same new binaries with the notification transport
inactive. This compares active policies, not the previous binary against a
new binary. Three short repeats cannot establish statistical significance.

## Development record

Three startups were discarded before any measured cells: missing FlashMLA
interface, then an incomplete DeepGEMM namespace cached before its vendor
files were restored, then a restart attempted before GPU memory cleanup had
finished. Verified zero allocated memory on all GPUs before the final launch.
Restored ignored dependencies from the previous working
checkout and verified DeepGEMM GEMM and attention exports in a fresh process
before restarting. Sources, hashes, stop records, and failed logs are retained
in the artifact directory.

The new default-false predictor-freeze option passed all eight predictor tests.
All staged implementation hooks passed. The notification mechanism remains
experimental and disabled by default.

## TP8 canary results

All 12 measured cells passed (768 measured requests). Every cell reported the
expected CPU-prefix tokens, zero GPU-prefix hits, zero preemptions, no measured
JIT, no runtime errors, and no unexpected GPU processes. All eight workers are
covered by the CPU samples, with no sampling errors. Training cells are excluded.

Both policies passed all 16 known-answer restore checks and all 16 distinct-fact
checks. Exact distinct-answer wording matched cold references in 14/16 bridge
and 15/16 coalesced cases; all facts and cache-token counts were correct.

Means across three short cells per policy; percentile columns are means of
each cell's request percentiles. CPU/request uses the bounded replay interval.

| Prompt | Mode | req/s | Mean TTFT ms | p95 TTFT ms | Total CPU ms/req | Engine CPU ms/req | Worker CPU ms/req | Compute rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60K | bridge | 6.98 | 3830.15 | 5724.89 | 1509.09 | 18.51 | 1428.90 | 11.00 |
| 60K | coalesced | 6.91 | 3884.37 | 5821.84 | 1545.49 | 17.84 | 1465.48 | 11.33 |
| 8K | bridge | 34.55 | 767.81 | 1012.03 | 314.76 | 3.78 | 301.68 | 9.00 |
| 8K | coalesced | 33.41 | 830.83 | 1079.17 | 322.59 | 3.73 | 309.90 | 9.67 |

| Prompt | Throughput change | Mean TTFT change | p95 TTFT change | Total CPU/request change | Engine CPU/request change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60K | -0.99% | +1.42% | +1.69% | +2.41% | -3.61% |
| 8K | -3.30% | +8.21% | +6.63% | +2.49% | -1.37% |

A positive CPU or TTFT change means higher cost. Throughput differences reverse
direction across repeats: paired changes are retained below. These observations
do not establish a statistically significant production regression or benefit.

| Prompt | Repeat | Throughput change | Total CPU/request change |
| --- | ---: | ---: | ---: |
| 60K | 0 | +0.84% | +3.85% |
| 60K | 1 | -3.49% | +2.61% |
| 60K | 2 | -0.24% | +0.76% |
| 8K | 0 | -10.20% | +9.26% |
| 8K | 1 | +3.11% | -2.70% |
| 8K | 2 | -2.75% | +1.36% |

## Progress and latency

Progress counts are waits returning progress, including an already pending
notification; they are not kernel scheduler wakeup counts. Fine waits retain
the existing 2 ms interval. Readiness latency pools target-reaching windows
across the three cells. It uses the observed preparation ACK caveat above.

| Prompt | Mode | Coarse progress resumes/cell | Coarse timer returns/cell | Bridge scans/cell | Fine waits/cell | Target latency median ms | Target latency p95 ms | Target latency max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60K | bridge | 263.000 | 2.333 | 2741.333 | 352.667 | 1.085 | 4.704 | 44.548 |
| 60K | coalesced | 41.000 | 3.333 | 0.000 | 449.667 | 0.452 | 1.959 | 2.053 |
| 8K | bridge | 78.000 | 1.333 | 228.667 | 125.000 | 1.472 | 2.108 | 2.313 |
| 8K | coalesced | 24.667 | 3.667 | 0.000 | 88.333 | 1.061 | 2.390 | 3.401 |

Coarse progress resumes fell 68.4% for 8K and 84.4% for 60K; bridge scans fell
to zero. Both traces show functioning producer notifications. One 60K bridge
window has a 44.55 ms target-response outlier, with 43.63 ms after its last wait
returned; its mean and p99 should not be read as typical scanner delay. Phase
breakdowns and all window samples are retained in the aggregate/cell JSON.

| Prompt | Mode | Late admissions/cell | Recovered compute opportunities | Opportunity admission ratio |
| --- | --- | ---: | --- | ---: |
| 60K | bridge | 58.67 | 156/169 | 92.31% |
| 60K | coalesced | 56.67 | 149/166 | 89.76% |
| 8K | bridge | 45.00 | 91/144 | 63.19% |
| 8K | coalesced | 44.67 | 115/169 | 68.05% |

Opportunities are unique requests whose all-rank KV completion falls inside an
observed preceding compute interval. The denominator changes with execution
timing; an increased ratio alone does not demonstrate better batching. Here,
coalescing did not reduce compute rounds on average.

## Decision

The mechanism eliminates coarse bridge scanning and improves typical coarse
readiness response. This evaluation does **not** show net CPU savings or a
consistent throughput/TTFT gain. Keep it experimental and disabled by default.

The CPU replay isolates additional worker bookkeeping. Preparation currently
opens, maps, and closes the notification channel on every ACK. Reusing that
mapping within the existing preparation worker is a concrete next optimization
to evaluate separately. Whole-model CPU differences also include changes in
compute rounds and cannot be attributed entirely to notification bookkeeping.

Measurements used implementation commit `7b3140e`. A launch preflight and a
guard against silently reusing saved training/cells were added after measurement
completed. These do not affect the measured serving implementation.

Summary, paired comparisons, CPU replay rows, source/vendor provenance, and
compressed measured traces are retained beside this report and in the artifact
directory. No 300-request benchmark was run.

All server and worker processes exited. GPU inventory is empty and host memory
was released. The post-run dependency/GPU-memory preflight passed. All staged
pre-commit checks passed. Cleanup-time resource-tracker warnings were outside
the measurement intervals; their full server log is retained.
