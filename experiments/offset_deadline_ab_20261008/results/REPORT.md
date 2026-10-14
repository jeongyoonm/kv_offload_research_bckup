# Latest offset predictor: deadline enabled versus disabled

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. Both arms use
the identical frozen offset predictor and optimized preparation/transfer.
The disabled arm computes and records the deadline in shadow mode.

<!-- markdownlint-disable MD013 -->

| Workload | Deadline off req/s | Deadline on req/s | Aggregate gain | Paired gain (95% interval) | Mean TTFT off/on ms | p95 TTFT off/on ms | GPU gap off/on ms per episode | Compute rounds off/on |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| familiar60k | 6.371 | 6.771 | +6.28% | +6.28% [-0.09%, +13.05%] | 4317.5 / 4068.5 | 5474.5 / 5474.9 | 1991.8 / 1004.2 | 80 / 86 |
| short16k | 22.894 | 23.981 | +4.75% | +4.49% [-6.28%, +16.50%] | 1219.7 / 1165.1 | 1578.3 / 1523.1 | 454.9 / 96.2 | 36 / 43 |
| heterogeneous | 1.722 | 1.722 | -0.01% | -0.01% [-0.45%, +0.43%] | 15536.6 / 15577.6 | 21426.9 / 21434.7 | 1317.2 / 1319.0 | 99 / 99 |

<!-- markdownlint-enable MD013 -->

Eight pairs for 60K and four pairs each for 16K and mixed workloads.
Each episode has 75 requests and separate cache reset/priming.
These are not uninterrupted 300-request cells. Six 32-request warmups
and all startup/priming are excluded. Intervals use paired log throughput
ratios and Student-t critical values; the pair is the sample unit.
Mixed workloads have few predictor decisions. Forecast MAE populations
depend on the policy and are not an architecture comparison.
GPU spans include submission stalls and output joins. Gaps exclude
startup and the final tail; they do not measure isolated kernel time.
All requests passed CPU restoration, exact token accounting, unique
ordered completion, and original output-token-count checks. No selector
oracle, model/weight modification or DMA audit was used.

Full artifacts: `/mnt/localnvme/glm53_offset_deadline_ab_20261008`.
