# Load-only half-ready policy with and without predictor deadline

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. All four arms use the same
frozen predictor and transfer/preparation path.

<!-- markdownlint-disable MD013 -->

| Workload | Policy | req/s | Mean TTFT ms | p95 TTFT ms | Compute rounds | GPU gap ms/episode | Load-only gap ms/episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| familiar60k | all_off | 6.983 | 3928.0 | 4980.8 | 47 | 438.3 | 0.0 |
| familiar60k | half_off | 6.818 | 4040.3 | 5092.5 | 44 | 859.2 | 480.2 |
| familiar60k | all_on | 6.577 | 4197.5 | 5512.3 | 41 | 1311.1 | 1029.6 |
| familiar60k | half_on | 6.849 | 4012.1 | 5110.9 | 44 | 826.2 | 483.9 |
| short16k | all_off | 23.708 | 1172.9 | 1542.7 | 39 | 257.8 | 178.0 |
| short16k | half_off | 24.395 | 1141.1 | 1522.1 | 39 | 232.8 | 181.6 |
| short16k | all_on | 24.466 | 1134.4 | 1492.3 | 43 | 93.7 | 0.0 |
| short16k | half_on | 25.847 | 1072.7 | 1374.7 | 43 | 87.7 | 0.0 |
| heterogeneous | all_off | 1.720 | 15607.1 | 21435.9 | 98 | 1378.7 | 0.0 |
| heterogeneous | half_off | 1.720 | 15622.6 | 21472.9 | 97 | 1370.4 | 0.0 |
| heterogeneous | all_on | 1.722 | 15592.6 | 21446.9 | 97 | 1346.1 | 0.0 |
| heterogeneous | half_on | 1.723 | 15505.0 | 21409.5 | 99 | 1320.2 | 0.0 |

| Workload | Comparison | Aggregate gain | Paired gain (95% interval) | Mean TTFT reduction |
| --- | --- | ---: | --- | ---: |
| familiar60k | half_ready_deadline_off | -2.36% | -2.34% [-6.63%, +2.14%] | -2.86% |
| familiar60k | half_ready_deadline_on | +4.13% | +4.02% [-1.07%, +9.37%] | +4.42% |
| familiar60k | deadline_wait_all | -5.81% | -5.69% [-16.20%, +6.12%] | -6.86% |
| familiar60k | deadline_half_ready | +0.46% | +0.45% [-6.17%, +7.53%] | +0.70% |
| short16k | half_ready_deadline_off | +2.90% | +2.80% [-8.95%, +16.07%] | +2.71% |
| short16k | half_ready_deadline_on | +5.65% | +5.65% [+0.06%, +11.56%] | +5.44% |
| short16k | deadline_wait_all | +3.20% | +3.05% [-7.69%, +15.04%] | +3.28% |
| short16k | deadline_half_ready | +5.96% | +5.91% [-2.47%, +15.00%] | +5.99% |
| heterogeneous | half_ready_deadline_off | -0.01% | -0.01% [-0.43%, +0.41%] | -0.10% |
| heterogeneous | half_ready_deadline_on | +0.03% | +0.03% [-0.17%, +0.22%] | +0.56% |
| heterogeneous | deadline_wait_all | +0.11% | +0.11% [-0.27%, +0.50%] | +0.09% |
| heterogeneous | deadline_half_ready | +0.15% | +0.15% [-0.06%, +0.36%] | +0.75% |

<!-- markdownlint-enable MD013 -->

Each arm/workload has four reset-and-reprimed 75-request episodes: 300 requests
total, not one continuous 300-request run. Twelve 32-request warmups, startup,
and priming are excluded. Position and immediate carryover are balanced across
the four repeats. Confidence intervals use paired log throughput ratios with
episodes as samples. Four repeats provide limited precision; intervals spanning
zero do not establish a repeatable throughput benefit.

GPU gaps use all-rank execution timing and exclude startup/final tail. Compute
spans include submission stalls and output joins. Gap categories are classified
by the successor seal's compute state. The fraction denominator is the frozen
published candidate cohort (up to 16), not all outstanding transfers.
Cancellations cap the target. Compute-active readiness rules remain unchanged.

Audit: exact scheduled token accounting, unique ordered completion, equal
transfer bytes within every four-arm block, CPU restoration counts, all-rank
optimized preparation, frozen checkpoint replay, deadline/fraction arithmetic,
and no measured compilation/errors. Original selector and model weights are
preserved.

Artifacts: `/mnt/localnvme/glm53_load_only_deadline_ab_20261008`.
