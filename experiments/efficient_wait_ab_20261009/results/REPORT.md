# Efficient waiting ablation with unchanged sealing conditions

GLM 5.3 NVFP4, TP8, concurrency 32, one output token.
All three arms use the same frozen predictor and transfer/preparation path.

<!-- markdownlint-disable MD013 -->

| Workload | Policy | req/s | Mean TTFT ms | p95 TTFT ms | Compute rounds | GPU gap ms/episode | Load-only gap ms/episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| familiar60k | full | 6.836 | 3993.8 | 4957.2 | 61 | 1108.0 | 939.8 |
| familiar60k | bridge | 6.786 | 4054.5 | 5556.0 | 64 | 1040.2 | 734.4 |
| familiar60k | poll | 6.832 | 4013.4 | 4970.0 | 65 | 954.9 | 643.6 |
| short16k | full | 24.670 | 1115.9 | 1494.2 | 62 | 164.8 | 89.5 |
| short16k | bridge | 24.349 | 1130.1 | 1436.5 | 64 | 173.6 | 87.5 |
| short16k | poll | 24.655 | 1118.1 | 1481.3 | 60 | 230.8 | 174.1 |
| heterogeneous | full | 1.721 | 15600.8 | 21435.7 | 150 | 1353.9 | 0.0 |
| heterogeneous | bridge | 1.721 | 15606.2 | 21436.6 | 149 | 1336.5 | 0.0 |
| heterogeneous | poll | 1.721 | 15570.7 | 21433.4 | 150 | 1347.8 | 0.0 |

| Workload | Comparison | Aggregate gain | Paired gain (95% interval) | Mean TTFT reduction |
| --- | --- | ---: | --- | ---: |
| familiar60k | efficient_wait_vs_poll | +0.05% | +0.08% [-5.93%, +6.47%] | +0.49% |
| familiar60k | coalescing_vs_bridge | +0.73% | +0.79% [-4.28%, +6.13%] | +1.50% |
| familiar60k | hybrid_bridge_vs_poll | -0.67% | -0.70% [-5.16%, +3.96%] | -1.02% |
| short16k | efficient_wait_vs_poll | +0.06% | -0.01% [-7.49%, +8.09%] | +0.20% |
| short16k | coalescing_vs_bridge | +1.32% | +1.27% [-3.64%, +6.42%] | +1.25% |
| short16k | hybrid_bridge_vs_poll | -1.24% | -1.26% [-9.84%, +8.14%] | -1.07% |
| heterogeneous | efficient_wait_vs_poll | +0.01% | +0.01% [-0.10%, +0.13%] | -0.19% |
| heterogeneous | coalescing_vs_bridge | -0.01% | -0.01% [-0.03%, +0.01%] | +0.04% |
| heterogeneous | hybrid_bridge_vs_poll | +0.03% | +0.03% [-0.09%, +0.15%] | -0.23% |

| Workload | Policy | Progress polls/episode | Wait returns/episode | Wait CPU ms/episode | EngineCore CPU ms/episode | Exposed compute hold ms/episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| familiar60k | full | 1595.5 | 600.0 | 91.05 | 1593.3 | 0.39 |
| familiar60k | bridge | 2026.0 | 946.0 | 205.06 | 1673.3 | 5.21 |
| familiar60k | poll | 3157.8 | 1896.5 | 202.16 | 1678.3 | 5.57 |
| short16k | full | 435.3 | 158.8 | 40.09 | 471.7 | 1.46 |
| short16k | bridge | 744.2 | 379.7 | 97.48 | 550.0 | 0.63 |
| short16k | poll | 837.5 | 529.7 | 66.52 | 485.0 | 0.57 |
| heterogeneous | full | 250.5 | 21.0 | 3.50 | 2988.3 | 0.00 |
| heterogeneous | bridge | 290.2 | 19.7 | 3.45 | 3000.0 | 0.00 |
| heterogeneous | poll | 252.0 | 19.3 | 2.86 | 3033.3 | 0.00 |

<!-- markdownlint-enable MD013 -->

Each arm/workload has six reset-and-reprimed 75-request episodes:
450 requests total, not one continuous 450-request run. Nine
32-request warmups, startup, and priming are excluded. Position and
within-block carryover are balanced across six repeats. Confidence
intervals use paired log throughput ratios with episodes as samples.
Six repeats provide limited precision; intervals spanning zero do
not establish a repeatable throughput benefit.

GPU gaps use all-rank execution timing and exclude startup/final tail.
Compute spans include submission stalls and output joins. Gap
categories are classified by the successor seal's compute state.
The fraction denominator is the frozen published candidate cohort
(up to 16), not all outstanding transfers. Cancellations cap the
target. Compute-active readiness rules remain unchanged.

Audit: exact scheduled token accounting, unique ordered completion,
equal transfer bytes within every three-arm block, CPU restoration
counts, all-rank optimized preparation, frozen checkpoint replay,
deadline/fraction arithmetic, and no measured compilation/errors.
Original selector and model weights are preserved.

Artifacts: `/mnt/localnvme/glm53_efficient_wait_ab_20261009`.
