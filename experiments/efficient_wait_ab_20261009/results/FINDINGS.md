# Efficient waiting: final findings

## Conclusion

Efficient waiting reduces scheduler progress CPU on the cached workloads, but
this study does not establish a repeatable throughput or TTFT advantage over
plain 2 ms polling. Keep it as a CPU efficiency option; plain 2 ms polling is a
reasonable simpler alternative under the measured configuration. No serving
defaults were changed.

## Controlled comparison

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. The control is the latest
prototype, including optimized preparation, async chunked H2D, earlier KV
submission, frozen predictor with a 5 ms deadline lead, 16 candidates, and the
50% load-only target. All arms retain identical sealing rules and model
execution.

- **full:** uncapped prediction-guided initial wait minus a 50 ms guard,
  producer-coalesced notifications, then 2 ms fine polling.
- **bridge:** the same wait, with the older 1 ms transport polling observer and
  partial-rank notifications.
- **poll:** plain 2 ms scheduler polling; predictor-based sealing deadlines
  remain enabled.

Six matched seed blocks use all six arm permutations. Each arm/workload has six
reset-and-reprimed 75-request episodes (450 requests), not a continuous
450-request run. Total: 54 episodes and 4,050 measured requests. Nine 32-request
warmups, cache priming, and startup are excluded. Load-only waiting is the same
2 ms polling in every arm.

## Serving results

<!-- markdownlint-disable MD013 -->

| Workload | Arm | Throughput req/s | Mean TTFT ms | p95 TTFT ms |
| --- | --- | ---: | ---: | ---: |
| short16k | poll | 24.6550 | 1118.1 | 1481.3 |
| short16k | bridge | 24.3493 | 1130.1 | 1436.5 |
| short16k | full | 24.6702 | 1115.9 | 1494.2 |
| familiar60k | poll | 6.8324 | 4013.4 | 4970.0 |
| familiar60k | bridge | 6.7864 | 4054.5 | 5556.0 |
| familiar60k | full | 6.8357 | 3993.8 | 4957.2 |
| heterogeneous | poll | 1.7209 | 15570.7 | 21433.4 |
| heterogeneous | bridge | 1.7213 | 15606.2 | 21436.6 |
| heterogeneous | full | 1.7211 | 15600.8 | 21435.7 |

| Workload | Full vs poll aggregate throughput gain | Paired gain, nominal 95% interval | Mean TTFT reduction |
| --- | ---: | --- | ---: |
| short16k | +0.06% | -0.01% [-7.49%, +8.09%] | +0.20% |
| familiar60k | +0.05% | +0.08% [-5.93%, +6.47%] | +0.49% |
| heterogeneous | +0.01% | +0.01% [-0.10%, +0.13%] | -0.19% |

All throughput intervals span zero. Six blocks give limited precision for the
cached workloads; this is not an equivalence proof. The intervals use paired
episode log ratios and are unadjusted for the nine contrasts. Tail TTFT does not
show a consistent improvement.

## CPU benefit and physical mechanism

| Workload | Full progress CPU ms/episode | Poll progress CPU ms/episode | Progress CPU reduction | Whole-engine CPU reduction |
| --- | ---: | ---: | ---: | ---: |
| short16k | 40.09 | 66.52 | +39.7% | +2.7% |
| familiar60k | 91.05 | 202.16 | +55.0% | +5.1% |
| heterogeneous | 3.50 | 2.86 | -22.3% | +1.5% |

Cached-workload progress CPU savings are repeatable in paired measurements:

- 16K: full minus poll is −26.43 ms/episode, 95% interval [−42.79, −10.07].
- 60K: full minus poll is −111.11 ms/episode, 95% interval [−132.05, −90.18].

The full arm avoids the bridge observer scans and reduces coarse returns caused
by partial readiness. On 60K, the bridge averages 2,757 scans and 309 coarse
returns per episode, versus zero scans and 52 coarse returns for full. On 16K,
these are 1,131 scans and 297 coarse returns versus zero and 56. Compared with
the bridge, full reduces progress CPU by 55.6% on 60K and 58.9% on 16K. Plain
polling instead returns from waits 1,897 times per 60K episode and 530 per 16K
episode, versus 600 and 159 for full. These counters are observed progress
operations, not kernel wakeup counts.

Whole-engine CPU point reductions versus plain polling are only 5.1% (60K) and
2.7% (16K), and their paired intervals include zero. Absolute progress savings
are about 1.48 ms/request and 0.35 ms/request, respectively. Most serving CPU
work is outside this progress loop. Mixed episodes average fewer than one coarse
wait; their progress CPU differences are tiny and do not establish a benefit.

## Batching and GPU gaps

| Workload | Arm | Compute rounds across 450 requests | Recovered compute-ready opportunities / total | Mean GPU gap ms/episode | Compute span ms/episode | Exposed compute hold ms/episode |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| short16k | poll | 60 | 366 / 384 | 230.8 | 2683.2 | 0.57 |
| short16k | bridge | 64 | 383 / 398 | 173.6 | 2767.9 | 0.63 |
| short16k | full | 62 | 392 / 400 | 164.8 | 2740.7 | 1.46 |
| familiar60k | poll | 65 | 370 / 380 | 954.9 | 9700.3 | 5.57 |
| familiar60k | bridge | 64 | 363 / 378 | 1040.2 | 9690.2 | 5.21 |
| familiar60k | full | 61 | 355 / 369 | 1108.0 | 9535.4 | 0.39 |
| heterogeneous | poll | 150 | 5 / 219 | 1347.8 | 42169.7 | 0.00 |
| heterogeneous | bridge | 149 | 5 / 215 | 1336.5 | 42179.3 | 0.00 |
| heterogeneous | full | 150 | 6 / 217 | 1353.9 | 42178.0 | 0.00 |

<!-- markdownlint-enable MD013 -->

Efficient waiting does not consistently reduce total GPU gaps or improve
opportunity recovery. On 60K, full reduces compute spans by about 165 ms/episode
relative to poll but increases inter-compute gaps by about 153 ms/episode. On
16K, it reduces gaps by about 66 ms but increases compute spans by about 58 ms.
These offsets are consistent with the near-identical throughput. The changed
wait cadence changes batch trajectories despite identical sealing conditions;
this ablation does not isolate a specific kernel or synchronization bottleneck.

## Verification and limits

- CPU protocol harness: 198 passed, 8 GPU-gated skips.
- All 54 live measurement audits and 18 matched-block audits passed.
- All 290 issued native predictions replay exactly against the frozen
  checkpoint.
- Exact uncached token accounting, expected CPU restoration, zero GPU prefix
  hits/preemptions, ordered unique retirement, all-rank compact/pinned
  preparation, equal transfer volume within matched blocks, and deadline/target
  arithmetic passed.
- No measured compilation or runtime errors. Original selector and weights
  preserved; no model-quality or bitwise-output equivalence claim.

The progress CPU counter covers instrumented progress iterations and
bridge-thread work, not every final-loop instruction or producer notification
cost. Whole-engine and worker CPU are separate process measurements. Common
submission-progress diagnostics also run at polling opportunities, so the
measured savings apply to this instrumented prototype; untraced production
savings were not measured. GPU compute spans include submission stalls and
output joins. Exposed compute hold is only forecast-conditioned waiting after
physical predecessor completion and base readiness, not the entire GPU gap.
Issued forecast populations may differ as batch trajectories change.

Raw evidence: `/mnt/localnvme/glm53_efficient_wait_ab_20261009`. Reproducible
experiment source and archived evidence: branch
`experiment/glm53-efficient-wait-ablation-20261009`, under
`experiments/efficient_wait_ab_20261009`. See `comparison.json`, `REPORT.md`,
and `waiting_ablation.png`.
