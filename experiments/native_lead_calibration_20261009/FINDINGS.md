# Setup-lead comparison after background D2H

Completed and stopped: two matched continuous 300-request episodes per
arm, 600 measured requests each. Both arms use native full stores,
allocator avoidance, the frozen predictor and its 90 ms subtraction.
Only setup lead changes. Warmups and priming are excluded. Every request
restores the exact CPU prefix with zero GPU hits; preparation is verified.

| Lead | Req/s | Rounds | Gaps ms | Recovery % | Mean TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 ms | 8.178 | 41.5 | 255.4 | 95.28 | 3753.9 |
| 25 ms | 8.138 | 40.5 | 292.7 | 98.65 | 3769.4 |

Mean paired throughput changes -0.49%, with +0.16% and -1.13% across
repeats. Mean episode p95 TTFT improves from 4694.3 to 4594.1 ms.
Compute envelopes remain similar, 36167.3 versus 36163.5 ms.
Exposed pre-forward intervals rise from 16.1 to 84.7 ms per episode;
forward-to-GPU-start exposed intervals fall from 23.1 to 15.3 ms. These
are clipped gap overlaps, not total batch setup costs or additive phases.

The shorter lead improves admissions and consolidates one compute round
on average, but does not improve measured throughput. Increased exposed
setup and gap variation remain. This does not establish a universal best
lead or statistical significance. Keep the tested 100 ms configuration
for the selected worker; do not promote 25 ms from admission gains alone.
Further lead calibration would need more repeats and diverse workloads.

The allocation finished in 1395.5 seconds, within its 1650-second owned
server timeout. No GPU tests or other serving allocations overlapped it.
Verified raw traces, launch state, client bookends and member hashes are
preserved in `artifacts/paired_continuous`. Serving acceptance is not a
whole-model accuracy claim; original model execution remains unchanged.
