# Decisions from the polling and prediction study

## Result

Keep the new policies opt-in. There is no universal throughput winner across
the three tested GLM 5.3 TP8 workloads. Larger polling intervals reduce polling
CPU work; the prediction followed by readiness policy makes that reduction
without relying on the prediction to finalize membership.

## Larger polling intervals

The screen tested 1, 2, 5 and 10 ms for both readiness and original prediction.
The selected policies were repeated with a fresh seed and reversed order.
Only post-acknowledgment adaptive polling changes; pre-ack polling stays 25 us.

Across the two seeds, readiness at 10 ms versus 1 ms gives:

| Workload | Throughput change | Measured polling CPU reduction |
| --- | ---: | ---: |
| 8K / 95% cached | +3.91% | 82.42% |
| 60K / 95% cached | -0.24% | 76.98% |
| 60K / 99% cached | +0.36% | 83.81% |

The short-workload throughput improvement repeats in direction, but its size
varies: +6.60% in screening and +1.36% in confirmation. The long-workload changes
are small. The 2 and 5 ms intervals have screening measurements only; these
results do not establish an optimal interval. Polling CPU is the thread-time
counter for iterations leading to adaptive sleeps, rather than total engine CPU.

## Prediction changes

The recent preparation reserve uses nearby batch shapes from the same context
regime and recent history, with a smaller percentile and a 10 ms margin. Compared
with original prediction at 1 ms, its two-seed mean throughput changes are
-3.77%, +0.18% and -1.12% for the three workloads above. It does not establish
a consistent improvement. Context regime alone cannot distinguish a loaded
CPU prefix from a partially computed cold prompt; the 10 ms fallback can also
underestimate preparation.

The initial hybrid with 20 ms coarse waits still allows the forecast to seal.
Its two-seed mean throughput changes versus fixed 200 + 100 ms are -3.64%,
+1.04% and -1.59%. Its modest long/95% improvement repeats, while it regresses
on the other workloads.

## Prediction followed by readiness

The separate branch implements the proposed combination:

1. Use the first conditioned forecast to set the transition into fine polling.
2. Initially sleep in intervals up to 50 ms, bounded by that transition and the
   safety deadline; refresh progress at every wake.
3. In the last 50 ms before the forecast seal deadline, switch permanently to
   2 ms polling. Missing forecasts use fine polling immediately.
4. Seal when all active published candidates are KV-ready and acknowledged,
   preceding physical compute completes, or the 2 s safety limit is reached.
   With only loads pending, the target is the smaller of four and the number
   of active candidates. Cancellation reduces the target.

The forecast never seals this policy. Conditions arriving during a sleep are
observed at the next wake; fine sleeps retain their original fixed-duration
behavior. This is a coarse checking phase followed by fine polling, rather
than an uninterrupted blind sleep until predicted completion.

The matched comparison on a new server uses seed 56:

| Policy | 8K / 95% req/s | 60K / 95% req/s | 60K / 99% req/s |
| --- | ---: | ---: | ---: |
| Fixed 200 + 100 ms | 40.797 | 6.939 | 11.551 |
| Original prediction, 1 ms | 37.663 | 6.997 | 11.636 |
| Readiness, 2 ms | 39.940 | 7.115 | 11.303 |
| Prediction followed by readiness | 39.590 | 7.118 | 11.629 |

Relative to readiness at 2 ms, the new hybrid changes throughput by -0.88%,
+0.04% and +2.88%. Relative to original prediction, the changes are +5.12%,
+1.73% and -0.06%. These are single replays within the new server, not repeated
proof of a throughput gain. It beats original prediction in the first two
cells and is effectively tied in the third, but does not improve every
readiness or fixed-hold control.

On 8K, measured polling CPU decreases from 78.09 to 65.17 ms and exposed hold's
lower bound decreases from 96.32 to 17.84 ms, yet throughput is slightly lower
and compute rounds increase from 38 to 40. On 60K/95%, polling CPU decreases
from 108.31 to 52.55 ms, with 13 compute rounds in both cells and essentially
equal throughput. Lower polling CPU and lower exposed hold do not independently
guarantee a throughput gain. See the full findings for all three diagnostics.

On 60K/99%, polling CPU decreases from 123.7 to 84.6 ms, while throughput
improves by 2.88% and mean TTFT falls by 2.14% versus readiness. Both use 13
compute rounds. The exposed-hold lower bound increases from 326.6 to 333.6 ms;
the coarse wait is not guaranteed to reduce exposure on every workload.

## Verification and reproducibility

The registered matrix has 78 replays and 11,960 measured requests. All replay
requests have CPU prefix hits; 95%/99% is the cached fraction of each prompt,
rounded to 64-token blocks. Short cells have 300 requests and long cells 80,
at concurrency 32 with one output token. Every policy independently resets
and primes its cache. The worker control already includes asynchronous H2D,
two-chunk pipelining and early KV submission; it is not unoptimized vLLM.

The new branch fixes a cancellation snapshot race when A completes during
the adaptive sleep. The old runtime deterministically fails the added CPU
regression; the corrected runtime passes all 70 targeted protocol/predictor
checks. Twelve selected option checks and all commit hooks pass. Each of the
four new policies passes 16/16 known answers and 16/16 distinct facts. The
fixed and original-prediction distinct-fact probes have two wording differences
each; readiness and the new hybrid match all cold answers exactly.

Two broader inherited fixture/artifact failures are retained and reproduced on
the unchanged evaluation source. They are separate from the targeted passing
checks. Aggregate replay gates, native SHA checks and equal uncached token work
are recorded in [verification](verification.json).

One 8K readiness-at-1ms confirmation batch (step 7958) lacks rank 6's GPU
execution timing in the engine snapshot. All eight primary completions are
valid. The batch-shape report initially coupled token accounting to timing
availability and omitted its 4,480 tokens; that report bug is corrected. The
cell retains all 300 responses and all 134,400 uncached tokens. Its GPU interval
and opportunity diagnostics omit the unavailable interval, with no imputed
timing. All other replay batches have complete execution timing. Raw throughput
and TTFT are unaffected by this reporting correction.

Prediction uses only past observations, but training history persists across
policies and priming. Reversed order helps assess sensitivity; it does not give
each predictor identical history. The new server has its own matched controls
and results should be compared within that phase. The experiment covers three
prompt/cache shapes at one concurrency, rather than long generation or other
models. No policy is promoted to a default.

## Artifacts

- [Complete numerical findings](FINDINGS.md)
- [Full tables](report.html)
- [Polling plot](polling_screen.png)
- [Two-seed strategy plot](strategy_comparison.png)
- [New hybrid throughput and TTFT plot](predicted_readiness_comparison.png)
- Raw evidence: `/mnt/localnvme/glm53_poll_prediction_20261003`
- Separate implementations and evidence archives: paths in
  `/mnt/localnvme/glm53_poll_prediction_20261003/branches.json`
