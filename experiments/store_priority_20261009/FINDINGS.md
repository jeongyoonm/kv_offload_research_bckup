# Critical store submission order

The candidate stably submits reuse-critical deferred stores first when their
CPU destination sets are disjoint. Shared or unknown destinations retain FIFO
order. Already submitted transfers are unchanged and retirement still requires
actual completion. Both arms use the same GPU reuse fences.

Four separately primed/reset 75-request episodes per arm measured 300 requests
per arm. GLM 5.3 NVFP4 TP8, the original selector, FP8 KV, 96 GiB GPU cache per
rank, 2.13 TB aggregate private CPU cache, 60K prompts with 95% CPU-restored
prefixes, concurrency 32, optimized preparation, 8 MiB/depth-two H2D and the
frozen scheduler predictor remained fixed.

| Metric | FIFO | Critical first |
| --- | ---: | ---: |
| Mean throughput (requests/s) | 7.7181 | 7.8251 |
| Mean TTFT (ms) | 3581.33 | 3519.92 |
| Mean episode p95 TTFT (ms) | 4841.55 | 4694.35 |
| Inter-compute gaps per episode (ms) | 189.72 | 91.32 |
| Pre-forward CPU phase per episode (ms) | 92.87 | 43.57 |
| Forward-to-GPU-start wait per episode (ms) | 62.32 | 39.77 |
| Compute rounds per episode | 12.25 | 12.25 |

Paired geometric throughput change is +1.392%; per-repeat changes are +0.011%,
-0.377%, +2.922%, +3.063%. These are descriptive repeats, not a significance
claim. Rank-zero traces contain 16 deferred batches whose candidate submission
order differs from increasing job IDs, confirming that the mechanism operates.
The trace does not record the counterfactual order or critical set, so this
count is evidence of reordering rather than a complete priority audit.

The conservative GPU span decreases 113.98 ms: compute envelopes decrease
15.57 ms and gaps decrease 98.40 ms. Exposed deferred submission decreases
70.71 to 24.25 ms, including copy API intervals of 63.87 to 21.50 ms. These
nested intervals overlap and must not be added. Time before first compute also
decreases 47.74 ms; bookends do not identify its cause. Thus the full throughput
change cannot be attributed to priority alone. The allocator preference remains
the simpler remedy because it removes these observed reuse dependencies.

Verification: two backend and five connector checks passed; all eight live
cells passed restoration counts, zero GPU prefix hits, no preemption, one-token
output, frozen predictor and all-rank preparation checks. Original selector
nondeterminism remains unchanged. No model accuracy claim follows from these
serving controls.

Implementation: 9b5308951978a265e5a3bc96c6a6b77e2c859148.
Raw traces: `/mnt/localnvme/jeongyoon/sp1`; the complete verified archive,
comparison, bookends and diagnostics are retained under `results/`.
