# Full future load reservations in private reuse gates

The candidate classifies the entire future load request's GPU reservation,
including suffix blocks that are not H2D destinations. Reuse dependencies gate
that request's prefix copy on a private stream. Its completion therefore also
protects later computation into the reserved suffix. Current compute uses its
own dependency set; unsafe lifecycle cases retain the broad fence fallback.
No store is retired before true completion.

Both arms use the same GLM 5.3 NVFP4 TP8 model, original selector, FP8 KV,
96 GiB per-rank GPU pool, 2.13 TB aggregate private CPU pool, optimized input
preparation, 8 MiB/depth-two H2D pipeline, frozen predictor, 90 ms prediction
subtraction and 100 ms setup lead. Four separately primed/reset 75-request
61,440-token episodes per arm give 300 measured requests per arm at concurrency
32, 58,368 CPU-restored tokens and one generated token per request.

| Metric | Global fence | Full-reservation private load gates |
| --- | ---: | ---: |
| Mean throughput (requests/s) | 7.7342 | 7.7965 |
| Mean TTFT (ms) | 3544.72 | 3516.15 |
| Mean episode p95 TTFT (ms) | 4689.01 | 4604.69 |
| Inter-compute gaps per episode (ms) | 158.37 | 137.65 |
| Forward-to-GPU-start wait per episode (ms) | 48.30 | 26.85 |
| Compute rounds per episode | 12.50 | 12.50 |
| Recovered / KV-ready-during-compute opportunities | 91.10% | 91.72% |

Paired geometric throughput change: +0.818%. Per-repeat changes are -0.699%,
+3.101%, +0.824%, +0.085%. These four repeats are descriptive, not a significance
claim. Conservative GPU spans decrease 41.8 ms on average; time before the
first compute also decreases 49.2 ms. Bookends cannot identify the cause of
that initial difference, so do not attribute the whole throughput change to
narrower fences. Compute envelopes decrease 21.1 ms and gaps decrease 20.7 ms.

Private dependency construction takes 5.08 ms per episode, with 1.52 ms inside
inter-compute gaps. The broader load-gate interval takes 50.27 ms and includes
H2D descriptor preparation/submission; these nested diagnostics must not be
added to the pre-forward phase. The allocator preference proposal removes
these observed dependencies entirely and remains the clearer root-cause fix.

Verification: 18 focused and 220 regression checks passed. All live cells
passed exact restoration counts, zero GPU prefix hits, no preemptions,
one-output-token, frozen-state and all-rank preparation checks. This does not
supersede the documented original sparse-selector nondeterminism or establish
model accuracy for all workloads.

Implementation: b48977eb06cbf4c4317f5fb305d6eb6aabcae7a2.
Raw traces: /mnt/localnvme/jeongyoon/sr1 and the verified
results/raw_traces.tar.gz. See comparison.json, bookends.json and
store_diagnostics.json for replayable evidence.
