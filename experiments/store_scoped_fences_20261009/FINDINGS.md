# Scoped store fence findings

Four matched 75-request episodes per arm; 300 measured requests per arm.
Both arms use the previous GPU store fence, fixed predictor and scheduler.

| Metric | Global fence | Per-load fences |
| --- | ---: | ---: |
| Mean throughput (requests/s) | 7.8077 | 7.8176 |
| Mean TTFT (ms) | 3521.31 | 3521.20 |
| Mean episode p95 TTFT (ms) | 4574.82 | 4765.92 |
| Compute rounds per episode | 12.25 | 12.50 |
| Sum inter-compute gaps per episode (ms) | 145.72 | 217.25 |
| Forward enqueue to GPU start in gaps (ms) | 77.21 | 21.24 |

Paired geometric throughput change: +0.128%. Per-pair changes range from -1.14%
to +1.90%; this does not demonstrate a throughput improvement. GPU waiting after
forward enqueue falls, but earlier pipeline stages lose more time. The narrowed
fence does not address reserved suffix blocks outside H2D destination lists.

The first long-path startup was rejected before measurement because its TMPDIR
exceeded FlashInfer Unix socket limits. The valid allocation used the short root
`/mnt/localnvme/jeongyoon/sf1`, had no such warnings, passed all cache restoration,
one-token, frozen-estimator and preparation checks, and released all GPUs.

Twenty-two focused tests passed, including real delayed-DMA byte/order tests and
progress of unrelated GPU work. Original sparse selection, weights and KV
precision were unchanged. These tests do not constitute model accuracy evals.
Raw traces, launch configuration, result files and logs are preserved in results;
archive member hashes were reread and verified. Keep this proposal disabled.
