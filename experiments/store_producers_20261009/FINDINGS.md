# Specific producer event result

Four matched 75-request episodes per arm, 300 measured requests per arm.
All eight cells passed cache-hit, no-preemption, one-token, frozen-estimator
and all-rank optimized-preparation checks. Both arms retain the GPU reuse fence.

| Metric | Broad stream dependency | Specific producer event |
| --- | ---: | ---: |
| Throughput, req/s | 7.817 | 7.568 |
| Mean TTFT, ms | 3520.081 | 3614.485 |
| Mean episode p95 TTFT, ms | 4786.000 | 4763.443 |
| Compute rounds | 12.000 | 12.000 |
| Inter-compute gaps, ms/episode | 144.771 | 528.305 |
| Pre-forward exposed phase, ms/episode | 88.477 | 101.379 |
| Forward to GPU start, ms/episode | 47.795 | 12.491 |

Paired geometric throughput change: −3.34%. Per-repeat changes were +0.014%,
−12.321%, +0.064%, and −0.533%. Keep the negative repeat; do not interpret
three near-ties as a demonstrated throughput gain.

At rank 0, 192 of 300 stores (64%) had a ready producer and a busy newer
compute stream. The mechanism therefore removes a real dependency. Two
stores still had an unfinished producer; real-GPU tests verify that this
producer remains protected while unrelated later compute can continue.
The mean copy API duration rose from 3.343 to 4.280 ms per store. These
observations do not identify a CUDA mutex or establish its contention cost.

The slow repeat had 1.497 s of hold without compute and 1.821 s of compute
gaps. Its first 473.4 ms load-only seal stall ended 14.0 ms before the
first changed store dependency. This does not establish that dependency
change caused the initial stall. Producer recording begins earlier, and
arrival/pipeline variation remains a possible contributor. A separate
preencoded-client control will reduce the approximately 80 ms spread in
initial request starts.

Conclusion: verified mechanism, no reliable throughput improvement in this
allocation. Keep the feature disabled and compare further proposals with
matched controls. Four descriptive repeats are not a significance estimate.

Raw traces and replay inputs are preserved in `results/raw_traces.tar.gz`;
`archive_manifest.json` verifies every member hash. Artifacts are also at
`/mnt/localnvme/jeongyoon/se2`. Two negative-control GPU tests fail when the
new producer dependency is replaced by the original broad stream wait.
