# D2H descriptor merging: live fragmentation prevents merges

Both arms use global GPU reuse fences and the same latest GLM 5.3 TP8
prototype, frozen predictor, preparation, 8 MiB/depth-two H2D pipeline,
90 ms prediction subtraction and 100 ms setup lead. Original model, selector,
FP8 KV, GPU and private CPU cache capacities are unchanged. Four separately
primed/reset 75-request 60K episodes per arm give 300 requests per arm at
concurrency 32, 95% CPU-prefix hits and one generated token.

The candidate sorts source pointers and merges only exact adjacent source
and destination address ranges. No producer ordering, STREAM access ordering,
FIFO or descriptor-buffer ownership is relaxed.

## Serving and mechanism results

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Mean throughput (requests/s), all four pairs | 7.7524 | 7.8266 |
| Mean TTFT (ms), all four pairs | 3569.60 | 3503.45 |
| Mean episode p95 TTFT (ms), all four pairs | 4903.13 | 4783.52 |
| Inter-compute gaps per episode (ms), all four pairs | 197.82 | 117.01 |
| Store descriptors before and after merging | 4752 | 4752 |

All four measured candidate episodes on all eight ranks have descriptor
ratio exactly 1.0. There are no physically mergeable address pairs. The
candidate still sorts pointers, so this is not an identical-code control,
but no throughput change can be attributed to merging fewer descriptors.

A broad preexisting regression suite unintentionally included GPU allocation
checks during the r2 control priming/measurement window. It was stopped;
subsequent tests were explicitly CPU-filtered. Keep the raw result and its
quality audit, and exclude the entire r2 pair for serving inference. Across
all four pairs, geometric throughput change is +0.958%; across the three
remaining pairs it is +0.350%. Individual changes are +2.084%, -1.570%,
+2.805% (potentially contaminated), and +0.571%. These few descriptive repeats
do not establish a reliable serving improvement.

## Next step and evidence

The allocation-order proposal assigns a request's already-selected GPU blocks
in ascending physical order before its logical block table is created. It
preserves capacity and references, and may permit merging against consecutive
CPU pages. Its actual merge rate, byte safety and throughput require new tests.

Initial coalescing verification: 9 real GPU checks passed, including preserved
source fences. Every live cell passed restored-token, zero-GPU-hit,
no-preemption, one-output-token, frozen-state and all-rank preparation checks.
The established original sparse-selector nondeterminism is unchanged.

Implementation: 8d999c11282a5b08322a245063813f8663295e05.
Raw traces: /mnt/localnvme/jeongyoon/sc1, also verified in
results/raw_traces.tar.gz. See quality_audit.json and store_diagnostics.json.
