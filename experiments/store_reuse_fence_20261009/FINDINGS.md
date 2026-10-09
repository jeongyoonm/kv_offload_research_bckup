# Why stores block forward, and the GPU fence experiment

## Root cause

The offloading scheduler logically releases a finished request's GPU blocks
while its GPU-to-CPU store may still be pending. The ordinary block allocator
chooses from its free queue without checking pending stores. The connector
detects intersections between newly allocated block IDs and pending store
source IDs and adds those stores to `jobs_to_flush`. The worker submits deferred
stores and then calls `wait()` before forward or new H2D loads. On CUDA, the
store handler implements this as `event.synchronize()`, blocking the CPU worker.

The safety dependency is real once those particular source blocks are chosen:
the old GPU bytes must reach CPU memory before a load or forward overwrites
them. The store transfer stream first waits for the current compute stream,
and store jobs are serialized behind the preceding store's completion event.
A selected store's completion therefore also depends on earlier queued stores.
This is a per-store event wait, not a global device synchronization.

There are two distinct costs: host-side store submission and waiting for store
completion. New direct markers divide submission into descriptor preparation,
stream dependencies, and the copy API wrapper. In the matched control, the
wrapper accounts for approximately 87.5% of submission wall time. Python
descriptor preparation is about 11.1%. The wrapper timer includes Torch and
CUDA API work, including possible internal blocking; it does not identify a
particular driver lock or prove all this time is active CPU work.

This is not evidence of physical HBM exhaustion. The allocation uses a fixed
96 GiB GPU KV budget per device and reports 2,160,960 KV tokens. Thirty-two
61,440-token contexts would occupy approximately 91% of that reported token
capacity. An early memory snapshot showed substantial remaining physical HBM.
All measured episodes had zero preemptions. Finite pool recycling and allocator
policy can create collisions without OOM. These traces do not establish whether
alternative safe free blocks existed at each allocation.

## Implemented change

Branch: `experiment/store-reuse-gpu-fence-20261009`.
Runtime and serving-driver commit: `03b84ce`.
Artifacts: `/mnt/localnvme/jeongyoon/store_reuse_fence_20261009/paired`.

`store_reuse_gpu_fence=true` replaces the ordinary CUDA store-reuse host wait
with `current_stream.wait_event(store_end)`. It lets the CPU enqueue the remaining
forward work while the store is pending. Both model writes and H2D writes stay
ordered after the store: native H2D submission records its ready event on that
same compute stream before the independent copy worker waits for it.

Completed stores are still acknowledged only by normal event completion. Event,
stream, and descriptor buffers retain their existing completion-based ownership.
Reset metadata explicitly requests a host drain, and mixed load/store flushes
retain the original host wait. Generic backends and non-CUDA paths use the host
fallback. No accounting, membership, predictor weights, model kernels, weights,
KV precision, or sparse attention selector were changed. The flag defaults off.

## Matched GLM 5.3 TP8 comparison

Each arm ran four independently primed/reset 75-request episodes: 300 measured
requests per arm, not a continuous 300-request stream. Separate 32-request
warmups are excluded. The workload is 61,440 prompt tokens, a 58,368-token CPU
prefix (95%), 3,072 uncached suffix tokens, concurrency 32, and one output token.
Both arms use the latest input preparation, private CPU cache initialization,
8 MiB/depth 2 H2D worker, readiness exits, 50% load-only target, efficient waiting,
frozen predictor, 90 ms prediction subtraction, and 100 ms setup lead. The only
policy difference is the completion fence. Arm order alternates by repeat;
seeds are 9301-9304.

Levels below are episode means. Percentages for throughput and TTFT use the
geometric mean of matched per-repeat ratios. Timing percentages compare the
reported means. Times are per 75-request episode, unless explicitly noted.

| Metric | Existing host wait | GPU stream fence | Change |
| --- | ---: | ---: | ---: |
| Throughput, req/s | 7.771 | 7.718 | -0.69% |
| Mean TTFT, ms | 3536.5 | 3545.5 | +0.25% |
| p95 TTFT, ms | 4767.6 | 4740.2 | -0.54% |
| Inter-compute gaps, ms | 287.9 | 233.3 | -18.97% |
| Exposed pre-forward phase, ms | 184.5 | 96.4 | -47.74% |
| Forward enqueue to GPU start, ms | 42.4 | 99.2 | +133.66% |
| Exposed host store completion wait, ms | 91.7 | 0.0 | Removed |
| Full host store completion wait, ms | 1171.2 | 0.0 | Removed |
| Full GPU fence enqueue cost, ms | 0.0 | 0.104 | About 104 us/episode |
| Full store submission wall time, ms | 282.2 | 293.4 | +3.97% |
| Exposed store submission wall time, ms | 46.2 | 62.2 | +34.68% |
| Full store copy API wrapper, ms | 246.9 | 256.4 | +3.84% |
| Full store descriptor preparation, ms | 31.5 | 33.5 | +6.59% |
| Conservative execution envelope, ms | 9091.5 | 9092.2 | +0.009% |
| Compute rounds | 12.5 | 12.5 | Equal |
| Late admissions | 68.25 | 68.00 | Nearly equal |
| Recovered / compute-period KV-ready | 93.47% | 93.10% | Nearly equal |

Throughput changes by matched repeat: +0.88%, -2.09%, -1.11%, -0.41%.
B's forward submission precedes A's conservative end in 17/46 transitions
with the host wait, versus 33/46 with the GPU fence. These direct call markers
confirm removal of the CPU stall. The larger forward-to-GPU-start phase shows
that some waiting moves to the GPU dependency. The GPU fence does not remove
store DMA, serialize fewer stores, or shorten the copy API calls.

Store diagnostics sum the largest per-rank span per transition. Different
rank maxima can refer to different ranks, so they are not an additive breakdown
of the all-rank host phase partition. The host phase partition itself sums
exactly to each measured gap. Overlapping execution envelopes count as zero gap.

## Why throughput does not follow the gap reduction

The conservative first-to-last GPU span improves by 53.9 ms on average, while
mean client elapsed time worsens by 67.3 ms. The recorded client time is:

| Mean time, ms/episode | Host wait | GPU fence |
| --- | ---: | ---: |
| Earliest request start to first compute | 198.2 | 295.8 |
| First-to-last conservative GPU span | 9379.4 | 9325.6 |
| Last conservative GPU end to final response | 72.8 | 84.2 |
| Additional client harness time | 0.5 | 12.7 |
| Total client elapsed | 9650.9 | 9718.2 |

Before the first compute batch, rank 0 records zero store submissions and zero
host/GPU reuse fences in all eight episodes. The first compute batch contains
one ordinary request in every episode. The approximately 98 ms difference in
initial waiting occurs before the changed fence executes; it cannot be directly
attributed to executing that fence. Startup arrival/load-submission behavior
and inherited process state need separate controls to explain it. Final-response
and client scheduling variation also matter. Four descriptive repeats do not
establish a significant slowdown or speedup. There is no demonstrated serving
throughput benefit, so the feature remains disabled by default.

## Remaining work suggested by the evidence

The tested change resolves unnecessary CPU completion blocking. The remaining
GPU ordering requirement follows from the selected blocks and the serial store
queue. Avoiding that dependency needs another experiment: prefer free blocks
without pending stores, prioritize required stores, or scope store fences to
the affected H2D loads when current computation does not overwrite those blocks.
The last option needs explicit source/destination classification and separate
compute versus load dependencies. These approaches were not implemented or
measured in this allocation. The measurements do not establish that every
existing global forward fence is necessary.

## Verification and cleanup

- CPU seal harness: 206 passed, 8 skipped.
- Reset/source-reuse scheduler checks: 5 passed.
- Selected connector/handler checks: 17 passed, including four real-GPU
  byte/order checks for forward and independent H2D overwrites. Three connector
  checks were rerun after a test typing fix and passed; they are included in 17.
- Selected prediction/readiness configuration checks: 15 passed.
- Deliberately removing the fence produced 100% byte mismatches in both forward
  and native-H2D negative controls, demonstrating that the regression detects
  the source-overwrite race. With the fence, stored and final GPU bytes match.
- Exactly-once store submission/completion checks passed. Reset and mixed-load
  drains preserve the host completion boundary.
- All eight measured and both warmup episodes passed expected CPU restoration,
  zero unexpected GPU prefix hits/preemptions, one output token, unchanged
  frozen predictor, and optimized preparation coverage on all eight ranks.
  These serving checks and GPU-unit byte checks are not an all-rank live DMA
  checksum audit or a semantic model accuracy evaluation.
- No JIT compilation, traceback, or error messages were found in measured
  server-log windows. Startup, priming, and warmups are excluded.
- Applicable normal pre-commit hooks passed, including mypy; none bypassed.
- The owned API, engine, and all eight workers exited. All GPU allocations are
  zero and no GPU compute processes remain. Shutdown emitted inherited
  resource-tracker cleanup warnings; final cleanup was independently checked.

Reproduce the timing report with the command in `README.md`. The committed
results include the comparison, bookend decomposition, frozen checkpoint,
verification, cleanup, and an archived set of raw serving/timing records.
