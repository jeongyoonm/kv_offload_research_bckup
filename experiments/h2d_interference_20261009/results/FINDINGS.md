# Interference-aware H2D calibration: measured findings

Status: complete. 1,143 unprofiled calibration cells, seven diagnostic cells,
and 18
accepted serving episodes (576 measured requests). Nine CPU checks pass.

## Fixed-cohort confirmation

Unchanged GLM 5.3 NVFP4 TP8, FP8 KV, eight user-identified B300 GPUs (reported
as L20D).
Nine combinations of three real cached compute cohorts and three reconstructed
native
transfer volumes. Each compute cohort contains eight requests. Four FIFO copy
jobs
repeat the same source/destination mapping within a cell; these private copy
buffers are
independent of the real model KV tensors. Sampling, input preparation,
allocation,
priming and validation are outside the timed forward window.

<!-- markdownlint-disable MD013 -->
| Compute shape | GPU-resident context/request | Uncached input tokens/request |
| --- | ---: | ---: |
| Short | 4,096 | 64 |
| Medium | 15,552 | 512 |
| Long | 58,368 | 2,048 |
<!-- markdownlint-enable MD013 -->

Copy profiles restore 4,096 / 15,552 / 58,368 tokens: 0.195 / 0.742 / 2.784 GB
per
GPU/job, or 0.782 / 2.967 / 11.137 GB across the four-job cohort. Decimal GB/s
uses
all-rank completion, including submission gaps.

The screen covered 8/16/32/64/128 MiB and depths 1/2, five randomized repeats:
615 cells
(150 copy-only, 15 compute-only, 450 concurrent). It froze three nominees:
128/2, 8/2,
16/2. An independent eight-repeat confirmation completed 528 cells (72
copy-only, 24
compute-only, 216 concurrent, 216 after-enqueue diagnostic). All rank geometry,
FIFO/status, sampled-byte and cleanup checks passed. The maximum native GPU-end
interval
uncertainty in confirmation was 0.612 ms.

<!-- markdownlint-disable MD013 -->
| Chunk / depth | Equal-geometry joint completion speedup | Nominal 95% interval | Joint latency reduction | Compute enqueue latency reduction |
| --- | ---: | --- | ---: | ---: |
| 8 MiB / 2 | +10.30% | [+9.56%, +11.04%] | 9.34% | 20.13% |
| 16 MiB / 2 | +9.52% | [+9.14%, +9.90%] | 8.69% | 16.54% |
| 128 MiB / 2 | +0.00% | [+0.00%, +0.00%] | 0.00% | 0.00% |
<!-- markdownlint-enable MD013 -->

Ratios receive equal weight across the nine geometries. Within-repeat averages
precede
paired log-ratio intervals; intervals are nominal and unadjusted for multiple
comparisons. Each cell changes a suffix-leading token to avoid suffix-cache
hits, so
this is not bit-identical model routing replay. Calibration speedups are not
serving
throughput gains.

## Submission interference evidence

A separate bounded Nsight capture includes one 60K copy job per rank, one
profiled
cell/config and one compute-only control. These profiled cells are excluded from
selection and unprofiled performance claims. Each rank launches 2,130 matched
forward
kernels in each captured case.

<!-- markdownlint-disable MD013 -->
| Setting | Mean rank host forward ms | Mean rank maximum copy API ms | Gaps of at least 0.5 ms between compute launches, summed ms | Portion overlapping copy APIs ms |
| --- | ---: | ---: | ---: | ---: |
| Compute only | 136.60 | 0.000 | 0.65 | 0.00 |
| 8 MiB / 2 | 152.78 | 0.779 | 6.31 | 6.24 |
| 128 MiB / 2 | 182.24 | 3.424 | 48.37 | 48.14 |
| 16 MiB / 2 | 159.39 | 1.527 | 8.18 | 7.67 |
<!-- markdownlint-enable MD013 -->

The 128 MiB copy API calls coincide with long gaps between compute launches.
Smaller
chunks split approximately the same total copy API time into shorter calls and
permit
more compute submission progress. The recorded compute-launch API union remains
about
10–11 ms; much of the slowdown is between those recorded calls. The default
trace omits
some CUDA query APIs and host work, so it does not identify the internal mutex
or prove
that recorded launch calls themselves block. Kernel-resident time includes NCCL
waiting
kernels and is not useful compute utilization.

Initial physical GPU kernel start remains around 0.14–0.16 ms in the concurrent
profiled
cells. The benefit is sustained submission progress and forward completion.

Starting copies only after compute enqueue restores host submission time, but
postpones
first KV readiness. With short compute and four 60K jobs, 128 MiB concurrent
joint
completion is 291.84 ms versus 367.45 ms after enqueue; 8 MiB is 265.77 versus
380.48
ms. With long compute, later submission can overlap remaining GPU work, but its
first KV
readiness is around 432–436 ms versus 65–71 ms for concurrent submission. This
timing
change can forfeit scheduler admission opportunities. The after-enqueue
diagnostic uses
a dedicated idle stream for its independent buffers; concurrent cells retain the
production caller-stream fence.

## Serving validation

All arms use the latest preparation/scheduling prototype, depth 2 and a frozen
predictor. The 128 MiB arm is its current transfer configuration. Each setting
has a
separate allocation, three excluded warmups and two matched 32-request
episodes/workload, concurrency 32, one output token. Priming and initialization
are
excluded. These are burst/reset/primed episodes, not continuous 300-request
throughput
cells.

<!-- markdownlint-disable MD013 -->
| Workload | Chunk MiB | Mean requests/s | Paired throughput gain | Mean TTFT ms | Paired mean TTFT reduction | Mean episode p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short16k | 128 | 20.900 | +0.00% | 1112.7 | -0.00% | 1514.7 |
| short16k | 8 | 24.572 | +17.69% | 894.2 | +19.62% | 1285.1 |
| short16k | 16 | 24.679 | +18.35% | 941.0 | +15.44% | 1275.7 |
| familiar60k | 128 | 6.481 | +0.00% | 3783.3 | -0.00% | 4864.8 |
| familiar60k | 8 | 7.139 | +10.17% | 3086.0 | +18.44% | 4430.4 |
| familiar60k | 16 | 7.089 | +9.39% | 3149.1 | +16.76% | 4447.9 |
| heterogeneous | 128 | 1.705 | +0.00% | 11413.9 | -0.00% | 18733.4 |
| heterogeneous | 8 | 1.705 | -0.02% | 11233.7 | +1.59% | 18610.8 |
| heterogeneous | 16 | 1.702 | -0.18% | 11262.5 | +1.33% | 18641.0 |
<!-- markdownlint-enable MD013 -->

Levels are arithmetic episode means; changes are geometric means of matched
episode
ratios. p95 is averaged across episode percentiles, not the pooled-request
percentile.
Two repeats and separate allocations leave order, runtime arrival/batch
composition and
allocation confounds. The small serving gains are descriptive, not statistical
significance or equivalence claims. All measured episodes pass exact
CPU-cache-token
accounting, zero additional GPU prefix-cache hits/preemptions, one output token,
frozen
state and all-rank optimized preparation coverage. No measured-window JIT or
error
notifications were found.

The cached workloads use context/suffix 15,552/832 and 58,368/3,072 tokens
respectively,
with every request CPU-prefix cached. The heterogeneous workload has 50% request
cache
hits: cached requests have a 4,096-token prefix, uncached requests a
24,576-token
prefix, and suffixes are 2,048 tokens for every third request and 8,192
otherwise. More
cold computation reduces the relative importance of H2D submission.

<!-- markdownlint-disable MD013 -->
| Workload | Mean compute cells at 128 MiB | At 8 MiB | At 16 MiB |
| --- | ---: | ---: | ---: |
| short16k | 4.5 | 5.0 | 4.5 |
| familiar60k | 4.0 | 5.0 | 5.0 |
| heterogeneous | 11.5 | 12.5 | 11.5 |
<!-- markdownlint-enable MD013 -->

The gains do not require fewer compute cells. In the 60K episodes, both
smaller-chunk
settings execute five compute cells instead of four yet improve throughput and
TTFT.
Fixed-cohort measurements directly show improved submission progress; serving
batch
composition also reacts to changed readiness. These data do not assign exact
percentages
of the live gain to each mechanism.

## Decision

Keep depth 2 and shortlist 8–16 MiB for this GLM TP8 pair. Eight MiB is the
strongest
equal-geometry calibration nominee and has the best mean TTFT on the cached
validation
workloads. Sixteen MiB is close in throughput, has fewer copy API calls and
remains a
readiness compromise. The small serving study does not establish a unique
throughput
optimum between them. Both improve cached-workload throughput roughly 9–18%
relative to
the current 128 MiB setting; mixed-workload throughput changes by less than
0.2%.

Startup selection should measure joint completion on representative compute and
transfer
shapes, then choose a fixed setting for the expected workload mix. Transfer-only
bandwidth misses the submission cost. Full screening and confirmation here took
516.8
and 459.8 seconds including separate model startup; a production probe should
reuse
warmed model/buffers and screen fewer representative combinations. No further
GPU
serving benchmark was run beyond the planned small validation. No production
default was
changed.

## Verification and implementation boundary

Nine CPU checks pass. The final calibration counts exclude four early rejected
launch/cohort attempts and the 19-cell mechanism pilot. The pilot after-enqueue
fence
originally serialized behind compute and is not pooled with the corrected
independent-stream diagnostic. One 8 MiB serving attempt rejected the original
predictor
snapshot before measurements; its directory and traceback are preserved under
`serving_8mib_rejected_snapshot`.

The trained predictor includes chunk size in compatibility identity. For this
frozen-weight experiment, only `context.h2d_chunk_bytes` is explicitly
retargeted for
8/16 MiB; every weight, history and other identity field is retained. Production
compatibility validation remains active. CPU replay of 80 control request-shape
events
produces exactly identical predictions in all three states. No retraining or
online
adaptation is allowed. The original checkpoint hash and loaded states are
recorded. Live
prediction accuracy can still change with submission interference.

Code is opt-in and isolated in `/mnt/localnvme/vllm_h2d_interference_20261009`,
branch
`experiment/h2d-compute-interference-calibration-20261009`. Native binaries,
model
weights/selector, ordinary scheduler source and production defaults are
unchanged. No
automatic startup selector or per-job chunk adaptation is deployed. The harness
is GLM
TP8-specific and does not establish an optimum for other hardware/model pairs.

Calibration shutdowns emitted inherited EngineCore force-kill/resource-tracker
warnings
after all eight explicit worker cleanup acknowledgments. All owned servers and
GPU
contexts were stopped and checked at completion. The 13 reused native binaries
retain
their setup hashes. Compact logs/JSON traces and the Nsight report are archived
on the
experiment branch; the exported SQLite remains under the artifact root.
