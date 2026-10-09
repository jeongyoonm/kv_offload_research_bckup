# Tensor-major CPU cache layout results

All modes preserve the model, original selector, FP8 KV, TP8, cache
capacities, frozen predictor and scheduler. Each has two continuous
300-request episodes over the same 75-prefix working set. GPU prefix
hits are zero; every request restores exactly 58,368 CPU-cached tokens.

| Mode | R0 req/s | R1 req/s | Mean req/s | Gap ms |
| --- | ---: | ---: | ---: | ---: |
| st0 | 8.142 | 8.207 | 8.175 | 248.9 |
| st1 | 8.164 | 8.186 | 8.175 | 251.2 |
| stm | 8.181 | 7.148 | 7.665 | 3428.6 |

Gap values are per-episode sums between conservative all-rank compute
bounds. They are not device utilization or additive per-rank CPU times.

Tensor-major layout alone (st1) is neutral versus block-major (st0).
Adding ascending GPU assignment, descriptor merging and packet alignment
(stm) reduces 90,288 load descriptors to roughly 1,000–1,700 and 4,752
store descriptors to roughly 100–160. Both repeats retain this reduction.

Despite that reduction, stm repeat 1 loses 12.9% throughput versus the
same-seed control and accumulates 6,682.5 ms of inter-compute gaps. The
largest gaps occur around load-only dispatches with candidate-target
seals. They include roughly 440 ms in scheduling and 134–140 ms from
forward submission to the recorded GPU start. Repeat 0 improves 0.48%.
The layout is therefore not selected as a reliable serving improvement.

These traces identify delayed phases, not a particular CUDA driver lock
or memory-system cause. The queued independent-load readiness experiment
tests one narrower dependency hypothesis. Do not infer that CPU recycled
row reversal caused this regression: load descriptor reductions persist
and the observed CPU source heads remain ascending.

Verification includes 48 CPU checks and real GPU transfers covering all
99 tested GLM reference geometries with contiguous and fragmented IDs.
Separate actual-layout matrix probes validate loaded and untouched bytes.
These byte checks do not establish whole-model accuracy.

Each artifacts subdirectory contains a hash-verified raw archive, source
provenance, replay plans, predictor checkpoint and accepted-cell reports.
The timed source commit was 06be8de. No GPU tests ran during these serving
allocations; the preceding tensor gate finished before the first launch.
