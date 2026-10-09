# Recycled CPU row ordering

The proposal sorts only the selected CPU block IDs before assigning cache
keys; eviction choices, protected references and capacity remain fixed.
All 31 CPU manager checks passed before serving.

Two continuous 300-request episodes share the seeds and 75-prefix working
set of st0 and stm. Every request restores 58,368 CPU tokens, has zero GPU
prefix hits, produces one token and incurs no preemption.

| Repeat | Block req/s | Merged req/s | Sorted req/s | Sorted gap ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.142 | 8.181 | 7.867 | 1329.5 |
| 1 | 8.207 | 7.148 | 8.002 | 1262.4 |

Sorting is slower than block-major in both repeats. The merged comparator
has an unstable second repeat, so its mean is not evidence that sorting
is a reliable throughput gain. Descriptor merging remains substantial:
rank-zero mean load descriptors are 1,104 and 1,780 versus 90,288 original;
mean store descriptors are 120 and 158 versus 4,752 original.

The sorted mode still exposes about 1,296 ms of inter-compute gaps per
episode, including about 1,056 ms from host forward submission to the
recorded GPU start. CPU source heads were already ascending in the merged
comparator, so row reversal does not explain that earlier slow repeat.
The proposal is retained as experimental and not selected for the
resident-cache reference comparison.

The verified raw archive retains launch, ownership, counters, predictor,
responses and all-rank timing. Byte-transfer tests on contiguous and
fragmented row mappings remain separate from whole-model correctness.
