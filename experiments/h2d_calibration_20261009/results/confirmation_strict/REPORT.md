# Transfer-only chunk and depth calibration

<!-- markdownlint-disable MD013 -->

| Chunk MiB | Depth | 4K GB/s/GPU | 16K GB/s/GPU | 60K GB/s/GPU | Geometric speedup vs 128/2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 44.15 | 44.34 | 43.79 | 1.0737 |
| 16 | 8 | 42.71 | 44.98 | 44.58 | 1.0733 |
| 16 | 2 | 42.77 | 44.57 | 44.71 | 1.0716 |
| 64 | 2 | 36.91 | 44.48 | 45.24 | 1.0236 |
| 128 | 2 | 33.94 | 44.86 | 45.48 | 1.0000 |

<!-- markdownlint-enable MD013 -->

GB/s is decimal effective bandwidth per GPU using completion across all eight
ranks, including native submission and copy gaps. Each trial queues 4
identical FIFO loads per rank. Descriptor construction, allocation,
population, registration, warmups, and sampled byte checks are excluded.
No model compute
is active. These are reconstructed GLM page geometry and randomized source
and destination block mappings, not recorded serving transfer descriptors.

Best score: 8 MiB / depth 4.
Plateau selection: 16 MiB / depth 2.

The 4,096-descriptor cap is unchanged. Nominal chunk sizes can therefore
exceed the actual bytes per submission. Per-job chunk sizes and submission
counts are recorded in the raw measurements.
