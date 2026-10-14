# Transfer-only chunk and depth calibration

<!-- markdownlint-disable MD013 -->

| Chunk MiB | Depth | 4K GB/s/GPU | 16K GB/s/GPU | 60K GB/s/GPU | Geometric speedup vs 128/2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 43.85 | 45.63 | 45.51 | 1.0827 |
| 8 | 4 | 44.56 | 44.67 | 44.29 | 1.0712 |
| 16 | 2 | 42.74 | 44.79 | 45.09 | 1.0637 |
| 64 | 2 | 37.33 | 45.19 | 45.71 | 1.0244 |
| 128 | 2 | 34.21 | 45.41 | 46.17 | 1.0000 |

<!-- markdownlint-enable MD013 -->

GB/s is decimal effective bandwidth per GPU using completion across all eight
ranks, including native submission and copy gaps. Each trial queues 4
identical FIFO loads per rank. Descriptor construction, allocation,
population, registration, warmups, and sampled byte checks are excluded.
No model compute
is active. These are reconstructed GLM page geometry and randomized source
and destination block mappings, not recorded serving transfer descriptors.

Best score: 16 MiB / depth 8.
Plateau selection: 16 MiB / depth 2.

The 4,096-descriptor cap is unchanged. Nominal chunk sizes can therefore
exceed the actual bytes per submission. Per-job chunk sizes and submission
counts are recorded in the raw measurements.
