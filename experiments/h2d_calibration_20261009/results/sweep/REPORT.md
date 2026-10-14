# Transfer-only chunk and depth calibration

<!-- markdownlint-disable MD013 -->

| Chunk MiB | Depth | 4K GB/s/GPU | 16K GB/s/GPU | 60K GB/s/GPU | Geometric speedup vs 128/2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 43.75 | 46.01 | 45.97 | 1.0907 |
| 16 | 4 | 43.71 | 45.97 | 45.70 | 1.0878 |
| 8 | 8 | 44.61 | 44.80 | 44.76 | 1.0784 |
| 8 | 4 | 44.93 | 44.74 | 44.40 | 1.0776 |
| 16 | 2 | 43.53 | 45.01 | 45.17 | 1.0747 |
| 32 | 8 | 41.51 | 45.57 | 46.00 | 1.0685 |
| 32 | 4 | 41.20 | 45.48 | 45.67 | 1.0626 |
| 32 | 2 | 41.34 | 44.80 | 45.59 | 1.0579 |
| 8 | 2 | 43.18 | 43.81 | 44.08 | 1.0535 |
| 64 | 8 | 37.48 | 46.04 | 46.37 | 1.0391 |
| 64 | 2 | 37.45 | 45.45 | 45.99 | 1.0315 |
| 64 | 4 | 37.46 | 45.27 | 46.14 | 1.0314 |
| 128 | 8 | 34.11 | 45.86 | 46.35 | 1.0055 |
| 128 | 4 | 34.15 | 45.78 | 46.37 | 1.0054 |
| 128 | 2 | 34.06 | 45.52 | 46.00 | 1.0000 |
| 256 | 4 | 32.75 | 45.43 | 46.21 | 0.9879 |
| 512 | 4 | 32.74 | 45.43 | 46.15 | 0.9873 |
| 512 | 8 | 32.79 | 45.12 | 46.30 | 0.9866 |
| 512 | 2 | 32.93 | 45.11 | 45.94 | 0.9854 |
| 256 | 8 | 32.74 | 45.13 | 46.12 | 0.9849 |
| 256 | 2 | 32.57 | 44.92 | 45.71 | 0.9787 |
| 512 | 1 | 27.03 | 27.08 | 26.82 | 0.6505 |
| 256 | 1 | 26.88 | 26.99 | 26.77 | 0.6482 |
| 64 | 1 | 26.74 | 26.73 | 26.71 | 0.6445 |
| 128 | 1 | 26.65 | 26.69 | 26.66 | 0.6430 |
| 32 | 1 | 26.37 | 26.57 | 26.55 | 0.6390 |
| 16 | 1 | 26.08 | 26.20 | 26.19 | 0.6308 |
| 8 | 1 | 25.48 | 25.57 | 25.56 | 0.6158 |

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
