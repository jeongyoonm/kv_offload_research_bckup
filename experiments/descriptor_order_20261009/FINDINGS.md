# Actual-layout transfer matrix and negative order-preservation finding

All 12 cases completed with all expected loaded bytes and untouched rows
checked: 33.4 GB of final copied data was validated. Every native job was
waited and retired exactly once. Each case uses six measured copies after two
warmups, GPU zero, NUMA zero, 8 MiB chunks, depth two and process-local THP
disabling. No model or concurrent GPU computation was present.

| CPU layout / GPU IDs | Mode | Descriptors | GPU ms | Wall ms |
| --- | --- | ---: | ---: | ---: |
| Block major / ascending | Off | 90,288 | 60.97 | 62.95 |
| Block major / ascending | Sorted | 90,288 | 54.48 | 57.67 |
| Block major / ascending | Preserved | 90,288 | 60.96 | 64.12 |
| Block major / descending | Off | 90,288 | 60.98 | 62.98 |
| Block major / descending | Sorted | 90,288 | 56.24 | 59.40 |
| Block major / descending | Preserved | 90,288 | 60.93 | 64.08 |
| Tensor major / ascending | Off | 90,288 | 60.79 | 62.73 |
| Tensor major / ascending | Sorted | 430 | 52.47 | 55.19 |
| Tensor major / ascending | Preserved | 430 | 52.48 | 55.05 |
| Tensor major / descending | Off | 90,288 | 60.77 | 62.72 |
| Tensor major / descending | Sorted | 90,288 | 60.79 | 63.68 |
| Tensor major / descending | Preserved | 90,288 | 60.79 | 63.58 |

Values are medians from this ordered single allocation, not confidence
intervals. The matrix does not identify a particular GPU or driver mechanism.
The descriptor counts confirm that block-major CPU geometry prevents merging,
and reversed GPU assignments defeat tensor-major merging too. Packet-aligned
tensor-major merging uses 332 native chunks versus 333 raw chunks, so this
comparison holds chunk count nearly fixed while changing descriptor geometry.

Preserving the original order without merging did not improve transfers.
Legacy block-major sorting improved the measured GPU envelope by 4.7–6.5 ms,
while adding approximately 1.2 ms of CPU descriptor work. Preserve mode kept
the sorting analysis overhead but lost the measured transfer benefit. There
is no evidence to promote this proposal or spend a serving run on it now.
Tensor-major preserve versus sorted differed negligibly in this probe.

Ten targeted CPU helper tests and two actual-layout store/load GPU byte tests
passed. The tensor-major layout also passed contiguous and fragmented GLM
GPU byte tests on its own branch. No model, precision, selection, cache
capacity or production default was changed. All raw samples and verification
logs are in `artifacts`; source provenance is in the master gate process log.
