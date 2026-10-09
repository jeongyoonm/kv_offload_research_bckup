# Disjoint-buffer submission contention

This GPU-zero probe uses private CPU/GPU tensors with the measured GLM page
geometry: 99 references, 48 store rows (146,534,400 bytes) and 912 load rows
(2,784,153,600 bytes). Store and load sources/destinations are disjoint. Two
background native H2D jobs use the unchanged 8 MiB/depth-two configuration.
NUMA-zero binding is preserved. Twelve measured repetitions follow two warmups.
Every case validates exact transferred bytes and exactly-once retirement.

| Cap | Store geometry | Loads | CPU submit (ms) | GPU transfer (ms) |
| --- | --- | ---: | ---: | ---: |
| 0 | 4,752 descriptors | 0 | 2.996 | 5.051 |
| 0 | 4,752 descriptors | 2 | 4.995 | 7.266 |
| 0 | 99 merged ranges | 0 | 0.619 | 2.618 |
| 0 | 99 merged ranges | 2 | 0.805 | 3.070 |
| 256 | 4,752 descriptors | 0 | 3.304 | 2.805 |
| 256 | 4,752 descriptors | 2 | 5.508 | 4.842 |
| 512 | 4,752 descriptors | 0 | 3.159 | 2.881 |
| 512 | 4,752 descriptors | 2 | 5.460 | 4.801 |

Small driver batches can start DMA earlier, reducing the idle-source GPU
envelope without reducing total main-thread submission work. Coalescing removes
most descriptor work when both source and destination are contiguous. Reversed
sources cannot merge and reproduce the larger submission cost. Native async
D2H and ascending GPU allocation are separate serving proposals motivated by
these observations.

A separate Nsight capture contains two measured repetitions per case. With
4,752 descriptors, the store `cuMemcpyBatchAsync` call increases from 3.156 to
5.126 ms with background H2D; 5.006 ms of the latter interval overlaps native
H2D copy calls on another thread. Merged calls are 0.073 and 0.103 ms. Profiler
overhead makes these absolute durations unsuitable for comparison with the
unprofiled medians. The trace locates the additional time inside the CUDA API,
but does not measure a specific internal lock or distinguish CPU, driver and
shared-device resource contention. No source-reuse dependency exists in this
probe; no full TP8 serving performance conclusion follows from it.

Raw JSON, logs, Nsight capture and SQLite export are in the complete hashed
`results/raw_probe.tar.gz`. The trace analysis and per-cap JSON are also directly
readable in `results/`. Artifacts originated in
`/mnt/localnvme/jeongyoon/store_overlap_overnight_20261009`.
Implementation: beec3d0fc1e28b51e371ac2242c38a98e2f6b340.
