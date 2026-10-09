# Private tensor-major CPU layout

The live block-major private CPU cache stores all layer pages within each block
row. For GLM, the row stride is 3,052,800 bytes while individual pages are 36,864
or 8,448 bytes. Consecutive pages of one layer are therefore not contiguous, even
when GPU block assignment is sorted. This explains why the physical-order
proposal merges no measured descriptors.

`cpu_cache_layout="private_tensor"` stores each tensor's rows contiguously in
the existing per-rank mmap. Logical block IDs, tensor shapes, CPU byte budget,
logical capacity, file ownership, contiguous population and local CUDA
registration remain unchanged. The normal default is still `block`. As with
`private`, only direct sharded single-node CPU offload is supported; canonical,
replicated and secondary layouts are rejected.

Compare three independent allocations with identical workload seeds:

1. `--layout private`: the latest scheduler and store-aware allocator.
2. `--layout private_tensor`: isolates CPU physical layout.
3. `--layout private_tensor --physical-optimization`: additionally enables
   ascending selected GPU assignment, H2D/D2H descriptor merging and packet
   alignment within the unchanged 8 MiB/depth-two H2D pipeline.

Each measured episode contains 300 continuous requests at concurrency 32,
75 distinct 58,368-token CPU prefixes and four distinct 3,072-token suffixes per
prefix. Prime the 75 prefixes once, reset GPU cache once, and do not reset or
prime again during measurement. Exact expected CPU restoration, zero GPU hits,
one generated token, no preemption, frozen prediction and all-rank preparation
coverage remain acceptance gates. `--requests 75` reproduces earlier episodes.

CPU tests verify disjoint layer storage, full byte coverage, bounds, rank
ownership and capacity. Two real GPU cases cover all 99 GLM page geometries,
contiguous and fragmented mappings, synchronous stores, native pipelined loads,
exact bytes, untouched rows, retirement and shutdown. Run them only with idle
GPUs before serving. No original selector or model-execution change is included.
