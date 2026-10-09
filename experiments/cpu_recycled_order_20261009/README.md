# Ordered assignment of recycled CPU blocks

The CPU manager selects fresh IDs and pops recycled IDs from a LIFO pool.
Ascending GPU assignment does not align with descending reused CPU IDs, so
recycled rows can defeat physical descriptor merging in tensor-major storage.

This opt-in `cpu_allocator_ascending` connector option defaults to false and
requires `cpu_cache_layout="private_tensor"`. It sorts only the already-selected
CPU blocks before assigning keys and constructing the authoritative store spec.
The selected IDs, eviction candidates, capacity, load/store references, store
completion and cache policy remain unchanged. Logical cache keys are assigned
to their new physical rows before any DMA descriptor is published.

All 31 CPU manager tests passed; the raw output is in `CPU_TESTS.txt`.
CPU checks cover mixed fresh/reused allocation, protected live load references,
key-to-row mapping across eviction, failed stores and original default behavior.
The existing contiguous and fragmented tensor-major GPU byte tests verify the
transfer primitive for both kinds of mapping. Serving verification uses the
same continuous 300-request episodes as the tensor-major control, with no GPU
prefix hits and the exact expected CPU prefix restoration on every request.

Compare `--layout private_tensor --physical-optimization --cpu-ascending`
against the same tensor-major physical optimization with default CPU order.
Report actual descriptor counts and per-repeat serving performance; do not
infer a throughput benefit from the sorted IDs alone.
