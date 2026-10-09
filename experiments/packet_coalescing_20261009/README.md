# Packet-aligned descriptor splitting

CPU simulation with measured GLM page sizes found that independently splitting
merged physical ranges increases greedy native byte-chunk counts: dense 60K
prefixes use 410 chunks versus 333 raw chunks; dense 16K prefixes use 161 versus
89. This is a layout calculation, not a GPU timing or serving result.

The candidate splits merged ranges at cumulative 8 MiB byte boundaries, allowing
fragments from adjacent ranges to share a copy chunk. It preserves exact byte
mapping and original pinned-buffer ownership. If splitting would need more
slots than the original descriptors, it uses ordinary bounded splitting.

The experimental flag `load_descriptor_packet_alignment` defaults to false
and requires load coalescing. Both arms enable store-aware allocation, ascending
GPU assignment and H2D/D2H merging. Only the splitter differs. Original model,
selector, FP8 KV, cache sizes, native H2D depth two and scheduling remain fixed.
CPU byte-map and packing checks and six real GPU row/producer/retirement checks
are required before serving. Actual merge rate and serving effects determine
whether this calculation matters to the model workload.
