# Bounded load and store descriptor coalescing

The D2H microprobe measured about 2.5 ms of submission for 4,752 descriptors.
A 95% hit at 60K has 90,288 H2D descriptors before merging, so the same physical
contiguity may offer a larger transfer submission saving. This is a separate
proposal; it must establish its effect in real serving.

Merge only adjacent source and destination byte ranges. Split merged H2D ranges
back into at most 8 MiB pieces. The original pinned descriptor arrays retain
ownership until actual completion, and the original producer/destination gates
and native two-chunk submission pipeline remain in force. If an original page
exceeds the chunk limit, retain the original descriptors.

Both serving arms enable D2H coalescing. Only the experimental arm enables
H2D merging. Report actual load descriptor reductions and the maximum emitted
descriptor size as well as throughput, TTFT, gaps and admission statistics.
Run bounded byte-mapping CPU tests and real GPU destination-write/retirement
checks before benchmarking. Both new coalescing flags default to disabled.
