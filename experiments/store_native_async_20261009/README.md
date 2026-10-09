# Native D2H submission

This proposal uses the existing native batch-copy I/O worker for stores, with
8 MiB chunks and depth two. The source uses STREAM access ordering and waits
on the producer captured from the caller's current stream. Descriptor tensors
remain owned until real completion. Reset and source reuse wait for actual
completion; no unrecorded completion event is used as a GPU fence.

The connector options are `async_d2h_submission` (default false) and
`async_d2h_chunk_bytes` (default 8 MiB). The matched serving driver holds
pending-store allocator preference, model, cache capacity, H2D and scheduler
policy constant. Run separate fresh roots with and without `--native-store`.
Each mode has four separately primed/reset 75-request episodes, 300 requests
in total. Independent allocations cannot establish a within-process paired
comparison, although workload seeds and all serving parameters are matched.

The first gate is byte-exact delayed-producer, source-reuse and exactly-once
retirement testing on idle GPUs. Native H2D descriptors and completion-ring
semantics remain separate from the store submitter.
