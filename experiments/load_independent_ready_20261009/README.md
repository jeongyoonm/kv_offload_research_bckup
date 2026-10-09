# Guarded independent H2D readiness

Native H2D records its ready event on the caller's compute stream. That protects
destination initialization but also inherits unrelated earlier computation.
This proposal uses a private ready stream only when the authoritative final
SchedulerOutput proves that the load destinations are disjoint from this step's
zeroing and copy-on-write footprints. GPU cache ownership already prevents
allocation of live compute blocks; CPU load references protect source pages.

The opt-in `load_independent_ready_stream` flag defaults to false and requires
early submission. The worker retains ordinary ordering for absent or mismatched
context, synchronous loads, preemptions, any store-reuse flush, Mamba cache state,
encoder work, speculative tokens, destination zeroing or copies involving a
destination as either source or target. Other media and non-native H2D retain
their existing submission path. No broad fence is removed in those cases.

GLM's scheduler already excludes fully overwritten async load destinations from
zeroing. Unrelated suffix zeroing therefore need not block a prefix transfer.
Native load FIFO, immutable CPU sources, descriptor ownership, actual completion
and all-rank retirement remain unchanged. The private stream is bounded to one
per worker and has no compute work. Reset still drains the original handlers.

CPU checks cover each footprint guard and authoritative context forwarding.
Two real GPU cases hold unrelated compute while submitting a load: ordinary
submission must wait, and independent submission must complete before the held
compute. Both cases validate loaded and untouched bytes and retire exactly once.
Both GPU cases passed in the serialized idle gate. The 36 CPU guard and
context tests passed with GPU allocation cases excluded. Logs are retained
in `GPU_TESTS.txt` and `CPU_TESTS.txt`. Run GPU checks only between serving
allocations.

The serving comparison keeps tensor-major storage, allocator avoidance, ascending
GPU assignment, descriptor merging, packet alignment, model, original selector,
FP8 KV, cache capacity, scheduler prediction and deadline configuration fixed.
Only the new readiness policy differs. It uses continuous 300-request episodes
with the reviewed 75-prefix CPU working set and distinct suffixes. Record guard
reasons, actual compute-stream readiness and performance; do not infer overlap
from the flag alone.
