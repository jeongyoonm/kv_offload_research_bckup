# GPU stream fence for store source reuse

Branch: `experiment/store-reuse-gpu-fence-20261009`.

The ordinary allocator can recycle a finished request's blocks before its
GPU-to-CPU store completes. Offloading detects intersecting source blocks and
flushes the corresponding store jobs before new writes. Previously this called
`event.synchronize()` on the forward worker thread. The experimental option
`store_reuse_gpu_fence=true` instead enqueues `stream.wait_event()` on the
current compute stream. Both forward writes and H2D loads remain ordered after
the store: the native H2D worker records its ready event on that compute stream.

This preserves the physical dependency. It allows CPU preparation and forward
enqueue to overlap pending stores. It does not make D2H DMA faster or allow a
GPU block to be overwritten before its store finishes. Store completion is
still reported only after the transfer event completes. Generic backends and
non-CUDA platforms retain host waits. Cache reset and mixed load/store drains
explicitly retain host waits. The option is disabled by default.

Direct markers separate store descriptor preparation, stream dependencies,
copy API enqueue, and completion fencing. Copy API wall time may include
internal CUDA blocking; it is not necessarily active CPU work.

Run a fresh matched allocation:

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.store_reuse_fence_20261009.run \
  --output /mnt/localnvme/jeongyoon/store_reuse_fence_20261009/paired
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.store_reuse_fence_20261009.report \
  /mnt/localnvme/jeongyoon/store_reuse_fence_20261009/paired
```

Control and GPU-fence arms both retain the latest 90 ms prediction correction,
100 ms setup lead, optimized input preparation, and 8 MiB/depth 2 H2D worker.
Four alternately ordered, matched 75-request episodes per arm constitute 300
measured requests per arm. They are not a continuous 300-request run. Each has
61,440 prompt tokens, a 58,368-token CPU prefix, a 3,072-token uncached suffix,
concurrency 32, and one output token. Separate 32-request warmups are excluded.

Tests exercise exactly-once submission and retirement, byte preservation for
forward and independent-H2D overwrites, and reset/load host drains. The GPU
regression extends a real store with a delayed DMA read so it can detect a
missing fence independently of any blocking inside batched memcpy submission.
