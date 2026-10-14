# Coalesced seal notifications

Branch: `experiment/glm53-coalesced-seal-notify-20261004`.
Artifacts: `/mnt/localnvme/glm53_coalesced_notify_20261004`.

This experiment replaces the 1 ms background bridge during the predicted
coarse wait with producer-driven notifications. It remains disabled by default.
Fine polling and load-only polling retain their existing intervals. The seal
policy, scheduler accounting, authoritative KV ring consumer, worker RPC FIFO,
and ordered retirement remain unchanged.

## Mechanism

A shared table holds at most 16 candidates, 256 transfer-job references, and
16 preceding compute steps. The engine installs candidate job identities
before publishing preparation. Each candidate has a preparation rank mask;
each referenced job has a KV completion rank mask. The existing native H2D
worker updates the KV mask after publishing its ordinary completion record.
The preparation lane updates its mask after publishing its ordinary ACK.

Only the transition to all-rank KV readiness plus all-rank preparation for
one candidate sends a FIFO notification. The table handles multiple jobs per
candidate and different completion orders across ranks. Partial updates perform
producer bookkeeping without notifying or waking the engine. These updates are
serialized by a process-shared `flock` and a local mutex; this implementation
is not lock-free.

The existing compute observer signals after publishing an all-rank compute
completion. Cancellation signals after insertion into the authoritative abort
queue. Transfer, overflow, preparation, and compute failures also signal.
The engine blocks in `select` on a named FIFO or its prediction deadline.
There is no new observer thread, periodic scan, or CUDA host callback.
Existing worker-side GPU completion observation still runs.

The engine drains the FIFO and takes a sequence ticket under the same producer
lock before checking authoritative progress. A notification between the
snapshot and wait therefore cannot be lost. A full FIFO retains progress via
the sequence. Terminal compute/candidate readiness remains actionable across
brief authoritative-reader contention without repeated producer notifications.
Every wake is advisory: only the engine decides membership and performs
admission accounting.

Preparation carries the window generation, and KV and compute publications
match transfer-job or compute-step identities. Already drained KV completions
are seeded into the table so a completion before candidate publication cannot
be missed. Disarm clears the active window and wakes a waiting reader; the next
window replaces all candidate and compute masks. Dynamic control descriptors
are removed at seal and are not retained in the channel's ownership list.

## Enable and build

Start with connector extra configuration `coalesced_seal_notifications: true`.
This creates transports before workers start. Then enable experimental option
`late_seal_coalesced_notifications: true` together with the existing full,
conditioned prediction-to-readiness policy. Startup checks the native KV ABI;
a native execution observer without the notification ABI is rejected. Use the
observer built from this directory, rather than an older observer library.

The targeted CMake build compiles the KV extension, compute observer, and CPU
publisher harness without rebuilding core kernels:

```bash
seal_torch_cmake="$PWD/.venv/lib/python3.12/site-packages/torch/share/cmake"
.venv/bin/cmake -S experiments/coalesced_notify_20261004 \
  -B /mnt/localnvme/glm53_coalesced_notify_20261004/build -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_PREFIX_PATH="$seal_torch_cmake" \
  -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build \
  /mnt/localnvme/glm53_coalesced_notify_20261004/build -j 4
```

For this worktree, `_glm53_async.so` is a private copy of this build. Other
worktrees retain their original native libraries. The source observer derives
from the prior timing-interference observer, with notifications added after its
ordinary timing publication.

## Initial verification scope

The CPU harness covers partial masks, multiple jobs, out-of-order reports,
eight concurrent worker processes, notification-before-wait races, cancellation,
producer failure, terminal-read retries, full FIFO, reset and cleanup. The
integrated seal path checks compute/KV/cancellation events for both margins and
all three fine intervals while A's ordered RPC remains held. The existing
preparation, immediate-seal, execute-enqueue, and retirement checks also run.

Small CUDA tests use 16 KiB transfers and a small compute operation. They check
that the actual H2D and native compute publishers signal, and that notifications
consume neither KV ring records nor ordinary completions. No model is loaded
and no serving benchmark is run. This verifies the mechanism; throughput and
net CPU savings, including producer locking costs, remain unmeasured.

## Matched serving comparison

The subsequent CPU replay and twelve GLM TP8 serving cells are documented in
[AB_REPORT.md](results/AB_REPORT.md). Raw artifacts are under
`/mnt/localnvme/glm53_coalesced_notify_20261004/ab`.

```bash
mkdir -p /mnt/localnvme/glm53_coalesced_notify_20261004/cpu-comparison/tmp
TMPDIR=/mnt/localnvme/glm53_coalesced_notify_20261004/cpu-comparison/tmp \
  PYTHONPATH="$PWD" .venv/bin/python \
  experiments/coalesced_notify_20261004/cpu_compare.py
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/coalesced_notify_20261004/canary.py \
  --name gpu --policies bridge,coalesced \
  --workloads 8192:95:100,61440:95:100 \
  --requests 64 --repeats 3 --max-model-len 65536 --distinct-fact-probe
.venv/bin/python experiments/coalesced_notify_20261004/analyze_ab.py
```

These scripts use fixed artifact locations. Archive the existing `ab` directory
before a fresh serving comparison; the driver rejects saved measured cells to
avoid skipping predictor training on a new server. Native build artifacts remain
in the parent directory. The launch preflight checks vendor dependencies and GPU
memory cleanup. Fresh worktrees must also restore their ignored precompiled
vendor sources and libraries; restoration source paths and hashes for this run
are retained in the artifact directory.
