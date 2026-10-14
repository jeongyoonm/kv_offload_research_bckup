# Coalesced seal notification mechanism verification

Implemented on `experiment/glm53-coalesced-seal-notify-20261004`.
Artifacts: `/mnt/localnvme/glm53_coalesced_notify_20261004`.

## Results

| Check | Result |
| --- | --- |
| CPU protocol and predictor suite | 135 passed |
| Configuration checks | 31 passed |
| Small CUDA mechanism checks | 9 passed |
| Native KV extension and compute observer | Built successfully |
| Partial candidate rank updates | No notification |
| Eight concurrent worker processes | One candidate notification |
| Long wait interrupted by a producer | One blocking `select` call |
| New background observer threads or scans | Zero |
| Serving/model benchmark | Not run |

The integrated CPU tests exercise both prediction margins (0 and 50 ms), all
fine intervals (2, 5 and 10 ms), and compute/KV/cancellation/fine transitions.
The held preceding RPC stays held while preparation and sealing complete.
The existing harness also checks execute enqueue before A's release and
exactly-once retirement.

Eight separate processes publish preparation and out-of-order KV job
completions concurrently. Shared masks retain all ranks and signal once for
the same candidate, including candidates that require multiple transfer jobs.
Notifications arriving before blocking, a full FIFO, stale preparation
generations, disarm/reset, and cleanup also pass. Terminal readiness remains
visible when the engine has to retry an authoritative read.

The CUDA checks copy 16 KiB through the actual async H2D worker and observe a
small compute operation through the native observer. Both signal after their
ordinary completion publication. KV ring records and ordinary completions
remain available to their original consumers. No model weights are loaded.
All CUDA processes exited after testing.

## Scope and remaining measurements

The 1 ms background scanner is eliminated during coalesced coarse waits.
Fine polling remains unchanged. Existing worker GPU completion observation
still runs, and producers acquire a shared lock to update masks. Throughput and
total CPU savings, including this producer bookkeeping, are unmeasured.
The feature stays experimental and disabled by default.

## Development record

Initial native publisher tests passed. An existing enqueue fixture exposed a
truthy unset `MagicMock` for the new opt-in flag; the runtime now requires an
explicit boolean `True`. A subsequent terminal-readiness latch made three
blocking-wait fixtures immediately eligible to seal; those fixtures were
corrected to keep a candidate pending while testing independent events.
The final protocol/predictor suite passes all 135 cases.

Initial hook failures were formatting and missing type annotations. They were
corrected. The inherited snapshot lacked its C++ formatting configuration;
the existing project configuration was restored from the Qwen FP8 worktree to
avoid unrelated formatting changes. Build and test logs and native source and
binary hashes are retained alongside this report. Earlier failed-run logs
remain in the artifact directory.
