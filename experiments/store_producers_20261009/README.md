# Store-specific KV producer events

Branch: `experiment/store-specific-producer-events-20261009`.
Parent: `6aeab52`.
Artifacts: `/mnt/localnvme/jeongyoon/se1`.

The original deferred D2H path waits on the current compute stream at submission,
which can contain newer unrelated GPU work. Record one event per real computed
batch in post-forward, associating it with that batch's request IDs. Capture the
request's last producer event when queueing its store; wait on that event when
submitting the deferred D2H copy. Preserve STREAM source-access order, D2H FIFO,
block-reuse fences, descriptor ownership, and all-rank retirement.

The producer cache holds at most 256 request IDs. A miss falls back to the
original current-stream dependency; the Transfer retains the captured event
through completion. Guard against preemption, reset, zeroing, copy-on-write,
sliding-window and special CoW state geometries with the broad original path.
The option defaults to false. Tests must prove both actual producer ordering
and progress while newer unrelated compute is still pending. Traces report when
the specific producer is complete but the current compute stream is not.
