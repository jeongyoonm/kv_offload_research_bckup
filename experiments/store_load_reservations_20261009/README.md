# Full request reservation gates

Branch: `experiment/store-reuse-load-reservations-20261009`.
Parent: `1f526ff` (scoped fences).
Artifacts: `/mnt/localnvme/jeongyoon/sr1`, with links from the
overnight report root.

The first scoped variant classifies the actual H2D destination pages. A loading
request may also reserve future suffix pages outside that transfer. No forward
for that request can run until its KV load becomes ready, so its load's private
stream gate can protect those reservations too. Record all allocated block IDs
for each asynchronous loading request while building the scheduler transaction;
classify both prefix destinations and future suffix reservations as dependencies
of that request's load. Load completion still implies that all protected stores
have completed on every rank. Current-compute block collisions remain global.
The reservation map is scheduler-private, bounded by one batch and cleared at
metadata construction; no block lists are added to worker RPC payloads.

Guard tests distinguish a store touching both load and current-compute blocks
from one touching load and future-suffix reservations, and retain broad handling
for reset, synchronous loads, preemption, zeroing and copy-on-write batches.
The serving comparison holds the previous global GPU-fence configuration fixed.
