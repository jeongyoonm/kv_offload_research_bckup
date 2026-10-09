# Pending-store-aware free block preference

Branch: `experiment/store-reuse-allocator-preference-20261009`.
Parent: `6aeab52`.
Artifacts: `/mnt/localnvme/jeongyoon/store_overlap_overnight_20261009/allocator`.

Free blocks are currently reused in queue order, including recently freed
uncached blocks prepended while GPU-to-CPU stores may still read them. Bind the
connector's live pending-store block map to the GPU block pool. The opt-in policy
prefers unfenced blocks and preserves the free queue order of skipped blocks.
When safe blocks are insufficient, ordinary fenced reuse remains available;
capacity, admission and existing copy-before-overwrite enforcement are preserved.
Completion removes hazards only after all ranks acknowledge the real store.
No additional reference counts, reservations, or GPU buffers are introduced.

Audit allocation choices, pending counts and CPU selection time in both arms.
Matched runs use global GPU stream fences in both arms and differ only in block
preference. New stores created after allocation can still collide, and prefix
cache eviction decisions can change; the measurements must distinguish these
limits from actual safe-block selection savings. CPU-only tests cover safe
alternatives, capacity exhaustion, exact ownership, and reset/completion updates.
