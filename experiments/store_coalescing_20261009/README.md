# Coalesce adjacent D2H descriptors

Branch: `experiment/store-descriptor-coalescing-20261009`.
Parent: `6aeab52`.
Artifacts: `/mnt/localnvme/jeongyoon/sc1`.

A measured GLM store job contains 4,752 descriptors for 146,534,400 bytes. Its
median copy-wrapper CPU call takes about 2.78 ms on rank zero. Sort source/dest
pairs by physical source address and merge only ranges whose source and dest
addresses are both adjacent. CUDA batch copies already have unordered internal
execution. Descriptor ownership, total bytes, stream ordering and store reuse
fences are unchanged. H2D retains its calibrated descriptor/chunk configuration.
Default disabled. Record original and merged descriptor counts on each store.

The byte mapping tests cover ascending, descending and nonmergeable destination
relationships. Real-GPU checks will validate the copied bytes before any serving
comparison. Gains depend on physical allocation order; low coalescing rates are
an explicit possible negative result.
